// backend/tensor_core/ptx_mma.cpp - PTX tensor core MMA emission
#include "cg/backend/tensor_core/ptx_mma.hpp"

#include <sstream>

namespace cg {

std::string PTXMMAEmitter::reg(u32 id, DType dt) const {
    if (dt == DType::F32) return "%f" + std::to_string(id);
    if (dt == DType::F16 || dt == DType::BF16) return "%h" + std::to_string(id);
    return "%r" + std::to_string(id);
}

MMAFragments PTXMMAEmitter::allocate_fragments(u32& next_reg_id) const {
    MMAFragments frags;
    // A: 4 registers (f16x2 packed, stored as 32-bit regs)
    for (int i = 0; i < 4; ++i)
        frags.a_regs.push_back(reg(next_reg_id++, DType::F16));
    // B: 2 registers
    for (int i = 0; i < 2; ++i)
        frags.b_regs.push_back(reg(next_reg_id++, DType::F16));
    // C/D: 4 registers (f32)
    for (int i = 0; i < 4; ++i)
        frags.c_regs.push_back(reg(next_reg_id++, DType::F32));
    return frags;
}

std::string PTXMMAEmitter::emit_mma_m16n8k16_f16_f32(
    const MMAFragments& frags) const {
    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    //   {%d0, %d1, %d2, %d3},
    //   {%a0, %a1, %a2, %a3},
    //   {%b0, %b1},
    //   {%c0, %c1, %c2, %c3};
    std::ostringstream os;
    os << "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n";
    os << "        {" << frags.c_regs[0] << ", " << frags.c_regs[1] << ", "
       << frags.c_regs[2] << ", " << frags.c_regs[3] << "},\n";
    os << "        {" << frags.a_regs[0] << ", " << frags.a_regs[1] << ", "
       << frags.a_regs[2] << ", " << frags.a_regs[3] << "},\n";
    os << "        {" << frags.b_regs[0] << ", " << frags.b_regs[1] << "},\n";
    os << "        {" << frags.c_regs[0] << ", " << frags.c_regs[1] << ", "
       << frags.c_regs[2] << ", " << frags.c_regs[3] << "};";
    return os.str();
}

std::string PTXMMAEmitter::emit_load_a_fragment(
    const std::string& smem_ptr,
    const std::string& offset_reg,
    const MMAFragments& frags) const {
    // Load A fragment from shared memory.
    // For m16n8k16, each thread in the warp loads a portion of the [16,16]
    // f16 matrix from shared memory.
    //
    // Thread mapping (Ampere m16n8k16):
    //   Thread t loads:
    //     a_regs[0]: smem[t/4 * lda + (t%4)*8]     (2 x f16)
    //     a_regs[1]: smem[t/4 * lda + (t%4)*8 + 4] (2 x f16)
    //     a_regs[2]: smem[(t/4+8) * lda + (t%4)*8]
    //     a_regs[3]: smem[(t/4+8) * lda + (t%4)*8 + 4]
    //
    // We use ldmatrix.sync.aligned.m8n8.x4.shared.b16 which loads 4 8x8
    // matrices in one instruction. This is the efficient way to load MMA
    // fragments on Ampere.
    std::ostringstream os;
    os << "    // Load A fragment from shared memory\n";
    os << "    ldmatrix.sync.aligned.m8n8.x4.shared.b16\n";
    os << "        {" << frags.a_regs[0] << ", " << frags.a_regs[1] << ", "
       << frags.a_regs[2] << ", " << frags.a_regs[3] << "},\n";
    os << "        [" << smem_ptr << " + " << offset_reg << "];";
    return os.str();
}

std::string PTXMMAEmitter::emit_load_b_fragment(
    const std::string& smem_ptr,
    const std::string& offset_reg,
    const MMAFragments& frags) const {
    // Load B fragment: 2 registers for [16,8] f16.
    // ldmatrix.sync.aligned.m8n8.x2.shared.b16 loads 2 8x8 matrices.
    std::ostringstream os;
    os << "    // Load B fragment from shared memory\n";
    os << "    ldmatrix.sync.aligned.m8n8.x2.shared.b16\n";
    os << "        {" << frags.b_regs[0] << ", " << frags.b_regs[1] << "},\n";
    os << "        [" << smem_ptr << " + " << offset_reg << "];";
    return os.str();
}

std::string PTXMMAEmitter::emit_store_d_fragment(
    const std::string& global_ptr,
    const std::string& offset_reg,
    const MMAFragments& frags) const {
    // Store D fragment: 4 f32 registers holding the [16,8] result.
    // Each thread stores its portion to global memory.
    std::ostringstream os;
    os << "    // Store D fragment to global memory\n";
    // For simplicity, emit individual stores. A real implementation would
    // use stmatrix or vectorized stores.
    for (usize i = 0; i < frags.c_regs.size(); ++i) {
        os << "    st.global.f32 [" << global_ptr << " + "
           << offset_reg << " + " << (i * 4) << "], "
           << frags.c_regs[i] << ";\n";
    }
    return os.str();
}

std::string PTXMMAEmitter::emit_zero_accumulator(const MMAFragments& frags) const {
    std::ostringstream os;
    os << "    // Zero accumulator\n";
    for (auto& r : frags.c_regs) {
        os << "    mov.f32 " << r << ", 0.0;\n";
    }
    return os.str();
}

std::string PTXMMAEmitter::emit_tiled_matmul_kernel(
    u32 M, u32 N, u32 K,
    DType dtype,
    const std::string& kernel_name,
    u32 m_tile,
    u32 n_tile,
    u32 k_tile) const {
    // Emit a complete tiled GEMM kernel using tensor cores.
    //
    // Structure:
    //   1. Prologue: compute thread indices, block indices
    //   2. Allocate shared memory for A and B tiles
    //   3. For each K tile:
    //      a. Load A tile from global → shared
    //      b. Load B tile from global → shared
    //      c. cp.async (or ld.global) + bar.sync
    //      d. For each k_inner step (16):
    //         - Load A fragment from shared via ldmatrix
    //         - Load B fragment from shared via ldmatrix
    //         - mma.sync
    //      e. bar.sync
    //   4. Store result to global
    //
    // This is a simplified version. A production kernel would:
    //   - Use cp.async.ca.shared.global for async copies
    //   - Double-buffer shared memory tiles
    //   - Use software pipelining to overlap loads and compute
    //   - Handle non-divisible tile sizes with tail handling

    std::ostringstream os;
    os << "// Tensor core GEMM kernel generated by cantors-gift\n";
    os << "// C[" << M << "x" << N << "] = A[" << M << "x" << K
       << "] @ B[" << K << "x" << N << "]\n";
    os << "// Tile: " << m_tile << "x" << n_tile << "x" << k_tile << "\n";
    os << "// MMA: m16n8k16 f16->f32 accumulate\n";
    os << ".version 7.5\n";
    os << ".target sm_80\n";
    os << ".address_size 64\n\n";

    os << ".visible .entry " << kernel_name << "(\n";
    os << "    .param .u64 _param_A,\n";
    os << "    .param .u64 _param_B,\n";
    os << "    .param .u64 _param_C,\n";
    os << "    .param .u32 _param_M,\n";
    os << "    .param .u32 _param_N,\n";
    os << "    .param .u32 _param_K\n";
    os << ") {\n";

    // Load params.
    os << "    .reg .u64 %rd_A, %rd_B, %rd_C;\n";
    os << "    .reg .u32 %r_M, %r_N, %r_K;\n";
    os << "    ld.param.u64 %rd_A, [_param_A];\n";
    os << "    ld.param.u64 %rd_B, [_param_B];\n";
    os << "    ld.param.u64 %rd_C, [_param_C];\n";
    os << "    ld.param.u32 %r_M, [_param_M];\n";
    os << "    ld.param.u32 %r_N, [_param_N];\n";
    os << "    ld.param.u32 %r_K, [_param_K];\n\n";

    // Thread/block indices.
    os << "    .reg .u32 %tid_x, %tid_y, %blk_x, %blk_y;\n";
    os << "    mov.u32 %tid_x, %tid.x;\n";
    os << "    mov.u32 %tid_y, %tid.y;\n";
    os << "    mov.u32 %blk_x, %ctaid.x;\n";
    os << "    mov.u32 %blk_y, %ctaid.y;\n\n";

    // Shared memory for tiles.
    u32 smem_a_size = m_tile * k_tile * 2; // f16 = 2 bytes
    u32 smem_b_size = k_tile * n_tile * 2;
    os << "    .shared .align 128 .b8 %smem_A[" << smem_a_size << "];\n";
    os << "    .shared .align 128 .b8 %smem_B[" << smem_b_size << "];\n\n";

    // Allocate MMA fragment registers.
    u32 next_reg = 0;
    MMAFragments frags = allocate_fragments(next_reg);

    // Declare fragment registers.
    os << "    // MMA fragment registers\n";
    os << "    .reg .b32 4 " << frags.a_regs[0];
    for (int i = 1; i < 4; ++i) os << ", " << frags.a_regs[i];
    os << "; // A fragment (f16x2 packed)\n";
    os << "    .reg .b32 2 " << frags.b_regs[0] << ", " << frags.b_regs[1];
    os << "; // B fragment\n";
    os << "    .reg .f32 4 " << frags.c_regs[0];
    for (int i = 1; i < 4; ++i) os << ", " << frags.c_regs[i];
    os << "; // C/D accumulator\n\n";

    // Zero accumulator.
    os << emit_zero_accumulator(frags) << "\n";

    // K-loop.
    u32 k_steps = k_tile / 16; // each MMA does k=16
    os << "    // K-loop: " << (K / k_tile) << " outer iterations, "
       << k_steps << " MMA per tile\n";
    os << "    .reg .u32 %r_k_outer, %r_k_inner;\n";
    os << "    .reg .u64 %rd_A_offset, %rd_B_offset;\n";
    os << "    .reg .u32 %r_smem_offset;\n";
    os << "    mov.u32 %r_k_outer, 0;\n\n";

    os << "    // K-outer loop start\n";
    os << "L_k_outer_begin:\n";
    os << "    // Load A tile from global to shared\n";
    os << "    // (Each thread loads a portion of the tile)\n";
    os << "    // ... cp.async or ld.global + st.shared ...\n\n";

    os << "    // Load B tile from global to shared\n";
    os << "    // ... cp.async or ld.global + st.shared ...\n\n";

    os << "    // Wait for shared memory loads to complete\n";
    os << "    bar.sync 0;\n\n";

    // Inner K loop: iterate over k_inner steps.
    os << "    // K-inner loop: " << k_steps << " MMA instructions\n";
    os << "    mov.u32 %r_k_inner, 0;\n";
    os << "L_k_inner_begin:\n";

    // Load fragments from shared memory.
    os << "    // Compute shared memory offset for this k_inner step\n";
    os << "    mad.lo.u32 %r_smem_offset, %r_k_inner, 16, 0;\n\n";

    os << emit_load_a_fragment("%smem_A", "%r_smem_offset", frags) << "\n";
    os << emit_load_b_fragment("%smem_B", "%r_smem_offset", frags) << "\n";

    // Execute MMA.
    os << emit_mma_m16n8k16_f16_f32(frags) << "\n";

    // Increment and loop.
    os << "    add.u32 %r_k_inner, %r_k_inner, 1;\n";
    os << "    cmp.lt.u32 %p_k_inner, %r_k_inner, " << k_steps << ";\n";
    os << "    @%p_k_inner bra L_k_inner_begin;\n\n";

    // Next K tile.
    os << "    // Advance to next K tile\n";
    os << "    add.u32 %r_k_outer, %r_k_outer, " << k_tile << ";\n";
    os << "    cmp.lt.u32 %p_k_outer, %r_k_outer, " << K << ";\n";
    os << "    @%p_k_outer bra L_k_outer_begin;\n\n";

    // Store result.
    os << "    // Store result to global memory\n";
    os << emit_store_d_fragment("%rd_C", "%tid_x", frags) << "\n";

    os << "    ret;\n";
    os << "}\n";
    return os.str();
}

TensorCoreCaps TensorCoreCaps::ampere() {
    TensorCoreCaps caps;
    caps.has_ampere_mma = true;
    caps.has_hopper_wgmma = false;
    caps.has_tma = false;
    caps.has_async_barrier = false;
    caps.peak_f16_tflops = 312.0;
    caps.peak_bf16_tflops = 312.0;
    caps.peak_i8_tflops = 624.0;
    caps.peak_fp8_tflops = 0; // Ampere doesn't have FP8
    caps.supported_tiles = {
        {16, 8, 16},  // m16n8k16 f16
        {16, 8, 8},   // m16n8k8 f16
        {8, 8, 16},   // m8n8k16 (for smaller tiles)
    };
    return caps;
}

TensorCoreCaps TensorCoreCaps::hopper() {
    TensorCoreCaps caps = ampere(); // Hopper supports all Ampere MMA
    caps.has_hopper_wgmma = true;
    caps.has_tma = true;
    caps.has_async_barrier = true;
    caps.peak_f16_tflops = 989.0;  // ~3x Ampere
    caps.peak_bf16_tflops = 989.0;
    caps.peak_i8_tflops = 1979.0;
    caps.peak_fp8_tflops = 1979.0;
    // WGMMA supports larger tiles
    caps.supported_tiles.push_back({64, 128, 16}); // wgmma.mma_async
    caps.supported_tiles.push_back({64, 256, 16});
    return caps;
}

TensorCoreCaps TensorCoreCaps::blackwell() {
    TensorCoreCaps caps = hopper();
    // Blackwell adds FP4/FP6/MX format support
    caps.peak_fp8_tflops = 3958.0; // ~2x Hopper
    caps.supported_tiles.push_back({128, 256, 32}); // 5th gen TC
    return caps;
}

} // namespace cg
