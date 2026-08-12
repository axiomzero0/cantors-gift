// backend/tensor_core/ptx_mma.hpp - PTX tensor core MMA instruction emission
//
// Emits NVIDIA tensor core MMA (Matrix Multiply-Accnowledge) instructions
// in PTX. These are the instructions that deliver 312 TFLOPs on A100
// (vs 15 TFLOPs for scalar F32 FMA) — a 20x throughput difference.
//
// Supported MMA shapes (Ampere SM_80+):
//
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//     D[16x8] = A[16x16] @ B[16x8] + C[16x8]
//     A: f16, B: f16, C/D: f32 (accumulated in f32 for accuracy)
//     This is the workhorse — ~600 TFLOPs on A100.
//
//   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
//     Smaller K tile, useful for tail handling.
//
//   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//     F16 accumulation — faster but less accurate.
//
// Register layout (Ampere m16n8k16):
//   A fragments (4 registers, f16x2 packed):
//     %a0 = A[0:8, 0:4]   %a1 = A[0:8, 4:8]
//     %a2 = A[8:16, 0:4]  %a3 = A[8:16, 4:8]
//   B fragments (2 registers, f16x2 packed):
//     %b0 = B[0:8, 0:4]   %b1 = B[0:8, 4:8]
//   C/D fragments (4 registers, f32):
//     %c0 = D[0:4, 0:4]   %c1 = D[4:8, 0:4]
//     %c2 = D[8:12, 0:4]  %c3 = D[12:16, 0:4]
//
// Each thread in a warp (32 threads) holds a different slice of the
// fragment. The .sync suffix means all 32 threads must execute the
// instruction together (warp-synchronous).
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <string>
#include <vector>

namespace cg {

// MMA tile shape: M x N x K
struct MMAShape {
    u32 m;
    u32 n;
    u32 k;
};

// MMA operand types
struct MMATypes {
    DType a_type;   // input A dtype (typically F16 or BF16)
    DType b_type;   // input B dtype
    DType c_type;   // accumulator dtype (F32 for accuracy, F16 for speed)
    DType d_type;   // output dtype
};

// Register allocation for MMA fragments.
struct MMAFragments {
    // A fragment registers (f16x2 packed, so each holds 2 f16 values)
    std::vector<std::string> a_regs;
    // B fragment registers
    std::vector<std::string> b_regs;
    // C/D fragment registers (accumulator)
    std::vector<std::string> c_regs;
};

// Ampere m16n8k16 MMA: 4 A regs, 2 B regs, 4 C/D regs.
// This is the most common tensor core instruction.
class PTXMMAEmitter {
public:
    PTXMMAEmitter() = default;

    // Emit a single mma.sync.aligned.m16n8k16 instruction.
    //
    // D = A @ B + C
    // where A is [16,16] f16, B is [16,8] f16, C/D are [16,8] f32.
    //
    // Returns the PTX text for the instruction.
    std::string emit_mma_m16n8k16_f16_f32(
        const MMAFragments& frags) const;

    // Emit a load of A fragment from shared memory.
    // A is [16,16] f16, stored in shared memory in row-major.
    // Each thread loads its portion of the fragment.
    //
    // Thread mapping (Ampere): lane (t) loads:
    //   a_regs[0] = smem[t/4][ (t%4)*8     : (t%4)*8 + 4 ]  (as f16x2)
    //   a_regs[1] = smem[t/4][ (t%4)*8 + 4 : (t%4)*8 + 8 ]
    //   a_regs[2] = smem[t/4 + 8][ ... ]
    //   a_regs[3] = smem[t/4 + 8][ ... ]
    std::string emit_load_a_fragment(
        const std::string& smem_ptr,
        const std::string& offset_reg,
        const MMAFragments& frags) const;

    // Emit a load of B fragment from shared memory.
    // B is [16,8] f16.
    std::string emit_load_b_fragment(
        const std::string& smem_ptr,
        const std::string& offset_reg,
        const MMAFragments& frags) const;

    // Emit a store of D (result) fragment to global memory.
    std::string emit_store_d_fragment(
        const std::string& global_ptr,
        const std::string& offset_reg,
        const MMAFragments& frags) const;

    // Emit a zero-initialize of the accumulator (C = 0).
    std::string emit_zero_accumulator(const MMAFragments& frags) const;

    // Emit a full tiled matmul loop using MMA:
    //   for k_outer in range(0, K, k_tile):
    //     load A tile [M, k_tile] from global → shared
    //     load B tile [k_tile, N] from global → shared
    //     sync
    //     for k_inner in range(0, k_tile, 16):
    //       load A fragment from shared
    //       load B fragment from shared
    //       mma.sync
    //     sync
    //   store D to global
    //
    // This is the complete GEMM kernel using tensor cores.
    std::string emit_tiled_matmul_kernel(
        u32 M, u32 N, u32 K,
        DType dtype,
        const std::string& kernel_name,
        u32 m_tile = 64,
        u32 n_tile = 64,
        u32 k_tile = 32) const;

    // Allocate fragment registers for m16n8k16.
    MMAFragments allocate_fragments(u32& next_reg_id) const;

private:
    // Helper: generate register name.
    std::string reg(u32 id, DType dt) const;
};

// Hardware capabilities for tensor cores.
struct TensorCoreCaps {
    bool has_ampere_mma = false;     // sm_80+
    bool has_hopper_wgmma = false;  // sm_90+
    bool has_tma = false;           // sm_90+ (cp.async.bulk.tensor)
    bool has_async_barrier = false; // sm_90+ (mbarrier)

    // Peak tensor core FLOPs per second (varies by dtype).
    double peak_f16_tflops = 0;
    double peak_bf16_tflops = 0;
    double peak_i8_tflops = 0;
    double peak_fp8_tflops = 0;

    // MMA tile shapes supported.
    std::vector<MMAShape> supported_tiles;

    static TensorCoreCaps ampere();   // sm_80 (A100)
    static TensorCoreCaps hopper();   // sm_90 (H100)
    static TensorCoreCaps blackwell(); // sm_100 (B200)
};

} // namespace cg
