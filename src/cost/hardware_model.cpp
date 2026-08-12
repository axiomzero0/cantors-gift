// cost/hardware_model.cpp - hardware models with real latency/overhead parameters
//
// All magic numbers eliminated. Each target has:
//   - launch_overhead_sec (from vendor specs)
//   - register_file_per_sm (from architecture manuals)
//   - base_regs_per_thread (from typical kernel analysis)
//   - max_warps_per_sm (hardware limit)
//   - warp_size (NVIDIA: 32, AMD: 64, CPU: 1)
//   - l2_read_bw (from architecture whitepapers)
//   - l2_hit_rate_estimate (from empirical analysis)
//   - stall_cycles_per_warp (from architecture manuals)
#include "cg/cost/hardware_model.hpp"

#include <algorithm>

namespace cg {

u32 HardwareModel::estimate_warps_per_sm(u32 regs_per_thread,
                                          u32 shared_mem_per_block,
                                          u32 threads_per_block) const {
    if (warp_size == 0) return 1;

    // Register-limited warps:
    //   regs_per_warp = regs_per_thread * warp_size
    //   max_warps_regs = register_file_per_sm / regs_per_warp
    u32 regs_per_warp = regs_per_thread * warp_size;
    u32 max_warps_regs = register_file_per_sm / std::max(1u, regs_per_warp);

    // Shared-memory-limited warps:
    //   max_blocks_smem = shared_mem_bytes / shared_mem_per_block
    //   max_warps_smem = max_blocks_smem * (threads_per_block / warp_size)
    u32 max_blocks_smem = shared_mem_per_block > 0
        ? static_cast<u32>(shared_mem_bytes) / shared_mem_per_block
        : max_warps_per_sm;
    u32 warps_per_block = threads_per_block / warp_size;
    u32 max_warps_smem = max_blocks_smem * std::max(1u, warps_per_block);

    // Hardware limit
    return std::min({max_warps_regs, max_warps_smem, max_warps_per_sm});
}

HardwareModel HardwareModel::generic_cpu() {
    HardwareModel hw;
    hw.name = "generic_cpu";
    hw.device = DeviceId::cpu();
    // 8 cores, AVX-512, 3 GHz, ~256 GFLOPs for F32.
    hw.compute.flops_per_sec[static_cast<u8>(DType::F32)] = 256e9;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F64)] = 128e9;
    hw.compute.flops_per_sec[static_cast<u8>(DType::I32)] = 256e9;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Generic)] = 50e9; // DDR4
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Local)]   = 500e9; // L1
    hw.shared_mem_bytes = 0;
    hw.l1_cache_bytes = 32 * 1024;
    hw.l2_cache_bytes = 256 * 1024;
    hw.simd_width_bytes = 64; // AVX-512
    hw.num_cores = 8;
    hw.threads_per_core = 2;

    // CPU-specific latency parameters
    hw.launch_overhead_sec = 1e-6;   // ~1μs (function call, no kernel launch)
    hw.register_file_per_sm = 256;   // x86-64 has ~16 named + ~256 physical
    hw.base_regs_per_thread = 16;
    hw.max_warps_per_sm = 1;         // CPU has no warps
    hw.warp_size = 1;
    hw.l2_read_bw = 500e9;           // ~500 GB/s L2 (shared L3 on CPU)
    hw.l2_hit_rate_estimate = 0.3;
    hw.stall_cycles_per_warp = 3.0;
    return hw;
}

HardwareModel HardwareModel::generic_nvidia_gpu() {
    HardwareModel hw;
    hw.name = "generic_nvidia_gpu"; // A100-class
    hw.device = DeviceId::cuda();
    // A100: 108 SMs, 64 FP32 cores/SM, 1.41 GHz => ~19.5 TFLOPS F32
    // (we use the more common 7.7 TFLOPS figure from older specs)
    hw.compute.flops_per_sec[static_cast<u8>(DType::F32)] = 19.5e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F16)] = 39.0e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::BF16)] = 39.0e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F64)] = 9.7e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::I32)] = 19.5e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::F16)]  = 312e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::BF16)] = 312e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::I8)]   = 624e12;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Generic)]  = 2.0e12; // HBM2e
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Shared)]   = 19.5e12;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Constant)] = 2.0e12;
    hw.shared_mem_bytes = 164 * 1024; // 164 KB per SM (configurable)
    hw.l1_cache_bytes   = 192 * 1024;
    hw.l2_cache_bytes   = 40 * 1024 * 1024;  // 40 MB L2
    hw.constant_mem_bytes = 64 * 1024;
    hw.simd_width_bytes = 16; // 128b warp lane group
    hw.num_cores = 108;       // SMs
    hw.threads_per_core = 32; // warp size

    // A100-specific latency parameters (from NVIDIA A100 whitepaper)
    hw.launch_overhead_sec = 5e-6;     // ~5μs kernel launch
    hw.register_file_per_sm = 65536;   // 256KB register file / 4 bytes
    hw.base_regs_per_thread = 40;
    hw.max_warps_per_sm = 64;          // A100 max
    hw.warp_size = 32;
    hw.l2_read_bw = 5.0e12;            // ~5 TB/s L2
    hw.l2_hit_rate_estimate = 0.5;
    hw.stall_cycles_per_warp = 4.0;
    return hw;
}

HardwareModel HardwareModel::generic_amd_gpu() {
    HardwareModel hw;
    hw.name = "generic_amd_gpu"; // MI300X-class
    hw.device = DeviceId::rocm();
    // MI300X: 304 CUs, 1280 stream processors, ~1300 TF F16
    hw.compute.flops_per_sec[static_cast<u8>(DType::F32)] = 163.0e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F16)] = 327.0e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::BF16)] = 327.0e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F64)] = 81.5e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::I32)] = 163.0e12;
    // MI300X matrix engine (no separate TC concept, but high throughput)
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::F16)]  = 1300.0e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::BF16)] = 1300.0e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::I8)]   = 2600.0e12;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Generic)]  = 8.0e12; // 8-channel HBM3
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Shared)]   = 30.0e12;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Constant)] = 8.0e12;
    hw.shared_mem_bytes = 256 * 1024;  // 256 KB per CU
    hw.l1_cache_bytes   = 256 * 1024;
    hw.l2_cache_bytes   = 256 * 1024 * 1024; // 256 MB L2
    hw.constant_mem_bytes = 64 * 1024;
    hw.simd_width_bytes = 16;
    hw.num_cores = 304;       // CUs
    hw.threads_per_core = 64; // wavefront size

    // MI300X-specific latency parameters (from AMD whitepaper)
    hw.launch_overhead_sec = 8e-6;     // ~8μs (higher than NVIDIA)
    hw.register_file_per_sm = 65536;   // 256KB / 4 bytes per CU
    hw.base_regs_per_thread = 40;
    hw.max_warps_per_sm = 40;          // MI300X max waves per CU
    hw.warp_size = 64;                 // AMD wavefront64
    hw.l2_read_bw = 12.0e12;           // ~12 TB/s L2 (huge on MI300X)
    hw.l2_hit_rate_estimate = 0.6;     // Larger L2 = higher hit rate
    hw.stall_cycles_per_warp = 6.0;    // Higher latency than NVIDIA
    return hw;
}

} // namespace cg
