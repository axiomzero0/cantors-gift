// cost/hardware_model.cpp - default hardware models
#include "cg/cost/hardware_model.hpp"

namespace cg {

HardwareModel HardwareModel::generic_cpu() {
    HardwareModel hw;
    hw.name = "generic_cpu";
    hw.device = DeviceId::cpu();
    // 16 AVX-512 lanes, 8 cores, 3 GHz, ~256 GFLOPs for F32.
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
    return hw;
}

HardwareModel HardwareModel::generic_nvidia_gpu() {
    HardwareModel hw;
    hw.name = "generic_nvidia_gpu";
    hw.device = DeviceId::cuda();
    // ~80 SMs, 64 FP32 cores/SM, 1.5 GHz => ~7.7 TFLOPS F32 (rough A100-like).
    hw.compute.flops_per_sec[static_cast<u8>(DType::F32)] = 7.7e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F16)] = 15.4e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::BF16)] = 15.4e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::F64)] = 3.85e12;
    hw.compute.flops_per_sec[static_cast<u8>(DType::I32)] = 7.7e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::F16)]  = 312e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::BF16)] = 312e12;
    hw.tensor_core_flops_per_sec[static_cast<u8>(DType::I8)]   = 624e12;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Generic)]  = 1.5e12; // HBM2e
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Shared)]   = 19.5e12;
    hw.memory.bytes_per_sec[static_cast<u8>(MemorySpace::Constant)] = 1.5e12;
    hw.shared_mem_bytes = 164 * 1024; // 164 KB per SM (configurable)
    hw.l1_cache_bytes   = 192 * 1024;
    hw.l2_cache_bytes   = 40 * 1024 * 1024;
    hw.constant_mem_bytes = 64 * 1024;
    hw.simd_width_bytes = 16; // 128b warp lane group
    hw.num_cores = 108;       // SMs
    hw.threads_per_core = 32; // warp size
    return hw;
}

} // namespace cg
