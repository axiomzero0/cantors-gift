// cost/hardware_profile.hpp - optimizer-facing hardware profile
//
// A HardwareProfile is a *purely descriptive* view of a target. It does not
// depend on any backend class and is consumed by the optimizer / scheduler
// to make target-aware decisions without #including backend headers.
//
// The MachineBackend constructs a HardwareProfile from its concrete
// HardwareModel; the optimizer only ever sees the profile.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/cost/hardware_model.hpp"

#include <string>
#include <tuple>
#include <vector>

namespace cg {

class HardwareProfile {
public:
    std::string name;
    DeviceId    device;

    u32 vector_width_bytes = 16;
    u32 cache_line_bytes   = 64;
    u64 l1_bytes           = 32 * 1024;
    u64 l2_bytes           = 256 * 1024;
    u64 shared_memory_bytes = 0;
    u64 constant_memory_bytes = 0;
    u32 num_cores           = 1;
    u32 threads_per_core    = 1;
    u32 warp_width          = 1;     // GPU only; 1 on CPU

    // Supported vector dtypes.
    std::vector<DType> vector_dtypes;

    // Tensor-core / MMA tile shapes (M, N, K) per dtype.
    std::vector<std::tuple<DType, u32, u32, u32>> tensor_core_tiles;

    // Peak rates (for the cost model).
    double peak_flops_f32 = 0.0;
    double peak_flops_f16 = 0.0;
    double peak_flops_i8  = 0.0;
    double peak_bytes_per_sec_global = 0.0;
    double peak_bytes_per_sec_shared = 0.0;

    // Build a profile from a HardwareModel.
    static HardwareProfile from_model(const HardwareModel& hw) {
        HardwareProfile p;
        p.name = hw.name;
        p.device = hw.device;
        p.vector_width_bytes = static_cast<u32>(hw.simd_width_bytes);
        p.l1_bytes = hw.l1_cache_bytes;
        p.l2_bytes = hw.l2_cache_bytes;
        p.shared_memory_bytes = hw.shared_mem_bytes;
        p.constant_memory_bytes = hw.constant_mem_bytes;
        p.num_cores = hw.num_cores;
        p.threads_per_core = hw.threads_per_core;
        p.warp_width = (hw.device.kind == DeviceId::Kind::CPU) ? 1 : 32;
        p.peak_flops_f32 = hw.peak_flops(DType::F32, false);
        p.peak_flops_f16 = hw.peak_flops(DType::F16, true);
        p.peak_flops_i8  = hw.peak_flops(DType::I8,  true);
        p.peak_bytes_per_sec_global = hw.memory.get(MemorySpace::Generic);
        p.peak_bytes_per_sec_shared = hw.memory.get(MemorySpace::Shared);
        p.vector_dtypes = {DType::F32, DType::F16, DType::BF16, DType::I8,
                           DType::I32};
        if (hw.supports_tensor_core(DType::F16))
            p.tensor_core_tiles.push_back({DType::F16, 16, 8, 16});
        if (hw.supports_tensor_core(DType::BF16))
            p.tensor_core_tiles.push_back({DType::BF16, 16, 8, 16});
        if (hw.supports_tensor_core(DType::I8))
            p.tensor_core_tiles.push_back({DType::I8, 8, 8, 16});
        return p;
    }

    bool supports_dtype(DType dt) const {
        for (auto d : vector_dtypes) if (d == dt) return true;
        return false;
    }

    bool supports_tensor_core(DType dt) const {
        for (auto& [d, _, _2, _3] : tensor_core_tiles)
            if (d == dt) return true;
        return false;
    }

    bool is_gpu() const {
        return device.kind == DeviceId::Kind::CUDA ||
               device.kind == DeviceId::Kind::ROCM ||
               device.kind == DeviceId::Kind::METAL;
    }
};

} // namespace cg
