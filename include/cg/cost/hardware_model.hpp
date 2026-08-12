// cost/hardware_model.hpp - hardware specification
//
// A HardwareModel describes the *static* characteristics of a target:
// compute throughput per dtype, memory bandwidth by memory space, cache sizes,
// vector widths, tensor-core shapes, and shared-memory size. The cost model
// uses this to estimate runtime without benchmarking.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/ir/type.hpp"
#include "cg/layout/layout.hpp"

#include <unordered_map>

namespace cg {

struct ComputeThroughput {
    // ops/sec for each dtype (e.g. FMA throughput).
    std::unordered_map<u8, double> flops_per_sec;

    double get(DType dt) const {
        auto it = flops_per_sec.find(static_cast<u8>(dt));
        return it != flops_per_sec.end() ? it->second : 0.0;
    }
};

struct MemoryBandwidth {
    // bytes/sec for each memory space.
    std::unordered_map<u8, double> bytes_per_sec;

    double get(MemorySpace ms) const {
        auto it = bytes_per_sec.find(static_cast<u8>(ms));
        return it != bytes_per_sec.end() ? it->second : 0.0;
    }
};

class HardwareModel {
public:
    std::string name;
    DeviceId device;

    ComputeThroughput compute;
    MemoryBandwidth memory;

    // Shared / L1 / L2 / constant cache sizes (bytes).
    u64 shared_mem_bytes = 0;
    u64 l1_cache_bytes   = 0;
    u64 l2_cache_bytes   = 0;
    u64 constant_mem_bytes = 0;

    // Vector widths (bytes) per dtype family.
    u64 simd_width_bytes = 16;

    // Tensor-core / MMA peak (FLOPs/sec), per dtype pair.
    std::unordered_map<u8, double> tensor_core_flops_per_sec;

    // Number of hardware threads / SMs / cores.
    u32 num_cores = 1;
    u32 threads_per_core = 1;

    // Compute peak FLOPs/sec for `dt` (scalar or tensor-core).
    double peak_flops(DType dt, bool use_tensor_core = false) const {
        if (use_tensor_core) {
            auto it = tensor_core_flops_per_sec.find(static_cast<u8>(dt));
            if (it != tensor_core_flops_per_sec.end()) return it->second;
        }
        return compute.get(dt);
    }

    // True iff the target has tensor-core / MMA throughput for `dt`.
    bool supports_tensor_core(DType dt) const {
        return tensor_core_flops_per_sec.count(static_cast<u8>(dt)) > 0;
    }

    // Factory: a generic CPU model.
    static HardwareModel generic_cpu();
    // Factory: a generic NVIDIA GPU model.
    static HardwareModel generic_nvidia_gpu();
};

} // namespace cg
