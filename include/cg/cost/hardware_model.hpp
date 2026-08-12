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

    // ---- Latency / overhead parameters (no magic numbers) ----

    // Kernel launch overhead in seconds. Varies by ~5x across targets:
    //   H100: ~2μs, A100: ~5μs, MI300X: ~8μs, CPU: ~1μs
    double launch_overhead_sec = 5e-6;

    // Register file size per SM (or per core on CPU), in registers.
    //   A100: 65536, H100: 65536, MI300X: 65536
    u32 register_file_per_sm = 65536;

    // Base register usage per thread (before any fusion).
    //   Typical: 20-40 depending on kernel complexity.
    u32 base_regs_per_thread = 32;

    // Maximum warps per SM (hardware limit).
    //   A100: 64, H100: 64, MI300X: 40
    u32 max_warps_per_sm = 64;

    // Warp size (threads per warp).
    //   NVIDIA: 32, AMD: 64, CPU: 1 (no warps)
    u32 warp_size = 32;

    // L2 cache read bandwidth (bytes/sec). Typically 5-10x HBM bandwidth.
    //   H100: ~12 TB/s, A100: ~5 TB/s
    double l2_read_bw = 0;

    // L2 cache hit rate estimate for the second-most-touched tensor
    // (the one that gets reused across CTA blocks). 0.0 = no hit, 1.0 = always.
    double l2_hit_rate_estimate = 0.5;

    // Stall cycles per warp when occupancy is too low to hide latency.
    //   A100: ~4 cycles, H100: ~2 cycles
    double stall_cycles_per_warp = 4.0;

    // ---- Derived properties ----

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

    // Compute the roofline ridge: peak_flops / peak_memory_bw.
    // This is the arithmetic intensity at which a kernel transitions from
    // memory-bound to compute-bound. For H100 F16 TC: 989e12 / 3.35e12 ≈ 295.
    double roofline_ridge(DType dt = DType::F32,
                          bool use_tensor_core = false) const {
        double peak = peak_flops(dt, use_tensor_core);
        double bw = memory.get(MemorySpace::Generic);
        if (bw <= 0) return 16.0; // fallback
        return peak / bw;
    }

    // Estimate warps per SM given register and shared memory constraints.
    // Returns the minimum of register-limited, shared-mem-limited, and hw max.
    u32 estimate_warps_per_sm(u32 regs_per_thread,
                               u32 shared_mem_per_block,
                               u32 threads_per_block) const;

    // Factory: a generic CPU model.
    static HardwareModel generic_cpu();
    // Factory: a generic NVIDIA GPU model (A100-class).
    static HardwareModel generic_nvidia_gpu();
    // Factory: a generic AMD GPU model (MI300X-class).
    static HardwareModel generic_amd_gpu();
};

} // namespace cg
