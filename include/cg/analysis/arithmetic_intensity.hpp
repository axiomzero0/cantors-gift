// analysis/arithmetic_intensity.hpp - per-op and per-region FLOPs / bytes
//
// Arithmetic intensity = FLOPs / bytes-moved.
//
// We distinguish THREE levels of arithmetic intensity:
//
//   1. GRAPH arithmetic intensity
//      Total FLOPs of the (unfused) IR graph divided by total bytes moved
//      between ops. This is the "naive" intensity the user would compute
//      from the source program. It ignores fusion, ignores scheduling,
//      ignores cache effects. Useful for: "should we even bother fusing?"
//
//   2. KERNEL arithmetic intensity
//      After fusion + scheduling, each kernel has its own FLOPs and its
//      own bytes-touched. The kernel intensity is the intensity of the
//      fused kernel as it would be launched. This already accounts for
//      fusion eliminating intermediate traffic.
//
//   3. EFFECTIVE arithmetic intensity
//      The intensity that actually determines performance: FLOPs divided
//      by bytes that *miss* every level of the cache hierarchy. This
//      accounts for:
//        - L2 cache reuse (B matrix reused across M/m_tile blocks)
//        - Shared-memory reuse (A,B tiles loaded once per K-tile, reused
//          across the reduction dimension)
//        - Register reuse (accumulator held in registers, no traffic)
//        - Coalesced global-memory transactions (vectorized loads)
//
// Classifies each operation (and the module as a whole) as:
//   - memory-bound   (low intensity, dominated by bytes)
//   - compute-bound  (high intensity, dominated by FLOPs)
//   - launch-bound   (few FLOPs and bytes, dominated by launch overhead)
//   - latency-bound  (small, sequential, no parallelism)
//
// These classifications drive the global fusion / vectorization / scheduling
// decisions.
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/module.hpp"
#include "cg/schedule/schedule.hpp"

#include <unordered_map>

namespace cg {

enum class BoundClass : u8 {
    MemoryBound,
    ComputeBound,
    LaunchBound,
    LatencyBound,
    Balanced,
};

struct OpIntensity {
    u64 flops = 0;
    u64 bytes_read = 0;
    u64 bytes_written = 0;
    double intensity = 0.0;       // flops / (bytes_read + bytes_written)
    BoundClass bound = BoundClass::Balanced;

    u64 total_bytes() const { return bytes_read + bytes_written; }
};

// A reusable-cache estimate used to compute effective intensity.
struct CacheReuse {
    // Fraction of B-matrix bytes that hit in L2 (0..1).
    // For matmul: B[K,N] is reused across M/m_tile CTA blocks. If K*N*elem_size
    // fits in L2, the hit rate approaches 1.0; otherwise it scales as
    // L2_size / B_size.
    double l2_hit_rate_b = 0.0;

    // Fraction of A-matrix bytes that hit in L2.
    // A[M,K] is reused across N/n_tile CTA blocks. Symmetric to B.
    double l2_hit_rate_a = 0.0;

    // Shared-memory reuse factor: how many times each global byte is reused
    // from shared memory. For matmul with k_tile reduction, this is K/k_tile.
    // The bytes that ACTUALLY go to global memory = bytes / reuse_factor.
    double shared_reuse_factor = 1.0;

    // Register reuse factor: the accumulator (M_tile * N_tile elements) lives
    // entirely in registers and never touches memory. We don't charge any
    // bytes for it.
    bool accumulator_in_registers = false;
};

// Per-kernel intensity after fusion + scheduling.
struct KernelIntensity {
    u64 flops = 0;
    u64 graph_bytes = 0;          // bytes if no fusion, no cache
    u64 kernel_bytes = 0;         // bytes after fusion (intermediates dropped)
    u64 effective_bytes = 0;      // bytes that actually miss every cache level

    double graph_intensity() const {
        return graph_bytes > 0 ? double(flops) / double(graph_bytes) : 0.0;
    }
    double kernel_intensity() const {
        return kernel_bytes > 0 ? double(flops) / double(kernel_bytes) : 0.0;
    }
    double effective_intensity() const {
        return effective_bytes > 0 ? double(flops) / double(effective_bytes) : 0.0;
    }

    BoundClass bound = BoundClass::Balanced;
};

class ArithmeticIntensityAnalysis : public AnalysisBase {
public:
    explicit ArithmeticIntensityAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    const OpIntensity& intensity_of(const Operation& op) const {
        static OpIntensity empty{};
        auto it = per_op_.find(op.id);
        return it != per_op_.end() ? it->second : empty;
    }

    BoundClass module_bound_class() const { return module_bound_; }
    double module_intensity() const { return module_intensity_; }
    // Effective module intensity: total flops / total effective bytes
    // (after L2/shared/register reuse). This is the number that actually
    // determines whether the *fused* kernel is memory- or compute-bound.
    double module_effective_intensity() const { return module_effective_intensity_; }

    u64 total_flops() const { return total_flops_; }
    u64 total_bytes() const { return total_bytes_; }
    u64 total_effective_bytes() const { return total_effective_bytes_; }

    void invalidate() {
        per_op_.clear();
        kernel_intensities_.clear();
        module_bound_ = BoundClass::Balanced;
        module_intensity_ = 0.0;
        module_effective_intensity_ = 0.0;
        total_flops_ = 0;
        total_bytes_ = 0;
        total_effective_bytes_ = 0;
    }

    void set_hardware(HardwareModel hw) {
        hw_ = std::move(hw);
        compute();
    }

    // Compute the effective bytes for a matmul (M, K, N) given a schedule
    // and the hardware's L2/shared memory parameters.
    //
    // effective_bytes = (A_bytes * (1 - l2_hit_a) + B_bytes * (1 - l2_hit_b))
    //                  / shared_reuse_factor
    //                  + C_bytes (always written once)
    //
    // The accumulator lives in registers, so we charge ZERO bytes for it
    // (the M_tile * N_tile accumulator is never spilled to memory).
    static u64 matmul_effective_bytes(u64 M, u64 K, u64 N, DType dt,
                                       const Schedule& s,
                                       const HardwareModel& hw);

    // Compute the kernel-level bytes for a matmul (M, K, N) under a schedule.
    // This is the bytes the kernel actually touches in global memory after
    // shared-memory reuse, but BEFORE L2 effects.
    static u64 matmul_kernel_bytes(u64 M, u64 K, u64 N, DType dt,
                                    const Schedule& s);

    // Compute the L2 hit rate for a tensor of `tensor_bytes`.
    // If the tensor fits entirely in L2, the hit rate is ~0.8 (some conflict
    // misses). Otherwise, it scales linearly with L2_size / tensor_size.
    static double l2_hit_rate(u64 tensor_bytes, const HardwareModel& hw);

    const HardwareModel& hardware() const { return hw_; }

    // Per-kernel intensity lookup (keyed by a kernel id, typically the
    // id of the matmul op or fused group root).
    const std::unordered_map<u32, KernelIntensity>& kernel_intensities() const {
        return kernel_intensities_;
    }

private:
    void compute();
    BoundClass classify(u64 flops, u64 bytes, usize parallelism_estimate);

    AnalysisManager& am_;
    std::unordered_map<u32, OpIntensity> per_op_;
    std::unordered_map<u32, KernelIntensity> kernel_intensities_;
    BoundClass module_bound_ = BoundClass::Balanced;
    double module_intensity_ = 0.0;
    double module_effective_intensity_ = 0.0;
    u64 total_flops_ = 0;
    u64 total_bytes_ = 0;
    u64 total_effective_bytes_ = 0;
    HardwareModel hw_; // defaults to generic CPU
};

} // namespace cg
