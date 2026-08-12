// cost/estimator.hpp - analytical cost estimator
//
// Given a Tensor IR program (represented abstractly here), a Schedule, and a
// HardwareModel, produce an estimate of:
//   - FLOPs
//   - memory traffic (bytes) per memory space
//   - launch / overhead count
//   - rough runtime estimate
//
// This is NOT a full performance model. It is a cheap analytical filter used
// to prune obviously-bad schedule candidates before benchmarking.
#pragma once

#include "cg/cost/hardware_model.hpp"
#include "cg/core/dtype.hpp"
#include "cg/ir/module.hpp"
#include "cg/schedule/schedule.hpp"

#include <unordered_map>

namespace cg {

struct CostEstimate {
    u64   flops = 0;
    u64   bytes_global = 0;
    u64   bytes_shared = 0;
    u64   bytes_constant = 0;
    u32   kernel_launches = 0;
    u32   parallel_axes = 0;
    double estimated_runtime_sec = 0.0;
};

class CostEstimator {
public:
    explicit CostEstimator(HardwareModel hw) : hw_(std::move(hw)) {}

    const HardwareModel& hardware() const { return hw_; }

    // Estimate the cost of running the entire module under `schedule`.
    CostEstimate estimate(const Module& m, const Schedule& schedule) const;

    // Estimate the cost of a single matmul-shaped op (M, K, N) at `dt`.
    // This is the kernel-level estimator used by the autotuner.
    CostEstimate estimate_matmul(i64 M, i64 K, i64 N, DType dt,
                                 const Schedule& s) const;

    // Prune a ScheduleSpace: returns a list of (schedule, estimate) pairs
    // sorted by estimated runtime, keeping only the top `k` candidates.
    std::vector<std::pair<Schedule, CostEstimate>>
    rank(const ScheduleSpace& space, usize k) const;

private:
    HardwareModel hw_;
};

} // namespace cg
