// analysis/arithmetic_intensity.hpp - per-op and per-region FLOPs / bytes
//
// Arithmetic intensity = FLOPs / bytes-moved.
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
#include "cg/ir/module.hpp"

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

    u64 total_flops() const { return total_flops_; }
    u64 total_bytes() const { return total_bytes_; }

    void invalidate() {
        per_op_.clear();
        module_bound_ = BoundClass::Balanced;
        module_intensity_ = 0.0;
        total_flops_ = 0;
        total_bytes_ = 0;
    }

private:
    void compute();
    BoundClass classify(u64 flops, u64 bytes, usize parallelism_estimate);

    AnalysisManager& am_;
    std::unordered_map<u32, OpIntensity> per_op_;
    BoundClass module_bound_ = BoundClass::Balanced;
    double module_intensity_ = 0.0;
    u64 total_flops_ = 0;
    u64 total_bytes_ = 0;
};

} // namespace cg
