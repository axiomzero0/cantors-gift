// analysis/parallelism_analysis.hpp - per-op and per-region parallelism estimate
//
// Estimates how much parallel work each operation exposes. Used by the
// scheduler and by the cost model.
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/ir/module.hpp"

#include <unordered_map>

namespace cg {

struct ParallelismInfo {
    // Estimated number of independent work items (e.g. tiles, output elements).
    u64 independent_items = 0;
    // Whether the op has a reduction dimension (sequential dependency).
    bool has_reduction = false;
    // Reduction length (if applicable).
    u64 reduction_length = 0;
    // Estimated parallelism ratio = independent_items / max(reduction_length, 1)
    double parallelism_ratio = 0.0;
};

class ParallelismAnalysis : public AnalysisBase {
public:
    explicit ParallelismAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    const ParallelismInfo& info_of(const Operation& op) const {
        static ParallelismInfo empty{};
        auto it = per_op_.find(op.id);
        return it != per_op_.end() ? it->second : empty;
    }

    void invalidate() { per_op_.clear(); }

private:
    void compute();

    AnalysisManager& am_;
    std::unordered_map<u32, ParallelismInfo> per_op_;
};

} // namespace cg
