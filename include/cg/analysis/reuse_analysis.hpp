// analysis/reuse_analysis.hpp - global materialize-vs-recompute analysis
//
// For every value with multiple consumers, decide whether to:
//   - materialize it (store once, read many times)
//   - recompute it (compute again at each consumer)
//
// Heuristic (foundational, will be refined by the cost model):
//   - single consumer       -> always materialize (or fuse into consumer)
//   - cheap op + many cons  -> recompute
//   - expensive op + many   -> materialize
//   - large output + few    -> recompute (to save memory)
//
// Output: a ReuseDecision per value.
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/ir/module.hpp"

#include <unordered_map>

namespace cg {

enum class ReuseDecision : u8 {
    Materialize,
    Recompute,
    Fuse,         // single consumer -> fuse in
    Undecided,
};

struct ReuseInfo {
    ReuseDecision decision = ReuseDecision::Undecided;
    usize num_consumers = 0;
    u64   flops_per_producer = 0;
    u64   bytes_per_producer = 0;
    double estimated_savings_sec = 0.0;
};

class ReuseAnalysis : public AnalysisBase {
public:
    explicit ReuseAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    const ReuseInfo& info_of(Value v) const {
        static ReuseInfo empty{};
        auto it = per_value_.find(v.id());
        return it != per_value_.end() ? it->second : empty;
    }

    void invalidate() { per_value_.clear(); }

private:
    void compute();

    AnalysisManager& am_;
    std::unordered_map<ValueId, ReuseInfo> per_value_;
};

} // namespace cg
