// analysis/global_alias_analysis.hpp - tensor-specific alias analysis
//
// Tracks relationships that matter for tensor programs:
//   - view-of:   B = view(A)              -> B aliases A
//   - slice-of:  B = A[i:j]               -> B aliases a subset of A
//   - broadcast-of: B = broadcast(A, ...) -> B's logical indices map into A
//
// Used by:
//   - fusion (to avoid breaking alias-based in-place patterns)
//   - memory planning (to know when two values can share storage)
//   - parallelism analysis (disjoint slices can run in parallel)
//
// In the foundational implementation, two pure-SSA values never alias
// unless one is a transpose/reshape/slice of the other. This is sound and
// lets downstream optimizations make the common-case decision quickly.
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/ir/module.hpp"

#include <unordered_map>
#include <unordered_set>

namespace cg {

enum class TensorAliasKind : u8 {
    NoAlias,
    MayAlias,
    MustAlias,
    ViewOf,
    SliceOf,
    BroadcastOf,
};

class GlobalAliasAnalysis : public AnalysisBase {
public:
    explicit GlobalAliasAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    TensorAliasKind alias(Value a, Value b) const;

    // The set of values that are views/slices of `v` (direct children).
    const std::unordered_set<ValueId>& views_of(Value v) const {
        static const std::unordered_set<ValueId> empty;
        auto it = views_.find(v.id());
        return it != views_.end() ? it->second : empty;
    }

    void invalidate() {
        views_.clear();
        defining_view_.clear();
    }

private:
    void compute();

    AnalysisManager& am_;
    // parent -> set of child values that are views/slices
    std::unordered_map<ValueId, std::unordered_set<ValueId>> views_;
    // child -> parent (the value it is a view of)
    std::unordered_map<ValueId, ValueId> defining_view_;
};

} // namespace cg
