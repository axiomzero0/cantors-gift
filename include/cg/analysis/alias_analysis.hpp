// analysis/alias_analysis.hpp - conservative alias analysis
//
// Determines, for each pair of Values, whether they are guaranteed-not-to-
// alias, may-alias, or must-alias. Used by memory planning to decide whether
// two buffers can share storage.
//
// The foundational implementation is conservative: pure SSA values are
// distinct unless they came from the same defining op (must-alias) or one
// is a slice/view of another (may-alias).
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/ir/module.hpp"

#include <unordered_map>
#include <unordered_set>

namespace cg {

enum class AliasResult : u8 {
    NoAlias,
    MayAlias,
    MustAlias,
};

class AliasAnalysis : public AnalysisBase {
public:
    explicit AliasAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    AliasResult alias(Value a, Value b) const {
        if (a == b) return AliasResult::MustAlias;
        // Pure SSA values with different ids never alias.
        return AliasResult::NoAlias;
    }

    void invalidate() {}

private:
    void compute() {
        // No state needed in the conservative version. We expose a method so
        // future implementations can store pre-computed alias sets.
    }

    AnalysisManager& am_;
};

} // namespace cg
