// optimization/canonicalize.hpp - canonicalization pass
//
// Normalizes the IR so that downstream pattern matchers and CSE see a single
// syntactic form for mathematically equivalent operations.
//
//   - x + 0 -> x
//   - 0 + x -> x
//   - x * 1 -> x
//   - x * 0 -> 0 (if x is a pure tensor op)
//   - x - 0 -> x
//   - transpose(transpose(x)) -> x
//   - reshape(reshape(x)) -> reshape(x)
//   - commutative ops: order operands by a stable key
//   - constant folding of arithmetic on constant tensors
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class CanonicalizePass : public Pass {
public:
    std::string name() const override { return "canonicalize"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
