// optimization/algebraic.hpp - algebraic simplification
//
// Applies algebraic identities that canonicalization doesn't catch:
//   - x - x -> 0
//   - x * x -> x (only if we had a sq op; we don't, so skip)
//   - neg(neg(x)) -> x
//   - transpose(transpose(x)) -> x   (already in canonicalize, kept for safety)
//   - reshape(reshape(x, s1), s2) -> reshape(x, s2)  when s1 is the original
//   - broadcast(x, same_shape) -> x
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class AlgebraicSimplificationPass : public Pass {
public:
    std::string name() const override { return "algebraic"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
