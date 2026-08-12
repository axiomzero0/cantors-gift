// optimization/reduction_opt.hpp - reduction optimization
//
// Optimizes reduction patterns:
//   - reduce_sum(x) followed by elementwise div by constant -> reduce_mean
//   - chained reductions collapse
//   - reduction over a contiguous dim can use tree reduction
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class ReductionOptimizationPass : public Pass {
public:
    std::string name() const override { return "reduction_opt"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
