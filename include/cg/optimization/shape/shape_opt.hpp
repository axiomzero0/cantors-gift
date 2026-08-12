// optimization/shape_opt.hpp - shape optimization
//
// Simplifies shape expressions across the module using the constraint solver.
//   - Collapses ceildiv(N, 32) -> N/32 when N % 32 == 0 is provable.
//   - Propagates constant shapes through reshape/broadcast.
//   - Removes redundant broadcast/reshape pairs.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class ShapeOptimizationPass : public Pass {
public:
    std::string name() const override { return "shape_opt"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
