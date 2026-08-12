// optimization/layout_opt.hpp - global layout optimization
//
// Chooses layouts for every value such that total graph cost is minimized.
// Uses the dataflow analysis to see producer-consumer chains and the cost
// model to evaluate layout choices.
//
// Foundational heuristic:
//   - If a transpose feeds a matmul, prefer the matmul-friendly layout and
//     drop the transpose.
//   - If two consumers prefer different layouts, pick the one that minimizes
//     total cost (producer + all consumers).
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class LayoutOptimizationPass : public Pass {
public:
    std::string name() const override { return "layout_opt"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
