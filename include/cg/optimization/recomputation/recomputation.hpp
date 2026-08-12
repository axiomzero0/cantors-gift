// optimization/recomputation.hpp - recomputation vs materialization pass
//
// Uses ReuseAnalysis to decide whether to materialize or recompute each
// multi-consumer value. For values marked "Recompute", it duplicates the
// defining op at each consumer (removing the need to store the intermediate).
// For values marked "Materialize", it ensures the value is stored to memory.
//
// This pass runs after fusion and before memory planning, so memory planning
// sees the final set of live tensors.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class RecomputationPass : public Pass {
public:
    std::string name() const override { return "recomputation"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
