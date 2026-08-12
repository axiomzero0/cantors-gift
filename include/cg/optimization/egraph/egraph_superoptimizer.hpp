// optimization/egraph_superoptimizer.hpp - tensor-aware e-graph superoptimizer
//
// The e-graph is used as a *superoptimizer* for specific tensor
// sub-expressions. The pattern is:
//
//     Tensor IR
//        |
//        v
//   extract optimizable region (pure, acyclic sub-DAG)
//        |
//        v
//      e-graph
//        |
//        v
//     saturate with tensor rewrite rules
//        |
//        v
//   extract (tensor-aware cost: FLOPs + memory traffic)
//        |
//        v
//     replace region in Tensor IR
//
// The main compiler remains deterministic; the e-graph is a specialized
// superoptimizer that runs on eligible regions.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class EGraphSuperoptimizerPass : public Pass {
public:
    std::string name() const override { return "egraph_superopt"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
