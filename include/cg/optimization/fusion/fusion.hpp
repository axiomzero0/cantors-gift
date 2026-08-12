// optimization/fusion.hpp - global fusion with profitability
//
// Fusion combines producer+consumer into a single kernel. The decision is
// NOT just "can we fuse?" but "should we fuse, globally?".
//
// Profitability model:
//   Δcost = Δmemory_traffic + Δlaunch_overhead + Δregister_pressure
//           - Δlocality_gain - Δparallelism_gain
//
// A fusion is applied only if Δcost < 0.
//
// Patterns supported in this foundational implementation:
//   - elementwise + elementwise  (e.g. add + relu)
//   - matmul + elementwise       (e.g. matmul + bias + relu)
//   - reduction + elementwise
//
// Fusion is represented by replacing the producer+consumer pair with a
// single OP_FUSE marker op carrying the fused opcode sequence as an
// attribute. The actual kernel generation happens during lowering.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class FusionPass : public Pass {
public:
    std::string name() const override { return "fusion"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
