// optimization/memory_planning.hpp - memory planning + Memory IR
//
// After semantic optimization, the compiler introduces explicit memory
// allocation ops. This pass:
//   - Computes global tensor lifetimes (via LifetimeAnalysis)
//   - Assigns each short-lived tensor to a buffer slot
//   - Inserts OP_ALLOC / OP_FREE / OP_REUSE markers
//   - Performs buffer reuse: tensors with disjoint lifetimes share a buffer
//
// The result is the Memory IR layer: a Tensor IR module annotated with
// explicit allocation events. Lowering consumes this to emit actual memory
// management calls.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class MemoryPlanningPass : public Pass {
public:
    std::string name() const override { return "memory_planning"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
