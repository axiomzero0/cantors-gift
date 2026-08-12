// optimization/copy_elimination.hpp - copy elimination
//
// Removes OP_COPY ops whose output is provably equivalent to the input
// (same layout, same shape, no aliasing concern). Also removes copies whose
// only consumer is itself a copy.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class CopyEliminationPass : public Pass {
public:
    std::string name() const override { return "copy_elim"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
