// optimization/dce.hpp - dead code elimination
//
// Removes operations whose results have no uses and that have no side effects.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class DCEPass : public Pass {
public:
    std::string name() const override { return "dce"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
