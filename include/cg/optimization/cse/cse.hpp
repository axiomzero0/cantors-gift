// optimization/cse.hpp - common subexpression elimination
//
// Hash-cons pure operations within a function and replace duplicate
// computations with a reference to the first.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class CSEPass : public Pass {
public:
    std::string name() const override { return "cse"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
