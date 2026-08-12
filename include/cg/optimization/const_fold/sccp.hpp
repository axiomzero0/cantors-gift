// optimization/sccp.hpp - sparse conditional constant propagation
//
// Propagates constants through the IR and removes dead branches whose
// predicate is provably constant. Uses the shape constraint solver to
// evaluate branch conditions involving symbolic expressions.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class SCCPPass : public Pass {
public:
    std::string name() const override { return "sccp"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
