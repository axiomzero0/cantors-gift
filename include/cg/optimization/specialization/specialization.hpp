// optimization/specialization.hpp - shape/dtype/alignment specialization
//
// Specializes ops based on provable constraints:
//   - N % 64 == 0  -> use the vectorized kernel without tail handling
//   - 128-byte aligned -> use aligned loads
//   - static shape -> specialize loop bounds
//
// Specialized ops carry a `specialized` attribute with the predicate that
// must hold at runtime. The runtime dispatches to the specialized kernel if
// the predicate holds, else falls back to the generic kernel.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class SpecializationPass : public Pass {
public:
    std::string name() const override { return "specialization"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
