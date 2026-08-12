// optimization/specialization.cpp - shape/dtype/alignment specialization
#include "cg/optimization/specialization/specialization.hpp"
#include "cg/ir/ops.hpp"
#include "cg/shape/solver.hpp"

namespace cg {

PreservedAnalyses SpecializationPass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    Solver solver(m.constraints());

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            // For every operand that is a tensor, check if its leading
            // dimension is provably divisible by 16. If so, mark the op as
            // "specialized:aligned_vector".
            for (auto& v : op.operands) {
                auto t = v.as_tensor();
                if (!t || t->shape.empty()) continue;
                auto d0 = t->shape[0];
                if (!d0->is_symbol()) continue;
                if (solver.prove_divisible(d0, 16) == SolverResult::ProvedTrue) {
                    op.attributes.set("specialized", Attribute::make_string("aligned_vector_16"));
                    changed = true;
                    break;
                }
            }
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
