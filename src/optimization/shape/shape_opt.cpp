// optimization/shape_opt.cpp - shape optimization
#include "cg/optimization/shape/shape_opt.hpp"
#include "cg/ir/ops.hpp"
#include "cg/shape/simplifier.hpp"
#include "cg/shape/solver.hpp"

namespace cg {

PreservedAnalyses ShapeOptimizationPass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    Solver solver(m.constraints());

    // Walk every tensor type in every op result and simplify its shape.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            for (auto& r : op.results) {
                auto t = const_cast<TensorType*>(
                    dynamic_cast<const TensorType*>(r.type().get()));
                if (!t) continue;
                bool shape_changed = false;
                for (auto& d : t->shape.dims()) {
                    auto simplified = simplify_dim(d);
                    if (!simplified->structurally_equal(*d)) {
                        d = simplified;
                        shape_changed = true;
                    }
                }
                if (shape_changed) changed = true;
            }
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
