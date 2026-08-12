// optimization/dce.cpp - dead code elimination
#include "cg/optimization/dce/dce.hpp"
#include "cg/ir/ops.hpp"

#include <unordered_set>

namespace cg {

PreservedAnalyses DCEPass::run(Module& m, AnalysisManager&) {
    bool changed = false;

    for (auto& f : m.functions()) {
        // Collect used values.
        std::unordered_set<ValueId> used;
        for (auto& op : *f->entry()) {
            for (auto& v : op.operands) used.insert(v.id());
        }
        for (auto& arg : f->entry()->arguments()) used.insert(arg.id());

        // Walk in reverse, removing pure ops whose results are unused.
        std::vector<Operation*> to_remove;
        for (auto it = f->entry()->head(); it; it = it->next) {
            if (!it->is_pure()) continue;
            bool any_used = false;
            for (auto& r : it->results) {
                if (used.count(r.id())) { any_used = true; break; }
            }
            if (!any_used) to_remove.push_back(it);
        }
        for (Operation* op : to_remove) {
            f->entry()->remove(op);
            changed = true;
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
