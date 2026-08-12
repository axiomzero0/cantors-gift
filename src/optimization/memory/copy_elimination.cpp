// optimization/copy_elimination.cpp - copy elimination
#include "cg/optimization/memory/copy_elimination.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

PreservedAnalyses CopyEliminationPass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    for (auto& f : m.functions()) {
        std::vector<Operation*> to_remove;
        for (auto& op : *f->entry()) {
            if (op.opcode != OP_COPY) continue;
            if (op.operands.size() != 1 || op.results.empty()) continue;
            // Replace all uses of the copy result with the original operand.
            m.replace_all_uses(op.results[0], op.operands[0]);
            to_remove.push_back(&op);
            changed = true;
        }
        for (Operation* op : to_remove) f->entry()->remove(op);
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
