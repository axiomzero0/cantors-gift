// optimization/layout_opt.cpp - global layout optimization
#include "cg/optimization/layout/layout_opt.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

PreservedAnalyses LayoutOptimizationPass::run(Module& m, AnalysisManager& am) {
    bool changed = false;
    auto& df = am.get<DataflowAnalysis>();

    // Pattern: transpose(A) -> matmul(transpose(A), B)
    // If the transpose is the sole producer for the matmul, we can mark the
    // matmul as wanting a transposed-A layout and drop the transpose.
    for (auto& f : m.functions()) {
        std::vector<Operation*> to_remove;
        for (auto& op : *f->entry()) {
            if (op.opcode != OP_TRANSPOSE) continue;
            if (op.results.empty()) continue;
            Value t = op.results[0];
            const auto& users = df.users(t);
            if (users.size() != 1) continue;

            Operation* consumer = nullptr;
            for (auto& c : *f->entry()) if (c.id == users[0]) { consumer = &c; break; }
            if (!consumer) continue;

            // The consumer must be a matmul that takes our transpose as an
            // operand. If so, mark the matmul as "A is transposed" and
            // replace the operand with the transpose's input.
            if (consumer->opcode == OP_MATMUL) {
                for (auto& v : consumer->operands) {
                    if (v == t) {
                        v = op.operands[0];
                        consumer->attributes.set("transposed_operand",
                            Attribute::make_integer(static_cast<i64>(&v - consumer->operands.data())));
                        m.replace_all_uses(t, op.operands[0]);
                        to_remove.push_back(&op);
                        changed = true;
                        break;
                    }
                }
            }
        }
        for (Operation* op : to_remove) f->entry()->remove(op);
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
