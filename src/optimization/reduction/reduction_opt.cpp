// optimization/reduction_opt.cpp - reduction optimization
#include "cg/optimization/reduction/reduction_opt.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

PreservedAnalyses ReductionOptimizationPass::run(Module& m, AnalysisManager& am) {
    bool changed = false;
    auto& df = am.get<DataflowAnalysis>();

    // Pattern: reduce_sum(x) -> div(result, N)  becomes  reduce_mean(x)
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.opcode != OP_DIV || op.operands.size() != 2) continue;
            // Find defining op of operand 0.
            Operation* red = nullptr;
            for (auto& c : *f->entry()) {
                if (!c.results.empty() && c.results[0] == op.operands[0]) {
                    red = &c; break;
                }
            }
            if (!red || red->opcode != OP_REDUCE_SUM) continue;

            // Operand 1 must be a constant.
            Operation* divisor = nullptr;
            for (auto& c : *f->entry()) {
                if (!c.results.empty() && c.results[0] == op.operands[1]) {
                    divisor = &c; break;
                }
            }
            if (!divisor || divisor->opcode != OP_CONSTANT) continue;

            // Single consumer of the reduce_sum result.
            if (df.fanout(red->results[0]) != 1) continue;

            // Rewrite: change reduce_sum to reduce_mean and remove the div.
            red->opcode = OP_REDUCE_MEAN;
            red->name = "reduce_mean";
            m.replace_all_uses(op.results[0], red->results[0]);
            changed = true;
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
