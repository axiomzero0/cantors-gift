// optimization/recomputation.cpp - recomputation pass
#include "cg/optimization/recomputation/recomputation.hpp"
#include "cg/analysis/reuse_analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

PreservedAnalyses RecomputationPass::run(Module& m, AnalysisManager& am) {
    bool changed = false;
    auto& reuse = am.get<ReuseAnalysis>();

    // For each value marked "Recompute" with multiple consumers, duplicate
    // the defining op at each consumer site. This eliminates the need to
    // store the intermediate tensor.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            Value v = op.results[0];
            auto info = reuse.info_of(v);
            if (info.decision != ReuseDecision::Recompute) continue;
            if (info.num_consumers <= 1) continue;

            // Find all consumers.
            std::vector<Operation*> consumers;
            for (auto& c : *f->entry()) {
                for (auto& operand : c.operands) {
                    if (operand == v) {
                        consumers.push_back(&c);
                        break;
                    }
                }
            }

            if (consumers.size() <= 1) continue;

            // For each consumer (except the first), create a duplicate of
            // the defining op and replace the operand.
            Builder b(f.get());
            for (usize i = 1; i < consumers.size(); ++i) {
                Operation* consumer = consumers[i];
                // Duplicate the defining op.
                auto* dup = b.create(op.opcode, op.operands, op.attributes);
                // Replace the operand in the consumer.
                for (auto& operand : consumer->operands) {
                    if (operand == v) {
                        operand = dup->results[0];
                        changed = true;
                        break;
                    }
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
