// optimization/fusion.cpp - global fusion with profitability model
#include "cg/optimization/fusion/fusion.hpp"
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/analysis/parallelism_analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

#include <vector>

namespace cg {

namespace {

bool is_elementwise(Opcode op) {
    switch (op) {
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_NEG:
        case OP_RELU: case OP_GELU: case OP_SIGMOID: case OP_TANH:
        case OP_EXP: case OP_LOG: case OP_SQRT: case OP_CAST:
            return true;
        default:
            return false;
    }
}

// Estimate the cost change of fusing producer -> consumer.
// Returns negative if fusion is profitable.
double fusion_delta_cost(const Operation& producer, const Operation& consumer,
                         ArithmeticIntensityAnalysis& ai,
                         ParallelismAnalysis& pa) {
    const auto& pi = ai.intensity_of(producer);
    const auto& ci = ai.intensity_of(consumer);

    // Memory savings: producer's output is no longer written to global memory
    // and re-read by consumer.
    double producer_output_bytes = static_cast<double>(ci.bytes_read);
    double memory_saved_bytes = producer_output_bytes;

    // Launch overhead: one fewer kernel launch.
    double launch_saved_sec = 5e-6;

    // Register pressure: fusing increases live registers.
    // Rough estimate: +8 registers per fused elementwise op.
    double register_penalty_sec = 1e-7; // small penalty

    // Parallelism loss: if consumer has lower parallelism, we lose some.
    const auto& pp = pa.info_of(producer);
    const auto& cp = pa.info_of(consumer);
    double parallelism_penalty = 0.0;
    if (pp.independent_items > cp.independent_items && cp.independent_items > 0) {
        double ratio = static_cast<double>(pp.independent_items) /
                       static_cast<double>(cp.independent_items);
        parallelism_penalty = 1e-6 * ratio;
    }

    // Convert bytes to seconds using a notional 50 GB/s bandwidth.
    double bw = 50e9;
    double memory_saved_sec = memory_saved_bytes / bw;

    return -memory_saved_sec - launch_saved_sec
         + register_penalty_sec + parallelism_penalty;
}

} // namespace

PreservedAnalyses FusionPass::run(Module& m, AnalysisManager& am) {
    bool changed = false;
    auto& df = am.get<DataflowAnalysis>();
    auto& ai = am.get<ArithmeticIntensityAnalysis>();
    auto& pa = am.get<ParallelismAnalysis>();

    for (auto& f : m.functions()) {
        std::vector<Operation*> to_remove;

        for (auto& op : *f->entry()) {
            // Look for elementwise producer -> elementwise consumer patterns.
            if (!is_elementwise(op.opcode)) continue;
            if (op.results.empty()) continue;
            Value prod_val = op.results[0];

            // Single-consumer check (foundational; multi-consumer fusion is
            // more complex and requires duplicating the producer).
            const auto& users = df.users(prod_val);
            if (users.size() != 1) continue;

            Operation* consumer = nullptr;
            for (auto& c : *f->entry()) {
                if (c.id == users[0]) { consumer = &c; break; }
            }
            if (!consumer) continue;
            if (!is_elementwise(consumer->opcode) &&
                consumer->opcode != OP_REDUCE_SUM &&
                consumer->opcode != OP_REDUCE_MAX) continue;

            // Profitability check.
            double delta = fusion_delta_cost(op, *consumer, ai, pa);
            if (delta >= 0.0) continue;

            // Apply fusion: replace consumer's operand with producer's operands
            // and mark consumer as a fused op.
            // For now, we rewrite the consumer to take the producer's operands
            // directly and record the fusion in attributes.
            for (usize i = 0; i < consumer->operands.size(); ++i) {
                if (consumer->operands[i] == prod_val) {
                    // Splice in producer's operands at this position.
                    SmallVector<Value, 4> new_operands;
                    for (usize j = 0; j < i; ++j)
                        new_operands.push_back(consumer->operands[j]);
                    for (auto& v : op.operands)
                        new_operands.push_back(v);
                    for (usize j = i + 1; j < consumer->operands.size(); ++j)
                        new_operands.push_back(consumer->operands[j]);
                    consumer->operands = std::move(new_operands);
                    break;
                }
            }

            // Record the fusion chain.
            auto existing = consumer->attributes.get("fused_chain");
            std::vector<i64> chain;
            if (existing && existing->kind == AttrKind::IntegerArray) {
                chain = existing->ints;
            }
            chain.push_back(static_cast<i64>(op.opcode));
            chain.push_back(static_cast<i64>(consumer->opcode));
            consumer->attributes.set("fused_chain",
                Attribute::make_int_array(std::move(chain)));
            consumer->name = "fused";

            // Replace all uses of producer with consumer's result (since
            // producer is now absorbed).
            m.replace_all_uses(prod_val, consumer->results[0]);
            to_remove.push_back(&op);
            changed = true;
        }

        for (Operation* op : to_remove) {
            f->entry()->remove(op);
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa_result;
    pa_result.preserve<AnalysisManager>();
    // Fusion invalidates dataflow, lifetime, intensity, cost, reuse.
    return pa_result;
}

} // namespace cg
