// optimization/graph_pattern.cpp - graph-level pattern matching
#include "cg/optimization/graph/graph_pattern.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

namespace {

// Check if an op is a specific opcode.
bool is_op(const Operation& op, Opcode code) {
    return op.opcode == code;
}

// Find the single consumer of a value (or nullptr if multiple/none).
Operation* single_consumer(Value v, Block& block) {
    Operation* consumer = nullptr;
    for (auto& c : block) {
        for (auto& operand : c.operands) {
            if (operand == v) {
                if (consumer) return nullptr; // multiple consumers
                consumer = &c;
            }
        }
    }
    return consumer;
}

} // namespace

PreservedAnalyses GraphPatternMatchingPass::run(Module& m, AnalysisManager&) {
    bool changed = false;

    for (auto& f : m.functions()) {
        Block& block = *f->entry();

        // Pattern 1: matmul + add(bias) + relu -> mark as fused chain
        // This doesn't create a new op (fusion handles that) but annotates
        // the pattern for the fusion pass to find.
        for (auto& op : block) {
            if (op.opcode != OP_MATMUL) continue;
            if (op.results.empty()) continue;
            Value mm_result = op.results[0];

            // Check if the single consumer is an add.
            Operation* add_op = single_consumer(mm_result, block);
            if (!add_op || !is_op(*add_op, OP_ADD)) continue;
            if (add_op->results.empty()) continue;
            Value add_result = add_op->results[0];

            // Check if the single consumer of add is relu.
            Operation* relu_op = single_consumer(add_result, block);
            if (!relu_op || !is_op(*relu_op, OP_RELU)) continue;

            // Annotate the relu op as a "matmul_bias_relu" pattern.
            relu_op->attributes.set("pattern", Attribute::make_string("matmul_bias_relu"));
            changed = true;
        }

        // Pattern 2: scale -> softmax -> matmul (attention pattern)
        // Q @ K^T -> scale -> softmax -> @ V
        // We look for: matmul -> mul(constant) -> softmax -> matmul
        for (auto& op : block) {
            if (op.opcode != OP_MATMUL) continue;
            if (op.results.empty()) continue;
            Value qk_result = op.results[0];

            Operation* scale_op = single_consumer(qk_result, block);
            if (!scale_op || !is_op(*scale_op, OP_MUL)) continue;
            if (scale_op->results.empty()) continue;
            Value scale_result = scale_op->results[0];

            Operation* softmax_op = single_consumer(scale_result, block);
            if (!softmax_op || !is_op(*softmax_op, OP_SOFTMAX)) continue;
            if (softmax_op->results.empty()) continue;
            Value sm_result = softmax_op->results[0];

            Operation* av_op = single_consumer(sm_result, block);
            if (!av_op || !is_op(*av_op, OP_MATMUL)) continue;

            // Annotate the second matmul as an "attention" pattern.
            av_op->attributes.set("pattern", Attribute::make_string("attention"));
            changed = true;
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
