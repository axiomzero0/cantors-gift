// optimization/algebraic.cpp - algebraic simplification
#include "cg/optimization/canonicalize/algebraic.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

PreservedAnalyses AlgebraicSimplificationPass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            // neg(neg(x)) -> x
            if (op.opcode == OP_NEG && !op.operands.empty()) {
                // Find the defining op of the operand.
                Operation* inner = nullptr;
                for (auto& c : *f->entry()) {
                    if (!c.results.empty() && c.results[0] == op.operands[0]) {
                        inner = &c; break;
                    }
                }
                if (inner && inner->opcode == OP_NEG && !inner->operands.empty()) {
                    m.replace_all_uses(op.results[0], inner->operands[0]);
                    changed = true;
                }
            }

            // x - x -> 0  (requires same operand value)
            if (op.opcode == OP_SUB && op.operands.size() == 2) {
                if (op.operands[0] == op.operands[1]) {
                    // Replace with a zero constant of the right type.
                    auto t = op.results[0].as_tensor();
                    if (t) {
                        std::string zero_bytes(
                            static_cast<usize>(t->shape.num_elements() *
                                               dtype_size(t->dtype)), '\0');
                        AttributeDict attrs;
                        std::vector<i64> shape_vec;
                        for (auto& d : t->shape) {
                            if (d->is_constant()) shape_vec.push_back(d->value);
                            else { shape_vec.clear(); break; }
                        }
                        if (!shape_vec.empty()) {
                            attrs.set("shape", Attribute::make_int_array(shape_vec));
                            attrs.set("dtype", Attribute::make_dtype(t->dtype));
                            attrs.set("bytes", Attribute::make_string(std::move(zero_bytes)));
                            Builder b(f.get());
                            auto* zero = b.create(OP_CONSTANT, {}, attrs);
                            m.replace_all_uses(op.results[0], zero->results[0]);
                            changed = true;
                        }
                    }
                }
            }

            // broadcast(x, same_shape) -> x
            if (op.opcode == OP_BROADCAST && !op.operands.empty()) {
                auto in = op.operands[0].as_tensor();
                auto out = op.results[0].as_tensor();
                if (in && out && in->shape == out->shape) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    changed = true;
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
