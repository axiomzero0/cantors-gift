// analysis/parallelism_analysis.cpp
//
// Computes per-op parallelism metadata: how many independent work items
// each op exposes, whether it has a reduction, and the reduction length.
// Used by the cost model and the schedule optimizer to decide tiling
// and parallelization strategy.
//
// Shape dimension access uses `dim_value_or_zero()` rather than
// `static_cast<u64>(d->value)` because the latter would silently wrap
// negative `i64` values into huge unsigned values. A negative shape
// dimension is invalid IR but we still must not produce garbage
// numbers from it.
#include "cg/analysis/parallelism_analysis.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

void ParallelismAnalysis::compute() {
    Module& m = am_.module();
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            ParallelismInfo pi;

            auto numel = [](const Value& v) -> u64 {
                if (auto t = v.as_tensor()) {
                    u64 n = 1;
                    for (auto& d : t->shape) {
                        if (!d->is_constant()) return 0;
                        u64 dv = dim_value_or_zero(d);
                        if (dv == 0) return 0;
                        n *= dv;
                    }
                    return n;
                }
                return 0;
            };

            if (op.opcode == OP_MATMUL && op.operands.size() == 2) {
                auto a = op.operands[0].as_tensor();
                auto b = op.operands[1].as_tensor();
                if (a && b && a->shape.rank() >= 2 && b->shape.rank() >= 2) {
                    u64 M = dim_value_or_zero(a->shape[a->shape.rank() - 2]);
                    u64 K = dim_value_or_zero(a->shape[a->shape.rank() - 1]);
                    u64 N = dim_value_or_zero(b->shape[b->shape.rank() - 1]);
                    u64 batch = 1;
                    for (usize i = 0; i + 2 < a->shape.rank(); ++i) {
                        u64 bv = dim_value_or_zero(a->shape[i]);
                        if (bv == 0) { batch = 0; break; }
                        batch *= bv;
                    }
                    pi.independent_items = batch * M * N;
                    pi.has_reduction = true;
                    pi.reduction_length = K;
                }
            } else if (op.has_trait(OpTrait::Reduction)) {
                if (!op.operands.empty()) {
                    auto in = op.operands[0].as_tensor();
                    if (in) {
                        u64 in_n = numel(op.operands[0]);
                        u64 out_n = !op.results.empty() ? numel(op.results[0]) : 1;
                        pi.independent_items = out_n;
                        pi.has_reduction = true;
                        pi.reduction_length = in_n / std::max<u64>(out_n, 1);
                    }
                }
            } else if (op.has_trait(OpTrait::Elementwise)) {
                if (!op.results.empty()) {
                    pi.independent_items = numel(op.results[0]);
                }
            }

            pi.parallelism_ratio = pi.reduction_length > 0
                ? static_cast<double>(pi.independent_items) /
                  static_cast<double>(pi.reduction_length)
                : static_cast<double>(pi.independent_items);

            per_op_[op.id] = pi;
        }
    }
}

} // namespace cg
