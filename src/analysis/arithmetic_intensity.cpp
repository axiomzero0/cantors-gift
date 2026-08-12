// analysis/arithmetic_intensity.cpp
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

BoundClass ArithmeticIntensityAnalysis::classify(u64 flops, u64 bytes,
                                                  usize parallelism) {
    if (flops == 0 && bytes == 0) return BoundClass::LaunchBound;
    if (flops == 0) return BoundClass::MemoryBound;
    if (bytes == 0) return BoundClass::ComputeBound;
    double intensity = static_cast<double>(flops) / static_cast<double>(bytes);
    // Thresholds are deliberately conservative; the cost model will refine.
    if (parallelism <= 1 && flops < 1024) return BoundClass::LatencyBound;
    if (intensity < 1.0)   return BoundClass::MemoryBound;
    if (intensity > 16.0)  return BoundClass::ComputeBound;
    if (flops < 4096 && bytes < 4096) return BoundClass::LaunchBound;
    return BoundClass::Balanced;
}

void ArithmeticIntensityAnalysis::compute() {
    Module& m = am_.module();
    total_flops_ = 0;
    total_bytes_ = 0;

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            OpIntensity oi;

            // FLOPs
            if (op.opcode == OP_MATMUL && op.operands.size() == 2) {
                auto a = op.operands[0].as_tensor();
                auto b = op.operands[1].as_tensor();
                if (a && b && a->shape.rank() >= 2 && b->shape.rank() >= 2) {
                    u64 M = a->shape[a->shape.rank() - 2]->is_constant()
                        ? a->shape[a->shape.rank() - 2]->value : 0;
                    u64 K = a->shape[a->shape.rank() - 1]->is_constant()
                        ? a->shape[a->shape.rank() - 1]->value : 0;
                    u64 N = b->shape[b->shape.rank() - 1]->is_constant()
                        ? b->shape[b->shape.rank() - 1]->value : 0;
                    u64 batch = 1;
                    for (usize i = 0; i + 2 < a->shape.rank(); ++i) {
                        batch *= a->shape[i]->is_constant()
                            ? a->shape[i]->value : 0;
                    }
                    oi.flops = 2 * batch * M * K * N;
                }
            } else if (op.has_trait(OpTrait::Reduction)) {
                if (!op.operands.empty()) {
                    auto a = op.operands[0].as_tensor();
                    if (a) {
                        u64 n = 1;
                        for (auto& d : a->shape)
                            n *= d->is_constant() ? d->value : 0;
                        oi.flops = n;
                    }
                }
            } else if (op.has_trait(OpTrait::Elementwise) && !op.results.empty()) {
                auto t = op.results[0].as_tensor();
                if (t) {
                    u64 n = 1;
                    for (auto& d : t->shape)
                        n *= d->is_constant() ? d->value : 0;
                    oi.flops = n;
                }
            }

            // Bytes
            auto count_bytes = [](const Value& v) -> u64 {
                if (auto t = v.as_tensor()) {
                    u64 n = 1;
                    for (auto& d : t->shape)
                        if (!d->is_constant()) return 0;
                        else n *= static_cast<u64>(d->value);
                    return n * dtype_size(t->dtype);
                }
                return 0;
            };
            for (auto& v : op.operands) oi.bytes_read += count_bytes(v);
            for (auto& r : op.results) oi.bytes_written += count_bytes(r);

            oi.intensity = oi.total_bytes() > 0
                ? static_cast<double>(oi.flops) / static_cast<double>(oi.total_bytes())
                : (oi.flops > 0 ? 1e18 : 0.0);

            oi.bound = classify(oi.flops, oi.total_bytes(),
                                op.has_trait(OpTrait::Reduction) ? 1 : 16);

            total_flops_ += oi.flops;
            total_bytes_ += oi.total_bytes();
            per_op_[op.id] = oi;
        }
    }

    module_intensity_ = total_bytes_ > 0
        ? static_cast<double>(total_flops_) / static_cast<double>(total_bytes_)
        : 0.0;
    module_bound_ = classify(total_flops_, total_bytes_, 32);
}

} // namespace cg
