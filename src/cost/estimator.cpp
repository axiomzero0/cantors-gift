// cost/estimator.cpp - analytical cost model implementation
#include "cg/cost/estimator.hpp"
#include "cg/ir/ops.hpp"

#include <algorithm>

namespace cg {

namespace {

// Compute FLOPs for an operation.
u64 flops_for_op(const Operation& op) {
    if (op.opcode == OP_MATMUL) {
        if (op.operands.size() != 2) return 0;
        auto a = op.operands[0].as_tensor();
        auto b = op.operands[1].as_tensor();
        if (!a || !b) return 0;
        // [..., M, K] x [..., K, N] -> [..., M, N], FLOPs = 2 * batch * M * K * N
        u64 batch = 1;
        for (usize i = 0; i + 2 < a->shape.rank(); ++i) {
            if (!a->shape[i]->is_constant()) return 0;
            batch *= static_cast<u64>(a->shape[i]->value);
        }
        if (!a->shape[a->shape.rank() - 2]->is_constant() ||
            !a->shape[a->shape.rank() - 1]->is_constant() ||
            !b->shape[b->shape.rank() - 1]->is_constant()) return 0;
        u64 M = a->shape[a->shape.rank() - 2]->value;
        u64 K = a->shape[a->shape.rank() - 1]->value;
        u64 N = b->shape[b->shape.rank() - 1]->value;
        return 2 * batch * M * K * N;
    }
    if (op.opcode == OP_REDUCE_SUM || op.opcode == OP_REDUCE_MAX ||
        op.opcode == OP_REDUCE_MEAN) {
        // Reduction: 1 FLOP per reduced element (mean adds a divide).
        if (op.operands.size() != 1) return 0;
        auto a = op.operands[0].as_tensor();
        if (!a) return 0;
        auto axes_attr = op.attributes.get("axes");
        if (!axes_attr) return 0;
        u64 total = 1;
        for (auto& d : a->shape) {
            if (!d->is_constant()) return 0;
            total *= static_cast<u64>(d->value);
        }
        return total;
    }
    // Elementwise ops: 1 FLOP per element.
    if (op.has_trait(OpTrait::Elementwise)) {
        if (!op.results.empty() && op.results[0].as_tensor()) {
            auto t = op.results[0].as_tensor();
            u64 n = 1;
            for (auto& d : t->shape) {
                if (!d->is_constant()) return 0;
                n *= static_cast<u64>(d->value);
            }
            return n;
        }
    }
    return 0;
}

u64 bytes_for_op(const Operation& op, u64& shared_bytes) {
    shared_bytes = 0;
    u64 total = 0;
    auto add_operand = [&](const Value& v) -> void {
        if (auto t = v.as_tensor()) {
            u64 n = 1;
            for (auto& d : t->shape) {
                if (!d->is_constant()) return;
                n *= static_cast<u64>(d->value);
            }
            total += n * dtype_size(t->dtype);
        }
    };
    for (auto& v : op.operands) add_operand(v);
    for (auto& r : op.results) {
        if (auto t = r.as_tensor()) {
            u64 n = 1;
            for (auto& d : t->shape) {
                if (!d->is_constant()) return total;
                n *= static_cast<u64>(d->value);
            }
            total += n * dtype_size(t->dtype);
        }
    }
    return total;
}

} // namespace

CostEstimate CostEstimator::estimate(const Module& m, const Schedule&) const {
    CostEstimate out;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            out.flops += flops_for_op(op);
            u64 shared = 0;
            u64 bytes = bytes_for_op(op, shared);
            out.bytes_global += bytes - shared;
            out.bytes_shared += shared;
            if (!op.is_pure()) out.kernel_launches++;
        }
    }
    // Rough runtime estimate: max(compute, memory) on the critical path.
    double compute_sec = out.flops / hw_.peak_flops(DType::F32);
    double mem_sec = (out.bytes_global + out.bytes_shared) /
                     std::max(1.0, hw_.memory.get(MemorySpace::Generic));
    out.estimated_runtime_sec = std::max(compute_sec, mem_sec);
    return out;
}

CostEstimate CostEstimator::estimate_matmul(i64 M, i64 K, i64 N, DType dt,
                                            const Schedule&) const {
    CostEstimate out;
    out.flops = 2u * static_cast<u64>(M) * static_cast<u64>(K) * static_cast<u64>(N);
    out.bytes_global = (static_cast<u64>(M) * static_cast<u64>(K) +
                        static_cast<u64>(K) * static_cast<u64>(N) +
                        static_cast<u64>(M) * static_cast<u64>(N)) * dtype_size(dt);
    out.kernel_launches = 1;
    out.parallel_axes = 2;
    bool use_tc = (dt == DType::F16 || dt == DType::BF16 || dt == DType::I8);
    double compute_sec = out.flops / hw_.peak_flops(dt, use_tc);
    double mem_sec = out.bytes_global / std::max(1.0, hw_.memory.get(MemorySpace::Generic));
    out.estimated_runtime_sec = std::max(compute_sec, mem_sec);
    return out;
}

std::vector<std::pair<Schedule, CostEstimate>>
CostEstimator::rank(const ScheduleSpace& space, usize k) const {
    std::vector<std::pair<Schedule, CostEstimate>> all;
    all.reserve(space.size());
    for (auto& s : space.schedules()) {
        // We assume the schedule applies to a matmul; in real use, the
        // autotuner passes the relevant op's shape & dtype.
        all.emplace_back(s, estimate_matmul(1024, 1024, 1024, DType::F32, s));
    }
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) {
                  return a.second.estimated_runtime_sec < b.second.estimated_runtime_sec;
              });
    if (all.size() > k) all.resize(k);
    return all;
}

} // namespace cg
