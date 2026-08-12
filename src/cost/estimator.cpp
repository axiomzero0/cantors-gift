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

CostEstimate CostEstimator::estimate(const Module& m, const Schedule& schedule) const {
    CostEstimate out;

    // Extract schedule parameters that affect cost.
    u64 m_tile = 64, n_tile = 64, k_tile = 32;
    u64 vector_width = 8;
    bool uses_shared = false;
    bool uses_tc = false;
    for (const auto& t : schedule.transforms()) {
        switch (t.kind) {
            case TransformKind::Tile:
                if (t.dim == "m") m_tile = static_cast<u64>(t.factor);
                else if (t.dim == "n") n_tile = static_cast<u64>(t.factor);
                else if (t.dim == "k") k_tile = static_cast<u64>(t.factor);
                break;
            case TransformKind::Vectorize:
                vector_width = static_cast<u64>(t.factor);
                break;
            case TransformKind::Cache:
                if (t.mem == MemorySpace::Shared) uses_shared = true;
                break;
            case TransformKind::Bind:
                if (t.target == "tensor_core") uses_tc = true;
                break;
            default: break;
        }
    }

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            out.flops += flops_for_op(op);
            u64 shared = 0;
            u64 bytes = bytes_for_op(op, shared);
            // If the schedule uses shared memory, count some bytes as shared.
            // The bytes_for_op function returns shared=0 because it doesn't
            // know about the schedule; we estimate shared bytes here based
            // on the schedule's tile sizes.
            if (uses_shared && op.opcode == OP_MATMUL) {
                // For matmul with shared memory, the A and B tiles are
                // loaded into shared memory.
                u64 tile_bytes = (m_tile * k_tile + k_tile * n_tile) * 4; // F32
                shared = std::min(bytes, tile_bytes);
            }
            if (uses_shared) {
                out.bytes_shared += shared;
                out.bytes_global += (bytes - shared);
            } else {
                out.bytes_global += bytes;
            }
            if (!op.is_pure()) out.kernel_launches++;
        }
    }

    // Adjust for vectorization: wider vectors mean fewer load/store ops.
    if (vector_width > 1) {
        out.bytes_global /= std::max(u64(1), vector_width / 4);
    }

    // Tensor cores increase effective compute throughput.
    double compute_throughput = hw_.peak_flops(DType::F32, uses_tc);
    double compute_sec = out.flops / std::max(1.0, compute_throughput);
    double mem_sec = (out.bytes_global + out.bytes_shared) /
                     std::max(1.0, hw_.memory.get(MemorySpace::Generic));
    out.estimated_runtime_sec = std::max(compute_sec, mem_sec);
    return out;
}

CostEstimate CostEstimator::estimate_matmul(i64 M, i64 K, i64 N, DType dt,
                                            const Schedule& schedule) const {
    CostEstimate out;
    out.flops = 2u * static_cast<u64>(M) * static_cast<u64>(K) * static_cast<u64>(N);
    out.bytes_global = (static_cast<u64>(M) * static_cast<u64>(K) +
                        static_cast<u64>(K) * static_cast<u64>(N) +
                        static_cast<u64>(M) * static_cast<u64>(N)) * dtype_size(dt);
    out.kernel_launches = 1;
    out.parallel_axes = 2;

    // Extract schedule parameters.
    u64 m_tile = 64, n_tile = 64, k_tile = 32;
    u64 vector_width = 1;
    bool uses_shared = false;
    bool uses_tc = false;
    for (const auto& t : schedule.transforms()) {
        switch (t.kind) {
            case TransformKind::Tile:
                if (t.dim == "m") m_tile = static_cast<u64>(t.factor);
                else if (t.dim == "n") n_tile = static_cast<u64>(t.factor);
                else if (t.dim == "k") k_tile = static_cast<u64>(t.factor);
                break;
            case TransformKind::Vectorize:
                vector_width = static_cast<u64>(t.factor);
                break;
            case TransformKind::Cache:
                if (t.mem == MemorySpace::Shared) uses_shared = true;
                break;
            case TransformKind::Bind:
                if (t.target == "tensor_core") uses_tc = true;
                break;
            default: break;
        }
    }

    // Shared memory reduces global memory traffic (tiles are loaded once
    // and reused across the reduction dimension).
    if (uses_shared) {
        u64 tile_reuse = std::max<u64>(1, K / k_tile);
        u64 shared_bytes = (m_tile * k_tile + k_tile * n_tile) * dtype_size(dt);
        out.bytes_shared = shared_bytes;
        out.bytes_global = out.bytes_global / tile_reuse;
    }

    // Vectorization reduces load/store count.
    if (vector_width > 1) {
        out.bytes_global /= std::max(u64(1), vector_width);
    }

    double compute_sec = out.flops / hw_.peak_flops(dt, uses_tc);
    double mem_sec = out.bytes_global / std::max(1.0, hw_.memory.get(MemorySpace::Generic));
    out.estimated_runtime_sec = std::max(compute_sec, mem_sec);
    return out;
}

std::vector<std::pair<Schedule, CostEstimate>>
CostEstimator::rank(const ScheduleSpace& space, usize k) const {
    std::vector<std::pair<Schedule, CostEstimate>> all;
    all.reserve(space.size());
    for (auto& s : space.schedules()) {
        // Use the schedule's own tile sizes to estimate cost.
        // Extract M, N, K from the schedule if available; default to 1024.
        i64 M = 1024, N = 1024, K = 1024;
        for (const auto& t : s.transforms()) {
            if (t.kind == TransformKind::Tile) {
                if (t.dim == "m") M = t.factor;
                else if (t.dim == "n") N = t.factor;
                else if (t.dim == "k") K = t.factor;
            }
        }
        all.emplace_back(s, estimate_matmul(M, K, N, DType::F32, s));
    }
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) {
                  return a.second.estimated_runtime_sec < b.second.estimated_runtime_sec;
              });
    if (all.size() > k) all.resize(k);
    return all;
}

} // namespace cg
