// analysis/arithmetic_intensity.cpp
//
// Rewritten to compute FLOPs/bytes for ALL ops, not just matmul/reduction/
// elementwise. Every op now has a real FLOP count and byte count based on
// its semantics.
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

BoundClass ArithmeticIntensityAnalysis::classify(u64 flops, u64 bytes,
                                                  usize parallelism) {
    if (flops == 0 && bytes == 0) return BoundClass::LaunchBound;
    if (flops == 0) return BoundClass::MemoryBound;
    if (bytes == 0) return BoundClass::ComputeBound;
    double intensity = static_cast<double>(flops) / static_cast<double>(bytes);
    if (parallelism <= 1 && flops < 1024) return BoundClass::LatencyBound;
    if (intensity < 1.0)   return BoundClass::MemoryBound;
    if (intensity > 16.0)  return BoundClass::ComputeBound;
    if (flops < 4096 && bytes < 4096) return BoundClass::LaunchBound;
    return BoundClass::Balanced;
}

namespace {

// Count elements in a tensor type.
u64 numel(const Value& v) {
    if (auto t = v.as_tensor()) {
        u64 n = 1;
        for (auto& d : t->shape) {
            if (!d->is_constant()) return 0;
            n *= static_cast<u64>(d->value);
        }
        return n;
    }
    return 0;
}

u64 elem_bytes(const Value& v) {
    if (auto t = v.as_tensor()) return dtype_size(t->dtype);
    return 4;
}

// Compute FLOPs for an operation based on its opcode and operands.
u64 compute_flops(const Operation& op) {
    switch (op.opcode) {
        // ---- Matmul: 2*M*K*N FLOPs ----
        case OP_MATMUL: {
            if (op.operands.size() != 2) return 0;
            auto a = op.operands[0].as_tensor();
            auto b = op.operands[1].as_tensor();
            if (!a || !b || a->shape.rank() < 2 || b->shape.rank() < 2) return 0;
            u64 M = a->shape[a->shape.rank() - 2]->is_constant()
                ? a->shape[a->shape.rank() - 2]->value : 0;
            u64 K = a->shape[a->shape.rank() - 1]->is_constant()
                ? a->shape[a->shape.rank() - 1]->value : 0;
            u64 N = b->shape[b->shape.rank() - 1]->is_constant()
                ? b->shape[b->shape.rank() - 1]->value : 0;
            u64 batch = 1;
            for (usize i = 0; i + 2 < a->shape.rank(); ++i) {
                batch *= a->shape[i]->is_constant() ? a->shape[i]->value : 0;
            }
            return 2 * batch * M * K * N;
        }

        // ---- Conv2D: 2 * N * C_out * H_out * W_out * kH * kW ----
        case OP_CONV2D: {
            if (op.operands.size() != 2) return 0;
            auto input = op.operands[0].as_tensor();
            auto weight = op.operands[1].as_tensor();
            if (!input || !weight || input->shape.rank() != 4 ||
                weight->shape.rank() != 4) return 0;
            u64 N = input->shape[0]->is_constant() ? input->shape[0]->value : 0;
            u64 C_out = weight->shape[0]->is_constant() ? weight->shape[0]->value : 0;
            u64 H_out = 0, W_out = 0;
            if (op.results.size() > 0) {
                auto out = op.results[0].as_tensor();
                if (out && out->shape.rank() == 4) {
                    H_out = out->shape[2]->is_constant() ? out->shape[2]->value : 0;
                    W_out = out->shape[3]->is_constant() ? out->shape[3]->value : 0;
                }
            }
            u64 kH = weight->shape[2]->is_constant() ? weight->shape[2]->value : 0;
            u64 kW = weight->shape[3]->is_constant() ? weight->shape[3]->value : 0;
            // FLOPs = 2 * N * C_out * H_out * W_out * C_in * kH * kW
            // But C_in * kH * kW is the K dimension
            u64 C_in = weight->shape[1]->is_constant() ? weight->shape[1]->value : 0;
            return 2 * N * C_out * H_out * W_out * C_in * kH * kW;
        }

        // ---- Reductions: 1 FLOP per reduced element ----
        case OP_REDUCE_SUM:
        case OP_REDUCE_MAX:
        case OP_REDUCE_MEAN: {
            if (op.operands.empty()) return 0;
            u64 in_n = numel(op.operands[0]);
            u64 out_n = op.results.empty() ? 1 : numel(op.results[0]);
            // reduce_mean has an extra division
            if (op.opcode == OP_REDUCE_MEAN) return in_n + out_n;
            return in_n;
        }

        // ---- Softmax: 3 passes (max, exp+sum, divide) = ~3*N FLOPs ----
        case OP_SOFTMAX: {
            if (op.operands.empty()) return 0;
            u64 n = numel(op.operands[0]);
            // Pass 1: max (N comparisons = N FLOPs)
            // Pass 2: exp + sum (2N FLOPs)
            // Pass 3: divide (N FLOPs)
            return n * 4;
        }

        // ---- LayerNorm: mean + var + normalize + scale+shift = ~5*N FLOPs ----
        case OP_LAYERNORM: {
            if (op.operands.empty()) return 0;
            u64 n = numel(op.operands[0]);
            // Pass 1: sum (N FLOPs)
            // Pass 2: sum of squares (N FLOPs)
            // Pass 3: normalize (3N FLOPs: subtract mean, divide by sqrt(var), multiply gamma)
            return n * 5;
        }

        // ---- BatchNorm: similar to LayerNorm ----
        case OP_BATCHNORM: {
            if (op.operands.empty()) return 0;
            u64 n = numel(op.operands[0]);
            return n * 4;
        }

        // ---- Elementwise: 1 FLOP per output element ----
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_NEG: case OP_RELU: case OP_GELU:
        case OP_SIGMOID: case OP_TANH: case OP_EXP:
        case OP_LOG: case OP_SQRT: case OP_CAST: {
            if (op.results.empty()) return 0;
            // GELU is more expensive: ~8 FLOPs (tanh approximation)
            if (op.opcode == OP_GELU) return numel(op.results[0]) * 8;
            // EXP, LOG, SQRT are transcendental: ~4 FLOPs each
            if (op.opcode == OP_EXP || op.opcode == OP_LOG ||
                op.opcode == OP_SQRT) return numel(op.results[0]) * 4;
            // SIGMOID, TANH: ~3 FLOPs
            if (op.opcode == OP_SIGMOID || op.opcode == OP_TANH)
                return numel(op.results[0]) * 3;
            return numel(op.results[0]);
        }

        // ---- Gather: 1 address computation per output element ----
        case OP_GATHER: {
            if (op.results.empty()) return 0;
            return numel(op.results[0]); // ~1 FLOP per element (address calc)
        }

        // ---- Scatter: 1 address computation + 1 write per element ----
        case OP_SCATTER: {
            if (op.operands.size() >= 3) return numel(op.operands[2]);
            return 0;
        }

        // ---- Concat: ~0 FLOPs (memory copy) ----
        case OP_CONCAT:
            return 0;

        // ---- Slice: ~0 FLOPs (memory copy) ----
        case OP_SLICE:
            return 0;

        // ---- Copy: 0 FLOPs ----
        case OP_COPY:
            return 0;

        // ---- Layout ops: 0 FLOPs ----
        case OP_BROADCAST:
        case OP_RESHAPE:
        case OP_TRANSPOSE:
            return 0;

        // ---- Constants, inputs, outputs: 0 FLOPs ----
        case OP_CONSTANT:
        case OP_INPUT:
        case OP_OUTPUT:
        case OP_RETURN:
        case OP_ALLOC:
        case OP_FREE:
            return 0;

        default:
            return 0;
    }
}

} // namespace

void ArithmeticIntensityAnalysis::compute() {
    Module& m = am_.module();
    total_flops_ = 0;
    total_bytes_ = 0;

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            OpIntensity oi;

            // FLOPs: now computed for ALL ops, not just matmul/reduction/elementwise.
            oi.flops = compute_flops(op);

            // Bytes: read from operands, written to results.
            auto count_bytes = [](const Value& v) -> u64 {
                if (auto t = v.as_tensor()) {
                    u64 n = 1;
                    for (auto& d : t->shape) {
                        if (!d->is_constant()) return 0;
                        n *= static_cast<u64>(d->value);
                    }
                    return n * dtype_size(t->dtype);
                }
                return 0;
            };
            for (auto& v : op.operands) oi.bytes_read += count_bytes(v);
            for (auto& r : op.results) oi.bytes_written += count_bytes(r);

            // Intensity = FLOPs / total bytes
            oi.intensity = oi.total_bytes() > 0
                ? static_cast<double>(oi.flops) / static_cast<double>(oi.total_bytes())
                : (oi.flops > 0 ? 1e18 : 0.0);

            // Classify using the real roofline.
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
