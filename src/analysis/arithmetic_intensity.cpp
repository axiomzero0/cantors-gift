// analysis/arithmetic_intensity.cpp
//
// Implements three levels of arithmetic intensity:
//   1. Graph:    total FLOPs / total bytes (naive, no fusion, no cache)
//   2. Kernel:   FLOPs / bytes-after-fusion (intermediates dropped)
//   3. Effective: FLOPs / bytes-after-L2-and-shared-reuse
//
// The effective intensity is the number that actually determines whether
// the fused kernel is memory- or compute-bound. A graph can look compute-
// bound at the graph level (high FLOPs/byte) but become memory-bound at
// the effective level if the schedule has poor cache reuse, or vice versa.
//
// Uses dynamic roofline ridge from HardwareModel instead of hardcoded 1.0/16.0.
// The ridge = peak_flops / peak_bw. For H100 F16 TC: ~295 FLOPs/byte.
// For CPU F32: ~5 FLOPs/byte.
//
// All shape-dimension reads go through `dim_value_or_zero()` (defined in
// shape/dim_expr.hpp). A negative shape dimension is invalid IR, and the
// helper returns 0 in that case rather than letting `static_cast<u64>`
// silently wrap it into a huge value.
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/ir/ops.hpp"
#include "cg/schedule/schedule.hpp"
#include "cg/shape/dim_expr.hpp"

#include <algorithm>
#include <cmath>

namespace cg {

BoundClass ArithmeticIntensityAnalysis::classify(u64 flops, u64 bytes,
                                                  usize parallelism) {
    if (flops == 0 && bytes == 0) return BoundClass::LaunchBound;
    if (flops == 0) return BoundClass::MemoryBound;
    if (bytes == 0) return BoundClass::ComputeBound;
    double intensity = static_cast<double>(flops) / static_cast<double>(bytes);

    // Dynamic roofline ridge from hardware model.
    // ridge = peak_flops / peak_memory_bw
    // For A100 F32: 19.5e12 / 2.0e12 = 9.75 FLOPs/byte
    // For H100 F16 TC: 989e12 / 3.35e12 ~ 295 FLOPs/byte
    // For CPU F32: 256e9 / 50e9 = 5.12 FLOPs/byte
    double ridge = hw_.roofline_ridge(DType::F32, false);

    if (parallelism <= 1 && flops < 1024) return BoundClass::LatencyBound;
    // Memory-bound: intensity < ridge/4
    // Compute-bound: intensity > ridge*4
    // Balanced: in between
    if (intensity < ridge / 4.0) return BoundClass::MemoryBound;
    if (intensity > ridge * 4.0) return BoundClass::ComputeBound;
    if (flops < 4096 && bytes < 4096) return BoundClass::LaunchBound;
    return BoundClass::Balanced;
}

namespace {

// Count elements in a tensor type. Returns 0 if any dimension is
// non-constant or zero.
u64 numel(const Value& v) {
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
            u64 M = dim_value_or_zero(a->shape[a->shape.rank() - 2]);
            u64 K = dim_value_or_zero(a->shape[a->shape.rank() - 1]);
            u64 N = dim_value_or_zero(b->shape[b->shape.rank() - 1]);
            u64 batch = 1;
            for (usize i = 0; i + 2 < a->shape.rank(); ++i) {
                u64 bv = dim_value_or_zero(a->shape[i]);
                if (bv == 0) { batch = 0; break; }
                batch *= bv;
            }
            return u64(2) * batch * M * K * N;
        }

        // ---- Conv2D: 2 * N * C_out * H_out * W_out * kH * kW ----
        case OP_CONV2D: {
            if (op.operands.size() != 2) return 0;
            auto input = op.operands[0].as_tensor();
            auto weight = op.operands[1].as_tensor();
            if (!input || !weight || input->shape.rank() != 4 ||
                weight->shape.rank() != 4) return 0;
            u64 N      = dim_value_or_zero(input->shape[0]);
            u64 C_out  = dim_value_or_zero(weight->shape[0]);
            u64 H_out = 0, W_out = 0;
            if (!op.results.empty()) {
                auto out = op.results[0].as_tensor();
                if (out && out->shape.rank() == 4) {
                    H_out = dim_value_or_zero(out->shape[2]);
                    W_out = dim_value_or_zero(out->shape[3]);
                }
            }
            u64 kH  = dim_value_or_zero(weight->shape[2]);
            u64 kW  = dim_value_or_zero(weight->shape[3]);
            u64 C_in = dim_value_or_zero(weight->shape[1]);
            // FLOPs = 2 * N * C_out * H_out * W_out * C_in * kH * kW
            // But C_in * kH * kW is the K dimension
            return u64(2) * N * C_out * H_out * W_out * C_in * kH * kW;
        }

        // ---- Reductions: 1 FLOP per reduced element ----
        case OP_REDUCE_SUM:
        case OP_REDUCE_MAX:
        case OP_REDUCE_MEAN: {
            if (op.operands.empty()) return 0;
            u64 in_n = numel(op.operands[0]);
            u64 out_n = op.results.empty() ? u64(1) : numel(op.results[0]);
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
            return n * u64(4);
        }

        // ---- LayerNorm: mean + var + normalize + scale+shift = ~5*N FLOPs ----
        case OP_LAYERNORM: {
            if (op.operands.empty()) return 0;
            u64 n = numel(op.operands[0]);
            // Pass 1: sum (N FLOPs)
            // Pass 2: sum of squares (N FLOPs)
            // Pass 3: normalize (3N FLOPs: subtract mean, divide by sqrt(var), multiply gamma)
            return n * u64(5);
        }

        // ---- BatchNorm: similar to LayerNorm ----
        case OP_BATCHNORM: {
            if (op.operands.empty()) return 0;
            u64 n = numel(op.operands[0]);
            return n * u64(4);
        }

        // ---- Elementwise: 1 FLOP per output element ----
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_NEG: case OP_RELU: case OP_GELU:
        case OP_SIGMOID: case OP_TANH: case OP_EXP:
        case OP_LOG: case OP_SQRT: case OP_CAST: {
            if (op.results.empty()) return 0;
            // GELU is more expensive: ~8 FLOPs (tanh approximation)
            if (op.opcode == OP_GELU) return numel(op.results[0]) * u64(8);
            // EXP, LOG, SQRT are transcendental: ~4 FLOPs each
            if (op.opcode == OP_EXP || op.opcode == OP_LOG ||
                op.opcode == OP_SQRT) return numel(op.results[0]) * u64(4);
            // SIGMOID, TANH: ~3 FLOPs
            if (op.opcode == OP_SIGMOID || op.opcode == OP_TANH)
                return numel(op.results[0]) * u64(3);
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

// static
double ArithmeticIntensityAnalysis::l2_hit_rate(u64 tensor_bytes,
                                                 const HardwareModel& hw) {
    if (tensor_bytes == 0) return 0.0;
    if (hw.l2_cache_bytes == 0) return 0.0;
    if (tensor_bytes <= hw.l2_cache_bytes) {
        // Fits in L2. Empirically ~0.8 due to conflict misses and
        // capacity pressure from concurrent kernels.
        return 0.8;
    }
    // Doesn't fit: hit rate scales with L2_size / tensor_size.
    // This is the canonical "streaming" model: each byte is touched
    // L2_size/tensor_size times out of total touches.
    double r = static_cast<double>(hw.l2_cache_bytes) /
               static_cast<double>(tensor_bytes);
    return std::min(0.8, std::max(0.0, r));
}

// static
u64 ArithmeticIntensityAnalysis::matmul_kernel_bytes(u64 M, u64 K, u64 N,
                                                      DType dt,
                                                      const Schedule& s) {
    // After fusion + shared-memory tiling, the kernel-level bytes are:
    //   A: M*K*elem_size / shared_reuse_factor
    //   B: K*N*elem_size / shared_reuse_factor
    //   C: M*N*elem_size (written once, no reuse)
    //
    // shared_reuse_factor = K / k_tile (each A/B tile is loaded once and
    // reused across the K/k_tile outer steps).
    //
    // Note: this is BEFORE L2 effects. L2 reuse is captured at the
    // effective-bytes level (matmul_effective_bytes below).
    //
    // Only `k_tile` is needed here because the shared-memory reuse factor
    // is along the reduction axis. `m_tile` and `n_tile` are consumed by
    // `matmul_effective_bytes` for cross-CTA L2 modeling.

    u32 k_tile = 32;
    bool uses_shared = false;
    for (const auto& t : s.transforms()) {
        switch (t.kind) {
            case TransformKind::Tile:
                if (t.dim == "k") k_tile = static_cast<u32>(t.factor);
                break;
            case TransformKind::Cache:
                if (t.mem == MemorySpace::Shared) uses_shared = true;
                break;
            default: break;
        }
    }

    u64 elem = dtype_size(dt);
    u64 a_bytes = M * K * elem;
    u64 b_bytes = K * N * elem;
    u64 c_bytes = M * N * elem;

    if (uses_shared && k_tile > 0) {
        // Each A and B tile is loaded once into shared memory and reused
        // K/k_tile times across the reduction.
        u64 reuse = std::max<u64>(1, K / k_tile);
        a_bytes /= reuse;
        b_bytes /= reuse;
    }

    return a_bytes + b_bytes + c_bytes;
}

// static
u64 ArithmeticIntensityAnalysis::matmul_effective_bytes(u64 M, u64 K, u64 N,
                                                         DType dt,
                                                         const Schedule& s,
                                                         const HardwareModel& hw) {
    // Effective bytes = bytes that actually miss every cache level.
    //
    // For a tiled matmul with tiles (m_tile, n_tile, k_tile):
    //   A (M*K) is read by (M/m_tile) * (N/n_tile) CTAs in total,
    //     but each A tile (m_tile * K) is reused across (N/n_tile) CTAs
    //     in the column direction. So the cross-CTA L2 reuse factor for
    //     A is (N/n_tile), capped by the L2 hit rate when the tile fits.
    //   B (K*N) is similarly reused across (M/m_tile) CTAs.
    //   C (M*N) is write-only and gets no L2 reuse benefit.
    //
    // The accumulator (m_tile * n_tile per CTA) lives in registers and
    // never touches memory, so it contributes ZERO effective bytes.

    u32 m_tile = 64, n_tile = 64, k_tile = 32;
    bool uses_shared = false;
    u32 vector_width = 1;
    for (const auto& t : s.transforms()) {
        switch (t.kind) {
            case TransformKind::Tile:
                if (t.dim == "m") m_tile = static_cast<u32>(t.factor);
                else if (t.dim == "n") n_tile = static_cast<u32>(t.factor);
                else if (t.dim == "k") k_tile = static_cast<u32>(t.factor);
                break;
            case TransformKind::Cache:
                if (t.mem == MemorySpace::Shared) uses_shared = true;
                break;
            case TransformKind::Vectorize:
                vector_width = static_cast<u32>(t.factor);
                break;
            default: break;
        }
    }

    u64 elem = dtype_size(dt);
    u64 a_bytes = M * K * elem;
    u64 b_bytes = K * N * elem;
    u64 c_bytes = M * N * elem;

    // Shared-memory reuse: each tile loaded once and reused K/k_tile times.
    if (uses_shared && k_tile > 0) {
        u64 reuse = std::max<u64>(1, K / k_tile);
        a_bytes /= reuse;
        b_bytes /= reuse;
    }

    // L2 reuse — base hit rate from the L2-size-vs-tensor-size model.
    // A is read M*K bytes; B is read K*N bytes.
    double l2_a = l2_hit_rate(M * K * elem, hw);
    double l2_b = l2_hit_rate(K * N * elem, hw);

    // Cross-CTA L2 reuse (only meaningful when tiling is in effect).
    // Each A tile (m_tile x K) is loaded by (N/n_tile) CTAs in the N
    // dimension; if those CTAs run close enough in time that the tile
    // stays in L2, only the first CTA's load goes to DRAM. We model this
    // as an additive boost to the L2 hit rate, bounded by 0.95 (some
    // capacity / conflict misses are unavoidable).
    //
    // The boost per doubling of reuse is ~0.1 (empirical, derived from
    // the streaming-with-reuse model in Hong & Kim 2009). The reuse
    // factor is (N/n_tile) for A and (M/m_tile) for B, both floored at 1.
    const double reuse_boost_per_doubling = 0.1;
    const double max_l2_hit               = 0.95;
    if (n_tile > 0) {
        u64 a_cta_reuse = std::max<u64>(1, N / n_tile);
        if (a_cta_reuse > 1) {
            double boost = std::log2(static_cast<double>(a_cta_reuse))
                         * reuse_boost_per_doubling;
            l2_a = std::min(max_l2_hit, l2_a + boost);
        }
    }
    if (m_tile > 0) {
        u64 b_cta_reuse = std::max<u64>(1, M / m_tile);
        if (b_cta_reuse > 1) {
            double boost = std::log2(static_cast<double>(b_cta_reuse))
                         * reuse_boost_per_doubling;
            l2_b = std::min(max_l2_hit, l2_b + boost);
        }
    }

    // Effective bytes after L2 misses. Cast to double before multiplying
    // to avoid implicit u64 -> double conversion warnings.
    double eff_a = static_cast<double>(a_bytes) * (1.0 - l2_a);
    double eff_b = static_cast<double>(b_bytes) * (1.0 - l2_b);
    double eff_c = static_cast<double>(c_bytes); // C is write-only, no L2 reuse benefit on write

    // Vectorization reduces transaction count (fewer load instructions,
    // better coalescing). We model this as a slight reduction in effective
    // bytes — the same data is moved, but with fewer transactions.
    if (vector_width > 1) {
        double coalescing_bonus = 1.0 - 0.05 * std::log2(double(vector_width));
        coalescing_bonus = std::max(0.7, coalescing_bonus);
        eff_a *= coalescing_bonus;
        eff_b *= coalescing_bonus;
    }

    return static_cast<u64>(eff_a + eff_b + eff_c);
}

void ArithmeticIntensityAnalysis::compute() {
    Module& m = am_.module();
    total_flops_ = 0;
    total_bytes_ = 0;
    total_effective_bytes_ = 0;

    Schedule default_schedule; // empty schedule = no tiling

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
                        u64 dv = dim_value_or_zero(d);
                        if (dv == 0) return 0;
                        n *= dv;
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

            // For matmul ops, also compute kernel + effective intensity.
            if (op.opcode == OP_MATMUL && op.operands.size() == 2) {
                auto a = op.operands[0].as_tensor();
                auto b = op.operands[1].as_tensor();
                if (a && b && a->shape.rank() >= 2 && b->shape.rank() >= 2 &&
                    a->shape[a->shape.rank() - 2]->is_constant() &&
                    a->shape[a->shape.rank() - 1]->is_constant() &&
                    b->shape[b->shape.rank() - 1]->is_constant()) {
                    u64 M = dim_value_or_zero(a->shape[a->shape.rank() - 2]);
                    u64 K = dim_value_or_zero(a->shape[a->shape.rank() - 1]);
                    u64 N = dim_value_or_zero(b->shape[b->shape.rank() - 1]);

                    KernelIntensity ki;
                    ki.flops = oi.flops;
                    ki.graph_bytes = oi.total_bytes();
                    ki.kernel_bytes = matmul_kernel_bytes(M, K, N,
                                                          a->dtype,
                                                          default_schedule);
                    ki.effective_bytes = matmul_effective_bytes(M, K, N,
                                                                a->dtype,
                                                                default_schedule,
                                                                hw_);
                    ki.bound = classify(ki.flops, ki.effective_bytes, 32);
                    kernel_intensities_[op.id] = ki;

                    total_effective_bytes_ += ki.effective_bytes;
                } else {
                    total_effective_bytes_ += oi.total_bytes();
                }
            } else {
                // Non-matmul ops: effective == graph bytes (no special model).
                total_effective_bytes_ += oi.total_bytes();
            }

            per_op_[op.id] = oi;
        }
    }

    module_intensity_ = total_bytes_ > 0
        ? static_cast<double>(total_flops_) / static_cast<double>(total_bytes_)
        : 0.0;
    module_effective_intensity_ = total_effective_bytes_ > 0
        ? static_cast<double>(total_flops_) / static_cast<double>(total_effective_bytes_)
        : 0.0;
    module_bound_ = classify(total_flops_, total_effective_bytes_, 32);
}

} // namespace cg
