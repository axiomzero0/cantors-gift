// cost/cost_model_v2.cpp - next-generation cost model implementation
#include "cg/cost/cost_model_v2.hpp"

#include <algorithm>
#include <cmath>

namespace cg {

u32 AnalyticalCostModelV2::estimate_warps_per_sm(
    u32 regs_per_thread, u32 shared_mem_per_block,
    u32 threads_per_block) const {
    // GPU occupancy is limited by three resources:
    //   1. Register file: SM has a fixed register file (e.g., 65536 on A100).
    //      Max warps = register_file / (regs_per_thread * 32).
    //   2. Shared memory: Max blocks = shared_mem_per_sm / shared_mem_per_block.
    //      Max warps = max_blocks * (threads_per_block / warp_size).
    //   3. Max warps per SM (hardware limit, e.g., 64 on A100).

    u32 warp_size = 32;
    u32 regs_per_warp = regs_per_thread * warp_size;
    u32 register_file = 65536; // A100 default
    u32 max_warps_regs = register_file / std::max(1u, regs_per_warp);

    u32 shared_mem_per_sm = static_cast<u32>(hw_.shared_mem_bytes);
    u32 max_blocks_smem = shared_mem_per_sm / std::max(1u, shared_mem_per_block);
    u32 max_warps_smem = max_blocks_smem * (threads_per_block / warp_size);

    u32 max_warps_hw = 64; // A100: 64 warps/SM max

    return std::min({max_warps_regs, max_warps_smem, max_warps_hw});
}

double AnalyticalCostModelV2::wave_quantization_penalty(
    u32 num_blocks, u32 num_sms) const {
    if (num_sms == 0) return 0;

    u32 full_waves = num_blocks / num_sms;
    u32 remainder = num_blocks % num_sms;

    if (remainder == 0) return 0;

    // The last wave uses only `remainder` SMs out of `num_sms`.
    // The time for the last wave is the same as a full wave (max time),
    // but only `remainder/num_sms` fraction of the work is done.
    // This means the effective time is (full_waves + 1) wave times,
    // but only `full_waves + remainder/num_sms` waves of work.
    // The penalty is the wasted fraction:
    //   penalty = 1 - (full_waves + remainder/num_sms) / (full_waves + 1)
    //           = (1 - remainder/num_sms) / (full_waves + 1)
    double work_ratio = static_cast<double>(remainder) / num_sms;
    double penalty = (1.0 - work_ratio) / (full_waves + 1);
    return penalty;
}

double AnalyticalCostModelV2::bank_conflict_penalty(
    u64 shared_bytes, u32 conflict_ways) const {
    if (conflict_ways <= 1) return 0;

    // N-way bank conflict divides shared memory bandwidth by N.
    // The extra time is: (shared_bytes / bandwidth) * (N - 1)
    double shared_bw = hw_.memory.get(MemorySpace::Shared);
    if (shared_bw <= 0) return 0;
    double base_time = static_cast<double>(shared_bytes) / shared_bw;
    return base_time * (conflict_ways - 1);
}

AnalyticalCostModelV2::CostBreakdown
AnalyticalCostModelV2::estimate(const CostFeatures& f) const {
    CostBreakdown out;

    // 1. Compute time
    double peak = hw_.peak_flops(f.tc_dtype, f.uses_tensor_core);
    if (peak <= 0) peak = hw_.peak_flops(DType::F32);
    out.compute_sec = static_cast<double>(f.flops) / peak;

    // 2. Global memory time
    double global_bw = hw_.memory.get(MemorySpace::Generic);
    if (global_bw <= 0) global_bw = 1e12;
    u64 total_global_bytes = f.bytes_global_load + f.bytes_global_store;
    out.memory_global_sec = static_cast<double>(total_global_bytes) / global_bw;

    // 3. Shared memory time (with bank conflict penalty)
    double shared_bw = hw_.memory.get(MemorySpace::Shared);
    if (shared_bw <= 0) shared_bw = 1e12;
    u64 total_shared_bytes = f.bytes_shared_load + f.bytes_shared_store;
    out.memory_shared_sec = static_cast<double>(total_shared_bytes) / shared_bw;
    out.bank_conflict_sec = bank_conflict_penalty(total_shared_bytes, f.bank_conflict_ways);

    // 4. Pipeline overlap
    // With N stages: overlapped = max(C, M_global + M_shared + bank_conflict)
    // adjusted by overlap factor.
    double total_memory = out.memory_global_sec + out.memory_shared_sec + out.bank_conflict_sec;
    if (f.num_pipeline_stages > 1) {
        // Overlap: the compute for stage K overlaps with memory loads for stage K+N.
        // Effective overlapped time ≈ max(C, M) + min(C, M) / N
        double min_tm = std::min(out.compute_sec, total_memory);
        double max_tm = std::max(out.compute_sec, total_memory);
        out.overlapped_sec = max_tm + min_tm / f.num_pipeline_stages;
    } else {
        out.overlapped_sec = std::max(out.compute_sec, total_memory);
    }

    // 5. Occupancy stall
    u32 warps_per_sm = estimate_warps_per_sm(
        f.registers_per_thread, f.shared_mem_per_block, f.threads_per_block);
    out.estimated_warps_per_sm = warps_per_sm;
    u32 max_warps = 64; // A100
    out.estimated_occupancy_pct = (warps_per_sm * 100) / max_warps;

    // Low occupancy → can't hide latency → stall
    // If occupancy < 25%, add stall proportional to (1 - occupancy)
    if (out.estimated_occupancy_pct < 50) {
        double occupancy_factor = static_cast<double>(out.estimated_occupancy_pct) / 100.0;
        out.stall_sec = out.overlapped_sec * (1.0 - occupancy_factor) * 0.5;
    }

    // 6. Wave quantization
    u32 num_sms = hw_.num_cores; // SM count
    double wave_penalty = wave_quantization_penalty(f.num_blocks, num_sms);
    out.wave_quant_sec = out.overlapped_sec * wave_penalty;

    // 7. Total
    out.total_sec = out.overlapped_sec + out.stall_sec + out.wave_quant_sec;

    return out;
}

CostFeatures AnalyticalCostModelV2::extract_features(
    const Schedule& schedule,
    u64 M, u64 N, u64 K,
    DType dtype) const {
    CostFeatures f;
    f.flops = 2 * M * N * K;
    f.useful_flops = f.flops;
    f.uses_tensor_core = false;
    f.tc_dtype = dtype;

    // Extract from schedule
    for (const auto& t : schedule.transforms()) {
        switch (t.kind) {
            case TransformKind::Tile:
                if (t.dim == "m") f.m_tile = static_cast<u32>(t.factor);
                else if (t.dim == "n") f.n_tile = static_cast<u32>(t.factor);
                else if (t.dim == "k") f.k_tile = static_cast<u32>(t.factor);
                break;
            case TransformKind::Vectorize:
                f.vector_width = static_cast<u32>(t.factor);
                break;
            case TransformKind::Cache:
                if (t.mem == MemorySpace::Shared) f.has_async_copy = true;
                break;
            case TransformKind::Bind:
                if (t.target == "tensor_core") {
                    f.uses_tensor_core = true;
                    f.tc_dtype = DType::F16;
                }
                break;
            case TransformKind::Pipeline:
                f.num_pipeline_stages = static_cast<u32>(t.factor);
                break;
            default: break;
        }
    }

    // Memory bytes
    u32 elem_size = dtype_size(dtype);
    f.bytes_global_load = (M * K + K * N) * elem_size;
    f.bytes_global_store = M * N * elem_size;

    // Shared memory
    if (f.has_async_copy) {
        f.bytes_shared_load = (f.m_tile * f.k_tile + f.k_tile * f.n_tile) * elem_size;
    }

    // Shared memory per block
    f.shared_mem_per_block = static_cast<u32>(f.bytes_shared_load);

    // Number of blocks
    u32 blocks_m = static_cast<u32>((M + f.m_tile - 1) / f.m_tile);
    u32 blocks_n = static_cast<u32>((N + f.n_tile - 1) / f.n_tile);
    f.num_blocks = blocks_m * blocks_n;
    f.num_sms = hw_.num_cores;

    // Estimate register usage
    // For MMA: 4 A + 2 B + 4 C/D = 10 registers per fragment set
    // Plus address computation: ~20 more
    f.registers_per_thread = f.uses_tensor_core ? 40 : 32;

    // Threads per block
    f.threads_per_block = 256; // 8 warps

    // L2 hit rate for B tensor (rough estimate)
    // B is [K, N], each element is reused M/m_tile times.
    // If K*N*elem_size < L2 cache, B fits in L2 → high hit rate.
    u64 b_size = K * N * elem_size;
    if (b_size < hw_.l2_cache_bytes) {
        f.l2_hit_rate = 0.8;
    } else {
        f.l2_hit_rate = static_cast<double>(hw_.l2_cache_bytes) / b_size;
    }

    return f;
}

// ---- LearnedCostModel ----

void LearnedCostModel::train(
    const std::vector<std::pair<CostFeatures, double>>& data) {
    if (data.empty()) return;

    // Simple linear regression on key features.
    // In a real implementation, this would be XGBoost.
    // Features: flops, bytes_global, num_pipeline_stages, m_tile, n_tile,
    //           k_tile, registers_per_thread, shared_mem_per_block
    usize n = data.size();
    const usize num_features = 8;

    // Normalize features and compute least-squares solution.
    // For simplicity, we use gradient descent.
    weights_.assign(num_features, 0.0);
    bias_ = 0.0;

    auto extract = [](const CostFeatures& f) -> std::vector<double> {
        return {
            static_cast<double>(f.flops) / 1e9,
            static_cast<double>(f.bytes_global_load + f.bytes_global_store) / 1e6,
            static_cast<double>(f.num_pipeline_stages),
            static_cast<double>(f.m_tile),
            static_cast<double>(f.n_tile),
            static_cast<double>(f.k_tile),
            static_cast<double>(f.registers_per_thread),
            static_cast<double>(f.shared_mem_per_block) / 1024.0,
        };
    };

    double lr = 0.001;
    for (int iter = 0; iter < 1000; ++iter) {
        std::vector<double> grads(num_features, 0.0);
        double bias_grad = 0;
        for (const auto& [feat, target] : data) {
            auto x = extract(feat);
            double pred = bias_;
            for (usize i = 0; i < num_features; ++i)
                pred += weights_[i] * x[i];
            double error = pred - target;
            for (usize i = 0; i < num_features; ++i)
                grads[i] += error * x[i];
            bias_grad += error;
        }
        for (usize i = 0; i < num_features; ++i)
            weights_[i] -= lr * grads[i] / n;
        bias_ -= lr * bias_grad / n;
    }

    trained_ = true;
}

std::optional<double> LearnedCostModel::predict(const CostFeatures& f) const {
    if (!trained_) return std::nullopt;

    auto extract = [](const CostFeatures& f) -> std::vector<double> {
        return {
            static_cast<double>(f.flops) / 1e9,
            static_cast<double>(f.bytes_global_load + f.bytes_global_store) / 1e6,
            static_cast<double>(f.num_pipeline_stages),
            static_cast<double>(f.m_tile),
            static_cast<double>(f.n_tile),
            static_cast<double>(f.k_tile),
            static_cast<double>(f.registers_per_thread),
            static_cast<double>(f.shared_mem_per_block) / 1024.0,
        };
    };

    auto x = extract(f);
    double pred = bias_;
    for (usize i = 0; i < weights_.size(); ++i)
        pred += weights_[i] * x[i];
    return std::max(0.0, pred);
}

} // namespace cg
