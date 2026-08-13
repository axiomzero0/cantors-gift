// cost/cost_model_v2.hpp - next-generation cost model
//
// Replaces the simple max(compute, memory) roofline with a model that
// captures what actually matters:
//
//   T_kernel = max(T_compute overlapped with T_memory) + T_stall
//
// Where T_stall includes:
//   - Wave quantization: 81 waves on 80 SMs = 2x time of 80 waves
//   - Bank conflicts: shared memory load throughput drops 4x with 4-way conflicts
//   - Register pressure → occupancy → latency hiding
//   - L2 cache hit rate for the second-most-touched tensor
//   - Tensor core utilization = useful FLOPs / peak TC FLOPs
//
// The model has two modes:
//   1. Analytical: fast, used for pruning the search space
//   2. Learned (XGBoost on features): trained on (features, measured_runtime) pairs
//      to correct the analytical model's systematic biases
#pragma once

#include "cg/cost/hardware_model.hpp"
#include "cg/core/util.hpp"
#include "cg/schedule/schedule.hpp"

#include <optional>
#include <vector>

namespace cg {

// Features extracted from a (kernel, schedule, hardware) triple.
// These are the inputs to both the analytical and learned cost models.
struct CostFeatures {
    // Compute features
    u64 flops = 0;
    u64 useful_flops = 0;       // FLOPs that contribute to the output (excludes padding)
    bool uses_tensor_core = false;
    DType tc_dtype = DType::F32; // tensor core compute dtype

    // Memory features
    u64 bytes_global_load = 0;
    u64 bytes_global_store = 0;
    u64 bytes_shared_load = 0;
    u64 bytes_shared_store = 0;
    u64 bytes_constant = 0;

    // Pipeline features
    u32 num_pipeline_stages = 1;
    bool has_async_copy = false;

    // Occupancy features
    u32 registers_per_thread = 32;     // estimated register usage
    u32 shared_mem_per_block = 0;      // bytes
    u32 threads_per_block = 256;
    u32 warp_size = 32;

    // Wave quantization
    u32 num_blocks = 0;                // total blocks needed
    u32 num_sms = 0;                   // SMs on the GPU

    // Bank conflict estimate
    u32 bank_conflict_ways = 1;        // 1 = no conflicts, 4 = 4-way conflicts

    // L2 cache
    double l2_hit_rate = 0.0;          // estimated L2 hit rate for B tensor

    // Tile shape
    u32 m_tile = 64;
    u32 n_tile = 64;
    u32 k_tile = 32;

    // Vector width
    u32 vector_width = 4;              // floats per load/store
};

// The analytical cost model.
class AnalyticalCostModelV2 {
public:
    explicit AnalyticalCostModelV2(HardwareModel hw) : hw_(std::move(hw)) {}

    struct CostBreakdown {
        double compute_sec = 0;
        double memory_global_sec = 0;
        double memory_shared_sec = 0;
        double overlapped_sec = 0;    // after pipeline overlap
        double wave_quant_sec = 0;    // wave quantization penalty
        double bank_conflict_sec = 0; // bank conflict penalty
        double stall_sec = 0;         // register pressure / occupancy stall
        double total_sec = 0;         // final estimate

        // Occupancy estimate
        u32 estimated_warps_per_sm = 0;
        u32 estimated_occupancy_pct = 0;

        // ---- Wave / SM utilization (first-class metrics) ----
        // "1 wave" is NOT the same as "perfect utilization". If 64 blocks
        // run on 108 SMs, only 64 SMs receive work; the other 44 sit idle.
        // The SM utilization is the fraction of SMs that actually receive
        // a block in the steady state, averaged across waves.
        u32 num_blocks = 0;
        u32 num_sms = 0;
        u32 num_waves = 0;                // ceil(num_blocks / num_sms)
        double sm_utilization_pct = 0.0;  // (active SMs / total SMs) * 100, averaged
        double tail_efficiency_pct = 0.0; // work done / wave time, in [0,100]
        // For partial last wave: how many SMs sit idle in the tail wave.
        u32 idle_sms_in_tail = 0;
    };

    CostBreakdown estimate(const CostFeatures& features) const;

    // Quick estimate (just the total, no breakdown).
    double estimate_total(const CostFeatures& features) const {
        return estimate(features).total_sec;
    }

    // Extract features from a Schedule + hardware.
    CostFeatures extract_features(
        const Schedule& schedule,
        u64 M, u64 N, u64 K,
        DType dtype) const;

    const HardwareModel& hardware() const { return hw_; }

private:
    HardwareModel hw_;

    // Occupancy estimation: registers_per_thread + shared_mem → warps/SM → occupancy.
    u32 estimate_warps_per_sm(u32 regs_per_thread, u32 shared_mem_per_block,
                               u32 threads_per_block) const;

    // Wave quantization: ceil(num_blocks / num_sms) determines the number of
    // "waves" of blocks. Each wave takes max(compute, memory) time. If
    // num_blocks doesn't divide evenly, the last wave is partial (wasted SMs).
    double wave_quantization_penalty(u32 num_blocks, u32 num_sms) const;

    // Bank conflict penalty: N-way bank conflict divides shared memory
    // bandwidth by N.
    double bank_conflict_penalty(u64 shared_bytes, u32 conflict_ways) const;
};

// Learned cost model (XGBoost on features).
// This is a stub for the interface; the actual training would be done
// offline and the trained model loaded at runtime.
class LearnedCostModel {
public:
    LearnedCostModel() = default;

    // Train on (features, measured_runtime) pairs.
    void train(const std::vector<std::pair<CostFeatures, double>>& data);

    // Predict runtime for a new feature vector.
    std::optional<double> predict(const CostFeatures& features) const;

    // True if the model has been trained.
    bool is_trained() const { return trained_; }

private:
    bool trained_ = false;
    // In a real implementation, this would hold the XGBoost model.
    // For now, we use a simple linear model trained on the features.
    std::vector<double> weights_;
    double bias_ = 0;
};

} // namespace cg
