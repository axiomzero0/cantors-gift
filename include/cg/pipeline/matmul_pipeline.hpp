// pipeline/matmul_pipeline.hpp - end-to-end matmul optimization pipeline
//
// Wires together:
//   1. IR construction (relu(matmul(A, B) + bias))
//   2. E-graph saturation with TC eligibility + FMA + layout rules
//   3. Expression extraction via cost function
//   4. Schedule-space generation (tiles × vector widths × TC on/off)
//   5. V2 cost model pruning (with proper wave quantization + SM utilization)
//   6. Bayesian autotuner (GP + EI) on the top-k candidates
//
// The pipeline reports:
//   - Compile-time breakdown (IR, GTA, e-graph, fusion, scheduling,
//     autotuning, codegen, total)
//   - Roofline breakdown (graph / kernel / effective intensity)
//   - SM utilization (so "1 wave" is no longer conflated with "perfect")
//   - Best schedule found by the autotuner
//
// The compile-time vs autotuner separation is preserved:
//   - The analytical cost model narrows the search space aggressively
//   - The Bayesian autotuner spends measurements only where they're informative
//   - This is NOT brute force with a fancy hat.
#pragma once

#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/autotuner/bayesian_optimizer.hpp"
#include "cg/cost/cost_model_v2.hpp"
#include "cg/cost/estimator.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/egraph/egraph.hpp"
#include "cg/egraph/rewrite_rules.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/module.hpp"
#include "cg/numerical/semantics.hpp"
#include "cg/schedule/schedule.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace cg {

// Per-stage compile-time breakdown (seconds).
struct CompileTiming {
    double ir_construction_sec = 0;
    double gta_sec = 0;            // global tensor analysis
    double egraph_saturation_sec = 0;
    double fusion_sec = 0;
    double scheduling_sec = 0;     // schedule-space enumeration + v2 prune
    double autotuning_sec = 0;
    double codegen_sec = 0;
    double total_sec = 0;
};

// Roofline breakdown for the matmul workload.
struct RooflineBreakdown {
    // Graph-level: total FLOPs / total bytes (no fusion, no cache).
    double graph_intensity = 0;
    // Kernel-level: FLOPs / bytes-after-fusion (intermediates dropped).
    double kernel_intensity = 0;
    // Effective: FLOPs / bytes-after-L2-and-shared-reuse.
    // This is what actually determines performance.
    double effective_intensity = 0;
    // Hardware roofline ridge (FLOP/byte). Above = compute-bound,
    // below = memory-bound.
    double ridge_f32 = 0;
    double ridge_f16_tc = 0;
    // Bound classification of the effective kernel.
    BoundClass effective_bound = BoundClass::Balanced;
    // Raw numbers.
    u64 flops = 0;
    u64 graph_bytes = 0;
    u64 kernel_bytes = 0;
    u64 effective_bytes = 0;
};

// Result of running the full matmul pipeline.
struct MatmulPipelineResult {
    // The optimized schedule chosen by the autotuner.
    Schedule best_schedule;
    // The runtime (seconds) of the best schedule, as measured by the
    // autotuner's benchmark function.
    double best_runtime_sec = 0;
    // History of best-runtime as the autotuner progressed.
    std::vector<double> autotune_history;
    // Total number of benchmark evaluations the autotuner used.
    usize autotune_benchmarks = 0;
    // The initial estimate (v2 cost model) for the best schedule,
    // before any benchmarking.
    double analytical_estimate_sec = 0;
    // Cost-model breakdown for the best schedule.
    AnalyticalCostModelV2::CostBreakdown cost_breakdown;
    // Compile-time breakdown.
    CompileTiming timing;
    // Roofline breakdown.
    RooflineBreakdown roofline;
    // Number of candidate schedules considered before pruning.
    usize schedule_space_size = 0;
    // Number of candidates passed to the autotuner after v2 pruning.
    usize pruned_space_size = 0;
    // Whether tensor cores were used in the best schedule.
    bool uses_tensor_core = false;
};

// The matmul pipeline orchestrator.
class MatmulPipeline {
public:
    struct Config {
        // Workload shape.
        u64 M = 1024;
        u64 K = 1024;
        u64 N = 1024;
        DType dtype = DType::F32;
        // Whether to include the bias+relu epilogue.
        bool fuse_bias_relu = true;
        // Schedule-space parameters.
        std::vector<i64> m_tiles = {32, 64, 128};
        std::vector<i64> n_tiles = {32, 64, 128};
        std::vector<i64> k_tiles = {16, 32, 64};
        std::vector<i64> vector_widths = {4, 8, 16};
        // Autotuner budget.
        usize max_benchmarks = 20;
        usize initial_random = 5;
        usize top_k_prune = 8;
        // Numerical mode (controls which e-graph rules fire).
        NumericalMode mode = NumericalMode::FastMath;
    };

    MatmulPipeline(HardwareModel hw, Config cfg)
        : hw_(std::move(hw)), cfg_(std::move(cfg)),
          cost_v2_(hw_), cost_v1_(hw_) {}

    // Run the full pipeline. The `benchmark` function is what the autotuner
    // calls to measure each candidate schedule. In a real run, this would
    // compile + launch + time the kernel; in tests, it can be an analytical
    // surrogate.
    MatmulPipelineResult run(BenchmarkFn benchmark) const;

    // Convenience: use the v2 cost model as the "benchmark" function.
    // This is what you'd use when you can't actually run kernels (e.g. in
    // CI). It exercises the full pipeline without hardware.
    MatmulPipelineResult run_with_analytical_benchmark() const;

    const HardwareModel& hardware() const { return hw_; }
    const Config& config() const { return cfg_; }

private:
    HardwareModel hw_;
    Config cfg_;
    AnalyticalCostModelV2 cost_v2_;
    CostEstimator cost_v1_;

    // Stage 1: build the IR module relu(matmul(A, B) + bias).
    std::pair<std::shared_ptr<Module>, Operation*>
    build_ir(std::chrono::steady_clock::time_point& t0,
             CompileTiming& timing) const;

    // Stage 2: run GTA (arithmetic intensity analysis).
    void run_gta(Module& m, ArithmeticIntensityAnalysis& ai,
                 std::chrono::steady_clock::time_point& t0,
                 CompileTiming& timing,
                 RooflineBreakdown& roofline) const;

    // Stage 3: build e-graph from IR, saturate with rewrite rules.
    // Returns the root e-class id of the matmul expression.
    EClassId run_egraph(std::chrono::steady_clock::time_point& t0,
                        CompileTiming& timing) const;

    // Stage 4: generate schedule space (tiles × vector × TC on/off).
    ScheduleSpace gen_schedule_space(std::chrono::steady_clock::time_point& t0,
                                     CompileTiming& timing) const;

    // Stage 5: prune schedule space via v2 cost model.
    std::vector<std::pair<Schedule, AnalyticalCostModelV2::CostBreakdown>>
    prune_schedule_space(const ScheduleSpace& space,
                         std::chrono::steady_clock::time_point& t0,
                         CompileTiming& timing) const;

    // Stage 6: run Bayesian autotuner on the top-k candidates.
    AutotuneResult run_autotuner(
        const std::vector<Schedule>& top_k,
        BenchmarkFn benchmark,
        std::chrono::steady_clock::time_point& t0,
        CompileTiming& timing) const;

    // Build the e-graph rewrite rule set filtered by the numerical mode.
    std::vector<EGraph::Rewrite> build_rewrite_rules() const;
};

} // namespace cg
