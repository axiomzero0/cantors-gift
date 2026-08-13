// pipeline/matmul_pipeline.cpp - end-to-end matmul optimization pipeline
//
// This file implements the orchestrator declared in matmul_pipeline.hpp.
// It wires together IR construction, e-graph saturation, schedule-space
// generation, v2 cost-model pruning, and Bayesian autotuning.
//
// The compile-time vs autotuner separation is preserved:
//   - The v2 cost model prunes the schedule space from O(100s) to O(10s)
//     of candidates using analytical modeling (wave quantization, occupancy,
//     L2 hit rate, bank conflicts, pipeline overlap).
//   - The Bayesian autotuner then spends actual benchmark measurements
//     only on the top-k promising candidates, using GP+EI to decide
//     where each measurement is most informative.
#include "cg/pipeline/matmul_pipeline.hpp"

#include "cg/ir/printer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace cg {

namespace {

// Helper: elapsed seconds between two time points.
double elapsed_sec(std::chrono::steady_clock::time_point t0,
                   std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
}

} // namespace

// ---------------------------------------------------------------------------
// Stage 1: IR construction
// ---------------------------------------------------------------------------
std::pair<std::shared_ptr<Module>, Operation*>
MatmulPipeline::build_ir(std::chrono::steady_clock::time_point& t0,
                          CompileTiming& timing) const {
    auto t1 = std::chrono::steady_clock::now();
    auto module = std::make_shared<Module>();
    auto M = static_cast<i64>(cfg_.M);
    auto K = static_cast<i64>(cfg_.K);
    auto N = static_cast<i64>(cfg_.N);

    std::vector<TypePtr> operand_types = {
        make_tensor_type({M, K}, cfg_.dtype),  // A
        make_tensor_type({K, N}, cfg_.dtype),  // B
    };
    if (cfg_.fuse_bias_relu) {
        operand_types.push_back(make_tensor_type({N}, cfg_.dtype)); // bias
    }
    std::vector<TypePtr> result_types = {
        make_tensor_type({M, N}, cfg_.dtype),
    };
    auto f = module->create_function("matmul_fused", operand_types, result_types);
    Builder b(f);
    auto A = f->args()[0];
    auto B = f->args()[1];
    auto mm = b.matmul(A, B);
    Value epilogue = mm;
    if (cfg_.fuse_bias_relu) {
        auto bias = f->args()[2];
        // bias is shape [N]; reshape to [1, N] then broadcast to [M, N].
        auto bias_2d = b.reshape(bias, {1, N});
        auto bias_bc = b.broadcast(bias_2d, {M, N});
        auto biased = b.add(mm, bias_bc);
        epilogue = b.relu(biased);
    }
    b.output_tensor(epilogue);

    // Find the matmul op for later reference.
    Operation* mm_op = nullptr;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_MATMUL) { mm_op = &op; break; }
    }

    timing.ir_construction_sec = elapsed_sec(t0, t1);
    t0 = std::chrono::steady_clock::now();
    return {module, mm_op};
}

// ---------------------------------------------------------------------------
// Stage 2: GTA (arithmetic intensity analysis with three-level breakdown)
// ---------------------------------------------------------------------------
void MatmulPipeline::run_gta(Module& m, ArithmeticIntensityAnalysis& ai,
                              std::chrono::steady_clock::time_point& t0,
                              CompileTiming& timing,
                              RooflineBreakdown& roofline) const {
    auto t1 = std::chrono::steady_clock::now();
    ai.set_hardware(hw_);

    // Find the matmul op.
    u64 M = cfg_.M, K = cfg_.K, N = cfg_.N;
    u64 elem = dtype_size(cfg_.dtype);

    roofline.flops = 2 * M * K * N;
    roofline.graph_bytes = (M * K + K * N + M * N) * elem;
    // Kernel bytes: after fusion (bias+relu fused into the matmul epilogue,
    // so bias is read once per CTA from constant / shared, not from global).
    // For now, use a default schedule (no tiling) — the v2 cost model will
    // refine this per-schedule.
    Schedule default_sched;
    roofline.kernel_bytes = ArithmeticIntensityAnalysis::matmul_kernel_bytes(
        M, K, N, cfg_.dtype, default_sched);
    roofline.effective_bytes = ArithmeticIntensityAnalysis::matmul_effective_bytes(
        M, K, N, cfg_.dtype, default_sched, hw_);

    roofline.graph_intensity = roofline.graph_bytes > 0
        ? double(roofline.flops) / double(roofline.graph_bytes) : 0.0;
    roofline.kernel_intensity = roofline.kernel_bytes > 0
        ? double(roofline.flops) / double(roofline.kernel_bytes) : 0.0;
    roofline.effective_intensity = roofline.effective_bytes > 0
        ? double(roofline.flops) / double(roofline.effective_bytes) : 0.0;
    roofline.ridge_f32 = hw_.roofline_ridge(DType::F32, false);
    roofline.ridge_f16_tc = hw_.roofline_ridge(DType::F16, true);
    // Classify using effective intensity vs F32 ridge.
    if (roofline.effective_intensity > roofline.ridge_f32 * 4.0)
        roofline.effective_bound = BoundClass::ComputeBound;
    else if (roofline.effective_intensity < roofline.ridge_f32 / 4.0)
        roofline.effective_bound = BoundClass::MemoryBound;
    else
        roofline.effective_bound = BoundClass::Balanced;

    timing.gta_sec = elapsed_sec(t0, t1);
    t0 = std::chrono::steady_clock::now();
    (void)m; // unused — analysis is computed via ai
}

// ---------------------------------------------------------------------------
// Stage 3: E-graph saturation
// ---------------------------------------------------------------------------
std::vector<EGraph::Rewrite> MatmulPipeline::build_rewrite_rules() const {
    // Get the rule library filtered by numerical mode.
    auto rule_set = get_rewrite_rules(cfg_.mode);
    std::vector<EGraph::Rewrite> out;
    out.reserve(rule_set.size());
    for (auto& r : rule_set) {
        out.push_back(r.rule);
    }
    return out;
}

EClassId MatmulPipeline::run_egraph(std::chrono::steady_clock::time_point& t0,
                                     CompileTiming& timing) const {
    auto t1 = std::chrono::steady_clock::now();
    EGraph g;

    // Build e-graph nodes for: relu(matmul(A, B) + bias)
    // (or just matmul(A, B) if no epilogue).
    auto A = g.add({"var", {}, cfg_.dtype, {}});
    auto B = g.add({"var", {}, cfg_.dtype, {}});
    auto mm = g.add({"matmul", {A, B}, cfg_.dtype, {}});

    EClassId root = mm;
    if (cfg_.fuse_bias_relu) {
        auto bias = g.add({"var", {}, cfg_.dtype, {}});
        auto add = g.add({"add", {mm, bias}});
        auto relu = g.add({"relu", {add}});
        root = relu;
    }

    // Saturate with the full rule library.
    auto rules = build_rewrite_rules();
    g.saturate(rules, 8);

    timing.egraph_saturation_sec = elapsed_sec(t0, t1);
    t0 = std::chrono::steady_clock::now();
    return root;
}

// ---------------------------------------------------------------------------
// Stage 4: Schedule-space generation
// ---------------------------------------------------------------------------
ScheduleSpace MatmulPipeline::gen_schedule_space(
    std::chrono::steady_clock::time_point& t0,
    CompileTiming& timing) const {
    auto t1 = std::chrono::steady_clock::now();

    // Start with the basic grid: m_tiles × n_tiles × k_tiles × vector_widths.
    ScheduleSpace space = ScheduleSpace::grid_matmul(
        cfg_.m_tiles, cfg_.n_tiles, cfg_.k_tiles, cfg_.vector_widths);

    // For each base schedule, also create a TC variant (Bind tensor_core).
    // The TC variant unlocks the F16 tensor-core throughput (312 TF on A100).
    std::vector<Schedule> tc_schedules;
    for (auto& s : space.schedules()) {
        Schedule tc = s;
        tc.add({TransformKind::Bind, "tc", 0, 0, "tensor_core", MemorySpace::Generic});
        tc_schedules.push_back(tc);
    }
    for (auto& s : tc_schedules) space.add(std::move(s));

    timing.scheduling_sec = elapsed_sec(t0, t1);
    t0 = std::chrono::steady_clock::now();
    return space;
}

// ---------------------------------------------------------------------------
// Stage 5: Prune schedule space via v2 cost model
// ---------------------------------------------------------------------------
std::vector<std::pair<Schedule, AnalyticalCostModelV2::CostBreakdown>>
MatmulPipeline::prune_schedule_space(const ScheduleSpace& space,
                                      std::chrono::steady_clock::time_point& t0,
                                      CompileTiming& timing) const {
    auto t1 = std::chrono::steady_clock::now();

    std::vector<std::pair<Schedule, AnalyticalCostModelV2::CostBreakdown>> all;
    all.reserve(space.size());
    for (auto& s : space.schedules()) {
        auto features = cost_v2_.extract_features(
            s, cfg_.M, cfg_.N, cfg_.K, cfg_.dtype);
        // Override tc_dtype if the schedule binds tensor_core.
        for (const auto& t : s.transforms()) {
            if (t.kind == TransformKind::Bind && t.target == "tensor_core") {
                features.uses_tensor_core = true;
                features.tc_dtype = DType::F16;
                break;
            }
        }
        auto bd = cost_v2_.estimate(features);
        all.emplace_back(s, bd);
    }
    // Sort by total_sec ascending.
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_sec < b.second.total_sec;
              });
    if (all.size() > cfg_.top_k_prune) all.resize(cfg_.top_k_prune);

    // The scheduling_sec is the time spent in gen_schedule_space + this prune.
    // gen_schedule_space already wrote its own time; we add the prune time
    // to scheduling_sec here.
    timing.scheduling_sec += elapsed_sec(t0, t1);
    t0 = std::chrono::steady_clock::now();
    return all;
}

// ---------------------------------------------------------------------------
// Stage 6: Bayesian autotuner
// ---------------------------------------------------------------------------
AutotuneResult MatmulPipeline::run_autotuner(
    const std::vector<Schedule>& top_k,
    BenchmarkFn benchmark,
    std::chrono::steady_clock::time_point& t0,
    CompileTiming& timing) const {
    auto t1 = std::chrono::steady_clock::now();
    // Pack the top_k into a ScheduleSpace.
    ScheduleSpace pruned;
    for (auto& s : top_k) pruned.add(s);
    auto result = bayesian_autotune(pruned, benchmark,
                                     cfg_.max_benchmarks, cfg_.initial_random);
    timing.autotuning_sec = elapsed_sec(t0, t1);
    t0 = std::chrono::steady_clock::now();
    return result;
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------
MatmulPipelineResult MatmulPipeline::run(BenchmarkFn benchmark) const {
    MatmulPipelineResult result;
    auto t_start = std::chrono::steady_clock::now();
    auto t0 = t_start;

    // Stage 1: IR
    auto [module, mm_op] = build_ir(t0, result.timing);

    // Stage 2: GTA
    AnalysisManager am(*module);
    auto& ai = am.get<ArithmeticIntensityAnalysis>();
    run_gta(*module, ai, t0, result.timing, result.roofline);

    // Stage 3: E-graph
    // (We don't actually feed the e-graph output back into the IR yet —
    // the e-graph is a separate optimization path that discovers equivalent
    // expressions. For now we run it to verify the rules fire and to
    // measure the saturation time.)
    [[maybe_unused]] auto root = run_egraph(t0, result.timing);

    // Stage 4: Schedule space
    auto space = gen_schedule_space(t0, result.timing);
    result.schedule_space_size = space.size();

    // Stage 5: Prune
    auto pruned = prune_schedule_space(space, t0, result.timing);
    result.pruned_space_size = pruned.size();

    // Stage 6: Autotune
    std::vector<Schedule> top_k;
    top_k.reserve(pruned.size());
    for (auto& [s, bd] : pruned) top_k.push_back(s);
    auto tune_result = run_autotuner(top_k, benchmark, t0, result.timing);

    result.best_schedule = tune_result.best_schedule;
    result.best_runtime_sec = tune_result.best_runtime;
    result.autotune_history = tune_result.runtime_history;
    result.autotune_benchmarks = tune_result.total_benchmarks;

    // Compute cost breakdown for the best schedule.
    auto best_features = cost_v2_.extract_features(
        result.best_schedule, cfg_.M, cfg_.N, cfg_.K, cfg_.dtype);
    for (const auto& t : result.best_schedule.transforms()) {
        if (t.kind == TransformKind::Bind && t.target == "tensor_core") {
            best_features.uses_tensor_core = true;
            best_features.tc_dtype = DType::F16;
            break;
        }
    }
    result.cost_breakdown = cost_v2_.estimate(best_features);
    result.analytical_estimate_sec = result.cost_breakdown.total_sec;
    result.uses_tensor_core = best_features.uses_tensor_core;

    // Stage 7: Codegen (placeholder — actual PTX emission is in ptx_mma.cpp).
    auto t_codegen_end = std::chrono::steady_clock::now();
    result.timing.codegen_sec = elapsed_sec(t0, t_codegen_end);

    // Total compile time.
    result.timing.total_sec = elapsed_sec(t_start, t_codegen_end);

    (void)mm_op;
    return result;
}

// Convenience: use v2 cost model as the benchmark function.
MatmulPipelineResult MatmulPipeline::run_with_analytical_benchmark() const {
    // The benchmark function uses the v2 cost model's total_sec as the
    // "measured" runtime. This lets us exercise the full pipeline without
    // any actual GPU hardware.
    auto benchmark = [this](const Schedule& s) -> double {
        auto features = cost_v2_.extract_features(
            s, cfg_.M, cfg_.N, cfg_.K, cfg_.dtype);
        for (const auto& t : s.transforms()) {
            if (t.kind == TransformKind::Bind && t.target == "tensor_core") {
                features.uses_tensor_core = true;
                features.tc_dtype = DType::F16;
                break;
            }
        }
        return cost_v2_.estimate(features).total_sec;
    };
    return run(benchmark);
}

} // namespace cg
