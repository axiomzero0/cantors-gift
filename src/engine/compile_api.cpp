// engine/compile_api.cpp - the engine that turns a task into a result.
//
// This file is the single entry point for the Python API. It:
//   1. Builds IR from the task description
//   2. Selects hardware + numerical mode
//   3. Runs the IterativeDriver (full optimization pipeline)
//   4. Evaluates the cost model on the optimized IR
//   5. Collects stats
//   6. Returns everything as a CompileResult
//
// The Python side never touches any of this directly.
#include "cg/engine/compile_api.hpp"

#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/cost/cost_model_v2.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/iterative_driver.hpp"
#include "cg/optimization/unified/migrated_passes.hpp"

#include <iostream>
#include <sstream>

namespace cg {

namespace {

// ---------------------------------------------------------------------------
// IR construction for each workload kind.
// ---------------------------------------------------------------------------
std::shared_ptr<Module> build_matmul_ir(const CompileTask& task) {
    auto m = std::make_shared<Module>();
    auto M = static_cast<i64>(task.M);
    auto K = static_cast<i64>(task.K);
    auto N = static_cast<i64>(task.N);

    std::vector<TypePtr> operand_types = {
        make_tensor_type({M, K}, task.dtype),
        make_tensor_type({K, N}, task.dtype),
    };
    bool has_bias = (task.kind == "matmul_bias" ||
                     task.kind == "matmul_bias_relu" ||
                     task.kind == "matmul_bias_gelu");
    if (has_bias) {
        operand_types.push_back(make_tensor_type({1, N}, task.dtype));
    }
    std::vector<TypePtr> result_types = {
        make_tensor_type({M, N}, task.dtype),
    };

    auto f = m->create_function("kernel", operand_types, result_types);
    Builder b(f);
    auto A = f->args()[0];
    auto B = f->args()[1];
    auto mm = b.matmul(A, B);

    if (!has_bias) {
        b.output_tensor(mm);
        return m;
    }

    auto bias = f->args()[2];
    auto bd = b.add(mm, bias);
    if (task.kind == "matmul_bias") {
        b.output_tensor(bd);
    } else if (task.kind == "matmul_bias_relu") {
        b.output_tensor(b.relu(bd));
    } else if (task.kind == "matmul_bias_gelu") {
        b.output_tensor(b.gelu(bd));
    }
    return m;
}

std::shared_ptr<Module> build_elementwise_chain_ir(const CompileTask& task) {
    auto m = std::make_shared<Module>();
    auto width = static_cast<i64>(task.M);
    auto f = m->create_function(
        "chain",
        {make_tensor_type({width}, task.dtype)},
        {make_tensor_type({width}, task.dtype)});
    Builder b(f);
    auto x = f->args()[0];
    Value cur = x;
    for (u32 i = 0; i < task.chain_depth; ++i) {
        switch (i % 4) {
            case 0: cur = b.relu(cur); break;
            case 1: cur = b.gelu(cur); break;
            case 2: cur = b.sigmoid(cur); break;
            case 3: cur = b.tanh(cur); break;
        }
    }
    b.output_tensor(cur);
    return m;
}

std::shared_ptr<Module> build_reduction_ir(const CompileTask& task) {
    auto m = std::make_shared<Module>();
    auto M = static_cast<i64>(task.M);
    auto K = static_cast<i64>(task.K);
    auto f = m->create_function(
        "reduction",
        {make_tensor_type({M, K}, task.dtype)},
        {make_tensor_type({M}, task.dtype)});
    Builder b(f);
    auto x = f->args()[0];
    auto r = b.reduce_sum(x, {1}, false);
    b.output_tensor(r);
    return m;
}

std::shared_ptr<Module> build_ir(const CompileTask& task) {
    if (task.kind == "matmul" || task.kind == "matmul_bias" ||
        task.kind == "matmul_bias_relu" || task.kind == "matmul_bias_gelu") {
        return build_matmul_ir(task);
    }
    if (task.kind == "elementwise_chain") {
        return build_elementwise_chain_ir(task);
    }
    if (task.kind == "reduction") {
        return build_reduction_ir(task);
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Hardware + numerical mode selection.
// ---------------------------------------------------------------------------
HardwareModel select_hardware(const std::string& name) {
    if (name == "cpu")     return HardwareModel::generic_cpu();
    if (name == "a100")    return HardwareModel::generic_nvidia_gpu();
    if (name == "mi300x")  return HardwareModel::generic_amd_gpu();
    return HardwareModel::generic_nvidia_gpu();  // default
}

NumericalMode select_mode(const std::string& name) {
    if (name == "strict")     return NumericalMode::Strict;
    if (name == "relaxed")    return NumericalMode::Relaxed;
    if (name == "fast_math")  return NumericalMode::FastMath;
    return NumericalMode::FastMath;
}

// ---------------------------------------------------------------------------
// Stats collection helpers.
// ---------------------------------------------------------------------------
u32 count_ops(const Module& m) {
    return static_cast<u32>(m.num_operations());
}

std::string module_to_string(const Module& m) {
    std::ostringstream os;
    for (auto& f : m.functions()) {
        os << to_string(*f) << "\n";
    }
    return os.str();
}

} // namespace

// ---------------------------------------------------------------------------
// The main compile entry point.
// ---------------------------------------------------------------------------
CompileResult compile(const CompileTask& task, bool print_ir) {
    CompileResult result;

    // 1. Build the IR.
    auto module = build_ir(task);
    if (!module) {
        result.error = "Unknown workload kind: " + task.kind;
        return result;
    }

    result.ops_before = count_ops(*module);
    result.ir_text_before = module_to_string(*module);

    if (print_ir) {
        std::cout << "=== IR BEFORE OPTIMIZATION ===\n";
        std::cout << result.ir_text_before << "\n";
    }

    // 2. Set up the analysis manager + driver.
    AnalysisManager am(*module);
    IterativeDriverOptions opts;
    opts.max_iterations = 3;
    opts.run_global_barrier = true;
    IterativeDriver driver(am, opts);
    driver.set_hardware(select_hardware(task.hardware));

    // 3. Run the full optimization pipeline.
    auto report = driver.run(*module);
    result.converged = report.converged;
    result.iterations = report.iterations_run;
    result.ops_after = count_ops(*module);
    result.ir_text = module_to_string(*module);

    if (print_ir) {
        std::cout << "=== IR AFTER OPTIMIZATION ===\n";
        std::cout << result.ir_text << "\n";
    }

    // 4. Run the unified analyzer on the optimized IR to get final facts.
    UnifiedAnalyzer analyzer(*module);
    analyzer.set_hardware(select_hardware(task.hardware));
    analyzer.set_numerical_mode(select_mode(task.numerical_mode));
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    auto& g = store.graph_facts();
    result.facts_discovered = g.facts_discovered;
    result.analyzer_latency_sec = g.analysis_latency_sec;
    result.analyzer_runs = 1;

    if (g.graph_arithmetic_intensity.known)
        result.graph_intensity = g.graph_arithmetic_intensity.value;
    if (g.effective_arithmetic_intensity.known)
        result.effective_intensity = g.effective_arithmetic_intensity.value;
    if (g.roofline_ridge.known)
        result.roofline_ridge = g.roofline_ridge.value;

    // 5. Cost model: estimate runtime + SM utilization for matmul.
    if (task.kind.find("matmul") != std::string::npos) {
        AnalyticalCostModelV2 cost_model(select_hardware(task.hardware));
        // Use a default schedule (no tiling) for the estimate.
        Schedule s;
        auto features = cost_model.extract_features(
            s, task.M, task.N, task.K, task.dtype);
        auto bd = cost_model.estimate(features);
        result.predicted_runtime_sec = bd.total_sec;
        result.sm_utilization_pct = bd.sm_utilization_pct;
        result.num_blocks = bd.num_blocks;
        result.num_waves = bd.num_waves;
    }

    // 6. Collect per-pass stats by re-running the unified pipeline with
    //    stats tracking. (This is a second pass over the optimized IR,
    //    but it gives us the per-pass breakdown the user wants.)
    UnifiedPassPipeline pipeline;
    pipeline.run(*module, am);
    const auto& ps = pipeline.stats();
    result.analyzer_runs += ps.analyzer_runs;
    result.analyzer_latency_sec += ps.analyzer_latency_sec;

    // Record per-pass stats as string-keyed maps for easy Python consumption.
    auto add_pass = [&](const std::string& name,
                        std::unordered_map<std::string, std::string> stats) {
        result.pass_stats.push_back({name, std::move(stats)});
    };
    add_pass("const_fold", {
        {"constants_folded", std::to_string(ps.const_fold.constants_folded)},
        {"zero_propagations", std::to_string(ps.const_fold.zero_propagations)},
    });
    add_pass("canonicalize", {
        {"add_zero_simplified", std::to_string(ps.canonicalize.add_zero_simplified)},
        {"mul_one_simplified", std::to_string(ps.canonicalize.mul_one_simplified)},
        {"mul_zero_simplified", std::to_string(ps.canonicalize.mul_zero_simplified)},
        {"transpose_pair_eliminated", std::to_string(ps.canonicalize.transpose_pair_eliminated)},
        {"commutative_reordered", std::to_string(ps.canonicalize.commutative_reordered)},
        {"total_rewrites", std::to_string(ps.canonicalize.total_rewrites)},
    });
    add_pass("cse", {
        {"duplicates_removed", std::to_string(ps.cse.duplicates_removed)},
    });
    add_pass("copy_elim", {
        {"transposes_eliminated", std::to_string(ps.copy_elim.transposes_eliminated)},
        {"reshapes_eliminated", std::to_string(ps.copy_elim.reshapes_eliminated)},
        {"broadcasts_eliminated", std::to_string(ps.copy_elim.broadcasts_eliminated)},
    });
    add_pass("recompute", {
        {"materialize_decisions", std::to_string(ps.recompute.materialize_decisions)},
        {"recompute_decisions", std::to_string(ps.recompute.recompute_decisions)},
        {"fuse_decisions", std::to_string(ps.recompute.fuse_decisions)},
    });
    add_pass("dce", {
        {"ops_removed", std::to_string(ps.dce.ops_removed)},
    });

    return result;
}

} // namespace cg
