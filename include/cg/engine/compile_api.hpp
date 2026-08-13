// engine/compile_api.hpp - the clean user-facing API.
//
// The user gives a TASK (what to compute), the engine handles everything
// (IR construction, analysis, optimization, codegen), and returns a RESULT.
//
//   task = {
//     kind: "matmul_bias_relu",
//     M: 1024, K: 1024, N: 1024,
//     dtype: F32,
//     fuse_bias_relu: true,
//     numerical_mode: "fast_math",
//     hardware: "a100"
//   }
//
//   result = compile(task, print_ir=False)
//   print(result.ir_text)           # the optimized IR
//   print(result.predicted_runtime) # seconds
//   print(result.stats)             # per-pass stats
//
// The Python side never touches IR construction, analysis, or passes
// directly. It just describes the task and reads the result.
#pragma once

#include "cg/cost/hardware_model.hpp"
#include "cg/ir/module.hpp"
#include "cg/numerical/semantics.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cg {

// ---------------------------------------------------------------------------
// Task: what the user wants to compute.
// ---------------------------------------------------------------------------
struct CompileTask {
    // Workload kind. Currently supported:
    //   "matmul"            : C = A @ B
    //   "matmul_bias"       : C = A @ B + bias
    //   "matmul_bias_relu"  : C = relu(A @ B + bias)
    //   "matmul_bias_gelu"  : C = gelu(A @ B + bias)
    //   "elementwise_chain" : C = relu(gelu(sigmoid(tanh(...(x)...)))
    //                          (depth from `chain_depth`)
    //   "reduction"         : C = reduce_sum(x, axis=1)
    std::string kind = "matmul_bias_relu";

    // Shape (used by matmul* kinds).
    u64 M = 1024;
    u64 K = 1024;
    u64 N = 1024;

    // Elementwise chain depth (used by "elementwise_chain").
    u32 chain_depth = 10;

    // Data type.
    DType dtype = DType::F32;

    // Numerical mode: "strict", "relaxed", "fast_math".
    std::string numerical_mode = "fast_math";

    // Hardware target: "cpu", "a100", "mi300x".
    std::string hardware = "a100";

    // Optimization level: 0 = none, 1 = basic, 2 = full (default).
    u32 opt_level = 2;
};

// ---------------------------------------------------------------------------
// Result: what the engine produces.
// ---------------------------------------------------------------------------
struct CompileResult {
    // The optimized IR as text (post-optimization).
    std::string ir_text;

    // The IR text BEFORE optimization (only if print_ir=true).
    std::string ir_text_before;

    // Predicted runtime in seconds (from the cost model).
    double predicted_runtime_sec = 0.0;

    // Number of ops before / after optimization.
    u32 ops_before = 0;
    u32 ops_after = 0;

    // Whether the optimizer converged.
    bool converged = false;
    u32 iterations = 0;

    // Per-pass stats (pass_name -> {key: value}).
    // Keys are strings; values are strings for easy Python dict conversion.
    std::vector<std::pair<std::string,
                          std::unordered_map<std::string, std::string>>> pass_stats;

    // Analyzer stats.
    u32 analyzer_runs = 0;
    double analyzer_latency_sec = 0.0;
    u32 facts_discovered = 0;

    // Roofline summary.
    double graph_intensity = 0.0;
    double effective_intensity = 0.0;
    double roofline_ridge = 0.0;

    // SM utilization (for matmul workloads).
    double sm_utilization_pct = 0.0;
    u32 num_blocks = 0;
    u32 num_waves = 0;

    // Error message (empty on success).
    std::string error;
};

// ---------------------------------------------------------------------------
// The main entry point.
//
//   CompileResult result = compile(task, print_ir=false);
//
// If `print_ir` is true, the IR is printed to stdout before AND after
// optimization, and `ir_text_before` is populated in the result.
//
// The engine handles:
//   1. IR construction from the task description
//   2. Hardware model selection
//   3. Numerical mode selection
//   4. AnalysisManager setup
//   5. IterativeDriver run (full optimization pipeline)
//   6. Cost model evaluation
//   7. Stats collection
//
// Python never touches any of that directly.
// ---------------------------------------------------------------------------
CompileResult compile(const CompileTask& task, bool print_ir = false);

} // namespace cg
