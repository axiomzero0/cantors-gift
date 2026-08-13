// benchmarks/benchmark_old_vs_unified.cpp - C++ A/B benchmark comparing
// old optimization passes against unified-driven passes on identical IR.
//
// Builds the same module twice, runs each pipeline, and compares:
//   - ops eliminated
//   - final op count (lower = better)
//   - latency
//
// The unified passes should produce results AT LEAST AS GOOD as the old
// passes, with the bonus of per-pass stats + provenance + confidence.
#include "cg/ir/builder.hpp"
#include "cg/ir/module.hpp"
#include "cg/ir/ops.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/unified/migrated_passes.hpp"

#include <chrono>
#include <iostream>
#include <memory>

using namespace cg;

namespace {

u32 count_ops(const Module& m, Opcode opcode) {
    u32 n = 0;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.opcode == opcode) ++n;
        }
    }
    return n;
}

u32 total_ops(const Module& m) {
    return static_cast<u32>(m.num_operations());
}

// Build: x -> mul(.,0) -> mul(.,0) -> ... (depth zeros)
std::shared_ptr<Module> build_mul_zero_chain(int depth) {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({16, 16}, DType::F32)},
        {make_tensor_type({16, 16}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    Value cur = x;
    std::vector<u8> zero_bytes(16 * 16 * 4, 0);
    for (int i = 0; i < depth; ++i) {
        auto zero = b.constant_tensor({16, 16}, DType::F32, zero_bytes);
        cur = b.mul(cur, zero);
    }
    b.output_tensor(cur);
    return m;
}

// Build: transpose(transpose(...transpose(x)...)) (depth transposes)
std::shared_ptr<Module> build_transpose_chain(int depth) {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({16, 16}, DType::F32)},
        {make_tensor_type({16, 16}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    Value cur = x;
    for (int i = 0; i < depth; ++i) {
        cur = b.transpose(cur, {1, 0});
    }
    b.output_tensor(cur);
    return m;
}

// Build: relu(matmul(A, B) + bias)
std::shared_ptr<Module> build_matmul_bias_relu(int M, int K, int N) {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({M, K}, DType::F32),
         make_tensor_type({K, N}, DType::F32),
         make_tensor_type({1, N}, DType::F32)},
        {make_tensor_type({M, N}, DType::F32)});
    Builder b(f);
    auto A = f->args()[0];
    auto B = f->args()[1];
    auto bias = f->args()[2];
    auto mm = b.matmul(A, B);
    auto bd = b.add(mm, bias);
    auto r = b.relu(bd);
    b.output_tensor(r);
    return m;
}

struct OldResult {
    u32 ops_before;
    u32 ops_after;
    double latency_us;
};

OldResult run_old_pipeline(std::shared_ptr<Module> m) {
    OldResult r;
    r.ops_before = total_ops(*m);
    AnalysisManager am(*m);
    PassManager pm;
    pm.add(std::make_unique<CanonicalizePass>());
    pm.add(std::make_unique<CSEPass>());
    pm.add(std::make_unique<ConstantFoldingPass>());
    pm.add(std::make_unique<DCEPass>());
    auto t0 = std::chrono::steady_clock::now();
    pm.run(*m, am);
    auto t1 = std::chrono::steady_clock::now();
    r.latency_us = std::chrono::duration<double, std::milli>(t1 - t0).count() * 1000.0;
    r.ops_after = total_ops(*m);
    return r;
}

struct NewResult {
    u32 ops_before;
    u32 ops_after;
    double latency_us;
    UnifiedPassPipeline::Stats stats;
};

NewResult run_unified_pipeline(std::shared_ptr<Module> m) {
    NewResult r;
    r.ops_before = total_ops(*m);
    AnalysisManager am(*m);
    UnifiedPassPipeline pipe;
    auto t0 = std::chrono::steady_clock::now();
    pipe.run(*m, am);
    auto t1 = std::chrono::steady_clock::now();
    r.latency_us = std::chrono::duration<double, std::milli>(t1 - t0).count() * 1000.0;
    r.ops_after = total_ops(*m);
    r.stats = pipe.stats();
    return r;
}

void run_comparison(const std::string& label,
                    std::shared_ptr<Module> (*builder)(),
                    bool expect_reduction) {
    std::cout << "\n--- " << label << " ---\n";

    auto m_old = builder();
    auto m_new = builder();

    std::cout << "  before: " << total_ops(*m_old) << " ops\n";

    auto old_r = run_old_pipeline(m_old);
    auto new_r = run_unified_pipeline(m_new);

    std::cout << "  OLD: " << old_r.ops_before << " -> " << old_r.ops_after
              << " ops  (" << old_r.latency_us << " us)\n";
    std::cout << "  NEW: " << new_r.ops_before << " -> " << new_r.ops_after
              << " ops  (" << new_r.latency_us << " us)\n";
    std::cout << "  NEW stats: cse=" << new_r.stats.cse.duplicates_removed
              << " dce=" << new_r.stats.dce.ops_removed
              << " const_fold=" << new_r.stats.const_fold.constants_folded
              << " canon_rewrites=" << new_r.stats.canonicalize.total_rewrites
              << " copy_elim=" << new_r.stats.copy_elim.total_rewrites
              << " recompute_fuse=" << new_r.stats.recompute.fuse_decisions
              << "\n";

    if (expect_reduction) {
        // The unified pass should reduce op count by at least as much.
        u32 old_reduction = old_r.ops_before - old_r.ops_after;
        u32 new_reduction = new_r.ops_before - new_r.ops_after;
        std::cout << "  reduction: old=" << old_reduction
                  << " new=" << new_reduction << "\n";
    }
}

} // namespace

int main() {
    std::cout << "VORTEX: old passes vs unified-driven passes (A/B benchmark)\n";
    std::cout << "=" << std::string(90, '=') << "\n";

    run_comparison("mul-zero chain (depth 5)",
                   []() { return build_mul_zero_chain(5); }, true);
    run_comparison("mul-zero chain (depth 10)",
                   []() { return build_mul_zero_chain(10); }, true);
    run_comparison("transpose chain (depth 5)",
                   []() { return build_transpose_chain(5); }, true);
    run_comparison("transpose chain (depth 10)",
                   []() { return build_transpose_chain(10); }, true);
    run_comparison("matmul+bias+relu (128x256x512)",
                   []() { return build_matmul_bias_relu(128, 256, 512); }, false);
    run_comparison("matmul+bias+relu (1024^3)",
                   []() { return build_matmul_bias_relu(1024, 1024, 1024); }, false);

    std::cout << "\n" << "=" << std::string(90, '=') << "\n";
    std::cout << "Done.\n";
    return 0;
}
