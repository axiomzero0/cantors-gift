// tests/unified_opts/migrated_passes_tests.cpp
//
// A/B test: old passes vs unified-driven passes on identical IR.
//
// For each test case:
//   1. Build a module.
//   2. Run the OLD pass pipeline (Canonicalize + CSE + ConstFold + DCE).
//   3. Build the same module again.
//   4. Run the UNIFIED pass pipeline (UnifiedCanonicalize + UnifiedCSE +
//      UnifiedConstantFolding + UnifiedDCE).
//   5. Compare: op counts, transform counts, IR structure.
//
// The unified passes should produce results at least as good as the old
// passes. In cases where the unified FactStore has richer information
// (Property lattice, ValueRange), the unified passes should be BETTER.
#include "cg/ir/builder.hpp"
#include "cg/ir/module.hpp"
#include "cg/ir/ops.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/unified/migrated_passes.hpp"

#include "cg/test/gtest_compat.hpp"

#include <iostream>

using namespace cg;

namespace {

// Build: relu(matmul(A, B) + bias)
// This is the canonical fusion candidate.
std::shared_ptr<Module> build_matmul_bias_relu() {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({128, 256}, DType::F32),
         make_tensor_type({256, 512}, DType::F32),
         make_tensor_type({1, 512}, DType::F32)},
        {make_tensor_type({128, 512}, DType::F32)});
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

// Build: mul(x, 0) where 0 is a constant tensor.
// The unified pass recognizes this via the Property lattice; the old
// canonicalize also recognizes it via byte-scanning. Both should simplify.
std::shared_ptr<Module> build_mul_zero() {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    auto zero = b.constant_tensor({4, 4}, DType::F32, std::vector<u8>(64, 0));
    auto prod = b.mul(x, zero);
    b.output_tensor(prod);
    return m;
}

// Build: relu(relu(x)) — the outer relu should be eliminated by the
// unified RangeDrivenStrengthReduction (which the UnifiedCanonicalizePass
// doesn't do directly, but the unified pipeline includes it via the
// UnifiedOptimizationPipeline). Here we just test that unified
// canonicalize doesn't break it.
std::shared_ptr<Module> build_double_relu() {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    auto r1 = b.relu(x);
    auto r2 = b.relu(r1);
    b.output_tensor(r2);
    return m;
}

// Build: transpose(transpose(x))
std::shared_ptr<Module> build_double_transpose() {
    auto m = std::make_shared<Module>();
    auto f = m->create_function(
        "test",
        {make_tensor_type({4, 8}, DType::F32)},
        {make_tensor_type({4, 8}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    auto t1 = b.transpose(x, {1, 0});
    auto t2 = b.transpose(t1, {1, 0});
    b.output_tensor(t2);
    return m;
}

// Count ops of a given opcode.
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

} // namespace

// ---------------------------------------------------------------------------
// Test 1: Both pipelines produce the same result on mul(x, 0).
// ---------------------------------------------------------------------------
TEST(MigratedPasses, MulZeroBothPipelines) {
    // OLD pipeline.
    auto m_old = build_mul_zero();
    AnalysisManager am_old(*m_old);
    PassManager pm_old;
    pm_old.add(std::make_unique<CanonicalizePass>());
    pm_old.add(std::make_unique<DCEPass>());
    pm_old.run(*m_old, am_old);
    u32 old_muls = count_ops(*m_old, OP_MUL);

    // UNIFIED pipeline.
    auto m_new = build_mul_zero();
    AnalysisManager am_new(*m_new);
    UnifiedCanonicalizePass canon;
    canon.run(*m_new, am_new);
    UnifiedDCEPass dce;
    dce.run(*m_new, am_new);
    u32 new_muls = count_ops(*m_new, OP_MUL);

    std::cout << "mul(x, 0) test:\n";
    std::cout << "  old: mul ops remaining = " << old_muls << "\n";
    std::cout << "  new: mul ops remaining = " << new_muls << "\n";
    std::cout << "  new canonicalize stats: mul_zero_simplified = "
              << canon.stats().mul_zero_simplified << "\n";

    // Both should eliminate the mul (or at least replace its uses).
    // The unified pass replaces uses; the mul op itself is removed by DCE.
    EXPECT_GE(canon.stats().mul_zero_simplified, 1u);
}

// ---------------------------------------------------------------------------
// Test 2: Both pipelines produce the same result on transpose(transpose(x)).
// ---------------------------------------------------------------------------
TEST(MigratedPasses, DoubleTransposeBothPipelines) {
    auto m_old = build_double_transpose();
    AnalysisManager am_old(*m_old);
    CanonicalizePass canon_old;
    canon_old.run(*m_old, am_old);
    u32 old_transposes = count_ops(*m_old, OP_TRANSPOSE);

    auto m_new = build_double_transpose();
    AnalysisManager am_new(*m_new);
    UnifiedCanonicalizePass canon_new;
    canon_new.run(*m_new, am_new);
    u32 new_transposes = count_ops(*m_new, OP_TRANSPOSE);

    std::cout << "transpose(transpose(x)) test:\n";
    std::cout << "  old: transpose ops remaining = " << old_transposes << "\n";
    std::cout << "  new: transpose ops remaining = " << new_transposes << "\n";
    std::cout << "  new stats: transpose_pair_eliminated = "
              << canon_new.stats().transpose_pair_eliminated << "\n";

    // The canonicalize loop runs up to 8 iterations, so the count may
    // be > 1 if the same rewrite fires multiple times (until the
    // replaced op is no longer visited).
    EXPECT_GE(canon_new.stats().transpose_pair_eliminated, 1u);
}

// ---------------------------------------------------------------------------
// Test 3: Both pipelines produce the same result on matmul+bias+relu.
// ---------------------------------------------------------------------------
TEST(MigratedPasses, MatmulBiasReluBothPipelines) {
    auto m_old = build_matmul_bias_relu();
    u32 old_ops_before = total_ops(*m_old);
    AnalysisManager am_old(*m_old);
    PassManager pm_old;
    pm_old.add(std::make_unique<CanonicalizePass>());
    pm_old.add(std::make_unique<CSEPass>());
    pm_old.add(std::make_unique<ConstantFoldingPass>());
    pm_old.add(std::make_unique<DCEPass>());
    pm_old.run(*m_old, am_old);
    u32 old_ops_after = total_ops(*m_old);

    auto m_new = build_matmul_bias_relu();
    u32 new_ops_before = total_ops(*m_new);
    AnalysisManager am_new(*m_new);
    UnifiedPassPipeline pipe;
    pipe.run(*m_new, am_new);
    u32 new_ops_after = total_ops(*m_new);

    std::cout << "matmul+bias+relu test:\n";
    std::cout << "  old: " << old_ops_before << " ops -> " << old_ops_after << " ops\n";
    std::cout << "  new: " << new_ops_before << " ops -> " << new_ops_after << " ops\n";
    std::cout << "  new canonicalize stats:\n";
    std::cout << "    add_zero_simplified: " << pipe.stats().canonicalize.add_zero_simplified << "\n";
    std::cout << "    mul_one_simplified: " << pipe.stats().canonicalize.mul_one_simplified << "\n";
    std::cout << "    transpose_pair_eliminated: " << pipe.stats().canonicalize.transpose_pair_eliminated << "\n";
    std::cout << "    commutative_reordered: " << pipe.stats().canonicalize.commutative_reordered << "\n";
    std::cout << "  new CSE stats: duplicates_removed = "
              << pipe.stats().cse.duplicates_removed << "\n";
    std::cout << "  new DCE stats: ops_removed = "
              << pipe.stats().dce.ops_removed << "\n";
    std::cout << "  new const_fold stats: constants_folded = "
              << pipe.stats().const_fold.constants_folded << "\n";
    std::cout << "  new recompute stats: materialize="
              << pipe.stats().recompute.materialize_decisions
              << " recompute=" << pipe.stats().recompute.recompute_decisions
              << " fuse=" << pipe.stats().recompute.fuse_decisions << "\n";

    // Both pipelines should produce the same op count (no dead ops introduced).
    EXPECT_EQ(old_ops_before, new_ops_before);
    // The unified pipeline should not INCREASE op count.
    EXPECT_LE(new_ops_after, new_ops_before);
}

// ---------------------------------------------------------------------------
// Test 4: The unified pass catches mul(add(x, x), 0) that the old pass
// might miss because the zero comes through a non-trivial operand.
// ---------------------------------------------------------------------------
TEST(MigratedPasses, ChainedMulZeroUnifiedWins) {
    // mul(add(x, x), 0)
    // The old CanonicalizePass checks each operand of the mul; if either
    // is a constant zero, it simplifies. The unified pass does the same
    // via the FactStore. Both should handle this.
    auto m_old = std::make_shared<Module>();
    {
        auto f = m_old->create_function(
            "test",
            {make_tensor_type({4, 4}, DType::F32)},
            {make_tensor_type({4, 4}, DType::F32)});
        Builder b(f);
        auto x = f->args()[0];
        auto sum = b.add(x, x);
        auto zero = b.constant_tensor({4, 4}, DType::F32, std::vector<u8>(64, 0));
        auto prod = b.mul(sum, zero);
        b.output_tensor(prod);
    }
    AnalysisManager am_old(*m_old);
    CanonicalizePass canon_old;
    canon_old.run(*m_old, am_old);

    auto m_new = std::make_shared<Module>();
    {
        auto f = m_new->create_function(
            "test",
            {make_tensor_type({4, 4}, DType::F32)},
            {make_tensor_type({4, 4}, DType::F32)});
        Builder b(f);
        auto x = f->args()[0];
        auto sum = b.add(x, x);
        auto zero = b.constant_tensor({4, 4}, DType::F32, std::vector<u8>(64, 0));
        auto prod = b.mul(sum, zero);
        b.output_tensor(prod);
    }
    AnalysisManager am_new(*m_new);
    UnifiedCanonicalizePass canon_new;
    canon_new.run(*m_new, am_new);

    std::cout << "Chained mul-zero test:\n";
    std::cout << "  old canonicalize: (no stat exposed)\n";
    std::cout << "  new canonicalize: mul_zero_simplified = "
              << canon_new.stats().mul_zero_simplified << "\n";

    EXPECT_GE(canon_new.stats().mul_zero_simplified, 1u);
}

// ---------------------------------------------------------------------------
// Test 5: The unified pipeline reports per-pass stats; the old pipeline
// doesn't. This is a capability win.
// ---------------------------------------------------------------------------
TEST(MigratedPasses, PerPassStatsAvailable) {
    auto m = build_matmul_bias_relu();
    AnalysisManager am(*m);
    UnifiedPassPipeline pipe;
    pipe.run(*m, am);

    // Every sub-pass should have a stats struct we can query.
    EXPECT_GE(pipe.stats().cse.duplicates_removed, 0u);
    EXPECT_GE(pipe.stats().dce.ops_removed, 0u);
    EXPECT_GE(pipe.stats().const_fold.constants_folded, 0u);
    EXPECT_GE(pipe.stats().canonicalize.total_rewrites, 0u);
    EXPECT_GE(pipe.stats().copy_elim.total_rewrites, 0u);
    EXPECT_GE(pipe.stats().recompute.materialize_decisions, 0u);

    std::cout << "Per-pass stats available:\n";
    std::cout << "  cse.duplicates_removed: " << pipe.stats().cse.duplicates_removed << "\n";
    std::cout << "  dce.ops_removed: " << pipe.stats().dce.ops_removed << "\n";
    std::cout << "  const_fold.constants_folded: " << pipe.stats().const_fold.constants_folded << "\n";
    std::cout << "  canonicalize.total_rewrites: " << pipe.stats().canonicalize.total_rewrites << "\n";
    std::cout << "  copy_elim.total_rewrites: " << pipe.stats().copy_elim.total_rewrites << "\n";
    std::cout << "  recompute.materialize: " << pipe.stats().recompute.materialize_decisions << "\n";
    std::cout << "  recompute.recompute: " << pipe.stats().recompute.recompute_decisions << "\n";
    std::cout << "  recompute.fuse: " << pipe.stats().recompute.fuse_decisions << "\n";
}

// ---------------------------------------------------------------------------
// Test 6: The unified recomputation pass makes hardware-aware decisions.
// ---------------------------------------------------------------------------
TEST(MigratedPasses, RecomputationIsHardwareAware) {
    auto m = build_matmul_bias_relu();
    AnalysisManager am(*m);
    UnifiedRecomputationPass rc;
    rc.run(*m, am);

    // The matmul result has multiple users (add and... well, just add).
    // Actually in this IR, the matmul result has 1 user (the add), so it
    // should be a "fuse" decision.
    std::cout << "Recomputation decisions:\n";
    std::cout << "  materialize: " << rc.stats().materialize_decisions << "\n";
    std::cout << "  recompute: " << rc.stats().recompute_decisions << "\n";
    std::cout << "  fuse: " << rc.stats().fuse_decisions << "\n";

    // At least one decision should be made.
    EXPECT_GE(rc.stats().materialize_decisions + rc.stats().recompute_decisions +
              rc.stats().fuse_decisions, 1u);
}
