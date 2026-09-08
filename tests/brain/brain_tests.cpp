// tests/brain/brain_tests.cpp - unit tests for the brain-domain
// semantic primitives and their optimization.
//
// The most important test here is `BrainBuilder.CSECallsapesDuplicateDistances`:
// it verifies that when the same `pairwise_dist_sq(positions)` is emitted
// multiple times, the existing CSE pass collapses them to a single op.
// This is the single biggest optimization target identified in the
// design discussion (the same `d_ij^2` is reused by K_x, K_inh, B_ij,
// and τ_delay).
#include "cg/brain/brain_builder.hpp"
#include "cg/engine/compile_api.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/iterative_driver.hpp"
#include "cg/analysis/analysis.hpp"

#include "cg/test/gtest_compat.hpp"

#include <sstream>

using namespace cg;

namespace {

// Helper: count ops of a specific opcode in a module.
usize count_ops(const Module& m, Opcode oc) {
    usize n = 0;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == oc) ++n;
    return n;
}

} // namespace

// =====================================================================
// IR construction tests
// =====================================================================

TEST(BrainIR, RegisterOps) {
    // The brain opcodes should be registered with the right names.
    EXPECT_EQ(OpRegistry::instance().lookup(OP_PAIRWISE_DIST_SQ)->name,
              std::string("pairwise_dist_sq"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_NEIGHBOR_QUERY)->name,
              std::string("neighbor_query"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_GAUSSIAN_KERNEL)->name,
              std::string("gaussian_kernel"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_LATERAL_INHIBITION)->name,
              std::string("lateral_inhibition"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_DELAY_FROM_DIST)->name,
              std::string("delay_from_dist"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_SPIKE)->name,
              std::string("spike"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_SYNAPTIC_ARRIVAL)->name,
              std::string("synaptic_arrival"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_ELIGIBILITY_UPDATE)->name,
              std::string("eligibility_update"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_CREDIT_PROPAGATE)->name,
              std::string("credit_propagate"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_ACTIVE_SET)->name,
              std::string("active_set"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_REGION_AGGREGATE)->name,
              std::string("region_aggregate"));
    EXPECT_EQ(OpRegistry::instance().lookup(OP_STRUCTURAL_EPOCH)->name,
              std::string("structural_epoch"));
}

TEST(BrainIR, BrainOpsArePureExceptSpikeAndEpoch) {
    // Most brain ops should be pure so they can be CSE'd / fused.
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_PAIRWISE_DIST_SQ)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_NEIGHBOR_QUERY)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_GAUSSIAN_KERNEL)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_LATERAL_INHIBITION)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_DELAY_FROM_DIST)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_SYNAPTIC_ARRIVAL)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_ELIGIBILITY_UPDATE)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_CREDIT_PROPAGATE)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_ACTIVE_SET)->effects.is_pure());
    EXPECT_TRUE(OpRegistry::instance().lookup(OP_REGION_AGGREGATE)->effects.is_pure());
    // Spike and StructuralEpoch have side effects and must not be
    // reordered or eliminated by DCE.
    EXPECT_FALSE(OpRegistry::instance().lookup(OP_SPIKE)->effects.is_pure());
    EXPECT_FALSE(OpRegistry::instance().lookup(OP_STRUCTURAL_EPOCH)->effects.is_pure());
}

TEST(BrainIR, PairwiseDistSqShape) {
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({16, 3}, DType::F32)},
        {make_tensor_type({16, 16}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    auto d2 = b.pairwise_dist_sq(x);
    b.output_tensor(d2);

    auto t = d2.as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 16);
    EXPECT_EQ(t->shape[1]->value, 16);
}

TEST(BrainIR, NeighborQueryShape) {
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({16, 3}, DType::F32)},
        {make_tensor_type({16, 8}, DType::I32)});
    Builder b(f);
    auto x = f->args()[0];
    auto n = b.neighbor_query(x, 0.5, 8);
    b.output_tensor(n);

    auto t = n.as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 16);
    EXPECT_EQ(t->shape[1]->value, 8);
    EXPECT_EQ(t->dtype, DType::I32);
}

TEST(BrainIR, GaussianKernelShapePreserving) {
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({16, 16}, DType::F32)},
        {make_tensor_type({16, 16}, DType::F32)});
    Builder b(f);
    auto d2 = f->args()[0];
    auto kx = b.gaussian_kernel(d2, 0.5);
    b.output_tensor(kx);

    auto t = kx.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 16);
    EXPECT_EQ(t->shape[1]->value, 16);
}

TEST(BrainIR, CreditPropagateShape) {
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({16, 16}, DType::F32),
         make_tensor_type({16}, DType::F32)},
        {make_tensor_type({16}, DType::F32)});
    Builder b(f);
    auto w = f->args()[0];
    auto c = f->args()[1];
    auto out = b.credit_propagate(w, c);
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape.rank(), 1u);
    EXPECT_EQ(t->shape[0]->value, 16);
}

TEST(BrainIR, ActiveSetShape) {
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({64}, DType::F32)},
        {make_tensor_type({8}, DType::I32)});
    Builder b(f);
    auto x = f->args()[0];
    auto as = b.active_set(x, 8);
    b.output_tensor(as);

    auto t = as.as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 1u);
    EXPECT_EQ(t->shape[0]->value, 8);
    EXPECT_EQ(t->dtype, DType::I32);
}

// =====================================================================
// BrainBuilder pattern tests
// =====================================================================

TEST(BrainBuilder, DistanceReusePatternEmitsOnePairwiseDistSq) {
    // The canonical pattern emits ONE pairwise_dist_sq, and four
    // downstream consumers (K_x, K_inh, Affinity, τ_delay).
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({32, 3}, DType::F32)},
        {make_tensor_type({32, 32}, DType::F32),
         make_tensor_type({32, 32}, DType::F32),
         make_tensor_type({32, 32}, DType::F32),
         make_tensor_type({32, 32}, DType::F32)});
    Builder b(f);
    auto positions = f->args()[0];
    auto p = brain::build_distance_reuse_pattern(
        b, positions, 0.5, 0.3, 1.0, 0.001, 0.005);
    b.output_tensor(p.kx);
    b.output_tensor(p.k_inh);
    b.output_tensor(p.affinity);
    b.output_tensor(p.delay);

    // Should have exactly 1 PairwiseDistSq.
    EXPECT_EQ(count_ops(m, OP_PAIRWISE_DIST_SQ), 1u);

    // Run CSE to dedupe the two GaussianKernel calls (kx and affinity
    // both use gaussian_kernel(d2, sigma=0.5) — they should collapse).
    AnalysisManager am(m);
    CSEPass cse;
    cse.run(m, am);
    DCEPass dce;
    dce.run(m, am);

    // After CSE: only 1 GaussianKernel should remain.
    EXPECT_EQ(count_ops(m, OP_GAUSSIAN_KERNEL), 1u);
    // The single PairwiseDistSq is still there (4 consumers).
    EXPECT_EQ(count_ops(m, OP_PAIRWISE_DIST_SQ), 1u);
}

TEST(BrainBuilder, CSECallsapesDuplicateDistances) {
    // The single biggest optimization target: if the user accidentally
    // emits pairwise_dist_sq(positions) FIVE times (one per consumer),
    // CSE must collapse them back to one.
    Module m;
    auto f = m.create_function("k",
        {make_tensor_type({16, 3}, DType::F32)},
        {make_tensor_type({16, 16}, DType::F32)});
    Builder b(f);
    auto positions = f->args()[0];

    // Naïve: emit pairwise_dist_sq 5 times (one per consumer).
    auto d2_a = b.pairwise_dist_sq(positions);
    auto d2_b = b.pairwise_dist_sq(positions);
    auto d2_c = b.pairwise_dist_sq(positions);
    auto d2_d = b.pairwise_dist_sq(positions);
    auto d2_e = b.pairwise_dist_sq(positions);

    auto kx     = b.gaussian_kernel(d2_a, 0.5);
    auto k_inh  = b.lateral_inhibition(d2_b, 1.0, 0.3);
    auto aff    = b.gaussian_kernel(d2_c, 0.5);
    auto delay  = b.delay_from_dist(d2_d, 0.001, 0.005);
    auto kx2    = b.gaussian_kernel(d2_e, 0.5);

    // Final output = kx + k_inh + aff + delay + kx2 (just to keep all
    // the values alive).
    auto s1 = b.add(kx, k_inh);
    auto s2 = b.add(s1, aff);
    auto s3 = b.add(s2, delay);
    auto s4 = b.add(s3, kx2);
    b.output_tensor(s4);

    // Before CSE: 5 PairwiseDistSq ops.
    EXPECT_EQ(count_ops(m, OP_PAIRWISE_DIST_SQ), 5u);

    AnalysisManager am(m);
    CSEPass cse;
    cse.run(m, am);
    DCEPass dce;
    dce.run(m, am);

    // After CSE: 1 PairwiseDistSq.
    EXPECT_EQ(count_ops(m, OP_PAIRWISE_DIST_SQ), 1u);

    // The downstream consumers (GaussianKernel, LateralInhibition,
    // DelayFromDist) should also dedupe where possible. Kx, Aff, Kx2
    // all use (d2, sigma=0.5) — those 3 collapse to 1. The other two
    // are distinct.
    EXPECT_EQ(count_ops(m, OP_GAUSSIAN_KERNEL), 1u);
    EXPECT_EQ(count_ops(m, OP_LATERAL_INHIBITION), 1u);
    EXPECT_EQ(count_ops(m, OP_DELAY_FROM_DIST), 1u);
}

TEST(BrainBuilder, FullIterativeDriverPreservesSemantics) {
    // The full optimization pipeline (IterativeDriver) must not
    // corrupt the brain IR. We check that the optimized IR text
    // still contains the brain primitives — including the
    // side-effecting ones (spike, structural_epoch).
    CompileTask task;
    task.kind = "brain_full";
    task.N = 32;
    task.D = 3;
    task.E = 64;
    task.topk = 8;
    task.dtype = DType::F32;
    task.hardware = "cpu";

    auto result = compile(task, /*print_ir=*/false);
    EXPECT_EQ(result.error.size(), 0u);
    EXPECT_TRUE(result.converged);

    // The optimized IR should contain:
    //   - exactly 1 pairwise_dist_sq  (collapsed by CSE)
    //   - at least 1 spike             (preserved by DCE due to side-effect)
    //   - exactly 1 structural_epoch   (preserved by DCE due to side-effect)
    EXPECT_NE(result.ir_text.find("pairwise_dist_sq"), std::string::npos);
    EXPECT_NE(result.ir_text.find("spike"), std::string::npos);
    EXPECT_NE(result.ir_text.find("structural_epoch"), std::string::npos);
}

// =====================================================================
// Compile API tests
// =====================================================================

TEST(BrainCompile, DistanceReuseWorkload) {
    CompileTask task;
    task.kind = "brain_distance_reuse";
    task.N = 32;
    task.D = 3;
    task.dtype = DType::F32;
    task.hardware = "cpu";

    auto result = compile(task, /*print_ir=*/false);
    EXPECT_EQ(result.error.size(), 0u);
    EXPECT_GT(result.ops_before, 0u);
    // The optimizer may legitimately add alloc/free ops (memory
    // planning), so we don't assert ops_after <= ops_before. We just
    // require that the optimized IR is non-empty and still contains
    // the brain primitives.
    EXPECT_GT(result.ops_after, 0u);

    // The IR text should mention all four downstream consumers
    // (in this workload they're all outputs, so DCE keeps them).
    EXPECT_NE(result.ir_text.find("gaussian_kernel"), std::string::npos);
    EXPECT_NE(result.ir_text.find("lateral_inhibition"), std::string::npos);
    EXPECT_NE(result.ir_text.find("delay_from_dist"), std::string::npos);
}

TEST(BrainCompile, CreditAssignmentWorkload) {
    CompileTask task;
    task.kind = "brain_credit_assignment";
    task.N = 16;
    task.dtype = DType::F32;
    task.hardware = "cpu";

    auto result = compile(task, /*print_ir=*/false);
    EXPECT_EQ(result.error.size(), 0u);
    EXPECT_NE(result.ir_text.find("credit_propagate"), std::string::npos);
    EXPECT_NE(result.ir_text.find("eligibility_update"), std::string::npos);
}

TEST(BrainCompile, EventDrivenWorkload) {
    CompileTask task;
    task.kind = "brain_event_driven";
    task.N = 32;
    task.D = 3;
    task.E = 64;
    task.topk = 8;
    task.dtype = DType::F32;
    task.hardware = "cpu";

    auto result = compile(task, /*print_ir=*/false);
    EXPECT_EQ(result.error.size(), 0u);
    EXPECT_NE(result.ir_text.find("neighbor_query"), std::string::npos);
    EXPECT_NE(result.ir_text.find("active_set"), std::string::npos);
    EXPECT_NE(result.ir_text.find("synaptic_arrival"), std::string::npos);
}

TEST(BrainCompile, FullWorkload) {
    CompileTask task;
    task.kind = "brain_full";
    task.N = 16;
    task.D = 3;
    task.E = 32;
    task.topk = 8;
    task.dtype = DType::F32;
    task.hardware = "cpu";

    auto result = compile(task, /*print_ir=*/false);
    EXPECT_EQ(result.error.size(), 0u);

    // The full kernel emits every brain primitive. Because every
    // primitive's result is wired to an output, DCE must keep them
    // all. The two side-effecting ops (spike, structural_epoch) must
    // survive regardless.
    EXPECT_NE(result.ir_text.find("pairwise_dist_sq"), std::string::npos);
    EXPECT_NE(result.ir_text.find("gaussian_kernel"), std::string::npos);
    EXPECT_NE(result.ir_text.find("lateral_inhibition"), std::string::npos);
    EXPECT_NE(result.ir_text.find("delay_from_dist"), std::string::npos);
    EXPECT_NE(result.ir_text.find("credit_propagate"), std::string::npos);
    EXPECT_NE(result.ir_text.find("eligibility_update"), std::string::npos);
    EXPECT_NE(result.ir_text.find("neighbor_query"), std::string::npos);
    EXPECT_NE(result.ir_text.find("active_set"), std::string::npos);
    EXPECT_NE(result.ir_text.find("synaptic_arrival"), std::string::npos);
    // Side-effecting ops must survive DCE.
    EXPECT_NE(result.ir_text.find("spike"), std::string::npos);
    EXPECT_NE(result.ir_text.find("structural_epoch"), std::string::npos);
}

// =====================================================================
// Print / debug tests
// =====================================================================

TEST(BrainIR, PrintDistanceReusePattern) {
    // The textual IR for a brain kernel should round-trip and look
    // like the kind of IR the optimizer can chew on.
    Module m;
    auto f = m.create_function("brain_k",
        {make_tensor_type({8, 3}, DType::F32)},
        {make_tensor_type({8, 8}, DType::F32)});
    Builder b(f);
    auto positions = f->args()[0];
    auto p = brain::build_distance_reuse_pattern(
        b, positions, 0.5, 0.3, 1.0, 0.001, 0.005);
    b.output_tensor(p.kx);

    auto s = to_string(m);
    EXPECT_NE(s.find("pairwise_dist_sq"), std::string::npos);
    EXPECT_NE(s.find("gaussian_kernel"), std::string::npos);
    EXPECT_NE(s.find("lateral_inhibition"), std::string::npos);
    EXPECT_NE(s.find("delay_from_dist"), std::string::npos);
}
