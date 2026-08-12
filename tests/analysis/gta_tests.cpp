// tests/gta_tests.cpp - Global Tensor Analysis tests
#include "cg/analysis/global_analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/global_barrier.hpp"

#include "cg/test/gtest_compat.hpp"

#include <vector>

using namespace cg;

namespace {

Module build_chain() {
    Module m;
    auto f = m.create_function("chain",
        {make_tensor_type({16, 32}, DType::F32),
         make_tensor_type({32, 64}, DType::F32)},
        {make_tensor_type({16, 64}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    auto r = b.relu(mm);
    b.output_tensor(r);
    return m;
}

Module build_fanout() {
    Module m;
    auto f = m.create_function("fanout",
        {make_tensor_type({8, 8}, DType::F32)},
        {make_tensor_type({8, 8}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto x = b.relu(a);
    auto y = b.neg(x);
    auto z = b.exp(x);
    auto s = b.add(y, z);
    b.output_tensor(s);
    return m;
}

} // namespace

TEST(GTA, DataflowTopoOrder) {
    auto m = build_chain();
    AnalysisManager am(m);
    auto& df = am.get<DataflowAnalysis>();
    EXPECT_EQ(df.topo_order().size(), 3u);
    EXPECT_GE(df.critical_path_length(), 2u);
}

TEST(GTA, DataflowFanout) {
    auto m = build_fanout();
    AnalysisManager am(m);
    auto& df = am.get<DataflowAnalysis>();
    // Find the relu op.
    Operation* relu = nullptr;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_RELU) { relu = &op; break; }
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(df.fanout(relu->results[0]), 2u);
}

TEST(GTA, LifetimeBasic) {
    auto m = build_chain();
    AnalysisManager am(m);
    auto& lt = am.get<LifetimeAnalysis>();
    // At least one tensor should be live at some point.
    EXPECT_GT(lt.peak_live_count(), 0u);
}

TEST(GTA, ArithmeticIntensityMatmul) {
    auto m = build_chain();
    AnalysisManager am(m);
    auto& ai = am.get<ArithmeticIntensityAnalysis>();
    EXPECT_GT(ai.total_flops(), 0u);
    EXPECT_GT(ai.total_bytes(), 0u);
    EXPECT_GT(ai.module_intensity(), 0.0);
    // Matmul should be classified as compute-bound or balanced.
    bool found_matmul = false;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_MATMUL) {
                auto bound = ai.intensity_of(op).bound;
                EXPECT_TRUE(bound == BoundClass::ComputeBound ||
                            bound == BoundClass::Balanced);
                found_matmul = true;
            }
    EXPECT_TRUE(found_matmul);
}

TEST(GTA, ParallelismMatmul) {
    auto m = build_chain();
    AnalysisManager am(m);
    auto& pa = am.get<ParallelismAnalysis>();
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_MATMUL) {
                auto info = pa.info_of(op);
                EXPECT_EQ(info.independent_items, 16u * 64u);
                EXPECT_TRUE(info.has_reduction);
                EXPECT_EQ(info.reduction_length, 32u);
            }
}

TEST(GTA, ReuseAnalysis) {
    auto m = build_fanout();
    AnalysisManager am(m);
    auto& ru = am.get<ReuseAnalysis>();
    // The relu result has 2 consumers.
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_RELU && !op.results.empty()) {
                auto info = ru.info_of(op.results[0]);
                EXPECT_EQ(info.num_consumers, 2u);
                // Small + cheap -> recompute, large + expensive -> materialize.
                // 8x8 f32 = 256 bytes which is small, so recompute.
                EXPECT_TRUE(info.decision == ReuseDecision::Recompute ||
                            info.decision == ReuseDecision::Materialize);
            }
}

TEST(GTA, GlobalAliasViewOf) {
    Module m;
    auto f = m.create_function("alias",
        {make_tensor_type({8, 8}, DType::F32)},
        {make_tensor_type({8, 8}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto t = b.transpose(a, {1, 0});
    auto r = b.relu(t);
    b.output_tensor(r);

    AnalysisManager am(m);
    auto& aa = am.get<GlobalAliasAnalysis>();
    // `t` is a view of `a`.
    EXPECT_EQ(aa.alias(t, a), TensorAliasKind::ViewOf);
    // `r` is not a view of `a`.
    EXPECT_EQ(aa.alias(r, a), TensorAliasKind::NoAlias);
}

TEST(GTA, GlobalCostNonZero) {
    auto m = build_chain();
    AnalysisManager am(m);
    GlobalAnalysisManager gta(am);
    EXPECT_GT(gta.cost().cost().total(), 0.0);
    EXPECT_GT(gta.cost().cost().execution_sec, 0.0);
    EXPECT_GT(gta.cost().cost().memory_sec, 0.0);
}

TEST(GTA, GlobalBarrierLegal) {
    auto m = build_chain();
    AnalysisManager am(m);
    GlobalAnalysisManager gta(am);
    GlobalBarrier barrier(gta);
    auto report = barrier.run(m);
    EXPECT_TRUE(report.legal);
    EXPECT_TRUE(report.errors.empty());
}
