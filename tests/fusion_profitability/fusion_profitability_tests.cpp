// tests/fusion_profitability/fusion_profitability_tests.cpp
#include "cg/analysis/analysis.hpp"
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/analysis/global_alias_analysis.hpp"
#include "cg/analysis/global_analysis.hpp"
#include "cg/analysis/parallelism_analysis.hpp"
#include "cg/analysis/reuse_analysis.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/fusion/fusion.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

namespace {
Module build_elementwise_chain() {
    Module m;
    auto f = m.create_function("chain",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r = b.relu(f->args()[0]);
    auto e = b.exp(r);
    b.output_tensor(e);
    return m;
}

Module build_multi_consumer() {
    Module m;
    auto f = m.create_function("multi",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r = b.relu(f->args()[0]);
    auto e = b.exp(r);
    auto n = b.neg(r);
    auto s = b.add(e, n);
    b.output_tensor(s);
    return m;
}

Module build_horizontal_fusion() {
    Module m;
    auto f = m.create_function("h",
        {make_tensor_type({32, 32}, DType::F32),
         make_tensor_type({32, 32}, DType::F32),
         make_tensor_type({32, 32}, DType::F32),
         make_tensor_type({32, 32}, DType::F32)},
        {make_tensor_type({32, 32}, DType::F32)});
    Builder b(f);
    auto s1 = b.add(f->args()[0], f->args()[1]);
    auto s2 = b.add(f->args()[1], f->args()[2]);
    auto s3 = b.add(f->args()[2], f->args()[3]);
    b.output_tensor(s1);
    b.output_tensor(s2);
    b.output_tensor(s3);
    return m;
}
} // namespace

TEST(FusionProfitability, ElementwiseChainFuses) {
    auto m = build_elementwise_chain();
    AnalysisManager am(m);
    FusionPass fusion;
    fusion.run(m, am);
    DCEPass dce;
    dce.run(m, am);
    bool has_fused = false;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.attributes.get("fused_chain")) has_fused = true;
    EXPECT_TRUE(has_fused);
}

TEST(FusionProfitability, MultiConsumerFusion) {
    auto m = build_multi_consumer();
    AnalysisManager am(m);
    FusionPass fusion;
    fusion.run(m, am);
    bool has_fused = false;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.attributes.get("fused_chain")) has_fused = true;
    EXPECT_TRUE(has_fused);
}

TEST(FusionProfitability, HorizontalFusionFindsGroups) {
    auto m = build_horizontal_fusion();
    AnalysisManager am(m);
    FusionPass fusion;
    fusion.run(m, am);
    bool has_horizontal = false;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.attributes.get("horizontal_group")) has_horizontal = true;
    EXPECT_TRUE(has_horizontal);
}

TEST(FusionProfitability, GPUHardwareModelFusion) {
    auto m = build_elementwise_chain();
    AnalysisManager am(m);
    GlobalAnalysisManager gta(am);
    gta.set_hardware(HardwareModel::generic_nvidia_gpu());
    FusionPass fusion;
    fusion.run(m, am);
    bool has_fused = false;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.attributes.get("fused_chain")) has_fused = true;
    EXPECT_TRUE(has_fused);
}

TEST(FusionProfitability, ArithmeticIntensityAllOps) {
    Module m;
    auto f = m.create_function("all_ops",
        {make_tensor_type({8, 16}, DType::F32),
         make_tensor_type({16, 32}, DType::F32)},
        {make_tensor_type({8, 32}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    auto r = b.relu(mm);
    auto sm = b.softmax(r);
    auto ln = b.layernorm(sm);
    b.output_tensor(ln);
    AnalysisManager am(m);
    auto& ai = am.get<ArithmeticIntensityAnalysis>();
    usize ops_with_flops = 0;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (ai.intensity_of(op).flops > 0) ops_with_flops++;
    EXPECT_GE(ops_with_flops, 4u);
}

TEST(FusionProfitability, Conv2DFlops) {
    Module m;
    auto f = m.create_function("conv",
        {make_tensor_type({1, 3, 32, 32}, DType::F32),
         make_tensor_type({16, 3, 3, 3}, DType::F32)},
        {make_tensor_type({1, 16, 30, 30}, DType::F32)});
    Builder b(f);
    auto conv = b.conv2d(f->args()[0], f->args()[1]);
    b.output_tensor(conv);
    AnalysisManager am(m);
    auto& ai = am.get<ArithmeticIntensityAnalysis>();
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_CONV2D) {
                auto oi = ai.intensity_of(op);
                EXPECT_GT(oi.flops, 0);
                EXPECT_EQ(oi.flops, 2u * 1 * 16 * 30 * 30 * 3 * 3 * 3);
                return;
            }
    SUCCEED(); return;
}

TEST(FusionProfitability, SoftmaxFlops) {
    Module m;
    auto f = m.create_function("sm",
        {make_tensor_type({8, 16}, DType::F32)},
        {make_tensor_type({8, 16}, DType::F32)});
    Builder b(f);
    auto sm = b.softmax(f->args()[0]);
    b.output_tensor(sm);
    AnalysisManager am(m);
    auto& ai = am.get<ArithmeticIntensityAnalysis>();
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_SOFTMAX) {
                auto oi = ai.intensity_of(op);
                EXPECT_EQ(oi.flops, 4u * 8 * 16);
                return;
            }
    SUCCEED(); return;
}

TEST(FusionProfitability, GeluMoreExpensiveThanRelu) {
    Module m1;
    auto f1 = m1.create_function("relu_test",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b1(f1);
    auto r = b1.relu(f1->args()[0]);
    b1.output_tensor(r);

    Module m2;
    auto f2 = m2.create_function("gelu_test",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b2(f2);
    auto g = b2.gelu(f2->args()[0]);
    b2.output_tensor(g);

    AnalysisManager am1(m1);
    AnalysisManager am2(m2);
    auto& ai1 = am1.get<ArithmeticIntensityAnalysis>();
    auto& ai2 = am2.get<ArithmeticIntensityAnalysis>();

    u64 relu_flops = 0, gelu_flops = 0;
    for (auto& f : m1.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_RELU) relu_flops = ai1.intensity_of(op).flops;
    for (auto& f : m2.functions())
        for (auto& op : *f->entry())
            if (op.opcode == OP_GELU) gelu_flops = ai2.intensity_of(op).flops;

    EXPECT_GT(gelu_flops, relu_flops);
    EXPECT_EQ(gelu_flops, relu_flops * 8);
}
