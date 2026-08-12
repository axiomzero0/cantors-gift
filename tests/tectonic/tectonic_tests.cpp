// tests/tectonic/tectonic_tests.cpp - tests for the tectonic pieces
#include "cg/backend/tensor_core/ptx_mma.hpp"
#include "cg/cost/cost_model_v2.hpp"
#include "cg/egraph/rewrite_rules.hpp"
#include "cg/numerical/semantics.hpp"
#include "cg/pipeline/software_pipeline.hpp"
#include "cg/schedule/schedule.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(Tectonic, PTXMMAEmitsValidInstruction) {
    PTXMMAEmitter emitter;
    u32 next_reg = 0;
    auto frags = emitter.allocate_fragments(next_reg);
    std::string ptx = emitter.emit_mma_m16n8k16_f16_f32(frags);
    EXPECT_NE(ptx.find("mma.sync.aligned.m16n8k16"), std::string::npos);
    EXPECT_NE(ptx.find("f32.f16.f16.f32"), std::string::npos);
}

TEST(Tectonic, PTXMMAZeroAccumulator) {
    PTXMMAEmitter emitter;
    u32 next_reg = 0;
    auto frags = emitter.allocate_fragments(next_reg);
    std::string ptx = emitter.emit_zero_accumulator(frags);
    EXPECT_NE(ptx.find("mov.f32"), std::string::npos);
    EXPECT_NE(ptx.find("0.0"), std::string::npos);
}

TEST(Tectonic, PTXMMAFullKernel) {
    PTXMMAEmitter emitter;
    std::string ptx = emitter.emit_tiled_matmul_kernel(128, 128, 128, DType::F16, "gemm_tc", 64, 64, 32);
    EXPECT_NE(ptx.find("mma.sync.aligned.m16n8k16"), std::string::npos);
    EXPECT_NE(ptx.find("sm_80"), std::string::npos);
    EXPECT_NE(ptx.find("ldmatrix"), std::string::npos);
    EXPECT_NE(ptx.find("gemm_tc"), std::string::npos);
}

TEST(Tectonic, TensorCoreCaps) {
    auto ampere = TensorCoreCaps::ampere();
    EXPECT_TRUE(ampere.has_ampere_mma);
    EXPECT_GT(ampere.peak_f16_tflops, 300.0);

    auto hopper = TensorCoreCaps::hopper();
    EXPECT_TRUE(hopper.has_hopper_wgmma);
    EXPECT_TRUE(hopper.has_tma);
    EXPECT_GT(hopper.peak_f16_tflops, 900.0);

    auto blackwell = TensorCoreCaps::blackwell();
    EXPECT_GT(blackwell.peak_fp8_tflops, 3000.0);
}

TEST(Tectonic, PipelinePrologue) {
    SoftwarePipelineEmitter emitter;
    PipelineConfig config;
    config.num_stages = 3;
    config.use_async_copy = true;
    PipelineStage stage;
    stage.loads.push_back({"A", "%rd_A", "%smem_A", "%r_off", 4096});
    stage.computes.push_back({"mma.sync"});
    std::string ptx = emitter.emit_prologue(config, stage, 128, 32);
    EXPECT_NE(ptx.find("cp.async"), std::string::npos);
    EXPECT_NE(ptx.find("commit_group"), std::string::npos);
}

TEST(Tectonic, PipelineSteadyState) {
    SoftwarePipelineEmitter emitter;
    PipelineConfig config;
    config.num_stages = 3;
    PipelineStage stage;
    stage.loads.push_back({"A", "%rd_A", "%smem_A", "%r_off", 4096});
    stage.computes.push_back({"mma.sync"});
    std::string ptx = emitter.emit_steady_state(config, stage, 128, 32);
    EXPECT_NE(ptx.find("L_pipeline_steady"), std::string::npos);
}

TEST(Tectonic, PipelineCompleteLoop) {
    SoftwarePipelineEmitter emitter;
    PipelineConfig config;
    config.num_stages = 3;
    PipelineStage stage;
    stage.loads.push_back({"A", "%rd_A", "%smem_A", "%r_off", 4096});
    stage.computes.push_back({"mma.sync"});
    std::string ptx = emitter.emit_pipelined_loop(config, stage, 128, 32);
    EXPECT_NE(ptx.find("Prologue"), std::string::npos);
    EXPECT_NE(ptx.find("Steady State"), std::string::npos);
    EXPECT_NE(ptx.find("Epilogue"), std::string::npos);
}

TEST(Tectonic, PipelineOverlapEstimate) {
    auto est0 = estimate_pipeline_overlap(1.0, 2.0, 1, 4, 65536, 4096);
    EXPECT_NEAR(est0.overlapped_sec, 2.0, 0.01);
    EXPECT_NEAR(est0.overlap_factor, 0.0, 0.01);

    auto est3 = estimate_pipeline_overlap(1.0, 2.0, 3, 4, 65536, 4096);
    EXPECT_LT(est3.overlapped_sec, 3.0);
    EXPECT_GT(est3.overlap_factor, 0.5);
}

TEST(Tectonic, CostModelV2Breakdown) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    AnalyticalCostModelV2 model(hw);
    CostFeatures f;
    f.flops = 2e9;
    f.bytes_global_load = 8e6;
    f.bytes_global_store = 4e6;
    f.num_blocks = 256;
    f.num_sms = 108;
    auto bd = model.estimate(f);
    EXPECT_GT(bd.compute_sec, 0);
    EXPECT_GT(bd.memory_global_sec, 0);
    EXPECT_GT(bd.total_sec, 0);
}

TEST(Tectonic, CostModelV2WaveQuantization) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    AnalyticalCostModelV2 model(hw);
    u32 num_sms = hw.num_cores;

    // num_sms + 1 blocks = 2 waves with wasted SMs
    CostFeatures f;
    f.flops = 1e9;
    f.num_blocks = num_sms + 1;
    f.num_sms = num_sms;
    f.threads_per_block = 256;
    auto bd = model.estimate(f);
    EXPECT_GT(bd.wave_quant_sec, 0);

    // Exactly num_sms blocks = 1 wave, no penalty
    f.num_blocks = num_sms;
    auto bd2 = model.estimate(f);
    EXPECT_EQ(bd2.wave_quant_sec, 0.0);
}

TEST(Tectonic, CostModelV2BankConflict) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    AnalyticalCostModelV2 model(hw);
    CostFeatures f;
    f.bytes_shared_load = 1e6;
    f.bank_conflict_ways = 4;
    auto bd = model.estimate(f);
    EXPECT_GT(bd.bank_conflict_sec, 0);
}

TEST(Tectonic, CostModelV2PipelineOverlap) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    AnalyticalCostModelV2 model(hw);
    CostFeatures f1;
    f1.flops = 1e12;
    f1.bytes_global_load = 1e9;
    f1.num_pipeline_stages = 1;
    CostFeatures f2 = f1;
    f2.num_pipeline_stages = 3;
    auto b1 = model.estimate(f1);
    auto b2 = model.estimate(f2);
    // Pipelined version should be faster or equal (overlap helps)
    EXPECT_LE(b2.total_sec, b1.total_sec * 1.1);
}

TEST(Tectonic, CostModelV2ExtractFeatures) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    AnalyticalCostModelV2 model(hw);
    Schedule s;
    s.add({TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Tile, "n", 64, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Bind, "tc", 0, 0, "tensor_core", MemorySpace::Generic});
    auto features = model.extract_features(s, 1024, 1024, 1024, DType::F32);
    EXPECT_EQ(features.flops, 2ull * 1024 * 1024 * 1024);
    EXPECT_TRUE(features.uses_tensor_core);
    EXPECT_EQ(features.m_tile, 64u);
}

TEST(Tectonic, LearnedCostModelTrain) {
    LearnedCostModel model;
    EXPECT_FALSE(model.is_trained());
    std::vector<std::pair<CostFeatures, double>> data;
    for (int i = 0; i < 20; ++i) {
        CostFeatures f;
        f.flops = i * 1e9;
        f.bytes_global_load = i * 1e6;
        data.push_back({f, i * 0.001});
    }
    model.train(data);
    EXPECT_TRUE(model.is_trained());
    CostFeatures test_f;
    test_f.flops = 5e9;
    test_f.bytes_global_load = 5e6;
    auto pred = model.predict(test_f);
    ASSERT_TRUE(pred.has_value());
    EXPECT_GE(*pred, 0);
}

TEST(Tectonic, RewriteRulesNonEmpty) {
    auto rules = get_rewrite_rules(NumericalMode::Relaxed);
    EXPECT_FALSE(rules.empty());
}

TEST(Tectonic, RewriteRulesFilteredByMode) {
    auto strict_rules = get_rewrite_rules(NumericalMode::Strict);
    auto relaxed_rules = get_rewrite_rules(NumericalMode::Relaxed);
    auto fast_rules = get_rewrite_rules(NumericalMode::FastMath);
    EXPECT_LE(strict_rules.size(), relaxed_rules.size());
    EXPECT_LE(relaxed_rules.size(), fast_rules.size());
}

TEST(Tectonic, CommutativityRulesPresent) {
    auto rules = commutativity_rules();
    EXPECT_GE(rules.size(), 4u);
    bool has_add = false, has_mul = false;
    for (const auto& r : rules) {
        if (r.name.find("add") != std::string::npos) has_add = true;
        if (r.name.find("mul") != std::string::npos) has_mul = true;
    }
    EXPECT_TRUE(has_add);
    EXPECT_TRUE(has_mul);
}

TEST(Tectonic, TCEligibilityRulesRequireFastMath) {
    auto rules = tc_eligibility_rules();
    for (const auto& r : rules) {
        EXPECT_EQ(r.min_mode, NumericalMode::FastMath);
        EXPECT_EQ(r.kind, RuleKind::TCEligibility);
    }
}
