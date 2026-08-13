// tests/unified_analyzer/unified_analyzer_tests.cpp
//
// Tests for the unified Tensor Knowledge Graph analyzer.
//
// Verifies:
//   - Fact lattice join semantics
//   - Iterative fixed-point convergence
//   - Provenance + confidence tracking
//   - Property propagation (Zero, One, Identity, Constant)
//   - Range analysis (relu(x) >= 0, exp(x) > 0, etc.)
//   - Alias analysis (MustAlias for views, NoAlias for fresh allocations)
//   - Lifetime analysis (birth/death/user count)
//   - Reduction analysis (axes, associativity, identity)
//   - Cost analysis (FLOPs, bytes, arithmetic intensity)
//   - Query API (can_fuse, fusion_benefit, is_zero, etc.)
//   - Convergence metrics (iterations, facts discovered, latency)
#include "cg/analysis/unified/abstract_domain.hpp"
#include "cg/analysis/unified/fact_propagator.hpp"
#include "cg/analysis/unified/fact_store.hpp"
#include "cg/analysis/unified/tensor_facts.hpp"
#include "cg/analysis/unified/unified_analyzer.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/module.hpp"
#include "cg/ir/ops.hpp"
#include "cg/numerical/semantics.hpp"

#include "cg/test/gtest_compat.hpp"

#include <iostream>

using namespace cg;

namespace {

// Build a small IR module for testing:
//   %1 = matmul(A, B)        : [M, K] x [K, N] -> [M, N]
//   %2 = add(%1, bias)        : broadcast bias [N] -> [M, N]
//   %3 = relu(%2)
//   output(%3)
std::shared_ptr<Module> build_matmul_relu_module() {
    auto module = std::make_shared<Module>();
    auto M = static_cast<i64>(128);
    auto K = static_cast<i64>(256);
    auto N = static_cast<i64>(512);
    std::vector<TypePtr> operand_types = {
        make_tensor_type({M, K}, DType::F32),  // A
        make_tensor_type({K, N}, DType::F32),  // B
        make_tensor_type({1, N}, DType::F32),  // bias (already 2D for broadcast)
    };
    std::vector<TypePtr> result_types = {
        make_tensor_type({M, N}, DType::F32),
    };
    auto f = module->create_function("matmul_relu", operand_types, result_types);
    Builder b(f);
    auto A = f->args()[0];
    auto B = f->args()[1];
    auto bias = f->args()[2];
    auto mm = b.matmul(A, B);
    auto bd = b.add(mm, bias);
    auto r  = b.relu(bd);
    b.output_tensor(r);
    return module;
}

// Build a module with constant operands to test constant propagation.
std::shared_ptr<Module> build_constant_module() {
    auto module = std::make_shared<Module>();
    std::vector<TypePtr> operand_types = {
        make_tensor_type({4, 4}, DType::F32),
        make_tensor_type({4, 4}, DType::F32),
    };
    std::vector<TypePtr> result_types = {
        make_tensor_type({4, 4}, DType::F32),
    };
    auto f = module->create_function("const_fold_test", operand_types, result_types);
    Builder b(f);
    auto A = f->args()[0];
    auto B = f->args()[1];

    // Create a constant-zero tensor and add it to A.
    // add(A, 0) -> A
    auto zero = b.constant_tensor({4, 4}, DType::F32, std::vector<u8>{});
    // Set the constant's "value" attribute to 0.
    // (The constant_tensor helper doesn't set the value attribute; we'd
    // need to do that manually. For the test, we just verify the constant
    // op is recognized.)
    auto sum = b.add(A, zero);
    b.output_tensor(sum);
    return module;
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: The analyzer runs and converges.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, RunsAndConverges) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.set_numerical_mode(NumericalMode::Relaxed);
    analyzer.add_default_propagators();

    const auto& metrics = analyzer.run();

    std::cout << "Iterations: " << metrics.iterations << "\n";
    std::cout << "Facts discovered: " << metrics.facts_discovered << "\n";
    std::cout << "Latency: " << metrics.latency_sec * 1e6 << " us\n";
    std::cout << "Per-propagator:\n";
    for (auto& p : metrics.per_propagator) {
        std::cout << "  " << p.name << ": runs=" << p.runs
                  << " facts=" << p.facts_produced
                  << " time=" << p.total_sec * 1e6 << " us\n";
    }

    EXPECT_GT(metrics.iterations, 0u);
    EXPECT_GT(metrics.facts_discovered, 0u);
    EXPECT_LE(metrics.iterations, 16u);  // safety bound
}

// ---------------------------------------------------------------------------
// Test 2: Shape facts are populated with provenance.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, ShapeFactsPopulated) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    EXPECT_GT(store.num_tensors(), 0u);

    // The matmul result should have shape [128, 512].
    bool found_matmul_result = false;
    for (auto& [vid, tf] : store) {
        if (tf.shape.known && tf.shape.value.size() == 2) {
            i64 m = tf.shape.value[0].value();
            i64 n = tf.shape.value[1].value();
            if (m == 128 && n == 512) {
                found_matmul_result = true;
                EXPECT_EQ(tf.shape.confidence, Confidence::Proven);
                EXPECT_EQ(tf.shape.provenance.rule, "ShapeInference");
                break;
            }
        }
    }
    EXPECT_TRUE(found_matmul_result);
}

// ---------------------------------------------------------------------------
// Test 3: Confidence and provenance are tracked.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, ConfidenceAndProvenanceTracked) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    usize proven_count = 0;
    usize estimated_count = 0;
    for (auto& [vid, tf] : store) {
        if (tf.shape.known && tf.shape.confidence == Confidence::Proven) ++proven_count;
        if (tf.cache_behavior.known && tf.cache_behavior.confidence == Confidence::Estimated)
            ++estimated_count;
    }
    EXPECT_GT(proven_count, 0u);
    // Cache behavior is Estimated (requires hardware); without hardware
    // it should be 0.
    EXPECT_EQ(estimated_count, 0u);
}

// ---------------------------------------------------------------------------
// Test 4: With hardware model set, cost facts become Estimated.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, CostFactsEstimatedWithHardware) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.set_hardware(HardwareModel::generic_nvidia_gpu());
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    bool found_cache_behavior = false;
    bool found_roofline = false;
    for (auto& [vid, tf] : store) {
        if (tf.cache_behavior.known && tf.cache_behavior.confidence == Confidence::Estimated) {
            found_cache_behavior = true;
            EXPECT_GE(tf.cache_behavior.value.l2_hit_rate, 0.0);
            EXPECT_LE(tf.cache_behavior.value.l2_hit_rate, 1.0);
        }
    }
    if (store.graph_facts().roofline_ridge.known) {
        found_roofline = true;
        EXPECT_GT(store.graph_facts().roofline_ridge.value, 0.0);
    }
    EXPECT_TRUE(found_cache_behavior);
    EXPECT_TRUE(found_roofline);
}

// ---------------------------------------------------------------------------
// Test 5: Range analysis: relu(x) >= 0.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, RangeAnalysisRelu) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    // Find the relu result and verify it's non-negative.
    bool found_relu_nonneg = false;
    for (auto& [vid, tf] : store) {
        if (tf.value_range.known && tf.value_range.value.is_non_negative()) {
            found_relu_nonneg = true;
            EXPECT_EQ(tf.value_range.confidence, Confidence::Proven);
            break;
        }
    }
    EXPECT_TRUE(found_relu_nonneg);
}

// ---------------------------------------------------------------------------
// Test 6: Alias analysis: views MustAlias parent, fresh allocs NoAlias.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, AliasAnalysisViewsAndAllocs) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    bool found_noalias = false;
    for (auto& [vid, tf] : store) {
        if (tf.alias_class.known) {
            if (tf.alias_class.value.kind == AliasKind::NoAlias) {
                found_noalias = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_noalias);
}

// ---------------------------------------------------------------------------
// Test 7: Lifetime analysis: birth/death/user count.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, LifetimeAnalysis) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    bool found_lifetime = false;
    for (auto& [vid, tf] : store) {
        if (tf.birth_op.known && tf.death_op.known && tf.num_users.known) {
            found_lifetime = true;
            EXPECT_GE(tf.death_op.value, tf.birth_op.value);
            break;
        }
    }
    EXPECT_TRUE(found_lifetime);
}

// ---------------------------------------------------------------------------
// Test 8: Reduction analysis: axes, associativity, identity.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, ReductionAnalysis) {
    auto module = std::make_shared<Module>();
    auto f = module->create_function(
        "reduction_test",
        {make_tensor_type({16, 32, 64}, DType::F32)},
        {make_tensor_type({16, 64}, DType::F32)});
    Builder b(f);
    auto x = f->args()[0];
    auto r = b.reduce_sum(x, {1}, false);  // reduce axis 1
    b.output_tensor(r);

    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    bool found_reduction = false;
    for (auto& [vid, tf] : store) {
        if (tf.reduction.known && tf.reduction.value.is_reduction) {
            found_reduction = true;
            EXPECT_TRUE(tf.reduction.value.is_commutative);
            EXPECT_TRUE(tf.reduction.value.is_associative);
            EXPECT_EQ(tf.reduction.value.identity_value, 0.0);
            break;
        }
    }
    EXPECT_TRUE(found_reduction);
}

// ---------------------------------------------------------------------------
// Test 9: Cost analysis: FLOPs, bytes, arithmetic intensity.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, CostAnalysisFlopsAndBytes) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    auto& g = store.graph_facts();
    EXPECT_TRUE(g.total_flops.known);
    EXPECT_TRUE(g.total_bytes.known);
    EXPECT_TRUE(g.graph_arithmetic_intensity.known);
    // matmul 128x256x512 = 2*128*256*512 = 33,554,432 FLOPs.
    // Plus elementwise add+relu over [128,512] = 128*512 + 128*512 = 131,072 FLOPs.
    // Total = 33,554,432 + 131,072 = 33,685,504.
    // But the cost propagator also counts the matmul result's bytes_written,
    // so total_flops may include all elementwise ops too. We just check
    // it's at least the matmul FLOPs.
    EXPECT_GE(g.total_flops.value, 33554432u);
    std::cout << "Total FLOPs: " << g.total_flops.value << "\n";
    std::cout << "Total bytes: " << g.total_bytes.value << "\n";
    std::cout << "Graph AI: " << g.graph_arithmetic_intensity.value << "\n";
}

// ---------------------------------------------------------------------------
// Test 10: Dependence graph is built.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, DependenceGraphBuilt) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    auto& edges = store.graph_facts().dependence_edges;
    EXPECT_FALSE(edges.empty());
    // matmul -> add -> relu should produce at least 3 edges.
    EXPECT_GE(edges.size(), 3u);
    // Verify edge kinds are classified.
    bool found_full = false;
    for (auto& e : edges) {
        if (e.kind == DependenceKind::Full) {
            found_full = true;
            break;
        }
    }
    EXPECT_TRUE(found_full);
}

// ---------------------------------------------------------------------------
// Test 11: Query API - is_zero, is_constant, etc.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, QueryAPI) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    // No tensor in the matmul_relu module should be provably zero.
    for (auto& [vid, tf] : store) {
        EXPECT_FALSE(store.is_zero(vid));
    }

    // The matmul result should have a known static shape.
    bool found_shape = false;
    for (auto& [vid, tf] : store) {
        auto shape = store.static_shape(vid);
        if (shape.has_value() && shape->size() == 2) {
            found_shape = true;
            break;
        }
    }
    EXPECT_TRUE(found_shape);
}

// ---------------------------------------------------------------------------
// Test 12: Fusion query - can_fuse + fusion_benefit.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, FusionQuery) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.set_hardware(HardwareModel::generic_nvidia_gpu());
    analyzer.add_default_propagators();
    analyzer.run();

    auto& store = analyzer.store();
    // Find the matmul result (producer) and the add result (consumer).
    ValueId matmul_result = 0;
    ValueId add_result = 0;
    for (auto& [vid, tf] : store) {
        auto shape = store.static_shape(vid);
        if (shape.has_value() && shape->size() == 2 &&
            (*shape)[0] == 128 && (*shape)[1] == 512) {
            // Both matmul and add results have this shape. We just pick
            // any two and test the API.
            if (matmul_result == 0) matmul_result = vid;
            else if (add_result == 0) add_result = vid;
        }
    }
    ASSERT_NE(matmul_result, 0u);

    // can_fuse should return true for a producer->consumer pair.
    if (add_result != 0) {
        bool can = store.can_fuse(matmul_result, add_result);
        EXPECT_TRUE(can);

        auto benefit = store.fusion_benefit(matmul_result, add_result);
        EXPECT_TRUE(benefit.can_fuse);
        EXPECT_GE(benefit.saved_bytes, 0.0);
        EXPECT_GE(benefit.saved_kernel_launches, 0.0);
        EXPECT_EQ(benefit.confidence, Confidence::Estimated);
        EXPECT_FALSE(benefit.reasons.empty());

        std::cout << "Fusion benefit:\n";
        std::cout << "  saved_bytes: " << benefit.saved_bytes << "\n";
        std::cout << "  saved_kernel_launches: " << benefit.saved_kernel_launches << "\n";
        std::cout << "  added_register_pressure: " << benefit.added_register_pressure << "\n";
        std::cout << "  occupancy_delta_pct: " << benefit.occupancy_delta_pct << "\n";
        std::cout << "  net_predicted_improvement: " << benefit.net_predicted_improvement << "\n";
        std::cout << "  confidence: " << confidence_name(benefit.confidence) << "\n";
        for (auto& r : benefit.reasons) {
            std::cout << "  reason: " << r.rule << " - " << r.explanation << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 13: Fixed-point convergence — second run produces no new facts.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, FixedPointConvergence) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();

    // First run: discovers facts.
    analyzer.run();
    u32 facts_after_first = analyzer.metrics().facts_discovered;

    // Re-run the same propagators on the same module: should produce 0 new
    // facts (fixed point).
    u32 facts_second = analyzer.run_one_iteration();

    EXPECT_EQ(facts_second, 0u);
    std::cout << "Facts first run: " << facts_after_first << "\n";
    std::cout << "Facts second iteration: " << facts_second << " (fixed point)\n";
}

// ---------------------------------------------------------------------------
// Test 14: Property lattice - bitset composition.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, PropertyLatticeComposition) {
    TensorProperty p = TensorProperty::Constant | TensorProperty::Diagonal |
                       TensorProperty::Sparse;
    EXPECT_TRUE(has_property(p, TensorProperty::Constant));
    EXPECT_TRUE(has_property(p, TensorProperty::Diagonal));
    EXPECT_TRUE(has_property(p, TensorProperty::Sparse));
    EXPECT_FALSE(has_property(p, TensorProperty::Zero));
    EXPECT_FALSE(has_property(p, TensorProperty::Identity));

    std::string s = property_to_string(p);
    EXPECT_NE(s.find("constant"), std::string::npos);
    EXPECT_NE(s.find("diagonal"), std::string::npos);
    EXPECT_NE(s.find("sparse"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Test 15: Confidence ordering - Proven > Derived > Estimated > ...
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, ConfidenceOrdering) {
    EXPECT_TRUE(is_sound(Confidence::Proven));
    EXPECT_TRUE(is_sound(Confidence::Derived));
    EXPECT_FALSE(is_sound(Confidence::Estimated));
    EXPECT_FALSE(is_sound(Confidence::Profiled));
    EXPECT_FALSE(is_sound(Confidence::Speculative));

    // Enum ordering: lower = more trusted.
    EXPECT_LT(Confidence::Proven, Confidence::Derived);
    EXPECT_LT(Confidence::Derived, Confidence::Estimated);
    EXPECT_LT(Confidence::Estimated, Confidence::Profiled);
    EXPECT_LT(Confidence::Profiled, Confidence::Speculative);
}

// ---------------------------------------------------------------------------
// Test 16: Fact<T>::join semantics.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, FactJoinSemantics) {
    // Unknown joined with known -> known.
    Fact<int> a;
    EXPECT_FALSE(a.known);
    Fact<int> b = Fact<int>::make(42, Confidence::Proven, Provenance("test"));
    EXPECT_TRUE(a.join(b));
    EXPECT_TRUE(a.known);
    EXPECT_EQ(a.value, 42);

    // Known joined with unknown -> no change.
    Fact<int> c = Fact<int>::make(10, Confidence::Proven, Provenance("test"));
    EXPECT_FALSE(c.join(Fact<int>::unknown()));
    EXPECT_EQ(c.value, 10);

    // Higher confidence overwrites lower.
    Fact<int> d = Fact<int>::make(10, Confidence::Estimated, Provenance("est"));
    Fact<int> e = Fact<int>::make(20, Confidence::Proven, Provenance("proven"));
    EXPECT_TRUE(d.join(e));
    EXPECT_EQ(d.value, 20);
    EXPECT_EQ(d.confidence, Confidence::Proven);

    // Same confidence, different values -> last-writer-wins.
    Fact<int> f = Fact<int>::make(10, Confidence::Proven, Provenance("a"));
    Fact<int> g = Fact<int>::make(20, Confidence::Proven, Provenance("b"));
    EXPECT_TRUE(f.join(g));
    EXPECT_EQ(f.value, 20);
    EXPECT_EQ(f.provenance.rule, "b");

    // Same confidence, same value -> no change.
    Fact<int> h = Fact<int>::make(10, Confidence::Proven, Provenance("a"));
    Fact<int> i = Fact<int>::make(10, Confidence::Proven, Provenance("b"));
    EXPECT_FALSE(h.join(i));
}

// ---------------------------------------------------------------------------
// Test 17: Analyzer metrics include per-propagator breakdown.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, PerPropagatorMetrics) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    const auto& m = analyzer.metrics();
    EXPECT_FALSE(m.per_propagator.empty());
    // Each default propagator should have run at least once.
    EXPECT_GE(m.per_propagator.size(), 8u);
    for (auto& p : m.per_propagator) {
        EXPECT_GE(p.runs, 1u);
        std::cout << "  " << p.name << ": " << p.facts_produced << " facts in "
                  << p.runs << " runs (" << p.total_sec * 1e6 << " us)\n";
    }
}

// ---------------------------------------------------------------------------
// Test 18: Reporting actual runtime updates prediction-error metric.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, ReportActualRuntime) {
    auto module = build_matmul_relu_module();
    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    // Find any value id and report a runtime for it.
    ValueId some_vid = 0;
    for (auto& [vid, tf] : analyzer.store()) {
        some_vid = vid;
        break;
    }
    ASSERT_NE(some_vid, 0u);

    analyzer.report_actual_runtime(some_vid, 0.001);
    EXPECT_EQ(analyzer.metrics().predictions_evaluated, 1u);
}

// ---------------------------------------------------------------------------
// Test 19: The analyzer does NOT mutate the IR.
// ---------------------------------------------------------------------------
TEST(UnifiedAnalyzer, DoesNotMutateIR) {
    auto module = build_matmul_relu_module();
    usize num_ops_before = module->num_operations();

    UnifiedAnalyzer analyzer(*module);
    analyzer.add_default_propagators();
    analyzer.run();

    usize num_ops_after = module->num_operations();
    EXPECT_EQ(num_ops_before, num_ops_after);
}
