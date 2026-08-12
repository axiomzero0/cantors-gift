// tests/autotuner_tests.cpp - Bayesian autotuner tests
#include "cg/autotuner/bayesian_optimizer.hpp"
#include "cg/schedule/schedule.hpp"

#include "cg/test/gtest_compat.hpp"

#include <cmath>

using namespace cg;

TEST(Autotuner, ExtractFeatures) {
    Schedule s;
    s.add({TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Tile, "n", 128, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Tile, "k", 32, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Vectorize, "n_inner", 8, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Parallelize, "m", 0, 0, "", MemorySpace::Generic});
    s.add({TransformKind::Cache, "a", 0, 0, "A", MemorySpace::Shared});

    auto f = extract_features(s);
    EXPECT_EQ(f.m_tile, 64.0);
    EXPECT_EQ(f.n_tile, 128.0);
    EXPECT_EQ(f.k_tile, 32.0);
    EXPECT_EQ(f.vector_width, 8.0);
    EXPECT_EQ(f.num_parallel_axes, 1.0);
    EXPECT_EQ(f.uses_shared_memory, 1.0);
}

TEST(Autotuner, GaussianProcessObserve) {
    GaussianProcess gp;
    gp.observe({1.0, 0.0}, 10.0);
    gp.observe({0.0, 1.0}, 20.0);
    EXPECT_EQ(gp.num_observations(), 2u);
}

TEST(Autotuner, GaussianProcessPredict) {
    GaussianProcess gp;
    gp.observe({1.0, 0.0}, 10.0);
    gp.observe({0.0, 1.0}, 20.0);

    auto pred = gp.predict({1.0, 0.0});
    EXPECT_NEAR(pred.mean, 10.0, 1.0);
    EXPECT_GE(pred.variance, 0.0);
}

TEST(Autotuner, ExpectedImprovement) {
    GaussianProcess gp;
    gp.observe({1.0, 0.0}, 10.0);
    gp.observe({0.0, 1.0}, 20.0);

    double ei = gp.expected_improvement({0.5, 0.5}, 10.0);
    EXPECT_GE(ei, 0.0);
}

TEST(Autotuner, BayesianAutotuneFindsOptimum) {
    // Build a schedule space.
    auto space = ScheduleSpace::grid_matmul(
        {32, 64, 128}, {32, 64, 128}, {16, 32}, {4, 8, 16});
    EXPECT_GT(space.size(), 0u);

    // Benchmark function with a known optimum at tile=64, vector=8.
    auto benchmark = [](const Schedule& s) -> double {
        auto f = extract_features(s);
        double cost = 1000.0;
        cost += std::abs(f.m_tile - 64) * 2.0;
        cost += std::abs(f.n_tile - 64) * 2.0;
        cost += std::abs(f.k_tile - 32) * 1.0;
        cost += std::abs(f.vector_width - 8) * 3.0;
        return cost;
    };

    auto result = bayesian_autotune(space, benchmark, 15, 3);
    EXPECT_GT(result.total_benchmarks, 0u);
    EXPECT_LT(result.best_runtime, std::numeric_limits<double>::infinity());
    // The best runtime should be close to the optimum (1000).
    EXPECT_LT(result.best_runtime, 1100.0);
}

TEST(Autotuner, BayesianAutotuneHistoryDecreasing) {
    auto space = ScheduleSpace::grid_matmul({64, 128}, {64, 128}, {32}, {8, 16});
    auto benchmark = [](const Schedule& s) -> double {
        auto f = extract_features(s);
        return 1000.0 + std::abs(f.m_tile - 64) + std::abs(f.vector_width - 8);
    };

    auto result = bayesian_autotune(space, benchmark, 10, 3);
    ASSERT_FALSE(result.runtime_history.empty());
    // The last entry should be <= the first.
    EXPECT_LE(result.runtime_history.back(), result.runtime_history.front());
}
