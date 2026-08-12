// tests/cost_tests.cpp - cost model tests
#include "cg/cost/hardware_model.hpp"
#include "cg/cost/estimator.hpp"
#include "cg/schedule/schedule.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(Cost, GenericCpuModel) {
    auto hw = HardwareModel::generic_cpu();
    EXPECT_GT(hw.peak_flops(DType::F32), 0.0);
    EXPECT_GT(hw.memory.get(MemorySpace::Generic), 0.0);
}

TEST(Cost, GenericGpuModel) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    EXPECT_GT(hw.peak_flops(DType::F16), 0.0);
    EXPECT_GT(hw.peak_flops(DType::F16, true), 0.0);
    EXPECT_GT(hw.shared_mem_bytes, 0u);
}

TEST(Cost, MatmulEstimate) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    CostEstimator est(hw);
    Schedule s;  // empty schedule
    auto c = est.estimate_matmul(1024, 1024, 1024, DType::F32, s);
    EXPECT_EQ(c.flops, 2ull * 1024 * 1024 * 1024);
    EXPECT_GT(c.bytes_global, 0u);
    EXPECT_GT(c.estimated_runtime_sec, 0.0);
}

TEST(Cost, ScheduleSpace) {
    auto space = ScheduleSpace::grid_matmul({64, 128}, {64, 128}, {32}, {8, 16});
    EXPECT_EQ(space.size(), 2u * 2u * 1u * 2u);
}

TEST(Cost, Rank) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    CostEstimator est(hw);
    auto space = ScheduleSpace::grid_matmul({64, 128}, {64, 128}, {32}, {8, 16});
    auto ranked = est.rank(space, 3);
    EXPECT_LE(ranked.size(), 3u);
    // Sorted by estimated runtime.
    for (usize i = 1; i < ranked.size(); ++i) {
        EXPECT_LE(ranked[i - 1].second.estimated_runtime_sec,
                  ranked[i].second.estimated_runtime_sec);
    }
}
