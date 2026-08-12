// autotuner/bayesian_optimizer.hpp - Bayesian optimization over ScheduleSpace
//
// Implements a Gaussian Process (GP) cost model with Expected Improvement
// (EI) acquisition function. The autotuner:
//
//   1. Starts with a few random schedules as initial observations.
//   2. Fits a GP to (schedule_features -> measured_runtime).
//   3. Computes EI for each untested candidate.
//   4. Benchmarks the top-k EI candidates.
//   5. Updates the GP and repeats.
//   6. Returns the best schedule found.
//
// The GP uses a squared-exponential (RBF) kernel. The features are extracted
// from the schedule (tile sizes, vector width, unroll factor, etc.).
#pragma once

#include "cg/core/util.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/schedule/schedule.hpp"

#include <functional>
#include <optional>
#include <vector>

namespace cg {

// Features extracted from a schedule for the learned cost model.
struct ScheduleFeatures {
    double m_tile = 0;
    double n_tile = 0;
    double k_tile = 0;
    double vector_width = 0;
    double unroll_factor = 0;
    double num_parallel_axes = 0;
    double uses_shared_memory = 0;
    double uses_tensor_core = 0;
};

// Extract features from a schedule.
ScheduleFeatures extract_features(const Schedule& s);

// Convert features to a vector for the GP.
std::vector<double> features_to_vector(const ScheduleFeatures& f);

// A Gaussian Process with a squared-exponential kernel.
class GaussianProcess {
public:
    GaussianProcess() = default;

    // Add an observation.
    void observe(const std::vector<double>& features, double runtime);

    // Predict mean and variance at a new point.
    struct Prediction { double mean; double variance; };
    Prediction predict(const std::vector<double>& features) const;

    // Expected Improvement at a point, given the current best observed value.
    double expected_improvement(const std::vector<double>& features,
                                 double best_observed) const;

    usize num_observations() const { return observations_.size(); }

    // Hyperparameters.
    double length_scale = 10.0;
    double signal_variance = 1.0;
    double noise_variance = 1e-4;

private:
    struct Observation {
        std::vector<double> features;
        double runtime;
    };
    std::vector<Observation> observations_;

    double kernel(const std::vector<double>& a, const std::vector<double>& b) const;
};

// A benchmark function: given a schedule, return its measured runtime.
using BenchmarkFn = std::function<double(const Schedule&)>;

// The result of an autotuning run.
struct AutotuneResult {
    Schedule best_schedule;
    double best_runtime = std::numeric_limits<double>::infinity();
    std::vector<double> runtime_history;
    usize total_benchmarks = 0;
};

// Run Bayesian optimization over a ScheduleSpace.
AutotuneResult bayesian_autotune(
    const ScheduleSpace& space,
    BenchmarkFn benchmark,
    usize max_benchmarks = 20,
    usize initial_random = 3);

} // namespace cg
