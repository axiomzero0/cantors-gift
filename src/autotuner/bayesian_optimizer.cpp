// autotuner/bayesian_optimizer.cpp - GP + EI implementation
#include "cg/autotuner/bayesian_optimizer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

namespace cg {

// Normal CDF (Phi).
static double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Standard normal PDF.
static double normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

ScheduleFeatures extract_features(const Schedule& s) {
    ScheduleFeatures f;
    f.num_parallel_axes = 0;
    f.uses_shared_memory = 0;
    f.uses_tensor_core = 0;
    f.vector_width = 1;
    f.unroll_factor = 1;

    for (const auto& t : s.transforms()) {
        switch (t.kind) {
            case TransformKind::Tile:
                if (t.dim == "m") f.m_tile = static_cast<double>(t.factor);
                else if (t.dim == "n") f.n_tile = static_cast<double>(t.factor);
                else if (t.dim == "k") f.k_tile = static_cast<double>(t.factor);
                break;
            case TransformKind::Vectorize:
                f.vector_width = static_cast<double>(t.factor);
                break;
            case TransformKind::Unroll:
                f.unroll_factor = static_cast<double>(t.factor);
                break;
            case TransformKind::Parallelize:
                f.num_parallel_axes += 1.0;
                break;
            case TransformKind::Cache:
                if (t.mem == MemorySpace::Shared) f.uses_shared_memory = 1.0;
                break;
            case TransformKind::Bind:
                if (t.target == "tensor_core") f.uses_tensor_core = 1.0;
                break;
            default: break;
        }
    }
    return f;
}

std::vector<double> features_to_vector(const ScheduleFeatures& f) {
    return {f.m_tile, f.n_tile, f.k_tile, f.vector_width, f.unroll_factor,
            f.num_parallel_axes, f.uses_shared_memory, f.uses_tensor_core};
}

double GaussianProcess::kernel(const std::vector<double>& a,
                                const std::vector<double>& b) const {
    // Squared-exponential (RBF) kernel.
    double sum_sq = 0.0;
    usize n = std::min(a.size(), b.size());
    for (usize i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum_sq += d * d;
    }
    return signal_variance * std::exp(-0.5 * sum_sq / (length_scale * length_scale));
}

void GaussianProcess::observe(const std::vector<double>& features, double runtime) {
    observations_.push_back({features, runtime});
}

GaussianProcess::Prediction GaussianProcess::predict(
    const std::vector<double>& features) const {
    if (observations_.empty()) {
        return {0.0, signal_variance};
    }

    usize n = observations_.size();

    // Build K (n x n) and k_star (n x 1).
    std::vector<double> k_star(n);
    for (usize i = 0; i < n; ++i) {
        k_star[i] = kernel(observations_[i].features, features);
    }

    // Build K matrix with noise.
    std::vector<std::vector<double>> K(n, std::vector<double>(n));
    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < n; ++j) {
            K[i][j] = kernel(observations_[i].features, observations_[j].features);
            if (i == j) K[i][j] += noise_variance;
        }
    }

    // Solve K * alpha = y via Gaussian elimination with partial pivoting.
    // y = observations runtimes.
    std::vector<double> y(n);
    for (usize i = 0; i < n; ++i) y[i] = observations_[i].runtime;

    // Augmented matrix [K | y].
    for (usize i = 0; i < n; ++i) K[i].push_back(y[i]);

    // Forward elimination.
    for (usize i = 0; i < n; ++i) {
        // Partial pivot.
        usize pivot = i;
        for (usize k = i + 1; k < n; ++k) {
            if (std::abs(K[k][i]) > std::abs(K[pivot][i])) pivot = k;
        }
        if (pivot != i) std::swap(K[i], K[pivot]);

        if (std::abs(K[i][i]) < 1e-12) continue; // singular

        for (usize k = i + 1; k < n; ++k) {
            double factor = K[k][i] / K[i][i];
            for (usize j = i; j <= n; ++j) {
                K[k][j] -= factor * K[i][j];
            }
        }
    }

    // Back substitution.
    std::vector<double> alpha(n, 0.0);
    for (isize i = static_cast<isize>(n) - 1; i >= 0; --i) {
        alpha[i] = K[i][n];
        for (usize j = i + 1; j < n; ++j) {
            alpha[i] -= K[i][j] * alpha[j];
        }
        if (std::abs(K[i][i]) > 1e-12)
            alpha[i] /= K[i][i];
    }

    // Mean prediction: k_star^T * alpha.
    double mean = 0.0;
    for (usize i = 0; i < n; ++i) mean += k_star[i] * alpha[i];

    // Variance: k(x,x) - k_star^T * K^{-1} * k_star
    // We already have alpha = K^{-1} * y, but we need K^{-1} * k_star.
    // Solve K * z = k_star.
    std::vector<std::vector<double>> K2(n, std::vector<double>(n + 1));
    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < n; ++j) K2[i][j] = kernel(observations_[i].features, observations_[j].features);
        if (i < n) K2[i][i] += noise_variance;
        K2[i][n] = k_star[i];
    }
    // Gaussian elimination.
    for (usize i = 0; i < n; ++i) {
        usize pivot = i;
        for (usize k = i + 1; k < n; ++k) {
            if (std::abs(K2[k][i]) > std::abs(K2[pivot][i])) pivot = k;
        }
        if (pivot != i) std::swap(K2[i], K2[pivot]);
        if (std::abs(K2[i][i]) < 1e-12) continue;
        for (usize k = i + 1; k < n; ++k) {
            double factor = K2[k][i] / K2[i][i];
            for (usize j = i; j <= n; ++j) K2[k][j] -= factor * K2[i][j];
        }
    }
    std::vector<double> z(n, 0.0);
    for (isize i = static_cast<isize>(n) - 1; i >= 0; --i) {
        z[i] = K2[i][n];
        for (usize j = i + 1; j < n; ++j) z[i] -= K2[i][j] * z[j];
        if (std::abs(K2[i][i]) > 1e-12) z[i] /= K2[i][i];
    }

    double k_xx = kernel(features, features);
    double variance = k_xx;
    for (usize i = 0; i < n; ++i) variance -= k_star[i] * z[i];
    if (variance < 0.0) variance = 0.0;

    return {mean, variance};
}

double GaussianProcess::expected_improvement(
    const std::vector<double>& features, double best_observed) const {
    auto pred = predict(features);
    if (pred.variance < 1e-12) return 0.0;

    // EI = (best - mean) * Phi(Z) + sigma * phi(Z)
    // where Z = (best - mean) / sigma
    // For minimization: improvement = max(0, best - mean)
    double sigma = std::sqrt(pred.variance);
    double delta = best_observed - pred.mean;
    double Z = delta / sigma;

    return delta * normal_cdf(Z) + sigma * normal_pdf(Z);
}

AutotuneResult bayesian_autotune(
    const ScheduleSpace& space,
    BenchmarkFn benchmark,
    usize max_benchmarks,
    usize initial_random) {
    AutotuneResult result;
    if (space.size() == 0) return result;

    GaussianProcess gp;
    std::mt19937 rng(42);

    // Track which schedules have been benchmarked.
    std::vector<bool> tested(space.size(), false);
    std::vector<double> runtimes(space.size(), 0.0);

    double best = std::numeric_limits<double>::infinity();
    usize best_idx = 0;

    // Initial random samples.
    usize n = std::min(initial_random, space.size());
    std::vector<usize> indices(space.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    usize benchmarked = 0;
    for (usize i = 0; i < n && benchmarked < max_benchmarks; ++i) {
        usize idx = indices[i];
        double rt = benchmark(space.schedules()[idx]);
        tested[idx] = true;
        runtimes[idx] = rt;
        gp.observe(features_to_vector(extract_features(space.schedules()[idx])), rt);
        if (rt < best) { best = rt; best_idx = idx; }
        result.runtime_history.push_back(best);
        ++benchmarked;
    }

    // GP-based selection.
    while (benchmarked < max_benchmarks) {
        // Find the untested candidate with the highest EI.
        double best_ei = -1.0;
        usize best_ei_idx = 0;
        for (usize i = 0; i < space.size(); ++i) {
            if (tested[i]) continue;
            double ei = gp.expected_improvement(
                features_to_vector(extract_features(space.schedules()[i])), best);
            if (ei > best_ei) {
                best_ei = ei;
                best_ei_idx = i;
            }
        }

        if (best_ei < 1e-10) {
            // No more improvement expected; fall back to random.
            for (usize i = 0; i < space.size(); ++i) {
                if (!tested[i]) { best_ei_idx = i; break; }
            }
        }

        // Benchmark the selected candidate.
        double rt = benchmark(space.schedules()[best_ei_idx]);
        tested[best_ei_idx] = true;
        runtimes[best_ei_idx] = rt;
        gp.observe(features_to_vector(extract_features(space.schedules()[best_ei_idx])), rt);
        if (rt < best) { best = rt; best_idx = best_ei_idx; }
        result.runtime_history.push_back(best);
        ++benchmarked;
    }

    result.best_schedule = space.schedules()[best_idx];
    result.best_runtime = best;
    result.total_benchmarks = benchmarked;
    return result;
}

} // namespace cg
