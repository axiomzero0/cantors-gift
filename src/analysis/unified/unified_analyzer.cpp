// analysis/unified/unified_analyzer.cpp - the iterative fixed-point driver.
#include "cg/analysis/unified/unified_analyzer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace cg {

u32 UnifiedAnalyzer::run_one_iteration() {
    u32 total_discovered = 0;
    for (auto& p : propagators_) {
        auto t0 = std::chrono::steady_clock::now();
        u32 found = p->run(store_);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;

        // Record per-propagator stats.
        AnalyzerMetrics::PropagatorStats* ps = nullptr;
        for (auto& s : metrics_.per_propagator) {
            if (s.name == p->name()) { ps = &s; break; }
        }
        if (!ps) {
            metrics_.per_propagator.push_back({p->name(), 0, 0, 0.0});
            ps = &metrics_.per_propagator.back();
        }
        ps->runs++;
        ps->facts_produced += found;
        ps->total_sec += sec;

        total_discovered += found;
    }
    return total_discovered;
}

const AnalyzerMetrics& UnifiedAnalyzer::run() {
    auto t_start = std::chrono::steady_clock::now();
    metrics_ = AnalyzerMetrics{};

    const u32 max_iterations = 16;  // safety bound; convergence is guaranteed
                                     // by finite-height lattices
    for (u32 iter = 0; iter < max_iterations; ++iter) {
        ++metrics_.iterations;
        u32 discovered = run_one_iteration();
        metrics_.facts_discovered += discovered;
        if (discovered == 0) break;  // fixed point
    }

    auto t_end = std::chrono::steady_clock::now();
    metrics_.latency_sec =
        std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000.0;
    metrics_.contradictions = store_.stats().contradictions;
    metrics_.worklist_processed = store_.facts_discovered();

    // Propagate metrics into the graph facts for downstream consumers.
    store_.graph_facts().iterations_to_converge = metrics_.iterations;
    store_.graph_facts().facts_discovered = metrics_.facts_discovered;
    store_.graph_facts().analysis_latency_sec = metrics_.latency_sec;
    store_.graph_facts().worklist_processed = metrics_.worklist_processed;

    return metrics_;
}

void UnifiedAnalyzer::report_actual_runtime(ValueId vid, double actual_sec) {
    actual_runtimes_[vid] = actual_sec;
    // Compute mean prediction error across all reported values.
    // (The cost model doesn't yet store per-value runtime predictions, so
    // this is a stub for the future cost-model integration. When the cost
    // model produces a Fact<double> predicted_runtime, we compare here.)
    double total_error = 0.0;
    u32 n = 0;
    for (auto& [id, actual] : actual_runtimes_) {
        // Placeholder: in a full implementation, query the predicted runtime
        // from the fact store and compute |pred - actual| / actual.
        (void)id; (void)actual;
        ++n;
    }
    if (n > 0) {
        metrics_.predictions_evaluated = n;
        metrics_.mean_prediction_error = total_error / static_cast<double>(n);
    }
}

} // namespace cg
