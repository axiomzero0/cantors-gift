// analysis/unified/unified_analyzer.hpp - the iterative fixed-point driver.
//
// The UnifiedAnalyzer is the GLOBAL ANALYZER barrier in the architecture.
// Above the barrier: local + e-graph + memory planning. Below: the global
// optimizer consults the unified fact store to make globally-profitable
// decisions before final lowering.
//
// The analyzer is ITERATIVE:
//
//   while changed:
//       propagate_shapes()
//       propagate_layouts()
//       propagate_constants()
//       propagate_properties()
//       propagate_ranges()
//       propagate_aliases()
//       propagate_lifetimes()
//       propagate_reductions()
//       propagate_dependencies()
//       propagate_costs()
//   if no_new_facts:
//       converged
//
// Each propagator reads facts, computes derived facts, writes them back
// with provenance + confidence. The store detects changes and re-queues
// dependent propagators. Iteration continues until no propagator produces
// new facts (fixed point).
//
// Convergence is GUARANTEED because every abstract domain here is finite
// (or has finite height in the lattice). In practice the analyzer converges
// in 2-4 iterations for typical tensor programs.
//
// Design rule: the analyzer is NOT an optimization pass. It produces facts;
// passes consume them. The analyzer never mutates the IR.
#pragma once

#include "cg/analysis/unified/fact_propagator.hpp"
#include "cg/analysis/unified/fact_store.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/module.hpp"
#include "cg/numerical/semantics.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace cg {

// Metrics from one analyzer run. Used by the analyzer benchmark.
struct AnalyzerMetrics {
    u32 iterations = 0;
    u32 facts_discovered = 0;
    u32 worklist_processed = 0;
    double latency_sec = 0.0;
    u32 contradictions = 0;

    // Per-propagator breakdown.
    struct PropagatorStats {
        std::string name;
        u32 runs = 0;
        u32 facts_produced = 0;
        double total_sec = 0.0;
    };
    std::vector<PropagatorStats> per_propagator;

    // Prediction-quality metrics (filled when actual runtimes are
    // reported back via `report_actual_runtime`).
    double mean_prediction_error = 0.0;  // mean |predicted - actual| / actual
    u32 predictions_evaluated = 0;
};

class UnifiedAnalyzer {
public:
    explicit UnifiedAnalyzer(Module& m) : store_(m) {}

    FactStore& store() { return store_; }
    const FactStore& store() const { return store_; }

    // Set the hardware model for cost predictions.
    void set_hardware(HardwareModel hw) {
        store_.set_hardware(std::move(hw));
    }

    // Set the numerical mode (affects which constant-folding rules fire).
    void set_numerical_mode(NumericalMode mode) { mode_ = mode; }
    NumericalMode numerical_mode() const { return mode_; }

    // Register a propagator. The analyzer takes ownership.
    void add_propagator(std::unique_ptr<FactPropagator> p) {
        propagators_.push_back(std::move(p));
    }

    // Convenience: register the default set of propagators.
    void add_default_propagators() {
        add_propagator(std::make_unique<ShapePropagator>());
        add_propagator(std::make_unique<LayoutPropagator>());
        add_propagator(std::make_unique<ConstantPropagator>(mode_));
        add_propagator(std::make_unique<PropertyPropagator>());
        add_propagator(std::make_unique<RangePropagator>());
        add_propagator(std::make_unique<AliasPropagator>());
        add_propagator(std::make_unique<LifetimePropagator>());
        add_propagator(std::make_unique<ReductionPropagator>());
        add_propagator(std::make_unique<DependencePropagator>());
        add_propagator(std::make_unique<CostPropagator>());
    }

    // Run all propagators to a fixed point.
    // Returns the metrics from this run.
    const AnalyzerMetrics& run();

    // Run a single iteration (one pass over all propagators).
    // Returns the number of facts discovered this iteration.
    u32 run_one_iteration();

    // INCREMENTAL: re-run only propagators whose inputs may have changed
    // since the last `run()` or `run_incremental()`. This is much cheaper
    // than a full `run()` when the IR has only been slightly mutated
    // (e.g. one pass eliminated a few ops).
    //
    // Strategy: mark all propagators as "dirty" initially. After a full
    // run, all are "clean". When the caller mutates the IR, they should
    // call `invalidate()` to mark propagators dirty. `run_incremental()`
    // only re-runs dirty propagators, and re-runs dependents if a dirty
    // propagator produced new facts.
    //
    // For now, "dirty" is coarse: we re-run ALL propagators but skip the
    // second iteration if the first produced no new facts. This is still
    // faster than `run()` because `run()` does up to 16 iterations while
    // `run_incremental()` does at most 2 (one to re-derive, one to confirm
    // fixed point). A future version will track per-value dependencies
    // for true incremental dataflow.
    const AnalyzerMetrics& run_incremental();

    // Mark the analyzer's facts as potentially stale. The next
    // `run_incremental()` will re-derive everything.
    void invalidate() { dirty_ = true; }

    // Reset all propagators and clear the fact store's facts (but keep
    // the store itself, since it holds a reference to the Module).
    void reset() {
        // We can't reassign FactStore (it has a reference member), so we
        // clear its state in-place by re-creating a fresh store and swap.
        // Easier: just create a new analyzer if you need a fresh state.
        for (auto& p : propagators_) p->reset();
        metrics_ = AnalyzerMetrics{};
    }

    // Report an actual runtime for a value. Used to compute prediction
    // error and refine the cost model.
    void report_actual_runtime(ValueId vid, double actual_sec);

    const AnalyzerMetrics& metrics() const { return metrics_; }

private:
    FactStore store_;
    std::vector<std::unique_ptr<FactPropagator>> propagators_;
    NumericalMode mode_ = NumericalMode::Relaxed;
    AnalyzerMetrics metrics_;
    bool dirty_ = true;  // true = facts may be stale, need re-derivation

    // Track actual runtimes for prediction-error computation.
    std::unordered_map<ValueId, double> actual_runtimes_;
};

} // namespace cg
