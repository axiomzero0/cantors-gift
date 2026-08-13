// analysis/unified/fact_store.hpp - the shared Tensor Knowledge Graph.
//
// The FactStore is the single source of truth for every fact about every
// SSA value in the module. Analyses do NOT talk to each other directly;
// they read from and write to the FactStore. The store:
//
//   - tracks provenance + confidence for every fact
//   - detects when a fact changes and re-queues dependent analyses
//   - exposes a query API for optimization passes
//
// The store is parameterized by a HardwareModel for cost queries, but
// the target-independent facts (shape, layout, properties, alias, lifetime)
// are stored WITHOUT hardware context — they describe the program, not
// the program's behavior on a specific machine.
//
// Design rule: the analysis engine should NEVER be an optimization pass.
// It is a persistent source of truth that passes query.
#pragma once

#include "cg/analysis/unified/abstract_domain.hpp"
#include "cg/analysis/unified/tensor_facts.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/module.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

// Which analysis produced (or consumes) a fact?
enum class AnalysisId : u8 {
    ShapeInference,
    LayoutInference,
    StrideInference,
    ConstantPropagation,
    PropertyPropagation,
    RangeAnalysis,
    AliasAnalysis,
    LifetimeAnalysis,
    ReductionAnalysis,
    DependenceAnalysis,
    ReuseAnalysis,
    CostAnalysis,
    ParallelismAnalysis,
    ProfileFeedback,
};

inline std::string_view analysis_name(AnalysisId a) {
    switch (a) {
        case AnalysisId::ShapeInference:        return "ShapeInference";
        case AnalysisId::LayoutInference:       return "LayoutInference";
        case AnalysisId::StrideInference:       return "StrideInference";
        case AnalysisId::ConstantPropagation:   return "ConstantPropagation";
        case AnalysisId::PropertyPropagation:   return "PropertyPropagation";
        case AnalysisId::RangeAnalysis:         return "RangeAnalysis";
        case AnalysisId::AliasAnalysis:         return "AliasAnalysis";
        case AnalysisId::LifetimeAnalysis:      return "LifetimeAnalysis";
        case AnalysisId::ReductionAnalysis:     return "ReductionAnalysis";
        case AnalysisId::DependenceAnalysis:    return "DependenceAnalysis";
        case AnalysisId::ReuseAnalysis:         return "ReuseAnalysis";
        case AnalysisId::CostAnalysis:          return "CostAnalysis";
        case AnalysisId::ParallelismAnalysis:   return "ParallelismAnalysis";
        case AnalysisId::ProfileFeedback:       return "ProfileFeedback";
    }
    return "?";
}

// A fact-update callback. When a fact for `value` changes, all registered
// dependents are notified with (value, analysis_that_changed_it).
using FactListener = std::function<void(ValueId, AnalysisId)>;

class FactStore {
public:
    explicit FactStore(Module& m) : module_(m) {}

    // ---- Tensor-fact access ----

    // Get (lazily creating) the TensorFacts for a value.
    TensorFacts& facts_for(ValueId vid) {
        auto it = tensor_facts_.find(vid);
        if (it != tensor_facts_.end()) return it->second;
        TensorFacts tf;
        tf.value_id = vid;
        tensor_facts_[vid] = std::move(tf);
        return tensor_facts_[vid];
    }

    const TensorFacts* facts_for(ValueId vid) const {
        auto it = tensor_facts_.find(vid);
        return it != tensor_facts_.end() ? &it->second : nullptr;
    }

    // ---- Graph-fact access ----

    GraphFacts& graph_facts() { return graph_; }
    const GraphFacts& graph_facts() const { return graph_; }

    // ---- Hardware parameterization ----

    void set_hardware(HardwareModel hw) {
        hw_ = std::move(hw);
        hw_set_ = true;
        // Invalidate cost facts; they need to be recomputed for the new HW.
        invalidate_cost_facts();
    }

    const HardwareModel& hardware() const { return hw_; }
    bool has_hardware() const { return hw_set_; }

    // ---- Dependency tracking ----
    //
    // When analysis A reads a fact about value V, it should call
    // `register_dependency(V, A)`. When that fact changes, A will be
    // re-queued by the worklist driver.

    void register_dependency(ValueId vid, AnalysisId dependent) {
        dependencies_[vid].push_back(dependent);
    }

    // Returns the list of analyses that depend on facts about `vid`.
    const std::vector<AnalysisId>& dependents_of(ValueId vid) const {
        static const std::vector<AnalysisId> empty;
        auto it = dependencies_.find(vid);
        return it != dependencies_.end() ? it->second : empty;
    }

    // ---- Fact-update notification ----
    //
    // Called by an analysis after it writes a new fact. Returns the list
    // of analyses that should be re-queued.

    std::vector<AnalysisId> notify_fact_changed(ValueId vid, AnalysisId producer) {
        ++facts_discovered_;
        auto& deps = dependents_of(vid);
        std::vector<AnalysisId> out;
        out.reserve(deps.size());
        for (auto d : deps) if (d != producer) out.push_back(d);
        return out;
    }

    // ---- Convenience queries used by optimization passes ----
    //
    // These are the high-level questions passes actually want to ask.
    // They hide the lattice-join + provenance machinery behind a clean API.
    //
    // Each query returns a result with confidence + provenance so the
    // optimizer can decide whether to trust it.

    bool is_zero(ValueId vid) const;
    bool is_one(ValueId vid) const;
    bool is_identity(ValueId vid) const;
    bool is_constant(ValueId vid) const;
    bool is_non_negative(ValueId vid) const;
    bool is_strictly_positive(ValueId vid) const;
    bool is_diagonal(ValueId vid) const;
    bool is_sparse(ValueId vid) const;

    // Get the constant value (if IsConstant). Returns nullopt if unknown.
    std::optional<double> constant_value(ValueId vid) const;

    // Get the exact shape if statically known.
    std::optional<std::vector<i64>> static_shape(ValueId vid) const;

    // Get the dtype if known.
    std::optional<DType> dtype_of(ValueId vid) const;

    // Get the layout if known.
    LayoutPtr layout_of(ValueId vid) const;

    // Get the strides if known.
    std::optional<std::vector<i64>> static_strides(ValueId vid) const;

    // ---- Fusion queries (the big one) ----
    //
    // `can_fuse(producer, consumer)` returns true iff fusing is legal.
    // `fusion_benefit(producer, consumer)` returns a full report with
    // savings, costs, net predicted improvement, and confidence.

    bool can_fuse(ValueId producer, ValueId consumer) const;
    FusionBenefitReport fusion_benefit(ValueId producer, ValueId consumer) const;

    // ---- Module + iteration count ----

    Module& module() { return module_; }
    const Module& module() const { return module_; }

    usize num_tensors() const { return tensor_facts_.size(); }
    u32 facts_discovered() const { return facts_discovered_; }

    // ---- Statistics for the analyzer benchmark ----

    struct Stats {
        u32 fact_writes = 0;
        u32 fact_reads = 0;
        u32 dependency_edges = 0;
        u32 contradictions = 0;  // two Proven facts disagreed
    };

    Stats& stats() { return stats_; }
    const Stats& stats() const { return stats_; }

    // ---- Iteration over all tensor facts ----

    auto begin() { return tensor_facts_.begin(); }
    auto end()   { return tensor_facts_.end(); }
    auto begin() const { return tensor_facts_.begin(); }
    auto end()   const { return tensor_facts_.end(); }

private:
    void invalidate_cost_facts() {
        for (auto& [_, tf] : tensor_facts_) {
            tf.estimated_flops.known = false;
            tf.estimated_bytes_read.known = false;
            tf.estimated_bytes_written.known = false;
            tf.arithmetic_intensity.known = false;
            tf.cache_behavior.known = false;
        }
        graph_.total_flops.known = false;
        graph_.total_bytes.known = false;
        graph_.graph_arithmetic_intensity.known = false;
        graph_.effective_arithmetic_intensity.known = false;
        graph_.roofline_ridge.known = false;
    }

    Module& module_;
    HardwareModel hw_;
    bool hw_set_ = false;

    std::unordered_map<ValueId, TensorFacts> tensor_facts_;
    GraphFacts graph_;

    std::unordered_map<ValueId, std::vector<AnalysisId>> dependencies_;
    Stats stats_;
    u32 facts_discovered_ = 0;
};

} // namespace cg
