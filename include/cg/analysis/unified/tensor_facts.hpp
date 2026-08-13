// analysis/unified/tensor_facts.hpp - the rich abstract state carried by
// every SSA value in the unified Tensor Knowledge Graph.
//
// Conceptually, every Value gets a TensorFacts object. Each field is an
// abstract domain (see abstract_domain.hpp) that supports lattice join.
// Fields are populated lazily as analyses discover facts; "unknown" is
// a legitimate state for every field.
//
// The design rule: TensorFacts carries TARGET-INDEPENDENT facts. Hardware-
// specific estimates (predicted kernel runtime, cache hit rate, occupancy)
// live in CostFacts, parameterized by a HardwareModel. The same TensorFacts
// can be queried against multiple hardware models without re-derivation.
#pragma once

#include "cg/analysis/unified/abstract_domain.hpp"
#include "cg/core/dtype.hpp"
#include "cg/ir/type.hpp"
#include "cg/ir/value.hpp"
#include "cg/layout/layout.hpp"
#include "cg/shape/dim_expr.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

// ---------------------------------------------------------------------------
// TensorFacts - the abstract state of one SSA tensor value.
//
// Every field is wrapped in Fact<T> so it carries confidence + provenance.
// Fields marked "unknown" do not contribute to optimization decisions.
// ---------------------------------------------------------------------------
struct TensorFacts {
    ValueId value_id = 0;

    // ---- Shape ----
    Fact<std::vector<Dimension>> shape;
    Fact<u32> rank;

    // ---- Type ----
    Fact<DType> dtype;
    Fact<DeviceId> device;
    Fact<MemorySpace> memory_space;
    Fact<u32> alignment_bytes;     // known minimum alignment

    // ---- Layout ----
    // LayoutPtr is the existing affine layout algebra (see layout/layout.hpp).
    // We wrap it in Fact<> to track provenance.
    Fact<LayoutPtr> layout;
    Fact<std::vector<DimExprPtr>> strides;  // per-dim stride (symbolic)
    Fact<bool> is_row_major_contiguous;
    Fact<bool> is_innermost_dim_contiguous;

    // ---- Value + structure ----
    Fact<ValueRange> value_range;
    Fact<TensorProperty> properties;
    // For Constant tensors: the constant value (if scalar-broadcast).
    Fact<double> constant_value;
    Fact<bool> constant_value_known;

    // ---- Alias + lifetime ----
    Fact<AliasClass> alias_class;
    Fact<u32> birth_op;          // op index where value is defined
    Fact<u32> death_op;          // op index of last user
    Fact<u32> num_users;
    Fact<u32> num_distinct_consumers;

    // ---- Producer / consumers ----
    Fact<u32> producer_op;       // OpId of the operation that produces this value
    Fact<std::vector<ValueId>> consumers;  // values that consume this one

    // ---- Reduction info ----
    Fact<ReductionInfo> reduction;

    // ---- Cost (target-independent building blocks) ----
    Fact<u64> estimated_flops;
    Fact<u64> estimated_bytes_read;
    Fact<u64> estimated_bytes_written;
    Fact<double> arithmetic_intensity;  // FLOPs / bytes

    // ---- Reuse ----
    Fact<u32> reuse_distance;     // ops between consecutive uses
    Fact<double> reuse_factor;    // accesses / unique elements
    Fact<CacheBehavior> cache_behavior;

    // ---- Parallelism ----
    Fact<u64> independent_items;  // e.g. output elements or tiles
    Fact<bool> has_reduction_dim;
    Fact<u64> reduction_length;

    // ---- Bounds (per-dim) ----
    // Override the per-dim bounds in `shape` if more refined info is
    // available from range analysis (e.g. tile loops).
    Fact<std::vector<Bound>> dim_bounds;
};

// ---------------------------------------------------------------------------
// GraphFacts - facts about the entire module / function graph.
//
// These drive global decisions: "is the whole graph memory-bound?",
// "what's the peak memory pressure?", "which subgraphs are reusable?".
// ---------------------------------------------------------------------------
struct GraphFacts {
    // ---- Aggregate metrics ----
    Fact<u64> total_flops;
    Fact<u64> total_bytes;
    Fact<u64> peak_memory_bytes;
    Fact<u32> kernel_count;
    Fact<double> graph_arithmetic_intensity;
    Fact<double> effective_arithmetic_intensity;  // after cache reuse
    Fact<double> roofline_ridge;                  // F32 ridge of the target

    // ---- Structure ----
    Fact<std::vector<std::pair<ValueId, ValueId>>> fusion_candidates;
    Fact<std::vector<std::pair<u32, u32>>> synchronization_points;
    Fact<std::vector<u32>> parallel_regions;
    Fact<u32> critical_path_length;  // in ops
    Fact<std::vector<std::pair<ValueId, ValueId>>> reusable_subgraphs;

    // ---- Dependence graph ----
    std::vector<DependenceEdge> dependence_edges;

    // ---- Convergence metrics (filled by UnifiedAnalyzer) ----
    u32 iterations_to_converge = 0;
    u32 facts_discovered = 0;
    double analysis_latency_sec = 0.0;
    u32 worklist_processed = 0;
};

// ---------------------------------------------------------------------------
// FusionBenefitReport - the answer to "should I fuse these two ops?"
//
// This is the unit the optimizer consumes. Note the separation of
// legality (can_fuse) from profitability (predicted_improvement) and
// the explicit confidence on every estimate.
// ---------------------------------------------------------------------------
struct FusionBenefitReport {
    bool can_fuse = false;

    // Why not? (empty if can_fuse is true)
    std::string legality_reason;

    // Estimated savings (positive = good)
    double saved_bytes = 0.0;
    double saved_kernel_launches = 0.0;
    double saved_runtime_sec = 0.0;

    // Estimated costs (negative = good when offset by savings)
    double added_register_pressure = 0.0;
    double occupancy_delta_pct = 0.0;  // negative = occupancy dropped
    double critical_path_delta_pct = 0.0;

    // Net predicted improvement (fraction; 0.178 = 17.8% faster)
    double net_predicted_improvement = 0.0;

    // Confidence on the net prediction.
    Confidence confidence = Confidence::Estimated;

    // Why? Provenance chain.
    std::vector<Provenance> reasons;
};

} // namespace cg
