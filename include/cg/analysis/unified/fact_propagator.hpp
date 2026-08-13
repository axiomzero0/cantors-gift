// analysis/unified/fact_propagator.hpp - abstract base + concrete propagators.
//
// A FactPropagator is a single analysis that:
//   1. Reads facts from the FactStore for its operands.
//   2. Computes derived facts.
//   3. Writes them back to the FactStore (with provenance + confidence).
//   4. Registers dependencies so it will be re-run when inputs change.
//
// The UnifiedAnalyzer runs all propagators in a worklist until no new
// facts are produced (fixed point).
//
// Design rule: propagators must be IDEMPOTENT. Running the same propagator
// twice with the same inputs must produce the same output. This is what
// makes fixed-point iteration safe.
#pragma once

#include "cg/analysis/unified/abstract_domain.hpp"
#include "cg/analysis/unified/fact_store.hpp"
#include "cg/ir/module.hpp"
#include "cg/numerical/semantics.hpp"

#include <string>
#include <vector>

namespace cg {

class FactPropagator {
public:
    virtual ~FactPropagator() = default;

    // Unique name for debugging / metrics.
    virtual std::string name() const = 0;

    // The analysis id (used for dependency tracking).
    virtual AnalysisId id() const = 0;

    // Run the propagator over the entire module.
    // Returns the number of facts that were newly discovered or refined.
    virtual u32 run(FactStore& store) = 0;

    // Reset internal state. Called by UnifiedAnalyzer between iterations
    // if needed (most propagators don't need this since they re-derive
    // everything from the FactStore).
    virtual void reset() {}
};

// ---------------------------------------------------------------------------
// ShapePropagator
//
// Derives shape + rank facts from the IR. This is the foundational
// propagator — almost every other propagator depends on shape facts.
// ---------------------------------------------------------------------------
class ShapePropagator : public FactPropagator {
public:
    std::string name() const override { return "ShapePropagator"; }
    AnalysisId id() const override { return AnalysisId::ShapeInference; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// LayoutPropagator
//
// Propagates layout + strides + contiguity through the graph.
// Computes stride facts from the layout (symbolic strides when the shape
// is symbolic).
// ---------------------------------------------------------------------------
class LayoutPropagator : public FactPropagator {
public:
    std::string name() const override { return "LayoutPropagator"; }
    AnalysisId id() const override { return AnalysisId::LayoutInference; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// ConstantPropagator
//
// Propagates constant values through the graph. Recognizes:
//   - OP_CONSTANT results -> set ConstantValue + Constant property
//   - add(x, 0) -> x  (if 0 is provable)
//   - mul(x, 1) -> x  (if 1 is provable)
//   - mul(x, 0) -> 0  (FastMath only)
//   - add(0, x) -> x
//   - sub(x, x) -> 0
//   - neg(neg(x)) -> x
// ---------------------------------------------------------------------------
class ConstantPropagator : public FactPropagator {
public:
    explicit ConstantPropagator(NumericalMode mode = NumericalMode::Relaxed)
        : mode_(mode) {}

    std::string name() const override { return "ConstantPropagator"; }
    AnalysisId id() const override { return AnalysisId::ConstantPropagation; }
    u32 run(FactStore& store) override;

private:
    NumericalMode mode_;
};

// ---------------------------------------------------------------------------
// PropertyPropagator
//
// Propagates tensor-structure properties (Zero, One, Identity, Diagonal,
// Sparse, Symmetric, Permutation) through operations.
//
//   - matmul(A, Identity) -> A   (Identity property transfers to result)
//   - matmul(Identity, A) -> A
//   - add(Zero, x) -> x
//   - mul(One, x) -> x
//   - mul(Zero, x) -> Zero
//   - transpose(Symmetric) -> Symmetric
//   - transpose(Diagonal) -> Diagonal
// ---------------------------------------------------------------------------
class PropertyPropagator : public FactPropagator {
public:
    std::string name() const override { return "PropertyPropagator"; }
    AnalysisId id() const override { return AnalysisId::PropertyPropagation; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// RangePropagator
//
// Propagates ValueRange through operations.
//
//   - relu(x) -> x if x is non-negative
//   - abs(x)  -> x if x is non-negative
//   - square(x) -> non-negative
//   - exp(x) -> strictly positive
//   - sigmoid(x) -> [0, 1]
//   - tanh(x) -> [-1, 1]
// ---------------------------------------------------------------------------
class RangePropagator : public FactPropagator {
public:
    std::string name() const override { return "RangePropagator"; }
    AnalysisId id() const override { return AnalysisId::RangeAnalysis; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// AliasPropagator
//
// Computes alias classes. Pure SSA values start with NoAlias. Views/slices
// of a tensor inherit the parent's alias set (MustAlias with the parent).
// ---------------------------------------------------------------------------
class AliasPropagator : public FactPropagator {
public:
    std::string name() const override { return "AliasPropagator"; }
    AnalysisId id() const override { return AnalysisId::AliasAnalysis; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// LifetimePropagator
//
// Computes birth/death op indices and num_users for each value.
// Depends on DataflowAnalysis for topological order.
// ---------------------------------------------------------------------------
class LifetimePropagator : public FactPropagator {
public:
    std::string name() const override { return "LifetimePropagator"; }
    AnalysisId id() const override { return AnalysisId::LifetimeAnalysis; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// ReductionPropagator
//
// Identifies reduction operations and their axes, associativity,
// commutativity, and identity element.
// ---------------------------------------------------------------------------
class ReductionPropagator : public FactPropagator {
public:
    std::string name() const override { return "ReductionPropagator"; }
    AnalysisId id() const override { return AnalysisId::ReductionAnalysis; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// CostPropagator
//
// Computes target-independent cost building blocks (FLOPs, bytes read,
// bytes written, arithmetic intensity) and target-dependent predictions
// (cache hit rate, predicted runtime, occupancy).
//
// Target-independent facts are stored with Confidence::Proven (they
// follow from shape + dtype). Target-dependent predictions are stored
// with Confidence::Estimated.
// ---------------------------------------------------------------------------
class CostPropagator : public FactPropagator {
public:
    std::string name() const override { return "CostPropagator"; }
    AnalysisId id() const override { return AnalysisId::CostAnalysis; }
    u32 run(FactStore& store) override;
};

// ---------------------------------------------------------------------------
// DependencePropagator
//
// Builds the tensor dependence graph (producer -> consumer edges with
// DependenceKind annotations). Also computes reuse_distance + reuse_factor.
// ---------------------------------------------------------------------------
class DependencePropagator : public FactPropagator {
public:
    std::string name() const override { return "DependencePropagator"; }
    AnalysisId id() const override { return AnalysisId::DependenceAnalysis; }
    u32 run(FactStore& store) override;
};

} // namespace cg
