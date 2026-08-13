// optimization/unified/unified_passes.hpp - optimization passes that
// EXPLOIT the unified Tensor Knowledge Graph.
//
// These passes do NOT reimplement analysis logic. They query the
// FactStore for what's true about the IR and decide what to do based
// on Proven / Derived / Estimated facts. This is the GCC model:
// analyses produce facts, passes consume them.
//
// Passes in this file:
//
//   PropertyDrivenSimplification
//     - Uses TensorProperty lattice (Zero, One, Identity, Constant)
//     - mul(x, Zero) -> Zero         (Proven)
//     - mul(x, One)  -> x            (Proven)
//     - add(x, Zero) -> x            (Proven)
//     - matmul(A, Identity) -> A     (Proven)
//     - matmul(Identity, B) -> B     (Proven)
//
//   RangeDrivenStrengthReduction
//     - Uses ValueRange facts
//     - relu(x) -> x   when x >= 0   (Proven)
//     - abs(x)  -> x   when x >= 0   (Proven)
//     - sigmoid(x) -> bounded to [0,1] (annotation only)
//
//   CostGuidedFusion
//     - Uses fusion_benefit() with explicit confidence threshold
//     - Only fuses when net_predicted_improvement > threshold
//     - Only fuses when confidence is at least Estimated
//     - Logs WHY fusion was accepted/rejected (provenance chain)
//
//   LayoutAwareCopyElimination
//     - Uses alias_class + layout to eliminate redundant transposes
//     - transpose(transpose(x)) -> x          (Pure)
//     - transpose(reshape(transpose(x), ...)) -> folded layout (LayoutAware)
//
//   AliasAwareMemoryPlanning
//     - Uses AliasClass to share storage between MustAlias buffers
//     - Buffers with the same alias_set_id can share allocation
//
// All passes are PASSes (derive from cg::Pass) so they fit into the
// existing PassManager. They run a UnifiedAnalyzer internally to
// refresh facts before consuming them.
#pragma once

#include "cg/analysis/unified/fact_store.hpp"
#include "cg/analysis/unified/unified_analyzer.hpp"
#include "cg/optimization/pass.hpp"

#include <string>

namespace cg {

// ---------------------------------------------------------------------------
// Pass 1: Property-driven simplification (Zero / One / Identity / Constant)
// ---------------------------------------------------------------------------
class PropertyDrivenSimplification : public Pass {
public:
    std::string name() const override {
        return "PropertyDrivenSimplification";
    }

    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    // Statistics from the last run.
    struct Stats {
        u32 mul_zero_eliminated = 0;
        u32 mul_one_eliminated = 0;
        u32 add_zero_eliminated = 0;
        u32 matmul_identity_eliminated = 0;
        u32 total_rewrites = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_;
};

// ---------------------------------------------------------------------------
// Pass 2: Range-driven strength reduction (relu(x) -> x when x >= 0)
// ---------------------------------------------------------------------------
class RangeDrivenStrengthReduction : public Pass {
public:
    std::string name() const override {
        return "RangeDrivenStrengthReduction";
    }

    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    struct Stats {
        u32 relu_eliminated = 0;
        u32 abs_eliminated = 0;
        u32 total_rewrites = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_;
};

// ---------------------------------------------------------------------------
// Pass 3: Cost-guided fusion (uses fusion_benefit() with confidence)
// ---------------------------------------------------------------------------
class CostGuidedFusion : public Pass {
public:
    explicit CostGuidedFusion(double min_improvement = 0.05,
                               Confidence min_confidence = Confidence::Estimated)
        : min_improvement_(min_improvement),
          min_confidence_(min_confidence) {}

    std::string name() const override { return "CostGuidedFusion"; }

    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    struct Stats {
        u32 fusions_accepted = 0;
        u32 fusions_rejected_cost = 0;
        u32 fusions_rejected_confidence = 0;
        u32 fusions_rejected_legality = 0;
        double total_predicted_improvement = 0.0;
    };
    const Stats& stats() const { return stats_; }

    // The decision log lets us answer "why did you fuse / not fuse?".
    struct Decision {
        ValueId producer;
        ValueId consumer;
        bool accepted;
        std::string reason;
        double net_improvement;
        Confidence confidence;
    };
    const std::vector<Decision>& decisions() const { return decisions_; }

private:
    double min_improvement_;
    Confidence min_confidence_;
    Stats stats_;
    std::vector<Decision> decisions_;
};

// ---------------------------------------------------------------------------
// Pass 4: Layout-aware copy elimination
// ---------------------------------------------------------------------------
class LayoutAwareCopyElimination : public Pass {
public:
    std::string name() const override { return "LayoutAwareCopyElimination"; }

    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    struct Stats {
        u32 transpose_transpose_eliminated = 0;
        u32 total_rewrites = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_;
};

// ---------------------------------------------------------------------------
// Pass 5: Alias-aware memory planning
// ---------------------------------------------------------------------------
class AliasAwareMemoryPlanning : public Pass {
public:
    std::string name() const override { return "AliasAwareMemoryPlanning"; }

    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    struct Stats {
        u32 buffers_merged = 0;
        u64 bytes_saved = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_;
};

// ---------------------------------------------------------------------------
// Convenience: run all unified passes with a single call.
// ---------------------------------------------------------------------------
class UnifiedOptimizationPipeline : public Pass {
public:
    std::string name() const override { return "UnifiedOptimizationPipeline"; }

    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    // Aggregate stats from all sub-passes.
    struct Stats {
        PropertyDrivenSimplification::Stats property;
        RangeDrivenStrengthReduction::Stats range;
        CostGuidedFusion::Stats fusion;
        LayoutAwareCopyElimination::Stats layout;
        AliasAwareMemoryPlanning::Stats alias;
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_;
};

} // namespace cg
