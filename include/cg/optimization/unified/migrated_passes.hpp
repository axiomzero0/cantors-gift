// optimization/unified/migrated_passes.hpp - existing passes rewritten to
// consume the unified Tensor Knowledge Graph.
//
// These are NOT new passes. They are the SAME passes (CSE, DCE,
// ConstantFolding, Canonicalize, Fusion, CopyElimination, Recomputation)
// rewritten to query the FactStore instead of re-deriving facts from
// IR attributes.
//
// Why this matters: the old passes each re-implement "is this a zero
// constant?" / "is this op pure?" / "are these two ops equivalent?"
// independently, with subtly different logic. The unified versions all
// ask the FactStore, which means:
//   1. One source of truth (no drift between passes).
//   2. Richer facts available (Property lattice, ValueRange, alias_class).
//   3. Provenance + confidence on every decision.
//
// The old passes remain available for backward compatibility. The
// migrated passes are named with a `Unified` prefix so you can A/B test.
//
// ANALYZER REUSE: each pass accepts an optional `UnifiedAnalyzer*`
// via `set_shared_analyzer()`. When non-null, the pass uses that
// analyzer's FactStore instead of building + running a fresh one.
// This is critical for the pipeline: without it, N passes each spend
// ~50-200 µs re-running the analyzer on the same IR. With it, the
// analyzer runs ONCE and all passes share its facts. The pipeline
// re-runs the analyzer only when a pass reports it mutated the IR.
#pragma once

#include "cg/analysis/unified/unified_analyzer.hpp"
#include "cg/optimization/pass.hpp"

#include <memory>
#include <string>

namespace cg {

// Unified CSE: uses FactStore to identify equivalent ops.
class UnifiedCSEPass : public Pass {
public:
    std::string name() const override { return "unified_cse"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    // If set, the pass uses this analyzer's FactStore instead of
    // building a fresh one. The caller is responsible for ensuring
    // the analyzer is up-to-date with the IR.
    void set_shared_analyzer(UnifiedAnalyzer* a) { shared_ = a; }

    struct Stats {
        u32 duplicates_removed = 0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
    UnifiedAnalyzer* shared_ = nullptr;
};

// Unified DCE: uses FactStore's lifetime facts (num_users == 0).
class UnifiedDCEPass : public Pass {
public:
    std::string name() const override { return "unified_dce"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
    void set_shared_analyzer(UnifiedAnalyzer* a) { shared_ = a; }

    struct Stats {
        u32 ops_removed = 0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
    UnifiedAnalyzer* shared_ = nullptr;
};

// Unified ConstantFolding: uses FactStore's ConstantValue fact.
class UnifiedConstantFoldingPass : public Pass {
public:
    std::string name() const override { return "unified_const_fold"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
    void set_shared_analyzer(UnifiedAnalyzer* a) { shared_ = a; }

    struct Stats {
        u32 constants_folded = 0;
        u32 zero_propagations = 0;
        u32 one_propagations = 0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
    UnifiedAnalyzer* shared_ = nullptr;
};

// Unified Canonicalize: uses FactStore for zero/one detection.
class UnifiedCanonicalizePass : public Pass {
public:
    std::string name() const override { return "unified_canonicalize"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
    void set_shared_analyzer(UnifiedAnalyzer* a) { shared_ = a; }

    struct Stats {
        u32 add_zero_simplified = 0;
        u32 mul_one_simplified = 0;
        u32 mul_zero_simplified = 0;
        u32 sub_zero_simplified = 0;
        u32 transpose_pair_eliminated = 0;
        u32 commutative_reordered = 0;
        u32 total_rewrites = 0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
    UnifiedAnalyzer* shared_ = nullptr;
};

// Unified CopyElimination: uses FactStore's alias_class.
class UnifiedCopyEliminationPass : public Pass {
public:
    std::string name() const override { return "unified_copy_elim"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
    void set_shared_analyzer(UnifiedAnalyzer* a) { shared_ = a; }

    struct Stats {
        u32 transposes_eliminated = 0;
        u32 reshapes_eliminated = 0;
        u32 broadcasts_eliminated = 0;
        u32 total_rewrites = 0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
    UnifiedAnalyzer* shared_ = nullptr;
};

// Unified Recomputation: uses FactStore's reuse facts.
class UnifiedRecomputationPass : public Pass {
public:
    std::string name() const override { return "unified_recompute"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
    void set_shared_analyzer(UnifiedAnalyzer* a) { shared_ = a; }

    struct Stats {
        u32 materialize_decisions = 0;
        u32 recompute_decisions = 0;
        u32 fuse_decisions = 0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
    UnifiedAnalyzer* shared_ = nullptr;
};

// Convenience: run all migrated passes in sequence.
//
// The pipeline owns a shared UnifiedAnalyzer. The first pass builds +
// runs it; subsequent passes reuse the FactStore. When a pass mutates
// the IR, the pipeline re-runs the analyzer before the next pass so
// facts stay fresh. This cuts total analyzer invocations from N (one
// per pass) to ~1-3 (one initial + one after each IR-mutating pass).
class UnifiedPassPipeline : public Pass {
public:
    std::string name() const override { return "unified_pipeline"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

    struct Stats {
        UnifiedCSEPass::Stats cse;
        UnifiedDCEPass::Stats dce;
        UnifiedConstantFoldingPass::Stats const_fold;
        UnifiedCanonicalizePass::Stats canonicalize;
        UnifiedCopyEliminationPass::Stats copy_elim;
        UnifiedRecomputationPass::Stats recompute;
        // How many times the analyzer was actually run.
        u32 analyzer_runs = 0;
        // Total analyzer latency (sum across all runs).
        double analyzer_latency_sec = 0.0;
    };
    const Stats& stats() const { return stats_; }
private:
    Stats stats_;
};

} // namespace cg
