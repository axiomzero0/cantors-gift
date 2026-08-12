// analysis/global_analysis.hpp - Global Tensor Analysis (GTA) facade
//
// The GlobalAnalysisManager bundles every analysis the global optimization
// phase needs, and exposes them through a single typed access point:
//
//   GlobalAnalysisManager gta(am);
//   auto& df = gta.dataflow();
//   auto& lt = gta.lifetimes();
//   auto& ai = gta.intensity();
//   auto& gc = gta.cost();
//   ...
//
// Each access lazily computes & caches the analysis via the underlying
// AnalysisManager. When a pass invalidates analyses, the next GTA access
// recomputes them transparently.
//
// This is the "GLOBAL ANALYSIS" barrier in the architecture: above this
// barrier, the compiler runs local + e-graph + memory planning. Below the
// barrier, the global optimizer consults GTA to make *globally-profitable*
// decisions before final lowering.
#pragma once

#include "cg/analysis/alias_analysis.hpp"
#include "cg/analysis/analysis.hpp"
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/analysis/global_alias_analysis.hpp"
#include "cg/analysis/global_cost.hpp"
#include "cg/analysis/layout_analysis.hpp"
#include "cg/analysis/lifetime_analysis.hpp"
#include "cg/analysis/parallelism_analysis.hpp"
#include "cg/analysis/reuse_analysis.hpp"
#include "cg/analysis/shape_analysis.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/module.hpp"

namespace cg {

class GlobalAnalysisManager {
public:
    explicit GlobalAnalysisManager(AnalysisManager& am) : am_(am) {}

    AnalysisManager& underlying() { return am_; }
    const AnalysisManager& underlying() const { return am_; }

    // ---- Accessors (lazily compute + cache via AnalysisManager) ----
    DataflowAnalysis&             dataflow()  { return am_.get<DataflowAnalysis>(); }
    ShapeAnalysis&                shapes()    { return am_.get<ShapeAnalysis>(); }
    LayoutAnalysis&               layouts()   { return am_.get<LayoutAnalysis>(); }
    LifetimeAnalysis&             lifetimes() { return am_.get<LifetimeAnalysis>(); }
    ArithmeticIntensityAnalysis&  intensity() { return am_.get<ArithmeticIntensityAnalysis>(); }
    ParallelismAnalysis&          parallelism() { return am_.get<ParallelismAnalysis>(); }
    ReuseAnalysis&                reuse()     { return am_.get<ReuseAnalysis>(); }
    GlobalAliasAnalysis&          aliases()   { return am_.get<GlobalAliasAnalysis>(); }
    GlobalCostAnalysis&           cost()      { return am_.get<GlobalCostAnalysis>(); }

    // Configure the hardware model used by the global cost analysis.
    void set_hardware(HardwareModel hw) {
        am_.get<GlobalCostAnalysis>().set_hardware(std::move(hw));
    }

    // Invalidate everything except what `pa` preserves.
    void invalidate(const PreservedAnalyses& pa) {
        am_.invalidate(pa);
    }

    // Hard reset.
    void clear() { am_.clear(); }

private:
    AnalysisManager& am_;
};

} // namespace cg
