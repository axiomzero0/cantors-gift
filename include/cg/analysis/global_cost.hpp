// analysis/global_cost.hpp - global cost model
//
// Combines per-op intensity, lifetime pressure, fanout, layout, and hardware
// profile to produce a graph-wide cost. This is the cost function that the
// global optimizer minimizes.
//
// The cost is decomposed so that individual transformations (fusion, layout
// changes, recomputation, scheduling) can attribute their impact:
//
//   total_cost = execution_cost
//              + memory_cost
//              + launch_cost
//              + synchronization_cost
//              + specialization_cost
//              + code_size_cost
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/analysis/lifetime_analysis.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/module.hpp"

#include <unordered_map>

namespace cg {

struct GlobalCost {
    double execution_sec        = 0.0;
    double memory_sec           = 0.0;
    double launch_sec           = 0.0;
    double sync_sec             = 0.0;
    double specialization_sec   = 0.0;
    double code_size_sec        = 0.0;

    // Per-op contribution (keyed by op id).
    std::unordered_map<u32, double> per_op_sec;

    double total() const {
        return execution_sec + memory_sec + launch_sec + sync_sec
             + specialization_sec + code_size_sec;
    }
};

class GlobalCostAnalysis : public AnalysisBase {
public:
    explicit GlobalCostAnalysis(AnalysisManager& am)
        : am_(am), hw_(HardwareModel::generic_cpu()) { compute(); }

    void set_hardware(HardwareModel hw) {
        hw_ = std::move(hw);
        compute();
    }

    const HardwareModel& hardware() const { return hw_; }
    const GlobalCost& cost() const { return cost_; }

    // Cost of a single operation in seconds.
    double op_cost(u32 op_id) const {
        auto it = cost_.per_op_sec.find(op_id);
        return it != cost_.per_op_sec.end() ? it->second : 0.0;
    }

    void invalidate() { compute(); }

private:
    void compute();

    AnalysisManager& am_;
    HardwareModel hw_;
    GlobalCost cost_;
};

} // namespace cg
