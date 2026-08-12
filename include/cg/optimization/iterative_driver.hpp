// optimization/iterative_driver.hpp - iterative optimization with GTA feedback
//
// The driver runs the optimization pipeline in phases, recomputing analyses
// between phases:
//
//   phase 1: canonicalization + algebraic + SCCP + const fold + DCE
//   phase 2: shape optimization + layout optimization
//   phase 3: fusion (uses dataflow + intensity + parallelism)
//   phase 4: reduction optimization
//   phase 5: copy elimination + memory planning
//   phase 6: specialization
//   ----- Global Barrier -----
//   phase 7: Global Tensor Analysis + final decisions
//
// Between phases, the driver may iterate (re-run earlier phases) if a later
// phase exposes a new opportunity. The iteration budget is bounded.
#pragma once

#include "cg/analysis/global_analysis.hpp"
#include "cg/ir/module.hpp"
#include "cg/optimization/global_barrier.hpp"
#include "cg/optimization/pass.hpp"

namespace cg {

struct IterativeDriverOptions {
    usize max_iterations = 3;
    bool  run_global_barrier = true;
};

struct IterativeDriverReport {
    usize iterations_run = 0;
    GlobalBarrierReport barrier_report;
    bool converged = false;
};

class IterativeDriver {
public:
    IterativeDriver(AnalysisManager& am, IterativeDriverOptions opts = {})
        : am_(am), gta_(am), opts_(std::move(opts)) {}

    // Configure the hardware used by the cost model.
    void set_hardware(HardwareModel hw) { gta_.set_hardware(std::move(hw)); }

    IterativeDriverReport run(Module& m);

private:
    AnalysisManager& am_;
    GlobalAnalysisManager gta_;
    IterativeDriverOptions opts_;
};

} // namespace cg
