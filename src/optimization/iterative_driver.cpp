// optimization/iterative_driver.cpp - iterative optimization driver
#include "cg/optimization/iterative_driver.hpp"
#include "cg/optimization/canonicalize/algebraic.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/const_fold/sccp.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/egraph/egraph_superoptimizer.hpp"
#include "cg/optimization/fusion/fusion.hpp"
#include "cg/optimization/layout/layout_opt.hpp"
#include "cg/optimization/memory/copy_elimination.hpp"
#include "cg/optimization/memory/memory_planning.hpp"
#include "cg/optimization/reduction/reduction_opt.hpp"
#include "cg/optimization/shape/shape_opt.hpp"
#include "cg/optimization/specialization/specialization.hpp"

namespace cg {

namespace {

usize count_ops(const Module& m) {
    usize n = 0;
    for (auto& f : m.functions()) n += f->entry()->size();
    return n;
}

} // namespace

IterativeDriverReport IterativeDriver::run(Module& m) {
    IterativeDriverReport report;
    usize prev_ops = count_ops(m);

    for (usize iter = 0; iter < opts_.max_iterations; ++iter) {
        report.iterations_run = iter + 1;

        // Phase 1: canonicalization + algebraic + SCCP + const fold + DCE.
        {
            PassManager pm;
            pm.add(std::make_unique<CanonicalizePass>());
            pm.add(std::make_unique<AlgebraicSimplificationPass>());
            pm.add(std::make_unique<SCCPPass>());
            pm.add(std::make_unique<ConstantFoldingPass>());
            pm.add(std::make_unique<CSEPass>());
            pm.add(std::make_unique<DCEPass>());
            pm.run(m, am_);
        }

        // Phase 2: shape + layout optimization.
        {
            PassManager pm;
            pm.add(std::make_unique<ShapeOptimizationPass>());
            pm.add(std::make_unique<LayoutOptimizationPass>());
            pm.run(m, am_);
        }

        // Phase 3: e-graph superoptimizer (runs before fusion so fusion
        // sees the algebraically simplified IR).
        {
            PassManager pm;
            pm.add(std::make_unique<EGraphSuperoptimizerPass>());
            pm.add(std::make_unique<DCEPass>());
            pm.run(m, am_);
        }

        // Phase 4: fusion.
        {
            PassManager pm;
            pm.add(std::make_unique<FusionPass>());
            pm.add(std::make_unique<DCEPass>());
            pm.run(m, am_);
        }

        // Phase 5: reduction optimization.
        {
            PassManager pm;
            pm.add(std::make_unique<ReductionOptimizationPass>());
            pm.run(m, am_);
        }

        // Phase 6: copy elimination + memory planning.
        {
            PassManager pm;
            pm.add(std::make_unique<CopyEliminationPass>());
            pm.add(std::make_unique<MemoryPlanningPass>());
            pm.run(m, am_);
        }

        // Phase 7: specialization.
        {
            PassManager pm;
            pm.add(std::make_unique<SpecializationPass>());
            pm.run(m, am_);
        }

        usize cur_ops = count_ops(m);
        if (cur_ops == prev_ops) {
            report.converged = true;
            break;
        }
        prev_ops = cur_ops;
    }

    // Global Barrier.
    if (opts_.run_global_barrier) {
        GlobalBarrier barrier(gta_);
        report.barrier_report = barrier.run(m);
    }

    return report;
}

} // namespace cg
