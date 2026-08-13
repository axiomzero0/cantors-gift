// optimization/iterative_driver.cpp - iterative optimization driver
//
// The IterativeDriver runs the full optimization pipeline and then lowers
// the optimized Tensor IR to Codegen IR, connecting the two halves of the
// compiler. The resulting CGModule can be compiled by any backend.
//
// The pipeline now uses the unified Tensor Knowledge Graph as the primary
// analysis layer. The old canonicalize/constfold/cse/dce cluster is
// replaced by UnifiedPassPipeline (which shares a single analyzer run
// across all six migrated passes). The new exploitation passes
// (ReductionTreeSynthesis, CachePlacement, DeadStoreElimination,
// ConstantTensorMaterialization) run as Phase 5.5 to exploit facts the
// old passes don't compute.
#include "cg/optimization/iterative_driver.hpp"
#include "cg/optimization/canonicalize/algebraic.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/const_fold/sccp.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/egraph/egraph_superoptimizer.hpp"
#include "cg/optimization/fusion/fusion.hpp"
#include "cg/optimization/graph/graph_pattern.hpp"
#include "cg/optimization/layout/layout_opt.hpp"
#include "cg/optimization/memory/copy_elimination.hpp"
#include "cg/optimization/memory/memory_planning.hpp"
#include "cg/optimization/recomputation/recomputation.hpp"
#include "cg/optimization/reduction/reduction_opt.hpp"
#include "cg/optimization/shape/shape_opt.hpp"
#include "cg/optimization/specialization/specialization.hpp"
#include "cg/optimization/unified/migrated_passes.hpp"
#include "cg/optimization/unified/unified_passes.hpp"

namespace cg {

namespace {

usize count_optimizable_ops(const Module& m) {
    usize n = 0;
    for (auto& f : m.functions())
        for (auto& op : *f->entry())
            if (op.opcode != OP_ALLOC && op.opcode != OP_FREE)
                ++n;
    return n;
}

} // namespace

IterativeDriverReport IterativeDriver::run(Module& m) {
    IterativeDriverReport report;
    usize prev_ops = count_optimizable_ops(m);

    for (usize iter = 0; iter < opts_.max_iterations; ++iter) {
        report.iterations_run = iter + 1;

        // Phase 1: unified migration pipeline (replaces old canonicalize +
        // constfold + cse + dce + copy_elim + recompute cluster).
        // This shares a single analyzer run across all six migrated passes
        // via the shared-analyzer optimization (~3-4x faster than running
        // each pass with its own analyzer).
        {
            PassManager pm;
            pm.add(std::make_unique<UnifiedPassPipeline>());
            pm.run(m, am_);
        }

        // Phase 1b: legacy SCCP + algebraic (not yet migrated). These run
        // after the unified cluster so they see canonicalized IR.
        {
            PassManager pm;
            pm.add(std::make_unique<AlgebraicSimplificationPass>());
            pm.add(std::make_unique<SCCPPass>());
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

        // Phase 3: e-graph superoptimizer.
        {
            PassManager pm;
            pm.add(std::make_unique<EGraphSuperoptimizerPass>());
            pm.add(std::make_unique<DCEPass>());
            pm.run(m, am_);
        }

        // Phase 4: graph pattern matching (before fusion so fusion sees patterns).
        {
            PassManager pm;
            pm.add(std::make_unique<GraphPatternMatchingPass>());
            pm.run(m, am_);
        }

        // Phase 5: fusion.
        {
            PassManager pm;
            pm.add(std::make_unique<FusionPass>());
            pm.add(std::make_unique<DCEPass>());
            pm.run(m, am_);
        }

        // Phase 5b: unified exploitation passes. These consume the
        // unified fact store to exploit facts the old passes don't use:
        //   - ReductionTreeSynthesis: chooses tree/warp/block/hierarchical
        //     based on ReductionInfo (axes, associativity, identity).
        //   - CachePlacement: annotates tensors with register/shared/L2/global
        //     based on CacheBehavior (l2_hit_rate, reuse_factor).
        //   - DeadStoreElimination: removes stores with zero readers.
        //   - ConstantTensorMaterialization: replaces large Zero/One/Identity
        //     constants with symbolic markers (avoids materializing MB).
        {
            PassManager pm;
            pm.add(std::make_unique<ReductionTreeSynthesis>());
            pm.add(std::make_unique<CachePlacement>());
            pm.add(std::make_unique<DeadStoreElimination>());
            pm.add(std::make_unique<ConstantTensorMaterialization>());
            pm.run(m, am_);
        }

        // Phase 6: reduction optimization (legacy; runs after tree synthesis
        // so it can read the reduction_strategy attribute).
        {
            PassManager pm;
            pm.add(std::make_unique<ReductionOptimizationPass>());
            pm.run(m, am_);
        }

        // Phase 7: recomputation (before memory planning).
        {
            PassManager pm;
            pm.add(std::make_unique<RecomputationPass>());
            pm.run(m, am_);
        }

        // Phase 8: copy elimination + memory planning.
        {
            PassManager pm;
            pm.add(std::make_unique<CopyEliminationPass>());
            pm.add(std::make_unique<MemoryPlanningPass>());
            pm.run(m, am_);
        }

        // Phase 9: specialization.
        {
            PassManager pm;
            pm.add(std::make_unique<SpecializationPass>());
            pm.run(m, am_);
        }

        usize cur_ops = count_optimizable_ops(m);
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
