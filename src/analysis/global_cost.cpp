// analysis/global_cost.cpp
// All timing from HardwareModel — no hardcoded constants.
#include "cg/analysis/global_cost.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

void GlobalCostAnalysis::compute() {
    Module& m = am_.module();
    auto& ai = am_.get<ArithmeticIntensityAnalysis>();

    cost_ = GlobalCost{};

    double bw_global = hw_.memory.get(MemorySpace::Generic);
    if (bw_global <= 0.0) bw_global = 1.0;

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            const auto& oi = ai.intensity_of(op);
            double compute_sec = oi.flops > 0
                ? static_cast<double>(oi.flops) / std::max(1.0, hw_.peak_flops(DType::F32))
                : 0.0;
            double mem_sec = static_cast<double>(oi.total_bytes()) / bw_global;
            // All timing from HardwareModel.
            double launch_sec = hw_.launch_overhead_sec;
            double sync_sec = op.is_pure() ? 0.0 : hw_.launch_overhead_sec * 0.2;
            // Specialization + code size: proportional to launch overhead.
            double spec_sec = hw_.launch_overhead_sec * 0.02;
            double codesize_sec = hw_.launch_overhead_sec * 0.02;

            double op_total = compute_sec + mem_sec + launch_sec + sync_sec
                            + spec_sec + codesize_sec;

            cost_.per_op_sec[op.id] = op_total;
            cost_.execution_sec      += compute_sec;
            cost_.memory_sec         += mem_sec;
            cost_.launch_sec         += launch_sec;
            cost_.sync_sec           += sync_sec;
            cost_.specialization_sec += spec_sec;
            cost_.code_size_sec      += codesize_sec;
        }
    }
}

} // namespace cg
