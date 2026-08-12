// analysis/reuse_analysis.cpp
#include "cg/analysis/reuse_analysis.hpp"
#include "cg/analysis/arithmetic_intensity.hpp"

namespace cg {

void ReuseAnalysis::compute() {
    Module& m = am_.module();
    auto& df = am_.get<DataflowAnalysis>();
    auto& ai = am_.get<ArithmeticIntensityAnalysis>();

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            Value v = op.results[0];
            ReuseInfo info;
            info.num_consumers = df.fanout(v);
            const auto& oi = ai.intensity_of(op);
            info.flops_per_producer = oi.flops;
            info.bytes_per_producer = oi.total_bytes();

            if (info.num_consumers == 0) {
                info.decision = ReuseDecision::Materialize;
            } else if (info.num_consumers == 1) {
                info.decision = ReuseDecision::Fuse;
            } else {
                // Cheap producer (low FLOPs) and small output -> recompute.
                // Expensive producer or large output -> materialize.
                bool cheap = info.flops_per_producer < 1024;
                bool small = info.bytes_per_producer < 4 * 1024;
                if (cheap && small) {
                    info.decision = ReuseDecision::Recompute;
                    info.estimated_savings_sec = 0.0; // saves memory, costs compute
                } else {
                    info.decision = ReuseDecision::Materialize;
                }
            }
            per_value_[v.id()] = info;
        }
    }
}

} // namespace cg
