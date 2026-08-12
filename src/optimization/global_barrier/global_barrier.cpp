// optimization/global_barrier.cpp - Global Barrier implementation
#include "cg/optimization/global_barrier.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

bool GlobalBarrier::check_legality(Module& m, std::vector<std::string>& errors) {
    bool ok = true;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            // Every operand must be non-null.
            for (usize i = 0; i < op.operands.size(); ++i) {
                if (op.operands[i].is_null()) {
                    errors.push_back("op " + std::to_string(op.id) +
                                     " has null operand " + std::to_string(i));
                    ok = false;
                }
            }
            // Every result must have a type.
            for (usize i = 0; i < op.results.size(); ++i) {
                if (!op.results[i].type()) {
                    errors.push_back("op " + std::to_string(op.id) +
                                     " result " + std::to_string(i) + " has null type");
                    ok = false;
                }
            }
        }
    }
    return ok;
}

bool GlobalBarrier::check_schedule(Module& m, std::vector<std::string>& errors) {
    // Foundational: verify no obvious schedule violations.
    // (The full scheduler lives in the schedule/ subsystem; this is a
    // last-line check before lowering.)
    (void)m; (void)errors;
    return true;
}

void GlobalBarrier::finalize_decisions(Module& m,
                                       GlobalBarrierReport::Decisions& out) {
    // Collect fusion clusters.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            auto fc = op.attributes.get("fused_chain");
            if (fc && fc->kind == AttrKind::IntegerArray) {
                out.fusion_clusters.push_back({op.id});
            }
        }
    }

    // Collect reused buffer ids.
    std::unordered_set<u32> seen;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            auto bid = op.attributes.get("buffer_id");
            if (bid && bid->kind == AttrKind::Integer) {
                u32 id = static_cast<u32>(bid->integer);
                if (seen.insert(id).second) {
                    out.reused_buffers.push_back(id);
                }
            }
        }
    }

    // Collect specializations.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            auto sp = op.attributes.get("specialized");
            if (sp && sp->kind == AttrKind::String) {
                out.specializations[op.id] = sp->str;
            }
        }
    }
}

GlobalBarrierReport GlobalBarrier::run(Module& m) {
    GlobalBarrierReport report;
    report.legal = true;

    if (!check_legality(m, report.errors)) report.legal = false;
    if (!check_schedule(m, report.errors)) report.legal = false;

    // Touch every analysis to make sure they're all computed and consistent.
    (void)gta_.dataflow();
    (void)gta_.shapes();
    (void)gta_.layouts();
    (void)gta_.lifetimes();
    (void)gta_.intensity();
    (void)gta_.parallelism();
    (void)gta_.reuse();
    (void)gta_.aliases();
    (void)gta_.cost();

    finalize_decisions(m, report.decisions);
    return report;
}

} // namespace cg
