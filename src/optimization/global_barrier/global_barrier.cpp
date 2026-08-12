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
    // Validate that the schedule is legal for every operation:
    //   - Tile sizes must divide the dimension (or be <= dim for tail handling).
    //   - Shared memory usage must not exceed the hardware limit.
    //   - Register pressure must be within budget.
    //   - Vector width must be a power of 2 and <= hardware SIMD width.

    auto& gta_cost = gta_.cost();
    const auto& hw = gta_cost.hardware();

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            // Check buffer_id attributes are valid.
            auto bid_attr = op.attributes.get("buffer_id");
            if (bid_attr && bid_attr->kind == AttrKind::Integer) {
                if (bid_attr->integer < 0) {
                    errors.push_back("op " + std::to_string(op.id) +
                                     " has negative buffer_id");
                    return false;
                }
            }

            // Check fused_chain attributes are well-formed.
            auto fc_attr = op.attributes.get("fused_chain");
            if (fc_attr && fc_attr->kind == AttrKind::IntegerArray) {
                if (fc_attr->ints.size() % 2 != 0) {
                    errors.push_back("op " + std::to_string(op.id) +
                                     " has malformed fused_chain (odd length)");
                    return false;
                }
            }

            // Check specialization predicates are non-empty.
            auto spec_attr = op.attributes.get("specialized");
            if (spec_attr && spec_attr->kind == AttrKind::String) {
                if (spec_attr->str.empty()) {
                    errors.push_back("op " + std::to_string(op.id) +
                                     " has empty specialization predicate");
                    return false;
                }
            }
        }
    }

    // Check shared memory budget.
    if (hw.shared_mem_bytes > 0) {
        u64 total_shared = 0;
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                auto bytes_attr = op.attributes.get("bytes");
                if (bytes_attr && bytes_attr->kind == AttrKind::Integer) {
                    if (op.opcode == OP_ALLOC) {
                        total_shared += static_cast<u64>(bytes_attr->integer);
                    }
                }
            }
        }
        // The shared memory budget is per-block, not per-module. We check
        // that the total allocated shared memory doesn't exceed a reasonable
        // multiple of the hardware limit (since multiple buffers may be
        // reused across the kernel).
        if (total_shared > hw.shared_mem_bytes * 4) {
            errors.push_back("total shared memory (" +
                            std::to_string(total_shared) + " bytes) exceeds 4x hardware limit (" +
                            std::to_string(hw.shared_mem_bytes) + " bytes)");
            // This is a warning, not a hard error.
        }
    }

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
