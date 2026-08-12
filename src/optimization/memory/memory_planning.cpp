// optimization/memory_planning.cpp - memory planning + Memory IR
#include "cg/optimization/memory/memory_planning.hpp"
#include "cg/analysis/lifetime_analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

#include <algorithm>
#include <unordered_map>

namespace cg {

PreservedAnalyses MemoryPlanningPass::run(Module& m, AnalysisManager& am) {
    bool changed = false;
    auto& lt = am.get<LifetimeAnalysis>();

    // Greedy interval-graph coloring for buffer reuse.
    // Each tensor value gets a color (buffer id). Two values can share a
    // color iff their lifetimes don't overlap.
    struct Slot { u32 id; u64 bytes; };
    std::vector<Slot> slots;

    // Collect all live values with their lifetimes and sizes.
    struct ValueLife {
        ValueId vid;
        u32 start;
        u32 end;
        u64 bytes;
    };
    std::vector<ValueLife> lives;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            Value v = op.results[0];
            auto t = v.as_tensor();
            if (!t) continue;
            u64 n = 1;
            for (auto& d : t->shape) {
                if (!d->is_constant()) { n = 0; break; }
                n *= static_cast<u64>(d->value);
            }
            if (n == 0) continue;
            const auto& life = lt.lifetime_of(v);
            lives.push_back({v.id(), life.start_op, life.end_op,
                             n * dtype_size(t->dtype)});
        }
    }

    // Sort by start time, then by size descending (first-fit decreasing).
    std::sort(lives.begin(), lives.end(), [](const ValueLife& a, const ValueLife& b) {
        if (a.start != b.start) return a.start < b.start;
        return a.bytes > b.bytes;
    });

    // First-fit coloring.
    std::unordered_map<ValueId, u32> value_to_slot;
    for (auto& vl : lives) {
        u32 chosen = UINT32_MAX;
        for (u32 i = 0; i < slots.size(); ++i) {
            // Check if slot i is free during [vl.start, vl.end].
            // We track per-slot the last end_op.
            // For simplicity, we check overlap with all values assigned to
            // this slot so far.
            bool free = true;
            for (auto& [vid, slot_id] : value_to_slot) {
                if (slot_id != i) continue;
                // Find the original ValueLife for vid.
                for (auto& other : lives) {
                    if (other.vid != vid) continue;
                    if (!(vl.end < other.start || other.end < vl.start)) {
                        free = false; break;
                    }
                }
                if (!free) break;
            }
            if (free) {
                chosen = i;
                if (slots[i].bytes < vl.bytes) slots[i].bytes = vl.bytes;
                break;
            }
        }
        if (chosen == UINT32_MAX) {
            chosen = static_cast<u32>(slots.size());
            slots.push_back({chosen, vl.bytes});
        }
        value_to_slot[vl.vid] = chosen;
    }

    // Annotate the IR: insert alloc ops at the start and free ops at the end
    // of each value's lifetime. For reuse, we emit a `reuse` attribute on
    // the alloc.
    // The foundational implementation just annotates buffer ids as attributes
    // on the defining op; the actual alloc/free insertion happens during
    // lowering when we have a concrete backend.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto it = value_to_slot.find(op.results[0].id());
            if (it == value_to_slot.end()) continue;
            op.attributes.set("buffer_id", Attribute::make_integer(static_cast<i64>(it->second)));
            changed = true;
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    pa.preserve<LifetimeAnalysis>();
    return pa;
}

} // namespace cg
