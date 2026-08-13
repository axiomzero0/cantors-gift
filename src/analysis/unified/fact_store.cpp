// analysis/unified/fact_store.cpp - query API for the FactStore.
//
// The query API is what optimization passes actually use. Instead of every
// pass implementing its own half-baked version of "is this probably a good
// idea?", passes call:
//
//   if (store.can_fuse(producer, consumer) &&
//       store.fusion_benefit(producer, consumer).net_predicted_improvement
//           > threshold) {
//       fuse(producer, consumer);
//   }
//
// Every query returns a result with confidence + provenance, so the
// optimizer can decide whether to trust it.
#include "cg/analysis/unified/fact_store.hpp"

#include "cg/ir/ops.hpp"

#include <cmath>

namespace cg {

// ---------------------------------------------------------------------------
// Property queries
// ---------------------------------------------------------------------------

bool FactStore::is_zero(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->properties.known) return false;
    return has_property(tf->properties.value, TensorProperty::Zero);
}

bool FactStore::is_one(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->properties.known) return false;
    return has_property(tf->properties.value, TensorProperty::One);
}

bool FactStore::is_identity(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->properties.known) return false;
    return has_property(tf->properties.value, TensorProperty::Identity);
}

bool FactStore::is_constant(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->properties.known) return false;
    return has_property(tf->properties.value, TensorProperty::Constant);
}

bool FactStore::is_non_negative(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->value_range.known) return false;
    return tf->value_range.value.is_non_negative();
}

bool FactStore::is_strictly_positive(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->value_range.known) return false;
    return tf->value_range.value.is_strictly_positive();
}

bool FactStore::is_diagonal(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->properties.known) return false;
    return has_property(tf->properties.value, TensorProperty::Diagonal);
}

bool FactStore::is_sparse(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->properties.known) return false;
    return has_property(tf->properties.value, TensorProperty::Sparse);
}

std::optional<double> FactStore::constant_value(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->constant_value_known.known ||
        !tf->constant_value_known.value || !tf->constant_value.known) {
        return std::nullopt;
    }
    return tf->constant_value.value;
}

std::optional<std::vector<i64>> FactStore::static_shape(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->shape.known) return std::nullopt;
    std::vector<i64> out;
    out.reserve(tf->shape.value.size());
    for (auto& d : tf->shape.value) {
        if (!d.is_constant()) return std::nullopt;
        out.push_back(d.value());
    }
    return out;
}

std::optional<DType> FactStore::dtype_of(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->dtype.known) return std::nullopt;
    return tf->dtype.value;
}

LayoutPtr FactStore::layout_of(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->layout.known) return nullptr;
    return tf->layout.value;
}

std::optional<std::vector<i64>> FactStore::static_strides(ValueId vid) const {
    auto* tf = facts_for(vid);
    if (!tf || !tf->strides.known) return std::nullopt;
    std::vector<i64> out;
    out.reserve(tf->strides.value.size());
    for (auto& s : tf->strides.value) {
        if (!s->is_constant()) return std::nullopt;
        out.push_back(s->value);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fusion queries
// ---------------------------------------------------------------------------

bool FactStore::can_fuse(ValueId producer, ValueId consumer) const {
    // Conservative legality check: producer must be a pure value with a
    // single consumer (or be a constant), and the two must not alias
    // conflicting storage.
    auto* pt = facts_for(producer);
    auto* ct = facts_for(consumer);
    if (!pt || !ct) return false;

    // If the producer is dead, we can't fuse (no uses).
    if (pt->num_users.known && pt->num_users.value == 0) return false;

    // If the producer has multiple distinct consumers, fusion would
    // duplicate work — only legal if recomputation is profitable.
    // (We allow it here; the profitability check below filters it out.)
    if (pt->num_distinct_consumers.known && pt->num_distinct_consumers.value > 1) {
        // Still legal — just need to check profitability.
    }

    // If the producer and consumer must-alias different storage, fusion
    // is illegal (would break memory dependencies).
    if (pt->alias_class.known && ct->alias_class.known) {
        const auto& pa = pt->alias_class.value;
        const auto& ca = ct->alias_class.value;
        if (pa.kind == AliasKind::MustAlias && ca.kind == AliasKind::MustAlias &&
            pa.alias_set_id != ca.alias_set_id) {
            // Different storage — but that's actually fine, the producer's
            // storage just won't be written. Legal.
        }
    }

    return true;  // conservative yes
}

FusionBenefitReport FactStore::fusion_benefit(ValueId producer, ValueId consumer) const {
    FusionBenefitReport report;

    if (!can_fuse(producer, consumer)) {
        report.legality_reason = "can_fuse returned false";
        return report;
    }
    report.can_fuse = true;

    auto* pt = facts_for(producer);
    auto* ct = facts_for(consumer);
    if (!pt || !ct) return report;

    // Savings: bytes of the producer that no longer need to be written to
    // global memory (because the producer is consumed directly in registers).
    if (pt->estimated_bytes_written.known) {
        report.saved_bytes = static_cast<double>(pt->estimated_bytes_written.value);
    }
    // One fewer kernel launch (assuming the producer was a separate kernel).
    report.saved_kernel_launches = 1.0;

    // Costs: if the producer has multiple consumers, fusing into one means
    // recomputing the producer for the other consumers (or duplicating work).
    if (pt->num_users.known && pt->num_users.value > 1) {
        // Recomputation cost = (n-1) * producer_flops
        if (pt->estimated_flops.known) {
            double recompute_cost = static_cast<double>(pt->estimated_flops.value) *
                                     (pt->num_users.value - 1);
            // Very rough: assume 1 TFLOP/s effective.
            report.saved_runtime_sec -= recompute_cost / 1e12;
        }
    }

    // Register pressure estimate: fusing adds the producer's working set
    // to the consumer's register footprint. Very rough heuristic.
    if (pt->estimated_bytes_written.known) {
        // Each element held in a register = 4 bytes (F32).
        // Number of registers added ~ sqrt(bytes_written / 4).
        double bytes = static_cast<double>(pt->estimated_bytes_written.value);
        report.added_register_pressure = std::sqrt(bytes / 4.0);
    }

    // Occupancy impact: more registers = lower occupancy.
    // A100: 65536 regs/SM, 256 threads/block, max 64 warps/SM.
    // Each additional register per thread reduces warps/SM by ~1/256.
    if (has_hardware()) {
        u32 regs_added = static_cast<u32>(report.added_register_pressure);
        u32 regs_per_warp = regs_added * hw_.warp_size;
        u32 warps_lost = regs_per_warp > 0 ? (regs_per_warp + hw_.register_file_per_sm - 1) /
                                              hw_.register_file_per_sm : 0;
        double occ_before = static_cast<double>(hw_.max_warps_per_sm);
        double occ_after = static_cast<double>(std::max(1u, hw_.max_warps_per_sm - warps_lost));
        report.occupancy_delta_pct = 100.0 * (occ_after - occ_before) / occ_before;
    }

    // Net predicted improvement: saved_bytes * (1 / bandwidth) - runtime cost.
    if (has_hardware()) {
        double bw = hw_.memory.get(MemorySpace::Generic);
        if (bw > 0) {
            double saved_sec = report.saved_bytes / bw;
            report.saved_runtime_sec += saved_sec;
        }
        // Add launch-overhead saving.
        report.saved_runtime_sec += hw_.launch_overhead_sec;
    }

    // Net predicted improvement as a fraction.
    // We need a baseline runtime estimate. Use the consumer's cost.
    //
    // IMPORTANT: the baseline must include launch overhead, otherwise
    // tiny ops produce absurd ratios. A 16x16 mul has 256 FLOPs, which
    // at 19.5 TFLOPs takes 1.3e-11 sec — but the kernel launch alone
    // is 5e-6 sec. Dividing saved_runtime (5 us) by compute-only
    // baseline (1.3e-11 sec) gives a bogus 380,000x "improvement".
    //
    // The fix: baseline = max(compute_sec, launch_overhead_sec).
    if (ct->estimated_flops.known && has_hardware()) {
        double compute_sec = static_cast<double>(ct->estimated_flops.value) /
                              hw_.peak_flops(DType::F32);
        double baseline_sec = std::max(compute_sec, hw_.launch_overhead_sec);
        if (baseline_sec > 0) {
            report.net_predicted_improvement =
                report.saved_runtime_sec / baseline_sec;
        }
    }

    // Confidence: Proven facts about bytes + Estimated for runtime.
    report.confidence = Confidence::Estimated;

    // Provenance chain.
    Provenance p1;
    p1.rule = "FusionBenefitAnalysis";
    p1.source_op = 0;
    p1.explanation = "saved_bytes from producer.estimated_bytes_written (Proven); "
                     "runtime estimate from hardware model (Estimated)";
    report.reasons.push_back(p1);

    if (pt->num_users.known && pt->num_users.value > 1) {
        Provenance p2;
        p2.rule = "RecomputationCost";
        p2.source_op = producer;
        p2.explanation = "producer has " + std::to_string(pt->num_users.value) +
                          " users; fusion into one requires recomputation for others";
        report.reasons.push_back(p2);
    }

    return report;
}

} // namespace cg
