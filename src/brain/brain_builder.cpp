// brain/brain_builder.cpp - implementations of the canonical brain
// computation patterns.
//
// Each function emits a small cluster of brain-domain ops. The caller
// is responsible for declaring the input tensors (positions, weights,
// etc.) via Builder::input_tensor or by reusing function arguments.
#include "cg/brain/brain_builder.hpp"

namespace cg {
namespace brain {

DistanceReusePattern build_distance_reuse_pattern(
    Builder& b,
    Value positions,
    double sigma_x,
    double sigma_inh,
    double g0,
    double tau0,
    double inv_speed) {

    DistanceReusePattern out;

    // Compute d² ONCE. Every downstream kernel consumes the same Value.
    // If anything later in the program tries to recompute pairwise_dist_sq
    // on the same `positions` operand with the same (no) attributes, the
    // existing CSE pass will collapse it.
    out.d2 = b.pairwise_dist_sq(positions);

    // K_x = exp(-d² / (2 σ_x²))
    out.kx = b.gaussian_kernel(out.d2, sigma_x);

    // K_inh = g0 * exp(-d² / (2 σ_inh²))
    out.k_inh = b.lateral_inhibition(out.d2, g0, sigma_inh);

    // B_ij = K_x  (kept as a separate op for semantic clarity — analyses
    // that look for "connection affinity" can find B_ij without having
    // to know it's a Gaussian kernel).
    out.affinity = b.gaussian_kernel(out.d2, sigma_x);

    // τ_ij = τ0 + sqrt(d²) * inv_speed
    out.delay = b.delay_from_dist(out.d2, tau0, inv_speed);

    return out;
}

CreditAssignmentPattern build_credit_assignment_pattern(
    Builder& b,
    Value weights,
    Value input_credits,
    Value eligibility,
    double dt,
    double tau_e) {

    CreditAssignmentPattern out;

    // C_i = Σ_j w_ij * c_j
    out.credits = b.credit_propagate(weights, input_credits);

    // P_ij ← P_ij * exp(-Δt / τ_e)
    out.eligibility = b.eligibility_update(eligibility, dt, tau_e);

    return out;
}

EventDrivenPattern build_event_driven_pattern(
    Builder& b,
    Value positions,
    Value weights,
    Value events,
    Value activations,
    double radius,
    i64   max_neighbors,
    i64   topk) {

    EventDrivenPattern out;

    // Spatial neighbor lookup. Output is [N, K] of int32 indices.
    out.neighbors = b.neighbor_query(positions, radius, max_neighbors);

    // Sparse active set selection. Output is [K] of int32 indices.
    out.active = b.active_set(activations, topk);

    // Aggregate synaptic current from incoming events. Output is [N].
    out.current = b.synaptic_arrival(events, weights);

    return out;
}

} // namespace brain
} // namespace cg
