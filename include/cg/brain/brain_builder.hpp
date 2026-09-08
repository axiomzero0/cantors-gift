// brain/brain_builder.hpp - high-level builder for the brain-like
// spatial neural graph ML domain.
//
// BrainBuilder is a thin facade over the generic IR Builder that
// constructs canonical patterns of brain semantic primitives.
//
// It exists because the math we care about has a small number of
// recurring compositions that the user (or the higher-level driver)
// wants to emit without thinking about operand wiring:
//
//   1. distance_reuse_pattern(x, sigma_x, sigma_inh, g0, tau0, v)
//      - Computes pairwise distance² ONCE.
//      - Feeds the same d² value into:
//          * K_x       = GaussianKernel(d², σ_x)
//          * K_inh     = LateralInhibition(d², g0, σ_inh)
//          * B_ij      = K_x          (connection affinity)
//          * τ_delay   = DelayFromDist(d², τ0, v)
//      - This is the single biggest optimization target identified
//        in the design discussion: a naïve implementation would
//        compute d² 5 times. By having the front-end emit ONE
//        PairwiseDistSq op and reuse the Value, the existing CSE
//        pass keeps it as a single computation — and the fusion
//        pass can fuse the four elementwise consumers with it.
//
//   2. credit_assignment_pattern(weights, credits, p, dt, tau_e)
//      - CreditPropagate: C_i ← Σ_j w_ij * c_j
//      - EligibilityUpdate: P_ij ← P_ij * exp(-dt/τ_e)
//
//   3. event_drive_pattern(positions, weights, events, radius, topk)
//      - NeighborQuery + ActiveSet + SynapticArrival wiring
//
// The brain builder does NOT change the IR architecture. It produces
// the same Tensor IR operations the generic Builder would produce —
// it just makes the intent explicit and reduces the chance that the
// caller accidentally emits duplicate distance computations.
#pragma once

#include "cg/ir/builder.hpp"
#include "cg/ir/module.hpp"

#include <utility>
#include <vector>

namespace cg {
namespace brain {

// Output of the canonical "distance reuse" pattern. Each field is a
// Value in the IR; all of them consume the same d².
struct DistanceReusePattern {
    Value d2;        // [N, N]   pairwise distance²  (computed ONCE)
    Value kx;        // [N, N]   Gaussian affinity kernel K_x
    Value k_inh;     // [N, N]   lateral inhibition weights
    Value affinity;  // [N, N]   B_ij = K_x  (kept as a separate op for clarity)
    Value delay;     // [N, N]   τ_ij = τ0 + sqrt(d²) * inv_speed
};

// Build the canonical distance-reuse pattern. The returned struct
// references ONE PairwiseDistSq op (via `d2`) that is consumed by
// every downstream kernel. CSE will collapse any future references
// to `d2` automatically.
DistanceReusePattern build_distance_reuse_pattern(
    Builder& b,
    Value positions,           // [N, D]
    double sigma_x,           // affinity kernel width
    double sigma_inh,         // inhibition kernel width
    double g0,                // inhibition amplitude
    double tau0,              // base synaptic delay
    double inv_speed);        // inverse conduction velocity

// Output of the credit assignment pattern.
struct CreditAssignmentPattern {
    Value credits;       // [N]     propagated credit C_i
    Value eligibility;   // [N, N]  updated eligibility trace
};

CreditAssignmentPattern build_credit_assignment_pattern(
    Builder& b,
    Value weights,        // [N, N]
    Value input_credits,  // [N]
    Value eligibility,    // [N, N]
    double dt,            // time since last update
    double tau_e);        // eligibility decay constant

// Output of the event-driven pattern.
struct EventDrivenPattern {
    Value neighbors;   // [N, K]   neighbor indices
    Value active;      // [K]      active-set neuron indices
    Value current;     // [N]      aggregated synaptic current
};

EventDrivenPattern build_event_driven_pattern(
    Builder& b,
    Value positions,    // [N, D]
    Value weights,      // [N, N]
    Value events,       // [E, 3]
    Value activations,  // [N]
    double radius,
    i64   max_neighbors,
    i64   topk);

} // namespace brain
} // namespace cg
