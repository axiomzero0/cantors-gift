// egraph/rewrite_rules.hpp - tensor rewrite rule library
//
// A comprehensive library of ~50 tensor rewrite rules for the e-graph
// superoptimizer. Rules are tagged with:
//   - pure: always valid (algebraic identities)
//   - accuracy_risky: valid for real numbers but not for IEEE 754
//     (e.g., reassociation, contraction)
//   - layout_aware: valid only under certain stride/shape conditions
//   - tc_eligibility: unlocks tensor core throughput (e.g., f32→f16 cast)
//
// The rule library is the long tail of the optimizer — it's what makes
// the e-graph a defensible moat rather than a toy.
#pragma once

#include "cg/egraph/egraph.hpp"
#include "cg/numerical/semantics.hpp"

#include <functional>
#include <string>
#include <vector>

namespace cg {

enum class RuleKind : u8 {
    Pure,              // always valid (algebraic identity)
    AccuracyRisky,     // valid for reals, not for IEEE 754
    LayoutAware,       // requires stride/shape check
    TCEligibility,     // unlocks tensor cores (precision tradeoff)
};

struct RewriteRule {
    std::string name;
    EGraph::Rewrite rule;
    RuleKind kind = RuleKind::Pure;
    NumericalMode min_mode = NumericalMode::Strict; // minimum mode required
};

// Get the full rewrite rule library (~50 rules).
std::vector<RewriteRule> get_rewrite_rules(NumericalMode mode);

// Filter rules by numerical mode (strict modes exclude accuracy-risky rules).
std::vector<RewriteRule> filter_rules(
    const std::vector<RewriteRule>& rules,
    NumericalMode mode);

// ---- Rule categories ----

// Algebraic identities (always valid):
//   add(x, 0) → x                  [requires constant guard]
//   mul(x, 1) → x                  [requires constant guard]
//   mul(x, 0) → 0                  [requires constant guard]
//   sub(x, 0) → x                  [requires constant guard]
//   sub(x, x) → 0
//   neg(neg(x)) → x
//   add(x, x) → mul(x, 2)          [strength reduction]
std::vector<RewriteRule> algebraic_rules();

// Commutativity (always valid):
//   add(a, b) ↔ add(b, a)
//   mul(a, b) ↔ mul(b, a)
//   max(a, b) ↔ max(b, a)
//   min(a, b) ↔ min(b, a)
std::vector<RewriteRule> commutativity_rules();

// Associativity (accuracy-risky under IEEE 754):
//   add(add(a, b), c) ↔ add(a, add(b, c))
//   mul(mul(a, b), c) ↔ mul(a, mul(b, c))
std::vector<RewriteRule> associativity_rules();

// FMA formation (accuracy-risky — changes rounding):
//   add(mul(a, b), c) → fma(a, b, c)
//   add(c, mul(a, b)) → fma(a, b, c)
//   sub(mul(a, b), c) → fma(a, b, neg(c))
std::vector<RewriteRule> fma_formation_rules();

// Cast propagation:
//   cast(cast(x, f16), f32) → x     [if x was f32 originally]
//   mul(cast(x, f16), cast(y, f16)) → cast(mul(x, y), f16)
std::vector<RewriteRule> cast_propagation_rules();

// Transpose/reshape movement:
//   transpose(transpose(x, p), inv(p)) → x
//   reshape(reshape(x, s1), s2) → reshape(x, s2)
//   transpose(reshape(x, s), p) → reshape(transpose(x, p'), s')  [layout-aware]
std::vector<RewriteRule> layout_movement_rules();

// Reduction reassociation:
//   reduce_sum(reduce_sum(x, axis1), axis2) ↔ reduce_sum(x, [axis1, axis2])
//   reduce_sum(add(a, b), axis) → add(reduce_sum(a, axis), reduce_sum(b, axis))
std::vector<RewriteRule> reduction_rules();

// Tensor core eligibility:
//   matmul(A_f32, B_f32) → matmul(cast(A, f16), cast(B, f16))  [if accuracy allows]
//   This unlocks 20x throughput on A100+.
std::vector<RewriteRule> tc_eligibility_rules();

// Domain-specific:
//   softmax(x) → exp(x - max(x)) / sum(exp(x - max(x)))  [numerically stable]
//   relu(x) → max(x, 0)
//   gelu(x) → 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
std::vector<RewriteRule> domain_rules();

} // namespace cg
