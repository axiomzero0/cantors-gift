// shape/solver.hpp - query constraints over a ConstraintSet
//
// A pragmatic SMT-lite solver for the kind of questions a tensor compiler
// actually asks:
//   - "Is this expression provably equal to / less than / greater than
//      another expression under the constraints?"
//   - "Is this expression guaranteed to be divisible by c?"
//   - "What is a tight constant range [lo, hi] for this expression?"
//   - "Is this constraint set satisfiable / is it always true?"
//
// We do not implement a general SMT solver. For complex queries we return
// `Unknown` rather than an unsound answer. Soundness is preferred over
// completeness: a wrong "yes" can break a kernel; a wrong "no" only leaves
// optimization on the table.
#pragma once

#include "cg/shape/constraint.hpp"

#include <optional>
#include <utility>

namespace cg {

enum class SolverResult : u8 {
    ProvedTrue,
    ProvedFalse,
    Unknown,
};

class Solver {
public:
    explicit Solver(ConstraintSet cs) : cs_(std::move(cs)) {}

    // Returns true iff the constraint is provably true under cs.
    SolverResult prove(Constraint c) const;

    // Returns true iff `a == b` is provable.
    SolverResult prove_equal(DimExprPtr a, DimExprPtr b) const;

    // Returns true iff `a` is provably divisible by `modulus`.
    SolverResult prove_divisible(DimExprPtr a, i64 modulus) const;

    // Best-effort constant range [lo, hi] for the expression. Returns
    // nullopt if no finite bound can be derived.
    struct Range { i64 lo; i64 hi; };
    std::optional<Range> range(DimExprPtr e) const;

    const ConstraintSet& constraints() const { return cs_; }

private:
    ConstraintSet cs_;

    // Internal helpers
    i64 eval_constant(DimExprPtr e) const;
};

} // namespace cg
