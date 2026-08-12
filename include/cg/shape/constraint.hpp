// shape/constraint.hpp - shape constraints and their collection
//
// Constraints are predicates over DimExpr that the compiler may assume to be
// true. Example:
//   N > 0
//   K % 16 == 0
//   M == N
//
// The solver answers questions of the form:
//   - "Can K ever be 0 under these constraints?"
//   - "Is M guaranteed to equal N?"
//   - "Is K guaranteed to be divisible by 16?"
//   - "Does this branch always evaluate to false?"
//
// We use a pragmatic SMT-lite approach: linear integer arithmetic over
// symbolic variables, with congruence (mod c) tracking and min/max bounds.
// Non-linear divisibility constraints (e.g. K % 16 == 0) are tracked
// symbolically and propagated when possible.
#pragma once

#include "cg/shape/dim_expr.hpp"
#include "cg/shape/simplifier.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

enum class CmpKind : u8 {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    ModEq,  // lhs % modulus == rhs   (rhs is a constant)
    ModNe,
};

struct Constraint {
    CmpKind kind;
    DimExprPtr lhs;
    DimExprPtr rhs;     // for ModEq/ModNe this is the constant remainder
    i64 modulus = 0;    // only used for Mod* kinds

    static Constraint eq(DimExprPtr l, DimExprPtr r) {
        return {CmpKind::EQ, std::move(l), std::move(r), 0};
    }
    static Constraint ne(DimExprPtr l, DimExprPtr r) {
        return {CmpKind::NE, std::move(l), std::move(r), 0};
    }
    static Constraint lt(DimExprPtr l, DimExprPtr r) {
        return {CmpKind::LT, std::move(l), std::move(r), 0};
    }
    static Constraint le(DimExprPtr l, DimExprPtr r) {
        return {CmpKind::LE, std::move(l), std::move(r), 0};
    }
    static Constraint gt(DimExprPtr l, DimExprPtr r) {
        return {CmpKind::GT, std::move(l), std::move(r), 0};
    }
    static Constraint ge(DimExprPtr l, DimExprPtr r) {
        return {CmpKind::GE, std::move(l), std::move(r), 0};
    }
    static Constraint mod_eq(DimExprPtr l, i64 modulus, i64 remainder) {
        return {CmpKind::ModEq, std::move(l), DimExpr::make_constant(remainder), modulus};
    }
    static Constraint mod_ne(DimExprPtr l, i64 modulus, i64 remainder) {
        return {CmpKind::ModNe, std::move(l), DimExpr::make_constant(remainder), modulus};
    }

    std::string to_string() const;
};

class ConstraintSet {
public:
    ConstraintSet() = default;

    void add(Constraint c) {
        // Normalize: subtract rhs from lhs so we always have lhs - rhs <cmp> 0
        switch (c.kind) {
            case CmpKind::EQ:
            case CmpKind::NE:
            case CmpKind::LT:
            case CmpKind::LE:
            case CmpKind::GT:
            case CmpKind::GE: {
                auto diff = simplify_dim(DimExpr::make_sub(c.lhs, c.rhs));
                constraints_.push_back({c.kind, diff, DimExpr::make_constant(0), 0});
                break;
            }
            case CmpKind::ModEq:
            case CmpKind::ModNe:
                constraints_.push_back(c);
                break;
        }
    }

    void add_eq(DimExprPtr a, DimExprPtr b) { add(Constraint::eq(std::move(a), std::move(b))); }
    void add_mod_eq(DimExprPtr a, i64 mod, i64 rem) {
        add(Constraint::mod_eq(std::move(a), mod, rem));
    }

    bool empty() const { return constraints_.empty(); }
    usize size() const { return constraints_.size(); }

    const std::vector<Constraint>& constraints() const { return constraints_; }

private:
    std::vector<Constraint> constraints_;
};

} // namespace cg
