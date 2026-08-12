// schedule/domain.hpp - symbolic iteration domains
//
// An IterationDomain describes the integer range a loop variable ranges over:
//
//     i ∈ [lo, hi)  step step
//
// All three bounds are DimExprs, so the domain can be symbolic (e.g.
// `i ∈ [0, M)` for an input dimension M). Schedule transformations operate
// algebraically on these domains without materializing concrete loops.
#pragma once

#include "cg/shape/dim_expr.hpp"
#include "cg/shape/simplifier.hpp"

#include <string>

namespace cg {

class IterationDomain {
public:
    DimExprPtr lo;
    DimExprPtr hi;
    DimExprPtr step;

    IterationDomain() = default;
    IterationDomain(DimExprPtr lo_, DimExprPtr hi_, DimExprPtr step_ = nullptr)
        : lo(std::move(lo_)), hi(std::move(hi_)),
          step(step_ ? std::move(step_) : DimExpr::make_constant(1)) {}

    static IterationDomain range(DimExprPtr n) {
        return IterationDomain(DimExpr::make_constant(0), std::move(n));
    }

    DimExprPtr extent() const {
        // (hi - lo + step - 1) / step
        auto diff = s_sub(hi, lo);
        auto num  = s_add(diff, s_sub(step, DimExpr::make_constant(1)));
        return simplify_dim(DimExpr::make_floor_div(num, step));
    }

    bool is_constant_extent(i64* out) const {
        auto e = extent();
        if (e->is_constant()) { *out = e->value; return true; }
        return false;
    }

    std::string to_string() const;
};

} // namespace cg
