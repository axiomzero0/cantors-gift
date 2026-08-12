// shape/simplifier.hpp - normalize and fold dim expressions
#pragma once

#include "cg/shape/dim_expr.hpp"

#include <cstdint>
#include <utility>

namespace cg {

// Returns a possibly-simpler DimExpr equivalent to `e` under integer arithmetic.
//   - constant folding
//   - x + 0 = x, x * 1 = x, x * 0 = 0
//   - x - 0 = x, 0 - x = -x
//   - min(x,x) = max(x,x) = x
//   - x / 1 = x, x % 1 = 0
//   - -(-x) = x
//   - associativity for add/mul when at least one side is constant
DimExprPtr simplify_dim(DimExprPtr e);

inline DimExprPtr simplify(const DimExprPtr& e) { return simplify_dim(e); }

inline DimExprPtr s_add(DimExprPtr a, DimExprPtr b) {
    return simplify_dim(DimExpr::make_add(std::move(a), std::move(b)));
}
inline DimExprPtr s_sub(DimExprPtr a, DimExprPtr b) {
    return simplify_dim(DimExpr::make_sub(std::move(a), std::move(b)));
}
inline DimExprPtr s_mul(DimExprPtr a, DimExprPtr b) {
    return simplify_dim(DimExpr::make_mul(std::move(a), std::move(b)));
}

} // namespace cg
