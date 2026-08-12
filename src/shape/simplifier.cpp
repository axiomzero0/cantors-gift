// shape/simplifier.cpp - implementation of dim-expr simplification
#include "cg/shape/simplifier.hpp"

namespace cg {

namespace {

bool is_const(const DimExprPtr& e, i64 v) {
    return e && e->is_constant(v);
}

bool is_zero(const DimExprPtr& e)  { return is_const(e, 0); }
bool is_one(const DimExprPtr& e)   { return is_const(e, 1); }

// Recursively re-associate (e + c1) + c2 -> e + (c1+c2).
DimExprPtr try_reassociate_add(DimExprPtr a, DimExprPtr b) {
    if (a->kind == DimKind::Add && a->rhs->is_constant() && b->is_constant()) {
        i64 sum = a->rhs->value + b->value;
        return DimExpr::make_add(a->lhs, DimExpr::make_constant(sum));
    }
    if (b->kind == DimKind::Add && b->rhs->is_constant() && a->is_constant()) {
        i64 sum = a->value + b->rhs->value;
        return DimExpr::make_add(b->lhs, DimExpr::make_constant(sum));
    }
    return nullptr;
}

DimExprPtr try_reassociate_mul(DimExprPtr a, DimExprPtr b) {
    if (a->kind == DimKind::Mul && a->rhs->is_constant() && b->is_constant()) {
        i64 prod = a->rhs->value * b->value;
        return DimExpr::make_mul(a->lhs, DimExpr::make_constant(prod));
    }
    if (b->kind == DimKind::Mul && b->rhs->is_constant() && a->is_constant()) {
        i64 prod = a->value * b->rhs->value;
        return DimExpr::make_mul(b->lhs, DimExpr::make_constant(prod));
    }
    return nullptr;
}

} // namespace

DimExprPtr simplify_dim(DimExprPtr e) {
    if (!e) return e;

    switch (e->kind) {
        case DimKind::Constant:
        case DimKind::Symbol:
            return e;

        case DimKind::Add: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (a->is_constant() && b->is_constant())
                return DimExpr::make_constant(a->value + b->value);
            if (is_zero(a)) return b;
            if (is_zero(b)) return a;
            if (auto r = try_reassociate_add(a, b)) return simplify_dim(r);
            return DimExpr::make_add(std::move(a), std::move(b));
        }

        case DimKind::Sub: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (a->is_constant() && b->is_constant())
                return DimExpr::make_constant(a->value - b->value);
            if (is_zero(b)) return a;
            if (is_zero(a)) return simplify_dim(DimExpr::make_neg(b));
            return DimExpr::make_sub(std::move(a), std::move(b));
        }

        case DimKind::Mul: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (a->is_constant() && b->is_constant())
                return DimExpr::make_constant(a->value * b->value);
            if (is_zero(a) || is_zero(b)) return DimExpr::make_constant(0);
            if (is_one(a)) return b;
            if (is_one(b)) return a;
            if (auto r = try_reassociate_mul(a, b)) return simplify_dim(r);
            return DimExpr::make_mul(std::move(a), std::move(b));
        }

        case DimKind::FloorDiv: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (b->is_constant()) {
                if (b->value == 1) return a;
                if (b->value != 0 && a->is_constant()) {
                    // Python-style floor div on integers
                    i64 aa = a->value, bb = b->value;
                    i64 q = aa / bb;
                    if ((aa % bb != 0) && ((aa < 0) != (bb < 0))) --q;
                    return DimExpr::make_constant(q);
                }
            }
            return DimExpr::make_floor_div(std::move(a), std::move(b));
        }

        case DimKind::CeilDiv: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (b->is_constant()) {
                if (b->value == 1) return a;
                if (b->value != 0 && a->is_constant()) {
                    i64 aa = a->value, bb = b->value;
                    if (bb < 0) { aa = -aa; bb = -bb; }
                    if (aa <= 0) return DimExpr::make_constant(aa / bb);
                    i64 q = (aa + bb - 1) / bb;
                    return DimExpr::make_constant(q);
                }
            }
            return DimExpr::make_ceil_div(std::move(a), std::move(b));
        }

        case DimKind::Mod: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (b->is_constant()) {
                if (b->value == 1) return DimExpr::make_constant(0);
                if (b->value != 0 && a->is_constant()) {
                    i64 r = a->value % b->value;
                    if (r != 0 && ((r < 0) != (b->value < 0))) r += b->value;
                    return DimExpr::make_constant(r);
                }
            }
            return DimExpr::make_mod(std::move(a), std::move(b));
        }

        case DimKind::Min: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (a->structurally_equal(*b)) return a;
            if (a->is_constant() && b->is_constant())
                return DimExpr::make_constant(std::min(a->value, b->value));
            return DimExpr::make_min(std::move(a), std::move(b));
        }

        case DimKind::Max: {
            auto a = simplify_dim(e->lhs);
            auto b = simplify_dim(e->rhs);
            if (a->structurally_equal(*b)) return a;
            if (a->is_constant() && b->is_constant())
                return DimExpr::make_constant(std::max(a->value, b->value));
            return DimExpr::make_max(std::move(a), std::move(b));
        }

        case DimKind::Neg: {
            auto a = simplify_dim(e->lhs);
            if (a->is_constant()) return DimExpr::make_constant(-a->value);
            if (a->kind == DimKind::Neg) return a->lhs;
            return DimExpr::make_neg(std::move(a));
        }
    }
    return e;
}

} // namespace cg
