// shape/solver.cpp - implementation of the constraint solver
#include "cg/shape/solver.hpp"

#include <algorithm>
#include <climits>
#include <unordered_map>

namespace cg {

namespace {

// A small union-find for symbolic equalities between symbols.
struct UnionFind {
    std::unordered_map<u32, u32> parent;
    std::unordered_map<u32, i64> offset; // this = parent + offset

    u32 find(u32 x) {
        auto it = parent.find(x);
        if (it == parent.end()) {
            parent[x] = x;
            offset[x] = 0;
            return x;
        }
        if (it->second == x) return x;
        // path compression with offset
        u32 root = find(it->second);
        offset[x] += offset[it->second];
        parent[x] = root;
        return root;
    }

    // Records `a - b == c` if known, returns true if consistent.
    bool unify(u32 a, u32 b, i64 c) {
        u32 ra = find(a), rb = find(b);
        // a = offset[a] + ra
        // b = offset[b] + rb
        // a - b = c  =>  (offset[a] - offset[b]) + (ra - rb) = c
        if (ra == rb) {
            return (offset[a] - offset[b]) == c;
        }
        // Merge ra into rb: ra = rb + (offset[b] - offset[a] + c)
        i64 off = offset[b] - offset[a] + c;
        parent[ra] = rb;
        offset[ra] = off;
        return true;
    }

    std::optional<i64> relation(u32 a, u32 b) {
        u32 ra = find(a), rb = find(b);
        if (ra != rb) return std::nullopt;
        // a - b = offset[a] - offset[b]
        return offset[a] - offset[b];
    }
};

// Collect all symbols in a DimExpr.
void collect_symbols(const DimExprPtr& e, std::unordered_set<u32>& out) {
    if (!e) return;
    switch (e->kind) {
        case DimKind::Symbol: out.insert(e->symbol.id); return;
        case DimKind::Constant: return;
        case DimKind::Add: case DimKind::Sub: case DimKind::Mul:
        case DimKind::FloorDiv: case DimKind::CeilDiv: case DimKind::Mod:
        case DimKind::Min: case DimKind::Max:
            collect_symbols(e->lhs, out);
            collect_symbols(e->rhs, out);
            return;
        case DimKind::Neg:
            collect_symbols(e->lhs, out);
            return;
    }
}

// Map symbol-id -> offset from `e` assuming `e` is affine in those symbols.
// Returns true on success; the constant part is stored in `c`.
bool to_affine(const DimExprPtr& e,
               std::unordered_map<u32, i64>& coeffs,
               i64& c) {
    switch (e->kind) {
        case DimKind::Constant:
            c += e->value;
            return true;
        case DimKind::Symbol:
            coeffs[e->symbol.id] += 1;
            return true;
        case DimKind::Neg:
            return to_affine(e->lhs, coeffs, c) && ([&]{
                for (auto& [k, v] : coeffs) v = -v;
                c = -c;
                return true;
            }());
        case DimKind::Add: {
            i64 c1 = 0, c2 = 0;
            std::unordered_map<u32, i64> a, b;
            if (!to_affine(e->lhs, a, c1)) return false;
            if (!to_affine(e->rhs, b, c2)) return false;
            for (auto& [k, v] : b) a[k] += v;
            coeffs = std::move(a);
            c = c1 + c2;
            return true;
        }
        case DimKind::Sub: {
            i64 c1 = 0, c2 = 0;
            std::unordered_map<u32, i64> a, b;
            if (!to_affine(e->lhs, a, c1)) return false;
            if (!to_affine(e->rhs, b, c2)) return false;
            for (auto& [k, v] : b) a[k] -= v;
            coeffs = std::move(a);
            c = c1 - c2;
            return true;
        }
        case DimKind::Mul: {
            // one side must be constant
            if (e->lhs->is_constant()) {
                if (!to_affine(e->rhs, coeffs, c)) return false;
                i64 k = e->lhs->value;
                for (auto& [_, v] : coeffs) v *= k;
                c *= k;
                return true;
            }
            if (e->rhs->is_constant()) {
                if (!to_affine(e->lhs, coeffs, c)) return false;
                i64 k = e->rhs->value;
                for (auto& [_, v] : coeffs) v *= k;
                c *= k;
                return true;
            }
            return false;
        }
        case DimKind::FloorDiv: {
            // floor_div(a, c) only if a is affine and divisor is constant
            if (!e->rhs->is_constant()) return false;
            if (e->rhs->value == 0) return false;
            if (!to_affine(e->lhs, coeffs, c)) return false;
            // Only sound if all coeffs are divisible by divisor AND constant
            // is divisible by divisor (with proper floor semantics).
            i64 d = e->rhs->value;
            for (auto& [_, v] : coeffs) {
                if (v % d != 0) return false;
                v /= d;
            }
            // floor(c / d) - we approximate by Python-style floor
            i64 q = c / d;
            if ((c % d != 0) && ((c < 0) != (d < 0))) --q;
            c = q;
            return true;
        }
        case DimKind::CeilDiv: {
            if (!e->rhs->is_constant()) return false;
            if (e->rhs->value == 0) return false;
            if (!to_affine(e->lhs, coeffs, c)) return false;
            i64 d2 = e->rhs->value;
            for (auto& [_, v] : coeffs) {
                if (v % d2 != 0) return false;
                v /= d2;
            }
            // ceildiv(c, d2)
            i64 q2;
            if (d2 < 0) { c = -c; d2 = -d2; }
            if (c <= 0) q2 = c / d2;
            else q2 = (c + d2 - 1) / d2;
            c = q2;
            return true;
        }
        case DimKind::Mod:
        case DimKind::Min:
        case DimKind::Max:
            return false;
    }
    return false;
}

} // namespace

i64 Solver::eval_constant(DimExprPtr e) const {
    if (e->is_constant()) return e->value;
    auto r = range(e);
    if (r && r->lo == r->hi) return r->lo;
    return 0;
}

std::optional<Solver::Range> Solver::range(DimExprPtr e) const {
    if (e->is_constant()) return Range{e->value, e->value};

    // Collect symbol bounds and equalities.
    std::unordered_map<u32, Range> sym_bounds;
    UnionFind uf;

    for (const auto& c : cs_.constraints()) {
        switch (c.kind) {
            case CmpKind::EQ: {
                // Try to interpret `lhs - rhs == 0` as a symbol relation.
                if (c.lhs->kind == DimKind::Sub &&
                    c.lhs->lhs->is_symbol() &&
                    c.lhs->rhs->is_symbol() &&
                    c.rhs->is_constant(0)) {
                    uf.unify(c.lhs->lhs->symbol.id,
                             c.lhs->rhs->symbol.id,
                             c.rhs->value);
                }
                break;
            }
            case CmpKind::ModEq: {
                // ModEq(lhs, mod, rem): lhs == k * mod + rem
                // We can record congruence but not direct bounds.
                break;
            }
            default:
                break;
        }
    }

    // Affine decomposition
    std::unordered_map<u32, i64> coeffs;
    i64 constant = 0;
    if (!to_affine(e, coeffs, constant)) {
        // Non-affine: best-effort unknown range.
        return std::nullopt;
    }

    // If all symbols are unbounded, return nullopt.
    if (!coeffs.empty()) {
        // Use sym_bounds if known; otherwise nullopt.
        bool any_unknown = false;
        i64 lo = constant, hi = constant;
        for (auto& [k, v] : coeffs) {
            auto it = sym_bounds.find(k);
            if (it == sym_bounds.end()) { any_unknown = true; break; }
            if (v >= 0) {
                lo += v * it->second.lo;
                hi += v * it->second.hi;
            } else {
                lo += v * it->second.hi;
                hi += v * it->second.lo;
            }
        }
        if (any_unknown) return std::nullopt;
        return Range{lo, hi};
    }
    return Range{constant, constant};
}

SolverResult Solver::prove_equal(DimExprPtr a, DimExprPtr b) const {
    auto diff = simplify_dim(DimExpr::make_sub(a, b));
    if (diff->is_constant(0)) return SolverResult::ProvedTrue;
    if (diff->is_constant() && diff->value != 0) return SolverResult::ProvedFalse;

    // Affine check via union-find on symbolic equalities in the constraint set.
    std::unordered_map<u32, i64> coeffs;
    i64 constant = 0;
    if (!to_affine(diff, coeffs, constant)) return SolverResult::Unknown;
    if (coeffs.empty()) {
        return constant == 0 ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
    }

    // Apply equality constraints to substitute.
    UnionFind uf;
    for (const auto& c : cs_.constraints()) {
        if (c.kind != CmpKind::EQ) continue;
        if (c.rhs->is_constant(0) &&
            c.lhs->kind == DimKind::Sub &&
            c.lhs->lhs->is_symbol() &&
            c.lhs->rhs->is_symbol()) {
            uf.unify(c.lhs->lhs->symbol.id, c.lhs->rhs->symbol.id, 0);
        }
    }
    // Combine coefficients of unified symbols
    std::unordered_map<u32, i64> merged;
    for (auto& [k, v] : coeffs) {
        u32 r = uf.find(k);
        merged[r] += v;
    }
    bool all_zero = true;
    for (auto& [_, v] : merged) if (v != 0) { all_zero = false; break; }
    if (all_zero) {
        return constant == 0 ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
    }
    return SolverResult::Unknown;
}

SolverResult Solver::prove_divisible(DimExprPtr a, i64 modulus) const {
    if (modulus == 0) return SolverResult::Unknown;
    if (a->is_constant()) return (a->value % modulus == 0)
        ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;

    // Direct: search for ModEq(a, modulus, 0)
    for (const auto& c : cs_.constraints()) {
        if (c.kind == CmpKind::ModEq &&
            c.modulus == modulus &&
            c.rhs->is_constant(0) &&
            c.lhs->structurally_equal(*a)) {
            return SolverResult::ProvedTrue;
        }
    }

    // For an affine a, divisibility holds iff all symbol coeffs are divisible
    // AND the constant is divisible.
    std::unordered_map<u32, i64> coeffs;
    i64 constant = 0;
    if (to_affine(a, coeffs, constant)) {
        bool all_div = (constant % modulus) == 0;
        for (auto& [_, v] : coeffs) {
            if (v % modulus != 0) { all_div = false; break; }
        }
        if (all_div) return SolverResult::ProvedTrue;
    }

    return SolverResult::Unknown;
}

SolverResult Solver::prove(Constraint c) const {
    switch (c.kind) {
        case CmpKind::EQ: {
            return prove_equal(c.lhs, c.rhs);
        }
        case CmpKind::NE: {
            auto r = prove_equal(c.lhs, c.rhs);
            if (r == SolverResult::ProvedTrue) return SolverResult::ProvedFalse;
            if (r == SolverResult::ProvedFalse) return SolverResult::ProvedTrue;
            return SolverResult::Unknown;
        }
        case CmpKind::LT: {
            auto diff = simplify_dim(DimExpr::make_sub(c.lhs, c.rhs));
            if (diff->is_constant()) return diff->value < 0
                ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
            auto r = range(diff);
            if (r) {
                if (r->hi < 0) return SolverResult::ProvedTrue;
                if (r->lo >= 0) return SolverResult::ProvedFalse;
            }
            return SolverResult::Unknown;
        }
        case CmpKind::LE: {
            auto diff = simplify_dim(DimExpr::make_sub(c.lhs, c.rhs));
            if (diff->is_constant()) return diff->value <= 0
                ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
            auto r = range(diff);
            if (r) {
                if (r->hi <= 0) return SolverResult::ProvedTrue;
                if (r->lo > 0)  return SolverResult::ProvedFalse;
            }
            return SolverResult::Unknown;
        }
        case CmpKind::GT: {
            auto diff = simplify_dim(DimExpr::make_sub(c.lhs, c.rhs));
            if (diff->is_constant()) return diff->value > 0
                ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
            auto r = range(diff);
            if (r) {
                if (r->lo > 0)  return SolverResult::ProvedTrue;
                if (r->hi <= 0) return SolverResult::ProvedFalse;
            }
            return SolverResult::Unknown;
        }
        case CmpKind::GE: {
            auto diff = simplify_dim(DimExpr::make_sub(c.lhs, c.rhs));
            if (diff->is_constant()) return diff->value >= 0
                ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
            auto r = range(diff);
            if (r) {
                if (r->lo >= 0) return SolverResult::ProvedTrue;
                if (r->hi < 0)  return SolverResult::ProvedFalse;
            }
            return SolverResult::Unknown;
        }
        case CmpKind::ModEq: {
            // lhs % modulus == rhs
            if (!c.rhs->is_constant()) return SolverResult::Unknown;
            i64 rem = c.rhs->value;
            if (rem < 0 || rem >= c.modulus) return SolverResult::ProvedFalse;
            // direct
            for (const auto& cc : cs_.constraints()) {
                if (cc.kind == CmpKind::ModEq &&
                    cc.modulus == c.modulus &&
                    cc.lhs->structurally_equal(*c.lhs) &&
                    cc.rhs->is_constant(rem)) {
                    return SolverResult::ProvedTrue;
                }
            }
            // If lhs is provably divisible by modulus then ModEq holds iff rem==0.
            auto divres = prove_divisible(c.lhs, c.modulus);
            if (divres == SolverResult::ProvedTrue) {
                return rem == 0 ? SolverResult::ProvedTrue : SolverResult::ProvedFalse;
            }
            return SolverResult::Unknown;
        }
        case CmpKind::ModNe: {
            auto r = prove(Constraint::mod_eq(c.lhs, c.modulus, c.rhs->value));
            if (r == SolverResult::ProvedTrue) return SolverResult::ProvedFalse;
            if (r == SolverResult::ProvedFalse) return SolverResult::ProvedTrue;
            return SolverResult::Unknown;
        }
    }
    return SolverResult::Unknown;
}

} // namespace cg
