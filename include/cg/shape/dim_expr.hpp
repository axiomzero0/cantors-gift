// shape/dim_expr.hpp - symbolic dimension expressions
//
// Dimensions in Tensor IR are NOT plain integers. They form an expression DAG
// over constants and symbolic variables, with arithmetic and min/max/ceildiv
// nodes. This lets the compiler prove things like "K % 32 == 0 implies
// ceildiv(K, 32) == K / 32", and to specialize shapes at compile time.
//
// DimExpr is an immutable, interned value. Equality is structural and
// pointer-equality (after interning) coincide. Hashing is structural.
#pragma once

#include "cg/core/util.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

// A symbolic variable identified by a small integer id and an optional name.
struct DimSymbol {
    u32 id;
    StringRef name; // optional; for debugging only

    bool operator==(const DimSymbol& o) const { return id == o.id; }
    bool operator!=(const DimSymbol& o) const { return id != o.id; }
};

// Tagged union of dimension expression nodes. We use a discriminated union
// rather than a class hierarchy so the equality / interning logic stays
// trivial and hashable.
enum class DimKind : u8 {
    Constant,    // integer literal
    Symbol,      // symbolic variable
    Add,         // a + b
    Sub,         // a - b
    Mul,         // a * b   (one operand must be Constant)
    FloorDiv,    // a / b   (integer, towards zero semantics)
    CeilDiv,     // (a + b - 1) / b for non-negative operands
    Mod,         // a % b
    Min,
    Max,
    Neg,         // -a
};

class DimExpr;
using DimExprPtr = std::shared_ptr<const DimExpr>;

// An interned dimension expression. Equality is structural and interning
// guarantees that two structurally-equal expressions share the same pointer.
class DimExpr {
public:
    DimKind kind;

    // Constant
    i64 value = 0;

    // Symbol
    DimSymbol symbol{0, {}};

    // Binary / unary
    DimExprPtr lhs;
    DimExprPtr rhs;

    static DimExprPtr make_constant(i64 v);
    static DimExprPtr make_symbol(u32 id, StringRef name = {});
    static DimExprPtr make_add(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_sub(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_mul(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_floor_div(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_ceil_div(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_mod(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_min(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_max(DimExprPtr a, DimExprPtr b);
    static DimExprPtr make_neg(DimExprPtr a);

    // Convenience overloads taking i64.
    static DimExprPtr make_add(DimExprPtr a, i64 b) {
        return make_add(std::move(a), make_constant(b));
    }
    static DimExprPtr make_mul(DimExprPtr a, i64 b) {
        return make_mul(std::move(a), make_constant(b));
    }

    bool is_constant() const { return kind == DimKind::Constant; }
    bool is_symbol()   const { return kind == DimKind::Symbol; }

    bool is_constant(i64 v) const {
        return kind == DimKind::Constant && value == v;
    }

    std::size_t hash() const;
    bool structurally_equal(const DimExpr& other) const;

    // Public so std::make_shared can default-construct via allocator_traits.
    DimExpr() = default;
};

class DimExprInterner {
public:
    static DimExprInterner& instance() {
        static DimExprInterner i;
        return i;
    }

    DimExprPtr intern_constant(i64 v) {
        std::lock_guard<std::mutex> g(mu_);
        auto key = std::make_pair(DimKind::Constant, v);
        auto it = constants_.find(key);
        if (it != constants_.end()) return it->second;
        auto n = std::make_shared<DimExpr>();
        n->kind = DimKind::Constant;
        n->value = v;
        constants_[key] = n;
        return n;
    }

    DimExprPtr intern_symbol(u32 id, StringRef name) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = symbols_.find(id);
        if (it != symbols_.end()) return it->second;
        auto n = std::make_shared<DimExpr>();
        n->kind = DimKind::Symbol;
        n->symbol = {id, std::string(name)};
        symbols_[id] = n;
        return n;
    }

    DimExprPtr intern_binary(DimKind k, DimExprPtr a, DimExprPtr b) {
        std::lock_guard<std::mutex> g(mu_);
        // Canonicalize operand order for commutative ops so that a+b and b+a
        // hash to the same value.
        if (k == DimKind::Add || k == DimKind::Mul) {
            if (a->hash() > b->hash()) std::swap(a, b);
        }
        auto key = std::make_tuple(k, a.get(), b.get());
        auto it = binary_.find(key);
        if (it != binary_.end()) return it->second;
        auto n = std::make_shared<DimExpr>();
        n->kind = k;
        n->lhs = std::move(a);
        n->rhs = std::move(b);
        binary_[key] = n;
        return n;
    }

    DimExprPtr intern_neg(DimExprPtr a) {
        std::lock_guard<std::mutex> g(mu_);
        auto key = a.get();
        auto it = negs_.find(key);
        if (it != negs_.end()) return it->second;
        auto n = std::make_shared<DimExpr>();
        n->kind = DimKind::Neg;
        n->lhs = std::move(a);
        negs_[key] = n;
        return n;
    }

private:
    struct PairHash {
        template <typename A, typename B>
        std::size_t operator()(const std::pair<A, B>& p) const {
            std::size_t h = std::hash<A>{}(p.first);
            h ^= std::hash<B>{}(p.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct TupleHash {
        std::size_t operator()(const std::tuple<DimKind, const DimExpr*, const DimExpr*>& t) const {
            std::size_t h = std::hash<u8>{}(static_cast<u8>(std::get<0>(t)));
            hash_combine(h, reinterpret_cast<u64>(std::get<1>(t)));
            hash_combine(h, reinterpret_cast<u64>(std::get<2>(t)));
            return h;
        }
    };

    struct PtrHash {
        std::size_t operator()(const DimExpr* p) const {
            return reinterpret_cast<std::size_t>(p);
        }
    };

    std::mutex mu_;
    std::unordered_map<std::pair<DimKind, i64>, DimExprPtr, PairHash> constants_;
    std::unordered_map<u32, DimExprPtr> symbols_;
    std::unordered_map<std::tuple<DimKind, const DimExpr*, const DimExpr*>, DimExprPtr, TupleHash> binary_;
    std::unordered_map<const DimExpr*, DimExprPtr, PtrHash> negs_;
};

inline DimExprPtr DimExpr::make_constant(i64 v) {
    return DimExprInterner::instance().intern_constant(v);
}
inline DimExprPtr DimExpr::make_symbol(u32 id, StringRef name) {
    return DimExprInterner::instance().intern_symbol(id, name);
}
inline DimExprPtr DimExpr::make_add(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::Add, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_sub(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::Sub, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_mul(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::Mul, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_floor_div(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::FloorDiv, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_ceil_div(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::CeilDiv, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_mod(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::Mod, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_min(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::Min, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_max(DimExprPtr a, DimExprPtr b) {
    return DimExprInterner::instance().intern_binary(DimKind::Max, std::move(a), std::move(b));
}
inline DimExprPtr DimExpr::make_neg(DimExprPtr a) {
    return DimExprInterner::instance().intern_neg(std::move(a));
}

inline std::size_t DimExpr::hash() const {
    std::size_t h = std::hash<u8>{}(static_cast<u8>(kind));
    switch (kind) {
        case DimKind::Constant:
            hash_combine(h, std::hash<i64>{}(value));
            break;
        case DimKind::Symbol:
            hash_combine(h, std::hash<u32>{}(symbol.id));
            break;
        case DimKind::Add: case DimKind::Sub: case DimKind::Mul:
        case DimKind::FloorDiv: case DimKind::CeilDiv: case DimKind::Mod:
        case DimKind::Min: case DimKind::Max:
            hash_combine(h, lhs->hash());
            hash_combine(h, rhs->hash());
            break;
        case DimKind::Neg:
            hash_combine(h, lhs->hash());
            break;
    }
    return h;
}

inline bool DimExpr::structurally_equal(const DimExpr& other) const {
    if (this == &other) return true;
    if (kind != other.kind) return false;
    switch (kind) {
        case DimKind::Constant: return value == other.value;
        case DimKind::Symbol:   return symbol.id == other.symbol.id;
        case DimKind::Add: case DimKind::Sub: case DimKind::Mul:
        case DimKind::FloorDiv: case DimKind::CeilDiv: case DimKind::Mod:
        case DimKind::Min: case DimKind::Max:
            return lhs->structurally_equal(*other.lhs) &&
                   rhs->structurally_equal(*other.rhs);
        case DimKind::Neg:
            return lhs->structurally_equal(*other.lhs);
    }
    return false;
}

// ---------------------------------------------------------------------------
// Shape - a vector of dimension expressions
// ---------------------------------------------------------------------------
class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<i64> dims) {
        dims_.reserve(dims.size());
        for (i64 d : dims) dims_.push_back(DimExpr::make_constant(d));
    }
    Shape(std::initializer_list<DimExprPtr> dims) {
        dims_.reserve(dims.size());
        for (const auto& d : dims) dims_.push_back(d);
    }
    explicit Shape(SmallVector<DimExprPtr> dims) : dims_(std::move(dims)) {}

    static Shape from_constants(std::vector<i64> dims) {
        SmallVector<DimExprPtr> v;
        v.reserve(dims.size());
        for (i64 d : dims) v.push_back(DimExpr::make_constant(d));
        return Shape(std::move(v));
    }

    usize rank() const { return dims_.size(); }
    bool   empty() const { return dims_.empty(); }

    DimExprPtr operator[](usize i) const { return dims_[i]; }
    DimExprPtr& operator[](usize i) { return dims_[i]; }

    auto begin()       { return dims_.begin(); }
    auto begin() const { return dims_.begin(); }
    auto end()         { return dims_.end(); }
    auto end()   const { return dims_.end(); }

    bool operator==(const Shape& o) const {
        if (rank() != o.rank()) return false;
        for (usize i = 0; i < rank(); ++i)
            if (!dims_[i]->structurally_equal(*o.dims_[i])) return false;
        return true;
    }

    u64 num_elements() const {
        u64 n = 1;
        for (auto& d : dims_) {
            if (!d->is_constant()) return 0; // symbolic: unknown
            n *= static_cast<u64>(d->value);
        }
        return n;
    }

    SmallVector<DimExprPtr>& dims() { return dims_; }
    const SmallVector<DimExprPtr>& dims() const { return dims_; }

private:
    SmallVector<DimExprPtr> dims_;
};

} // namespace cg
