// analysis/unified/abstract_domain.hpp - abstract domain primitives for the
// unified Tensor Knowledge Graph.
//
// The central idea: every fact about a tensor lives in an ABSTRACT DOMAIN
// that supports a lattice join (∨), a "top" (unknown), and a "bottom"
// (contradiction). This lets analyses:
//
//   - exchange information through a shared fact store
//   - refine facts iteratively until a fixed point
//   - track confidence + provenance for every derived fact
//   - distinguish "proven" facts (used for legality) from "estimated"
//     facts (used for profitability)
//
// Domains defined here:
//
//   Confidence   - Proven / Derived / Estimated / Profiled / Speculative
//   Provenance   - rule + operand chain explaining HOW a fact was derived
//   Bound        - lower / upper / exact for a symbolic dimension
//   Dimension    - exact DimExpr + lower/upper bounds
//   ValueRange   - min / max / sign information for tensor element values
//   TensorProperty - bitset lattice (Zero, One, Identity, Diagonal, ...)
//   AliasClass   - MustAlias / MayAlias / NoAlias + alias-set id
//   Lifetime     - logical + physical + tile-level lifetime
//   ReductionInfo - axes, associativity, commutativity, identity
//
// The lattice structure is what makes constraint propagation work: when
// a new fact arrives, we join it with the existing fact and check whether
// the result changed. If it did, we re-queue dependents.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/ir/value.hpp"
#include "cg/shape/dim_expr.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cg {

// ---------------------------------------------------------------------------
// Confidence - how much to trust a fact.
//
// Optimization LEGALITY uses only Proven/Derived facts.
// Optimization PROFITABILITY may use Estimated/Profiled/Speculative facts.
// ---------------------------------------------------------------------------
enum class Confidence : u8 {
    Proven,      // derived from axioms via sound rules (e.g. shape inference)
    Derived,     // derived from Proven facts via heuristic rules
    Estimated,   // produced by a model (analytical cost, occupancy heuristic)
    Profiled,    // observed at runtime (PGO)
    Speculative, // a guess; never use for legality
};

inline std::string_view confidence_name(Confidence c) {
    switch (c) {
        case Confidence::Proven:     return "proven";
        case Confidence::Derived:    return "derived";
        case Confidence::Estimated:  return "estimated";
        case Confidence::Profiled:   return "profiled";
        case Confidence::Speculative:return "speculative";
    }
    return "?";
}

// True iff a fact with this confidence can be used for optimization legality.
inline bool is_sound(Confidence c) {
    return c == Confidence::Proven || c == Confidence::Derived;
}

// ---------------------------------------------------------------------------
// Provenance - the chain of rules + operands that produced a fact.
//
// When the compiler makes a terrible decision, you should be able to ask
// "why did you think this was profitable?" and get a proof chain back.
// This struct is the unit of that chain.
// ---------------------------------------------------------------------------
struct Provenance {
    // The rule or analysis that produced this fact.
    // e.g. "ShapeInference", "MulZero", "ConstantFolding", "BayesianCostModel"
    std::string rule;

    // The OpId (or 0 for module-level facts) the fact was derived from.
    u32 source_op = 0;

    // Optional: the operand ValueIds that fed into this derivation.
    std::vector<ValueId> operands;

    // Optional: a human-readable explanation.
    std::string explanation;

    Provenance() = default;
    explicit Provenance(std::string r) : rule(std::move(r)) {}
    Provenance(std::string r, u32 op) : rule(std::move(r)), source_op(op) {}
};

// ---------------------------------------------------------------------------
// A fact wrapper: value + confidence + provenance.
//
// This is the unit stored in the FactStore. The lattice join is:
//   - if either side is "unknown", take the other
//   - if both known, prefer higher confidence; on tie, prefer the
//     more-recently-derived fact (last-writer-wins on tie)
//   - if values conflict and both are Proven, that's a contradiction
//     (we currently pick one and log; a sound system would assert).
// ---------------------------------------------------------------------------
template <typename T>
struct Fact {
    T value;
    Confidence confidence = Confidence::Proven;
    Provenance provenance;
    bool known = false;       // false = "unknown" (top of the lattice)

    static Fact<T> unknown() { return Fact<T>{}; }

    static Fact<T> make(T v, Confidence c, Provenance p) {
        Fact<T> f;
        f.value = std::move(v);
        f.confidence = c;
        f.provenance = std::move(p);
        f.known = true;
        return f;
    }

    // Lattice join: refine `*this` with `other`. Returns true if `*this`
    // actually changed (so the caller can re-queue dependents).
    //
    // Confidence is an enum where LOWER numeric value = MORE trusted
    // (Proven=0 < Derived=1 < Estimated=2 < Profiled=3 < Speculative=4).
    // So when comparing confidences:
    //   - other < this  =>  other is MORE trusted  =>  overwrite
    //   - other > this  =>  other is LESS trusted  =>  keep this
    //   - other == this =>  same trust; last-writer-wins on value mismatch
    bool join(const Fact<T>& other) {
        if (!other.known) return false;
        if (!known) {
            *this = other;
            return true;
        }
        // Both known. Lower confidence enum value = more trusted.
        if (other.confidence < this->confidence) {
            *this = other;
            return true;
        }
        if (other.confidence > this->confidence) return false;
        // Same confidence. If values are equal, no change.
        // Otherwise, last-writer-wins (we update to `other`).
        if (!(value == other.value)) {
            *this = other;
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Bound - a symbolic or constant bound on a dimension.
//
// A dimension can be:
//   - exactly 1024                 (exact = 1024, lower = upper = 1024)
//   - exactly N                    (exact = N, lower/upper unknown)
//   - in [1, ∞)                    (exact unknown, lower = 1, upper = ∞)
//   - in [N, M]                    (exact unknown, lower = N, upper = M)
//
// This is much richer than "known/unknown" and lets the compiler answer
// questions like "is this dimension positive?" or "is it a multiple of 32?"
// ---------------------------------------------------------------------------
struct Bound {
    DimExprPtr exact;        // if known exactly
    DimExprPtr lower;        // lower bound (inclusive)
    DimExprPtr upper;        // upper bound (inclusive)

    bool is_exact_constant() const {
        return exact && exact->is_constant();
    }
    i64 exact_value() const {
        return exact && exact->is_constant() ? exact->value : 0;
    }
};

// ---------------------------------------------------------------------------
// Dimension - exact value + lower/upper bounds + divisibility constraints.
//
// Examples:
//   M = 1024                  -> exact=1024, lower=1024, upper=1024, divides={32}
//   N                         -> exact=N, lower=1, upper=∞, divides={}
//   K = N + 1                 -> exact=N+1, lower=2, upper=∞, divides={}
//   tile_m in [16, 256]       -> exact unknown, lower=16, upper=256, divides={16}
// ---------------------------------------------------------------------------
struct Dimension {
    Bound bound;
    // Set of constants known to divide this dimension (e.g. {16, 32} for
    // a tile dimension that's a multiple of both warp size and MMA K).
    std::vector<i64> known_divides;

    bool is_constant() const { return bound.is_exact_constant(); }
    i64 value() const { return bound.exact_value(); }

    bool operator==(const Dimension& o) const {
        // Structural equality on exact bound; lower/upper compared by pointer.
        if (bound.exact.get() != o.bound.exact.get()) {
            if (!bound.exact || !o.bound.exact) return false;
            if (!bound.exact->structurally_equal(*o.bound.exact)) return false;
        }
        return known_divides == o.known_divides;
    }

    // Lattice join: take the tightest bounds.
    bool join(const Dimension& o) {
        bool changed = false;
        // Prefer exact.
        if (!bound.exact && o.bound.exact) {
            bound.exact = o.bound.exact;
            changed = true;
        }
        // Tighten lower.
        if (o.bound.lower) {
            if (!bound.lower) { bound.lower = o.bound.lower; changed = true; }
            // Symbolic comparison is hard; we conservatively only update
            // when both are constant.
            else if (o.bound.lower->is_constant() && bound.lower->is_constant() &&
                     o.bound.lower->value > bound.lower->value) {
                bound.lower = o.bound.lower;
                changed = true;
            }
        }
        // Tighten upper.
        if (o.bound.upper) {
            if (!bound.upper) { bound.upper = o.bound.upper; changed = true; }
            else if (o.bound.upper->is_constant() && bound.upper->is_constant() &&
                     o.bound.upper->value < bound.upper->value) {
                bound.upper = o.bound.upper;
                changed = true;
            }
        }
        // Merge divides.
        for (i64 d : o.known_divides) {
            bool found = false;
            for (i64 e : known_divides) if (e == d) { found = true; break; }
            if (!found) { known_divides.push_back(d); changed = true; }
        }
        return changed;
    }
};

// ---------------------------------------------------------------------------
// ValueRange - sign + magnitude information for tensor element values.
//
// Lets the compiler simplify things like:
//   relu(x) -> x    when x >= 0
//   abs(x)  -> x    when x >= 0
//   log(x)  -> safe when x > 0
// ---------------------------------------------------------------------------
struct ValueRange {
    // Lower/upper bounds on element values. None = -inf / +inf.
    std::optional<double> lower;
    std::optional<double> upper;

    bool possibly_negative = true;
    bool possibly_zero = true;
    bool possibly_nan = true;
    bool possibly_inf = true;

    // Convenience predicates.
    bool is_non_negative() const {
        return lower.has_value() && *lower >= 0.0;
    }
    bool is_strictly_positive() const {
        return lower.has_value() && *lower > 0.0;
    }
    bool is_non_positive() const {
        return upper.has_value() && *upper <= 0.0;
    }
    bool is_strictly_negative() const {
        return upper.has_value() && *upper < 0.0;
    }
    bool is_constant_zero() const {
        return lower.has_value() && upper.has_value() &&
               *lower == 0.0 && *upper == 0.0;
    }
    bool is_constant_one() const {
        return lower.has_value() && upper.has_value() &&
               *lower == 1.0 && *upper == 1.0;
    }

    bool operator==(const ValueRange& o) const {
        return lower == o.lower && upper == o.upper &&
               possibly_negative == o.possibly_negative &&
               possibly_zero == o.possibly_zero &&
               possibly_nan == o.possibly_nan &&
               possibly_inf == o.possibly_inf;
    }

    // Lattice join: take the union (wider range).
    bool join(const ValueRange& o) {
        bool changed = false;
        if (o.lower.has_value()) {
            if (!lower.has_value() || *o.lower < *lower) {
                lower = o.lower;
                changed = true;
            }
        } else if (lower.has_value()) {
            // `o` says -inf; widen.
            lower.reset();
            changed = true;
        }
        if (o.upper.has_value()) {
            if (!upper.has_value() || *o.upper > *upper) {
                upper = o.upper;
                changed = true;
            }
        } else if (upper.has_value()) {
            upper.reset();
            changed = true;
        }
        // OR the flags (if either side says "possibly", we keep "possibly").
        changed = (possibly_negative |= o.possibly_negative) || changed;
        changed = (possibly_zero     |= o.possibly_zero)     || changed;
        changed = (possibly_nan      |= o.possibly_nan)      || changed;
        changed = (possibly_inf      |= o.possibly_inf)      || changed;
        return changed;
    }
};

// ---------------------------------------------------------------------------
// TensorProperty - bitset lattice.
//
// A tensor can be Constant + Diagonal + Sparse simultaneously. We use a
// bitset so all properties compose.
// ---------------------------------------------------------------------------
enum class TensorProperty : u32 {
    None             = 0,
    Constant         = 1u << 0,   // all elements equal a known scalar
    Zero             = 1u << 1,   // all elements are 0
    One              = 1u << 2,   // all elements are 1
    Identity         = 1u << 3,   // identity matrix
    Diagonal         = 1u << 4,   // only diagonal elements are non-zero
    Symmetric        = 1u << 5,   // A == A^T
    Permutation      = 1u << 6,   // each row/col has exactly one 1
    BroadcastConst   = 1u << 7,   // broadcast of a scalar
    Sparse           = 1u << 8,   // >50% zeros
    BlockSparse      = 1u << 9,   // structured block sparsity
    TriangularLower  = 1u << 10,  // lower triangular
    TriangularUpper  = 1u << 11,  // upper triangular
    Dense            = 1u << 12,  // known to be dense (no special structure)
};

inline TensorProperty operator|(TensorProperty a, TensorProperty b) {
    return static_cast<TensorProperty>(
        static_cast<u32>(a) | static_cast<u32>(b));
}

inline TensorProperty operator&(TensorProperty a, TensorProperty b) {
    return static_cast<TensorProperty>(
        static_cast<u32>(a) & static_cast<u32>(b));
}

inline bool has_property(TensorProperty set, TensorProperty p) {
    return (static_cast<u32>(set) & static_cast<u32>(p)) != 0;
}

inline std::string property_to_string(TensorProperty p) {
    std::string out;
    if (has_property(p, TensorProperty::Constant))        out += "constant|";
    if (has_property(p, TensorProperty::Zero))            out += "zero|";
    if (has_property(p, TensorProperty::One))             out += "one|";
    if (has_property(p, TensorProperty::Identity))        out += "identity|";
    if (has_property(p, TensorProperty::Diagonal))        out += "diagonal|";
    if (has_property(p, TensorProperty::Symmetric))       out += "symmetric|";
    if (has_property(p, TensorProperty::Permutation))     out += "permutation|";
    if (has_property(p, TensorProperty::BroadcastConst))  out += "broadcast_const|";
    if (has_property(p, TensorProperty::Sparse))          out += "sparse|";
    if (has_property(p, TensorProperty::BlockSparse))     out += "block_sparse|";
    if (has_property(p, TensorProperty::TriangularLower)) out += "tril|";
    if (has_property(p, TensorProperty::TriangularUpper)) out += "triu|";
    if (has_property(p, TensorProperty::Dense))           out += "dense|";
    if (!out.empty()) out.pop_back(); // strip trailing '|'
    return out;
}

// ---------------------------------------------------------------------------
// AliasClass - storage-identity information.
// ---------------------------------------------------------------------------
enum class AliasKind : u8 {
    NoAlias,
    MayAlias,
    MustAlias,
};

struct AliasClass {
    AliasKind kind = AliasKind::NoAlias;
    u32 alias_set_id = 0;  // 0 = no alias set assigned

    bool operator==(const AliasClass& o) const {
        return kind == o.kind && alias_set_id == o.alias_set_id;
    }

    bool join(const AliasClass& o) {
        // Widen: NoAlias -> MayAlias -> MustAlias.
        // (MustAlias only if both are MustAlias with the same set.)
        if (o.kind == AliasKind::MustAlias && kind == AliasKind::MustAlias &&
            o.alias_set_id == alias_set_id) return false;
        if (static_cast<u8>(o.kind) > static_cast<u8>(kind)) {
            kind = o.kind;
            if (alias_set_id == 0) alias_set_id = o.alias_set_id;
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// ReductionInfo - reduction-specific facts.
// ---------------------------------------------------------------------------
struct ReductionInfo {
    bool is_reduction = false;
    std::vector<i32> reduction_axes;
    bool is_associative = false;
    bool is_commutative = false;
    double identity_value = 0.0;  // identity element (0 for sum, -inf for max)

    bool operator==(const ReductionInfo& o) const {
        return is_reduction == o.is_reduction &&
               reduction_axes == o.reduction_axes &&
               is_associative == o.is_associative &&
               is_commutative == o.is_commutative &&
               identity_value == o.identity_value;
    }

    bool join(const ReductionInfo& o) {
        if (!is_reduction && o.is_reduction) {
            *this = o;
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// DependenceKind - annotates edges in the tensor dependency graph.
// ---------------------------------------------------------------------------
enum class DependenceKind : u8 {
    Full,         // consumer needs the entire producer tensor
    Slice,        // consumer only needs a sub-range
    Reduction,    // consumer reduces the producer
    Broadcast,    // consumer broadcasts the producer
    LayoutOnly,   // only the layout changes (transpose/reshape/etc.)
};

struct DependenceEdge {
    ValueId producer;
    ValueId consumer;
    DependenceKind kind = DependenceKind::Full;
    // Optional: which dims of `producer` does `consumer` touch?
    std::vector<i32> dims_accessed;
};

// ---------------------------------------------------------------------------
// CacheBehavior - predicted cache residency for a tensor.
// ---------------------------------------------------------------------------
struct CacheBehavior {
    double l2_hit_rate = 0.0;        // 0..1
    double shared_reuse_factor = 1.0; // how many times each global byte is reused from shared mem
    bool fits_in_l2 = false;
    bool fits_in_shared = false;
    bool accumulator_in_registers = false;

    bool operator==(const CacheBehavior& o) const {
        return l2_hit_rate == o.l2_hit_rate &&
               shared_reuse_factor == o.shared_reuse_factor &&
               fits_in_l2 == o.fits_in_l2 &&
               fits_in_shared == o.fits_in_shared &&
               accumulator_in_registers == o.accumulator_in_registers;
    }

    bool join(const CacheBehavior& o) {
        bool c = false;
        if (o.l2_hit_rate > l2_hit_rate) {
            l2_hit_rate = o.l2_hit_rate;
            c = true;
        }
        if (o.shared_reuse_factor > shared_reuse_factor) {
            shared_reuse_factor = o.shared_reuse_factor;
            c = true;
        }
        fits_in_l2 |= o.fits_in_l2;
        fits_in_shared |= o.fits_in_shared;
        accumulator_in_registers |= o.accumulator_in_registers;
        return c;
    }
};

} // namespace cg
