// numerical/semantics.hpp - floating-point numerical semantics
//
// Controls how the optimizer treats floating-point operations:
//
//   Strict:   full IEEE 754 compliance. No reassociation, no contraction,
//             no fast-math. (a+b)+c != a+(b+c) always.
//   Relaxed:  allows reassociation and contraction within a single kernel,
//             but preserves NaN/Inf behavior.
//   FastMath: allows any algebraic transformation that is valid for real
//             numbers, including assuming no NaNs/Infs, reassociation,
//             contraction, and reciprocal approximations.
//
// The numerical mode is a Module-level attribute. Passes check it before
// applying transformations.
#pragma once

#include "cg/core/util.hpp"

#include <string>

namespace cg {

enum class NumericalMode : u8 {
    Strict,     // full IEEE 754
    Relaxed,    // reassociation + contraction allowed
    FastMath,   // assume no NaN/Inf, allow all algebraic rewrites
};

inline std::string_view numerical_mode_name(NumericalMode m) {
    switch (m) {
        case NumericalMode::Strict:  return "strict";
        case NumericalMode::Relaxed: return "relaxed";
        case NumericalMode::FastMath: return "fast_math";
    }
    return "?";
}

// True iff reassociation (a+b)+c -> a+(b+c) is legal under the mode.
inline bool allows_reassociation(NumericalMode m) {
    return m == NumericalMode::Relaxed || m == NumericalMode::FastMath;
}

// True iff contraction (a*b + c -> fma(a,b,c)) is legal.
inline bool allows_contraction(NumericalMode m) {
    return m == NumericalMode::Relaxed || m == NumericalMode::FastMath;
}

// True iff we may assume no NaNs or Infs exist.
inline bool assumes_no_nan(NumericalMode m) {
    return m == NumericalMode::FastMath;
}

// True iff reciprocal approximation (1/x -> rcp(x)) is legal.
inline bool allows_reciprocal_approx(NumericalMode m) {
    return m == NumericalMode::FastMath;
}

// True iff we may replace x*0 -> 0 (unsafe if x is NaN/Inf).
inline bool allows_mul_zero_elimination(NumericalMode m) {
    return m == NumericalMode::FastMath;
}

} // namespace cg
