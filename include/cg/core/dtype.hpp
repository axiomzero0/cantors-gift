// core/dtype.hpp - tensor element data types
#pragma once

#include "cg/core/util.hpp"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string_view>
#include <stdexcept>

namespace cg {

enum class DType : u8 {
    // Floating point
    F16,  // IEEE half
    BF16, // Brain float
    F32,  // IEEE single
    F64,  // IEEE double

    // Integer
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,

    // Boolean
    BOOL,

    // Index (used internally for gather/scatter indices)
    INDEX,

    // Complex
    CF32,
    CF64,
};

// Byte size of one element of the given dtype. Returns 0 for unsupported
// combinations (which should be treated as a programming error).
inline usize dtype_size(DType dt) {
    switch (dt) {
        case DType::BOOL: return 1;
        case DType::I8:   return 1;
        case DType::U8:   return 1;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::I16:  return 2;
        case DType::U16:  return 2;
        case DType::F32:  return 4;
        case DType::I32:  return 4;
        case DType::U32:  return 4;
        case DType::CF32: return 8;
        case DType::F64:  return 8;
        case DType::I64:  return 8;
        case DType::U64:  return 8;
        case DType::CF64: return 16;
        case DType::INDEX: return 8;
    }
    return 0;
}

inline std::string_view dtype_name(DType dt) {
    switch (dt) {
        case DType::BOOL: return "bool";
        case DType::I8:   return "i8";
        case DType::U8:   return "u8";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::I16:  return "i16";
        case DType::U16:  return "u16";
        case DType::F32:  return "f32";
        case DType::I32:  return "i32";
        case DType::U32:  return "u32";
        case DType::CF32: return "cf32";
        case DType::F64:  return "f64";
        case DType::I64:  return "i64";
        case DType::U64:  return "u64";
        case DType::CF64: return "cf64";
        case DType::INDEX: return "index";
    }
    return "?";
}

inline std::optional<DType> dtype_from_name(std::string_view s) {
    if (s == "bool") return DType::BOOL;
    if (s == "i8")   return DType::I8;
    if (s == "u8")   return DType::U8;
    if (s == "f16")  return DType::F16;
    if (s == "bf16") return DType::BF16;
    if (s == "i16")  return DType::I16;
    if (s == "u16")  return DType::U16;
    if (s == "f32")  return DType::F32;
    if (s == "i32")  return DType::I32;
    if (s == "u32")  return DType::U32;
    if (s == "cf32") return DType::CF32;
    if (s == "f64")  return DType::F64;
    if (s == "i64")  return DType::I64;
    if (s == "u64")  return DType::U64;
    if (s == "cf64") return DType::CF64;
    if (s == "index") return DType::INDEX;
    return std::nullopt;
}

inline bool is_float(DType dt) {
    return dt == DType::F16 || dt == DType::BF16 ||
           dt == DType::F32 || dt == DType::F64 ||
           dt == DType::CF32 || dt == DType::CF64;
}

inline bool is_int(DType dt) {
    return dt == DType::I8  || dt == DType::I16 ||
           dt == DType::I32 || dt == DType::I64 ||
           dt == DType::U8  || dt == DType::U16 ||
           dt == DType::U32 || dt == DType::U64 ||
           dt == DType::INDEX;
}

inline bool is_signed(DType dt) {
    switch (dt) {
        case DType::I8: case DType::I16: case DType::I32: case DType::I64:
        case DType::F16: case DType::BF16: case DType::F32: case DType::F64:
        case DType::CF32: case DType::CF64: case DType::INDEX:
            return true;
        case DType::BOOL: case DType::U8: case DType::U16:
        case DType::U32: case DType::U64:
            return false;
    }
    return false;
}

// Promote two dtypes to a common one following a simplified C-style rule
// (no surprises in the tensor compiler).
inline DType promote(DType a, DType b) {
    if (a == b) return a;
    // Always up-cast integers to floats if either side is float
    if (is_float(a) && !is_float(b)) return a;
    if (is_float(b) && !is_float(a)) return b;
    if (is_float(a) && is_float(b)) {
        // bigger wins
        return dtype_size(a) >= dtype_size(b) ? a : b;
    }
    // both ints: bigger wins, signed wins ties if either is signed
    if (dtype_size(a) != dtype_size(b))
        return dtype_size(a) > dtype_size(b) ? a : b;
    return is_signed(a) ? a : b;
}

} // namespace cg
