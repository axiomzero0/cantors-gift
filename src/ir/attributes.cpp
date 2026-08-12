// ir/attributes.cpp - attribute helpers
#include "cg/ir/attributes.hpp"

#include <sstream>

namespace cg {

bool Attribute::structurally_equal(const Attribute& other) const {
    if (kind != other.kind) return false;
    switch (kind) {
        case AttrKind::Integer: return integer == other.integer;
        case AttrKind::Float:   return real == other.real;
        case AttrKind::Bool:    return flag == other.flag;
        case AttrKind::String:  return str == other.str;
        case AttrKind::IntegerArray: return ints == other.ints;
        case AttrKind::BoolArray:    return bools == other.bools;
        case AttrKind::DType:        return dtype == other.dtype;
        case AttrKind::DTypeArray:   return dtypes == other.dtypes;
    }
    return false;
}

std::string Attribute::to_string() const {
    std::ostringstream os;
    switch (kind) {
        case AttrKind::Integer: os << integer; break;
        case AttrKind::Float:   os << real; break;
        case AttrKind::Bool:    os << (flag ? "true" : "false"); break;
        case AttrKind::String:  os << "\"" << str << "\""; break;
        case AttrKind::IntegerArray: {
            os << "[";
            for (usize i = 0; i < ints.size(); ++i) {
                if (i) os << ", ";
                os << ints[i];
            }
            os << "]";
            break;
        }
        case AttrKind::BoolArray: {
            os << "[";
            for (usize i = 0; i < bools.size(); ++i) {
                if (i) os << ", ";
                os << (bools[i] ? "true" : "false");
            }
            os << "]";
            break;
        }
        case AttrKind::DType: os << dtype_name(dtype); break;
        case AttrKind::DTypeArray: {
            os << "[";
            for (usize i = 0; i < dtypes.size(); ++i) {
                if (i) os << ", ";
                os << dtype_name(dtypes[i]);
            }
            os << "]";
            break;
        }
    }
    return os.str();
}

} // namespace cg
