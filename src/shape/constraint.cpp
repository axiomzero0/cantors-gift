// shape/constraint.cpp - to_string for constraints
#include "cg/shape/constraint.hpp"

#include <sstream>

namespace cg {

namespace {
std::string dim_to_string(const DimExprPtr& e) {
    if (!e) return "?";
    switch (e->kind) {
        case DimKind::Constant:
            return std::to_string(e->value);
        case DimKind::Symbol:
            return e->symbol.name.empty()
                ? ("v" + std::to_string(e->symbol.id))
                : std::string(e->symbol.name);
        case DimKind::Add: return "(" + dim_to_string(e->lhs) + " + " + dim_to_string(e->rhs) + ")";
        case DimKind::Sub: return "(" + dim_to_string(e->lhs) + " - " + dim_to_string(e->rhs) + ")";
        case DimKind::Mul: return "(" + dim_to_string(e->lhs) + " * " + dim_to_string(e->rhs) + ")";
        case DimKind::FloorDiv: return "(" + dim_to_string(e->lhs) + " / " + dim_to_string(e->rhs) + ")";
        case DimKind::CeilDiv:  return "ceildiv(" + dim_to_string(e->lhs) + ", " + dim_to_string(e->rhs) + ")";
        case DimKind::Mod:      return "(" + dim_to_string(e->lhs) + " % " + dim_to_string(e->rhs) + ")";
        case DimKind::Min:      return "min(" + dim_to_string(e->lhs) + ", " + dim_to_string(e->rhs) + ")";
        case DimKind::Max:      return "max(" + dim_to_string(e->lhs) + ", " + dim_to_string(e->rhs) + ")";
        case DimKind::Neg:      return "-(" + dim_to_string(e->lhs) + ")";
    }
    return "?";
}
}

std::string Constraint::to_string() const {
    std::ostringstream os;
    switch (kind) {
        case CmpKind::EQ: os << dim_to_string(lhs) << " == " << dim_to_string(rhs); break;
        case CmpKind::NE: os << dim_to_string(lhs) << " != " << dim_to_string(rhs); break;
        case CmpKind::LT: os << dim_to_string(lhs) << " < "  << dim_to_string(rhs); break;
        case CmpKind::LE: os << dim_to_string(lhs) << " <= " << dim_to_string(rhs); break;
        case CmpKind::GT: os << dim_to_string(lhs) << " > "  << dim_to_string(rhs); break;
        case CmpKind::GE: os << dim_to_string(lhs) << " >= " << dim_to_string(rhs); break;
        case CmpKind::ModEq: os << dim_to_string(lhs) << " % " << modulus << " == " << dim_to_string(rhs); break;
        case CmpKind::ModNe: os << dim_to_string(lhs) << " % " << modulus << " != " << dim_to_string(rhs); break;
    }
    return os.str();
}

} // namespace cg
