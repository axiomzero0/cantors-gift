// schedule/domain.cpp
#include "cg/schedule/domain.hpp"

#include <sstream>

namespace cg {

std::string IterationDomain::to_string() const {
    std::ostringstream os;
    auto dim_str = [](const DimExprPtr& e) -> std::string {
        if (!e) return "?";
        if (e->is_constant()) return std::to_string(e->value);
        if (e->is_symbol()) return e->symbol.name.empty()
            ? "v" + std::to_string(e->symbol.id)
            : std::string(e->symbol.name);
        return "<expr>";
    };
    os << "[" << dim_str(lo) << ", " << dim_str(hi) << ")";
    if (step && !(step->is_constant() && step->value == 1))
        os << " step " << dim_str(step);
    return os.str();
}

} // namespace cg
