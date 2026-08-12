// ir/module.cpp - module / function / operation helpers
#include "cg/ir/module.hpp"

namespace cg {

void Module::replace_all_uses(Value old, Value new_v) {
    for (auto& f : functions_) {
        for (auto& op : *f->entry()) {
            for (auto& operand : op.operands) {
                if (operand == old) operand = new_v;
            }
        }
    }
}

} // namespace cg
