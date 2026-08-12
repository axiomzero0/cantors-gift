// ir/module.cpp - module / function / operation helpers
//
// replace_all_uses is O(N) per call: it walks the module once and replaces
// all occurrences of `old` with `new_v` in operand lists.
#include "cg/ir/module.hpp"

namespace cg {

void Module::replace_all_uses(Value old, Value new_v) {
    const ValueId old_id = old.id();
    const ValueId new_id = new_v.id();
    for (auto& f : functions_) {
        for (auto& op : *f->entry()) {
            for (auto& operand : op.operands) {
                if (operand.id() == old_id) {
                    operand = new_v;
                }
            }
        }
        // Block arguments are immutable; their uses are in operations
        // and are caught above.
    }
    (void)new_id;
}

} // namespace cg
