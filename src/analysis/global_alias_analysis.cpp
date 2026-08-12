// analysis/global_alias_analysis.cpp
#include "cg/analysis/global_alias_analysis.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

void GlobalAliasAnalysis::compute() {
    Module& m = am_.module();
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.opcode == OP_TRANSPOSE ||
                op.opcode == OP_RESHAPE ||
                op.opcode == OP_SLICE ||
                op.opcode == OP_BROADCAST) {
                if (!op.operands.empty() && !op.results.empty()) {
                    views_[op.operands[0].id()].insert(op.results[0].id());
                    defining_view_[op.results[0].id()] = op.operands[0].id();
                }
            }
        }
    }
}

TensorAliasKind GlobalAliasAnalysis::alias(Value a, Value b) const {
    if (a == b) return TensorAliasKind::MustAlias;
    // Look for direct view/slice relationships.
    auto it = defining_view_.find(a.id());
    if (it != defining_view_.end() && it->second == b.id())
        return TensorAliasKind::ViewOf;
    it = defining_view_.find(b.id());
    if (it != defining_view_.end() && it->second == a.id())
        return TensorAliasKind::ViewOf;
    // Same root?
    auto root_of = [&](Value v) -> ValueId {
        ValueId cur = v.id();
        while (true) {
            auto it2 = defining_view_.find(cur);
            if (it2 == defining_view_.end()) return cur;
            cur = it2->second;
        }
    };
    if (root_of(a) == root_of(b) && root_of(a) != a.id())
        return TensorAliasKind::MayAlias;
    return TensorAliasKind::NoAlias;
}

} // namespace cg
