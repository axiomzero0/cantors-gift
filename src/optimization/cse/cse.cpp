// optimization/cse.cpp - common subexpression elimination
#include "cg/optimization/cse/cse.hpp"
#include "cg/ir/ops.hpp"

#include <unordered_map>

namespace cg {

namespace {

u64 hash_attributes(const AttributeDict& attrs) {
    u64 h = 0;
    for (auto& [k, v] : attrs) {
        hash_combine(h, std::hash<std::string>{}(k));
        hash_combine(h, static_cast<u64>(v->kind));
        switch (v->kind) {
            case AttrKind::Integer: hash_combine(h, std::hash<i64>{}(v->integer)); break;
            case AttrKind::Float:   hash_combine(h, std::hash<double>{}(v->real)); break;
            case AttrKind::Bool:    hash_combine(h, std::hash<bool>{}(v->flag)); break;
            case AttrKind::String:  hash_combine(h, std::hash<std::string>{}(v->str)); break;
            case AttrKind::IntegerArray:
                for (auto x : v->ints) hash_combine(h, std::hash<i64>{}(x));
                break;
            case AttrKind::BoolArray:
                for (auto x : v->bools) hash_combine(h, std::hash<bool>{}(x));
                break;
            case AttrKind::DType: hash_combine(h, std::hash<u8>{}(static_cast<u8>(v->dtype))); break;
            case AttrKind::DTypeArray:
                for (auto x : v->dtypes) hash_combine(h, std::hash<u8>{}(static_cast<u8>(x)));
                break;
        }
    }
    return h;
}

bool attributes_equal(const AttributeDict& a, const AttributeDict& b) {
    if (a.size() != b.size()) return false;
    auto ait = a.begin(); auto bit = b.begin();
    for (; ait != a.end() && bit != b.end(); ++ait, ++bit) {
        if (ait->first != bit->first) return false;
        if (!ait->second->structurally_equal(*bit->second)) return false;
    }
    return true;
}

u64 hash_op(const Operation& op) {
    u64 h = std::hash<u32>{}(op.opcode);
    for (auto& v : op.operands) hash_combine(h, std::hash<ValueId>{}(v.id()));
    hash_combine(h, hash_attributes(op.attributes));
    return h;
}

bool ops_equivalent(const Operation& a, const Operation& b) {
    if (a.opcode != b.opcode) return false;
    if (a.operands.size() != b.operands.size()) return false;
    for (usize i = 0; i < a.operands.size(); ++i)
        if (a.operands[i] != b.operands[i]) return false;
    if (!attributes_equal(a.attributes, b.attributes)) return false;
    if (a.results.size() != b.results.size()) return false;
    return true;
}

} // namespace

PreservedAnalyses CSEPass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    for (auto& f : m.functions()) {
        // Hash table: hash -> first operation with that hash.
        std::unordered_map<u64, Operation*> existing;
        std::vector<Operation*> to_remove;

        for (auto& op : *f->entry()) {
            if (!op.is_pure()) continue;
            // Only CSE ops with no attributes that contain mutable state.
            // For simplicity, we only dedupe ops with matching opcode+operands+attrs.
            u64 h = hash_op(op);
            auto it = existing.find(h);
            if (it == existing.end()) {
                existing[h] = &op;
                continue;
            }
            // Check for false positive.
            if (ops_equivalent(*it->second, op)) {
                // Replace all uses of `op.results` with `it->second->results`.
                for (usize i = 0; i < op.results.size(); ++i) {
                    m.replace_all_uses(op.results[i], it->second->results[i]);
                }
                to_remove.push_back(&op);
                changed = true;
            } else {
                // Hash collision: insert as a separate entry (we keep the
                // first match; future queries may collide, which is fine).
                existing[h] = &op;
            }
        }

        // Remove the dead ops.
        for (Operation* op : to_remove) {
            f->entry()->remove(op);
        }
    }

    PreservedAnalyses pa;
    if (!changed) return PreservedAnalyses::all();
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
