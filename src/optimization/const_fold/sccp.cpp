// optimization/sccp.cpp - sparse conditional constant propagation
#include "cg/optimization/const_fold/sccp.hpp"
#include "cg/ir/ops.hpp"

#include <unordered_map>

namespace cg {

PreservedAnalyses SCCPPass::run(Module& m, AnalysisManager&) {
    bool changed = false;

    // Track which values are known constants (single scalar int for now).
    std::unordered_map<ValueId, i64> const_ints;

    // Seed: constant ops.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.opcode != OP_CONSTANT) continue;
            if (op.results.empty()) continue;
            auto shape_attr = op.attributes.get("shape");
            auto dtype_attr = op.attributes.get("dtype");
            auto bytes_attr = op.attributes.get("bytes");
            if (!shape_attr || !dtype_attr || !bytes_attr) continue;
            if (shape_attr->kind != AttrKind::IntegerArray) continue;
            // Only fold scalar constants.
            if (shape_attr->ints.size() != 0 && shape_attr->ints.size() != 1) continue;
            if (dtype_attr->kind != AttrKind::DType) continue;
            if (bytes_attr->kind != AttrKind::String) continue;
            const std::string& bytes = bytes_attr->str;
            switch (dtype_attr->dtype) {
                case DType::I32: {
                    if (bytes.size() == 4) {
                        i32 v; std::memcpy(&v, bytes.data(), 4);
                        const_ints[op.results[0].id()] = static_cast<i64>(v);
                    }
                    break;
                }
                case DType::I64: {
                    if (bytes.size() == 8) {
                        i64 v; std::memcpy(&v, bytes.data(), 8);
                        const_ints[op.results[0].id()] = v;
                    }
                    break;
                }
                default: break;
            }
        }
    }

    // Propagate through arithmetic ops.
    bool iterate = true;
    while (iterate) {
        iterate = false;
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                if (op.results.empty()) continue;
                ValueId rid = op.results[0].id();
                if (const_ints.count(rid)) continue;

                if (op.opcode == OP_ADD && op.operands.size() == 2) {
                    auto a = const_ints.find(op.operands[0].id());
                    auto b = const_ints.find(op.operands[1].id());
                    if (a != const_ints.end() && b != const_ints.end()) {
                        const_ints[rid] = a->second + b->second;
                        iterate = true;
                        changed = true;
                    }
                } else if (op.opcode == OP_MUL && op.operands.size() == 2) {
                    auto a = const_ints.find(op.operands[0].id());
                    auto b = const_ints.find(op.operands[1].id());
                    if (a != const_ints.end() && b != const_ints.end()) {
                        const_ints[rid] = a->second * b->second;
                        iterate = true;
                        changed = true;
                    }
                } else if (op.opcode == OP_SUB && op.operands.size() == 2) {
                    auto a = const_ints.find(op.operands[0].id());
                    auto b = const_ints.find(op.operands[1].id());
                    if (a != const_ints.end() && b != const_ints.end()) {
                        const_ints[rid] = a->second - b->second;
                        iterate = true;
                        changed = true;
                    }
                }
            }
        }
    }

    // Replace constant-computed results with explicit constant ops.
    // (We don't materialize new constant ops here; canonicalization +
    // constant folding handles that. SCCP just establishes the facts.)

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
