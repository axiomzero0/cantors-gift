// optimization/sccp.cpp - sparse conditional constant propagation
//
// Propagates constants through the IR and materializes the results as
// explicit constant ops. Unlike the previous version which only computed
// facts, this version actually replaces operations whose results are
// provably constant with OP_CONSTANT ops.
#include "cg/optimization/const_fold/sccp.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

#include <cmath>
#include <unordered_map>

namespace cg {

namespace {

// Build a constant tensor op from a scalar value.
Value materialize_constant(Function* f, i64 value, DType dt) {
    Builder b(f);
    std::string bytes(dtype_size(dt), '\0');
    switch (dt) {
        case DType::I32: {
            i32 v = static_cast<i32>(value);
            std::memcpy(bytes.data(), &v, 4);
            break;
        }
        case DType::I64: {
            std::memcpy(bytes.data(), &value, 8);
            bytes.resize(8);
            break;
        }
        case DType::F32: {
            float v = static_cast<float>(value);
            std::memcpy(bytes.data(), &v, 4);
            break;
        }
        case DType::F64: {
            double v = static_cast<double>(value);
            std::memcpy(bytes.data(), &v, 8);
            bytes.resize(8);
            break;
        }
        default: break;
    }
    AttributeDict attrs;
    attrs.set("shape", Attribute::make_int_array({}));
    attrs.set("dtype", Attribute::make_dtype(dt));
    attrs.set("bytes", Attribute::make_string(std::move(bytes)));
    auto* op = b.create(OP_CONSTANT, {}, attrs);
    return op->results[0];
}

} // namespace

PreservedAnalyses SCCPPass::run(Module& m, AnalysisManager&) {
    bool changed = false;

    // Track which values are known constants (scalar int for now).
    std::unordered_map<ValueId, std::pair<i64, DType>> const_vals;

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
            if (shape_attr->ints.size() != 0) continue;
            if (dtype_attr->kind != AttrKind::DType) continue;
            if (bytes_attr->kind != AttrKind::String) continue;
            const std::string& bytes = bytes_attr->str;
            DType dt = dtype_attr->dtype;
            switch (dt) {
                case DType::I32: {
                    if (bytes.size() == 4) {
                        i32 v; std::memcpy(&v, bytes.data(), 4);
                        const_vals[op.results[0].id()] = {static_cast<i64>(v), dt};
                    }
                    break;
                }
                case DType::I64: {
                    if (bytes.size() == 8) {
                        i64 v; std::memcpy(&v, bytes.data(), 8);
                        const_vals[op.results[0].id()] = {v, dt};
                    }
                    break;
                }
                case DType::F32: {
                    if (bytes.size() == 4) {
                        float v; std::memcpy(&v, bytes.data(), 4);
                        if (v == std::floor(v)) {
                            const_vals[op.results[0].id()] = {static_cast<i64>(v), dt};
                        }
                    }
                    break;
                }
                default: break;
            }
        }
    }

    // Propagate through arithmetic ops until fixpoint.
    bool iterate = true;
    while (iterate) {
        iterate = false;
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                if (op.results.empty()) continue;
                ValueId rid = op.results[0].id();
                if (const_vals.count(rid)) continue;

                if (op.opcode == OP_ADD && op.operands.size() == 2) {
                    auto a = const_vals.find(op.operands[0].id());
                    auto b = const_vals.find(op.operands[1].id());
                    if (a != const_vals.end() && b != const_vals.end()) {
                        const_vals[rid] = {a->second.first + b->second.first,
                                           a->second.second};
                        iterate = true;
                        changed = true;
                    }
                } else if (op.opcode == OP_MUL && op.operands.size() == 2) {
                    auto a = const_vals.find(op.operands[0].id());
                    auto b = const_vals.find(op.operands[1].id());
                    if (a != const_vals.end() && b != const_vals.end()) {
                        const_vals[rid] = {a->second.first * b->second.first,
                                           a->second.second};
                        iterate = true;
                        changed = true;
                    }
                } else if (op.opcode == OP_SUB && op.operands.size() == 2) {
                    auto a = const_vals.find(op.operands[0].id());
                    auto b = const_vals.find(op.operands[1].id());
                    if (a != const_vals.end() && b != const_vals.end()) {
                        const_vals[rid] = {a->second.first - b->second.first,
                                           a->second.second};
                        iterate = true;
                        changed = true;
                    }
                }
            }
        }
    }

    // Materialize: replace computed-constant results with explicit constant ops.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            ValueId rid = op.results[0].id();
            auto it = const_vals.find(rid);
            if (it == const_vals.end()) continue;
            // Don't replace existing constant ops.
            if (op.opcode == OP_CONSTANT) continue;

            // Create a new constant op and replace uses.
            Value new_const = materialize_constant(f.get(), it->second.first,
                                                    it->second.second);
            m.replace_all_uses(op.results[0], new_const);
            changed = true;
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
