// optimization/const_fold.cpp - constant folding implementation
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

#include <cstring>
#include <unordered_map>

namespace cg {

namespace {

bool get_constant_info(const Operation& op,
                       std::vector<i64>& shape,
                       DType& dt,
                       std::string& bytes) {
    if (op.opcode != OP_CONSTANT) return false;
    auto s = op.attributes.get("shape");
    auto d = op.attributes.get("dtype");
    auto b = op.attributes.get("bytes");
    if (!s || !d || !b) return false;
    if (s->kind != AttrKind::IntegerArray) return false;
    if (d->kind != AttrKind::DType) return false;
    if (b->kind != AttrKind::String) return false;
    shape = s->ints;
    dt = d->dtype;
    bytes = b->str;
    return true;
}

u64 numel(const std::vector<i64>& shape) {
    u64 n = 1;
    for (auto d : shape) n *= static_cast<u64>(d);
    return n;
}

template <typename T>
T read_scalar(const std::string& bytes, u64 idx) {
    T v;
    std::memcpy(&v, bytes.data() + idx * sizeof(T), sizeof(T));
    return v;
}

template <typename T>
void write_scalar(std::string& bytes, u64 idx, T v) {
    std::memcpy(bytes.data() + idx * sizeof(T), &v, sizeof(T));
}

template <typename T>
std::string fold_binary_arith(const std::string& a, const std::string& b,
                              u64 n, Opcode op) {
    std::string out(a.size(), '\0');
    for (u64 i = 0; i < n; ++i) {
        T x = read_scalar<T>(a, i);
        T y = read_scalar<T>(b, i);
        T r = T{};
        switch (op) {
            case OP_ADD: r = x + y; break;
            case OP_SUB: r = x - y; break;
            case OP_MUL: r = x * y; break;
            case OP_DIV: r = x / y; break;
            default: break;
        }
        write_scalar<T>(out, i, r);
    }
    return out;
}

std::string fold_binary(const std::string& a, const std::string& b,
                        DType dt, u64 n, Opcode op) {
    switch (dt) {
        case DType::F32: return fold_binary_arith<float>(a, b, n, op);
        case DType::F64: return fold_binary_arith<double>(a, b, n, op);
        case DType::I8:  return fold_binary_arith<i8>(a, b, n, op);
        case DType::I16: return fold_binary_arith<i16>(a, b, n, op);
        case DType::I32: return fold_binary_arith<i32>(a, b, n, op);
        case DType::I64: return fold_binary_arith<i64>(a, b, n, op);
        case DType::U8:  return fold_binary_arith<u8>(a, b, n, op);
        case DType::U16: return fold_binary_arith<u16>(a, b, n, op);
        case DType::U32: return fold_binary_arith<u32>(a, b, n, op);
        case DType::U64: return fold_binary_arith<u64>(a, b, n, op);
        default: return {};
    }
}

} // namespace

PreservedAnalyses ConstantFoldingPass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    for (auto& f : m.functions()) {
        std::unordered_map<ValueId, Operation*> defs;
        for (auto& op : *f->entry()) {
            if (!op.results.empty()) defs[op.results[0].id()] = &op;
        }

        std::vector<Operation*> to_remove;
        for (auto& op : *f->entry()) {
            if (op.opcode != OP_ADD && op.opcode != OP_SUB &&
                op.opcode != OP_MUL && op.opcode != OP_DIV)
                continue;

            // Both operands must be constants of the same dtype and shape.
            auto ldef = defs.find(op.operands[0].id());
            auto rdef = defs.find(op.operands[1].id());
            if (ldef == defs.end() || rdef == defs.end()) continue;

            std::vector<i64> ls, rs;
            DType ldt, rdt;
            std::string lb, rb;
            if (!get_constant_info(*ldef->second, ls, ldt, lb)) continue;
            if (!get_constant_info(*rdef->second, rs, rdt, rb)) continue;
            if (ldt != rdt) continue;
            if (ls != rs) continue;
            u64 n = numel(ls);
            if (n * dtype_size(ldt) > max_fold_bytes_) continue;

            std::string folded = fold_binary(lb, rb, ldt, n, op.opcode);
            if (folded.empty()) continue;

            // Build a new constant op replacing this one.
            AttributeDict attrs;
            attrs.set("shape", Attribute::make_int_array(ls));
            attrs.set("dtype", Attribute::make_dtype(ldt));
            attrs.set("bytes", Attribute::make_string(std::move(folded)));

            Builder b(f.get());
            // Emit a new constant before this op and replace uses.
            // We append at the end; canonicalization will reorder.
            // This is safe because constants are pure.
            auto* new_op = b.create(OP_CONSTANT, {}, attrs);
            m.replace_all_uses(op.results[0], new_op->results[0]);
            to_remove.push_back(&op);
            changed = true;
        }

        for (Operation* op : to_remove) {
            f->entry()->remove(op);
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
