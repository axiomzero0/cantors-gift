// optimization/canonicalize.cpp - canonicalization implementation
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/ir/ops.hpp"

#include <algorithm>

namespace cg {

namespace {

// Stable key for operand ordering of commutative ops.
u64 operand_key(const Value& v) {
    return static_cast<u64>(v.id());
}

// Returns true if `v` is defined by a `constant` op.
bool is_constant_value(const Value& v) {
    // We don't have back-pointers from Value to its defining op here; the
    // canonicalization operates by walking the block, so by the time we look
    // at an operand it has already been visited and we have a pointer to its
    // defining operation in the block. We pass that in.
    return false;
}

bool is_constant_op(const Operation& op) {
    return op.opcode == OP_CONSTANT;
}

// Returns the constant integer stored in a constant op (for scalar constants).
// For multi-element tensors we do not fold here.
std::optional<i64> get_scalar_int(const Operation& op) {
    if (op.opcode != OP_CONSTANT) return std::nullopt;
    auto shape_attr = op.attributes.get("shape");
    if (!shape_attr || shape_attr->kind != AttrKind::IntegerArray) return std::nullopt;
    if (shape_attr->ints.size() != 1 && shape_attr->ints.size() != 0) return std::nullopt;
    auto bytes_attr = op.attributes.get("bytes");
    if (!bytes_attr || bytes_attr->kind != AttrKind::String) return std::nullopt;
    auto dtype_attr = op.attributes.get("dtype");
    if (!dtype_attr || dtype_attr->kind != AttrKind::DType) return std::nullopt;
    const std::string& bytes = bytes_attr->str;
    switch (dtype_attr->dtype) {
        case DType::I8: case DType::U8:
            if (bytes.size() != 1) return std::nullopt;
            return static_cast<i64>(static_cast<u8>(bytes[0]));
        case DType::I16: case DType::U16:
            if (bytes.size() != 2) return std::nullopt;
            i16 v16; std::memcpy(&v16, bytes.data(), 2);
            return static_cast<i64>(v16);
        case DType::I32: case DType::U32:
            if (bytes.size() != 4) return std::nullopt;
            i32 v32; std::memcpy(&v32, bytes.data(), 4);
            return static_cast<i64>(v32);
        case DType::I64: case DType::U64:
            if (bytes.size() != 8) return std::nullopt;
            i64 v64; std::memcpy(&v64, bytes.data(), 8);
            return v64;
        default:
            return std::nullopt;
    }
}

// Returns true if the op is a constant whose every byte is zero.
bool is_constant_zero_tensor(const Operation& op) {
    if (op.opcode != OP_CONSTANT) return false;
    auto bytes_attr = op.attributes.get("bytes");
    if (!bytes_attr || bytes_attr->kind != AttrKind::String) return false;
    const std::string& bytes = bytes_attr->str;
    if (bytes.empty()) return false;
    for (char c : bytes) if (c != '\0') return false;
    return true;
}

// Returns true if the op is a constant whose every element is 1 (for any
// numeric dtype).
bool is_constant_one_tensor(const Operation& op) {
    if (op.opcode != OP_CONSTANT) return false;
    auto dtype_attr = op.attributes.get("dtype");
    auto bytes_attr = op.attributes.get("bytes");
    if (!dtype_attr || !bytes_attr ||
        dtype_attr->kind != AttrKind::DType ||
        bytes_attr->kind != AttrKind::String) return false;
    const std::string& bytes = bytes_attr->str;
    if (bytes.empty()) return false;
    DType dt = dtype_attr->dtype;
    usize elem_size = dtype_size(dt);
    if (bytes.size() % elem_size != 0) return false;
    usize n = bytes.size() / elem_size;
    auto is_one = [&](u64 idx) -> bool {
        const char* p = bytes.data() + idx * elem_size;
        switch (dt) {
            case DType::F32: { float v; std::memcpy(&v, p, 4); return v == 1.0f; }
            case DType::F64: { double v; std::memcpy(&v, p, 8); return v == 1.0; }
            case DType::I8:  { i8 v; std::memcpy(&v, p, 1); return v == 1; }
            case DType::U8:  { u8 v; std::memcpy(&v, p, 1); return v == 1; }
            case DType::I16: { i16 v; std::memcpy(&v, p, 2); return v == 1; }
            case DType::U16: { u16 v; std::memcpy(&v, p, 2); return v == 1; }
            case DType::I32: { i32 v; std::memcpy(&v, p, 4); return v == 1; }
            case DType::U32: { u32 v; std::memcpy(&v, p, 4); return v == 1; }
            case DType::I64: { i64 v; std::memcpy(&v, p, 8); return v == 1; }
            case DType::U64: { u64 v; std::memcpy(&v, p, 8); return v == 1; }
            default: return false;
        }
    };
    for (u64 i = 0; i < n; ++i) if (!is_one(i)) return false;
    return true;
}

// Replaces all uses of `old` with `new_v` in the module.
void replace_all_uses(Module& m, Value old, Value new_v) {
    m.replace_all_uses(old, new_v);
}

// Try to canonicalize a single operation. Returns true if the IR was changed.
bool canonicalize_op(Module& m, Block& block, Operation& op) {
    switch (op.opcode) {
        case OP_ADD: {
            // x + 0 -> x ; 0 + x -> x
            // We need pointers to the defining ops of the operands.
            // We do this by scanning the block to find them. (Cheap because
            // we walk once per canonicalize pass and the block is small.)
            // For now, we only fold when one operand is a constant zero.
            // Lookup operand 0's defining op:
            Operation* lhs = nullptr;
            Operation* rhs = nullptr;
            for (auto& cand : block) {
                if (!lhs && !cand.results.empty() && cand.results[0] == op.operands[0])
                    lhs = &cand;
                if (!rhs && !cand.results.empty() && cand.results[0] == op.operands[1])
                    rhs = &cand;
            }
            if (lhs && is_constant_op(*lhs)) {
                auto v = get_scalar_int(*lhs);
                bool zero_scalar = v && *v == 0;
                bool zero_tensor = is_constant_zero_tensor(*lhs);
                if (zero_scalar || zero_tensor) {
                    replace_all_uses(m, op.results[0], op.operands[1]);
                    return true;
                }
            }
            if (rhs && is_constant_op(*rhs)) {
                auto v = get_scalar_int(*rhs);
                bool zero_scalar = v && *v == 0;
                bool zero_tensor = is_constant_zero_tensor(*rhs);
                if (zero_scalar || zero_tensor) {
                    replace_all_uses(m, op.results[0], op.operands[0]);
                    return true;
                }
            }
            // Commutative reorder
            if (operand_key(op.operands[0]) > operand_key(op.operands[1])) {
                std::swap(op.operands[0], op.operands[1]);
                return true;
            }
            break;
        }
        case OP_MUL: {
            Operation* lhs = nullptr;
            Operation* rhs = nullptr;
            for (auto& cand : block) {
                if (!lhs && !cand.results.empty() && cand.results[0] == op.operands[0])
                    lhs = &cand;
                if (!rhs && !cand.results.empty() && cand.results[0] == op.operands[1])
                    rhs = &cand;
            }
            if (lhs && is_constant_op(*lhs)) {
                auto v = get_scalar_int(*lhs);
                bool one_scalar = v && *v == 1;
                bool one_tensor = is_constant_one_tensor(*lhs);
                bool zero_scalar = v && *v == 0;
                bool zero_tensor = is_constant_zero_tensor(*lhs);
                if (one_scalar || one_tensor) {
                    replace_all_uses(m, op.results[0], op.operands[1]);
                    return true;
                }
                if (zero_scalar || zero_tensor) {
                    // x * 0 -> 0
                    replace_all_uses(m, op.results[0], op.operands[0]);
                    return true;
                }
            }
            if (rhs && is_constant_op(*rhs)) {
                auto v = get_scalar_int(*rhs);
                bool one_scalar = v && *v == 1;
                bool one_tensor = is_constant_one_tensor(*rhs);
                bool zero_scalar = v && *v == 0;
                bool zero_tensor = is_constant_zero_tensor(*rhs);
                if (one_scalar || one_tensor) {
                    replace_all_uses(m, op.results[0], op.operands[0]);
                    return true;
                }
                if (zero_scalar || zero_tensor) {
                    replace_all_uses(m, op.results[0], op.operands[1]);
                    return true;
                }
            }
            if (operand_key(op.operands[0]) > operand_key(op.operands[1])) {
                std::swap(op.operands[0], op.operands[1]);
                return true;
            }
            break;
        }
        case OP_SUB: {
            Operation* rhs = nullptr;
            for (auto& cand : block) {
                if (!rhs && !cand.results.empty() && cand.results[0] == op.operands[1])
                    rhs = &cand;
            }
            if (rhs && is_constant_op(*rhs)) {
                auto v = get_scalar_int(*rhs);
                bool zero_scalar = v && *v == 0;
                bool zero_tensor = is_constant_zero_tensor(*rhs);
                if (zero_scalar || zero_tensor) {
                    replace_all_uses(m, op.results[0], op.operands[0]);
                    return true;
                }
            }
            break;
        }
        case OP_TRANSPOSE: {
            // transpose(transpose(x, p), inverse(p)) -> x
            Operation* operand = nullptr;
            for (auto& cand : block) {
                if (!operand && !cand.results.empty() && cand.results[0] == op.operands[0])
                    operand = &cand;
            }
            if (operand && operand->opcode == OP_TRANSPOSE) {
                auto outer_perm = op.attributes.get("perm");
                auto inner_perm = operand->attributes.get("perm");
                if (outer_perm && inner_perm &&
                    outer_perm->kind == AttrKind::IntegerArray &&
                    inner_perm->kind == AttrKind::IntegerArray) {
                    // Check: outer_perm ∘ inner_perm == identity
                    bool identity = true;
                    if (outer_perm->ints.size() == inner_perm->ints.size()) {
                        for (usize i = 0; i < outer_perm->ints.size(); ++i) {
                            if (outer_perm->ints[inner_perm->ints[i]] != static_cast<i64>(i)) {
                                identity = false; break;
                            }
                        }
                    } else {
                        identity = false;
                    }
                    if (identity) {
                        replace_all_uses(m, op.results[0], operand->operands[0]);
                        return true;
                    }
                }
            }
            break;
        }
        default:
            break;
    }
    return false;
}

} // namespace

PreservedAnalyses CanonicalizePass::run(Module& m, AnalysisManager&) {
    bool changed = false;
    // Iterate to fixpoint.
    for (int iter = 0; iter < 8; ++iter) {
        bool iter_changed = false;
        for (auto& f : m.functions()) {
            // Collect ops to remove (replace_all_uses doesn't delete the op).
            std::vector<Operation*> to_remove;
            for (auto it = f->entry()->begin(); it != f->entry()->end(); ++it) {
                Operation& op = *it;
                if (canonicalize_op(m, *f->entry(), op)) {
                    iter_changed = true;
                }
            }
            (void)to_remove;
        }
        if (!iter_changed) break;
        changed = true;
    }

    PreservedAnalyses pa;
    if (!changed) return PreservedAnalyses::all();
    // Canonicalization may change operands, results, and shape/layout info.
    // It does NOT change the global shape constraint set.
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
