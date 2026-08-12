// ir/ops.hpp - operation registry and information
//
// The OpRegistry is the source of truth for:
//   - opcode -> name
//   - name   -> opcode
//   - opcode -> OpTraits
//   - opcode -> default EffectSet
//   - opcode -> "infer shape" function
//   - opcode -> "infer dtype" function
//   - opcode -> canonicalization rule
//
// Built-in opcodes (OP_ADD, OP_MATMUL, ...) are pre-registered at process
// start. User opcodes (>= OP_USER_BEGIN) can be registered dynamically.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/ir/attributes.hpp"
#include "cg/ir/effects.hpp"
#include "cg/ir/operation.hpp"
#include "cg/ir/traits.hpp"
#include "cg/ir/type.hpp"
#include "cg/shape/inference.hpp"

#include <functional>
#include <string>
#include <unordered_map>

namespace cg {

struct OpInfo {
    Opcode opcode;
    std::string name;
    OpTraits traits;
    EffectSet effects;

    // Type inference: given operand types + attributes, produce result types.
    // On failure, returns an empty vector and sets *err.
    std::function<std::vector<TypePtr>(
        Span<const TypePtr> operands,
        const AttributeDict& attrs,
        std::string* err)> infer_types;

    // Canonicalization hook (optional). Returns true if the IR was modified.
    std::function<bool(Operation&)> canonicalize;
};

class OpRegistry {
public:
    static OpRegistry& instance();

    void register_op(OpInfo info);

    const OpInfo* lookup(Opcode op) const;
    const OpInfo* lookup_by_name(StringRef name) const;

    OpTraits traits(Opcode op) const;
    EffectSet effects(Opcode op) const;

private:
    OpRegistry();
    void register_builtins();

    std::unordered_map<u32, OpInfo> by_opcode_;
    std::unordered_map<std::string, OpInfo*> by_name_;
};

// Helper: register a new user op at runtime.
Opcode register_user_op(OpInfo info);

} // namespace cg
