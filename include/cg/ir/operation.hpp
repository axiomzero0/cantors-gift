// ir/operation.hpp - IR operations
//
// An Operation is the central node of Tensor IR. It owns:
//   - a list of operand Values
//   - a list of result Values
//   - an AttributeDict
//   - an EffectSet
//   - OpTraits
//   - a parent Block (may be null)
//   - a source location (for diagnostics)
//
// Operations live inside a Block. They are not recursively owned by their
// operands; SSA implies that an operand's defining op lives earlier in the
// function (or is a block argument).
//
// We use intrusive linked lists to keep the parent Block's operation sequence
// cheap to mutate.
#pragma once

#include "cg/core/util.hpp"
#include "cg/ir/attributes.hpp"
#include "cg/ir/effects.hpp"
#include "cg/ir/traits.hpp"
#include "cg/ir/type.hpp"
#include "cg/ir/value.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cg {

class Block;
class Operation;

using OpId = u32;

// Source location, used for diagnostics. Files are interned elsewhere.
struct SourceLocation {
    StringRef file;
    u32 line = 0;
    u32 col  = 0;
};

// Opcode - identifies what the operation does. New opcodes are registered via
// the OpRegistry (see ir/ops.hpp). The numeric value is stable across the
// process lifetime so it can be used as a switch label.
using Opcode = u32;

// Reserved opcodes; user-registered ops start at 0x1000.
constexpr Opcode OP_INVALID       = 0;
constexpr Opcode OP_CONSTANT      = 1;
constexpr Opcode OP_INPUT         = 2;
constexpr Opcode OP_OUTPUT        = 3;
constexpr Opcode OP_ADD           = 4;
constexpr Opcode OP_SUB           = 5;
constexpr Opcode OP_MUL           = 6;
constexpr Opcode OP_DIV           = 7;
constexpr Opcode OP_NEG           = 8;
constexpr Opcode OP_RELU          = 9;
constexpr Opcode OP_GELU          = 10;
constexpr Opcode OP_SIGMOID       = 11;
constexpr Opcode OP_TANH          = 12;
constexpr Opcode OP_EXP           = 13;
constexpr Opcode OP_LOG           = 14;
constexpr Opcode OP_SQRT          = 15;
constexpr Opcode OP_MATMUL        = 16;
constexpr Opcode OP_BROADCAST     = 17;
constexpr Opcode OP_RESHAPE       = 18;
constexpr Opcode OP_TRANSPOSE     = 19;
constexpr Opcode OP_REDUCE_SUM    = 20;
constexpr Opcode OP_REDUCE_MAX    = 21;
constexpr Opcode OP_REDUCE_MEAN   = 22;
constexpr Opcode OP_GATHER        = 23;
constexpr Opcode OP_SCATTER       = 24;
constexpr Opcode OP_CONCAT        = 25;
constexpr Opcode OP_SLICE         = 26;
constexpr Opcode OP_CAST          = 27;
constexpr Opcode OP_COPY          = 28;
constexpr Opcode OP_SOFTMAX       = 29;
constexpr Opcode OP_LAYERNORM     = 30;
constexpr Opcode OP_BATCHNORM     = 31;
constexpr Opcode OP_CONV2D        = 32;
constexpr Opcode OP_RETURN        = 33;
constexpr Opcode OP_FUSE          = 34;     // marker op produced by fusion pass
constexpr Opcode OP_KERNEL_CALL   = 35;     // call into a lowered kernel
constexpr Opcode OP_ALLOC         = 36;     // raw allocation (Memory IR)
constexpr Opcode OP_FREE          = 37;     // raw deallocation
constexpr Opcode OP_REUSE         = 38;     // buffer reuse marker
constexpr Opcode OP_USER_BEGIN    = 0x1000;

class Operation {
public:
    OpId        id;
    Opcode      opcode;
    std::string name;            // optional, for debugging

    SmallVector<Value> operands;
    SmallVector<Value> results;

    AttributeDict attributes;
    EffectSet     effects;
    OpTraits      traits;

    Block* parent = nullptr;
    SourceLocation loc;

    // Intrusive doubly-linked list pointers within `parent`.
    Operation* prev = nullptr;
    Operation* next = nullptr;

    Operation(OpId id, Opcode op)
        : id(id), opcode(op) {}

    bool is_pure() const { return effects.is_pure(); }
    bool has_trait(OpTrait t) const { return traits.has(t); }

    Value result(usize i) const { return results[i]; }
    Value operand(usize i) const { return operands[i]; }

    usize num_operands() const { return operands.size(); }
    usize num_results() const  { return results.size(); }

    void replace_operand(usize i, Value v) { operands[i] = v; }

    // Use Block::remove to detach from parent.
};

} // namespace cg
