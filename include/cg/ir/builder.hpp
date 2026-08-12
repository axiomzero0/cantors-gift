// ir/builder.hpp - ergonomic construction of IR
//
// The Builder owns a `Function*` and emits operations into its entry block.
// It tracks the current insert position and produces Values for results.
//
// High-level convenience methods (add, mul, matmul, ...) consult the
// OpRegistry to infer result types and to attach the correct traits/effects.
#pragma once

#include "cg/ir/module.hpp"
#include "cg/ir/operation.hpp"
#include "cg/ir/ops.hpp"

#include <utility>
#include <vector>

namespace cg {

class Builder {
public:
    explicit Builder(Function* fn) : fn_(fn) {}

    Function* function() const { return fn_; }
    Block*    block()    const { return fn_->entry(); }

    // Create an Operation with `operands`, `attrs` and the given opcode. The
    // result types are inferred via OpRegistry. Returns a pointer to the new
    // operation (the Builder retains ownership in the block).
    Operation* create(Opcode opcode,
                      SmallVector<Value, 4> operands,
                      AttributeDict attrs = {});

    // Create an Operation with explicit result types (skips inference).
    Operation* create_with_results(Opcode opcode,
                                   SmallVector<Value, 4> operands,
                                   SmallVector<TypePtr, 2> result_types,
                                   AttributeDict attrs = {});

    // ---------- Convenience: constants and inputs ----------
    Value constant_tensor(Shape shape, DType dt, std::vector<u8> /*bytes*/ = {});
    Value input_tensor(Shape shape, DType dt);
    void  output_tensor(Value v);

    // ---------- Convenience: arithmetic ----------
    Value add(Value a, Value b);
    Value sub(Value a, Value b);
    Value mul(Value a, Value b);
    Value div(Value a, Value b);
    Value neg(Value a);

    // ---------- Convenience: elementwise ----------
    Value relu(Value a);
    Value gelu(Value a);
    Value exp(Value a);
    Value sqrt(Value a);

    // ---------- Convenience: tensor ops ----------
    Value matmul(Value a, Value b);
    Value broadcast(Value a, std::vector<i64> target_shape);
    Value reshape(Value a, std::vector<i64> target_shape);
    Value transpose(Value a, std::vector<i32> perm);
    Value reduce_sum(Value a, std::vector<i32> axes, bool keep_dims = false);
    Value reduce_max(Value a, std::vector<i32> axes, bool keep_dims = false);
    Value cast(Value a, DType target_dtype);

private:
    Function* fn_;
};

} // namespace cg
