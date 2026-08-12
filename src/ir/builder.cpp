// ir/builder.cpp - implementation of the IR builder
#include "cg/ir/builder.hpp"

#include <stdexcept>

namespace cg {

namespace {

void attach_info(Operation* op) {
    auto* info = OpRegistry::instance().lookup(op->opcode);
    if (!info) return;
    op->traits = info->traits;
    op->effects = info->effects;
}

} // namespace

Operation* Builder::create_with_results(Opcode opcode,
                                        SmallVector<Value, 4> operands,
                                        SmallVector<TypePtr, 2> result_types,
                                        AttributeDict attrs) {
    auto op = std::make_unique<Operation>(fn_->allocate_op_id(), opcode);
    op->operands = std::move(operands);
    op->attributes = std::move(attrs);
    attach_info(op.get());
    for (auto& t : result_types) {
        ValueId vid = fn_->allocate_value_id();
        op->results.emplace_back(t, vid);
    }
    return fn_->entry()->append(std::move(op));
}

Operation* Builder::create(Opcode opcode,
                           SmallVector<Value, 4> operands,
                           AttributeDict attrs) {
    // Build operand type list.
    std::vector<TypePtr> operand_types;
    operand_types.reserve(operands.size());
    for (auto& v : operands) operand_types.push_back(v.type());

    std::string err;
    std::vector<TypePtr> result_types;
    if (auto* info = OpRegistry::instance().lookup(opcode)) {
        if (info->infer_types) {
            result_types = info->infer_types(make_span(operand_types), attrs, &err);
            if (!err.empty()) {
                throw std::invalid_argument(
                    "IR type inference failed for op " + std::to_string(opcode) + ": " + err);
            }
        }
    }
    SmallVector<TypePtr, 2> rt;
    for (auto& t : result_types) rt.push_back(t);
    return create_with_results(opcode, std::move(operands), std::move(rt), std::move(attrs));
}

Value Builder::constant_tensor(Shape shape, DType dt, std::vector<u8> bytes) {
    AttributeDict attrs;
    std::vector<i64> shape_vec;
    shape_vec.reserve(shape.rank());
    for (auto& d : shape) {
        if (!d->is_constant())
            throw std::invalid_argument("constant_tensor: shape must be static");
        shape_vec.push_back(d->value);
    }
    attrs.set("shape", Attribute::make_int_array(std::move(shape_vec)));
    attrs.set("dtype", Attribute::make_dtype(dt));
    if (!bytes.empty()) {
        // Bytes are referenced via attribute. For now we keep them as a string
        // attribute (raw bytes); a real impl would use a constant pool. We do
        // not assume the bytes are valid for any particular dtype.
        attrs.set("bytes", Attribute::make_string(
            std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size())));
    }
    auto* op = create(OP_CONSTANT, {}, std::move(attrs));
    return op->results[0];
}

Value Builder::input_tensor(Shape shape, DType dt) {
    AttributeDict attrs;
    std::vector<i64> shape_vec;
    shape_vec.reserve(shape.rank());
    for (auto& d : shape) {
        if (!d->is_constant())
            throw std::invalid_argument("input_tensor: shape must be static");
        shape_vec.push_back(d->value);
    }
    attrs.set("shape", Attribute::make_int_array(std::move(shape_vec)));
    attrs.set("dtype", Attribute::make_dtype(dt));
    auto* op = create(OP_INPUT, {}, std::move(attrs));
    return op->results[0];
}

void Builder::output_tensor(Value v) {
    create(OP_OUTPUT, {v}, {});
}

Value Builder::add(Value a, Value b) { return create(OP_ADD, {a, b})->results[0]; }
Value Builder::sub(Value a, Value b) { return create(OP_SUB, {a, b})->results[0]; }
Value Builder::mul(Value a, Value b) { return create(OP_MUL, {a, b})->results[0]; }
Value Builder::div(Value a, Value b) { return create(OP_DIV, {a, b})->results[0]; }
Value Builder::neg(Value a)          { return create(OP_NEG, {a})->results[0]; }
Value Builder::relu(Value a)         { return create(OP_RELU, {a})->results[0]; }
Value Builder::gelu(Value a)         { return create(OP_GELU, {a})->results[0]; }
Value Builder::exp(Value a)          { return create(OP_EXP, {a})->results[0]; }
Value Builder::sqrt(Value a)         { return create(OP_SQRT, {a})->results[0]; }

Value Builder::matmul(Value a, Value b) {
    return create(OP_MATMUL, {a, b})->results[0];
}

Value Builder::broadcast(Value a, std::vector<i64> target_shape) {
    AttributeDict attrs;
    attrs.set("shape", Attribute::make_int_array(std::move(target_shape)));
    return create(OP_BROADCAST, {a}, std::move(attrs))->results[0];
}

Value Builder::reshape(Value a, std::vector<i64> target_shape) {
    AttributeDict attrs;
    attrs.set("shape", Attribute::make_int_array(std::move(target_shape)));
    return create(OP_RESHAPE, {a}, std::move(attrs))->results[0];
}

Value Builder::transpose(Value a, std::vector<i32> perm) {
    AttributeDict attrs;
    std::vector<i64> perm64(perm.begin(), perm.end());
    attrs.set("perm", Attribute::make_int_array(std::move(perm64)));
    return create(OP_TRANSPOSE, {a}, std::move(attrs))->results[0];
}

Value Builder::reduce_sum(Value a, std::vector<i32> axes, bool keep_dims) {
    AttributeDict attrs;
    std::vector<i64> axes64(axes.begin(), axes.end());
    attrs.set("axes", Attribute::make_int_array(std::move(axes64)));
    attrs.set("keep_dims", Attribute::make_bool(keep_dims));
    return create(OP_REDUCE_SUM, {a}, std::move(attrs))->results[0];
}

Value Builder::reduce_max(Value a, std::vector<i32> axes, bool keep_dims) {
    AttributeDict attrs;
    std::vector<i64> axes64(axes.begin(), axes.end());
    attrs.set("axes", Attribute::make_int_array(std::move(axes64)));
    attrs.set("keep_dims", Attribute::make_bool(keep_dims));
    return create(OP_REDUCE_MAX, {a}, std::move(attrs))->results[0];
}

Value Builder::cast(Value a, DType target_dtype) {
    AttributeDict attrs;
    attrs.set("dtype", Attribute::make_dtype(target_dtype));
    return create(OP_CAST, {a}, std::move(attrs))->results[0];
}

} // namespace cg
