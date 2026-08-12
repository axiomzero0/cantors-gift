// ir/ops.cpp - registration of built-in ops
#include "cg/ir/ops.hpp"

#include <mutex>
#include <sstream>

namespace cg {

namespace {

// Shape inference helpers shared by built-in ops.
std::vector<TypePtr> infer_elementwise_binary_types(
    Span<const TypePtr> operands,
    const AttributeDict&,
    std::string* err) {
    if (operands.size() != 2) {
        if (err) *err = "expected 2 operands";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    auto b = std::dynamic_pointer_cast<const TensorType>(operands[1]);
    if (!a || !b) {
        if (err) *err = "operands must be tensors";
        return {};
    }
    auto r = infer_elementwise_binary(a->shape, b->shape);
    if (!r.ok) {
        if (err) *err = r.message;
        return {};
    }
    DType out_dt = promote(a->dtype, b->dtype);
    return { make_tensor_type(r.shape, out_dt, a->device) };
}

std::vector<TypePtr> infer_unary_same_types(
    Span<const TypePtr> operands,
    const AttributeDict&,
    std::string* err) {
    if (operands.size() != 1) {
        if (err) *err = "expected 1 operand";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    if (!a) {
        if (err) *err = "operand must be a tensor";
        return {};
    }
    return { a };
}

std::vector<TypePtr> infer_matmul_types(
    Span<const TypePtr> operands,
    const AttributeDict&,
    std::string* err) {
    if (operands.size() != 2) {
        if (err) *err = "matmul: expected 2 operands";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    auto b = std::dynamic_pointer_cast<const TensorType>(operands[1]);
    if (!a || !b) {
        if (err) *err = "matmul: operands must be tensors";
        return {};
    }
    auto r = infer_matmul(a->shape, b->shape);
    if (!r.ok) {
        if (err) *err = r.message;
        return {};
    }
    DType out_dt = promote(a->dtype, b->dtype);
    return { make_tensor_type(r.shape, out_dt, a->device) };
}

std::vector<TypePtr> infer_reduce_types(
    Span<const TypePtr> operands,
    const AttributeDict& attrs,
    std::string* err) {
    if (operands.size() != 1) {
        if (err) *err = "reduce: expected 1 operand";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    if (!a) {
        if (err) *err = "reduce: operand must be a tensor";
        return {};
    }
    auto axes_attr = attrs.get("axes");
    if (!axes_attr || axes_attr->kind != AttrKind::IntegerArray) {
        if (err) *err = "reduce: missing 'axes' attribute";
        return {};
    }
    auto keep_attr = attrs.get("keep_dims");
    bool keep = keep_attr && keep_attr->kind == AttrKind::Bool && keep_attr->flag;
    std::vector<i32> axes32;
    axes32.reserve(axes_attr->ints.size());
    for (i64 v : axes_attr->ints) axes32.push_back(static_cast<i32>(v));
    auto r = infer_reduction(a->shape, make_span(axes32), keep);
    if (!r.ok) {
        if (err) *err = r.message;
        return {};
    }
    return { make_tensor_type(r.shape, a->dtype, a->device) };
}

std::vector<TypePtr> infer_broadcast_types(
    Span<const TypePtr> operands,
    const AttributeDict& attrs,
    std::string* err) {
    if (operands.size() != 1) {
        if (err) *err = "broadcast: expected 1 operand";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    if (!a) {
        if (err) *err = "broadcast: operand must be a tensor";
        return {};
    }
    auto shape_attr = attrs.get("shape");
    if (!shape_attr || shape_attr->kind != AttrKind::IntegerArray) {
        if (err) *err = "broadcast: missing 'shape' attribute";
        return {};
    }
    Shape target = Shape::from_constants(shape_attr->ints);
    auto r = infer_broadcast(a->shape, target);
    if (!r.ok) {
        if (err) *err = r.message;
        return {};
    }
    return { make_tensor_type(r.shape, a->dtype, a->device) };
}

std::vector<TypePtr> infer_reshape_types(
    Span<const TypePtr> operands,
    const AttributeDict& attrs,
    std::string* err) {
    if (operands.size() != 1) {
        if (err) *err = "reshape: expected 1 operand";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    if (!a) {
        if (err) *err = "reshape: operand must be a tensor";
        return {};
    }
    auto shape_attr = attrs.get("shape");
    if (!shape_attr || shape_attr->kind != AttrKind::IntegerArray) {
        if (err) *err = "reshape: missing 'shape' attribute";
        return {};
    }
    Shape target = Shape::from_constants(shape_attr->ints);
    auto r = infer_reshape(a->shape, target);
    if (!r.ok) {
        if (err) *err = r.message;
        return {};
    }
    return { make_tensor_type(r.shape, a->dtype, a->device) };
}

std::vector<TypePtr> infer_transpose_types(
    Span<const TypePtr> operands,
    const AttributeDict& attrs,
    std::string* err) {
    if (operands.size() != 1) {
        if (err) *err = "transpose: expected 1 operand";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    if (!a) {
        if (err) *err = "transpose: operand must be a tensor";
        return {};
    }
    auto perm_attr = attrs.get("perm");
    if (!perm_attr || perm_attr->kind != AttrKind::IntegerArray) {
        if (err) *err = "transpose: missing 'perm' attribute";
        return {};
    }
    std::vector<i32> perm32;
    perm32.reserve(perm_attr->ints.size());
    for (i64 v : perm_attr->ints) perm32.push_back(static_cast<i32>(v));
    auto r = infer_transpose(a->shape, make_span(perm32));
    if (!r.ok) {
        if (err) *err = r.message;
        return {};
    }
    return { make_tensor_type(r.shape, a->dtype, a->device) };
}

std::vector<TypePtr> infer_constant_types(
    Span<const TypePtr> /*operands*/,
    const AttributeDict& attrs,
    std::string* err) {
    auto shape_attr = attrs.get("shape");
    auto dtype_attr = attrs.get("dtype");
    if (!shape_attr || shape_attr->kind != AttrKind::IntegerArray ||
        !dtype_attr || dtype_attr->kind != AttrKind::DType) {
        if (err) *err = "constant: missing 'shape' or 'dtype'";
        return {};
    }
    Shape shape = Shape::from_constants(shape_attr->ints);
    return { make_tensor_type(std::move(shape), dtype_attr->dtype) };
}

std::vector<TypePtr> infer_cast_types(
    Span<const TypePtr> operands,
    const AttributeDict& attrs,
    std::string* err) {
    if (operands.size() != 1) {
        if (err) *err = "cast: expected 1 operand";
        return {};
    }
    auto a = std::dynamic_pointer_cast<const TensorType>(operands[0]);
    if (!a) {
        if (err) *err = "cast: operand must be a tensor";
        return {};
    }
    auto dt_attr = attrs.get("dtype");
    if (!dt_attr || dt_attr->kind != AttrKind::DType) {
        if (err) *err = "cast: missing 'dtype' attribute";
        return {};
    }
    return { make_tensor_type(a->shape, dt_attr->dtype, a->device) };
}

std::vector<TypePtr> infer_input_types(
    Span<const TypePtr> /*operands*/,
    const AttributeDict& attrs,
    std::string* err) {
    auto shape_attr = attrs.get("shape");
    auto dtype_attr = attrs.get("dtype");
    if (!shape_attr || shape_attr->kind != AttrKind::IntegerArray ||
        !dtype_attr || dtype_attr->kind != AttrKind::DType) {
        if (err) *err = "input: missing 'shape' or 'dtype'";
        return {};
    }
    Shape shape = Shape::from_constants(shape_attr->ints);
    return { make_tensor_type(std::move(shape), dtype_attr->dtype) };
}

} // namespace

OpRegistry& OpRegistry::instance() {
    static OpRegistry r;
    return r;
}

OpRegistry::OpRegistry() {
    register_builtins();
}

void OpRegistry::register_op(OpInfo info) {
    Opcode op = info.opcode;
    by_name_[info.name] = &by_opcode_[op];
    by_opcode_[op] = std::move(info);
}

const OpInfo* OpRegistry::lookup(Opcode op) const {
    auto it = by_opcode_.find(op);
    if (it == by_opcode_.end()) return nullptr;
    return &it->second;
}

const OpInfo* OpRegistry::lookup_by_name(StringRef name) const {
    auto it = by_name_.find(std::string(name));
    if (it == by_name_.end()) return nullptr;
    return it->second;
}

OpTraits OpRegistry::traits(Opcode op) const {
    auto* info = lookup(op);
    return info ? info->traits : OpTraits{};
}
EffectSet OpRegistry::effects(Opcode op) const {
    auto* info = lookup(op);
    return info ? info->effects : EffectSet::read_write();
}

void OpRegistry::register_builtins() {
    auto reg = [this](OpInfo info) { register_op(std::move(info)); };

    {
        OpInfo i;
        i.opcode = OP_INPUT; i.name = "input";
        i.traits = OpTraits().with(OpTrait::MemoryRead);
        i.effects = EffectSet::read();
        i.infer_types = infer_input_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_OUTPUT; i.name = "output";
        i.traits = OpTraits().with(OpTrait::MemoryWrite);
        i.effects = EffectSet::write();
        i.infer_types = [](Span<const TypePtr> ops, const AttributeDict&, std::string* err) -> std::vector<TypePtr> {
            if (ops.empty()) { if (err) *err = "output: no operands"; return {}; }
            return {}; // outputs produce no SSA values
        };
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_CONSTANT; i.name = "constant";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = infer_constant_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_ADD; i.name = "add";
        i.traits = OpTraits()
            .with(OpTrait::Pure)
            .with(OpTrait::Commutative)
            .with(OpTrait::Associative)
            .with(OpTrait::Elementwise)
            .with(OpTrait::Broadcastable)
            .with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_elementwise_binary_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_SUB; i.name = "sub";
        i.traits = OpTraits()
            .with(OpTrait::Pure)
            .with(OpTrait::Associative)
            .with(OpTrait::Elementwise)
            .with(OpTrait::Broadcastable)
            .with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_elementwise_binary_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_MUL; i.name = "mul";
        i.traits = OpTraits()
            .with(OpTrait::Pure)
            .with(OpTrait::Commutative)
            .with(OpTrait::Associative)
            .with(OpTrait::Elementwise)
            .with(OpTrait::Broadcastable)
            .with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_elementwise_binary_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_DIV; i.name = "div";
        i.traits = OpTraits()
            .with(OpTrait::Pure)
            .with(OpTrait::Elementwise)
            .with(OpTrait::Broadcastable)
            .with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_elementwise_binary_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_NEG; i.name = "neg";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Elementwise).with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_unary_same_types;
        reg(std::move(i));
    }
    for (auto [oc, nm] : std::initializer_list<std::pair<Opcode, const char*>>{
            {OP_RELU,    "relu"},
            {OP_GELU,    "gelu"},
            {OP_SIGMOID, "sigmoid"},
            {OP_TANH,    "tanh"},
            {OP_EXP,     "exp"},
            {OP_LOG,     "log"},
            {OP_SQRT,    "sqrt"}}) {
        OpInfo i;
        i.opcode = oc; i.name = nm;
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Elementwise).with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_unary_same_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_MATMUL; i.name = "matmul";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction).with(OpTrait::TensorCore);
        i.effects = EffectSet::pure();
        i.infer_types = infer_matmul_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_BROADCAST; i.name = "broadcast";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = infer_broadcast_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_RESHAPE; i.name = "reshape";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = infer_reshape_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_TRANSPOSE; i.name = "transpose";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = infer_transpose_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_REDUCE_SUM; i.name = "reduce_sum";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction);
        i.effects = EffectSet::pure();
        i.infer_types = infer_reduce_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_REDUCE_MAX; i.name = "reduce_max";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction);
        i.effects = EffectSet::pure();
        i.infer_types = infer_reduce_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_REDUCE_MEAN; i.name = "reduce_mean";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction);
        i.effects = EffectSet::pure();
        i.infer_types = infer_reduce_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_CAST; i.name = "cast";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Elementwise).with(OpTrait::ShapePreserving);
        i.effects = EffectSet::pure();
        i.infer_types = infer_cast_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_COPY; i.name = "copy";
        i.traits = OpTraits().with(OpTrait::MemoryRead).with(OpTrait::MemoryWrite);
        i.effects = EffectSet::read_write();
        i.infer_types = infer_unary_same_types;
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_RETURN; i.name = "return";
        i.traits = OpTraits().with(OpTrait::HasSideEffect);
        i.effects = EffectSet(static_cast<u16>(EffectKind::HasSideEffect));
        i.infer_types = [](Span<const TypePtr>, const AttributeDict&, std::string*) {
            return std::vector<TypePtr>{};
        };
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_ALLOC; i.name = "alloc";
        i.traits = OpTraits().with(OpTrait::HasSideEffect);
        i.effects = EffectSet(static_cast<u16>(EffectKind::Allocate) | static_cast<u16>(EffectKind::HasSideEffect));
        reg(std::move(i));
    }
    {
        OpInfo i;
        i.opcode = OP_FREE; i.name = "free";
        i.traits = OpTraits().with(OpTrait::HasSideEffect);
        i.effects = EffectSet(static_cast<u16>(EffectKind::Free) | static_cast<u16>(EffectKind::HasSideEffect));
        reg(std::move(i));
    }
}

Opcode register_user_op(OpInfo info) {
    static std::mutex mu;
    static Opcode next = OP_USER_BEGIN;
    std::lock_guard<std::mutex> g(mu);
    if (info.opcode == OP_INVALID) info.opcode = next++;
    OpRegistry::instance().register_op(std::move(info));
    return info.opcode;
}

} // namespace cg
