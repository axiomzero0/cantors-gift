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

    // ---- Conv2D ----
    // input:  [N, C_in, H, W]
    // weight: [C_out, C_in, kH, kW]
    // output: [N, C_out, H_out, W_out]
    // H_out = (H + 2*pad_h - dilation*(kH-1) - 1) / stride_h + 1
    {
        OpInfo i;
        i.opcode = OP_CONV2D; i.name = "conv2d";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction);
        i.effects = EffectSet::pure();
        i.infer_types = [](Span<const TypePtr> operands,
                           const AttributeDict& attrs,
                           std::string* err) -> std::vector<TypePtr> {
            if (operands.size() != 2) {
                if (err) *err = "conv2d: expected 2 operands (input, weight)";
                return {};
            }
            auto input = std::dynamic_pointer_cast<const TensorType>(operands[0]);
            auto weight = std::dynamic_pointer_cast<const TensorType>(operands[1]);
            if (!input || !weight) {
                if (err) *err = "conv2d: operands must be tensors";
                return {};
            }
            if (input->shape.rank() != 4 || weight->shape.rank() != 4) {
                if (err) *err = "conv2d: input must be 4D (NCHW) and weight 4D";
                return {};
            }
            // Read stride, padding, dilation from attributes (defaults: 1, 0, 1).
            auto get_int = [&](const char* name, i64 def) -> i64 {
                auto a = attrs.get(name);
                if (a && a->kind == AttrKind::Integer) return a->integer;
                return def;
            };
            i64 stride_h = get_int("stride_h", 1);
            i64 stride_w = get_int("stride_w", 1);
            i64 pad_h = get_int("pad_h", 0);
            i64 pad_w = get_int("pad_w", 0);
            i64 dilation_h = get_int("dilation_h", 1);
            i64 dilation_w = get_int("dilation_w", 1);

            i64 N = input->shape[0]->is_constant() ? input->shape[0]->value : 0;
            i64 H = input->shape[2]->is_constant() ? input->shape[2]->value : 0;
            i64 W = input->shape[3]->is_constant() ? input->shape[3]->value : 0;
            i64 C_out = weight->shape[0]->is_constant() ? weight->shape[0]->value : 0;
            i64 kH = weight->shape[2]->is_constant() ? weight->shape[2]->value : 0;
            i64 kW = weight->shape[3]->is_constant() ? weight->shape[3]->value : 0;

            i64 H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
            i64 W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;

            Shape out_shape = Shape::from_constants({N, C_out, H_out, W_out});
            return { make_tensor_type(out_shape, input->dtype, input->device) };
        };
        reg(std::move(i));
    }

    // ---- Softmax ----
    // input: [..., D]  output: same shape
    {
        OpInfo i;
        i.opcode = OP_SOFTMAX; i.name = "softmax";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction);
        i.effects = EffectSet::pure();
        i.infer_types = infer_unary_same_types;
        reg(std::move(i));
    }

    // ---- LayerNorm ----
    // input: [..., D]  output: same shape
    {
        OpInfo i;
        i.opcode = OP_LAYERNORM; i.name = "layernorm";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::Reduction);
        i.effects = EffectSet::pure();
        i.infer_types = infer_unary_same_types;
        reg(std::move(i));
    }

    // ---- BatchNorm ----
    // input: [N, C, ...]  output: same shape
    {
        OpInfo i;
        i.opcode = OP_BATCHNORM; i.name = "batchnorm";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = infer_unary_same_types;
        reg(std::move(i));
    }

    // ---- Gather ----
    // input: [A, ...], indices: [B, ...]  output: [B, ...] (along axis)
    {
        OpInfo i;
        i.opcode = OP_GATHER; i.name = "gather";
        i.traits = OpTraits().with(OpTrait::Pure).with(OpTrait::MemoryRead);
        i.effects = EffectSet::pure();
        i.infer_types = [](Span<const TypePtr> operands,
                           const AttributeDict& attrs,
                           std::string* err) -> std::vector<TypePtr> {
            if (operands.size() != 2) {
                if (err) *err = "gather: expected 2 operands";
                return {};
            }
            auto input = std::dynamic_pointer_cast<const TensorType>(operands[0]);
            auto indices = std::dynamic_pointer_cast<const TensorType>(operands[1]);
            if (!input || !indices) {
                if (err) *err = "gather: operands must be tensors";
                return {};
            }
            // Output shape = indices shape (simplified).
            return { make_tensor_type(indices->shape, input->dtype, input->device) };
        };
        reg(std::move(i));
    }

    // ---- Scatter ----
    // input, indices, updates -> output (same shape as input)
    {
        OpInfo i;
        i.opcode = OP_SCATTER; i.name = "scatter";
        i.traits = OpTraits().with(OpTrait::MemoryWrite).with(OpTrait::HasSideEffect);
        i.effects = EffectSet::read_write();
        i.infer_types = [](Span<const TypePtr> operands,
                           const AttributeDict&,
                           std::string* err) -> std::vector<TypePtr> {
            if (operands.empty()) {
                if (err) *err = "scatter: expected at least 1 operand";
                return {};
            }
            return { operands[0] };
        };
        reg(std::move(i));
    }

    // ---- Concat ----
    // inputs: list of tensors  output: concatenated along axis
    {
        OpInfo i;
        i.opcode = OP_CONCAT; i.name = "concat";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = [](Span<const TypePtr> operands,
                           const AttributeDict& attrs,
                           std::string* err) -> std::vector<TypePtr> {
            if (operands.empty()) {
                if (err) *err = "concat: expected at least 1 operand";
                return {};
            }
            auto first = std::dynamic_pointer_cast<const TensorType>(operands[0]);
            if (!first) {
                if (err) *err = "concat: operands must be tensors";
                return {};
            }
            auto axis_attr = attrs.get("axis");
            i64 axis = (axis_attr && axis_attr->kind == AttrKind::Integer)
                ? axis_attr->integer : 0;
            if (axis < 0) axis += static_cast<i64>(first->shape.rank());

            // Sum the sizes along the concat axis.
            i64 total = 0;
            for (auto& t : operands) {
                auto tt = std::dynamic_pointer_cast<const TensorType>(t);
                if (!tt) continue;
                if (static_cast<usize>(axis) < tt->shape.rank() &&
                    tt->shape[axis]->is_constant()) {
                    total += tt->shape[axis]->value;
                }
            }

            // Build output shape.
            Shape out = first->shape;
            if (static_cast<usize>(axis) < out.rank()) {
                out.dims()[axis] = DimExpr::make_constant(total);
            }
            return { make_tensor_type(out, first->dtype, first->device) };
        };
        reg(std::move(i));
    }

    // ---- Slice ----
    // input: [...]  output: sub-range along each dim
    {
        OpInfo i;
        i.opcode = OP_SLICE; i.name = "slice";
        i.traits = OpTraits().with(OpTrait::Pure);
        i.effects = EffectSet::pure();
        i.infer_types = [](Span<const TypePtr> operands,
                           const AttributeDict& attrs,
                           std::string* err) -> std::vector<TypePtr> {
            if (operands.size() != 1) {
                if (err) *err = "slice: expected 1 operand";
                return {};
            }
            auto input = std::dynamic_pointer_cast<const TensorType>(operands[0]);
            if (!input) {
                if (err) *err = "slice: operand must be a tensor";
                return {};
            }
            // Read begins, ends from attributes.
            auto begins = attrs.get("begins");
            auto ends = attrs.get("ends");
            if (!begins || !ends ||
                begins->kind != AttrKind::IntegerArray ||
                ends->kind != AttrKind::IntegerArray) {
                if (err) *err = "slice: missing 'begins' or 'ends' attribute";
                return {};
            }
            Shape out;
            for (usize d = 0; d < input->shape.rank(); ++d) {
                i64 begin = d < begins->ints.size() ? begins->ints[d] : 0;
                i64 end = d < ends->ints.size()
                    ? ends->ints[d]
                    : (input->shape[d]->is_constant() ? input->shape[d]->value : 0);
                out.dims().push_back(DimExpr::make_constant(end - begin));
            }
            return { make_tensor_type(out, input->dtype, input->device) };
        };
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
