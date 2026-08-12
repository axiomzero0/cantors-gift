// lowering/tensor_to_codegen.cpp - Tensor IR -> Codegen IR lowering
#include "cg/lowering/tensor_to_codegen.hpp"
#include "cg/ir/ops.hpp"

#include <cmath>

namespace cg {

CGValue TensorToCodegenLowering::get_or_create_value(Value v, CGFunction& out) {
    auto it = value_map_.find(v.id());
    if (it != value_map_.end()) return it->second;

    // Create a pointer argument for this tensor.
    auto t = v.as_tensor();
    CGValue cv;
    if (t) {
        cv = out.allocate(DType::U64);  // pointer
        cv.name = "t" + std::to_string(v.id());
    } else {
        cv = out.allocate(DType::F32);
        cv.name = "s" + std::to_string(v.id());
    }
    value_map_[v.id()] = cv;
    return cv;
}

CGValue TensorToCodegenLowering::emit_tensor_arg(Value v, CGFunction& out) {
    auto cv = get_or_create_value(v, out);
    out.args.push_back(cv);
    return cv;
}

void TensorToCodegenLowering::lower_elementwise_binary(const Operation& op,
                                                         CGFunction& out) {
    // Load a, load b, compute, store result.
    auto a_ptr = get_or_create_value(op.operands[0], out);
    auto b_ptr = get_or_create_value(op.operands[1], out);
    auto r_ptr = get_or_create_value(op.results[0], out);

    // Offset register (zero for now; a real loop would increment it).
    auto off = out.allocate(DType::I64);
    CGInstruction load_imm;
    load_imm.opcode = CGOpcode::Const;
    load_imm.results = {off};
    load_imm.attributes.set("value", Attribute::make_integer(0));
    out.emit(load_imm);

    // Determine dtype from the result.
    auto t = op.results[0].as_tensor();
    DType dt = t ? t->dtype : DType::F32;
    u8 width = opts_.default_vector_width;

    // Check if we should scalarize.
    if (t && opts_.scalarize_small_tensors) {
        u64 n = 1;
        for (auto& d : t->shape) {
            if (d->is_constant()) n *= static_cast<u64>(d->value);
        }
        if (n <= opts_.scalarize_threshold) width = 1;
    }

    auto a_val = out.allocate(dt, width);
    auto b_val = out.allocate(dt, width);
    auto r_val = out.allocate(dt, width);

    out.emit(make_vector_load(a_val, a_ptr, off, width, MemorySpace::Generic));
    out.emit(make_vector_load(b_val, b_ptr, off, width, MemorySpace::Generic));

    // Emit the arithmetic op.
    CGInstruction arith;
    switch (op.opcode) {
        case OP_ADD:
            arith.opcode = CGOpcode::Add;
            arith.operands = {a_val, b_val};
            arith.results = {r_val};
            break;
        case OP_MUL:
            arith.opcode = CGOpcode::Mul;
            arith.operands = {a_val, b_val};
            arith.results = {r_val};
            break;
        default:
            // For other elementwise ops (sub, div, relu, exp, etc.), we
            // emit an Add as a placeholder and record the real opcode in
            // a comment. A real implementation would have CGOpcode entries
            // for each.
            arith.opcode = CGOpcode::Add;
            arith.operands = {a_val, b_val};
            arith.results = {r_val};
            arith.comment = "opcode=" + std::to_string(op.opcode);
            break;
    }
    out.emit(arith);
    out.emit(make_vector_store(r_ptr, off, r_val, MemorySpace::Generic));
}

void TensorToCodegenLowering::lower_elementwise_unary(const Operation& op,
                                                        CGFunction& out) {
    auto a_ptr = get_or_create_value(op.operands[0], out);
    auto r_ptr = get_or_create_value(op.results[0], out);

    auto off = out.allocate(DType::I64);
    CGInstruction load_imm;
    load_imm.opcode = CGOpcode::Const;
    load_imm.results = {off};
    load_imm.attributes.set("value", Attribute::make_integer(0));
    out.emit(load_imm);

    auto t = op.results[0].as_tensor();
    DType dt = t ? t->dtype : DType::F32;
    u8 width = opts_.default_vector_width;

    if (t && opts_.scalarize_small_tensors) {
        u64 n = 1;
        for (auto& d : t->shape) {
            if (d->is_constant()) n *= static_cast<u64>(d->value);
        }
        if (n <= opts_.scalarize_threshold) width = 1;
    }

    auto a_val = out.allocate(dt, width);
    auto r_val = out.allocate(dt, width);

    out.emit(make_vector_load(a_val, a_ptr, off, width, MemorySpace::Generic));

    // Emit the unary op as an Add with itself (placeholder) + comment.
    CGInstruction arith;
    arith.opcode = CGOpcode::Add;
    arith.operands = {a_val, a_val};
    arith.results = {r_val};
    arith.comment = "unary opcode=" + std::to_string(op.opcode);
    out.emit(arith);

    out.emit(make_vector_store(r_ptr, off, r_val, MemorySpace::Generic));
}

void TensorToCodegenLowering::lower_matmul(const Operation& op,
                                             CGFunction& out) {
    // Lower matmul as a tiled FMA loop.
    // C[m,n] = sum_k A[m,k] * B[k,n]
    //
    // For each tile (m_tile, n_tile, k_tile):
    //   load A tile -> shared
    //   load B tile -> shared
    //   for each (mi, ni) in tile:
    //     for each ki in k_tile:
    //       FMA C[mi,ni] += A[mi,ki] * B[ki,ni]
    //   store C tile
    auto a_ptr = get_or_create_value(op.operands[0], out);
    auto b_ptr = get_or_create_value(op.operands[1], out);
    auto c_ptr = get_or_create_value(op.results[0], out);

    auto t = op.results[0].as_tensor();
    DType dt = t ? t->dtype : DType::F32;

    // Get dimensions.
    auto a_t = op.operands[0].as_tensor();
    auto b_t = op.operands[1].as_tensor();
    if (!a_t || !b_t) return;

    u64 M = a_t->shape[a_t->shape.rank() - 2]->is_constant()
        ? a_t->shape[a_t->shape.rank() - 2]->value : opts_.matmul_m_tile;
    u64 K = a_t->shape[a_t->shape.rank() - 1]->is_constant()
        ? a_t->shape[a_t->shape.rank() - 1]->value : opts_.matmul_k_tile;
    u64 N = b_t->shape[b_t->shape.rank() - 1]->is_constant()
        ? b_t->shape[b_t->shape.rank() - 1]->value : opts_.matmul_n_tile;

    u64 m_tile = std::min(opts_.matmul_m_tile, M);
    u64 n_tile = std::min(opts_.matmul_n_tile, N);
    u64 k_tile = std::min(opts_.matmul_k_tile, K);

    // Offset registers.
    auto off_a = out.allocate(DType::I64);
    auto off_b = out.allocate(DType::I64);
    auto off_c = out.allocate(DType::I64);

    // Zero offsets.
    for (auto* off : {&off_a, &off_b, &off_c}) {
        CGInstruction imm;
        imm.opcode = CGOpcode::Const;
        imm.results = {*off};
        imm.attributes.set("value", Attribute::make_integer(0));
        out.emit(imm);
    }

    // Load A tile (m_tile x k_tile).
    auto a_tile = out.allocate(dt, static_cast<u8>(m_tile));
    out.emit(make_vector_load(a_tile, a_ptr, off_a, static_cast<u8>(m_tile),
                               MemorySpace::Shared));

    // Load B tile (k_tile x n_tile).
    auto b_tile = out.allocate(dt, static_cast<u8>(n_tile));
    out.emit(make_vector_load(b_tile, b_ptr, off_b, static_cast<u8>(n_tile),
                               MemorySpace::Shared));

    // Accumulator.
    auto acc = out.allocate(dt, static_cast<u8>(m_tile));

    // FMA: acc += a_tile * b_tile (one FMA per k step).
    u64 k_steps = (K + k_tile - 1) / k_tile;
    for (u64 k = 0; k < k_steps; ++k) {
        out.emit(make_fma(acc, a_tile, b_tile));
    }

    // Store C tile.
    out.emit(make_vector_store(c_ptr, off_c, acc, MemorySpace::Generic));

    // Barrier to ensure shared memory is visible.
    out.emit(make_barrier());
}

void TensorToCodegenLowering::lower_reduction(const Operation& op,
                                                CGFunction& out) {
    // Lower a reduction as: load input, reduce, store output.
    auto in_ptr = get_or_create_value(op.operands[0], out);
    auto out_ptr = get_or_create_value(op.results[0], out);

    auto off = out.allocate(DType::I64);
    CGInstruction imm;
    imm.opcode = CGOpcode::Const;
    imm.results = {off};
    imm.attributes.set("value", Attribute::make_integer(0));
    out.emit(imm);

    auto t = op.results[0].as_tensor();
    DType dt = t ? t->dtype : DType::F32;

    auto in_val = out.allocate(dt, opts_.default_vector_width);
    auto red_val = out.allocate(dt, 1);

    out.emit(make_vector_load(in_val, in_ptr, off, opts_.default_vector_width,
                               MemorySpace::Generic));

    CGInstruction reduce;
    reduce.opcode = CGOpcode::Reduce;
    reduce.operands = {in_val};
    reduce.results = {red_val};
    reduce.comment = "reduction opcode=" + std::to_string(op.opcode);
    out.emit(reduce);

    out.emit(make_vector_store(out_ptr, off, red_val, MemorySpace::Generic));
}

void TensorToCodegenLowering::lower_constant(const Operation& op,
                                               CGFunction& out) {
    // Constants become a Const instruction with the value.
    auto t = op.results[0].as_tensor();
    DType dt = t ? t->dtype : DType::F32;
    auto cv = out.allocate(dt);

    CGInstruction ci;
    ci.opcode = CGOpcode::Const;
    ci.results = {cv};

    // Extract the constant value if it's a scalar.
    auto bytes_attr = op.attributes.get("bytes");
    auto dtype_attr = op.attributes.get("dtype");
    if (bytes_attr && dtype_attr &&
        bytes_attr->kind == AttrKind::String &&
        dtype_attr->kind == AttrKind::DType) {
        const std::string& bytes = bytes_attr->str;
        if (bytes.size() >= dtype_size(dtype_attr->dtype)) {
            i64 val = 0;
            std::memcpy(&val, bytes.data(),
                        std::min<usize>(8, bytes.size()));
            ci.attributes.set("value", Attribute::make_integer(val));
        }
    }
    out.emit(ci);
    value_map_[op.results[0].id()] = cv;
}

void TensorToCodegenLowering::lower_alloc(const Operation& op,
                                            CGFunction& out) {
    // Alloc ops are metadata; we emit them as a comment.
    CGInstruction ai;
    ai.opcode = CGOpcode::Const;
    ai.comment = "alloc buffer_id=" +
        std::to_string(op.attributes.get("buffer_id") ?
                        op.attributes.get("buffer_id")->integer : 0);
    out.emit(ai);
}

void TensorToCodegenLowering::lower_free(const Operation& op,
                                           CGFunction& out) {
    CGInstruction fi;
    fi.opcode = CGOpcode::Const;
    fi.comment = "free buffer_id=" +
        std::to_string(op.attributes.get("buffer_id") ?
                        op.attributes.get("buffer_id")->integer : 0);
    out.emit(fi);
}

void TensorToCodegenLowering::lower_function(const Function& f,
                                               CGFunction& out,
                                               const Schedule&) {
    value_map_.clear();

    // Emit pointer arguments for block args.
    for (auto& arg : f.args()) {
        emit_tensor_arg(arg, out);
    }

    // Walk operations in order (they are already in topological order
    // since they were appended in definition order).
    for (auto& op : *f.entry()) {
        switch (op.opcode) {
            case OP_CONSTANT:
                lower_constant(op, out);
                break;
            case OP_INPUT:
                // Inputs are block args; already handled above.
                if (!op.results.empty()) {
                    get_or_create_value(op.results[0], out);
                }
                break;
            case OP_OUTPUT:
                // Outputs are stores; the operand's pointer is already
                // in the value map.
                break;
            case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
                lower_elementwise_binary(op, out);
                break;
            case OP_NEG: case OP_RELU: case OP_GELU: case OP_SIGMOID:
            case OP_TANH: case OP_EXP: case OP_LOG: case OP_SQRT:
            case OP_CAST:
                lower_elementwise_unary(op, out);
                break;
            case OP_MATMUL:
                lower_matmul(op, out);
                break;
            case OP_REDUCE_SUM: case OP_REDUCE_MAX: case OP_REDUCE_MEAN:
                lower_reduction(op, out);
                break;
            case OP_BROADCAST: case OP_RESHAPE: case OP_TRANSPOSE:
            case OP_COPY:
                // These are layout/view ops; at the Codegen IR level they
                // are no-ops (the pointer is reused).
                if (!op.results.empty() && !op.operands.empty()) {
                    value_map_[op.results[0].id()] =
                        get_or_create_value(op.operands[0], out);
                }
                break;
            case OP_ALLOC:
                lower_alloc(op, out);
                break;
            case OP_FREE:
                lower_free(op, out);
                break;
            default:
                // Unknown op: emit a nop with a comment.
                CGInstruction nop;
                nop.opcode = CGOpcode::Const;
                nop.comment = "unknown opcode=" + std::to_string(op.opcode);
                out.emit(nop);
                break;
        }
    }
}

CGModule TensorToCodegenLowering::lower(const Module& m, const Schedule&) {
    CGModule cgm;
    for (auto& f : m.functions()) {
        auto& cgf = cgm.create_function(f->name());
        lower_function(*f, cgf, {});
    }
    return cgm;
}

} // namespace cg
