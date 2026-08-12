// backend/ptx_emitter.cpp - PTX text emission
#include "cg/backend/ptx/ptx_emitter.hpp"

#include <iomanip>

namespace cg {

namespace {

std::string vreg_name(u32 id) {
    return "%r" + std::to_string(id);
}

std::string preg_name(u32 id) {
    return "%rd" + std::to_string(id);
}

} // namespace

std::string PTXEmitter::ptx_type_name(DType dt) const {
    switch (dt) {
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::F32:  return "f32";
        case DType::F64:  return "f64";
        case DType::I8:   return "s8";
        case DType::I16:  return "s16";
        case DType::I32:  return "s32";
        case DType::I64:  return "s64";
        case DType::U8:   return "u8";
        case DType::U16:  return "u16";
        case DType::U32:  return "u32";
        case DType::U64:  return "u64";
        case DType::BOOL: return "pred";
        default:          return "u32";
    }
}

std::string PTXEmitter::ptx_type_name_vec(DType dt, u8 width) const {
    if (width <= 1) return ptx_type_name(dt);
    return "v" + std::to_string(width) + "." + ptx_type_name(dt);
}

std::string PTXEmitter::reg_name(const CGValue& v) const {
    // Pointers use %rd (64-bit), floats use %f, ints use %r.
    if (!v.name.empty()) return "%" + v.name;
    if (v.dtype == DType::F32 || v.dtype == DType::F16 || v.dtype == DType::BF16)
        return "%f" + std::to_string(v.id);
    if (v.dtype == DType::F64)
        return "%d" + std::to_string(v.id);
    if (v.dtype == DType::U64 || v.dtype == DType::I64)
        return "%rd" + std::to_string(v.id);
    return "%r" + std::to_string(v.id);
}

std::string PTXEmitter::emit_instruction(const CGInstruction& inst) const {
    std::ostringstream os;
    os << "    ";

    auto op_str = [&](const CGValue& v) { return reg_name(v); };

    switch (inst.opcode) {
        case CGOpcode::Load: {
            // ld.{global,shared}.<type> %r, [%rd + offset];
            const auto& dst = inst.results[0];
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "ld." << space << "." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", [" << op_str(ptr) << "+" << op_str(off) << "];";
            break;
        }
        case CGOpcode::Store: {
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const auto& val = inst.operands[2];
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "st." << space << "." << ptx_type_name(val.dtype)
               << " [" << op_str(ptr) << "+" << op_str(off) << "], " << op_str(val) << ";";
            break;
        }
        case CGOpcode::VectorLoad: {
            const auto& dst = inst.results[0];
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            u8 width = dst.width;
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "ld." << space << "." << ptx_type_name_vec(dst.dtype, width)
               << " " << op_str(dst) << ", [" << op_str(ptr) << "+" << op_str(off) << "];";
            break;
        }
        case CGOpcode::VectorStore: {
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const auto& val = inst.operands[2];
            u8 width = val.width;
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "st." << space << "." << ptx_type_name_vec(val.dtype, width)
               << " [" << op_str(ptr) << "+" << op_str(off) << "], " << op_str(val) << ";";
            break;
        }
        case CGOpcode::FMA: {
            // fma.rn.f32 %f, %f, %f, %f;
            const auto& acc = inst.operands[0];
            const auto& a   = inst.operands[1];
            const auto& b   = inst.operands[2];
            const auto& dst = inst.results.empty() ? acc : inst.results[0];
            os << "fma.rn." << ptx_type_name(a.dtype)
               << " " << op_str(dst) << ", "
               << op_str(a) << ", " << op_str(b) << ", " << op_str(acc) << ";";
            break;
        }
        case CGOpcode::Add: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            const auto& b = inst.operands[1];
            os << "add." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ", " << op_str(b) << ";";
            break;
        }
        case CGOpcode::Mul: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            const auto& b = inst.operands[1];
            os << "mul." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ", " << op_str(b) << ";";
            break;
        }
        case CGOpcode::Reduce: {
            os << "// reduce";
            if (!inst.comment.empty()) os << " " << inst.comment;
            os << ";";
            break;
        }
        case CGOpcode::Barrier: {
            os << "bar.sync 0;";
            break;
        }
        case CGOpcode::AsyncCopy: {
            os << "// async_copy";
            if (!inst.comment.empty()) os << " " << inst.comment;
            os << ";";
            break;
        }
        case CGOpcode::Prefetch: {
            os << "// prefetch";
            if (!inst.comment.empty()) os << " " << inst.comment;
            os << ";";
            break;
        }
        case CGOpcode::Broadcast: {
            os << "// broadcast;";
            break;
        }
        case CGOpcode::Shuffle: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            os << "shfl.sync.bfly." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ", 1, 0x1f, 0xffffffff;";
            break;
        }
        case CGOpcode::Cmp: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            const auto& b = inst.operands[1];
            os << "setp.eq." << ptx_type_name(a.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ", " << op_str(b) << ";";
            break;
        }
        case CGOpcode::Select: {
            const auto& dst = inst.results[0];
            const auto& pred = inst.operands[0];
            const auto& a = inst.operands[1];
            const auto& b = inst.operands[2];
            os << "selp." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ", " << op_str(b)
               << ", " << op_str(pred) << ";";
            break;
        }
        case CGOpcode::Cast: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            os << "cvt." << ptx_type_name(dst.dtype) << "." << ptx_type_name(a.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ";";
            break;
        }
        case CGOpcode::Const: {
            const auto& dst = inst.results[0];
            auto val_attr = inst.attributes.get("value");
            if (val_attr && val_attr->kind == AttrKind::Integer) {
                os << "mov." << ptx_type_name(dst.dtype)
                   << " " << op_str(dst) << ", " << val_attr->integer << ";";
            } else {
                os << "mov." << ptx_type_name(dst.dtype)
                   << " " << op_str(dst) << ", 0; // const";
            }
            break;
        }
    }
    return os.str();
}

std::string PTXEmitter::emit_body(const CGFunction& fn) {
    std::ostringstream os;

    // Collect all virtual registers used.
    std::unordered_map<std::string, DType> regs;
    for (auto& arg : fn.args) {
        regs[reg_name(arg)] = arg.dtype;
    }
    for (auto& inst : fn.instructions) {
        for (auto& op : inst.operands) regs[reg_name(op)] = op.dtype;
        for (auto& r : inst.results) regs[reg_name(r)] = r.dtype;
    }

    // Emit register declarations.
    // Group by type.
    std::unordered_map<std::string, std::vector<std::string>> by_type;
    for (auto& [name, dt] : regs) {
        by_type[ptx_type_name(dt)].push_back(name);
    }
    for (auto& [type, names] : by_type) {
        os << "    .reg ." << type << " " << names.size() << " "
           << names[0];
        for (usize i = 1; i < names.size(); ++i) os << ", " << names[i];
        os << ";\n";
    }

    // Emit instructions.
    for (auto& inst : fn.instructions) {
        os << emit_instruction(inst) << "\n";
    }

    return os.str();
}

std::string PTXEmitter::emit_kernel(const CGFunction& fn,
                                     const std::string& kernel_name,
                                     const std::string& sm_target) {
    std::ostringstream os;
    os << "// PTX generated by cantors-gift\n";
    os << ".version 7.5\n";
    os << ".target " << sm_target << "\n";
    os << ".address_size 64\n\n";

    os << ".visible .entry " << kernel_name << "(";
    for (usize i = 0; i < fn.args.size(); ++i) {
        if (i) os << ",\n    ";
        else os << "\n    ";
        const auto& arg = fn.args[i];
        if (arg.dtype == DType::U64 || arg.dtype == DType::I64) {
            os << ".param .u64 " << reg_name(arg);
        } else {
            os << ".param ." << ptx_type_name(arg.dtype) << " " << reg_name(arg);
        }
    }
    os << "\n) {\n";

    os << emit_body(fn);

    os << "    ret;\n";
    os << "}\n";
    return os.str();
}

} // namespace cg
