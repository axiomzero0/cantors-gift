// backend/amd_backend.cpp - AMD GCN ISA emitter
#include "cg/backend/amd/amd_backend.hpp"

#include <sstream>

namespace cg {

std::string GCNEmitter::gcn_type_name(DType dt) const {
    switch (dt) {
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::F32:  return "f32";
        case DType::F64:  return "f64";
        case DType::I8:   return "i8";
        case DType::I16:  return "i16";
        case DType::I32:  return "i32";
        case DType::I64:  return "i64";
        case DType::U8:   return "u8";
        case DType::U16:  return "u16";
        case DType::U32:  return "u32";
        case DType::U64:  return "u64";
        default:          return "i32";
    }
}

std::string GCNEmitter::reg_name(const CGValue& v) const {
    if (!v.name.empty()) return "%" + v.name;
    // Scalar registers (s) for pointers; vector registers (v) for data.
    if (v.dtype == DType::U64 || v.dtype == DType::I64)
        return "%s" + std::to_string(v.id);
    return "%v" + std::to_string(v.id);
}

std::string GCNEmitter::emit_instruction(const CGInstruction& inst) const {
    std::ostringstream os;
    os << "  ";

    switch (inst.opcode) {
        case CGOpcode::Load: {
            const auto& dst = inst.results[0];
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const char* space = inst.mem_space == MemorySpace::Shared ? "lds" : "buffer";
            os << space << "_load_dword " << reg_name(dst)
               << ", off:" << reg_name(off) << " " << reg_name(ptr);
            break;
        }
        case CGOpcode::Store: {
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const auto& val = inst.operands[2];
            const char* space = inst.mem_space == MemorySpace::Shared ? "lds" : "buffer";
            os << space << "_store_dword " << reg_name(val)
               << ", off:" << reg_name(off) << " " << reg_name(ptr);
            break;
        }
        case CGOpcode::VectorLoad: {
            const auto& dst = inst.results[0];
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const char* space = inst.mem_space == MemorySpace::Shared ? "lds" : "buffer";
            os << space << "_load_dwordx" << static_cast<int>(dst.width)
               << " " << reg_name(dst) << ", off:" << reg_name(off)
               << " " << reg_name(ptr);
            break;
        }
        case CGOpcode::VectorStore: {
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const auto& val = inst.operands[2];
            const char* space = inst.mem_space == MemorySpace::Shared ? "lds" : "buffer";
            os << space << "_store_dwordx" << static_cast<int>(val.width)
               << " " << reg_name(val) << ", off:" << reg_name(off)
               << " " << reg_name(ptr);
            break;
        }
        case CGOpcode::FMA: {
            const auto& acc = inst.operands[0];
            const auto& a   = inst.operands[1];
            const auto& b   = inst.operands[2];
            const auto& dst = inst.results.empty() ? acc : inst.results[0];
            os << "v_fma_f32 " << reg_name(dst) << ", "
               << reg_name(a) << ", " << reg_name(b) << ", " << reg_name(acc);
            break;
        }
        case CGOpcode::Add: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            const auto& b = inst.operands[1];
            os << "v_add_f32 " << reg_name(dst) << ", "
               << reg_name(a) << ", " << reg_name(b);
            break;
        }
        case CGOpcode::Mul: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            const auto& b = inst.operands[1];
            os << "v_mul_f32 " << reg_name(dst) << ", "
               << reg_name(a) << ", " << reg_name(b);
            break;
        }
        case CGOpcode::Reduce:
            os << "// reduce";
            if (!inst.comment.empty()) os << " " << inst.comment;
            break;
        case CGOpcode::Barrier:
            os << "s_barrier";
            break;
        case CGOpcode::AsyncCopy:
            os << "// async_copy";
            break;
        case CGOpcode::Prefetch:
            os << "// prefetch";
            break;
        case CGOpcode::Broadcast:
            os << "// broadcast";
            break;
        case CGOpcode::Shuffle:
            os << "ds_bpermute";
            break;
        case CGOpcode::Cmp:
            os << "v_cmp_eq_f32";
            break;
        case CGOpcode::Select:
            os << "v_cndmask_b32";
            break;
        case CGOpcode::Cast: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            os << "v_cvt_f32_f32 " << reg_name(dst) << ", " << reg_name(a);
            break;
        }
        case CGOpcode::Const: {
            const auto& dst = inst.results[0];
            auto val_attr = inst.attributes.get("value");
            if (val_attr && val_attr->kind == AttrKind::Integer) {
                os << "s_mov_b32 " << reg_name(dst) << ", " << val_attr->integer;
            } else {
                os << "s_mov_b32 " << reg_name(dst) << ", 0";
            }
            break;
        }
    }
    if (!inst.comment.empty() && inst.opcode != CGOpcode::Reduce &&
        inst.opcode != CGOpcode::AsyncCopy && inst.opcode != CGOpcode::Prefetch &&
        inst.opcode != CGOpcode::Broadcast) {
        os << "  ; " << inst.comment;
    }
    return os.str();
}

std::string GCNEmitter::emit_body(const CGFunction& fn) {
    std::ostringstream os;
    for (auto& inst : fn.instructions) {
        os << emit_instruction(inst) << "\n";
    }
    return os.str();
}

std::string GCNEmitter::emit_kernel(const CGFunction& fn,
                                     const std::string& kernel_name,
                                     const std::string& gcn_target) {
    std::ostringstream os;
    os << "// GCN ISA generated by cantors-gift\n";
    os << ".target " << gcn_target << "\n";
    os << ".amdgpu_kernel " << kernel_name << "(";
    for (usize i = 0; i < fn.args.size(); ++i) {
        if (i) os << ", ";
        os << reg_name(fn.args[i]);
    }
    os << ")\n{\n";
    os << emit_body(fn);
    os << "  s_endpgm\n";
    os << "}\n";
    return os.str();
}

std::unique_ptr<Executable> AmdBackend::compile(const CGModule& module) {
    auto exe = std::make_unique<Executable>();
    exe->target_device = DeviceId::rocm();

    GCNEmitter emitter;
    for (const auto& fn : module.functions) {
        std::string gcn = emitter.emit_kernel(fn, fn.name, gcn_target_);
        // Store GCN text in the ptx_text field (it's a generic "text" field).
        exe->ptx_text += gcn + "\n";
        exe->entrypoints.push_back({fn.name, 0});
    }
    return exe;
}

std::optional<std::string> AmdBackend::emit_text(const CGModule& module) {
    GCNEmitter emitter;
    std::string out;
    for (const auto& fn : module.functions) {
        out += emitter.emit_kernel(fn, fn.name, gcn_target_);
        out += "\n";
    }
    return out;
}

} // namespace cg
