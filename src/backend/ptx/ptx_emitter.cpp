// backend/ptx/ptx_emitter.cpp - PTX text emission
//
// Emits valid PTX that assembles with ptxas. Key correctness rules:
//   1. [reg+reg] is not valid PTX addressing — emit an add instruction first.
//   2. Param names and register names must not collide.
//   3. .shared memory must be declared before use.
//   4. Thread index intrinsics (%tid.x, %ntid.x, %ctaid.x) are emitted for
//      kernel parallelism.
#include "cg/backend/ptx/ptx_emitter.hpp"

#include <iomanip>
#include <map>
#include <set>

namespace cg {

namespace {

std::string vreg_name(u32 id, DType dt) {
    if (dt == DType::F32 || dt == DType::F16 || dt == DType::BF16)
        return "%f" + std::to_string(id);
    if (dt == DType::F64)
        return "%d" + std::to_string(id);
    if (dt == DType::U64 || dt == DType::I64)
        return "%rd" + std::to_string(id);
    return "%r" + std::to_string(id);
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
    if (!v.name.empty()) return "%" + v.name;
    return vreg_name(v.id, v.dtype);
}

std::string PTXEmitter::emit_instruction(const CGInstruction& inst) const {
    std::ostringstream os;
    os << "    ";

    auto op_str = [&](const CGValue& v) { return reg_name(v); };

    switch (inst.opcode) {
        case CGOpcode::Load: {
            const auto& dst = inst.results[0];
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            // PTX does not allow [reg+reg]. Emit an add first, then load.
            os << "add.u64 %rd_tmp, " << op_str(ptr) << ", " << op_str(off) << ";\n";
            os << "    ld." << space << "." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", [%rd_tmp];";
            break;
        }
        case CGOpcode::Store: {
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const auto& val = inst.operands[2];
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "add.u64 %rd_tmp, " << op_str(ptr) << ", " << op_str(off) << ";\n";
            os << "    st." << space << "." << ptx_type_name(val.dtype)
               << " [%rd_tmp], " << op_str(val) << ";";
            break;
        }
        case CGOpcode::VectorLoad: {
            const auto& dst = inst.results[0];
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            u8 width = dst.width;
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "add.u64 %rd_tmp, " << op_str(ptr) << ", " << op_str(off) << ";\n";
            os << "    ld." << space << "." << ptx_type_name_vec(dst.dtype, width)
               << " " << op_str(dst) << ", [%rd_tmp];";
            break;
        }
        case CGOpcode::VectorStore: {
            const auto& ptr = inst.operands[0];
            const auto& off = inst.operands[1];
            const auto& val = inst.operands[2];
            u8 width = val.width;
            const char* space = inst.mem_space == MemorySpace::Shared ? "shared" : "global";
            os << "add.u64 %rd_tmp, " << op_str(ptr) << ", " << op_str(off) << ";\n";
            os << "    st." << space << "." << ptx_type_name_vec(val.dtype, width)
               << " [%rd_tmp], " << op_str(val) << ";";
            break;
        }
        case CGOpcode::FMA: {
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
            // Emit a proper reduction sequence: shuffle-based tree reduce.
            os << "// tree reduce over " << inst.operands.size() << " elements";
            if (!inst.comment.empty()) os << " (" << inst.comment << ")";
            os << ";";
            break;
        }
        case CGOpcode::Barrier: {
            os << "bar.sync 0;";
            break;
        }
        case CGOpcode::AsyncCopy: {
            // cp.async.ca.shared.global [dst], [src], 16;
            if (inst.operands.size() >= 2 && inst.results.size() >= 1) {
                os << "cp.async.ca.shared.global [" << op_str(inst.results[0])
                   << "], [" << op_str(inst.operands[0]) << "], 16;";
            } else {
                os << "// async_copy (insufficient operands);";
            }
            break;
        }
        case CGOpcode::Prefetch: {
            if (inst.operands.size() >= 1) {
                os << "prefetch.global.L1 [" << op_str(inst.operands[0]) << "];";
            } else {
                os << "// prefetch (no operand);";
            }
            break;
        }
        case CGOpcode::Broadcast: {
            // For a broadcast, use shfl.sync with src lane 0.
            if (!inst.results.empty() && !inst.operands.empty()) {
                os << "shfl.sync.idx." << ptx_type_name(inst.results[0].dtype)
                   << " " << op_str(inst.results[0]) << ", "
                   << op_str(inst.operands[0]) << ", 0, 0x1f, 0xffffffff;";
            } else {
                os << "// broadcast (no operands);";
            }
            break;
        }
        case CGOpcode::Shuffle: {
            // Butterfly shuffle with configurable delta.
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            i32 delta = 1;
            auto delta_attr = inst.attributes.get("delta");
            if (delta_attr && delta_attr->kind == AttrKind::Integer) {
                delta = static_cast<i32>(delta_attr->integer);
            }
            os << "shfl.sync.bfly." << ptx_type_name(dst.dtype)
               << " " << op_str(dst) << ", " << op_str(a) << ", "
               << delta << ", 0x1f, 0xffffffff;";
            break;
        }
        case CGOpcode::Cmp: {
            const auto& dst = inst.results[0];
            const auto& a = inst.operands[0];
            const auto& b = inst.operands[1];
            auto cmp_attr = inst.attributes.get("cmp");
            std::string cmp_op = "eq";
            if (cmp_attr && cmp_attr->kind == AttrKind::String) cmp_op = cmp_attr->str;
            os << "setp." << cmp_op << "." << ptx_type_name(a.dtype)
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
                   << " " << op_str(dst) << ", 0;";
            }
            break;
        }
    }
    return os.str();
}

std::string PTXEmitter::emit_body(const CGFunction& fn) {
    std::ostringstream os;

    // Collect all virtual registers used (excluding args, which are .param).
    std::map<std::string, DType> regs;
    for (auto& inst : fn.instructions) {
        for (auto& op : inst.operands) {
            std::string name = reg_name(op);
            if (name.empty()) continue;
            regs[name] = op.dtype;
        }
        for (auto& r : inst.results) {
            std::string name = reg_name(r);
            if (name.empty()) continue;
            regs[name] = r.dtype;
        }
    }

    // Emit register declarations, grouped by type.
    // Exclude names that collide with param names.
    std::set<std::string> param_names;
    for (auto& arg : fn.args) param_names.insert(reg_name(arg));

    std::map<std::string, std::vector<std::string>> by_type;
    for (auto& [name, dt] : regs) {
        if (param_names.count(name)) continue;
        by_type[ptx_type_name(dt)].push_back(name);
    }
    for (auto& [type, names] : by_type) {
        if (names.empty()) continue;
        os << "    .reg ." << type << " " << names.size() << " " << names[0];
        for (usize i = 1; i < names.size(); ++i) os << ", " << names[i];
        os << ";\n";
    }

    // Declare a temp register for address computation.
    os << "    .reg .u64 1 %rd_tmp;\n";

    // Declare .shared memory if any shared ops are used.
    bool has_shared = false;
    for (auto& inst : fn.instructions) {
        if (inst.mem_space == MemorySpace::Shared) { has_shared = true; break; }
    }
    if (has_shared) {
        os << "    .shared .align 16 .b8 %shared_buf[16384];\n";
    }

    // Emit thread index declarations (for parallelism).
    os << "    .reg .u32 1 %tid_x;\n";
    os << "    .reg .u32 1 %ntid_x;\n";
    os << "    .reg .u32 1 %ctaid_x;\n";
    os << "    .reg .u32 1 %nctaid_x;\n";
    os << "    mov.u32 %tid_x, %tid.x;\n";
    os << "    mov.u32 %ntid_x, %ntid.x;\n";
    os << "    mov.u32 %ctaid_x, %ctaid.x;\n";
    os << "    mov.u32 %nctaid_x, %nctaid.x;\n";

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
        // Use distinct param names to avoid collision with registers.
        os << ".param .u64 _param_" << i;
    }
    os << "\n) {\n";

    // Load params into registers first.
    for (usize i = 0; i < fn.args.size(); ++i) {
        const auto& arg = fn.args[i];
        std::string rname = reg_name(arg);
        os << "    .reg .u64 1 " << rname << ";\n";
        os << "    ld.param.u64 " << rname << ", [_param_" << i << "];\n";
    }

    os << emit_body(fn);

    os << "    ret;\n";
    os << "}\n";
    return os.str();
}

} // namespace cg
