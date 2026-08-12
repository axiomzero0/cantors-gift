// backend/lowering.cpp - Codegen IR -> machine code
#include "cg/backend/lowering.hpp"

namespace cg {

std::string lower_to_ptx(const CGFunction& fn, const std::string& kernel_name,
                         const std::string& sm_target) {
    PTXEmitter emitter;
    return emitter.emit_kernel(fn, kernel_name, sm_target);
}

std::vector<u8> lower_to_x86(const CGFunction& fn) {
    X86Emitter e;

    // Prologue: push callee-saved registers, align stack.
    e.push(X86Reg::RBP);
    e.mov_reg(X86Reg::RBP, X86Reg::RSP);

    // Allocate stack space for local temporaries (just a fixed 256-byte
    // scratch area for the foundational implementation).
    // sub rsp, 256
    e.emit_rex_public(true, false, false, false);
    e.emit_byte(0x81);  // SUB r/m64, imm32 (when REX.W and mod=11)
    e.emit_byte(0xEC);  // modrm: mod=11, reg=5 (SUB), rm=4 (RSP)
    e.emit_u32(256);

    // Walk instructions and emit x86 equivalents.
    // We use a simple register allocation strategy:
    //   - RAX, RCX, RDX, R8, R9, R10, R11 are caller-saved temporaries.
    //   - XMM0-XMM15 for vector ops.
    //   - The first arg (RDI) is treated as the base pointer for loads/stores.
    // This is NOT a real register allocator; it's a direct translation that
    // assumes the CGFunction's virtual registers can be mapped 1:1 to
    // physical registers for small kernels.

    for (const auto& inst : fn.instructions) {
        switch (inst.opcode) {
            case CGOpcode::Const: {
                // mov rax, imm64
                if (!inst.results.empty()) {
                    auto val_attr = inst.attributes.get("value");
                    i64 v = (val_attr && val_attr->kind == AttrKind::Integer)
                        ? val_attr->integer : 0;
                    e.mov_imm64(X86Reg::RAX, static_cast<u64>(v));
                }
                break;
            }
            case CGOpcode::Add: {
                if (inst.results.size() == 1 && inst.operands.size() >= 2) {
                    // add rcx, rdx
                    e.add_reg(X86Reg::RCX, X86Reg::RDX);
                }
                break;
            }
            case CGOpcode::Mul: {
                if (inst.results.size() == 1 && inst.operands.size() >= 2) {
                    e.imul_reg(X86Reg::RAX, X86Reg::RCX);
                }
                break;
            }
            case CGOpcode::FMA: {
                if (inst.operands.size() >= 3) {
                    // vfmadd231ps xmm0, xmm1, xmm2
                    e.vfmadd231ps(X86VReg::XMM0, X86VReg::XMM1, X86VReg::XMM2,
                                  VEXWidth::XMM);
                }
                break;
            }
            case CGOpcode::VectorLoad: {
                if (inst.results.size() == 1 && inst.operands.size() >= 2) {
                    // vmovaps xmm0, [rdi + offset]
                    // Use a fixed offset of 0 for the foundational impl.
                    e.vmovaps_load(X86VReg::XMM0, X86Reg::RDI, 0, VEXWidth::XMM);
                }
                break;
            }
            case CGOpcode::VectorStore: {
                if (inst.operands.size() >= 3) {
                    e.vmovaps_store(X86Reg::RDI, 0, X86VReg::XMM0, VEXWidth::XMM);
                }
                break;
            }
            case CGOpcode::Load: {
                if (inst.results.size() == 1 && inst.operands.size() >= 2) {
                    e.load_reg(X86Reg::RAX, X86Reg::RDI, 0);
                }
                break;
            }
            case CGOpcode::Store: {
                if (inst.operands.size() >= 3) {
                    e.store_reg(X86Reg::RDI, 0, X86Reg::RAX);
                }
                break;
            }
            case CGOpcode::Barrier: {
                // mfence
                e.emit_byte(0x0F);
                e.emit_byte(0xAE);
                e.emit_byte(0xF0);
                break;
            }
            case CGOpcode::Reduce:
            case CGOpcode::AsyncCopy:
            case CGOpcode::Prefetch:
            case CGOpcode::Broadcast:
            case CGOpcode::Shuffle:
            case CGOpcode::Cmp:
            case CGOpcode::Select:
            case CGOpcode::Cast:
                // Emit a NOP as placeholder; real implementation would emit
                // the corresponding x86 instruction.
                e.nop();
                break;
        }
    }

    // Epilogue: restore stack, pop, ret.
    // add rsp, 256
    e.emit_rex_public(true, false, false, false);
    e.emit_byte(0x81);
    e.emit_byte(0xC4);  // modrm: mod=11, reg=0 (ADD), rm=4 (RSP)
    e.emit_u32(256);

    e.pop(X86Reg::RBP);
    e.ret();

    return e.take_bytes();
}

} // namespace cg
