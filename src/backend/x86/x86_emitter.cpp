// backend/x86_emitter.cpp - real x86-64 machine code emission
#include "cg/backend/x86/x86_emitter.hpp"

namespace cg {

// ---- Prologue / epilogue ----

void X86Emitter::push(X86Reg r) {
    // 50+rd (no REX needed for R8-R15 because the bit is encoded in the opcode)
    if (static_cast<u8>(r) >= 8) {
        emit_rex(false, false, false, true);
    }
    bytes_.push_back(static_cast<u8>(0x50 + (static_cast<u8>(r) & 7)));
}

void X86Emitter::pop(X86Reg r) {
    if (static_cast<u8>(r) >= 8) {
        emit_rex(false, false, false, true);
    }
    bytes_.push_back(static_cast<u8>(0x58 + (static_cast<u8>(r) & 7)));
}

void X86Emitter::ret() {
    bytes_.push_back(0xC3);
}

void X86Emitter::nop() {
    bytes_.push_back(0x90);
}

void X86Emitter::int3() {
    bytes_.push_back(0xCC);
}

// ---- Integer arithmetic ----

void X86Emitter::mov_imm64(X86Reg dst, u64 imm) {
    // REX.W + B8+rd io
    bool b = static_cast<u8>(dst) >= 8;
    emit_rex(true, false, false, b);
    bytes_.push_back(static_cast<u8>(0xB8 + (static_cast<u8>(dst) & 7)));
    emit_u64(imm);
}

void X86Emitter::mov_reg(X86Reg dst, X86Reg src) {
    // REX.W + 89 /r  (mov src -> dst)
    // or REX.W + 8B /r (mov dst <- src)
    // We use 89 /r: MOV r/m64, r64
    bool r = static_cast<u8>(src) >= 8;
    bool b = static_cast<u8>(dst) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x89);
    bytes_.push_back(modrm(0b11, static_cast<u8>(src) & 7, static_cast<u8>(dst) & 7));
}

void X86Emitter::xor_reg(X86Reg dst, X86Reg src) {
    // REX.W + 31 /r  XOR r/m64, r64
    bool r = static_cast<u8>(src) >= 8;
    bool b = static_cast<u8>(dst) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x31);
    bytes_.push_back(modrm(0b11, static_cast<u8>(src) & 7, static_cast<u8>(dst) & 7));
}

void X86Emitter::add_reg(X86Reg dst, X86Reg src) {
    // REX.W + 01 /r  ADD r/m64, r64
    bool r = static_cast<u8>(src) >= 8;
    bool b = static_cast<u8>(dst) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x01);
    bytes_.push_back(modrm(0b11, static_cast<u8>(src) & 7, static_cast<u8>(dst) & 7));
}

void X86Emitter::sub_reg(X86Reg dst, X86Reg src) {
    // REX.W + 29 /r  SUB r/m64, r64
    bool r = static_cast<u8>(src) >= 8;
    bool b = static_cast<u8>(dst) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x29);
    bytes_.push_back(modrm(0b11, static_cast<u8>(src) & 7, static_cast<u8>(dst) & 7));
}

void X86Emitter::imul_reg(X86Reg dst, X86Reg src) {
    // REX.W + 0F AF /r  IMUL r64, r/m64
    bool r = static_cast<u8>(dst) >= 8;
    bool b = static_cast<u8>(src) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x0F);
    bytes_.push_back(0xAF);
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src) & 7));
}

// ---- Memory access ----

void X86Emitter::store_reg(X86Reg base, i32 offset, X86Reg src) {
    // REX.W + 89 /r  MOV [base + disp32], r64
    bool r = static_cast<u8>(src) >= 8;
    bool b = static_cast<u8>(base) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x89);
    // If base is RSP or R12, we need a SIB byte.
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) { // RSP/R12
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, base_lo));
    }
    // 32-bit displacement
    emit_u32(static_cast<u32>(offset));
}

void X86Emitter::load_reg(X86Reg dst, X86Reg base, i32 offset) {
    // REX.W + 8B /r  MOV r64, [base + disp32]
    bool r = static_cast<u8>(dst) >= 8;
    bool b = static_cast<u8>(base) >= 8;
    emit_rex(true, r, false, b);
    bytes_.push_back(0x8B);
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

// ---- VEX prefix ----
//
// 3-byte VEX prefix layout (Intel SDM Vol. 2A, Section 2.3.2):
//
//   Byte 1:  C4
//   Byte 2:  R(1) X(1) B(1) mmmmm(5)
//   Byte 3:  W(1) vvvv(4) L(1) pp(2)
//
// Field meanings:
//   R      = ~reg field of ModR/M (inverted). 0 = R8-R15, 1 = R0-R7.
//   X      = ~index field of SIB (inverted). 1 = no index extension.
//   B      = ~r/m field of ModR/M (inverted). 0 = R8-R15, 1 = R0-R7.
//   mmmmm  = opcode map: 00001 = 0F, 00010 = 0F 38, 00011 = 0F 3A.
//   W      = opcode-specific. For most VEX-encoded SSE/AVX ops, W=0.
//   vvvv   = ~src1 (inverted, 4 bits). 1111 = no src1.
//   L      = vector length: 0 = 128-bit (XMM), 1 = 256-bit (YMM).
//   pp     = prefix: 00 = none, 01 = 66, 10 = F3, 11 = F2.
//
// CRITICAL: the pp field selects between single/double precision variants:
//   VADDPS (single) = VEX.128.0F.58       -> pp=00 (no prefix)
//   VADDPD (double) = VEX.128.66.0F.58    -> pp=01 (66 prefix)
// Setting pp=01 for VADDPS produces VADDPD — a silent correctness bug.
//
// For all single-precision packed (PS) ops we use pp=00.
// For double-precision packed (PD) ops we would use pp=01.

void X86Emitter::emit_vex_3byte(VEXWidth w, X86VReg dst, X86VReg src1,
                                 X86Reg rm_or_base, u8 opcode_extension) {
    (void)opcode_extension;

    u8 r_bit = (static_cast<u8>(dst) >= 8) ? 0 : 1;         // inverted
    u8 x_bit = 1;                                             // no index
    u8 b_bit = (static_cast<u8>(rm_or_base) >= 8) ? 0 : 1;  // inverted
    u8 w_bit = 0;       // W=0 for single-precision packed ops
    u8 vvvv = static_cast<u8>(~static_cast<u8>(src1)) & 0xF; // inverted src1
    u8 l_bit = (w == VEXWidth::YMM) ? 1 : 0;                 // 0=XMM(128), 1=YMM(256)
    u8 pp = 0b00;       // pp=00 for PS (single-precision packed); pp=01 would be PD

    u8 mmmmm = 0b00001; // 0F map
    u8 byte2 = static_cast<u8>((r_bit << 7) | (x_bit << 6) | (b_bit << 5) | mmmmm);
    u8 byte3 = static_cast<u8>((w_bit << 7) | (vvvv << 3) | (l_bit << 2) | pp);

    bytes_.push_back(0xC4);
    bytes_.push_back(byte2);
    bytes_.push_back(byte3);
}

// ---- AVX/AVX2 vector ----

void X86Emitter::vmovaps_load(X86VReg dst, X86Reg base, i32 offset, VEXWidth w) {
    // VMOVAPS xmm/ymm, [m128/m256]
    // VEX.128.0F.58 — wait, that's VADDPS.
    // VMOVAPS is VEX.128.0F.28 (load) / VEX.128.0F.29 (store).
    // Actually: VMOVAPS r, m is VEX.[128|256].0F.28 /r
    emit_vex_3byte(w, dst, X86VReg::XMM0 /*unused as src1*/, base, 0);
    bytes_.push_back(0x28); // opcode for VMOVAPS load
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

void X86Emitter::vmovaps_store(X86Reg base, i32 offset, X86VReg src, VEXWidth w) {
    // VMOVAPS [m128/m256], xmm/ymm
    // VEX.[128|256].0F.29 /r
    emit_vex_3byte(w, X86VReg::XMM0, X86VReg::XMM0, base, 0);
    bytes_.push_back(0x29);
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

// VMOVUPS: unaligned load/store. Same encoding as VMOVAPS but opcodes
// 0x10 (load) and 0x11 (store) instead of 0x28/0x29.
void X86Emitter::vmovups_load(X86VReg dst, X86Reg base, i32 offset, VEXWidth w) {
    emit_vex_3byte(w, dst, X86VReg::XMM0, base, 0);
    bytes_.push_back(0x10); // VMOVUPS load opcode
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

void X86Emitter::vmovups_store(X86Reg base, i32 offset, X86VReg src, VEXWidth w) {
    emit_vex_3byte(w, X86VReg::XMM0, X86VReg::XMM0, base, 0);
    bytes_.push_back(0x11); // VMOVUPS store opcode
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

// VMOVSS: scalar single-precision load/store.
// VEX.128.F3.0F.10 /r (load): loads 32 bits from memory into low lane.
// VEX.128.F3.0F.11 /r (store): stores low 32 bits to memory.
// pp=F3=10 (binary), mmmmm=00001 (0F map), W=0, L=0.
void X86Emitter::vmovss_load(X86VReg dst, X86Reg base, i32 offset) {
    // 3-byte VEX: C4 [R X B mmmmm] [W vvvv L pp]
    u8 r_bit = (static_cast<u8>(dst) >= 8) ? 0 : 1;
    u8 x_bit = 1;
    u8 b_bit = (static_cast<u8>(base) >= 8) ? 0 : 1;
    u8 w_bit = 0;
    u8 vvvv = 0xF; // no src1
    u8 l_bit = 0;  // 128-bit
    u8 pp = 0b10;  // F3 prefix
    u8 mmmmm = 0b00001; // 0F map
    u8 byte2 = static_cast<u8>((r_bit << 7) | (x_bit << 6) | (b_bit << 5) | mmmmm);
    u8 byte3 = static_cast<u8>((w_bit << 7) | (vvvv << 3) | (l_bit << 2) | pp);
    bytes_.push_back(0xC4);
    bytes_.push_back(byte2);
    bytes_.push_back(byte3);
    bytes_.push_back(0x10); // VMOVSS load opcode
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(dst) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

void X86Emitter::vmovss_store(X86Reg base, i32 offset, X86VReg src) {
    u8 r_bit = (static_cast<u8>(src) >= 8) ? 0 : 1;
    u8 x_bit = 1;
    u8 b_bit = (static_cast<u8>(base) >= 8) ? 0 : 1;
    u8 w_bit = 0;
    u8 vvvv = 0xF;
    u8 l_bit = 0;
    u8 pp = 0b10;  // F3 prefix
    u8 mmmmm = 0b00001;
    u8 byte2 = static_cast<u8>((r_bit << 7) | (x_bit << 6) | (b_bit << 5) | mmmmm);
    u8 byte3 = static_cast<u8>((w_bit << 7) | (vvvv << 3) | (l_bit << 2) | pp);
    bytes_.push_back(0xC4);
    bytes_.push_back(byte2);
    bytes_.push_back(byte3);
    bytes_.push_back(0x11); // VMOVSS store opcode
    u8 base_lo = static_cast<u8>(base) & 7;
    if (base_lo == 4) {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, 4));
        bytes_.push_back(sib(0, 4, base_lo));
    } else {
        bytes_.push_back(modrm(0b10, static_cast<u8>(src) & 7, base_lo));
    }
    emit_u32(static_cast<u32>(offset));
}

void X86Emitter::vaddps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w) {
    // VADDPS xmm/ymm, xmm, xmm/mem
    // VEX.[128|256].0F.58 /r
    // src2 is in the r/m field. For register-register, mod=11.
    bool r2_high = static_cast<u8>(src2) >= 8;
    emit_vex_3byte(w, dst, src1, static_cast<X86Reg>(static_cast<u8>(src2) & 7), 0);
    bytes_.push_back(0x58);
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
    (void)r2_high;
}

void X86Emitter::vmulps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w) {
    // VMULPS: VEX.[128|256].0F.59 /r
    emit_vex_3byte(w, dst, src1, static_cast<X86Reg>(static_cast<u8>(src2) & 7), 0);
    bytes_.push_back(0x59);
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
}

void X86Emitter::vfmadd231ps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w) {
    // VFMADD231PS: VEX.128.66.0F38.W0 B8 /r
    u8 r_bit = (static_cast<u8>(dst) >= 8) ? 0 : 1;
    u8 x_bit = 1;
    u8 b_bit = (static_cast<u8>(src2) >= 8) ? 0 : 1;
    u8 w_bit = 0;
    u8 vvvv = static_cast<u8>(~static_cast<u8>(src1)) & 0xF;
    u8 l_bit = (w == VEXWidth::YMM) ? 1 : 0;
    u8 pp = 0b01;       // 66 prefix (packed)
    u8 mmmmm = 0b00010; // 0F38 map
    u8 byte2 = static_cast<u8>((r_bit << 7) | (x_bit << 6) | (b_bit << 5) | mmmmm);
    u8 byte3 = static_cast<u8>((w_bit << 7) | (vvvv << 3) | (l_bit << 2) | pp);
    bytes_.push_back(0xC4);
    bytes_.push_back(byte2);
    bytes_.push_back(byte3);
    bytes_.push_back(0xB8);
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
}

// VFMADD231SS: VEX.128.66.0F38.W0 B9 /r
// Scalar FMA: only operates on the low 32 bits (one float).
// Same pp as packed (66), different opcode (B9 vs B8).
void X86Emitter::vfmadd231ss(X86VReg dst, X86VReg src1, X86VReg src2) {
    u8 r_bit = (static_cast<u8>(dst) >= 8) ? 0 : 1;
    u8 x_bit = 1;
    u8 b_bit = (static_cast<u8>(src2) >= 8) ? 0 : 1;
    u8 w_bit = 0;
    u8 vvvv = static_cast<u8>(~static_cast<u8>(src1)) & 0xF;
    u8 l_bit = 0;       // 128-bit
    u8 pp = 0b01;       // 66 prefix (same as packed FMA)
    u8 mmmmm = 0b00010; // 0F38 map
    u8 byte2 = static_cast<u8>((r_bit << 7) | (x_bit << 6) | (b_bit << 5) | mmmmm);
    u8 byte3 = static_cast<u8>((w_bit << 7) | (vvvv << 3) | (l_bit << 2) | pp);
    bytes_.push_back(0xC4);
    bytes_.push_back(byte2);
    bytes_.push_back(byte3);
    bytes_.push_back(0xB9); // VFMADD231SS opcode (B9, not B8)
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
}

// ---- VXORPS: zero a vector register ----
// VEX.128.0F.57 /r
// pp=00 (no prefix) for single-precision packed.
void X86Emitter::vxorps(X86VReg dst, X86VReg src) {
    emit_vex_3byte(VEXWidth::XMM, dst, src, static_cast<X86Reg>(static_cast<u8>(src) & 7), 0);
    bytes_.push_back(0x57); // VXORPS opcode
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src) & 7));
}

// ---- VMAXPS: dst = max(src1, src2) ----
// VEX.128.0F.5F /r, pp=00
void X86Emitter::vmaxps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w) {
    emit_vex_3byte(w, dst, src1, static_cast<X86Reg>(static_cast<u8>(src2) & 7), 0);
    bytes_.push_back(0x5F); // VMAXPS opcode
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
}

// ---- VSUBPS: dst = src1 - src2 ----
// VEX.128.0F.5C /r, pp=00
void X86Emitter::vsubps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w) {
    emit_vex_3byte(w, dst, src1, static_cast<X86Reg>(static_cast<u8>(src2) & 7), 0);
    bytes_.push_back(0x5C); // VSUBPS opcode
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
}

// ---- Loop / control flow ----

void X86Emitter::mark_label(LabelId id) {
    label_positions_[id] = bytes_.size();
}

void X86Emitter::jne(LabelId id) {
    // JNZ rel32: 0F 85 cd
    bytes_.push_back(0x0F);
    bytes_.push_back(0x85);
    // Reserve 4 bytes for the displacement; patch later.
    pending_patches_.push_back({bytes_.size(), id});
    emit_u32(0);
}

void X86Emitter::je(LabelId id) {
    // JZ rel32: 0F 84 cd
    bytes_.push_back(0x0F);
    bytes_.push_back(0x84);
    pending_patches_.push_back({bytes_.size(), id});
    emit_u32(0);
}

void X86Emitter::jnz(LabelId id) { jne(id); }
void X86Emitter::jz(LabelId id)  { je(id); }

void X86Emitter::jmp(LabelId id) {
    // JMP rel32: E9 cd
    bytes_.push_back(0xE9);
    pending_patches_.push_back({bytes_.size(), id});
    emit_u32(0);
}

void X86Emitter::dec_reg(X86Reg r) {
    // REX.W + FF /1  DEC r/m64
    bool b = static_cast<u8>(r) >= 8;
    emit_rex(true, false, false, b);
    bytes_.push_back(0xFF);
    bytes_.push_back(modrm(0b11, 1, static_cast<u8>(r) & 7));
}

void X86Emitter::cmp_imm32(X86Reg r, i32 imm) {
    // REX.W + 81 /7  CMP r/m64, imm32
    bool b = static_cast<u8>(r) >= 8;
    emit_rex(true, false, false, b);
    bytes_.push_back(0x81);
    bytes_.push_back(modrm(0b11, 7, static_cast<u8>(r) & 7));
    emit_u32(static_cast<u32>(imm));
}

void X86Emitter::resolve_patches() {
    for (auto& p : pending_patches_) {
        auto it = label_positions_.find(p.target_label);
        if (it == label_positions_.end()) continue; // unresolved label
        // The displacement is relative to the instruction AFTER the jump.
        // The jump instruction is: opcode (2 bytes for Jcc, 1 for JMP) +
        // 4 bytes displacement. The displacement is measured from the end
        // of the jump instruction.
        usize jump_end = p.patch_offset + 4; // end of the 4-byte displacement
        i32 disp = static_cast<i32>(it->second) - static_cast<i32>(jump_end);
        // Patch the 4 bytes at p.patch_offset.
        bytes_[p.patch_offset + 0] = static_cast<u8>(disp & 0xFF);
        bytes_[p.patch_offset + 1] = static_cast<u8>((disp >> 8) & 0xFF);
        bytes_[p.patch_offset + 2] = static_cast<u8>((disp >> 16) & 0xFF);
        bytes_[p.patch_offset + 3] = static_cast<u8>((disp >> 24) & 0xFF);
    }
    pending_patches_.clear();
}

} // namespace cg
