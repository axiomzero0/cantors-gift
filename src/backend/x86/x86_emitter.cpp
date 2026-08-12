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
    // Here dst is the memory operand and src is the vector register.
    // In VEX encoding, the vvvv field is 0 (no src1).
    emit_vex_3byte(w, X86VReg::XMM0 /*unused*/, X86VReg::XMM0 /*unused*/, base, 0);
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
    // Unlike VADDPS (which is VEX.0F.58 with pp=00), FMA uses pp=01 (66).
    // The 0F38 map + pp=01 + opcode B8 = VFMADD231PS (single-precision).
    u8 r_bit = (static_cast<u8>(dst) >= 8) ? 0 : 1;
    u8 x_bit = 1;
    u8 b_bit = (static_cast<u8>(src2) >= 8) ? 0 : 1;
    u8 w_bit = 0;       // W=0 for single-precision FMA
    u8 vvvv = static_cast<u8>(~static_cast<u8>(src1)) & 0xF;
    u8 l_bit = (w == VEXWidth::YMM) ? 1 : 0;
    u8 pp = 0b01;       // 66 prefix (required for VFMADD231PS per SDM)
    u8 mmmmm = 0b00010; // 0F38 map
    u8 byte2 = static_cast<u8>((r_bit << 7) | (x_bit << 6) | (b_bit << 5) | mmmmm);
    u8 byte3 = static_cast<u8>((w_bit << 7) | (vvvv << 3) | (l_bit << 2) | pp);
    bytes_.push_back(0xC4);
    bytes_.push_back(byte2);
    bytes_.push_back(byte3);
    bytes_.push_back(0xB8); // VFMADD231PS opcode in 0F38 map
    bytes_.push_back(modrm(0b11, static_cast<u8>(dst) & 7, static_cast<u8>(src2) & 7));
}

} // namespace cg
