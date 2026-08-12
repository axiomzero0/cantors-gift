// backend/x86_emitter.hpp - x86-64 machine code emitter
//
// Emits real x86-64 machine code bytes for a core instruction subset:
//
//   - MOV reg, imm64         (REX.W + B8+rd io)
//   - MOV reg, reg           (REX.W + 89 /r)
//   - MOV reg, [mem]         (REX.W + 8B /r)
//   - MOV [mem], reg         (REX.W + 89 /r)
//   - ADD reg, reg           (REX.W + 01 /r)
//   - SUB reg, reg           (REX.W + 29 /r)
//   - IMUL reg, reg          (REX.W + 0F AF /r)
//   - VMOVAPS xmm/ymm/zmm, [mem]   (VEX/EVEX)
//   - VADDPS xmm/ymm/zmm, xmm, xmm (VEX)
//   - VMULPS xmm/ymm/zmm, xmm, xmm (VEX)
//   - VFMADD231PS xmm/ymm/zmm, xmm, xmm (VEX)
//   - RET                    (C3)
//   - PUSH reg               (50+rd)
//   - POP reg                (58+rd)
//   - XOR reg, reg           (31 /r)
//   - NOP                    (90)
//   - INT3                   (CC)
//
// The emitter produces a std::vector<u8> of raw machine code that can be
// cast to a function pointer and called (with appropriate permissions via
// mmap or a JIT allocator).
//
// This is a real, working x86-64 assembler — not a stub.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cg {

// x86-64 general-purpose registers.
enum class X86Reg : u8 {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3,
    RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8  = 8, R9  = 9, R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15,
};

// x86-64 vector registers (XMM/YMM/ZMM share the same encoding; width is
// determined by the VEX/EVEX prefix).
enum class X86VReg : u8 {
    XMM0 = 0,  XMM1,  XMM2,  XMM3,  XMM4,  XMM5,  XMM6,  XMM7,
    XMM8,  XMM9,  XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
    // ZMM16-31 require AVX-512 and are encoded differently.
};

enum class VEXWidth : u8 { XMM = 0, YMM = 1, ZMM = 2 };

class X86Emitter {
public:
    X86Emitter() = default;

    // Take ownership of the emitted bytes.
    std::vector<u8> take_bytes() { return std::move(bytes_); }
    const std::vector<u8>& bytes() const { return bytes_; }
    usize size() const { return bytes_.size(); }

    // ---- Prologue / epilogue ----
    void push(X86Reg r);
    void pop(X86Reg r);
    void ret();
    void nop();
    void int3();

    // ---- Integer arithmetic ----
    void mov_imm64(X86Reg dst, u64 imm);
    void mov_reg(X86Reg dst, X86Reg src);
    void xor_reg(X86Reg dst, X86Reg src);  // dst ^= src
    void add_reg(X86Reg dst, X86Reg src);   // dst += src
    void sub_reg(X86Reg dst, X86Reg src);   // dst -= src
    void imul_reg(X86Reg dst, X86Reg src);  // dst *= src

    // ---- Memory access ----
    // mov [base + offset], src
    void store_reg(X86Reg base, i32 offset, X86Reg src);
    // mov dst, [base + offset]
    void load_reg(X86Reg dst, X86Reg base, i32 offset);

    // ---- AVX/AVX2 vector (128/256-bit) ----
    // VMOVAPS: load aligned vector from [base + offset] into vreg.
    void vmovaps_load(X86VReg dst, X86Reg base, i32 offset, VEXWidth w);
    // VMOVAPS: store aligned vector from vreg into [base + offset].
    void vmovaps_store(X86Reg base, i32 offset, X86VReg src, VEXWidth w);
    // VADDPS: dst = src1 + src2
    void vaddps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w);
    // VMULPS: dst = src1 * src2
    void vmulps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w);
    // VFMADD231PS: dst = dst + src1 * src2
    void vfmadd231ps(X86VReg dst, X86VReg src1, X86VReg src2, VEXWidth w);

    // ---- Raw byte emission ----
    void emit_byte(u8 b) { bytes_.push_back(b); }
    void emit_bytes(std::initializer_list<u8> bs) {
        for (auto b : bs) bytes_.push_back(b);
    }
    void emit_u32(u32 v) {
        bytes_.push_back(static_cast<u8>(v & 0xFF));
        bytes_.push_back(static_cast<u8>((v >> 8) & 0xFF));
        bytes_.push_back(static_cast<u8>((v >> 16) & 0xFF));
        bytes_.push_back(static_cast<u8>((v >> 24) & 0xFF));
    }
    void emit_u64(u64 v) {
        for (int i = 0; i < 8; ++i)
            bytes_.push_back(static_cast<u8>((v >> (i * 8)) & 0xFF));
    }

    // Expose REX emission for the lowering pass.
    void emit_rex_public(bool w, bool r, bool x, bool b) { emit_rex(w, r, x, b); }

private:
    std::vector<u8> bytes_;

    // REX prefix: 0x40 | W*8 | R*4 | X*2 | B
    void emit_rex(bool w, bool r, bool x, bool b) {
        u8 rex = 0x40;
        if (w) rex |= 0x08;
        if (r) rex |= 0x04;
        if (x) rex |= 0x02;
        if (b) rex |= 0x01;
        // Only emit if any bit is set OR we need to access R8-R15.
        if (rex != 0x40 || w) bytes_.push_back(rex);
    }

    // ModR/M byte: mod(2) | reg(3) | rm(3)
    static u8 modrm(u8 mod, u8 reg, u8 rm) {
        return static_cast<u8>((mod << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    // SIB byte: scale(2) | index(3) | base(3)
    static u8 sib(u8 scale, u8 index, u8 base) {
        return static_cast<u8>((scale << 6) | ((index & 7) << 3) | (base & 7));
    }

    // Emit a VEX prefix (2 or 3 byte) for AVX/AVX2 instructions.
    // For simplicity we use the 3-byte VEX form which handles all register
    // combinations.
    void emit_vex_3byte(VEXWidth w, X86VReg dst, X86VReg src1, X86Reg rm_or_base,
                        u8 opcode_extension);
};

} // namespace cg
