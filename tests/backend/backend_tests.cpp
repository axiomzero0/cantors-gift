// tests/backend_tests.cpp - backend tests
#include "cg/backend/cpu_backend.hpp"
#include "cg/backend/nvidia_backend.hpp"
#include "cg/backend/lowering.hpp"
#include "cg/codegen/codegen_ir.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

namespace {

CGModule build_simple_module() {
    CGModule m;
    auto& fn = m.create_function("test_kernel");
    auto a = fn.allocate(DType::F32, 8);
    auto b = fn.allocate(DType::F32, 8);
    auto acc = fn.allocate(DType::F32, 8);
    auto ptr = fn.allocate(DType::U64);
    auto off = fn.allocate(DType::I64);
    fn.emit(make_vector_load(a, ptr, off, 8, MemorySpace::Generic));
    fn.emit(make_vector_load(b, ptr, off, 8, MemorySpace::Generic));
    fn.emit(make_fma(acc, a, b));
    fn.emit(make_vector_store(ptr, off, acc, MemorySpace::Generic));
    fn.emit(make_barrier());
    fn.args = {ptr, off};
    return m;
}

} // namespace

TEST(Backend, PTXEmitterProducesValidText) {
    auto m = build_simple_module();
    NvidiaBackend nv;
    auto exe = nv.compile(m);
    EXPECT_FALSE(exe->ptx_text.empty());
    EXPECT_NE(exe->ptx_text.find(".version"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find(".target"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find(".entry"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find("fma.rn"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find("bar.sync"), std::string::npos);
}

TEST(Backend, PTXEmitterHasCorrectEntrypoint) {
    auto m = build_simple_module();
    NvidiaBackend nv;
    auto exe = nv.compile(m);
    ASSERT_EQ(exe->entrypoints.size(), 1u);
    EXPECT_EQ(exe->entrypoints[0].first, "test_kernel");
}

TEST(Backend, X86EmitterProducesMachineCode) {
    auto m = build_simple_module();
    CpuBackend cpu;
    auto exe = cpu.compile(m);
    EXPECT_GT(exe->machine_code.size(), 0u);
    EXPECT_EQ(exe->machine_code.back(), 0xC3); // ret
    EXPECT_EQ(exe->machine_code[0], 0x55);     // push rbp
}

TEST(Backend, X86EmitterHasCorrectEntrypoint) {
    auto m = build_simple_module();
    CpuBackend cpu;
    auto exe = cpu.compile(m);
    ASSERT_EQ(exe->entrypoints.size(), 1u);
    EXPECT_EQ(exe->entrypoints[0].first, "test_kernel");
    EXPECT_EQ(exe->entrypoints[0].second, 0u);
}

TEST(Backend, X86EmitterMultipleFunctions) {
    CGModule m;
    m.create_function("first");
    m.create_function("second");

    CpuBackend cpu;
    auto exe = cpu.compile(m);
    ASSERT_EQ(exe->entrypoints.size(), 2u);
    EXPECT_EQ(exe->entrypoints[0].first, "first");
    EXPECT_EQ(exe->entrypoints[1].first, "second");
    EXPECT_GT(exe->entrypoints[1].second, exe->entrypoints[0].second);
}

TEST(Backend, EmitTextReturnsPTX) {
    auto m = build_simple_module();
    NvidiaBackend nv;
    auto text = nv.emit_text(m);
    ASSERT_TRUE(text.has_value());
    EXPECT_NE(text->find("test_kernel"), std::string::npos);
}

TEST(Backend, EmitTextReturnsDisassembly) {
    auto m = build_simple_module();
    CpuBackend cpu;
    auto text = cpu.emit_text(m);
    ASSERT_TRUE(text.has_value());
    EXPECT_NE(text->find("test_kernel"), std::string::npos);
}

TEST(Backend, X86EmitterIndividualInstructions) {
    X86Emitter e;
    e.push(X86Reg::RBP);
    e.mov_reg(X86Reg::RBP, X86Reg::RSP);
    e.mov_imm64(X86Reg::RAX, 0x123456789ABCDEF0ULL);
    e.add_reg(X86Reg::RAX, X86Reg::RCX);
    e.imul_reg(X86Reg::RAX, X86Reg::RCX);
    e.xor_reg(X86Reg::RAX, X86Reg::RAX);
    e.ret();

    auto bytes = e.take_bytes();
    EXPECT_GT(bytes.size(), 0u);
    EXPECT_EQ(bytes.back(), 0xC3);
}

TEST(Backend, X86EmitterVectorInstructions) {
    X86Emitter e;
    e.vmovaps_load(X86VReg::XMM0, X86Reg::RDI, 0, VEXWidth::XMM);
    e.vmovaps_load(X86VReg::XMM1, X86Reg::RSI, 0, VEXWidth::XMM);
    e.vaddps(X86VReg::XMM2, X86VReg::XMM0, X86VReg::XMM1, VEXWidth::XMM);
    e.vmovaps_store(X86Reg::RDI, 0, X86VReg::XMM2, VEXWidth::XMM);
    e.ret();

    auto bytes = e.take_bytes();
    EXPECT_GT(bytes.size(), 0u);
    EXPECT_EQ(bytes.back(), 0xC3);
}

TEST(Backend, TargetInfo) {
    CpuBackend cpu;
    const auto& ti = cpu.target_info();
    EXPECT_EQ(ti.device.kind, DeviceId::Kind::CPU);
    EXPECT_GT(ti.hardware.peak_flops(DType::F32), 0.0);

    NvidiaBackend nv;
    const auto& ti2 = nv.target_info();
    EXPECT_EQ(ti2.device.kind, DeviceId::Kind::CUDA);
    EXPECT_GT(ti2.hardware.peak_flops(DType::F16), 0.0);
}
