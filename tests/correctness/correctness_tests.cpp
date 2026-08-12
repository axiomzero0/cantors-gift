// tests/correctness/correctness_tests.cpp - tests that execute generated code
//
// These tests verify correctness on real hardware by JIT-compiling x86
// machine code and executing it. They also verify PTX structural validity.
#include "cg/backend/cpu_backend.hpp"
#include "cg/backend/jit.hpp"
#include "cg/backend/lowering.hpp"
#include "cg/backend/nvidia_backend.hpp"
#include "cg/backend/ptx/ptx_emitter.hpp"
#include "cg/backend/x86/x86_emitter.hpp"
#include "cg/codegen/codegen_ir.hpp"
#include "cg/cost/estimator.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/lowering/tensor_to_codegen.hpp"
#include "cg/optimization/const_fold/sccp.hpp"
#include "cg/analysis/analysis.hpp"
#include "cg/schedule/schedule.hpp"

#include "cg/test/gtest_compat.hpp"

#include <cmath>
#include <cstring>

using namespace cg;

// ---- VADDPS: verify single-precision add (not double!) ----

TEST(Correctness, VaddpsSinglePrecision) {
    X86Emitter e;
    e.vmovaps_load(X86VReg::XMM0, X86Reg::RDI, 0, VEXWidth::XMM);
    e.vmovaps_load(X86VReg::XMM1, X86Reg::RDI, 16, VEXWidth::XMM);
    e.vaddps(X86VReg::XMM0, X86VReg::XMM0, X86VReg::XMM1, VEXWidth::XMM);
    e.vmovaps_store(X86Reg::RDI, 32, X86VReg::XMM0, VEXWidth::XMM);
    e.ret();

    JITFunction<void(*)(float*)> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());

    alignas(16) float buf[12] = {1,2,3,4, 10,20,30,40, 0,0,0,0};
    jit(buf);

    EXPECT_FLOAT_EQ(buf[8],  11.0f);
    EXPECT_FLOAT_EQ(buf[9],  22.0f);
    EXPECT_FLOAT_EQ(buf[10], 33.0f);
    EXPECT_FLOAT_EQ(buf[11], 44.0f);
}

TEST(Correctness, VmulpsSinglePrecision) {
    X86Emitter e;
    e.vmovaps_load(X86VReg::XMM0, X86Reg::RDI, 0, VEXWidth::XMM);
    e.vmovaps_load(X86VReg::XMM1, X86Reg::RDI, 16, VEXWidth::XMM);
    e.vmulps(X86VReg::XMM0, X86VReg::XMM0, X86VReg::XMM1, VEXWidth::XMM);
    e.vmovaps_store(X86Reg::RDI, 32, X86VReg::XMM0, VEXWidth::XMM);
    e.ret();

    JITFunction<void(*)(float*)> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());

    alignas(16) float buf[12] = {1,2,3,4, 10,20,30,40, 0,0,0,0};
    jit(buf);

    EXPECT_FLOAT_EQ(buf[8],  10.0f);
    EXPECT_FLOAT_EQ(buf[9],  40.0f);
    EXPECT_FLOAT_EQ(buf[10], 90.0f);
    EXPECT_FLOAT_EQ(buf[11], 160.0f);
}

TEST(Correctness, Vfmadd231psSinglePrecision) {
    X86Emitter e;
    e.vmovaps_load(X86VReg::XMM0, X86Reg::RDI, 0, VEXWidth::XMM);
    e.vmovaps_load(X86VReg::XMM1, X86Reg::RDI, 16, VEXWidth::XMM);
    e.vmovaps_load(X86VReg::XMM2, X86Reg::RDI, 32, VEXWidth::XMM);
    e.vfmadd231ps(X86VReg::XMM0, X86VReg::XMM1, X86VReg::XMM2, VEXWidth::XMM);
    e.vmovaps_store(X86Reg::RDI, 48, X86VReg::XMM0, VEXWidth::XMM);
    e.ret();

    JITFunction<void(*)(float*)> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());

    alignas(16) float buf[16] = {1,2,3,4, 2,2,2,2, 3,3,3,3, 0,0,0,0};
    jit(buf);

    EXPECT_FLOAT_EQ(buf[12], 7.0f);
    EXPECT_FLOAT_EQ(buf[13], 8.0f);
    EXPECT_FLOAT_EQ(buf[14], 9.0f);
    EXPECT_FLOAT_EQ(buf[15], 10.0f);
}

// ---- Integer arithmetic ----

TEST(Correctness, IntegerAdd) {
    X86Emitter e;
    e.mov_imm64(X86Reg::RAX, 10);
    e.mov_imm64(X86Reg::RCX, 20);
    e.add_reg(X86Reg::RAX, X86Reg::RCX);
    e.ret();
    JITFunction<int(*)()> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());
    EXPECT_EQ(jit(), 30);
}

TEST(Correctness, IntegerMul) {
    X86Emitter e;
    e.mov_imm64(X86Reg::RAX, 6);
    e.mov_imm64(X86Reg::RCX, 7);
    e.imul_reg(X86Reg::RAX, X86Reg::RCX);
    e.ret();
    JITFunction<int(*)()> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());
    EXPECT_EQ(jit(), 42);
}

// ---- PTX structural validity ----

TEST(Correctness, PTXNoRegRegAddressing) {
    CGModule cgm;
    auto& fn = cgm.create_function("test");
    auto ptr = fn.allocate(DType::U64);
    auto off = fn.allocate(DType::I64);
    auto dst = fn.allocate(DType::F32, 4);
    fn.emit(make_vector_load(dst, ptr, off, 4, MemorySpace::Generic));
    fn.args = {ptr, off};

    PTXEmitter emitter;
    std::string ptx = emitter.emit_kernel(fn, "test", "sm_80");

    // Should NOT contain [reg+reg] — should have an add instruction first.
    EXPECT_EQ(ptx.find("+%"), std::string::npos);
    EXPECT_NE(ptx.find("add.u64"), std::string::npos);
}

TEST(Correctness, PTXNoParamRegCollision) {
    CGModule cgm;
    auto& fn = cgm.create_function("test");
    auto ptr = fn.allocate(DType::U64);
    auto off = fn.allocate(DType::I64);
    fn.args = {ptr, off};

    PTXEmitter emitter;
    std::string ptx = emitter.emit_kernel(fn, "test", "sm_80");

    EXPECT_NE(ptx.find("_param_0"), std::string::npos);
    EXPECT_NE(ptx.find("ld.param.u64"), std::string::npos);
}

TEST(Correctness, PTXDeclaresSharedMemory) {
    CGModule cgm;
    auto& fn = cgm.create_function("test");
    auto ptr = fn.allocate(DType::U64);
    auto off = fn.allocate(DType::I64);
    auto dst = fn.allocate(DType::F32, 4);
    fn.emit(make_vector_load(dst, ptr, off, 4, MemorySpace::Shared));
    fn.args = {ptr, off};

    PTXEmitter emitter;
    std::string ptx = emitter.emit_kernel(fn, "test", "sm_80");

    EXPECT_NE(ptx.find(".shared"), std::string::npos);
}

TEST(Correctness, PTXHasThreadIndexIntrinsics) {
    CGModule cgm;
    auto& fn = cgm.create_function("test");
    auto ptr = fn.allocate(DType::U64);
    auto off = fn.allocate(DType::I64);
    auto dst = fn.allocate(DType::F32, 4);
    fn.emit(make_vector_load(dst, ptr, off, 4, MemorySpace::Generic));
    fn.args = {ptr, off};

    PTXEmitter emitter;
    std::string ptx = emitter.emit_kernel(fn, "test", "sm_80");

    EXPECT_NE(ptx.find("%tid.x"), std::string::npos);
    EXPECT_NE(ptx.find("%ntid.x"), std::string::npos);
    EXPECT_NE(ptx.find("%ctaid.x"), std::string::npos);
}

// ---- SCCP materializes constants ----

TEST(Correctness, SCCPMaterializesConstants) {
    Module m;
    auto f = m.create_function("sccp_test", {}, {make_tensor_type({}, DType::I32)});
    Builder b(f);

    std::string b5(4, '\0'); i32 v5 = 5; std::memcpy(b5.data(), &v5, 4);
    std::string b7(4, '\0'); i32 v7 = 7; std::memcpy(b7.data(), &v7, 4);

    AttributeDict a5; a5.set("shape", Attribute::make_int_array({}));
    a5.set("dtype", Attribute::make_dtype(DType::I32));
    a5.set("bytes", Attribute::make_string(std::move(b5)));
    auto* c5 = b.create(OP_CONSTANT, {}, a5);

    AttributeDict a7; a7.set("shape", Attribute::make_int_array({}));
    a7.set("dtype", Attribute::make_dtype(DType::I32));
    a7.set("bytes", Attribute::make_string(std::move(b7)));
    auto* c7 = b.create(OP_CONSTANT, {}, a7);

    auto sum = b.add(c5->results[0], c7->results[0]);
    b.output_tensor(sum);

    AnalysisManager am(m);
    SCCPPass sccp;
    sccp.run(m, am);

    bool found_12 = false;
    for (auto& op : *f->entry()) {
        if (op.opcode != OP_CONSTANT) continue;
        auto bytes_attr = op.attributes.get("bytes");
        if (!bytes_attr || bytes_attr->kind != AttrKind::String) continue;
        if (bytes_attr->str.size() != 4) continue;
        i32 val;
        std::memcpy(&val, bytes_attr->str.data(), 4);
        if (val == 12) found_12 = true;
    }
    EXPECT_TRUE(found_12);
}

// ---- Cost estimator uses schedule ----

TEST(Correctness, CostEstimatorUsesSchedule) {
    auto hw = HardwareModel::generic_nvidia_gpu();
    CostEstimator est(hw);

    Module m;
    auto f = m.create_function("mm",
        {make_tensor_type({64, 64}, DType::F32),
         make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    b.output_tensor(mm);

    Schedule s_no_shared;
    s_no_shared.add({TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic});

    Schedule s_shared;
    s_shared.add({TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic});
    s_shared.add({TransformKind::Cache, "a", 0, 0, "A", MemorySpace::Shared});

    auto cost_no_shared = est.estimate(m, s_no_shared);
    auto cost_shared = est.estimate(m, s_shared);

    EXPECT_GT(cost_shared.bytes_shared, 0u);
}
