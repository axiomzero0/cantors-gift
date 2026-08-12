// tests/finishing/finishing_tests.cpp - tests for the finishing pieces
#include "cg/backend/amd/amd_backend.hpp"
#include "cg/backend/cpu_backend.hpp"
#include "cg/backend/jit.hpp"
#include "cg/backend/lowering.hpp"
#include "cg/backend/nvidia_backend.hpp"
#include "cg/codegen/codegen_ir.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/lowering/tensor_to_codegen.hpp"
#include "cg/numerical/semantics.hpp"
#include "cg/vendor/dispatch.hpp"

#include "cg/test/gtest_compat.hpp"

#include <vector>

using namespace cg;

// ---- Lowering: Tensor IR -> Codegen IR ----

TEST(Finishing, LoweringProducesCodegenIR) {
    Module m;
    auto f = m.create_function("test",
        {make_tensor_type({4, 4}, DType::F32),
         make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto sum = b.add(f->args()[0], f->args()[1]);
    b.output_tensor(sum);

    TensorToCodegenLowering lowering;
    CGModule cgm = lowering.lower(m);

    ASSERT_EQ(cgm.functions.size(), 1u);
    EXPECT_GT(cgm.functions[0].instructions.size(), 0u);
    EXPECT_GE(cgm.functions[0].instructions.size(), 4u);
}

TEST(Finishing, LoweringMatmul) {
    Module m;
    auto f = m.create_function("mm",
        {make_tensor_type({16, 32}, DType::F32),
         make_tensor_type({32, 64}, DType::F32)},
        {make_tensor_type({16, 64}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    b.output_tensor(mm);

    TensorToCodegenLowering lowering;
    CGModule cgm = lowering.lower(m);

    bool has_fma = false;
    for (auto& inst : cgm.functions[0].instructions) {
        if (inst.opcode == CGOpcode::FMA) has_fma = true;
    }
    EXPECT_TRUE(has_fma);
}

TEST(Finishing, LoweringElementwise) {
    Module m;
    auto f = m.create_function("ew",
        {make_tensor_type({8, 8}, DType::F32)},
        {make_tensor_type({8, 8}, DType::F32)});
    Builder b(f);
    auto r = b.relu(f->args()[0]);
    b.output_tensor(r);

    TensorToCodegenLowering lowering;
    CGModule cgm = lowering.lower(m);

    bool has_load = false, has_store = false;
    for (auto& inst : cgm.functions[0].instructions) {
        if (inst.opcode == CGOpcode::VectorLoad) has_load = true;
        if (inst.opcode == CGOpcode::VectorStore) has_store = true;
    }
    EXPECT_TRUE(has_load);
    EXPECT_TRUE(has_store);
}

// ---- JIT execution ----

TEST(Finishing, JITExecutesSimpleFunction) {
    // push rbp; mov rbp, rsp; xor rax, rax (returns 0); pop rbp; ret.
    X86Emitter e;
    e.push(X86Reg::RBP);
    e.mov_reg(X86Reg::RBP, X86Reg::RSP);
    e.xor_reg(X86Reg::RAX, X86Reg::RAX);
    e.pop(X86Reg::RBP);
    e.ret();

    JITFunction<int(*)()> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());
    int result = jit();
    EXPECT_EQ(result, 0);
}

TEST(Finishing, JITExecutesAddFunction) {
    // mov rax, 42; ret. Returns 42.
    X86Emitter e;
    e.mov_imm64(X86Reg::RAX, 42);
    e.ret();

    JITFunction<int(*)()> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());
    int result = jit();
    EXPECT_EQ(result, 42);
}

TEST(Finishing, JITExecutesArithmetic) {
    // mov rax, 10; mov rcx, 20; add rax, rcx; ret. Returns 30.
    X86Emitter e;
    e.mov_imm64(X86Reg::RAX, 10);
    e.mov_imm64(X86Reg::RCX, 20);
    e.add_reg(X86Reg::RAX, X86Reg::RCX);
    e.ret();

    JITFunction<int(*)()> jit(e.take_bytes());
    ASSERT_TRUE(jit.valid());
    int result = jit();
    EXPECT_EQ(result, 30);
}

// ---- Complete standard ops ----

TEST(Finishing, Conv2DShapeInference) {
    Module m;
    auto f = m.create_function("conv",
        {make_tensor_type({1, 3, 32, 32}, DType::F32),
         make_tensor_type({16, 3, 3, 3}, DType::F32)},
        {make_tensor_type({1, 16, 30, 30}, DType::F32)});
    Builder b(f);
    auto out = b.conv2d(f->args()[0], f->args()[1], 1, 1, 0, 0, 1, 1);
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 4u);
    EXPECT_EQ(t->shape[0]->value, 1);
    EXPECT_EQ(t->shape[1]->value, 16);
    EXPECT_EQ(t->shape[2]->value, 30);
    EXPECT_EQ(t->shape[3]->value, 30);
}

TEST(Finishing, Conv2DWithStride) {
    Module m;
    auto f = m.create_function("conv_s",
        {make_tensor_type({1, 3, 32, 32}, DType::F32),
         make_tensor_type({16, 3, 3, 3}, DType::F32)},
        {make_tensor_type({1, 16, 15, 15}, DType::F32)});
    Builder b(f);
    auto out = b.conv2d(f->args()[0], f->args()[1], 2, 2, 0, 0, 1, 1);
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape[2]->value, 15);
    EXPECT_EQ(t->shape[3]->value, 15);
}

TEST(Finishing, SoftmaxShapePreserved) {
    Module m;
    auto f = m.create_function("sm",
        {make_tensor_type({8, 16}, DType::F32)},
        {make_tensor_type({8, 16}, DType::F32)});
    Builder b(f);
    auto out = b.softmax(f->args()[0]);
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 8);
    EXPECT_EQ(t->shape[1]->value, 16);
}

TEST(Finishing, ConcatShapeInference) {
    Module m;
    auto f = m.create_function("cat",
        {make_tensor_type({4, 8}, DType::F32),
         make_tensor_type({4, 16}, DType::F32)},
        {make_tensor_type({4, 24}, DType::F32)});
    Builder b(f);
    std::vector<Value> inputs = {f->args()[0], f->args()[1]};
    auto out = b.concat(inputs, 1);
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape[1]->value, 24);
}

TEST(Finishing, SliceShapeInference) {
    Module m;
    auto f = m.create_function("sl",
        {make_tensor_type({8, 16}, DType::F32)},
        {make_tensor_type({4, 8}, DType::F32)});
    Builder b(f);
    auto out = b.slice(f->args()[0], {2, 4}, {6, 12});
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape[0]->value, 4);
    EXPECT_EQ(t->shape[1]->value, 8);
}

TEST(Finishing, GatherShapeInference) {
    Module m;
    auto f = m.create_function("g",
        {make_tensor_type({100}, DType::F32),
         make_tensor_type({10}, DType::I32)},
        {make_tensor_type({10}, DType::F32)});
    Builder b(f);
    auto out = b.gather(f->args()[0], f->args()[1]);
    b.output_tensor(out);

    auto t = out.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->shape.rank(), 1u);
    EXPECT_EQ(t->shape[0]->value, 10);
}

// ---- AMD backend ----

TEST(Finishing, AmdBackendProducesGCNText) {
    CGModule cgm;
    auto& fn = cgm.create_function("test_kernel");
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

    AmdBackend amd;
    auto exe = amd.compile(cgm);

    EXPECT_FALSE(exe->ptx_text.empty());
    EXPECT_NE(exe->ptx_text.find(".amdgpu_kernel"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find("v_fma_f32"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find("s_barrier"), std::string::npos);
    EXPECT_NE(exe->ptx_text.find("s_endpgm"), std::string::npos);
}

TEST(Finishing, AmdBackendTargetInfo) {
    AmdBackend amd("gfx90a");
    const auto& ti = amd.target_info();
    EXPECT_EQ(ti.device.kind, DeviceId::Kind::ROCM);
    EXPECT_NE(ti.name.find("gfx90a"), std::string::npos);
}

// ---- Vendor dispatch ----

TEST(Finishing, VendorDispatchFindsMatmul) {
    Module m;
    auto f = m.create_function("mm",
        {make_tensor_type({64, 64}, DType::F32),
         make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    b.output_tensor(mm);

    // Find the matmul op.
    Operation* mm_op = nullptr;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_MATMUL) { mm_op = &op; break; }
    }
    ASSERT_NE(mm_op, nullptr);
    auto* kernel = VendorDispatcher::instance().find_best(*mm_op);
    EXPECT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->vendor(), VendorKind::cuBLAS);
}

TEST(Finishing, VendorDispatchFindsConv2D) {
    Module m;
    auto f = m.create_function("conv",
        {make_tensor_type({1, 3, 32, 32}, DType::F32),
         make_tensor_type({16, 3, 3, 3}, DType::F32)},
        {make_tensor_type({1, 16, 30, 30}, DType::F32)});
    Builder b(f);
    auto conv = b.conv2d(f->args()[0], f->args()[1]);
    b.output_tensor(conv);

    Operation* conv_op = nullptr;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_CONV2D) { conv_op = &op; break; }
    }
    ASSERT_NE(conv_op, nullptr);
    auto* kernel = VendorDispatcher::instance().find_best(*conv_op);
    EXPECT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->vendor(), VendorKind::cuDNN);
}

TEST(Finishing, VendorHasMultipleKernels) {
    auto& d = VendorDispatcher::instance();
    EXPECT_GE(d.kernels().size(), 4u);
    EXPECT_TRUE(d.has_vendor(VendorKind::cuBLAS));
    EXPECT_TRUE(d.has_vendor(VendorKind::cuDNN));
    EXPECT_TRUE(d.has_vendor(VendorKind::rocBLAS));
    EXPECT_TRUE(d.has_vendor(VendorKind::oneDNN));
}

// ---- Numerical semantics ----

TEST(Finishing, NumericalModes) {
    EXPECT_FALSE(allows_reassociation(NumericalMode::Strict));
    EXPECT_TRUE(allows_reassociation(NumericalMode::Relaxed));
    EXPECT_TRUE(allows_reassociation(NumericalMode::FastMath));

    EXPECT_FALSE(allows_contraction(NumericalMode::Strict));
    EXPECT_TRUE(allows_contraction(NumericalMode::Relaxed));

    EXPECT_FALSE(assumes_no_nan(NumericalMode::Strict));
    EXPECT_FALSE(assumes_no_nan(NumericalMode::Relaxed));
    EXPECT_TRUE(assumes_no_nan(NumericalMode::FastMath));

    EXPECT_FALSE(allows_reciprocal_approx(NumericalMode::Strict));
    EXPECT_TRUE(allows_reciprocal_approx(NumericalMode::FastMath));
}

TEST(Finishing, NumericalModeNames) {
    EXPECT_EQ(numerical_mode_name(NumericalMode::Strict), "strict");
    EXPECT_EQ(numerical_mode_name(NumericalMode::Relaxed), "relaxed");
    EXPECT_EQ(numerical_mode_name(NumericalMode::FastMath), "fast_math");
}

// ---- End-to-end: Tensor IR -> optimize -> lower -> compile ----

TEST(Finishing, EndToEndPipeline) {
    Module m;
    auto f = m.create_function("mlp",
        {make_tensor_type({64, 128}, DType::F32),
         make_tensor_type({128, 256}, DType::F32),
         make_tensor_type({64, 256}, DType::F32)},
        {make_tensor_type({64, 256}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    auto bd = b.add(mm, f->args()[2]);
    auto r = b.relu(bd);
    b.output_tensor(r);

    TensorToCodegenLowering lowering;
    CGModule cgm = lowering.lower(m);
    ASSERT_EQ(cgm.functions.size(), 1u);
    EXPECT_GT(cgm.functions[0].instructions.size(), 0u);

    NvidiaBackend nv;
    auto exe_nv = nv.compile(cgm);
    EXPECT_FALSE(exe_nv->ptx_text.empty());
    EXPECT_NE(exe_nv->ptx_text.find("fma.rn"), std::string::npos);

    CpuBackend cpu;
    auto exe_cpu = cpu.compile(cgm);
    EXPECT_GT(exe_cpu->machine_code.size(), 0u);

    AmdBackend amd;
    auto exe_amd = amd.compile(cgm);
    EXPECT_FALSE(exe_amd->ptx_text.empty());
    EXPECT_NE(exe_amd->ptx_text.find("v_fma_f32"), std::string::npos);
}
