// tests/integration_tests.cpp - end-to-end integration tests
#include "cg/analysis/analysis.hpp"
#include "cg/codegen/codegen_ir.hpp"
#include "cg/cost/estimator.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/egraph/egraph.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/runtime/runtime.hpp"
#include "cg/schedule/schedule.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(Integration, FullPipelineBuildAndOptimize) {
    Module m;
    auto f = m.create_function("mlp_block",
        {make_tensor_type({128, 256}, DType::F32),
         make_tensor_type({256, 512}, DType::F32),
         make_tensor_type({128, 512}, DType::F32)},
        {make_tensor_type({128, 512}, DType::F32)});
    Builder b(f);
    auto A = f->args()[0];
    auto B = f->args()[1];
    auto bias = f->args()[2];
    auto mm = b.matmul(A, B);
    auto bd = b.add(mm, bias);     // broadcast add
    auto r  = b.relu(bd);
    b.output_tensor(r);

    AnalysisManager am(m);
    PassManager pm;
    pm.add(std::make_unique<CanonicalizePass>());
    pm.add(std::make_unique<CSEPass>());
    pm.add(std::make_unique<DCEPass>());
    pm.run(m, am);

    // The module should still print.
    auto s = to_string(m);
    EXPECT_NE(s.find("matmul"), std::string::npos);
    EXPECT_NE(s.find("relu"), std::string::npos);
}

TEST(Integration, CodegenIR) {
    CGModule cgm;
    auto& fn = cgm.create_function("simple_kernel");
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

    ASSERT_EQ(fn.instructions.size(), 5u);
    EXPECT_EQ(fn.instructions[0].opcode, CGOpcode::VectorLoad);
    EXPECT_EQ(fn.instructions[2].opcode, CGOpcode::FMA);
    EXPECT_EQ(fn.instructions[4].opcode, CGOpcode::Barrier);
}

TEST(Integration, CostPipeline) {
    Module m;
    auto f = m.create_function("cost_test",
        {make_tensor_type({64, 64}, DType::F32),
         make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    b.output_tensor(mm);

    auto hw = HardwareModel::generic_nvidia_gpu();
    CostEstimator est(hw);
    Schedule s;
    auto c = est.estimate(m, s);
    EXPECT_GT(c.flops, 0u);
    EXPECT_GT(c.bytes_global, 0u);
}

TEST(Integration, RuntimeCreation) {
    Runtime rt;
    // No devices added; get_device should return null.
    EXPECT_EQ(rt.get_device(DeviceId::cpu()), nullptr);
    EXPECT_EQ(rt.num_devices(), 0u);
}

TEST(Integration, EGraphSuperoptimize) {
    // Build: add(x, 0) -> x
    EGraph g;
    auto x = g.add({"var", {}, DType::F32, {}});
    auto zero = g.add({"const", {}, DType::F32, {}});
    auto sum = g.add({"add", {x, zero}});

    EGraph::Rewrite rw;
    rw.lhs = {"add", {0, 1}};
    rw.var_names = {"a", "b"};
    rw.rhs = [](const std::unordered_map<std::string, EClassId>& subst) {
        // Just return the first operand (x + 0 -> x).
        ENode n;
        n.op = "var";
        n.children = {subst.at("a")};
        return n;
    };
    g.saturate({rw}, 4);

    auto ext = g.extract(sum, [](const ENode& n) -> double {
        if (n.op == "var")   return 0.5;
        if (n.op == "const") return 0.1;
        if (n.op == "add")   return 2.0;
        return 1.0;
    });
    // The cheapest expression should be just "var" (cost 0.5), not "add"
    // (cost 2.6).
    EXPECT_EQ(ext.node.op, "var");
}
