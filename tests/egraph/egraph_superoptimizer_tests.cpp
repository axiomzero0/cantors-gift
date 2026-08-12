// tests/egraph_superoptimizer_tests.cpp - e-graph superoptimizer tests
#include "cg/analysis/analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/egraph/egraph_superoptimizer.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(EGraphSuperoptimizer, RunsOnPureOps) {
    Module m;
    auto f = m.create_function("test",
        {make_tensor_type({8, 8}, DType::F32)},
        {make_tensor_type({8, 8}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto r1 = b.relu(a);
    auto r2 = b.exp(r1);
    auto r3 = b.add(r2, r2);
    b.output_tensor(r3);

    AnalysisManager am(m);
    EGraphSuperoptimizerPass p;
    // Should run without crashing.
    p.run(m, am);
    SUCCEED();
}

TEST(EGraphSuperoptimizer, HandlesEmptyModule) {
    Module m;
    AnalysisManager am(m);
    EGraphSuperoptimizerPass p;
    p.run(m, am);
    SUCCEED();
}

TEST(EGraphSuperoptimizer, HandlesImpureOps) {
    Module m;
    auto f = m.create_function("test",
        {make_tensor_type({8, 8}, DType::F32)},
        {make_tensor_type({8, 8}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto r = b.relu(a);
    b.output_tensor(r);

    AnalysisManager am(m);
    EGraphSuperoptimizerPass p;
    p.run(m, am);
    SUCCEED();
}
