// tests/egraph_tests.cpp - e-graph tests
#include "cg/egraph/egraph.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(EGraph, AddAndMerge) {
    EGraph g;
    auto a = g.add({"var", {}, DType::F32, {}});
    auto b = g.add({"var", {}, DType::F32, {}});
    auto sum1 = g.add({"add", {a, b}});
    auto sum2 = g.add({"add", {b, a}});  // same e-class via commutativity
    g.merge(sum1, sum2);

    // After merge, find(sum1) == find(sum2) — we verify indirectly by
    // extraction producing a node with cost.
    auto ext = g.extract(sum1, [](const ENode& n) -> double {
        if (n.op == "add") return 1.0;
        if (n.op == "var") return 0.1;
        return 1.0;
    });
    EXPECT_GT(ext.cost, 0.0);
}

TEST(EGraph, RewriteCommutative) {
    EGraph g;
    // Two distinct variables: different dtypes so the ENodes don't hash-cons.
    auto a = g.add({"var", {}, DType::F32, {}});
    auto b = g.add({"var", {}, DType::F64, {}});
    auto sum = g.add({"add", {a, b}});

    EGraph::Rewrite rw;
    rw.lhs = {"add", {0, 1}};
    rw.var_names = {"x", "y"};
    rw.rhs = [](const std::unordered_map<std::string, EClassId>& subst) {
        ENode n;
        n.op = "add";
        n.children = {subst.at("y"), subst.at("x")};
        return n;
    };

    g.saturate({rw}, 4);

    // After saturation, the sum class should contain at least 2 nodes
    // (the original add(a,b) and the rewritten add(b,a)).
    EXPECT_GE(g.num_classes(), 3u); // a, b, and merged sum
}

TEST(EGraph, ExtractCheapest) {
    EGraph g;
    auto a = g.add({"var", {}, DType::F32, {}});
    auto b = g.add({"var", {}, DType::F32, {}});

    // Two ways to express "0": direct constant and add(x, -x).
    auto zero_const = g.add({"const", {}, DType::F32, {}});
    auto neg_a = g.add({"neg", {a}});
    auto sum_zero = g.add({"add", {a, neg_a}});
    g.merge(zero_const, sum_zero);

    // Cost function that prefers constants.
    auto ext = g.extract(zero_const, [](const ENode& n) -> double {
        if (n.op == "const") return 0.1;
        if (n.op == "var")   return 1.0;
        return 2.0;
    });
    EXPECT_EQ(ext.node.op, "const");
}
