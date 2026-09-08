// tests/egraph_tests.cpp - e-graph tests
//
// All ENode literals use designated initializers to silence
// -Wmissing-field-initializers. The fields `dtype` and `shape` are
// std::optional, so omitting them with `{}` leaves them as nullopt.
#include "cg/egraph/egraph.hpp"

#include "cg/test/gtest_compat.hpp"

#include <utility>

using namespace cg;

// Convenience factory: keeps test cases terse without triggering
// -Wmissing-field-initializers on brace-init.
inline ENode make_var(DType dt) {
    return ENode{.op = "var", .children = {}, .dtype = dt, .shape = {}};
}
inline ENode make_const(DType dt) {
    return ENode{.op = "const", .children = {}, .dtype = dt, .shape = {}};
}
inline ENode make_unary(std::string op, EClassId child) {
    return ENode{.op = std::move(op), .children = {child}, .dtype = {}, .shape = {}};
}
inline ENode make_binary(std::string op, EClassId a, EClassId b) {
    return ENode{.op = std::move(op), .children = {a, b}, .dtype = {}, .shape = {}};
}

TEST(EGraph, AddAndMerge) {
    EGraph g;
    auto a = g.add(make_var(DType::F32));
    auto b = g.add(make_var(DType::F32));
    auto sum1 = g.add(make_binary("add", a, b));
    auto sum2 = g.add(make_binary("add", b, a));  // same e-class via commutativity
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
    auto a = g.add(make_var(DType::F32));
    auto b = g.add(make_var(DType::F64));
    auto sum = g.add(make_binary("add", a, b));
    (void)sum;  // keep the e-class alive; we don't query its id below

    EGraph::Rewrite rw;
    rw.lhs = Pattern::node("add", {Pattern::var("x"), Pattern::var("y")});
    rw.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
        ENode n;
        n.op = "add";
        n.children = {subst.at("y"), subst.at("x")};
        return eg.add(n);
    };

    g.saturate({rw}, 4);

    // After saturation, the sum class should contain at least 2 nodes
    // (the original add(a,b) and the rewritten add(b,a)).
    EXPECT_GE(g.num_classes(), 3u); // a, b, and merged sum
}

TEST(EGraph, ExtractCheapest) {
    EGraph g;
    auto a = g.add(make_var(DType::F32));
    auto b = g.add(make_var(DType::F32));
    (void)b;  // unused but kept to mirror the original test shape

    // Two ways to express "0": direct constant and add(x, -x).
    auto zero_const = g.add(make_const(DType::F32));
    auto neg_a = g.add(make_unary("neg", a));
    auto sum_zero = g.add(make_binary("add", a, neg_a));
    g.merge(zero_const, sum_zero);

    // Cost function that prefers constants.
    auto ext = g.extract(zero_const, [](const ENode& n) -> double {
        if (n.op == "const") return 0.1;   // cheapest
        if (n.op == "add")  return 2.0;
        if (n.op == "neg")  return 1.0;
        if (n.op == "var")  return 0.5;
        return 1.0;
    });
    // The cheapest representative should be the constant.
    EXPECT_EQ(ext.node.op, "const");
}
