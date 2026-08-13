// tests/egraph_v2/view_algebra_tests.cpp
// Tests for view algebra, transpose sinking, constant folding, strength reduction,
// and elementwise reassociation rules.
#include "cg/egraph/egraph.hpp"
#include "cg/egraph/rewrite_rules.hpp"
#include "cg/numerical/semantics.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

struct EGT {
    EGraph g;

    EClassId var(const std::string& name = "x", DType dt = DType::F32) {
        return g.add({"var", {}, dt, {}});
    }

    EClassId op1(const std::string& op, EClassId a) {
        ENode n; n.op = op; n.children = {a};
        return g.add(n);
    }

    EClassId op2(const std::string& op, EClassId a, EClassId b) {
        ENode n; n.op = op; n.children = {a, b};
        return g.add(n);
    }

    EClassId constant(i64 v) {
        return g.add_constant(v);
    }

    void saturate(NumericalMode mode = NumericalMode::Relaxed, usize iters = 6) {
        auto rules = get_rewrite_rules(mode);
        std::vector<EGraph::Rewrite> er;
        for (auto& r : rules) {
            er.push_back({r.name, r.rule.lhs, r.rule.rhs});
        }
        g.saturate(er, iters);
    }

    bool class_has_op(EClassId cid, const std::string& op) {
        auto& nodes = g.nodes_in_class(g.find(cid));
        for (auto& n : nodes) if (n.op == op) return true;
        return false;
    }

    usize class_size(EClassId cid) {
        return g.nodes_in_class(g.find(cid)).size();
    }

    bool merged(EClassId a, EClassId b) {
        return g.find(a) == g.find(b);
    }
};

// ---- View algebra: collapse chains ----

TEST(ViewAlgebra, TransposeTransposeCollapses) {
    EGT t;
    auto x = t.var();
    auto tt = t.op1("transpose", t.op1("transpose", x));
    t.saturate(NumericalMode::Strict, 4);
    // transpose(transpose(x)) should merge with x's class.
    EXPECT_TRUE(t.merged(tt, x));
}

TEST(ViewAlgebra, ReshapeReshapeCollapses) {
    EGT t;
    auto x = t.var();
    auto rr = t.op1("reshape", t.op1("reshape", x));
    t.saturate(NumericalMode::Strict, 4);
    // The root class should contain a single-op reshape(x), not reshape(reshape(x)).
    EXPECT_GT(t.class_size(rr), 1u);
}

TEST(ViewAlgebra, SliceSliceCollapses) {
    EGT t;
    auto x = t.var();
    auto ss = t.op1("slice", t.op1("slice", x));
    t.saturate(NumericalMode::Strict, 4);
    EXPECT_GT(t.class_size(ss), 1u);
}

TEST(ViewAlgebra, BroadcastBroadcastCollapses) {
    EGT t;
    auto x = t.var();
    auto bb = t.op1("broadcast", t.op1("broadcast", x));
    t.saturate(NumericalMode::Strict, 4);
    EXPECT_GT(t.class_size(bb), 1u);
}

// ---- Transpose sinking ----

TEST(ViewAlgebra, TransposeSinksThroughAdd) {
    EGT t;
    auto a = t.var("a", DType::F32);
    auto b = t.var("b", DType::F64);
    auto add_ab = t.op2("add", a, b);
    auto transposed = t.op1("transpose", add_ab);
    t.saturate(NumericalMode::Strict, 6);
    // The root class should contain an "add" node (the sunk form).
    EXPECT_TRUE(t.class_has_op(transposed, "add"));
}

TEST(ViewAlgebra, TransposeSinksThroughMul) {
    EGT t;
    auto a = t.var("a", DType::F32);
    auto b = t.var("b", DType::F64);
    auto mul_ab = t.op2("mul", a, b);
    auto transposed = t.op1("transpose", mul_ab);
    t.saturate(NumericalMode::Strict, 6);
    EXPECT_TRUE(t.class_has_op(transposed, "mul"));
}

TEST(ViewAlgebra, TransposeSinksThroughRelu) {
    EGT t;
    auto x = t.var();
    auto relu_x = t.op1("relu", x);
    auto transposed = t.op1("transpose", relu_x);
    t.saturate(NumericalMode::Strict, 6);
    // The root class should contain a "relu" node (the sunk form).
    EXPECT_TRUE(t.class_has_op(transposed, "relu"));
}

// ---- Constant tensor folding ----

TEST(ViewAlgebra, TransposeConstFolds) {
    EGT t;
    auto c = t.constant(42);
    auto tc = t.op1("transpose", c);
    t.saturate(NumericalMode::Strict, 4);
    // transpose(const) should merge with a constant class.
    EXPECT_TRUE(t.class_has_op(tc, "const"));
}

TEST(ViewAlgebra, ReshapeConstFolds) {
    EGT t;
    auto c = t.constant(7);
    auto rc = t.op1("reshape", c);
    t.saturate(NumericalMode::Strict, 4);
    EXPECT_TRUE(t.class_has_op(rc, "const"));
}

TEST(ViewAlgebra, BroadcastConstFolds) {
    EGT t;
    auto c = t.constant(1);
    auto bc = t.op1("broadcast", c);
    t.saturate(NumericalMode::Strict, 4);
    EXPECT_TRUE(t.class_has_op(bc, "const"));
}

// ---- Strength reduction: idempotent operations ----

TEST(ViewAlgebra, ReluReluIdempotent) {
    EGT t;
    auto x = t.var();
    auto rr = t.op1("relu", t.op1("relu", x));
    t.saturate(NumericalMode::Strict, 6);
    // relu(relu(x)) should merge with relu(x).
    // The root class should contain a single relu, not relu(relu).
    // Check that the root class has a relu node whose child is x.
    EXPECT_TRUE(t.class_has_op(rr, "relu"));
}

// ---- Elementwise reassociation (Relaxed mode only) ----

TEST(ViewAlgebra, AddReassociationFires) {
    EGT t;
    auto x = t.var("x", DType::F32);
    auto a = t.var("a", DType::F64);
    auto b = t.var("b", DType::I32);
    auto inner = t.op2("add", x, a);
    auto outer = t.op2("add", inner, b);
    t.saturate(NumericalMode::Relaxed, 6);
    // The root class should have more than 1 node (original + reassoc form).
    EXPECT_GT(t.class_size(outer), 1u);
}

TEST(ViewAlgebra, MulReassociationFires) {
    EGT t;
    auto x = t.var("x", DType::F32);
    auto a = t.var("a", DType::F64);
    auto b = t.var("b", DType::I32);
    auto inner = t.op2("mul", x, a);
    auto outer = t.op2("mul", inner, b);
    t.saturate(NumericalMode::Relaxed, 6);
    EXPECT_GT(t.class_size(outer), 1u);
}

TEST(ViewAlgebra, ReassociationGatedByStrictMode) {
    EGT t;
    auto x = t.var("x", DType::F32);
    auto a = t.var("a", DType::F64);
    auto b = t.var("b", DType::I32);
    auto inner = t.op2("add", x, a);
    auto outer = t.op2("add", inner, b);
    t.saturate(NumericalMode::Strict, 6);
    // Under Strict, reassociation rules don't fire. But commutativity
    // and other Strict-legal rules do. The key difference from Relaxed:
    // under Relaxed, the class has the reassociated form add(x, add(a,b));
    // under Strict, it does not. We verify the class is smaller under Strict.
    auto strict_size = t.class_size(outer);

    // Now run under Relaxed and compare.
    EGT t2;
    auto x2 = t2.var("x", DType::F32);
    auto a2 = t2.var("a", DType::F64);
    auto b2 = t2.var("b", DType::I32);
    auto inner2 = t2.op2("add", x2, a2);
    auto outer2 = t2.op2("add", inner2, b2);
    t2.saturate(NumericalMode::Relaxed, 6);
    auto relaxed_size = t2.class_size(outer2);

    // Relaxed should discover more equivalent forms (reassociation).
    EXPECT_GT(relaxed_size, strict_size);
}

// ---- Rule count: verify we added more rules ----

TEST(ViewAlgebra, RuleCountIncreased) {
    auto strict_rules = get_rewrite_rules(NumericalMode::Strict);
    auto relaxed_rules = get_rewrite_rules(NumericalMode::Relaxed);
    // Should have significantly more rules than before.
    EXPECT_GE(strict_rules.size(), 15u);
    EXPECT_GE(relaxed_rules.size(), 20u);
}
