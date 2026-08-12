// tests/egraph_v2/egraph_rules_tests.cpp
// Verify that rewrite rules ACTUALLY FIRE in the e-graph.
#include "cg/egraph/egraph.hpp"
#include "cg/egraph/rewrite_rules.hpp"
#include "cg/numerical/semantics.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

// Helper: build a simple e-graph with two leaf vars and one op.
struct EGraphTest {
    EGraph g;
    EClassId a, b, root;

    void build_2arg(const std::string& op) {
        a = g.add({"var", {}, DType::F32, {}});
        b = g.add({"var", {}, DType::F64, {}});
        ENode n;
        n.op = op;
        n.children = {a, b};
        root = g.add(n);
    }

    void build_1arg(const std::string& op) {
        a = g.add({"var", {}, DType::F32, {}});
        ENode n;
        n.op = op;
        n.children = {a};
        root = g.add(n);
    }

    void build_nested(const std::string& outer_op, const std::string& inner_op) {
        a = g.add({"var", {}, DType::F32, {}});
        b = g.add({"var", {}, DType::F64, {}});
        ENode inner;
        inner.op = inner_op;
        inner.children = {a, b};
        EClassId inner_class = g.add(inner);
        ENode outer;
        outer.op = outer_op;
        outer.children = {inner_class};
        // need a third arg for add — use a var
        EClassId c = g.add({"var", {}, DType::I32, {}});
        ENode outer2;
        outer2.op = outer_op;
        outer2.children = {inner_class, c};
        root = g.add(outer2);
    }

    void saturate(NumericalMode mode = NumericalMode::Relaxed, usize iters = 6) {
        auto rules = get_rewrite_rules(mode);
        std::vector<EGraph::Rewrite> egraph_rules;
        for (auto& r : rules) {
            EGraph::Rewrite er;
            er.name = r.name;
            er.lhs = r.rule.lhs;
            er.rhs = r.rule.rhs;
            egraph_rules.push_back(std::move(er));
        }
        g.saturate(egraph_rules, iters);
    }

    bool class_has_op(EClassId cid, const std::string& op) {
        auto& nodes = g.nodes_in_class(g.find(cid));
        for (auto& n : nodes) {
            if (n.op == op) return true;
        }
        return false;
    }

    usize class_size(EClassId cid) {
        return g.nodes_in_class(g.find(cid)).size();
    }
};

// ---- FMA formation: add(mul(a,b), c) → fma(a,b,c) ----
TEST(EGraphRules, FMAFormationFires) {
    EGraphTest t;
    // Build: add(mul(a, b), c)
    t.a = t.g.add({"var", {}, DType::F32, {}});
    t.b = t.g.add({"var", {}, DType::F64, {}});
    ENode mul_ab;
    mul_ab.op = "mul";
    mul_ab.children = {t.a, t.b};
    EClassId mul_class = t.g.add(mul_ab);
    EClassId c = t.g.add({"var", {}, DType::I32, {}});
    ENode add_node;
    add_node.op = "add";
    add_node.children = {mul_class, c};
    t.root = t.g.add(add_node);

    t.saturate(NumericalMode::Relaxed, 6);

    // The root class should now contain an "fma" node.
    EXPECT_TRUE(t.class_has_op(t.root, "fma"));
}

// ---- Associativity: add(add(a,b), c) → add(a, add(b,c)) ----
TEST(EGraphRules, AssociativityFires) {
    EGraphTest t;
    t.a = t.g.add({"var", {}, DType::F32, {}});
    t.b = t.g.add({"var", {}, DType::F64, {}});
    ENode inner;
    inner.op = "add";
    inner.children = {t.a, t.b};
    EClassId inner_class = t.g.add(inner);
    EClassId c = t.g.add({"var", {}, DType::I32, {}});
    ENode outer;
    outer.op = "add";
    outer.children = {inner_class, c};
    t.root = t.g.add(outer);

    t.saturate(NumericalMode::Relaxed, 6);

    // After saturation, the root class should have more than 1 node
    // (the original add(add(a,b),c) and the rewritten add(a,add(b,c))).
    EXPECT_GT(t.class_size(t.root), 1u);
}

// ---- neg(neg(x)) → x ----
TEST(EGraphRules, NegNegFires) {
    EGraphTest t;
    t.a = t.g.add({"var", {}, DType::F32, {}});
    ENode inner;
    inner.op = "neg";
    inner.children = {t.a};
    EClassId inner_class = t.g.add(inner);
    ENode outer;
    outer.op = "neg";
    outer.children = {inner_class};
    t.root = t.g.add(outer);

    t.saturate(NumericalMode::Strict, 4);

    // The root class should be merged with the original variable's class.
    EXPECT_EQ(t.g.find(t.root), t.g.find(t.a));
}

// ---- relu(x) → max(x, 0) ----
TEST(EGraphRules, ReluToMaxFires) {
    EGraphTest t;
    t.build_1arg("relu");
    t.saturate(NumericalMode::Relaxed, 4);

    // The root class should contain a "max" node.
    EXPECT_TRUE(t.class_has_op(t.root, "max"));
}

// ---- Commutativity: add(a,b) ↔ add(b,a) ----
TEST(EGraphRules, CommutativityFires) {
    EGraphTest t;
    t.build_2arg("add");
    t.saturate(NumericalMode::Relaxed, 4);

    // The root class should have at least 2 nodes (original + commuted).
    EXPECT_GE(t.class_size(t.root), 2u);
}

// ---- TC eligibility: matmul(A,B) → matmul(cast(A,f16), cast(B,f16)) ----
TEST(EGraphRules, TCEligibilityFires) {
    EGraphTest t;
    t.build_2arg("matmul");
    t.saturate(NumericalMode::FastMath, 6);

    // The root class should contain a matmul with cast children.
    // Check that "cast" appears somewhere in the e-graph.
    bool has_cast = false;
    for (EClassId cid = 0; cid < t.g.num_classes(); ++cid) {
        if (t.class_has_op(cid, "cast")) {
            has_cast = true;
            break;
        }
    }
    EXPECT_TRUE(has_cast);
}

// ---- TC eligibility does NOT fire under Strict ----
TEST(EGraphRules, TCEligibilityGatedByMode) {
    EGraphTest t;
    t.build_2arg("matmul");
    t.saturate(NumericalMode::Strict, 6);

    // No cast nodes should be created under Strict mode.
    bool has_cast = false;
    for (EClassId cid = 0; cid < t.g.num_classes(); ++cid) {
        if (t.class_has_op(cid, "cast")) {
            has_cast = true;
            break;
        }
    }
    EXPECT_FALSE(has_cast);
}

// ---- Reduction distribution: reduce_sum(add(a,b)) → add(reduce_sum(a), reduce_sum(b)) ----
TEST(EGraphRules, ReductionDistributionFires) {
    EGraphTest t;
    t.a = t.g.add({"var", {}, DType::F32, {}});
    t.b = t.g.add({"var", {}, DType::F64, {}});
    ENode add_node;
    add_node.op = "add";
    add_node.children = {t.a, t.b};
    EClassId add_class = t.g.add(add_node);
    ENode red;
    red.op = "reduce_sum";
    red.children = {add_class};
    t.root = t.g.add(red);

    t.saturate(NumericalMode::Relaxed, 6);

    // The root class should contain an "add" node (the distributed form).
    EXPECT_TRUE(t.class_has_op(t.root, "add"));
}

// ---- Rule count: verify we have a substantial number of active rules ----
TEST(EGraphRules, RuleCountSignificant) {
    auto relaxed_rules = get_rewrite_rules(NumericalMode::Relaxed);
    auto fast_rules = get_rewrite_rules(NumericalMode::FastMath);

    // Relaxed should have at least 10 active rules (algebraic + commutativity
    // + associativity + FMA + cast + layout + reduction + domain).
    EXPECT_GE(relaxed_rules.size(), 10u);

    // FastMath should have more (adds TC eligibility).
    EXPECT_GT(fast_rules.size(), relaxed_rules.size());
}

// ---- E-graph rebuild discovers congruence ----
TEST(EGraphRules, RebuildDiscoversCongruence) {
    EGraph g;
    auto a = g.add({"var", {}, DType::F32, {}});
    auto b = g.add({"var", {}, DType::F64, {}});

    // Build add(a, b) and add(b, a) separately.
    ENode ab;
    ab.op = "add";
    ab.children = {a, b};
    EClassId ab_class = g.add(ab);

    ENode ba;
    ba.op = "add";
    ba.children = {b, a};
    EClassId ba_class = g.add(ba);

    // Apply commutativity rule: add(a,b) → add(b,a)
    EGraph::Rewrite rw;
    rw.lhs = Pattern::node("add", {Pattern::var("x"), Pattern::var("y")});
    rw.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
        ENode n;
        n.op = "add";
        n.children = {subst.at("y"), subst.at("x")};
        return eg.add(n);
    };

    g.saturate({rw}, 4);

    // After saturation + rebuild, ab and ba should be in the same e-class.
    EXPECT_EQ(g.find(ab_class), g.find(ba_class));
}
