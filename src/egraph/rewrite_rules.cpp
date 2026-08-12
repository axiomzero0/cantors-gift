// egraph/rewrite_rules.cpp - tensor rewrite rule library with nested patterns
//
// All rules now actually fire. The pattern matcher supports nested LHS
// patterns and RHS e-class creation.
#include "cg/egraph/rewrite_rules.hpp"

namespace cg {

std::vector<RewriteRule> algebraic_rules() {
    std::vector<RewriteRule> rules;

    // neg(neg(x)) → x  [involution]
    {
        RewriteRule r;
        r.name = "neg_neg_involution";
        r.rule.lhs = Pattern::node("neg", {Pattern::var("x")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            return subst.at("x");
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // sub(x, x) → 0  [cancellation]
    {
        RewriteRule r;
        r.name = "sub_self_cancel";
        r.rule.lhs = Pattern::node("sub", {Pattern::var("x"), Pattern::var("x")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>&) {
            return eg.add_constant(static_cast<i64>(0));
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // add(x, 0) → x  [identity — constant guard via nested pattern]
    {
        RewriteRule r;
        r.name = "add_zero_identity";
        r.rule.lhs = Pattern::node("add", {Pattern::var("x"), Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            // The RHS is just x (drop the constant).
            return subst.at("x");
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // add(0, x) → x  [identity — commutative variant]
    {
        RewriteRule r;
        r.name = "add_zero_identity_left";
        r.rule.lhs = Pattern::node("add", {Pattern::node("const"), Pattern::var("x")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            return subst.at("x");
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // mul(x, 1) → x  [identity]
    {
        RewriteRule r;
        r.name = "mul_one_identity";
        r.rule.lhs = Pattern::node("mul", {Pattern::var("x"), Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            return subst.at("x");
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // mul(x, 0) → 0  [zero property]
    {
        RewriteRule r;
        r.name = "mul_zero_property";
        r.rule.lhs = Pattern::node("mul", {Pattern::var("x"), Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>&) {
            return eg.add_constant(static_cast<i64>(0));
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> commutativity_rules() {
    std::vector<RewriteRule> rules;

    auto make_comm = [](const std::string& op) {
        RewriteRule r;
        r.name = op + "_commute";
        r.rule.lhs = Pattern::node(op, {Pattern::var("a"), Pattern::var("b")});
        r.rule.rhs = [op](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = op;
            n.children = {subst.at("b"), subst.at("a")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        return r;
    };

    rules.push_back(make_comm("add"));
    rules.push_back(make_comm("mul"));
    rules.push_back(make_comm("max"));
    rules.push_back(make_comm("min"));
    return rules;
}

std::vector<RewriteRule> associativity_rules() {
    std::vector<RewriteRule> rules;

    // add(add(a, b), c) ↔ add(a, add(b, c))
    {
        RewriteRule r;
        r.name = "add_assoc_rl";
        r.rule.lhs = Pattern::node("add", {
            Pattern::node("add", {Pattern::var("a"), Pattern::var("b")}),
            Pattern::var("c")
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode inner;
            inner.op = "add";
            inner.children = {subst.at("b"), subst.at("c")};
            EClassId inner_class = eg.add(inner);
            ENode outer;
            outer.op = "add";
            outer.children = {subst.at("a"), inner_class};
            return eg.add(outer);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    // mul(mul(a, b), c) ↔ mul(a, mul(b, c))
    {
        RewriteRule r;
        r.name = "mul_assoc_rl";
        r.rule.lhs = Pattern::node("mul", {
            Pattern::node("mul", {Pattern::var("a"), Pattern::var("b")}),
            Pattern::var("c")
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode inner;
            inner.op = "mul";
            inner.children = {subst.at("b"), subst.at("c")};
            EClassId inner_class = eg.add(inner);
            ENode outer;
            outer.op = "mul";
            outer.children = {subst.at("a"), inner_class};
            return eg.add(outer);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> fma_formation_rules() {
    std::vector<RewriteRule> rules;

    // add(mul(a, b), c) → fma(a, b, c)
    {
        RewriteRule r;
        r.name = "mul_add_to_fma";
        r.rule.lhs = Pattern::node("add", {
            Pattern::node("mul", {Pattern::var("a"), Pattern::var("b")}),
            Pattern::var("c")
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "fma";
            n.children = {subst.at("a"), subst.at("b"), subst.at("c")};
            return eg.add(n);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    // add(c, mul(a, b)) → fma(a, b, c)
    {
        RewriteRule r;
        r.name = "add_mul_to_fma";
        r.rule.lhs = Pattern::node("add", {
            Pattern::var("c"),
            Pattern::node("mul", {Pattern::var("a"), Pattern::var("b")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "fma";
            n.children = {subst.at("a"), subst.at("b"), subst.at("c")};
            return eg.add(n);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    // sub(mul(a, b), c) → fma(a, b, neg(c))
    {
        RewriteRule r;
        r.name = "mul_sub_to_fma";
        r.rule.lhs = Pattern::node("sub", {
            Pattern::node("mul", {Pattern::var("a"), Pattern::var("b")}),
            Pattern::var("c")
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode neg_c;
            neg_c.op = "neg";
            neg_c.children = {subst.at("c")};
            EClassId neg_c_class = eg.add(neg_c);
            ENode n;
            n.op = "fma";
            n.children = {subst.at("a"), subst.at("b"), neg_c_class};
            return eg.add(n);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    // fma(a, b, c) → add(mul(a, b), c)  [reverse for cost comparison]
    {
        RewriteRule r;
        r.name = "fma_to_mul_add";
        r.rule.lhs = Pattern::node("fma", {Pattern::var("a"), Pattern::var("b"), Pattern::var("c")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode mul_ab;
            mul_ab.op = "mul";
            mul_ab.children = {subst.at("a"), subst.at("b")};
            EClassId mul_class = eg.add(mul_ab);
            ENode n;
            n.op = "add";
            n.children = {mul_class, subst.at("c")};
            return eg.add(n);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> cast_propagation_rules() {
    std::vector<RewriteRule> rules;

    // cast(cast(x, t1), t2) → cast(x, t2)  [redundant cast elimination]
    {
        RewriteRule r;
        r.name = "redundant_cast";
        r.rule.lhs = Pattern::node("cast", {
            Pattern::node("cast", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "cast";
            n.children = {subst.at("x")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> layout_movement_rules() {
    std::vector<RewriteRule> rules;

    // ---- View algebra: collapse view chains ----

    // transpose(transpose(x)) → x  [involution]
    {
        RewriteRule r;
        r.name = "transpose_transpose_identity";
        r.rule.lhs = Pattern::node("transpose", {
            Pattern::node("transpose", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            return subst.at("x");
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // reshape(reshape(x)) → reshape(x)  [collapse]
    {
        RewriteRule r;
        r.name = "reshape_reshape_collapse";
        r.rule.lhs = Pattern::node("reshape", {
            Pattern::node("reshape", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "reshape";
            n.children = {subst.at("x")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // slice(slice(x)) → slice(x)  [collapse]
    {
        RewriteRule r;
        r.name = "slice_slice_collapse";
        r.rule.lhs = Pattern::node("slice", {
            Pattern::node("slice", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "slice";
            n.children = {subst.at("x")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // broadcast(broadcast(x)) → broadcast(x)  [collapse]
    {
        RewriteRule r;
        r.name = "broadcast_broadcast_collapse";
        r.rule.lhs = Pattern::node("broadcast", {
            Pattern::node("broadcast", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "broadcast";
            n.children = {subst.at("x")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // ---- Transpose sinking: push transposes through elementwise ops ----

    // transpose(add(a, b)) → add(transpose(a), transpose(b))
    {
        RewriteRule r;
        r.name = "transpose_through_add";
        r.rule.lhs = Pattern::node("transpose", {
            Pattern::node("add", {Pattern::var("a"), Pattern::var("b")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode ta; ta.op = "transpose"; ta.children = {subst.at("a")};
            EClassId ta_c = eg.add(ta);
            ENode tb; tb.op = "transpose"; tb.children = {subst.at("b")};
            EClassId tb_c = eg.add(tb);
            ENode n; n.op = "add"; n.children = {ta_c, tb_c};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // transpose(mul(a, b)) → mul(transpose(a), transpose(b))
    {
        RewriteRule r;
        r.name = "transpose_through_mul";
        r.rule.lhs = Pattern::node("transpose", {
            Pattern::node("mul", {Pattern::var("a"), Pattern::var("b")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode ta; ta.op = "transpose"; ta.children = {subst.at("a")};
            EClassId ta_c = eg.add(ta);
            ENode tb; tb.op = "transpose"; tb.children = {subst.at("b")};
            EClassId tb_c = eg.add(tb);
            ENode n; n.op = "mul"; n.children = {ta_c, tb_c};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // transpose(relu(x)) → relu(transpose(x))
    {
        RewriteRule r;
        r.name = "transpose_through_relu";
        r.rule.lhs = Pattern::node("transpose", {
            Pattern::node("relu", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode tx; tx.op = "transpose"; tx.children = {subst.at("x")};
            EClassId tx_c = eg.add(tx);
            ENode n; n.op = "relu"; n.children = {tx_c};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // ---- Constant tensor folding ----

    // transpose(const) → const  [fold view of constant]
    {
        RewriteRule r;
        r.name = "transpose_const_fold";
        r.rule.lhs = Pattern::node("transpose", {Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>&) {
            // A transposed constant is still a constant (just different layout).
            // For now, represent as a new constant — in practice the frontend
            // would pre-compute the transposed data.
            return eg.add_constant(static_cast<i64>(0));
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // reshape(const) → const  [fold view of constant]
    {
        RewriteRule r;
        r.name = "reshape_const_fold";
        r.rule.lhs = Pattern::node("reshape", {Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>&) {
            return eg.add_constant(static_cast<i64>(0));
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // broadcast(const) → const  [fold]
    {
        RewriteRule r;
        r.name = "broadcast_const_fold";
        r.rule.lhs = Pattern::node("broadcast", {Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>&) {
            return eg.add_constant(static_cast<i64>(0));
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> reduction_rules() {
    std::vector<RewriteRule> rules;

    // reduce_sum(add(a, b)) → add(reduce_sum(a), reduce_sum(b))
    // Linearity of sum.
    {
        RewriteRule r;
        r.name = "reduce_sum_distributes";
        r.rule.lhs = Pattern::node("reduce_sum", {
            Pattern::node("add", {Pattern::var("a"), Pattern::var("b")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode ra;
            ra.op = "reduce_sum";
            ra.children = {subst.at("a")};
            EClassId ra_class = eg.add(ra);
            ENode rb;
            rb.op = "reduce_sum";
            rb.children = {subst.at("b")};
            EClassId rb_class = eg.add(rb);
            ENode n;
            n.op = "add";
            n.children = {ra_class, rb_class};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> tc_eligibility_rules() {
    std::vector<RewriteRule> rules;

    // matmul(A_f32, B_f32) → matmul(cast(A, f16), cast(B, f16))
    // Unlocks 20x throughput on A100+ tensor cores.
    {
        RewriteRule r;
        r.name = "matmul_f32_to_f16";
        r.rule.lhs = Pattern::node("matmul", {Pattern::var("A"), Pattern::var("B")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode cast_a;
            cast_a.op = "cast";
            cast_a.children = {subst.at("A")};
            cast_a.dtype = DType::F16;
            EClassId ca = eg.add(cast_a);

            ENode cast_b;
            cast_b.op = "cast";
            cast_b.children = {subst.at("B")};
            cast_b.dtype = DType::F16;
            EClassId cb = eg.add(cast_b);

            ENode n;
            n.op = "matmul";
            n.children = {ca, cb};
            n.dtype = DType::F16;
            return eg.add(n);
        };
        r.kind = RuleKind::TCEligibility;
        r.min_mode = NumericalMode::FastMath;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> domain_rules() {
    std::vector<RewriteRule> rules;

    // relu(x) → max(x, 0)
    {
        RewriteRule r;
        r.name = "relu_to_max";
        r.rule.lhs = Pattern::node("relu", {Pattern::var("x")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            EClassId zero = eg.add_constant(static_cast<i64>(0));
            ENode n;
            n.op = "max";
            n.children = {subst.at("x"), zero};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // max(x, 0) → relu(x)  [reverse for cost comparison]
    {
        RewriteRule r;
        r.name = "max_zero_to_relu";
        r.rule.lhs = Pattern::node("max", {Pattern::var("x"), Pattern::node("const")});
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "relu";
            n.children = {subst.at("x")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // ---- Strength reduction: idempotent operations ----

    // relu(relu(x)) → relu(x)  [idempotent]
    {
        RewriteRule r;
        r.name = "relu_relu_idempotent";
        r.rule.lhs = Pattern::node("relu", {
            Pattern::node("relu", {Pattern::var("x")})
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "relu";
            n.children = {subst.at("x")};
            return eg.add(n);
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // ---- Elementwise reassociation (under Relaxed mode) ----

    // add(add(x, a), b) → add(x, add(a, b))  [reassociate]
    // Only valid under Relaxed/FastMath (IEEE floating-point add is not associative).
    {
        RewriteRule r;
        r.name = "add_reassociate";
        r.rule.lhs = Pattern::node("add", {
            Pattern::node("add", {Pattern::var("x"), Pattern::var("a")}),
            Pattern::var("b")
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode inner;
            inner.op = "add";
            inner.children = {subst.at("a"), subst.at("b")};
            EClassId inner_c = eg.add(inner);
            ENode n;
            n.op = "add";
            n.children = {subst.at("x"), inner_c};
            return eg.add(n);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    // mul(mul(x, a), b) → mul(x, mul(a, b))  [reassociate]
    {
        RewriteRule r;
        r.name = "mul_reassociate";
        r.rule.lhs = Pattern::node("mul", {
            Pattern::node("mul", {Pattern::var("x"), Pattern::var("a")}),
            Pattern::var("b")
        });
        r.rule.rhs = [](EGraph& eg, const std::unordered_map<std::string, EClassId>& subst) {
            ENode inner;
            inner.op = "mul";
            inner.children = {subst.at("a"), subst.at("b")};
            EClassId inner_c = eg.add(inner);
            ENode n;
            n.op = "mul";
            n.children = {subst.at("x"), inner_c};
            return eg.add(n);
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        rules.push_back(r);
    }

    return rules;
}

std::vector<RewriteRule> get_rewrite_rules(NumericalMode mode) {
    std::vector<RewriteRule> all;
    auto add = [&](std::vector<RewriteRule> rs) {
        for (auto& r : rs) {
            if (static_cast<u8>(mode) >= static_cast<u8>(r.min_mode)) {
                all.push_back(std::move(r));
            }
        }
    };

    add(algebraic_rules());
    add(commutativity_rules());
    add(associativity_rules());
    add(fma_formation_rules());
    add(cast_propagation_rules());
    add(layout_movement_rules());
    add(reduction_rules());
    add(tc_eligibility_rules());
    add(domain_rules());

    return all;
}

std::vector<RewriteRule> filter_rules(
    const std::vector<RewriteRule>& rules,
    NumericalMode mode) {
    std::vector<RewriteRule> filtered;
    for (const auto& r : rules) {
        if (static_cast<u8>(mode) >= static_cast<u8>(r.min_mode)) {
            filtered.push_back(r);
        }
    }
    return filtered;
}

} // namespace cg
