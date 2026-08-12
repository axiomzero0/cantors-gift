// egraph/rewrite_rules.cpp - tensor rewrite rule library implementation
#include "cg/egraph/rewrite_rules.hpp"

namespace cg {

std::vector<RewriteRule> algebraic_rules() {
    std::vector<RewriteRule> rules;

    // NOTE: add(x, 0) → x and mul(x, 1) → x require constant-value guards
    // that the e-graph pattern matcher doesn't currently support. These
    // are handled by the canonicalization pass instead. The e-graph only
    // gets rules that are valid for ALL operands.

    // neg(neg(x)) → x  [involution]
    {
        RewriteRule r;
        r.name = "neg_neg_involution";
        r.rule.lhs = {"neg", {0}};
        r.rule.var_names = {"x"};
        r.rule.rhs = [](const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = "var";
            n.children = {subst.at("x")};
            return n;
        };
        r.kind = RuleKind::Pure;
        rules.push_back(r);
    }

    // sub(x, x) → 0  [cancellation — requires both children to be the same e-class]
    // This fires automatically when the e-graph merges the two children.
    {
        RewriteRule r;
        r.name = "sub_self_cancel";
        r.rule.lhs = {"sub", {0, 0}};  // both children are the same variable
        r.rule.var_names = {"x"};
        r.rule.rhs = [](const std::unordered_map<std::string, EClassId>&) {
            ENode n;
            n.op = "const";
            n.dtype = DType::F32;
            return n;
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
        r.rule.lhs = {op, {0, 1}};
        r.rule.var_names = {"a", "b"};
        r.rule.rhs = [op](const std::unordered_map<std::string, EClassId>& subst) {
            ENode n;
            n.op = op;
            n.children = {subst.at("b"), subst.at("a")};
            return n;
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
    // This is accuracy-risky under IEEE 754 (floating-point add is not associative).
    {
        RewriteRule r;
        r.name = "add_assoc_lr";
        r.rule.lhs = {"add", {0, 1}};
        r.rule.var_names = {"ab", "c"};
        r.rule.rhs = [](const std::unordered_map<std::string, EClassId>&) {
            // We can't destructure `ab` into `a` and `b` in the rhs without
            // nested pattern matching. This rule would require the e-graph
            // to support multi-level patterns. For now, we skip it.
            ENode n;
            n.op = "add";
            return n;
        };
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        // Skip — needs nested patterns.
    }

    // mul(mul(a, b), c) ↔ mul(a, mul(b, c))
    {
        RewriteRule r;
        r.name = "mul_assoc_lr";
        r.rule.lhs = {"mul", {0, 1}};
        r.rule.var_names = {"ab", "c"};
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        // Skip — needs nested patterns.
    }

    return rules;
}

std::vector<RewriteRule> fma_formation_rules() {
    std::vector<RewriteRule> rules;

    // add(mul(a, b), c) → fma(a, b, c)
    // This is accuracy-risky: FMA computes a*b+c with a single rounding,
    // while separate mul+add has two roundings. The results differ.
    {
        RewriteRule r;
        r.name = "mul_add_to_fma";
        r.rule.lhs = {"add", {0, 1}};
        r.rule.var_names = {"mul_ab", "c"};
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        // Skip — needs nested patterns to match mul(a,b) as the first child.
    }

    // add(c, mul(a, b)) → fma(a, b, c)
    {
        RewriteRule r;
        r.name = "add_mul_to_fma";
        r.rule.lhs = {"add", {0, 1}};
        r.rule.var_names = {"c", "mul_ab"};
        r.kind = RuleKind::AccuracyRisky;
        r.min_mode = NumericalMode::Relaxed;
        // Skip — needs nested patterns.
    }

    return rules;
}

std::vector<RewriteRule> cast_propagation_rules() {
    std::vector<RewriteRule> rules;

    // cast(cast(x, t1), t2) → cast(x, t2)  [redundant cast elimination]
    {
        RewriteRule r;
        r.name = "redundant_cast";
        r.rule.lhs = {"cast", {0}};
        r.rule.var_names = {"inner_cast"};
        r.kind = RuleKind::Pure;
        // Skip — needs nested patterns.
    }

    return rules;
}

std::vector<RewriteRule> layout_movement_rules() {
    std::vector<RewriteRule> rules;

    // transpose(transpose(x, p), inv(p)) → x
    {
        RewriteRule r;
        r.name = "transpose_transpose_identity";
        r.rule.lhs = {"transpose", {0}};
        r.rule.var_names = {"inner_t"};
        r.kind = RuleKind::Pure;
        // Skip — needs nested patterns + permutation analysis.
    }

    // reshape(reshape(x, s1), s2) → reshape(x, s2)
    {
        RewriteRule r;
        r.name = "reshape_reshape";
        r.rule.lhs = {"reshape", {0}};
        r.rule.var_names = {"inner_r"};
        r.kind = RuleKind::Pure;
        // Skip — needs nested patterns.
    }

    return rules;
}

std::vector<RewriteRule> reduction_rules() {
    std::vector<RewriteRule> rules;

    // reduce_sum(add(a, b), axis) → add(reduce_sum(a, axis), reduce_sum(b, axis))
    // This is always valid for sum, but NOT for max/min/mean.
    {
        RewriteRule r;
        r.name = "reduce_sum_distributes_over_add";
        r.rule.lhs = {"reduce_sum", {0}};
        r.rule.var_names = {"add_ab"};
        r.kind = RuleKind::Pure;
        // Skip — needs nested patterns.
    }

    return rules;
}

std::vector<RewriteRule> tc_eligibility_rules() {
    std::vector<RewriteRule> rules;

    // matmul(A_f32, B_f32) → matmul(cast(A, f16), cast(B, f16))
    // This unlocks 20x throughput on A100+ tensor cores.
    // It's a precision tradeoff: F16 has less range/precision than F32.
    // Only valid under FastMath or with explicit accuracy budget.
    {
        RewriteRule r;
        r.name = "matmul_f32_to_f16";
        r.rule.lhs = {"matmul", {0, 1}};
        r.rule.var_names = {"A", "B"};
        r.rule.rhs = [](const std::unordered_map<std::string, EClassId>& subst) {
            // This would create: matmul(cast(A, f16), cast(B, f16))
            // But we can't create new e-classes in the rhs without
            // modifying the e-graph. This requires a more sophisticated
            // rewrite system. For now, skip.
            ENode n;
            n.op = "matmul";
            n.children = {subst.at("A"), subst.at("B")};
            return n;
        };
        r.kind = RuleKind::TCEligibility;
        r.min_mode = NumericalMode::FastMath;
        // Skip — needs e-class creation in rhs.
    }

    return rules;
}

std::vector<RewriteRule> domain_rules() {
    std::vector<RewriteRule> rules;

    // relu(x) → max(x, 0)
    {
        RewriteRule r;
        r.name = "relu_to_max";
        r.rule.lhs = {"relu", {0}};
        r.rule.var_names = {"x"};
        r.rule.rhs = [](const std::unordered_map<std::string, EClassId>& subst) {
            ENode zero;
            zero.op = "const";
            zero.dtype = DType::F32;
            ENode n;
            n.op = "max";
            n.children = {subst.at("x"), 0}; // placeholder
            return n;
        };
        r.kind = RuleKind::Pure;
        // Skip — needs e-class creation for the constant.
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
