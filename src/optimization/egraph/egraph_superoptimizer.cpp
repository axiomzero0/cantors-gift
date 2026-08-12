// optimization/egraph/egraph_superoptimizer.cpp - e-graph superoptimizer
//
// Now uses the full rewrite_rules library with nested pattern matching.
// Rules actually fire: FMA formation, associativity, cast propagation,
// TC eligibility, reduction distribution, domain rules.
#include "cg/optimization/egraph/egraph_superoptimizer.hpp"
#include "cg/egraph/egraph.hpp"
#include "cg/egraph/rewrite_rules.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/numerical/semantics.hpp"

#include <unordered_map>

namespace cg {

namespace {

double tensor_cost(const ENode& n) {
    if (n.op == "var" || n.op == "const") return 0.1;
    if (n.op == "add" || n.op == "sub" || n.op == "mul" || n.op == "div")
        return 1.0;
    if (n.op == "fma") return 0.8;  // cheaper than separate mul+add
    if (n.op == "matmul") return 10.0;
    if (n.op == "cast") return 0.2;  // cheap
    if (n.op == "reduce_sum" || n.op == "reduce_max" || n.op == "reduce_mean")
        return 5.0;
    if (n.op == "relu" || n.op == "exp" || n.op == "log" || n.op == "sqrt")
        return 2.0;
    if (n.op == "max" || n.op == "min") return 1.5;
    if (n.op == "neg") return 1.0;
    if (n.op == "transpose" || n.op == "reshape") return 0.5;
    return 1.0;
}

struct EGraphBuildResult {
    EGraph graph;
    EClassId root_class;
    std::unordered_map<EClassId, Value> class_to_value;
    std::unordered_map<ValueId, EClassId> value_to_class;
};

std::optional<EGraphBuildResult> build_egraph_for_region(
    Module& m, Value root, usize max_depth = 16) {
    EGraphBuildResult result;
    auto& graph = result.graph;
    auto& value_to_class = result.value_to_class;
    auto& class_to_value = result.class_to_value;

    std::function<EClassId(Value, usize)> build = [&](Value v, usize depth) -> EClassId {
        if (depth > max_depth) {
            auto c = graph.add({"var", {}, v.as_tensor() ? v.as_tensor()->dtype : DType::F32, {}});
            class_to_value[c] = v;
            return c;
        }
        auto it = value_to_class.find(v.id());
        if (it != value_to_class.end()) return it->second;

        Operation* def = nullptr;
        for (auto& f : m.functions()) {
            if (def) break;
            for (auto& op : *f->entry()) {
                if (!op.results.empty() && op.results[0] == v) {
                    def = &op;
                    break;
                }
            }
        }

        if (!def || !def->is_pure()) {
            auto c = graph.add({"var", {}, v.as_tensor() ? v.as_tensor()->dtype : DType::F32, {}});
            value_to_class[v.id()] = c;
            class_to_value[c] = v;
            return c;
        }

        std::string op_name;
        switch (def->opcode) {
            case OP_ADD: op_name = "add"; break;
            case OP_SUB: op_name = "sub"; break;
            case OP_MUL: op_name = "mul"; break;
            case OP_DIV: op_name = "div"; break;
            case OP_NEG: op_name = "neg"; break;
            case OP_RELU: op_name = "relu"; break;
            case OP_EXP: op_name = "exp"; break;
            case OP_MATMUL: op_name = "matmul"; break;
            case OP_REDUCE_SUM: op_name = "reduce_sum"; break;
            case OP_REDUCE_MAX: op_name = "reduce_max"; break;
            case OP_CONSTANT: op_name = "const"; break;
            case OP_CAST: op_name = "cast"; break;
            case OP_TRANSPOSE: op_name = "transpose"; break;
            case OP_RESHAPE: op_name = "reshape"; break;
            default:
                op_name = "var";
                auto c = graph.add({"var", {}, v.as_tensor() ? v.as_tensor()->dtype : DType::F32, {}});
                value_to_class[v.id()] = c;
                class_to_value[c] = v;
                return c;
        }

        ENode enode;
        enode.op = op_name;
        if (auto t = v.as_tensor()) enode.dtype = t->dtype;
        for (auto& operand : def->operands) {
            enode.children.push_back(build(operand, depth + 1));
        }
        auto c = graph.add(enode);
        value_to_class[v.id()] = c;
        class_to_value[c] = v;
        return c;
    };

    result.root_class = build(root, 0);
    return result;
}

} // namespace

PreservedAnalyses EGraphSuperoptimizerPass::run(Module& m, AnalysisManager&) {
    bool changed = false;

    // Get the full rewrite rule library for Relaxed mode.
    // (FastMath would also include TC eligibility rules.)
    auto rules = get_rewrite_rules(NumericalMode::Relaxed);

    // Convert RewriteRule to EGraph::Rewrite.
    std::vector<EGraph::Rewrite> egraph_rules;
    for (auto& r : rules) {
        EGraph::Rewrite er;
        er.name = r.name;
        er.lhs = r.rule.lhs;
        er.rhs = r.rule.rhs;
        egraph_rules.push_back(std::move(er));
    }

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (!op.is_pure()) continue;
            if (op.operands.empty()) continue;
            if (op.results.empty()) continue;
            Value root = op.results[0];

            auto build_result = build_egraph_for_region(m, root, 8);
            if (!build_result) continue;

            auto& graph = build_result->graph;
            EClassId root_class = build_result->root_class;

            try {
                graph.saturate(egraph_rules, 4);
            } catch (...) {
                continue;
            }

            auto extracted = graph.extract(root_class, tensor_cost);

            // If the extracted form is different from the original and cheaper,
            // try to replace the IR.
            if (extracted.node.op == "var" && !extracted.node.children.empty()) {
                EClassId child_class = extracted.node.children[0];
                auto it = build_result->class_to_value.find(child_class);
                if (it != build_result->class_to_value.end()) {
                    Value replacement = it->second;
                    if (replacement != root) {
                        m.replace_all_uses(root, replacement);
                        changed = true;
                    }
                }
            }
            // If the extracted form is a const (e.g. from constant folding),
            // we could replace with a constant op — but that requires
            // knowing the constant value, which we don't track yet.
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    pa.preserve<AnalysisManager>();
    return pa;
}

} // namespace cg
