// egraph/egraph.hpp - e-graph for tensor algebraic rewrites
//
// The e-graph is a *superoptimizer* used for specific tensor sub-expressions.
// It is NOT the main IR. The pattern is:
//
//     Tensor IR
//        |
//        v
//   extract optimizable region
//        |
//        v
//      e-graph
//        |
//        v
//     saturate
//        |
//        v
//   extract (tensor-aware cost)
//        |
//        v
//     Tensor IR
//
// We do not e-graph the entire compiler because saturation can explode.
//
// The e-graph's cost function is tensor-aware: it considers FLOPs, memory
// traffic, workspace, and layout conversions rather than just "smallest
// expression".
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace cg {

using ENodeId = u32;
using EClassId = u32;

// An ENode is an operation applied to e-class ids (not e-node ids).
struct ENode {
    std::string op;            // e.g. "add", "mul", "matmul"
    std::vector<EClassId> children;
    std::optional<DType> dtype;
    std::optional<std::vector<i64>> shape;  // optional shape tag

    bool operator==(const ENode& o) const {
        return op == o.op && children == o.children && dtype == o.dtype &&
               shape == o.shape;
    }
};

class EGraph {
public:
    EGraph() = default;

    // Add a standalone e-node, returning its e-class id.
    EClassId add(const ENode& n);

    // Merge two e-classes (assert equivalence).
    void merge(EClassId a, EClassId b);

    // Apply a rewrite rule. The rule's lhs is matched; on match, the rhs is
    // added and the matched class is merged with the rhs's class.
    struct Rewrite {
        ENode lhs;
        std::function<ENode(const std::unordered_map<std::string, EClassId>&)> rhs;
        std::vector<std::string> var_names;  // names for the wildcard children
    };

    void saturate(const std::vector<Rewrite>& rewrites, usize max_iters = 8);

    // Extract the best (cheapest) ENode for class `c` using a tensor-aware
    // cost function.
    struct ExtractedExpr {
        ENode node;
        double cost;
    };
    ExtractedExpr extract(EClassId c,
                          std::function<double(const ENode&)> cost_fn) const;

    usize num_classes() const { return classes_.size(); }

    const std::vector<ENode>& nodes_in_class(EClassId c) const {
        return classes_[c].nodes;
    }

private:
    struct EClass {
        std::vector<ENode> nodes;
        EClassId parent;   // union-find parent
        usize    rank = 0;
    };

    std::vector<EClass> classes_;
    std::unordered_map<u64, EClassId> hashcons_;

    EClassId find(EClassId c) {
        // Path compression.
        if (classes_[c].parent == c) return c;
        return classes_[c].parent = find(classes_[c].parent);
    }

    u64 node_hash(const ENode& n) const {
        u64 h = std::hash<std::string>{}(n.op);
        for (auto c : n.children) hash_combine(h, static_cast<u64>(c));
        if (n.dtype) hash_combine(h, static_cast<u8>(static_cast<u8>(*n.dtype)));
        if (n.shape) for (auto s : *n.shape) hash_combine(h, std::hash<i64>{}(s));
        return h;
    }
};

} // namespace cg
