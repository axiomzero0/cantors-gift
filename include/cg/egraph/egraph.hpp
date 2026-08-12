// egraph/egraph.hpp - e-graph with nested pattern matching and e-class creation
//
// Supports:
//   - Nested LHS patterns: add(mul(a, b), c) matches add(X, c) where X is mul(a, b)
//   - RHS that creates new e-classes: relu(x) → max(x, 0) creates a const node
//   - Rebuild after merge for congruence closure
//   - Saturation to fixpoint
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

using ENodeId = u32;
using EClassId = u32;

struct ENode {
    std::string op;
    std::vector<EClassId> children;
    std::optional<DType> dtype;
    std::optional<std::vector<i64>> shape;

    bool operator==(const ENode& o) const {
        return op == o.op && children == o.children && dtype == o.dtype &&
               shape == o.shape;
    }
};

// A Pattern is either a variable (matches any e-class, binds to a name)
// or a node pattern (matches op + recursively matches children).
struct Pattern {
    bool is_variable = false;
    std::string var_name;        // if is_variable
    std::string op;              // if not variable
    std::vector<Pattern> children;  // if not variable

    // Factory: create a variable pattern.
    static Pattern var(const std::string& name) {
        Pattern p;
        p.is_variable = true;
        p.var_name = name;
        return p;
    }

    // Factory: create a node pattern.
    static Pattern node(const std::string& op,
                        std::vector<Pattern> children = {}) {
        Pattern p;
        p.is_variable = false;
        p.op = op;
        p.children = std::move(children);
        return p;
    }
};

class EGraph;

// RHS generator: given the e-graph and the substitution, produce an e-class.
// This allows creating new e-classes (e.g. constants, cast ops) in the RHS.
using RHSGen = std::function<EClassId(
    EGraph& eg,
    const std::unordered_map<std::string, EClassId>& subst)>;

class EGraph {
public:
    EGraph() = default;

    // Add a standalone e-node, returning its e-class id.
    EClassId add(const ENode& n);

    // Add a constant e-node (convenience).
    EClassId add_constant(i64 value, DType dt = DType::F32);
    EClassId add_constant(double value, DType dt = DType::F32);

    // Merge two e-classes (assert equivalence).
    void merge(EClassId a, EClassId b);

    // Rewrite rule with nested pattern support.
    struct Rewrite {
        std::string name;
        Pattern lhs;
        RHSGen rhs;
    };

    // Saturate: apply rewrites until fixpoint or max_iters.
    // After each merge, rebuilds the hashcons for congruence closure.
    void saturate(const std::vector<Rewrite>& rewrites, usize max_iters = 8);

    // Extract the best (cheapest) ENode for class `c`.
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

    // Find the root class for `c` (non-const, does path compression).
    EClassId find(EClassId c) {
        if (c >= classes_.size()) return c;
        if (classes_[c].parent == c) return c;
        return classes_[c].parent = find(classes_[c].parent);
    }

    // Const version of find (no path compression).
    EClassId find_const(EClassId c) const {
        if (c >= classes_.size()) return c;
        EClassId root = c;
        while (classes_[root].parent != root) root = classes_[root].parent;
        return root;
    }

private:
    struct EClass {
        std::vector<ENode> nodes;
        EClassId parent;
        usize rank = 0;
    };

    std::vector<EClass> classes_;
    std::unordered_map<u64, EClassId> hashcons_;

    u64 node_hash(const ENode& n) const {
        u64 h = std::hash<std::string>{}(n.op);
        for (auto c : n.children) hash_combine(h, static_cast<u64>(c));
        if (n.dtype) hash_combine(h, static_cast<u8>(*n.dtype));
        if (n.shape) for (auto s : *n.shape) hash_combine(h, std::hash<i64>{}(s));
        return h;
    }

    // Try to match `pattern` against `class_id`. On success, fills `subst`
    // and returns true.
    bool match(const Pattern& pattern, EClassId class_id,
               std::unordered_map<std::string, EClassId>& subst) const;

    // Rebuild: rehash all e-nodes to discover new congruences from merges.
    void rebuild();
};

} // namespace cg
