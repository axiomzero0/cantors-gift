// egraph/egraph.cpp - implementation
#include "cg/egraph/egraph.hpp"

#include <algorithm>
#include <limits>
#include <queue>

namespace cg {

namespace {

// Substitute child e-class ids in `n` according to `subst`.
ENode substitute(const ENode& n, const std::unordered_map<EClassId, EClassId>& subst) {
    ENode out = n;
    for (auto& c : out.children) {
        auto it = subst.find(c);
        if (it != subst.end()) c = it->second;
    }
    return out;
}

} // namespace

EClassId EGraph::add(const ENode& n) {
    u64 h = node_hash(n);
    auto it = hashcons_.find(h);
    if (it != hashcons_.end()) return find(it->second);

    EClassId id = static_cast<EClassId>(classes_.size());
    EClass c;
    c.nodes.push_back(n);
    c.parent = id;
    classes_.push_back(std::move(c));
    hashcons_[h] = id;
    return id;
}

void EGraph::merge(EClassId a, EClassId b) {
    EClassId ra = find(a), rb = find(b);
    if (ra == rb) return;
    // Union by rank.
    if (classes_[ra].rank < classes_[rb].rank) std::swap(ra, rb);
    classes_[rb].parent = ra;
    if (classes_[ra].rank == classes_[rb].rank) classes_[ra].rank++;
    // Move nodes from rb into ra.
    for (auto& n : classes_[rb].nodes) classes_[ra].nodes.push_back(std::move(n));
    classes_[rb].nodes.clear();
}

void EGraph::saturate(const std::vector<Rewrite>& rewrites, usize max_iters) {
    for (usize iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        // For each class, attempt to match each rewrite.
        for (EClassId cid = 0; cid < classes_.size(); ++cid) {
            if (classes_[cid].nodes.empty()) continue;
            for (const auto& rw : rewrites) {
                // Match rw.lhs against each node in this class.
                for (const auto& n : classes_[cid].nodes) {
                    if (n.op != rw.lhs.op) continue;
                    if (n.children.size() != rw.lhs.children.size()) continue;
                    // Build substitution: rw.lhs.children[i] is a "variable"
                    // named by rw.var_names[i]; n.children[i] is the concrete
                    // class id (after find()).
                    std::unordered_map<std::string, EClassId> subst;
                    bool ok = true;
                    for (usize i = 0; i < n.children.size(); ++i) {
                        if (i < rw.var_names.size()) {
                            subst[rw.var_names[i]] = find(n.children[i]);
                        } else {
                            // No variable for this slot; require structural match.
                            if (rw.lhs.children[i] != n.children[i]) { ok = false; break; }
                        }
                    }
                    if (!ok) continue;
                    // Apply the rhs to build a new node.
                    ENode rhs = rw.rhs(subst);
                    EClassId rhs_class = add(rhs);
                    EClassId lhs_class = find(cid);
                    if (rhs_class != lhs_class) {
                        merge(lhs_class, rhs_class);
                        changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }
}

EGraph::ExtractedExpr
EGraph::extract(EClassId c, std::function<double(const ENode&)> cost_fn) const {
    // Greedy extraction: for each class, pick the cheapest node, recursing
    // into children.
    std::unordered_map<EClassId, ExtractedExpr> memo;

    std::function<ExtractedExpr(EClassId)> rec = [&](EClassId id) -> ExtractedExpr {
        // Find the root class for `id` without path compression (const method).
        EClassId root = id;
        std::unordered_set<EClassId> visited;
        while (classes_[root].parent != root) {
            if (!visited.insert(root).second) {
                // Cycle detected (shouldn't happen); break.
                break;
            }
            root = classes_[root].parent;
        }
        auto it = memo.find(root);
        if (it != memo.end()) return it->second;

        // Mark as in-progress to avoid infinite recursion on cycles.
        memo[root] = ExtractedExpr{{}, 0.0};

        ExtractedExpr best{{}, std::numeric_limits<double>::infinity()};
        for (const auto& n : classes_[root].nodes) {
            double c = cost_fn(n);
            for (auto ch : n.children) {
                auto sub = rec(ch);
                c += sub.cost;
            }
            if (c < best.cost) {
                best.node = n;
                best.cost = c;
            }
        }
        if (best.cost == std::numeric_limits<double>::infinity()) {
            best.cost = 0;
        }
        memo[root] = best;
        return best;
    };

    return rec(c);
}

} // namespace cg
