// egraph/egraph.cpp - implementation with nested pattern matching
#include "cg/egraph/egraph.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace cg {

EClassId EGraph::add(const ENode& n) {
    // Canonicalize children to their root classes.
    ENode canonical = n;
    for (auto& c : canonical.children) c = find(c);

    u64 h = node_hash(canonical);
    auto it = hashcons_.find(h);
    if (it != hashcons_.end()) return find(it->second);

    EClassId id = static_cast<EClassId>(classes_.size());
    EClass c;
    c.nodes.push_back(canonical);
    c.parent = id;
    classes_.push_back(std::move(c));
    hashcons_[h] = id;
    return id;
}

EClassId EGraph::add_constant(i64 value, DType dt) {
    ENode n;
    n.op = "const";
    n.dtype = dt;
    // Store the value in the shape field as a hack (we don't have a value field).
    n.shape = std::vector<i64>{value};
    return add(n);
}

EClassId EGraph::add_constant(double value, DType dt) {
    // Store float as bits in an i64.
    i64 bits;
    std::memcpy(&bits, &value, 8);
    return add_constant(bits, dt);
}

void EGraph::merge(EClassId a, EClassId b) {
    EClassId ra = find(a), rb = find(b);
    if (ra == rb) return;
    if (classes_[ra].rank < classes_[rb].rank) std::swap(ra, rb);
    classes_[rb].parent = ra;
    if (classes_[ra].rank == classes_[rb].rank) classes_[ra].rank++;
    for (auto& n : classes_[rb].nodes) classes_[ra].nodes.push_back(std::move(n));
    classes_[rb].nodes.clear();
}

bool EGraph::match(const Pattern& pattern, EClassId class_id,
                    std::unordered_map<std::string, EClassId>& subst) const {
    EClassId root = find_const(class_id);

    if (pattern.is_variable) {
        // Variable: bind to the root class. If already bound, check consistency.
        auto it = subst.find(pattern.var_name);
        if (it != subst.end()) {
            return find_const(it->second) == root;
        }
        subst[pattern.var_name] = root;
        return true;
    }

    // Node pattern: try to match against each e-node in the class.
    if (root >= classes_.size()) return false;
    for (const auto& n : classes_[root].nodes) {
        if (n.op != pattern.op) continue;
        if (n.children.size() != pattern.children.size()) continue;

        // Save subst so we can backtrack on failure.
        auto saved = subst;
        bool ok = true;
        for (usize i = 0; i < pattern.children.size(); ++i) {
            if (!match(pattern.children[i], n.children[i], subst)) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
        subst = saved;  // backtrack
    }
    return false;
}

void EGraph::rebuild() {
    // Rehash all e-nodes after merges to discover new congruences.
    // This is the "rebuild" step from egg/egglog.
    hashcons_.clear();
    for (EClassId cid = 0; cid < classes_.size(); ++cid) {
        EClassId root = find(cid);
        for (const auto& n : classes_[cid].nodes) {
            ENode canonical = n;
            for (auto& c : canonical.children) c = find(c);
            u64 h = node_hash(canonical);
            auto it = hashcons_.find(h);
            if (it != hashcons_.end()) {
                // Congruence found! Merge.
                EClassId existing = find(it->second);
                if (existing != root) {
                    merge(existing, root);
                }
            } else {
                hashcons_[h] = root;
            }
        }
    }
}

void EGraph::saturate(const std::vector<Rewrite>& rewrites, usize max_iters) {
    for (usize iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // Collect all (class_id, node) pairs to try matching against.
        // We snapshot because matching + adding can modify the e-graph.
        struct MatchTarget {
            EClassId class_id;
            ENode node;
        };
        std::vector<MatchTarget> targets;
        for (EClassId cid = 0; cid < classes_.size(); ++cid) {
            EClassId root = find(cid);
            if (root != cid) continue;  // skip non-root classes
            for (const auto& n : classes_[root].nodes) {
                targets.push_back({root, n});
            }
        }

        for (const auto& [class_id, node] : targets) {
            for (const auto& rw : rewrites) {
                std::unordered_map<std::string, EClassId> subst;
                if (!match(rw.lhs, class_id, subst)) continue;

                // Apply the RHS to create/lookup an e-class.
                EClassId rhs_class = rw.rhs(*this, subst);
                EClassId lhs_root = find(class_id);
                if (rhs_class != lhs_root) {
                    merge(lhs_root, rhs_class);
                    changed = true;
                }
            }
        }

        // Rebuild after all merges to discover congruences.
        if (changed) {
            rebuild();
        }

        if (!changed) break;
    }
}

EGraph::ExtractedExpr
EGraph::extract(EClassId c, std::function<double(const ENode&)> cost_fn) const {
    std::unordered_map<EClassId, ExtractedExpr> memo;

    std::function<ExtractedExpr(EClassId)> rec = [&](EClassId id) -> ExtractedExpr {
        EClassId root = find_const(id);
        auto it = memo.find(root);
        if (it != memo.end()) return it->second;

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
