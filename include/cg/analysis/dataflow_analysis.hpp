// analysis/dataflow_analysis.hpp - use-def / def-use chains, fanout, critical path
//
// Computes a complete dependency graph for the module:
//   - def-use chains: for every Value, every operation that uses it
//   - use-def chains: for every Operation, every Value it depends on
//   - fanout:        number of consumers of a Value
//   - fanin:         number of distinct producers feeding an Operation
//   - critical path: longest dependency chain through the module
//   - topological order: a valid schedule order respecting dependencies
//
// This is the foundation of Global Tensor Analysis. Every downstream
// optimization (fusion, recomputation, memory planning, layout) consults it.
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/ir/module.hpp"
#include "cg/ir/value.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

class DataflowAnalysis : public AnalysisBase {
public:
    explicit DataflowAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    // The set of operations that read `v` (def-use).
    const std::vector<u32>& users(Value v) const {
        static const std::vector<u32> empty;
        auto it = users_.find(v.id());
        return it != users_.end() ? it->second : empty;
    }

    // Number of distinct consumers of `v`.
    usize fanout(Value v) const {
        return users(v).size();
    }

    // Number of distinct producer operations feeding `op`.
    usize fanin(const Operation& op) const {
        std::unordered_set<u32> distinct;
        for (auto& v : op.operands) {
            for (auto uid : users(v)) distinct.insert(uid);
        }
        return distinct.size();
    }

    // Topologically sorted operation ids (def before use).
    const std::vector<u32>& topo_order() const { return topo_order_; }

    // Critical path length through the module (count of operations on the
    // longest dependency chain). Useful for parallelism estimation.
    usize critical_path_length() const { return critical_path_; }

    // Map from ValueId -> defining OpId (or 0 for block arguments).
    u32 defining_op(Value v) const {
        auto it = defining_op_.find(v.id());
        return it != defining_op_.end() ? it->second : 0;
    }

    // True iff `producer` is a single-consumer producer of `consumer`
    // (canonical fusion pattern).
    bool is_single_consumer(Value producer, const Operation& consumer) const {
        const auto& u = users(producer);
        return u.size() == 1 && u[0] == consumer.id;
    }

    void invalidate() {
        users_.clear();
        defining_op_.clear();
        topo_order_.clear();
        critical_path_ = 0;
    }

private:
    void compute() {
        Module& m = am_.module();

        // Build def-use + defining_op.
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                if (!op.results.empty())
                    defining_op_[op.results[0].id()] = op.id;
                for (auto& v : op.operands) {
                    users_[v.id()].push_back(op.id);
                }
            }
        }

        // Topological sort via Kahn's algorithm.
        // Build in-degree per op.
        std::unordered_map<u32, usize> indeg;
        std::unordered_map<u32, std::vector<u32>> succ;
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                indeg[op.id]; // ensure key exists
                for (auto& v : op.operands) {
                    u32 def = defining_op(v);
                    if (def == 0) continue; // block arg
                    if (def == op.id) continue;
                    succ[def].push_back(op.id);
                    indeg[op.id]++;
                }
            }
        }

        std::vector<u32> ready;
        for (auto& [id, d] : indeg) if (d == 0) ready.push_back(id);
        // Sort ready by op id for determinism.
        std::sort(ready.begin(), ready.end());

        while (!ready.empty()) {
            u32 cur = ready.back();
            ready.pop_back();
            topo_order_.push_back(cur);
            for (u32 s : succ[cur]) {
                if (--indeg[s] == 0) ready.push_back(s);
            }
            std::sort(ready.begin(), ready.end());
        }

        // Critical path via longest path in DAG.
        std::unordered_map<u32, usize> depth;
        usize max_depth = 0;
        for (u32 id : topo_order_) {
            usize d = 0;
            // Look at every predecessor of `id`.
            // We do this by walking the module; ops are cheap.
            for (auto& f : am_.module().functions()) {
                for (auto& op : *f->entry()) {
                    if (op.id != id) continue;
                    for (auto& v : op.operands) {
                        u32 def = defining_op(v);
                        if (def == 0 || def == id) continue;
                        auto it = depth.find(def);
                        if (it != depth.end())
                            d = std::max(d, it->second + 1);
                    }
                }
            }
            depth[id] = d;
            max_depth = std::max(max_depth, d);
        }
        critical_path_ = max_depth;
    }

    AnalysisManager& am_;
    std::unordered_map<ValueId, std::vector<u32>> users_;
    std::unordered_map<ValueId, u32> defining_op_;
    std::vector<u32> topo_order_;
    usize critical_path_ = 0;
};

} // namespace cg
