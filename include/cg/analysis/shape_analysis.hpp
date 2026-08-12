// analysis/shape_analysis.hpp - shape inference analysis
//
// Recomputes and caches the shape of every Value in the module. Used by
// passes that need to know shapes (canonicalization, fusion, layout, ...).
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/ir/module.hpp"
#include "cg/shape/dim_expr.hpp"

#include <unordered_map>

namespace cg {

class ShapeAnalysis : public AnalysisBase {
public:
    explicit ShapeAnalysis(AnalysisManager& am) : am_(am) {
        compute();
    }

    const Shape& shape_of(Value v) const {
        static Shape empty;
        auto it = shapes_.find(v.id());
        return it != shapes_.end() ? it->second : empty;
    }

    void invalidate() { shapes_.clear(); }

private:
    void compute() {
        Module& m = am_.module();
        for (auto& f : m.functions()) {
            for (auto& arg : f->args()) {
                if (auto t = arg.as_tensor()) shapes_[arg.id()] = t->shape;
            }
            for (auto& op : *f->entry()) {
                for (auto& r : op.results) {
                    if (auto t = r.as_tensor()) shapes_[r.id()] = t->shape;
                }
            }
        }
    }

    AnalysisManager& am_;
    std::unordered_map<ValueId, Shape> shapes_;
};

} // namespace cg
