// analysis/layout_analysis.hpp - layout inference analysis
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/ir/module.hpp"
#include "cg/layout/layout.hpp"

#include <unordered_map>

namespace cg {

class LayoutAnalysis : public AnalysisBase {
public:
    explicit LayoutAnalysis(AnalysisManager& am) : am_(am) { compute(); }

    LayoutPtr layout_of(Value v) const {
        auto it = layouts_.find(v.id());
        return it != layouts_.end() ? it->second : nullptr;
    }

    void invalidate() { layouts_.clear(); }

private:
    void compute() {
        Module& m = am_.module();
        for (auto& f : m.functions()) {
            for (auto& arg : f->args()) {
                if (auto t = arg.as_tensor()) layouts_[arg.id()] = t->layout;
            }
            for (auto& op : *f->entry()) {
                for (auto& r : op.results) {
                    if (auto t = r.as_tensor()) layouts_[r.id()] = t->layout;
                }
            }
        }
    }

    AnalysisManager& am_;
    std::unordered_map<ValueId, LayoutPtr> layouts_;
};

} // namespace cg
