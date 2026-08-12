// optimization/pass.hpp - pass base classes and pass manager
//
// A Pass transforms a Module. After running, it returns a PreservedAnalyses
// describing which cached analyses survived.
//
// Passes are registered with a PassManager, which runs them in order and
// forwards the PreservedAnalyses to the AnalysisManager.
#pragma once

#include "cg/analysis/analysis.hpp"
#include "cg/core/util.hpp"
#include "cg/ir/module.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cg {

class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    virtual PreservedAnalyses run(Module& m, AnalysisManager& am) = 0;
};

class PassManager {
public:
    void add(std::unique_ptr<Pass> p) { passes_.push_back(std::move(p)); }

    PreservedAnalyses run(Module& m, AnalysisManager& am) {
        PreservedAnalyses total = PreservedAnalyses::all();
        for (auto& p : passes_) {
            auto pa = p->run(m, am);
            am.invalidate(pa);
            // Merge: any analysis not preserved by some pass is gone globally.
            if (!pa.preserves_all()) {
                if (total.preserves_all()) {
                    total = pa;
                } else {
                    // Intersect: only analyses preserved by BOTH survive.
                    // We approximate by setting total to `pa` since
                    // preserved analyses from earlier passes are already
                    // invalidated by am.invalidate(pa) above.
                    total = pa;
                }
            }
        }
        return total;
    }

    usize size() const { return passes_.size(); }
    bool empty() const { return passes_.empty(); }

    const std::vector<std::unique_ptr<Pass>>& passes() const { return passes_; }

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

// A simple pipeline wraps a PassManager with a name and a flag to enable it.
class Pipeline {
public:
    explicit Pipeline(std::string name) : name_(std::move(name)) {}

    Pipeline& enable(bool e) { enabled_ = e; return *this; }
    bool enabled() const { return enabled_; }
    const std::string& name() const { return name_; }

    PassManager& manager() { return pm_; }
    const PassManager& manager() const { return pm_; }

private:
    std::string name_;
    bool enabled_ = true;
    PassManager pm_;
};

} // namespace cg
