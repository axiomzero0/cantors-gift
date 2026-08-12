// analysis/analysis.hpp - analysis framework with caching and invalidation
//
// An Analysis is a cached, queryable fact about a Module. Analyses register
// themselves via `AnalysisManager::get<T>()` and are computed lazily.
//
// Each pass returns a `PreservedAnalyses` describing which analyses survived
// the transformation. The AnalysisManager drops caches for analyses that were
// *not* preserved, so the next `get<T>()` recomputes them.
//
// This mirrors the spirit of LLVM's analysis infrastructure but is tailored
// to Tensor IR's pass model.
#pragma once

#include "cg/core/util.hpp"
#include "cg/ir/module.hpp"

#include <any>
#include <memory>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

namespace cg {

// Marks which analyses a pass preserved.
class PreservedAnalyses {
public:
    PreservedAnalyses() = default;

    static PreservedAnalyses all() {
        PreservedAnalyses p; p.all_ = true; return p;
    }
    static PreservedAnalyses none() {
        return PreservedAnalyses{};
    }

    template <typename T>
    void preserve() {
        preserved_.insert(std::type_index(typeid(T)));
    }

    bool preserves_all() const { return all_; }

    const std::unordered_set<std::type_index>& preserved_types() const {
        return preserved_;
    }

private:
    bool all_ = false;
    std::unordered_set<std::type_index> preserved_;
};

// Base class for analyses.
class AnalysisBase {
public:
    virtual ~AnalysisBase() = default;
};

class AnalysisManager {
public:
    explicit AnalysisManager(Module& m) : module_(m) {}

    Module& module() { return module_; }
    const Module& module() const { return module_; }

    // Get or compute an analysis. The analysis `T` must have a constructor
    // that takes `AnalysisManager&`.
    template <typename T>
    T& get() {
        auto key = std::type_index(typeid(T));
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return std::any_cast<T&>(it->second);
        }
        auto ptr = std::make_any<T>(T(*this));
        auto& ref = std::any_cast<T&>(ptr);
        cache_[key] = std::move(ptr);
        return ref;
    }

    template <typename T>
    bool has() const {
        return cache_.find(std::type_index(typeid(T))) != cache_.end();
    }

    // Invalidate analyses not preserved by `pa`.
    void invalidate(const PreservedAnalyses& pa) {
        if (pa.preserves_all()) return;
        const auto& preserved = pa.preserved_types();
        for (auto it = cache_.begin(); it != cache_.end(); ) {
            if (preserved.count(it->first)) { ++it; continue; }
            it = cache_.erase(it);
        }
    }

    void clear() { cache_.clear(); }

private:
    Module& module_;
    std::unordered_map<std::type_index, std::any> cache_;
};

} // namespace cg
