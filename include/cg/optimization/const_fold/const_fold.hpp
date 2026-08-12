// optimization/const_fold.hpp - constant folding
//
// Folds operations whose operands are all constant tensors into a new
// constant tensor. The current implementation folds:
//   - constant + constant -> constant
//   - constant * constant -> constant
//   - constant reshape / broadcast / transpose -> constant
//
// We deliberately do NOT fold operations that would produce a tensor larger
// than a configurable cap (default 1 MiB), to avoid materializing huge
// weight matrices at compile time.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class ConstantFoldingPass : public Pass {
public:
    explicit ConstantFoldingPass(u64 max_fold_bytes = 1ull << 20) // 1 MiB
        : max_fold_bytes_(max_fold_bytes) {}

    std::string name() const override { return "const_fold"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;

private:
    u64 max_fold_bytes_;
};

} // namespace cg
