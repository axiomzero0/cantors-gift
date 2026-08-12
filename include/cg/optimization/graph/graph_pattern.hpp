// optimization/graph_pattern.hpp - graph-level pattern matching
//
// Recognizes multi-op patterns and replaces them with higher-level domain
// ops. Examples:
//   - Q @ K^T -> scale -> softmax -> @ V  becomes  attention(Q, K, V)
//   - conv + bias + activation  becomes  fused_conv
//   - matmul + bias + relu  becomes  fused_matmul_relu
//
// This is where domain knowledge enters the optimizer without contaminating
// the core IR.
#pragma once

#include "cg/optimization/pass.hpp"

namespace cg {

class GraphPatternMatchingPass : public Pass {
public:
    std::string name() const override { return "graph_pattern"; }
    PreservedAnalyses run(Module& m, AnalysisManager& am) override;
};

} // namespace cg
