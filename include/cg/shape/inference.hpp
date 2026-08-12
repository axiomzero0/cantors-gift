// shape/inference.hpp - shape inference for tensor operations
//
// Each Tensor IR operation knows how to infer the shape of its result(s) from
// the shapes of its operands. This module provides a uniform API for that
// inference, used both by the builder (when constructing ops) and by the
// `ShapeInferencePass`.
#pragma once

#include "cg/shape/dim_expr.hpp"
#include "cg/shape/constraint.hpp"
#include "cg/shape/simplifier.hpp"

#include <optional>
#include <string>

namespace cg {

// Result of an inference call. On failure, `message` is set; on success,
// `shape` is the inferred result shape.
struct InferResult {
    bool ok = false;
    Shape shape;
    std::string message;
};

// Elementwise binary ops: shapes broadcast against each other.
InferResult infer_elementwise_binary(const Shape& a, const Shape& b);

// Reduction: axes is a sorted unique list of dimensions to reduce; keep_dims
// controls whether reduced axes are kept as size-1.
InferResult infer_reduction(const Shape& a,
                            Span<const i32> axes,
                            bool keep_dims);

// Broadcast: explicit shape to broadcast to.
InferResult infer_broadcast(const Shape& a, const Shape& to);

// Reshape: target shape must have the same number of elements as the input
// (with at most one dynamic (-1) dimension).
InferResult infer_reshape(const Shape& a, const Shape& to);

// Transpose: perm must be a permutation of [0, a.rank()).
InferResult infer_transpose(const Shape& a, Span<const i32> perm);

// Matmul: standard [..., M, K] x [..., K, N] -> [..., M, N] with batch
// broadcasting.
InferResult infer_matmul(const Shape& a, const Shape& b);

} // namespace cg
