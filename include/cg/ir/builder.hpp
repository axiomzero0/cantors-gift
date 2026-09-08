// ir/builder.hpp - ergonomic construction of IR
//
// The Builder owns a `Function*` and emits operations into its entry block.
// It tracks the current insert position and produces Values for results.
//
// High-level convenience methods (add, mul, matmul, ...) consult the
// OpRegistry to infer result types and to attach the correct traits/effects.
#pragma once

#include "cg/ir/module.hpp"
#include "cg/ir/operation.hpp"
#include "cg/ir/ops.hpp"

#include <utility>
#include <vector>

namespace cg {

class Builder {
public:
    explicit Builder(Function* fn) : fn_(fn) {}

    Function* function() const { return fn_; }
    Block*    block()    const { return fn_->entry(); }

    // Create an Operation with `operands`, `attrs` and the given opcode. The
    // result types are inferred via OpRegistry. Returns a pointer to the new
    // operation (the Builder retains ownership in the block).
    Operation* create(Opcode opcode,
                      SmallVector<Value, 4> operands,
                      AttributeDict attrs = {});

    // Create an Operation with explicit result types (skips inference).
    Operation* create_with_results(Opcode opcode,
                                   SmallVector<Value, 4> operands,
                                   SmallVector<TypePtr, 2> result_types,
                                   AttributeDict attrs = {});

    // ---------- Convenience: constants and inputs ----------
    Value constant_tensor(Shape shape, DType dt, std::vector<u8> /*bytes*/ = {});
    Value input_tensor(Shape shape, DType dt);
    void  output_tensor(Value v);

    // ---------- Convenience: arithmetic ----------
    Value add(Value a, Value b);
    Value sub(Value a, Value b);
    Value mul(Value a, Value b);
    Value div(Value a, Value b);
    Value neg(Value a);

    // ---------- Convenience: elementwise ----------
    Value relu(Value a);
    Value gelu(Value a);
    Value exp(Value a);
    Value sqrt(Value a);

    // ---------- Convenience: tensor ops ----------
    Value matmul(Value a, Value b);
    Value broadcast(Value a, std::vector<i64> target_shape);
    Value reshape(Value a, std::vector<i64> target_shape);
    Value transpose(Value a, std::vector<i32> perm);
    Value reduce_sum(Value a, std::vector<i32> axes, bool keep_dims = false);
    Value reduce_max(Value a, std::vector<i32> axes, bool keep_dims = false);
    Value cast(Value a, DType target_dtype);

    // ---------- Convenience: domain ops ----------
    Value conv2d(Value input, Value weight,
                 i64 stride_h = 1, i64 stride_w = 1,
                 i64 pad_h = 0, i64 pad_w = 0,
                 i64 dilation_h = 1, i64 dilation_w = 1);
    Value softmax(Value a);
    Value layernorm(Value a);
    Value batchnorm(Value a);
    Value gather(Value input, Value indices);
    Value concat(std::vector<Value> inputs, i64 axis);
    Value slice(Value input, std::vector<i64> begins, std::vector<i64> ends);
    Value sigmoid(Value a);
    Value tanh(Value a);
    Value log(Value a);

    // ---------- Convenience: brain-domain semantic primitives ----------
    // See include/cg/ir/operation.hpp for the semantics of each op.
    // These wrap `create(...)` so the caller doesn't have to construct
    // AttributeDict manually. Each pure op is CSE-eligible: two calls
    // with the same operands and same attributes will deduplicate.

    // d_ij² = Σ_k (x_i,k - x_j,k)²   for x:[N,D]  ->  d²:[N,N]
    Value pairwise_dist_sq(Value x);

    // N_r(i) = { j : |x_i-x_j|² < r² }  for x:[N,D]  ->  indices:[N,K]
    // K = max_neighbors (default 32). r = radius.
    Value neighbor_query(Value x, double radius, i64 max_neighbors = 32);

    // k(d², σ) = exp(-d² / (2σ²))   applied to d²:[N,N]  ->  k:[N,N]
    Value gaussian_kernel(Value d2, double sigma);

    // g_ij^inh = g0 * exp(-d² / (2σ_inh²))   applied to d²:[N,N]  ->  g:[N,N]
    Value lateral_inhibition(Value d2, double g0, double sigma_inh);

    // τ_ij = τ0 + sqrt(d²) * inv_speed   applied to d²:[N,N]  ->  τ:[N,N]
    Value delay_from_dist(Value d2, double tau0, double inv_speed);

    // Aggregate spike events:  events:[E,3]=(i,j,t), w:[N,N]  ->  I:[N]
    Value synaptic_arrival(Value events, Value weights);

    // P_ij ← P_ij * exp(-Δt / τ_e)   for p:[N,N]  ->  p':[N,N]
    Value eligibility_update(Value p, double dt, double tau_e);

    // C_i ← Σ_j w_ij * c_j   for w:[N,N], c:[N]  ->  C:[N]
    Value credit_propagate(Value weights, Value credits);

    // top-K sparse activation:  x:[N]  ->  indices:[K]
    Value active_set(Value x, i64 topk);

    // region-level reduction:  x:[N], region_ids:[N]  ->  out:[R]
    Value region_aggregate(Value x, Value region_ids, i64 num_regions);

    // Emit a spike event: v:[N], t: scalar  ->  (no result)
    void  spike(Value activations, Value t);

    // Structural plasticity epoch marker (no operands, no results).
    void  structural_epoch();

private:
    Function* fn_;
};

} // namespace cg
