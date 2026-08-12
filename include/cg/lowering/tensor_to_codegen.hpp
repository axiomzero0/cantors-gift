// lowering/tensor_to_codegen.hpp - lower Tensor IR to Codegen IR
//
// This is the pass that converts the optimized Tensor IR (after the Global
// Barrier) into Codegen IR (the input to the backend emitters). It walks
// each function in topological order and emits CGInstructions for each
// Tensor IR operation.
//
// The lowering is target-independent: it produces generic Codegen IR that
// any backend (PTX, x86, AMD) can consume. Target-specific decisions
// (vector width, register allocation, instruction selection) are made by
// the backend, not here.
//
// The lowering strategy:
//   - Each tensor value is assigned a virtual register (or a memory slot
//     for spilled values).
//   - Elementwise ops become a sequence of vector_load + op + vector_store.
//   - Matmul becomes a tiled loop of vector_load + FMA + vector_store.
//   - Reductions become a tree-reduction sequence.
//   - Allocs/frees from MemoryPlanning become alloc/free annotations on
//     the CGFunction.
#pragma once

#include "cg/codegen/codegen_ir.hpp"
#include "cg/ir/module.hpp"
#include "cg/schedule/schedule.hpp"

#include <unordered_map>

namespace cg {

struct LoweringOptions {
    // Default vector width (in elements) for the target. The backend may
    // override this based on its HardwareProfile.
    u8 default_vector_width = 8;

    // If true, elementwise ops are lowered as a single vector op over the
    // whole tensor. If false, they are lowered as a loop of vector ops.
    bool scalarize_small_tensors = true;

    // Tensors with fewer elements than this are scalarized.
    u64 scalarize_threshold = 64;

    // Tile sizes for matmul lowering.
    u64 matmul_m_tile = 64;
    u64 matmul_n_tile = 64;
    u64 matmul_k_tile = 32;
};

class TensorToCodegenLowering {
public:
    explicit TensorToCodegenLowering(LoweringOptions opts = {})
        : opts_(std::move(opts)) {}

    // Lower an entire Module to a CGModule.
    CGModule lower(const Module& m, const Schedule& schedule = {});

    // Lower a single Function to a CGFunction.
    void lower_function(const Function& f, CGFunction& out,
                         const Schedule& schedule);

    const LoweringOptions& options() const { return opts_; }

private:
    LoweringOptions opts_;

    // Map from Tensor IR ValueId -> CGValue (the virtual register holding
    // the tensor's base pointer or scalar value).
    std::unordered_map<ValueId, CGValue> value_map_;

    // Emit a simple elementwise kernel: load a, load b, op, store result.
    void lower_elementwise_binary(const Operation& op, CGFunction& out);
    void lower_elementwise_unary(const Operation& op, CGFunction& out);
    void lower_matmul(const Operation& op, CGFunction& out);
    void lower_reduction(const Operation& op, CGFunction& out);
    void lower_constant(const Operation& op, CGFunction& out);
    void lower_alloc(const Operation& op, CGFunction& out);
    void lower_free(const Operation& op, CGFunction& out);

    // Get or create the CGValue for a Tensor IR Value.
    CGValue get_or_create_value(Value v, CGFunction& out);

    // Emit a pointer argument for a tensor input.
    CGValue emit_tensor_arg(Value v, CGFunction& out);
};

} // namespace cg
