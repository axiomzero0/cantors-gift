// pipeline/software_pipeline.hpp - decoupled access/execute software pipelining
//
// This is Triton's actual secret: tl.load with num_stages makes the
// data-movement pipeline a first-class IR concept.
//
// Given a tile loop:
//   for k in range(0, K, k_tile):
//     load A_tile[k] from global → shared
//     load B_tile[k] from global → shared
//     compute C += A_tile @ B_tile
//     store C
//
// And num_stages=3, the pipelining pass emits:
//
//   // Prologue: issue loads for stages 0 and 1
//   load A[0], B[0]  → buffer 0
//   load A[1], B[1]  → buffer 1
//
//   // Steady state: overlap load and compute
//   for k in range(0, K - 2, k_tile):
//     load A[k+2], B[k+2]  → buffer (k+2) % 3  // async, overlaps with compute
//     wait for buffer k % 3
//     compute C += A[k%3] @ B[k%3]
//
//   // Epilogue: drain remaining stages
//   wait for buffer (K-2) % 3
//   compute C += A[(K-2)%3] @ B[(K-2)%3]
//   wait for buffer (K-1) % 3
//   compute C += A[(K-1)%3] @ B[(K-1)%3]
//
// This overlaps global memory latency with compute, which is the key to
// achieving high utilization on GPUs.
#pragma once

#include "cg/core/util.hpp"
#include "cg/schedule/schedule.hpp"

#include <string>
#include <vector>

namespace cg {

// A pipeline stage describes what happens in one stage of the pipeline.
struct PipelineStage {
    // The load operations (global → shared) for this stage.
    struct LoadOp {
        std::string tensor_name;   // "A" or "B"
        std::string src_ptr_reg;   // PTX register holding the global pointer
        std::string dst_smem_reg;  // PTX register holding the shared pointer
        std::string offset_reg;    // PTX register holding the byte offset
        u32 bytes;                 // number of bytes to load
    };
    std::vector<LoadOp> loads;

    // The compute operations for this stage.
    struct ComputeOp {
        std::string description;  // human-readable (e.g., "mma.sync m16n8k16")
    };
    std::vector<ComputeOp> computes;

    // The store operations (shared → global) for this stage, if any.
    struct StoreOp {
        std::string tensor_name;
        std::string dst_ptr_reg;
        std::string offset_reg;
        u32 bytes;
    };
    std::vector<StoreOp> stores;
};

// A software pipeline configuration.
struct PipelineConfig {
    u32 num_stages = 3;        // number of pipeline stages (default: 3)
    u32 num_buffers = 3;       // number of shared memory buffers (usually == num_stages)
    bool use_async_copy = true; // use cp.async (Ampere+) or wgmma (Hopper+)
    bool use_async_barrier = false; // use mbarrier (Hopper+)

    // Buffer sizes (bytes) for each tensor.
    u32 buffer_a_bytes = 0;
    u32 buffer_b_bytes = 0;
    u32 buffer_c_bytes = 0;
};

// Emit a software-pipelined tile loop in PTX.
//
// Given:
//   - A tile loop body (load + compute + store)
//   - num_stages
//   - K dimension and k_tile size
//
// Emit:
//   1. Prologue: issue the first num_stages-1 loads
//   2. Steady state: overlapped load + compute
//   3. Epilogue: drain remaining computes
class SoftwarePipelineEmitter {
public:
    SoftwarePipelineEmitter() = default;

    // Emit the prologue (issue first num_stages-1 loads).
    std::string emit_prologue(
        const PipelineConfig& config,
        const PipelineStage& stage,
        u32 k_total,
        u32 k_tile) const;

    // Emit the steady-state loop body (overlapped load + compute).
    std::string emit_steady_state(
        const PipelineConfig& config,
        const PipelineStage& stage,
        u32 k_total,
        u32 k_tile) const;

    // Emit the epilogue (drain remaining computes).
    std::string emit_epilogue(
        const PipelineConfig& config,
        const PipelineStage& stage,
        u32 k_total,
        u32 k_tile) const;

    // Emit the complete pipelined loop.
    std::string emit_pipelined_loop(
        const PipelineConfig& config,
        const PipelineStage& stage,
        u32 k_total,
        u32 k_tile) const;
};

// Analysis: estimate the overlap factor for a given pipeline config.
// With 0 stages: max(compute, memory) — no overlap.
// With N stages and infinite buffers: max(compute/N, memory/N) — full overlap.
// Reality is between, bounded by shared memory and register pressure.
struct PipelineOverlapEstimate {
    double compute_sec;
    double memory_sec;
    double overlapped_sec;    // estimated time with overlap
    double overlap_factor;    // 0.0 = no overlap, 1.0 = full overlap
};

PipelineOverlapEstimate estimate_pipeline_overlap(
    double compute_sec,
    double memory_sec,
    u32 num_stages,
    u32 num_warps,
    u32 shared_mem_available,
    u32 shared_mem_per_stage);

} // namespace cg
