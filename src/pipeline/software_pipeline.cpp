// pipeline/software_pipeline.cpp - software pipelining emission
#include "cg/pipeline/software_pipeline.hpp"

#include <sstream>
#include <algorithm>

namespace cg {

std::string SoftwarePipelineEmitter::emit_prologue(
    const PipelineConfig& config,
    const PipelineStage& stage,
    u32 k_total,
    u32 k_tile) const {
    std::ostringstream os;
    os << "    // === Pipeline Prologue ===\n";
    os << "    // Issue " << (config.num_stages - 1) << " async loads\n";

    u32 num_prologue_loads = std::min(config.num_stages - 1,
                                       k_total / k_tile);
    for (u32 s = 0; s < num_prologue_loads; ++s) {
        os << "    // Stage " << s << ": load tiles for k=" << (s * k_tile) << "\n";
        for (auto& load : stage.loads) {
            if (config.use_async_copy) {
                // cp.async.ca.shared.global [smem], [global], bytes;
                u32 buf = s % config.num_buffers;
                os << "    cp.async.ca.shared.global "
                   << "[" << load.dst_smem_reg << " + " << (buf * load.bytes) << "], "
                   << "[" << load.src_ptr_reg << " + " << load.offset_reg
                   << " + " << (s * k_tile) << "], " << load.bytes << ";\n";
            } else {
                // Fallback: ld.global + st.shared (synchronous)
                os << "    // Synchronous load (no cp.async support)\n";
            }
        }
    }

    if (config.use_async_copy) {
        os << "    // Commit all async copies\n";
        os << "    cp.async.commit_group;\n";
    }

    os << "    // Wait for first stage to complete\n";
    os << "    cp.async.wait_group " << (num_prologue_loads - 1) << ";\n";
    os << "    bar.sync 0;\n";
    return os.str();
}

std::string SoftwarePipelineEmitter::emit_steady_state(
    const PipelineConfig& config,
    const PipelineStage& stage,
    u32 k_total,
    u32 k_tile) const {
    std::ostringstream os;
    u32 num_iters = k_total / k_tile;
    u32 prologue_iters = std::min(config.num_stages - 1, num_iters);
    u32 steady_iters = num_iters - prologue_iters;

    os << "    // === Pipeline Steady State ===\n";
    os << "    // " << steady_iters << " iterations with overlapped load/compute\n";
    os << "    .reg .u32 %r_stage;\n";
    os << "    mov.u32 %r_stage, " << prologue_iters << ";\n";
    os << "L_pipeline_steady:\n";
    os << "    cmp.ge.u32 %p_steady_end, %r_stage, " << num_iters << ";\n";
    os << "    @%p_steady_end bra L_pipeline_epilogue;\n\n";

    // Issue next stage's load (async, overlaps with current compute).
    os << "    // Issue load for stage %r_stage (async)\n";
    for (auto& load : stage.loads) {
        if (config.use_async_copy) {
            u32 buf_offset_expr = config.num_buffers; // computed at runtime
            os << "    // Load " << load.tensor_name
               << " into buffer (%r_stage % " << config.num_buffers << ")\n";
            os << "    // cp.async.ca.shared.global [smem + buf_offset], [global + k_offset], "
               << load.bytes << "\n";
        }
    }
    if (config.use_async_copy) {
        os << "    cp.async.commit_group;\n";
    }

    // Wait for the compute stage's load to complete.
    os << "    // Wait for the compute stage's load\n";
    if (config.use_async_copy) {
        os << "    cp.async.wait_group " << (config.num_stages - 2) << ";\n";
    }
    os << "    bar.sync 0;\n\n";

    // Compute on current stage.
    os << "    // Compute on stage (%r_stage - " << (config.num_stages - 1) << ")\n";
    for (auto& compute : stage.computes) {
        os << "    // " << compute.description << "\n";
    }

    // Increment and loop.
    os << "    add.u32 %r_stage, %r_stage, 1;\n";
    os << "    bra L_pipeline_steady;\n\n";
    return os.str();
}

std::string SoftwarePipelineEmitter::emit_epilogue(
    const PipelineConfig& config,
    const PipelineStage& stage,
    u32 k_total,
    u32 k_tile) const {
    std::ostringstream os;
    u32 num_iters = k_total / k_tile;
    u32 prologue_iters = std::min(config.num_stages - 1, num_iters);
    u32 epilogue_iters = prologue_iters;

    os << "    // === Pipeline Epilogue ===\n";
    os << "    // Drain remaining " << epilogue_iters << " computes\n";
    os << "L_pipeline_epilogue:\n";

    for (u32 i = 0; i < epilogue_iters; ++i) {
        if (config.use_async_copy) {
            os << "    // Wait for stage " << (num_iters - epilogue_iters + i) << "\n";
            os << "    cp.async.wait_group " << (epilogue_iters - i - 1) << ";\n";
            os << "    bar.sync 0;\n";
        }
        os << "    // Compute on stage " << (num_iters - epilogue_iters + i) << "\n";
        for (auto& compute : stage.computes) {
            os << "    // " << compute.description << "\n";
        }
    }

    // Stores.
    os << "    // Store results\n";
    for (auto& store : stage.stores) {
        os << "    // Store " << store.tensor_name << " to global\n";
    }
    return os.str();
}

std::string SoftwarePipelineEmitter::emit_pipelined_loop(
    const PipelineConfig& config,
    const PipelineStage& stage,
    u32 k_total,
    u32 k_tile) const {
    std::ostringstream os;
    os << emit_prologue(config, stage, k_total, k_tile);
    os << emit_steady_state(config, stage, k_total, k_tile);
    os << emit_epilogue(config, stage, k_total, k_tile);
    return os.str();
}

PipelineOverlapEstimate estimate_pipeline_overlap(
    double compute_sec,
    double memory_sec,
    u32 num_stages,
    u32 num_warps,
    u32 shared_mem_available,
    u32 shared_mem_per_stage) {
    PipelineOverlapEstimate est;
    est.compute_sec = compute_sec;
    est.memory_sec = memory_sec;

    if (num_stages <= 1) {
        // No pipelining: max(compute, memory)
        est.overlapped_sec = std::max(compute_sec, memory_sec);
        est.overlap_factor = 0.0;
        return est;
    }

    // With N stages, compute and memory overlap.
    // The overlap factor depends on:
    //   1. The ratio of compute to memory time
    //   2. The number of stages (more stages = better overlap)
    //   3. Shared memory constraints (limited buffers = limited overlap)
    //   4. Register pressure (more warps = better latency hiding but more registers)

    // Maximum achievable overlap: max(C, M) / max(C/N, M/N) = N * max(C,M) / max(C,M) = N
    // But we can't exceed the total work: total = C + M, and overlapped >= max(C, M)
    // So overlap_factor = (C + M - overlapped) / (C + M - max(C, M))
    //                    = (min(C, M)) / overlapped (approximately)

    // Simple model: with N stages, overlapped ≈ max(C, M) + min(C, M) / N
    // This gives: overlap_factor = (min(C,M) - min(C,M)/N) / min(C,M) = 1 - 1/N
    double min_time = std::min(compute_sec, memory_sec);
    double max_time = std::max(compute_sec, memory_sec);

    // Adjust for shared memory constraints: if we can't fit all buffers,
    // we get fewer effective stages.
    u32 max_buffers = shared_mem_available / std::max(1u, shared_mem_per_stage);
    u32 effective_stages = std::min(num_stages, max_buffers);

    // Adjust for occupancy: more warps = better latency hiding.
    double occupancy_factor = std::min(1.0, 1.0 + 0.1 * (num_warps - 1));

    est.overlapped_sec = max_time + min_time / (effective_stages * occupancy_factor);
    est.overlap_factor = 1.0 - 1.0 / effective_stages;

    return est;
}

} // namespace cg
