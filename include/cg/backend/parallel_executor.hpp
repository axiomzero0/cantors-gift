// backend/parallel_executor.hpp - multi-threaded JIT execution via std::thread
//
// Takes a JIT'd function pointer and splits the work across N threads.
// Each thread processes a contiguous chunk of the output array.
//
// The JIT function signature is:
//   void fn(void* a, void* b, void* c, int64_t count)
// where count = number of 4-float groups to process.
//
// The executor:
//   1. Splits the total element count into N chunks
//   2. Spawns N std::threads, each calling fn with adjusted pointers
//   3. Joins all threads
//   4. Returns when all threads are done
//
// Uses std::thread (pthreads on Linux). No custom threading engine —
// just the standard library, which is battle-tested.
#pragma once

#include "cg/backend/jit.hpp"
#include "cg/core/util.hpp"

#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace cg {

// Function type for 3-argument JIT kernels with a count parameter.
// void fn(float* a, float* b, float* c, int64_t count)
// count = number of 4-float groups (each group = 16 bytes)
using Kernel3Fn = void (*)(float*, float*, float*, int64_t);

class ParallelExecutor {
public:
    explicit ParallelExecutor(u32 num_threads = 0)
        : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {}

    u32 num_threads() const { return num_threads_; }

    // Execute a 3-argument kernel in parallel.
    // Splits `total_elements` into chunks, each thread processes a contiguous
    // range. The kernel must process `count` groups of 4 floats.
    //
    // a, b, c must be contiguous arrays of `total_elements` floats each.
    // The kernel signature is void(float* a, float* b, float* c, int64_t count)
    // where count = number of 4-float groups.
    void execute(Kernel3Fn fn, float* a, float* b, float* c,
                 u64 total_elements) const {
        if (total_elements == 0) return;

        // Each group is 4 floats. We need total_elements to be divisible by
        // (4 * num_threads) for perfect splitting, but we handle remainder.
        u64 total_groups = total_elements / 4;
        u64 remainder_floats = total_elements % 4;  // leftover floats

        // Split groups across threads.
        u64 groups_per_thread = total_groups / num_threads_;
        u64 extra_groups = total_groups % num_threads_;

        std::vector<std::thread> threads;
        u64 group_offset = 0;

        for (u32 t = 0; t < num_threads_; ++t) {
            u64 chunk_groups = groups_per_thread + (t < extra_groups ? 1 : 0);
            if (chunk_groups == 0) continue;

            u64 float_offset = group_offset * 4;
            float* a_chunk = a + float_offset;
            float* b_chunk = b + float_offset;
            float* c_chunk = c + float_offset;

            threads.emplace_back([fn, a_chunk, b_chunk, c_chunk, chunk_groups]() {
                fn(a_chunk, b_chunk, c_chunk, static_cast<int64_t>(chunk_groups));
            });

            group_offset += chunk_groups;
        }

        // Handle remainder floats (not a multiple of 4) on the last thread.
        // For now, process them sequentially after the threads join.
        for (auto& t : threads) t.join();

        if (remainder_floats > 0) {
            // Process the remaining floats with a scalar loop.
            u64 offset = total_groups * 4;
            for (u64 i = 0; i < remainder_floats; ++i) {
                c[offset + i] = a[offset + i] + b[offset + i];
            }
        }
    }

    // Execute with a JIT entry point (convenience wrapper).
    void execute(uintptr_t jit_entry, float* a, float* b, float* c,
                 u64 total_elements) const {
        auto fn = reinterpret_cast<Kernel3Fn>(jit_entry);
        execute(fn, a, b, c, total_elements);
    }

private:
    u32 num_threads_;
};

} // namespace cg
