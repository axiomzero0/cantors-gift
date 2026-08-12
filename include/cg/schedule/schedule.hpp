// schedule/schedule.hpp - schedule representation
//
// A Schedule is a sequence of transformations applied to a Tensor IR
// computation: tiling, splitting, interchanging, vectorizing, parallelizing,
// unrolling, and caching operands to a memory space.
//
// A Schedule is immutable once finalized. The autotuner enumerates many
// schedules for the same Tensor IR program, compiles each, and benchmarks.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/layout/layout.hpp"
#include "cg/schedule/domain.hpp"
#include "cg/shape/dim_expr.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cg {

enum class TransformKind : u8 {
    Split,         // split a dim by a factor -> outer, inner
    Tile,          // multi-dim split
    Interchange,   // swap two dims
    Fuse,          // fuse two dims
    Vectorize,     // vectorize a dim by a factor
    Parallelize,   // mark a dim parallel
    Unroll,        // unroll a dim by a factor
    Cache,         // cache an operand into a memory space
    Pipeline,      // software-pipeline a stage
    Prefetch,      // prefetch a tile
    Bind,          // bind a dim to a hardware axis (warp, thread, ...)
};

struct Transform {
    TransformKind kind;
    std::string dim;       // dim name (or "i0", "i1" ...)
    i64 factor = 0;
    i64 factor2 = 0;       // second factor for tile / split
    std::string target;    // for cache: operand name; for bind: hw axis name
    MemorySpace mem = MemorySpace::Generic;

    std::string to_string() const;
};

class Schedule {
public:
    Schedule() = default;

    void add(Transform t) { transforms_.push_back(std::move(t)); }
    const std::vector<Transform>& transforms() const { return transforms_; }

    // Hashable identity for cache lookup.
    u64 hash() const;
    bool operator==(const Schedule& o) const;

private:
    std::vector<Transform> transforms_;
};

// ScheduleSpace: an enumerable collection of candidate schedules.
class ScheduleSpace {
public:
    ScheduleSpace() = default;

    void add(Schedule s) { schedules_.push_back(std::move(s)); }
    const std::vector<Schedule>& schedules() const { return schedules_; }
    usize size() const { return schedules_.size(); }

    // Generate a basic grid of tile sizes for a 2D matmul-like op.
    static ScheduleSpace grid_matmul(
        std::vector<i64> m_tiles,
        std::vector<i64> n_tiles,
        std::vector<i64> k_tiles,
        std::vector<i64> vector_widths);

private:
    std::vector<Schedule> schedules_;
};

} // namespace cg
