// schedule/schedule.cpp
#include "cg/schedule/schedule.hpp"

#include <sstream>

namespace cg {

namespace {
const char* transform_name(TransformKind k) {
    switch (k) {
        case TransformKind::Split:       return "split";
        case TransformKind::Tile:        return "tile";
        case TransformKind::Interchange: return "interchange";
        case TransformKind::Fuse:        return "fuse";
        case TransformKind::Vectorize:   return "vectorize";
        case TransformKind::Parallelize: return "parallelize";
        case TransformKind::Unroll:      return "unroll";
        case TransformKind::Cache:       return "cache";
        case TransformKind::Pipeline:    return "pipeline";
        case TransformKind::Prefetch:    return "prefetch";
        case TransformKind::Bind:        return "bind";
    }
    return "?";
}
}

std::string Transform::to_string() const {
    std::ostringstream os;
    os << transform_name(kind) << "(" << dim;
    if (factor) os << ", " << factor;
    if (factor2) os << ", " << factor2;
    if (!target.empty()) os << ", target=" << target;
    if (mem != MemorySpace::Generic) os << ", mem=" << memory_space_name(mem);
    os << ")";
    return os.str();
}

u64 Schedule::hash() const {
    u64 h = 0xdeadbeefULL;
    for (auto& t : transforms_) {
        hash_combine(h, std::hash<u8>{}(static_cast<u8>(t.kind)));
        hash_combine(h, std::hash<std::string>{}(t.dim));
        hash_combine(h, std::hash<i64>{}(t.factor));
        hash_combine(h, std::hash<i64>{}(t.factor2));
        hash_combine(h, std::hash<std::string>{}(t.target));
        hash_combine(h, std::hash<u8>{}(static_cast<u8>(t.mem)));
    }
    return h;
}

bool Schedule::operator==(const Schedule& o) const {
    if (transforms_.size() != o.transforms_.size()) return false;
    for (usize i = 0; i < transforms_.size(); ++i) {
        const auto& a = transforms_[i];
        const auto& b = o.transforms_[i];
        if (a.kind != b.kind) return false;
        if (a.dim != b.dim) return false;
        if (a.factor != b.factor) return false;
        if (a.factor2 != b.factor2) return false;
        if (a.target != b.target) return false;
        if (a.mem != b.mem) return false;
    }
    return true;
}

ScheduleSpace ScheduleSpace::grid_matmul(
    std::vector<i64> m_tiles,
    std::vector<i64> n_tiles,
    std::vector<i64> k_tiles,
    std::vector<i64> vector_widths) {
    ScheduleSpace out;
    for (i64 m : m_tiles) for (i64 n : n_tiles) for (i64 k : k_tiles) {
        for (i64 v : vector_widths) {
            Schedule s;
            s.add({TransformKind::Tile, "m", m, n, "", MemorySpace::Generic});
            s.add({TransformKind::Tile, "n", n, 0, "", MemorySpace::Generic});
            s.add({TransformKind::Tile, "k", k, 0, "", MemorySpace::Generic});
            s.add({TransformKind::Parallelize, "m", 0, 0, "", MemorySpace::Generic});
            s.add({TransformKind::Parallelize, "n", 0, 0, "", MemorySpace::Generic});
            s.add({TransformKind::Vectorize, "n_inner", v, 0, "", MemorySpace::Generic});
            s.add({TransformKind::Cache, "a", 0, 0, "A", MemorySpace::Shared});
            s.add({TransformKind::Cache, "b", 0, 0, "B", MemorySpace::Shared});
            out.add(std::move(s));
        }
    }
    return out;
}

} // namespace cg
