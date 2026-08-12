// ir/traits.hpp - operation traits
//
// Traits classify operations for generic optimization. They are queried
// via the OperationInfo registry (see ir/ops.hpp), which is the source of
// truth for "what does this op do?".
#pragma once

#include "cg/core/util.hpp"

#include <cstdint>

namespace cg {

enum class OpTrait : u32 {
    None             = 0,
    Pure             = 1u << 0,
    Commutative      = 1u << 1,
    Associative      = 1u << 2,
    Elementwise      = 1u << 3,
    Broadcastable    = 1u << 4,
    Reduction        = 1u << 5,
    MemoryRead       = 1u << 6,
    MemoryWrite      = 1u << 7,
    ShapePreserving  = 1u << 8,
    LayoutPreserving = 1u << 9,
    HasSideEffect    = 1u << 10,
    TensorCore       = 1u << 11,   // may use tensor-core / MMA instructions
};

class OpTraits {
public:
    constexpr OpTraits() = default;
    constexpr explicit OpTraits(u32 bits) : bits_(bits) {}

    constexpr bool has(OpTrait t) const {
        return (bits_ & static_cast<u32>(t)) != 0;
    }
    constexpr OpTraits with(OpTrait t) const {
        return OpTraits(bits_ | static_cast<u32>(t));
    }
    constexpr u32 bits() const { return bits_; }

private:
    u32 bits_ = 0;
};

} // namespace cg
