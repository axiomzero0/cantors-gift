// ir/effects.hpp - side-effect descriptions for IR operations
//
// Pure tensor operations can be freely reordered, CSE'd, and duplicated. Ops
// with effects (scatter, in-place updates, RNG, collective communication,
// device transfers, synchronization) must be treated more carefully.
//
// Every Operation carries an EffectSet. The scheduler / DCE / CSE consult
// this set before moving or deleting an op.
#pragma once

#include "cg/core/util.hpp"

#include <cstdint>

namespace cg {

enum class EffectKind : u16 {
    None        = 0,
    MemoryRead  = 1u << 0,
    MemoryWrite = 1u << 1,
    Allocate    = 1u << 2,
    Free        = 1u << 3,
    Synchronize = 1u << 4,
    DeviceTransfer = 1u << 5,
    Random      = 1u << 6,
    HasSideEffect = 1u << 7,    // catch-all for "cannot be moved"
};

class EffectSet {
public:
    constexpr EffectSet() = default;
    constexpr explicit EffectSet(u16 bits) : bits_(bits) {}

    constexpr bool has(EffectKind k) const {
        return (bits_ & static_cast<u16>(k)) != 0;
    }
    constexpr void add(EffectKind k) {
        bits_ |= static_cast<u16>(k);
    }
    constexpr void remove(EffectKind k) {
        bits_ &= ~static_cast<u16>(k);
    }
    constexpr bool is_pure() const {
        return bits_ == 0;
    }
    constexpr bool may_write() const {
        return has(EffectKind::MemoryWrite) || has(EffectKind::Allocate) ||
               has(EffectKind::Free) || has(EffectKind::HasSideEffect);
    }
    constexpr bool may_read() const {
        return has(EffectKind::MemoryRead);
    }
    constexpr u16 bits() const { return bits_; }

    static constexpr EffectSet pure()         { return EffectSet(0); }
    static constexpr EffectSet read()         { return EffectSet(static_cast<u16>(EffectKind::MemoryRead)); }
    static constexpr EffectSet write()        { return EffectSet(static_cast<u16>(EffectKind::MemoryWrite)); }
    static constexpr EffectSet read_write()   { return EffectSet(static_cast<u16>(EffectKind::MemoryRead) | static_cast<u16>(EffectKind::MemoryWrite)); }
    static constexpr EffectSet random()       { return EffectSet(static_cast<u16>(EffectKind::Random) | static_cast<u16>(EffectKind::HasSideEffect)); }
    static constexpr EffectSet sync()         { return EffectSet(static_cast<u16>(EffectKind::Synchronize)); }
    static constexpr EffectSet device_xfer()  { return EffectSet(static_cast<u16>(EffectKind::DeviceTransfer)); }

private:
    u16 bits_ = 0;
};

} // namespace cg
