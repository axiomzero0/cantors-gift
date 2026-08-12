// ir/value.hpp - SSA values
//
// A Value is an opaque handle to either:
//   - the result of an Operation (with a result index), or
//   - a BlockArgument.
//
// Values are immutable once created. Their type is fixed. Operations that
// produce multiple results emit one Value per result.
#pragma once

#include "cg/core/util.hpp"
#include "cg/ir/type.hpp"

#include <cstdint>

namespace cg {

class Operation;
class BlockArgument;

// A 32-bit value id is sufficient for any realistic module size; we
// deliberately do NOT use pointers so that IR can be relocated, serialized,
// and snapshot/diff'd cheaply.
using ValueId = u32;

class Value {
public:
    Value() = default;
    Value(TypePtr type, ValueId id)
        : type_(std::move(type)), id_(id) {}

    const TypePtr& type() const { return type_; }
    ValueId id() const { return id_; }

    bool is_null() const { return id_ == 0 && !type_; }
    explicit operator bool() const { return !is_null(); }

    bool operator==(const Value& o) const { return id_ == o.id_; }
    bool operator!=(const Value& o) const { return id_ != o.id_; }
    bool operator<(const Value& o)  const { return id_ < o.id_; }

    // Downcast helpers (return nullptr on mismatch).
    std::shared_ptr<const TensorType> as_tensor() const {
        if (type_ && type_->kind == TypeKind::Tensor)
            return std::static_pointer_cast<const TensorType>(type_);
        return nullptr;
    }
    std::shared_ptr<const ScalarType> as_scalar() const {
        if (type_ && type_->kind == TypeKind::Scalar)
            return std::static_pointer_cast<const ScalarType>(type_);
        return nullptr;
    }

private:
    TypePtr type_;
    ValueId id_ = 0;
};

} // namespace cg

namespace std {
template <>
struct hash<cg::Value> {
    std::size_t operator()(const cg::Value& v) const noexcept {
        return std::hash<cg::ValueId>{}(v.id());
    }
};
} // namespace std
