// ir/type.hpp - the type system of Tensor IR
//
// Types are interned, immutable, and form a sealed class hierarchy:
//
//   Type (abstract)
//     TensorType      - shape + dtype + layout + device + memory space
//     ScalarType      - dtype only (used for scalar args / indices)
//     VoidType        - the unit type
//     FunctionType    - operand types -> result types
//
// TensorType is the workhorse. It carries every piece of metadata that
// generic, layout-agnostic optimizations need to know about a value.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/layout/layout.hpp"
#include "cg/shape/dim_expr.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cg {

enum class TypeKind : u8 {
    Tensor,
    Scalar,
    Void,
    Function,
};

class Type;
using TypePtr = std::shared_ptr<const Type>;

// Identifies a (logical) device. We do not bind to a physical device at IR
// construction time; the runtime resolves `DeviceId` to a concrete backend.
struct DeviceId {
    enum class Kind : u8 { CPU, CUDA, ROCM, METAL };
    Kind kind;
    u32 index; // 0-based device index within the kind

    bool operator==(const DeviceId& o) const {
        return kind == o.kind && index == o.index;
    }
    bool operator!=(const DeviceId& o) const { return !(*this == o); }

    static DeviceId cpu(u32 i = 0)    { return {Kind::CPU, i}; }
    static DeviceId cuda(u32 i = 0)   { return {Kind::CUDA, i}; }
    static DeviceId rocm(u32 i = 0)   { return {Kind::ROCM, i}; }
    static DeviceId metal(u32 i = 0)  { return {Kind::METAL, i}; }

    std::string to_string() const;
};

class Type {
public:
    TypeKind kind;
    virtual ~Type() = default;
    virtual std::string to_string() const = 0;
    virtual bool structurally_equal(const Type& other) const = 0;
    virtual std::size_t hash() const = 0;

protected:
    explicit Type(TypeKind k) : kind(k) {}
};

class TensorType : public Type {
public:
    Shape shape;
    DType dtype = DType::F32;
    LayoutPtr layout;
    DeviceId device = DeviceId::cpu();
    MemorySpace memory_space = MemorySpace::Generic;

    TensorType() : Type(TypeKind::Tensor) {}

    static std::shared_ptr<const TensorType> make(
        Shape shape, DType dtype,
        LayoutPtr layout = nullptr,
        DeviceId device = DeviceId::cpu(),
        MemorySpace ms = MemorySpace::Generic);

    std::string to_string() const override;
    bool structurally_equal(const Type& other) const override;
    std::size_t hash() const override;
};

class ScalarType : public Type {
public:
    DType dtype;

    explicit ScalarType(DType dt) : Type(TypeKind::Scalar), dtype(dt) {}

    static std::shared_ptr<const ScalarType> make(DType dt) {
        return std::make_shared<ScalarType>(dt);
    }

    std::string to_string() const override;
    bool structurally_equal(const Type& other) const override;
    std::size_t hash() const override;
};

class VoidType : public Type {
public:
    VoidType() : Type(TypeKind::Void) {}
    static std::shared_ptr<const VoidType> make() {
        return std::make_shared<VoidType>();
    }
    std::string to_string() const override { return "void"; }
    bool structurally_equal(const Type& other) const override {
        return other.kind == TypeKind::Void;
    }
    std::size_t hash() const override { return 0; }
};

class FunctionType : public Type {
public:
    std::vector<TypePtr> operands;
    std::vector<TypePtr> results;

    FunctionType() : Type(TypeKind::Function) {}

    static std::shared_ptr<const FunctionType> make(
        std::vector<TypePtr> operands,
        std::vector<TypePtr> results) {
        auto f = std::make_shared<FunctionType>();
        f->operands = std::move(operands);
        f->results  = std::move(results);
        return f;
    }

    std::string to_string() const override;
    bool structurally_equal(const Type& other) const override;
    std::size_t hash() const override;
};

// Convenience: tensor type with default row-major layout.
inline std::shared_ptr<const TensorType> make_tensor_type(
    Shape shape, DType dtype,
    DeviceId device = DeviceId::cpu(),
    MemorySpace ms = MemorySpace::Generic) {
    return TensorType::make(std::move(shape), dtype, nullptr, device, ms);
}

} // namespace cg
