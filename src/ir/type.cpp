// ir/type.cpp - implementation of the type system
#include "cg/ir/type.hpp"

#include <sstream>

namespace cg {

std::string DeviceId::to_string() const {
    const char* k = "?";
    switch (kind) {
        case Kind::CPU:   k = "cpu"; break;
        case Kind::CUDA:  k = "cuda"; break;
        case Kind::ROCM:  k = "rocm"; break;
        case Kind::METAL: k = "metal"; break;
    }
    return std::string(k) + ":" + std::to_string(index);
}

std::shared_ptr<const TensorType> TensorType::make(
    Shape shape, DType dtype, LayoutPtr layout,
    DeviceId device, MemorySpace ms) {
    auto t = std::make_shared<TensorType>();
    t->shape = std::move(shape);
    t->dtype = dtype;
    if (!layout) {
        // Default to row-major.
        t->layout = Layout::make_row_major(t->shape);
    } else {
        t->layout = std::move(layout);
    }
    t->device = device;
    t->memory_space = ms;
    return t;
}

std::string TensorType::to_string() const {
    std::ostringstream os;
    os << "Tensor<";
    os << "[";
    for (usize i = 0; i < shape.rank(); ++i) {
        if (i) os << ",";
        if (shape[i]->is_constant()) os << shape[i]->value;
        else if (shape[i]->is_symbol()) os << shape[i]->symbol.name;
        else os << "?";
    }
    os << "], ";
    os << dtype_name(dtype);
    os << ", ";
    if (layout) {
        switch (layout->kind) {
            case LayoutKind::Strided:
                os << "strided";
                if (layout->is_row_major_contiguous()) os << "(row_major)";
                break;
            case LayoutKind::Broadcast: os << "broadcast"; break;
            case LayoutKind::Compose:   os << "compose"; break;
            case LayoutKind::Transpose: os << "transpose"; break;
            case LayoutKind::Reshape:   os << "reshape"; break;
            case LayoutKind::Slice:     os << "slice"; break;
            case LayoutKind::Symbolic:  os << "symbolic:" << layout->opaque_id; break;
        }
    } else {
        os << "no_layout";
    }
    os << ", " << device.to_string();
    os << ", " << memory_space_name(memory_space);
    os << ">";
    return os.str();
}

bool TensorType::structurally_equal(const Type& other) const {
    if (other.kind != TypeKind::Tensor) return false;
    const auto& o = static_cast<const TensorType&>(other);
    if (dtype != o.dtype) return false;
    if (device != o.device) return false;
    if (memory_space != o.memory_space) return false;
    if (!(shape == o.shape)) return false;
    if (layout && o.layout) return layout->structurally_equal(*o.layout);
    return !layout && !o.layout;
}

std::size_t TensorType::hash() const {
    std::size_t h = std::hash<u8>{}(static_cast<u8>(TypeKind::Tensor));
    hash_combine(h, std::hash<u8>{}(static_cast<u8>(dtype)));
    hash_combine(h, std::hash<u8>{}(static_cast<u8>(device.kind)));
    hash_combine(h, std::hash<u32>{}(device.index));
    hash_combine(h, std::hash<u8>{}(static_cast<u8>(memory_space)));
    for (auto& d : shape) hash_combine(h, d->hash());
    if (layout) hash_combine(h, layout->hash());
    return h;
}

std::string ScalarType::to_string() const {
    return std::string("Scalar<") + std::string(dtype_name(dtype)) + ">";
}

bool ScalarType::structurally_equal(const Type& other) const {
    if (other.kind != TypeKind::Scalar) return false;
    return dtype == static_cast<const ScalarType&>(other).dtype;
}

std::size_t ScalarType::hash() const {
    std::size_t h = std::hash<u8>{}(static_cast<u8>(TypeKind::Scalar));
    hash_combine(h, std::hash<u8>{}(static_cast<u8>(dtype)));
    return h;
}

std::string FunctionType::to_string() const {
    std::ostringstream os;
    os << "(";
    for (usize i = 0; i < operands.size(); ++i) {
        if (i) os << ", ";
        os << operands[i]->to_string();
    }
    os << ") -> (";
    for (usize i = 0; i < results.size(); ++i) {
        if (i) os << ", ";
        os << results[i]->to_string();
    }
    os << ")";
    return os.str();
}

bool FunctionType::structurally_equal(const Type& other) const {
    if (other.kind != TypeKind::Function) return false;
    const auto& o = static_cast<const FunctionType&>(other);
    if (operands.size() != o.operands.size() ||
        results.size() != o.results.size()) return false;
    for (usize i = 0; i < operands.size(); ++i)
        if (!operands[i]->structurally_equal(*o.operands[i])) return false;
    for (usize i = 0; i < results.size(); ++i)
        if (!results[i]->structurally_equal(*o.results[i])) return false;
    return true;
}

std::size_t FunctionType::hash() const {
    std::size_t h = std::hash<u8>{}(static_cast<u8>(TypeKind::Function));
    for (auto& t : operands) hash_combine(h, t->hash());
    for (auto& t : results)  hash_combine(h, t->hash());
    return h;
}

} // namespace cg
