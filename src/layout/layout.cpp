// layout/layout.cpp - layout implementation
#include "cg/layout/layout.hpp"

#include <algorithm>
#include <numeric>

namespace cg {

std::string_view memory_space_name(MemorySpace ms) {
    switch (ms) {
        case MemorySpace::Generic:    return "generic";
        case MemorySpace::Shared:     return "shared";
        case MemorySpace::Constant:   return "constant";
        case MemorySpace::Local:      return "local";
        case MemorySpace::GlobalHost: return "host";
    }
    return "?";
}

namespace {

u64 numel(const Shape& s) {
    u64 n = 1;
    for (auto& d : s) {
        if (!d->is_constant()) return 0;
        n *= static_cast<u64>(d->value);
    }
    return n;
}

// Verify that the strides describe a row-major contiguous layout.
bool strides_are_row_major(const Shape& shape, Span<const i64> strides) {
    if (shape.rank() != strides.size()) return false;
    i64 expected = 1;
    for (usize i = shape.rank(); i-- > 0;) {
        if (!shape[i]->is_constant()) return false;
        if (strides[i] != expected) return false;
        expected *= shape[i]->value;
    }
    return true;
}

} // namespace

LayoutPtr Layout::make_strided(Shape shape, SmallVector<i64> strides) {
    auto l = std::make_shared<Layout>();
    l->kind = LayoutKind::Strided;
    l->shape = std::move(shape);
    l->strides = std::move(strides);
    return l;
}

LayoutPtr Layout::make_row_major(Shape shape) {
    auto strides = row_major_strides(shape);
    return make_strided(std::move(shape), std::move(strides));
}

LayoutPtr Layout::make_col_major(Shape shape) {
    auto strides = col_major_strides(shape);
    return make_strided(std::move(shape), std::move(strides));
}

LayoutPtr Layout::make_broadcast(LayoutPtr base, Shape new_shape,
                                 SmallVector<i32> broadcast_axes) {
    auto l = std::make_shared<Layout>();
    l->kind = LayoutKind::Broadcast;
    l->shape = std::move(new_shape);
    l->outer = std::move(base);
    l->permutation = std::move(broadcast_axes);
    return l;
}

LayoutPtr Layout::make_transpose(LayoutPtr base, SmallVector<i32> perm) {
    auto l = std::make_shared<Layout>();
    l->kind = LayoutKind::Transpose;
    // The output shape is the permutation of the base shape.
    Shape out;
    out.dims().reserve(base->shape.rank());
    for (i32 p : perm) out.dims().push_back(base->shape[p]);
    l->shape = std::move(out);
    l->outer = std::move(base);
    l->permutation = std::move(perm);
    return l;
}

LayoutPtr Layout::make_reshape(LayoutPtr base, Shape target) {
    auto l = std::make_shared<Layout>();
    l->kind = LayoutKind::Reshape;
    l->shape = base->shape;
    l->target_shape = std::move(target);
    l->outer = std::move(base);
    return l;
}

LayoutPtr Layout::make_slice(LayoutPtr base,
                             SmallVector<std::pair<i64, i64>> ranges) {
    auto l = std::make_shared<Layout>();
    l->kind = LayoutKind::Slice;
    Shape out;
    out.dims().reserve(base->shape.rank());
    for (usize i = 0; i < base->shape.rank(); ++i) {
        auto [begin, extent] = ranges[i];
        out.dims().push_back(DimExpr::make_constant(extent));
    }
    l->shape = std::move(out);
    l->outer = std::move(base);
    l->slice_ranges = std::move(ranges);
    return l;
}

LayoutPtr Layout::make_symbolic(Shape shape, std::string opaque_id) {
    auto l = std::make_shared<Layout>();
    l->kind = LayoutKind::Symbolic;
    l->shape = std::move(shape);
    l->opaque_id = std::move(opaque_id);
    return l;
}

u64 Layout::bytes(DType dtype) const {
    return numel(shape) * dtype_size(dtype);
}

bool Layout::is_row_major_contiguous() const {
    if (kind != LayoutKind::Strided) return false;
    return strides_are_row_major(shape, make_span(strides));
}

std::optional<u64> Layout::byte_offset(Span<const i64> indices, DType dtype) const {
    if (indices.size() != shape.rank()) return std::nullopt;
    switch (kind) {
        case LayoutKind::Strided: {
            i64 off = 0;
            for (usize i = 0; i < shape.rank(); ++i) {
                if (i >= strides.size()) return std::nullopt;
                off += indices[i] * strides[i];
            }
            return static_cast<u64>(off) * dtype_size(dtype);
        }
        case LayoutKind::Transpose: {
            // Output index i selects base index permutation[i].
            SmallVector<i64> mapped;
            mapped.reserve(shape.rank());
            for (usize i = 0; i < shape.rank(); ++i)
                mapped.push_back(indices[permutation[i]]);
            return outer->byte_offset(make_span(mapped), dtype);
        }
        case LayoutKind::Broadcast: {
            // permutation contains base-axes corresponding to output axes;
            // size-1 output dims contribute no offset.
            SmallVector<i64> mapped;
            for (i32 axis : permutation) {
                if (axis >= 0 && static_cast<usize>(axis) < indices.size())
                    mapped.push_back(indices[axis]);
                else
                    mapped.push_back(0);
            }
            return outer->byte_offset(make_span(mapped), dtype);
        }
        case LayoutKind::Reshape: {
            // Linearize the input index then delinearize to the base shape.
            if (!outer) return std::nullopt;
            u64 linear = 0;
            // Compute linear index in this layout's coordinate space.
            SmallVector<i64> strides_local = row_major_strides(shape);
            for (usize i = 0; i < shape.rank(); ++i)
                linear += static_cast<u64>(indices[i]) * static_cast<u64>(strides_local[i]);
            // Delinearize to the base shape.
            SmallVector<i64> mapped;
            auto base_strides = row_major_strides(outer->shape);
            for (usize i = 0; i < outer->shape.rank(); ++i) {
                mapped.push_back(static_cast<i64>(linear / static_cast<u64>(base_strides[i])));
                linear %= static_cast<u64>(base_strides[i]);
            }
            return outer->byte_offset(make_span(mapped), dtype);
        }
        case LayoutKind::Slice: {
            SmallVector<i64> mapped;
            for (usize i = 0; i < shape.rank(); ++i)
                mapped.push_back(indices[i] + slice_ranges[i].first);
            return outer->byte_offset(make_span(mapped), dtype);
        }
        case LayoutKind::Compose:
        case LayoutKind::Symbolic:
            return std::nullopt;
    }
    return std::nullopt;
}

bool Layout::structurally_equal(const Layout& other) const {
    if (this == &other) return true;
    if (kind != other.kind) return false;
    if (!(shape == other.shape)) return false;
    if (strides != other.strides) return false;
    if (permutation != other.permutation) return false;
    if (target_shape != other.target_shape) return false;
    if (slice_ranges != other.slice_ranges) return false;
    if (opaque_id != other.opaque_id) return false;
    if (outer && other.outer) {
        if (!outer->structurally_equal(*other.outer)) return false;
    } else if (outer || other.outer) {
        return false;
    }
    if (inner && other.inner) {
        if (!inner->structurally_equal(*other.inner)) return false;
    } else if (inner || other.inner) {
        return false;
    }
    return true;
}

std::size_t Layout::hash() const {
    std::size_t h = std::hash<u8>{}(static_cast<u8>(kind));
    for (auto& d : shape) hash_combine(h, d->hash());
    for (auto s : strides) hash_combine(h, std::hash<i64>{}(s));
    for (auto p : permutation) hash_combine(h, std::hash<i32>{}(p));
    for (auto& t : target_shape) hash_combine(h, t->hash());
    for (auto& r : slice_ranges) {
        hash_combine(h, std::hash<i64>{}(r.first));
        hash_combine(h, std::hash<i64>{}(r.second));
    }
    hash_combine(h, std::hash<std::string>{}(opaque_id));
    if (outer) hash_combine(h, outer->hash());
    if (inner) hash_combine(h, inner->hash());
    return h;
}

} // namespace cg
