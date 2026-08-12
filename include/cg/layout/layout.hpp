// layout/layout.hpp - composable layout algebra
//
// A Layout is a *function* from logical indices to byte offsets:
//
//     address(i0, i1, ..., in) -> u64
//
// NOT merely a `std::vector<int64_t> strides`. By treating the layout as a
// composable function we can symbolically fold sequences of
// reshape/transpose/broadcast/cast operations and eliminate entire tensor
// copies from the compiled program.
//
// Layouts are immutable and interned for cheap structural comparison.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/shape/dim_expr.hpp"
#include "cg/core/util.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cg {

// Memory spaces a tensor (or tile thereof) may reside in.
enum class MemorySpace : u8 {
    Generic,        // device global memory
    Shared,         // GPU shared memory
    Constant,       // constant cache / constant memory
    Local,          // thread-local / register
    GlobalHost,     // host-pinned
};

std::string_view memory_space_name(MemorySpace ms);

// Forward
class Layout;
using LayoutPtr = std::shared_ptr<const Layout>;

enum class LayoutKind : u8 {
    Strided,         // explicit strides per dim
    Broadcast,       // dim doesn't advance the address
    Compose,         // outer.layout applied after inner
    Transpose,       // permute dims
    Reshape,         // remap dim order via shape product
    Slice,           // sub-range along one or more dims
    Symbolic,        // backend-provided opaque layout (e.g. vendor tile layout)
};

// Base layout node. Layouts are immutable and interned.
class Layout {
public:
    LayoutKind kind;

    // The shape *this layout operates over*. The address function takes
    // indices of this rank.
    Shape shape;

    // Strided: per-dim stride in *elements* (not bytes).
    SmallVector<i64> strides;

    // Compose: outer ∘ inner
    LayoutPtr outer;
    LayoutPtr inner;

    // Transpose: permutation[i] is the source dim for output dim i.
    SmallVector<i32> permutation;

    // Reshape: target shape (must have same numel as `shape`).
    Shape target_shape;

    // Slice: for each dim, (begin, extent).
    SmallVector<std::pair<i64, i64>> slice_ranges;

    // Symbolic: opaque string identifying the layout (e.g. "cutlass::RowMajor").
    std::string opaque_id;

    // ---- Factory helpers ---------------------------------------------------
    static LayoutPtr make_strided(Shape shape, SmallVector<i64> strides);
    static LayoutPtr make_row_major(Shape shape);
    static LayoutPtr make_col_major(Shape shape);
    static LayoutPtr make_broadcast(LayoutPtr base, Shape new_shape,
                                    SmallVector<i32> broadcast_axes);
    static LayoutPtr make_transpose(LayoutPtr base, SmallVector<i32> perm);
    static LayoutPtr make_reshape(LayoutPtr base, Shape target);
    static LayoutPtr make_slice(LayoutPtr base,
                                SmallVector<std::pair<i64, i64>> ranges);
    static LayoutPtr make_symbolic(Shape shape, std::string opaque_id);

    // Total number of bytes described by this layout for `dtype`.
    u64 bytes(DType dtype) const;

    // True iff the layout is "contiguous" in row-major order over its shape.
    bool is_row_major_contiguous() const;

    // Returns the byte offset of the element at `indices`.
    // On success, returns the byte offset. On failure (e.g. the layout cannot
    // be evaluated statically because it has symbolic strides), returns
    // nullopt.
    std::optional<u64> byte_offset(Span<const i64> indices, DType dtype) const;

    // Structural equality.
    bool structurally_equal(const Layout& other) const;
    std::size_t hash() const;
};

// ---------------------------------------------------------------------------
// Helper: compute row-major strides from a (constant) shape.
// ---------------------------------------------------------------------------
inline SmallVector<i64> row_major_strides(const Shape& s) {
    SmallVector<i64> out(s.rank(), 1);
    if (s.rank() == 0) return out;
    for (usize i = s.rank() - 1; i-- > 0;)
        out[i] = out[i + 1] * s[i + 1]->value;
    return out;
}

inline SmallVector<i64> col_major_strides(const Shape& s) {
    SmallVector<i64> out(s.rank(), 1);
    for (usize i = 1; i < s.rank(); ++i)
        out[i] = out[i - 1] * s[i - 1]->value;
    return out;
}

} // namespace cg
