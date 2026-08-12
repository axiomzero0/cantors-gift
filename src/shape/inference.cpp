// shape/inference.cpp - implementation of shape inference
#include "cg/shape/inference.hpp"

#include <algorithm>
#include <set>

namespace cg {

namespace {

Shape broadcast_shapes(const Shape& a, const Shape& b, std::string& err) {
    usize ra = a.rank(), rb = b.rank();
    usize r  = std::max(ra, rb);
    Shape out;
    out.dims().reserve(r);
    for (usize i = 0; i < r; ++i) {
        DimExprPtr da = i < r - ra ? DimExpr::make_constant(1) : a[i - (r - ra)];
        DimExprPtr db = i < r - rb ? DimExpr::make_constant(1) : b[i - (r - rb)];
        if (da->is_constant(1)) {
            out.dims().push_back(db);
        } else if (db->is_constant(1)) {
            out.dims().push_back(da);
        } else if (da->structurally_equal(*db)) {
            out.dims().push_back(da);
        } else {
            err = "incompatible broadcast dimensions";
            return {};
        }
    }
    return out;
}

} // namespace

InferResult infer_elementwise_binary(const Shape& a, const Shape& b) {
    InferResult r;
    std::string err;
    auto out = broadcast_shapes(a, b, err);
    if (!err.empty()) { r.message = err; return r; }
    r.ok = true;
    r.shape = std::move(out);
    return r;
}

InferResult infer_reduction(const Shape& a, Span<const i32> axes, bool keep_dims) {
    InferResult r;
    std::set<i32> unique_axes;
    for (i32 ax : axes) {
        if (ax < 0) ax += static_cast<i32>(a.rank());
        if (ax < 0 || static_cast<usize>(ax) >= a.rank()) {
            r.message = "reduction axis out of range";
            return r;
        }
        unique_axes.insert(ax);
    }
    Shape out;
    out.dims().reserve(a.rank());
    for (usize i = 0; i < a.rank(); ++i) {
        if (unique_axes.count(static_cast<i32>(i))) {
            if (keep_dims) out.dims().push_back(DimExpr::make_constant(1));
        } else {
            out.dims().push_back(a[i]);
        }
    }
    r.ok = true;
    r.shape = std::move(out);
    return r;
}

InferResult infer_broadcast(const Shape& a, const Shape& to) {
    InferResult r;
    if (a.rank() > to.rank()) {
        r.message = "cannot broadcast: input has more dims than target";
        return r;
    }
    usize pad = to.rank() - a.rank();
    for (usize i = 0; i < to.rank(); ++i) {
        if (i < pad) {
            if (!to[i]->is_constant(1)) {
                r.message = "broadcast dimension must be 1 for missing axes";
                return r;
            }
            continue;
        }
        DimExprPtr ai = a[i - pad];
        if (ai->is_constant(1)) continue;
        if (!ai->structurally_equal(*to[i])) {
            r.message = "broadcast shape mismatch";
            return r;
        }
    }
    r.ok = true;
    r.shape = to;
    return r;
}

InferResult infer_reshape(const Shape& a, const Shape& to) {
    InferResult r;
    // Handle -1 dimension.
    int dyn_idx = -1;
    u64 known = 1;
    for (usize i = 0; i < to.rank(); ++i) {
        if (to[i]->is_constant(-1)) {
            if (dyn_idx != -1) {
                r.message = "reshape: only one dimension may be -1";
                return r;
            }
            dyn_idx = static_cast<int>(i);
        } else if (to[i]->is_constant()) {
            known *= static_cast<u64>(to[i]->value);
            if (known == 0) {
                r.message = "reshape: target dimension is 0";
                return r;
            }
        } else {
            r.message = "reshape: target shape must be statically known";
            return r;
        }
    }
    u64 src = a.num_elements();
    if (src == 0) {
        r.message = "reshape: source shape has dynamic dimensions";
        return r;
    }
    if (dyn_idx == -1) {
        if (src != known) {
            r.message = "reshape: element count mismatch";
            return r;
        }
    } else {
        if (src % known != 0 || known == 0) {
            r.message = "reshape: cannot infer -1 dimension evenly";
            return r;
        }
        Shape out = to;
        out.dims()[dyn_idx] = DimExpr::make_constant(static_cast<i64>(src / known));
        r.ok = true;
        r.shape = std::move(out);
        return r;
    }
    r.ok = true;
    r.shape = to;
    return r;
}

InferResult infer_transpose(const Shape& a, Span<const i32> perm) {
    InferResult r;
    if (perm.size() != a.rank()) {
        r.message = "transpose: permutation rank mismatch";
        return r;
    }
    std::vector<bool> seen(a.rank(), false);
    for (i32 p : perm) {
        if (p < 0 || static_cast<usize>(p) >= a.rank() || seen[p]) {
            r.message = "transpose: invalid permutation";
            return r;
        }
        seen[p] = true;
    }
    Shape out;
    out.dims().reserve(a.rank());
    for (i32 p : perm) out.dims().push_back(a[p]);
    r.ok = true;
    r.shape = std::move(out);
    return r;
}

InferResult infer_matmul(const Shape& a, const Shape& b) {
    InferResult r;
    if (a.rank() < 2 || b.rank() < 2) {
        r.message = "matmul: operands must have rank >= 2";
        return r;
    }
    DimExprPtr ak = a[a.rank() - 1];
    DimExprPtr bk = b[b.rank() - 2];
    if (!ak->structurally_equal(*bk)) {
        // The compiler may still prove equality via the constraint solver
        // at a later phase; here we are syntactic.
        r.message = "matmul: inner dimensions do not syntactically match";
        return r;
    }
    // Batch broadcast
    Shape aBatch, bBatch;
    for (usize i = 0; i + 2 < a.rank(); ++i) aBatch.dims().push_back(a[i]);
    for (usize i = 0; i + 2 < b.rank(); ++i) bBatch.dims().push_back(b[i]);
    std::string err;
    Shape batch = broadcast_shapes(aBatch, bBatch, err);
    if (!err.empty()) { r.message = err; return r; }

    Shape out;
    out.dims().reserve(batch.rank() + 2);
    for (auto& d : batch) out.dims().push_back(d);
    out.dims().push_back(a[a.rank() - 2]); // M
    out.dims().push_back(b[b.rank() - 1]); // N
    r.ok = true;
    r.shape = std::move(out);
    return r;
}

} // namespace cg
