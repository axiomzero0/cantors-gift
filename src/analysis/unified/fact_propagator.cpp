// analysis/unified/fact_propagator.cpp - implementations of all propagators.
//
// Each propagator reads facts from the FactStore, derives new facts from
// the IR + existing facts, and writes them back with provenance + confidence.
// All propagators are idempotent — running them twice with the same inputs
// produces the same output, which makes fixed-point iteration safe.
#include "cg/analysis/unified/fact_propagator.hpp"

#include "cg/ir/ops.hpp"

#include <algorithm>
#include <cmath>

namespace cg {

namespace {

// Helper: count static numel from a Shape.
u64 numel_of(const Shape& s) {
    u64 n = 1;
    for (auto& d : s) {
        if (!d->is_constant()) return 0;
        n *= static_cast<u64>(d->value);
    }
    return n;
}

// Helper: extract shape dimensions as a vector<Dimension>.
std::vector<Dimension> dims_from_shape(const Shape& s) {
    std::vector<Dimension> out;
    out.reserve(s.rank());
    for (auto& d : s) {
        Dimension dim;
        dim.bound.exact = d;
        dim.bound.lower = d->is_constant() ? d : DimExpr::make_constant(1);
        // No upper bound by default (could be infinite).
        out.push_back(dim);
    }
    return out;
}

} // namespace

// ===========================================================================
// ShapePropagator
// ===========================================================================
u32 ShapePropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    for (auto& f : m.functions()) {
        for (auto& arg : f->args()) {
            auto t = arg.as_tensor();
            if (!t) continue;
            auto& tf = store.facts_for(arg.id());
            std::vector<Dimension> dims = dims_from_shape(t->shape);
            auto new_shape = Fact<std::vector<Dimension>>::make(
                dims, Confidence::Proven,
                Provenance("ShapeInference", 0));
            if (tf.shape.join(new_shape)) {
                ++discovered;
                store.notify_fact_changed(arg.id(), id());
            }
            auto new_rank = Fact<u32>::make(
                static_cast<u32>(t->shape.rank()),
                Confidence::Proven,
                Provenance("ShapeInference", 0));
            if (tf.rank.join(new_rank)) {
                ++discovered;
                store.notify_fact_changed(arg.id(), id());
            }
        }
        for (auto& op : *f->entry()) {
            for (auto& r : op.results) {
                auto t = r.as_tensor();
                if (!t) continue;
                auto& tf = store.facts_for(r.id());
                std::vector<Dimension> dims = dims_from_shape(t->shape);
                auto new_shape = Fact<std::vector<Dimension>>::make(
                    std::move(dims), Confidence::Proven,
                    Provenance("ShapeInference", op.id));
                if (tf.shape.join(new_shape)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
                auto new_rank = Fact<u32>::make(
                    static_cast<u32>(t->shape.rank()),
                    Confidence::Proven,
                    Provenance("ShapeInference", op.id));
                if (tf.rank.join(new_rank)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
            }
        }
    }
    return discovered;
}

// ===========================================================================
// LayoutPropagator
// ===========================================================================
u32 LayoutPropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    for (auto& f : m.functions()) {
        for (auto& arg : f->args()) {
            auto t = arg.as_tensor();
            if (!t) continue;
            auto& tf = store.facts_for(arg.id());
            if (t->layout) {
                auto new_layout = Fact<LayoutPtr>::make(
                    t->layout, Confidence::Proven,
                    Provenance("LayoutInference", 0));
                if (tf.layout.join(new_layout)) {
                    ++discovered;
                    store.notify_fact_changed(arg.id(), id());
                }
            }
            // Default row-major for input tensors.
            auto rm = Layout::make_row_major(t->shape);
            auto new_rmc = Fact<bool>::make(true, Confidence::Derived,
                                            Provenance("LayoutInference", 0));
            if (tf.is_row_major_contiguous.join(new_rmc)) {
                ++discovered;
                store.notify_fact_changed(arg.id(), id());
            }
            // Compute static strides if possible.
            if (t->shape.rank() > 0) {
                std::vector<DimExprPtr> strides(t->shape.rank());
                DimExprPtr acc = DimExpr::make_constant(1);
                for (isize i = static_cast<isize>(t->shape.rank()) - 1; i >= 0; --i) {
                    strides[i] = acc;
                    acc = DimExpr::make_mul(acc, t->shape[i]);
                }
                auto new_strides = Fact<std::vector<DimExprPtr>>::make(
                    strides, Confidence::Derived,
                    Provenance("LayoutInference", 0));
                if (tf.strides.join(new_strides)) {
                    ++discovered;
                    store.notify_fact_changed(arg.id(), id());
                }
            }
        }
        for (auto& op : *f->entry()) {
            for (auto& r : op.results) {
                auto t = r.as_tensor();
                if (!t) continue;
                auto& tf = store.facts_for(r.id());
                if (t->layout) {
                    auto new_layout = Fact<LayoutPtr>::make(
                        t->layout, Confidence::Proven,
                        Provenance("LayoutInference", op.id));
                    if (tf.layout.join(new_layout)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
                // Stride propagation.
                if (t->shape.rank() > 0) {
                    std::vector<DimExprPtr> strides(t->shape.rank());
                    DimExprPtr acc = DimExpr::make_constant(1);
                    for (isize i = static_cast<isize>(t->shape.rank()) - 1; i >= 0; --i) {
                        strides[i] = acc;
                        acc = DimExpr::make_mul(acc, t->shape[i]);
                    }
                    auto new_strides = Fact<std::vector<DimExprPtr>>::make(
                        strides, Confidence::Derived,
                        Provenance("LayoutInference", op.id));
                    if (tf.strides.join(new_strides)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
            }
        }
    }
    return discovered;
}

// ===========================================================================
// ConstantPropagator
// ===========================================================================
u32 ConstantPropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto r = op.results[0];
            auto& tf = store.facts_for(r.id());

            // OP_CONSTANT: the result is a known constant.
            if (op.opcode == OP_CONSTANT) {
                // Try to extract the constant value from attributes.
                auto val_attr = op.attributes.get("value");
                double v = 0.0;
                if (val_attr) {
                    if (val_attr->kind == AttrKind::Integer) v = static_cast<double>(val_attr->integer);
                    else if (val_attr->kind == AttrKind::Float) v = val_attr->real;
                }
                auto new_cv = Fact<double>::make(v, Confidence::Proven,
                                                  Provenance("ConstantPropagator", op.id));
                if (tf.constant_value.join(new_cv)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
                auto new_known = Fact<bool>::make(true, Confidence::Proven,
                                                   Provenance("ConstantPropagator", op.id));
                if (tf.constant_value_known.join(new_known)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
                // Set Constant + Zero/One properties.
                TensorProperty props = TensorProperty::Constant;
                if (v == 0.0) props = props | TensorProperty::Zero;
                if (v == 1.0) props = props | TensorProperty::One;
                auto new_props = Fact<TensorProperty>::make(
                    props, Confidence::Proven,
                    Provenance("ConstantPropagator", op.id));
                if (tf.properties.join(new_props)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
                continue;
            }

            // add(x, 0) -> x  ;  add(0, x) -> x
            if (op.opcode == OP_ADD && op.operands.size() == 2) {
                auto& a = store.facts_for(op.operands[0].id());
                auto& b = store.facts_for(op.operands[1].id());
                if (a.constant_value_known.known && a.constant_value_known.value &&
                    a.constant_value.known && a.constant_value.value == 0.0) {
                    // Result is the second operand.
                    auto& src = store.facts_for(op.operands[1].id());
                    if (src.constant_value_known.known && src.constant_value_known.value &&
                        src.constant_value.known) {
                        auto new_cv = Fact<double>::make(
                            src.constant_value.value, Confidence::Proven,
                            Provenance("add_zero_identity", op.id));
                        if (tf.constant_value.join(new_cv)) {
                            ++discovered;
                            store.notify_fact_changed(r.id(), id());
                        }
                    }
                }
            }

            // sub(x, x) -> 0
            if (op.opcode == OP_SUB && op.operands.size() == 2 &&
                op.operands[0].id() == op.operands[1].id()) {
                auto new_cv = Fact<double>::make(0.0, Confidence::Proven,
                                                  Provenance("sub_self_cancel", op.id));
                if (tf.constant_value.join(new_cv)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
                auto new_known = Fact<bool>::make(true, Confidence::Proven,
                                                   Provenance("sub_self_cancel", op.id));
                if (tf.constant_value_known.join(new_known)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
            }
        }
    }
    return discovered;
}

// ===========================================================================
// PropertyPropagator
// ===========================================================================
u32 PropertyPropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto r = op.results[0];
            auto& tf = store.facts_for(r.id());

            // matmul(A, Identity) -> A   (the result inherits A's properties)
            if (op.opcode == OP_MATMUL && op.operands.size() == 2) {
                auto& a = store.facts_for(op.operands[0].id());
                auto& b = store.facts_for(op.operands[1].id());
                if (a.properties.known &&
                    has_property(a.properties.value, TensorProperty::Identity)) {
                    // Result == B (matmul(Identity, B) = B).
                    if (b.properties.known) {
                        auto new_props = Fact<TensorProperty>::make(
                            b.properties.value, Confidence::Proven,
                            Provenance("matmul_identity_left", op.id));
                        if (tf.properties.join(new_props)) {
                            ++discovered;
                            store.notify_fact_changed(r.id(), id());
                        }
                    }
                }
                if (b.properties.known &&
                    has_property(b.properties.value, TensorProperty::Identity)) {
                    // Result == A (matmul(A, Identity) = A).
                    if (a.properties.known) {
                        auto new_props = Fact<TensorProperty>::make(
                            a.properties.value, Confidence::Proven,
                            Provenance("matmul_identity_right", op.id));
                        if (tf.properties.join(new_props)) {
                            ++discovered;
                            store.notify_fact_changed(r.id(), id());
                        }
                    }
                }
            }

            // mul(Zero, x) -> Zero
            if (op.opcode == OP_MUL && op.operands.size() == 2) {
                auto& a = store.facts_for(op.operands[0].id());
                auto& b = store.facts_for(op.operands[1].id());
                bool zero_a = a.properties.known &&
                              has_property(a.properties.value, TensorProperty::Zero);
                bool zero_b = b.properties.known &&
                              has_property(b.properties.value, TensorProperty::Zero);
                if (zero_a || zero_b) {
                    auto new_props = Fact<TensorProperty>::make(
                        TensorProperty::Zero | TensorProperty::Constant,
                        Confidence::Proven,
                        Provenance("mul_zero_property", op.id));
                    if (tf.properties.join(new_props)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
                // mul(One, x) -> x
                bool one_a = a.properties.known &&
                             has_property(a.properties.value, TensorProperty::One);
                bool one_b = b.properties.known &&
                             has_property(b.properties.value, TensorProperty::One);
                if (one_a && b.properties.known) {
                    auto new_props = Fact<TensorProperty>::make(
                        b.properties.value, Confidence::Proven,
                        Provenance("mul_one_identity", op.id));
                    if (tf.properties.join(new_props)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
                if (one_b && a.properties.known) {
                    auto new_props = Fact<TensorProperty>::make(
                        a.properties.value, Confidence::Proven,
                        Provenance("mul_one_identity", op.id));
                    if (tf.properties.join(new_props)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
            }

            // add(Zero, x) -> x  ;  add(x, Zero) -> x
            if (op.opcode == OP_ADD && op.operands.size() == 2) {
                auto& a = store.facts_for(op.operands[0].id());
                auto& b = store.facts_for(op.operands[1].id());
                bool zero_a = a.properties.known &&
                              has_property(a.properties.value, TensorProperty::Zero);
                bool zero_b = b.properties.known &&
                              has_property(b.properties.value, TensorProperty::Zero);
                if (zero_a && b.properties.known) {
                    auto new_props = Fact<TensorProperty>::make(
                        b.properties.value, Confidence::Proven,
                        Provenance("add_zero_identity", op.id));
                    if (tf.properties.join(new_props)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
                if (zero_b && a.properties.known) {
                    auto new_props = Fact<TensorProperty>::make(
                        a.properties.value, Confidence::Proven,
                        Provenance("add_zero_identity", op.id));
                    if (tf.properties.join(new_props)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
            }

            // transpose(Symmetric) -> Symmetric  ;  transpose(Diagonal) -> Diagonal
            if (op.opcode == OP_TRANSPOSE && op.operands.size() == 1) {
                auto& src = store.facts_for(op.operands[0].id());
                if (src.properties.known) {
                    TensorProperty preserved = TensorProperty::None;
                    if (has_property(src.properties.value, TensorProperty::Symmetric))
                        preserved = preserved | TensorProperty::Symmetric;
                    if (has_property(src.properties.value, TensorProperty::Diagonal))
                        preserved = preserved | TensorProperty::Diagonal;
                    if (has_property(src.properties.value, TensorProperty::Zero))
                        preserved = preserved | TensorProperty::Zero;
                    if (has_property(src.properties.value, TensorProperty::One))
                        preserved = preserved | TensorProperty::One;
                    if (has_property(src.properties.value, TensorProperty::Constant))
                        preserved = preserved | TensorProperty::Constant;
                    if (preserved != TensorProperty::None) {
                        auto new_props = Fact<TensorProperty>::make(
                            preserved, Confidence::Proven,
                            Provenance("transpose_preserves_structure", op.id));
                        if (tf.properties.join(new_props)) {
                            ++discovered;
                            store.notify_fact_changed(r.id(), id());
                        }
                    }
                }
            }
        }
    }
    return discovered;
}

// ===========================================================================
// RangePropagator
// ===========================================================================
u32 RangePropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto r = op.results[0];
            auto& tf = store.facts_for(r.id());

            ValueRange vr;

            switch (op.opcode) {
                case OP_RELU: {
                    // relu(x) >= 0
                    vr.lower = 0.0;
                    vr.possibly_negative = false;
                    break;
                }
                case OP_EXP: {
                    // exp(x) > 0
                    vr.lower = 0.0;
                    vr.possibly_negative = false;
                    vr.possibly_zero = false;
                    break;
                }
                case OP_SIGMOID: {
                    // sigmoid(x) ∈ (0, 1)
                    vr.lower = 0.0;
                    vr.upper = 1.0;
                    vr.possibly_negative = false;
                    break;
                }
                case OP_TANH: {
                    // tanh(x) ∈ [-1, 1]
                    vr.lower = -1.0;
                    vr.upper = 1.0;
                    break;
                }
                case OP_SQRT: {
                    // sqrt(x) >= 0
                    vr.lower = 0.0;
                    vr.possibly_negative = false;
                    break;
                }
                case OP_GELU: {
                    // gelu(x) ∈ approximately [-0.17, ∞)
                    vr.lower = -0.17;
                    vr.possibly_negative = true;
                    break;
                }
                case OP_MUL: {
                    // square-like patterns are non-negative.
                    if (op.operands.size() == 2 &&
                        op.operands[0].id() == op.operands[1].id()) {
                        vr.lower = 0.0;
                        vr.possibly_negative = false;
                    }
                    break;
                }
                default: continue;
            }

            auto new_range = Fact<ValueRange>::make(
                vr, Confidence::Proven,
                Provenance("RangePropagator", op.id));
            if (tf.value_range.join(new_range)) {
                ++discovered;
                store.notify_fact_changed(r.id(), id());
            }
        }
    }
    return discovered;
}

// ===========================================================================
// AliasPropagator
// ===========================================================================
u32 AliasPropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    u32 next_alias_set = 1;
    for (auto& f : m.functions()) {
        // Args: each gets its own alias set (NoAlias to each other).
        for (auto& arg : f->args()) {
            AliasClass ac;
            ac.kind = AliasKind::NoAlias;
            ac.alias_set_id = next_alias_set++;
            auto& tf = store.facts_for(arg.id());
            auto new_ac = Fact<AliasClass>::make(
                ac, Confidence::Proven,
                Provenance("AliasPropagator", 0));
            if (tf.alias_class.join(new_ac)) {
                ++discovered;
                store.notify_fact_changed(arg.id(), id());
            }
        }
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto r = op.results[0];
            auto& tf = store.facts_for(r.id());
            AliasClass ac;
            // Views/slices/transposes/reshapes inherit the parent's alias set.
            if (op.opcode == OP_TRANSPOSE || op.opcode == OP_RESHAPE ||
                op.opcode == OP_SLICE || op.opcode == OP_BROADCAST) {
                if (!op.operands.empty()) {
                    auto& src = store.facts_for(op.operands[0].id());
                    if (src.alias_class.known) {
                        ac.kind = AliasKind::MustAlias;
                        ac.alias_set_id = src.alias_class.value.alias_set_id;
                    }
                }
            } else {
                // Each result is a fresh allocation.
                ac.kind = AliasKind::NoAlias;
                ac.alias_set_id = next_alias_set++;
            }
            auto new_ac = Fact<AliasClass>::make(
                ac, Confidence::Proven,
                Provenance("AliasPropagator", op.id));
            if (tf.alias_class.join(new_ac)) {
                ++discovered;
                store.notify_fact_changed(r.id(), id());
            }
        }
    }
    return discovered;
}

// ===========================================================================
// LifetimePropagator
// ===========================================================================
u32 LifetimePropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    std::unordered_map<ValueId, u32> def_pos;
    std::unordered_map<ValueId, u32> last_use;
    std::unordered_map<ValueId, u32> use_count;

    u32 op_index = 0;
    for (auto& f : m.functions()) {
        for (auto& arg : f->args()) def_pos[arg.id()] = 0;
        for (auto& op : *f->entry()) {
            ++op_index;
            if (!op.results.empty()) def_pos[op.results[0].id()] = op_index;
            for (auto& v : op.operands) {
                last_use[v.id()] = std::max(last_use[v.id()], op_index);
                ++use_count[v.id()];
            }
        }
    }

    for (auto& [vid, sp] : def_pos) {
        auto& tf = store.facts_for(vid);
        auto new_birth = Fact<u32>::make(sp, Confidence::Proven,
                                          Provenance("LifetimePropagator", 0));
        if (tf.birth_op.join(new_birth)) {
            ++discovered;
            store.notify_fact_changed(vid, id());
        }
        u32 death = last_use.count(vid) ? last_use[vid] : sp;
        auto new_death = Fact<u32>::make(death, Confidence::Proven,
                                          Provenance("LifetimePropagator", 0));
        if (tf.death_op.join(new_death)) {
            ++discovered;
            store.notify_fact_changed(vid, id());
        }
        u32 n = use_count.count(vid) ? use_count[vid] : 0;
        auto new_users = Fact<u32>::make(n, Confidence::Proven,
                                          Provenance("LifetimePropagator", 0));
        if (tf.num_users.join(new_users)) {
            ++discovered;
            store.notify_fact_changed(vid, id());
        }
    }
    return discovered;
}

// ===========================================================================
// ReductionPropagator
// ===========================================================================
u32 ReductionPropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto r = op.results[0];
            auto& tf = store.facts_for(r.id());

            ReductionInfo ri;
            switch (op.opcode) {
                case OP_REDUCE_SUM: {
                    ri.is_reduction = true;
                    ri.is_associative = true;  // mathematically; numerically only under Relaxed
                    ri.is_commutative = true;
                    ri.identity_value = 0.0;
                    break;
                }
                case OP_REDUCE_MAX: {
                    ri.is_reduction = true;
                    ri.is_associative = true;
                    ri.is_commutative = true;
                    ri.identity_value = -std::numeric_limits<double>::infinity();
                    break;
                }
                case OP_REDUCE_MEAN: {
                    ri.is_reduction = true;
                    ri.is_associative = false;  // not associative (involves division)
                    ri.is_commutative = true;
                    ri.identity_value = 0.0;
                    break;
                }
                default: continue;
            }
            // Extract reduction axes from attributes.
            auto axes_attr = op.attributes.get("axes");
            if (axes_attr && axes_attr->kind == AttrKind::IntegerArray) {
                for (i64 a : axes_attr->ints) ri.reduction_axes.push_back(static_cast<i32>(a));
            }

            auto new_ri = Fact<ReductionInfo>::make(
                ri, Confidence::Proven,
                Provenance("ReductionPropagator", op.id));
            if (tf.reduction.join(new_ri)) {
                ++discovered;
                store.notify_fact_changed(r.id(), id());
            }
        }
    }
    return discovered;
}

// ===========================================================================
// DependencePropagator
// ===========================================================================
u32 DependencePropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    auto& edges = store.graph_facts().dependence_edges;
    edges.clear();

    // Track all consumers of each value.
    std::unordered_map<ValueId, std::vector<ValueId>> consumers;
    // Track op-positions for reuse distance.
    std::unordered_map<ValueId, std::vector<u32>> use_positions;
    u32 op_index = 0;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            ++op_index;
            for (auto& v : op.operands) {
                if (!op.results.empty()) {
                    DependenceEdge e;
                    e.producer = v.id();
                    e.consumer = op.results[0].id();
                    // Classify the dependence.
                    switch (op.opcode) {
                        case OP_SLICE: e.kind = DependenceKind::Slice; break;
                        case OP_REDUCE_SUM: case OP_REDUCE_MAX: case OP_REDUCE_MEAN:
                            e.kind = DependenceKind::Reduction; break;
                        case OP_BROADCAST: e.kind = DependenceKind::Broadcast; break;
                        case OP_TRANSPOSE: case OP_RESHAPE:
                            e.kind = DependenceKind::LayoutOnly; break;
                        default: e.kind = DependenceKind::Full; break;
                    }
                    edges.push_back(e);
                    consumers[v.id()].push_back(op.results[0].id());
                }
                use_positions[v.id()].push_back(op_index);
            }
        }
    }
    // Record consumers + reuse distance.
    for (auto& [vid, conss] : consumers) {
        auto& tf = store.facts_for(vid);
        auto new_cons = Fact<std::vector<ValueId>>::make(
            conss, Confidence::Proven,
            Provenance("DependencePropagator", 0));
        if (tf.consumers.join(new_cons)) {
            ++discovered;
            store.notify_fact_changed(vid, id());
        }
        // Reuse distance = max gap between consecutive uses.
        auto& positions = use_positions[vid];
        if (positions.size() >= 2) {
            u32 max_gap = 0;
            for (usize i = 1; i < positions.size(); ++i) {
                u32 gap = positions[i] - positions[i - 1];
                if (gap > max_gap) max_gap = gap;
            }
            auto new_rd = Fact<u32>::make(max_gap, Confidence::Derived,
                                           Provenance("DependencePropagator", 0));
            if (tf.reuse_distance.join(new_rd)) {
                ++discovered;
                store.notify_fact_changed(vid, id());
            }
        }
    }
    return discovered;
}

// ===========================================================================
// CostPropagator
// ===========================================================================
u32 CostPropagator::run(FactStore& store) {
    u32 discovered = 0;
    Module& m = store.module();
    u64 total_flops = 0;
    u64 total_bytes = 0;

    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            auto r = op.results[0];
            auto& tf = store.facts_for(r.id());

            // ---- Compute FLOPs ----
            u64 flops = 0;
            switch (op.opcode) {
                case OP_MATMUL: {
                    if (op.operands.size() == 2) {
                        auto a = op.operands[0].as_tensor();
                        auto b = op.operands[1].as_tensor();
                        if (a && b && a->shape.rank() >= 2 && b->shape.rank() >= 2) {
                            u64 M = a->shape[a->shape.rank()-2]->is_constant()
                                ? a->shape[a->shape.rank()-2]->value : 0;
                            u64 K = a->shape[a->shape.rank()-1]->is_constant()
                                ? a->shape[a->shape.rank()-1]->value : 0;
                            u64 N = b->shape[b->shape.rank()-1]->is_constant()
                                ? b->shape[b->shape.rank()-1]->value : 0;
                            flops = 2 * M * K * N;
                        }
                    }
                    break;
                }
                case OP_REDUCE_SUM: case OP_REDUCE_MAX: case OP_REDUCE_MEAN: {
                    if (!op.operands.empty()) {
                        auto a = op.operands[0].as_tensor();
                        if (a) flops = numel_of(a->shape);
                    }
                    break;
                }
                default: {
                    if (op.has_trait(OpTrait::Elementwise)) {
                        auto t = r.as_tensor();
                        if (t) flops = numel_of(t->shape);
                    }
                    break;
                }
            }
            if (flops > 0) {
                auto new_flops = Fact<u64>::make(flops, Confidence::Proven,
                                                  Provenance("CostPropagator", op.id));
                if (tf.estimated_flops.join(new_flops)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
                total_flops += flops;
            }

            // ---- Compute bytes ----
            u64 bytes_read = 0;
            u64 bytes_written = 0;
            for (auto& v : op.operands) {
                auto t = v.as_tensor();
                if (t) bytes_read += numel_of(t->shape) * dtype_size(t->dtype);
            }
            {
                auto t = r.as_tensor();
                if (t) bytes_written += numel_of(t->shape) * dtype_size(t->dtype);
            }
            if (bytes_read > 0) {
                auto new_br = Fact<u64>::make(bytes_read, Confidence::Proven,
                                               Provenance("CostPropagator", op.id));
                if (tf.estimated_bytes_read.join(new_br)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
            }
            if (bytes_written > 0) {
                auto new_bw = Fact<u64>::make(bytes_written, Confidence::Proven,
                                               Provenance("CostPropagator", op.id));
                if (tf.estimated_bytes_written.join(new_bw)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
            }
            // Arithmetic intensity.
            u64 total_b = bytes_read + bytes_written;
            if (total_b > 0 && flops > 0) {
                double ai = static_cast<double>(flops) / static_cast<double>(total_b);
                auto new_ai = Fact<double>::make(ai, Confidence::Derived,
                                                  Provenance("CostPropagator", op.id));
                if (tf.arithmetic_intensity.join(new_ai)) {
                    ++discovered;
                    store.notify_fact_changed(r.id(), id());
                }
            }
            total_bytes += total_b;
        }
    }

    // ---- Graph-level cost facts ----
    auto& g = store.graph_facts();
    if (total_flops > 0) {
        auto new_tf = Fact<u64>::make(total_flops, Confidence::Proven,
                                       Provenance("CostPropagator", 0));
        if (g.total_flops.join(new_tf)) ++discovered;
    }
    if (total_bytes > 0) {
        auto new_tb = Fact<u64>::make(total_bytes, Confidence::Proven,
                                       Provenance("CostPropagator", 0));
        if (g.total_bytes.join(new_tb)) ++discovered;
    }
    if (total_bytes > 0 && total_flops > 0) {
        double gai = static_cast<double>(total_flops) / static_cast<double>(total_bytes);
        auto new_gai = Fact<double>::make(gai, Confidence::Derived,
                                           Provenance("CostPropagator", 0));
        if (g.graph_arithmetic_intensity.join(new_gai)) ++discovered;
    }

    // ---- Cache behavior (Estimated, requires hardware) ----
    if (store.has_hardware()) {
        const auto& hw = store.hardware();
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                if (op.results.empty()) continue;
                auto r = op.results[0];
                auto& tf = store.facts_for(r.id());
                CacheBehavior cb;
                // L2 hit rate estimate from operand B in matmul.
                if (op.opcode == OP_MATMUL && op.operands.size() == 2) {
                    auto b = op.operands[1].as_tensor();
                    if (b) {
                        u64 b_bytes = numel_of(b->shape) * dtype_size(b->dtype);
                        if (b_bytes > 0 && hw.l2_cache_bytes > 0) {
                            if (b_bytes <= hw.l2_cache_bytes) {
                                cb.l2_hit_rate = 0.8;
                                cb.fits_in_l2 = true;
                            } else {
                                cb.l2_hit_rate = static_cast<double>(hw.l2_cache_bytes) /
                                                  static_cast<double>(b_bytes);
                            }
                        }
                        cb.shared_reuse_factor = 32.0; // rough: K / k_tile
                        cb.accumulator_in_registers = true;
                    }
                }
                if (cb.l2_hit_rate > 0 || cb.shared_reuse_factor > 1.0) {
                    auto new_cb = Fact<CacheBehavior>::make(
                        cb, Confidence::Estimated,
                        Provenance("CostPropagator", op.id));
                    if (tf.cache_behavior.join(new_cb)) {
                        ++discovered;
                        store.notify_fact_changed(r.id(), id());
                    }
                }
            }
        }
        // Effective intensity: total_flops / (total_bytes * (1 - l2_hit_avg)).
        if (g.total_flops.known && g.total_bytes.known) {
            // Very rough: assume average L2 hit rate of 0.5 across the graph.
            double eff = static_cast<double>(g.total_flops.value) /
                          (static_cast<double>(g.total_bytes.value) * 0.5);
            auto new_eff = Fact<double>::make(eff, Confidence::Estimated,
                                               Provenance("CostPropagator", 0));
            if (g.effective_arithmetic_intensity.join(new_eff)) ++discovered;
        }
        double ridge = hw.roofline_ridge(DType::F32, false);
        auto new_ridge = Fact<double>::make(ridge, Confidence::Proven,
                                              Provenance("CostPropagator", 0));
        if (g.roofline_ridge.join(new_ridge)) ++discovered;
    }

    return discovered;
}

} // namespace cg
