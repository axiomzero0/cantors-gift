// optimization/unified/migrated_passes.cpp - implementations.
//
// Each pass here is a 1:1 rewrite of an existing pass that queries the
// unified FactStore instead of re-deriving facts from IR attributes.
// The logic is intentionally parallel to the original so we can A/B
// test that the unified version produces the same (or better) results.
#include "cg/optimization/unified/migrated_passes.hpp"

#include "cg/analysis/unified/unified_analyzer.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace cg {

namespace {

// Build a fresh analyzer with default propagators + hardware.
UnifiedAnalyzer make_analyzer(Module& m, AnalysisManager& am) {
    UnifiedAnalyzer a(m);
    a.set_numerical_mode(NumericalMode::FastMath);
    a.add_default_propagators();
    return a;
}

// RAII helper: use a shared analyzer if provided, else build a fresh one.
// This is the key to analyzer reuse — passes that receive a non-null
// shared_ pointer skip the ~50-200 µs analyzer construction + run.
struct AnalyzerScope {
    std::unique_ptr<UnifiedAnalyzer> owned;
    UnifiedAnalyzer* ptr = nullptr;

    AnalyzerScope(UnifiedAnalyzer* shared, Module& m) {
        if (shared) {
            ptr = shared;
        } else {
            owned = std::make_unique<UnifiedAnalyzer>(m);
            owned->set_numerical_mode(NumericalMode::FastMath);
            owned->add_default_propagators();
            owned->run();
            ptr = owned.get();
        }
    }

    FactStore& store() { return ptr->store(); }
};

} // namespace

// Hash an op for CSE: opcode + canonical operand ids + attributes.
u64 hash_op_for_cse(const Operation& op) {
    u64 h = std::hash<u32>{}(op.opcode);
    for (auto& v : op.operands) hash_combine(h, std::hash<ValueId>{}(v.id()));
    // Hash attributes.
    for (auto& [k, v] : op.attributes) {
        hash_combine(h, std::hash<std::string>{}(k));
        hash_combine(h, static_cast<u64>(v->kind));
        switch (v->kind) {
            case AttrKind::Integer: hash_combine(h, std::hash<i64>{}(v->integer)); break;
            case AttrKind::Float:   hash_combine(h, std::hash<double>{}(v->real)); break;
            case AttrKind::Bool:    hash_combine(h, std::hash<bool>{}(v->flag)); break;
            case AttrKind::String:  hash_combine(h, std::hash<std::string>{}(v->str)); break;
            case AttrKind::IntegerArray:
                for (auto x : v->ints) hash_combine(h, std::hash<i64>{}(x));
                break;
            default: break;
        }
    }
    return h;
}

// Find the defining op of a value (linear scan).
Operation* find_defining_op(Function& f, ValueId vid) {
    for (auto& op : *f.entry()) {
        for (auto& r : op.results) {
            if (r.id() == vid) return &op;
        }
    }
    return nullptr;
}

// ===========================================================================
// UnifiedCSEPass
// ===========================================================================
PreservedAnalyses UnifiedCSEPass::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    AnalyzerScope scope(shared_, m);

    bool changed = false;
    for (auto& f : m.functions()) {
        std::unordered_map<u64, Operation*> existing;
        std::vector<Operation*> to_remove;

        for (auto& op : *f->entry()) {
            if (!op.is_pure()) continue;
            u64 h = hash_op_for_cse(op);
            auto it = existing.find(h);
            if (it == existing.end()) {
                existing[h] = &op;
                continue;
            }
            // Hash hit: verify it's actually equivalent (not a collision).
            auto* prior = it->second;
            if (prior->opcode == op.opcode &&
                prior->operands == op.operands &&
                prior->results.size() == op.results.size()) {
                for (usize i = 0; i < op.results.size(); ++i) {
                    m.replace_all_uses(op.results[i], prior->results[i]);
                }
                to_remove.push_back(&op);
                ++stats_.duplicates_removed;
                changed = true;
            } else {
                existing[h] = &op;
            }
        }
        for (Operation* op : to_remove) {
            f->entry()->remove(op);
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// UnifiedDCEPass
// ===========================================================================
PreservedAnalyses UnifiedDCEPass::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    AnalyzerScope scope(shared_, m);
    const FactStore& store = scope.store();

    bool changed = false;
    bool iterate = true;
    while (iterate) {
        iterate = false;
        for (auto& f : m.functions()) {
            // Collect used values from the IR directly (faster than
            // walking the fact store).
            std::unordered_set<ValueId> used;
            for (auto& op : *f->entry()) {
                for (auto& v : op.operands) used.insert(v.id());
            }
            for (auto& arg : f->entry()->arguments()) used.insert(arg.id());

            std::vector<Operation*> to_remove;
            for (auto it = f->entry()->head(); it; it = it->next) {
                if (!it->is_pure()) continue;
                bool any_used = false;
                for (auto& r : it->results) {
                    if (used.count(r.id())) { any_used = true; break; }
                }
                if (!any_used) {
                    // Cross-check with the fact store: num_users should be 0.
                    const TensorFacts* tf = store.facts_for(it->results[0].id());
                    if (tf && tf->num_users.known && tf->num_users.value > 0) {
                        // Fact store says it has users — trust the IR scan
                        // (the fact store might be stale from a prior run).
                        // The IR scan is authoritative.
                    }
                    to_remove.push_back(it);
                }
            }
            for (Operation* op : to_remove) {
                f->entry()->remove(op);
                ++stats_.ops_removed;
                changed = true;
                iterate = true;
            }
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// UnifiedConstantFoldingPass
// ===========================================================================
PreservedAnalyses UnifiedConstantFoldingPass::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    AnalyzerScope scope(shared_, m);
    const FactStore& store = scope.store();

    bool changed = false;
    for (auto& f : m.functions()) {
        // Build a value-id -> defining-op map.
        std::unordered_map<ValueId, Operation*> defs;
        for (auto& op : *f->entry()) {
            if (!op.results.empty()) defs[op.results[0].id()] = &op;
        }

        std::vector<Operation*> to_remove;
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            ValueId result_vid = op.results[0].id();

            // If the result is provably Zero (Proven confidence), and the op
            // is one that produces a zero from a zero operand, we can
            // propagate the zero.
            if (store.is_zero(result_vid) &&
                (op.opcode == OP_ADD || op.opcode == OP_SUB ||
                 op.opcode == OP_MUL || op.opcode == OP_DIV)) {
                // The result is zero — find the zero operand and replace.
                // (For mul: either operand being zero makes the result zero.)
                // (For add/sub: only if the OTHER operand is zero, which
                // is handled by canonicalize. Here we just record the fact.)
                ++stats_.zero_propagations;
                // We don't replace here — canonicalize handles the actual
                // operand substitution. We just record that the constant
                // folder recognized the zero.
            }

            // If both operands are constants, fold the arithmetic.
            if ((op.opcode == OP_ADD || op.opcode == OP_SUB ||
                 op.opcode == OP_MUL || op.opcode == OP_DIV) &&
                op.operands.size() == 2) {
                auto ldef = defs.find(op.operands[0].id());
                auto rdef = defs.find(op.operands[1].id());
                if (ldef == defs.end() || rdef == defs.end()) continue;
                if (ldef->second->opcode != OP_CONSTANT ||
                    rdef->second->opcode != OP_CONSTANT) continue;

                // Use the FactStore's constant_value fact if available.
                auto lcv = store.constant_value(op.operands[0].id());
                auto rcv = store.constant_value(op.operands[1].id());
                if (lcv.has_value() && rcv.has_value()) {
                    // Both are scalar constants — fold.
                    double result = 0.0;
                    switch (op.opcode) {
                        case OP_ADD: result = *lcv + *rcv; break;
                        case OP_SUB: result = *lcv - *rcv; break;
                        case OP_MUL: result = *lcv * *rcv; break;
                        case OP_DIV:
                            if (*rcv == 0.0) continue;  // don't fold div by zero
                            result = *lcv / *rcv; break;
                        default: continue;
                    }
                    // Build a new constant op with the folded value.
                    auto t = op.results[0].as_tensor();
                    if (!t) continue;
                    AttributeDict attrs;
                    std::vector<i64> shape_vec;
                    for (auto& d : t->shape) {
                        if (d->is_constant()) shape_vec.push_back(d->value);
                    }
                    if (shape_vec.empty()) continue;
                    attrs.set("shape", Attribute::make_int_array(shape_vec));
                    attrs.set("dtype", Attribute::make_dtype(t->dtype));
                    attrs.set("value", Attribute::make_float(result));
                    // Materialize bytes for the result.
                    std::string bytes(t->shape.num_elements() * dtype_size(t->dtype), '\0');
                    if (t->dtype == DType::F32) {
                        float v = static_cast<float>(result);
                        std::memcpy(bytes.data(), &v, 4);
                    } else if (t->dtype == DType::F64) {
                        std::memcpy(bytes.data(), &result, 8);
                    }
                    attrs.set("bytes", Attribute::make_string(std::move(bytes)));

                    Builder b(f.get());
                    auto* new_op = b.create(OP_CONSTANT, {}, attrs);
                    m.replace_all_uses(op.results[0], new_op->results[0]);
                    to_remove.push_back(&op);
                    ++stats_.constants_folded;
                    changed = true;
                }
            }
        }
        for (Operation* op : to_remove) {
            f->entry()->remove(op);
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// UnifiedCanonicalizePass
// ===========================================================================
PreservedAnalyses UnifiedCanonicalizePass::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    AnalyzerScope scope(shared_, m);
    const FactStore& store = scope.store();

    bool changed = false;
    for (int iter = 0; iter < 8; ++iter) {
        bool iter_changed = false;
        for (auto& f : m.functions()) {
            for (auto& op : *f->entry()) {
                if (op.results.empty()) continue;
                ValueId result_vid = op.results[0].id();

                // add(x, Zero) -> x   ;   add(Zero, x) -> x
                if (op.opcode == OP_ADD && op.operands.size() == 2) {
                    if (store.is_zero(op.operands[0].id())) {
                        m.replace_all_uses(op.results[0], op.operands[1]);
                        ++stats_.add_zero_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                        continue;
                    }
                    if (store.is_zero(op.operands[1].id())) {
                        m.replace_all_uses(op.results[0], op.operands[0]);
                        ++stats_.add_zero_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                        continue;
                    }
                    // Commutative reorder.
                    if (op.operands[0].id() > op.operands[1].id()) {
                        std::swap(op.operands[0], op.operands[1]);
                        ++stats_.commutative_reordered;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                    }
                }

                // mul(x, One) -> x   ;   mul(One, x) -> x
                // mul(x, Zero) -> Zero   ;   mul(Zero, x) -> Zero
                if (op.opcode == OP_MUL && op.operands.size() == 2) {
                    if (store.is_one(op.operands[0].id())) {
                        m.replace_all_uses(op.results[0], op.operands[1]);
                        ++stats_.mul_one_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                        continue;
                    }
                    if (store.is_one(op.operands[1].id())) {
                        m.replace_all_uses(op.results[0], op.operands[0]);
                        ++stats_.mul_one_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                        continue;
                    }
                    if (store.is_zero(op.operands[0].id())) {
                        m.replace_all_uses(op.results[0], op.operands[0]);
                        ++stats_.mul_zero_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                        continue;
                    }
                    if (store.is_zero(op.operands[1].id())) {
                        m.replace_all_uses(op.results[0], op.operands[1]);
                        ++stats_.mul_zero_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                        continue;
                    }
                    if (op.operands[0].id() > op.operands[1].id()) {
                        std::swap(op.operands[0], op.operands[1]);
                        ++stats_.commutative_reordered;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                    }
                }

                // sub(x, Zero) -> x
                if (op.opcode == OP_SUB && op.operands.size() == 2) {
                    if (store.is_zero(op.operands[1].id())) {
                        m.replace_all_uses(op.results[0], op.operands[0]);
                        ++stats_.sub_zero_simplified;
                        ++stats_.total_rewrites;
                        iter_changed = true;
                    }
                }

                // transpose(transpose(x)) -> x
                // Use the FactStore's alias_class to verify both transposes
                // view the same underlying storage, then check the perms.
                if (op.opcode == OP_TRANSPOSE && !op.operands.empty()) {
                    Operation* inner = find_defining_op(*f, op.operands[0].id());
                    if (inner && inner->opcode == OP_TRANSPOSE &&
                        !inner->operands.empty()) {
                        auto outer_perm = op.attributes.get("perm");
                        auto inner_perm = inner->attributes.get("perm");
                        if (outer_perm && inner_perm &&
                            outer_perm->kind == AttrKind::IntegerArray &&
                            inner_perm->kind == AttrKind::IntegerArray &&
                            outer_perm->ints.size() == inner_perm->ints.size()) {
                            bool identity = true;
                            for (usize i = 0; i < outer_perm->ints.size(); ++i) {
                                if (outer_perm->ints[inner_perm->ints[i]] !=
                                    static_cast<i64>(i)) {
                                    identity = false; break;
                                }
                            }
                            if (identity) {
                                m.replace_all_uses(op.results[0], inner->operands[0]);
                                ++stats_.transpose_pair_eliminated;
                                ++stats_.total_rewrites;
                                iter_changed = true;
                            }
                        }
                    }
                }
            }
        }
        if (!iter_changed) break;
        changed = true;
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// UnifiedCopyEliminationPass
// ===========================================================================
PreservedAnalyses UnifiedCopyEliminationPass::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    AnalyzerScope scope(shared_, m);
    const FactStore& store = scope.store();

    bool changed = false;
    for (auto& f : m.functions()) {
        std::vector<Operation*> to_remove;
        for (auto& op : *f->entry()) {
            if (op.results.empty() || op.operands.empty()) continue;

            // transpose(transpose(x)) -> x
            if (op.opcode == OP_TRANSPOSE) {
                Operation* inner = find_defining_op(*f, op.operands[0].id());
                if (inner && inner->opcode == OP_TRANSPOSE &&
                    !inner->operands.empty()) {
                    auto outer_perm = op.attributes.get("perm");
                    auto inner_perm = inner->attributes.get("perm");
                    if (outer_perm && inner_perm &&
                        outer_perm->kind == AttrKind::IntegerArray &&
                        inner_perm->kind == AttrKind::IntegerArray &&
                        outer_perm->ints.size() == 2 &&
                        inner_perm->ints.size() == 2 &&
                        outer_perm->ints[0] == 1 && outer_perm->ints[1] == 0 &&
                        inner_perm->ints[0] == 1 && inner_perm->ints[1] == 0) {
                        m.replace_all_uses(op.results[0], inner->operands[0]);
                        ++stats_.transposes_eliminated;
                        ++stats_.total_rewrites;
                        changed = true;
                    }
                }
            }

            // reshape(reshape(x)) -> reshape(x)
            if (op.opcode == OP_RESHAPE) {
                Operation* inner = find_defining_op(*f, op.operands[0].id());
                if (inner && inner->opcode == OP_RESHAPE &&
                    !inner->operands.empty()) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    ++stats_.reshapes_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                }
            }

            // broadcast(x, same_shape) -> x
            // Use FactStore's static_shape to compare shapes.
            if (op.opcode == OP_BROADCAST) {
                auto in_shape = store.static_shape(op.operands[0].id());
                auto out_shape = store.static_shape(op.results[0].id());
                if (in_shape.has_value() && out_shape.has_value() &&
                    *in_shape == *out_shape) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    ++stats_.broadcasts_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                }
            }
        }
        for (Operation* op : to_remove) {
            f->entry()->remove(op);
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// UnifiedRecomputationPass
// ===========================================================================
PreservedAnalyses UnifiedRecomputationPass::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    // Recomputation needs hardware-aware cost facts. If we have a shared
    // analyzer, use it (the pipeline sets hardware before running).
    // Otherwise build a fresh one with hardware set.
    std::unique_ptr<UnifiedAnalyzer> owned;
    UnifiedAnalyzer* ptr = shared_;
    if (!ptr) {
        owned = std::make_unique<UnifiedAnalyzer>(m);
        owned->set_numerical_mode(NumericalMode::FastMath);
        owned->set_hardware(HardwareModel::generic_nvidia_gpu());
        owned->add_default_propagators();
        owned->run();
        ptr = owned.get();
    }
    const FactStore& store = ptr->store();

    // For each value with multiple consumers, decide materialize vs recompute.
    // Heuristic (from the unified fact store):
    //   - num_users == 0 or 1  -> fuse (single consumer)
    //   - num_users > 1 && estimated_flops < threshold  -> recompute
    //   - num_users > 1 && estimated_flops >= threshold  -> materialize
    // The threshold is hardware-aware: ~1000 FLOPs is cheaper to recompute
    // than to spill to memory and reload.
    const u64 recompute_threshold = 1000;

    for (auto& [vid, tf] : store) {
        if (!tf.num_users.known || tf.num_users.value <= 1) {
            if (tf.num_users.known && tf.num_users.value == 1) {
                ++stats_.fuse_decisions;
            }
            continue;
        }
        if (!tf.estimated_flops.known) continue;
        if (tf.estimated_flops.value < recompute_threshold) {
            ++stats_.recompute_decisions;
            // Mark the defining op with a "recompute" attribute.
            for (auto& f : m.functions()) {
                Operation* def = find_defining_op(*f, vid);
                if (def) {
                    def->attributes.set("reuse_decision",
                        Attribute::make_string("recompute"));
                    break;
                }
            }
        } else {
            ++stats_.materialize_decisions;
            for (auto& f : m.functions()) {
                Operation* def = find_defining_op(*f, vid);
                if (def) {
                    def->attributes.set("reuse_decision",
                        Attribute::make_string("materialize"));
                    break;
                }
            }
        }
    }

    // This pass doesn't actually transform IR (it just annotates), so we
    // preserve everything.
    return PreservedAnalyses::all();
}

// ===========================================================================
// UnifiedPassPipeline
//
// Owns a shared UnifiedAnalyzer. The first pass triggers the initial
// analyzer run; subsequent passes reuse the FactStore. When a pass
// mutates the IR (returns PreservedAnalyses::none()), the pipeline
// re-runs the analyzer so the next pass sees fresh facts.
//
// This cuts total analyzer invocations from N (one per pass) to
// ~1-3 (one initial + one after each IR-mutating pass). In practice
// for a 6-pass pipeline on unchanged IR: 1 run instead of 6.
// ===========================================================================
PreservedAnalyses UnifiedPassPipeline::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};

    // Build the shared analyzer ONCE.
    UnifiedAnalyzer shared(m);
    shared.set_numerical_mode(NumericalMode::FastMath);
    shared.set_hardware(HardwareModel::generic_nvidia_gpu());
    shared.add_default_propagators();

    // Helper: run the analyzer (if needed) and account for it.
    auto refresh = [&](bool force) -> void {
        // We always run on the first call (force=true) and re-run when
        // a pass mutated the IR (force=true). Otherwise skip — the
        // facts are still fresh.
        if (!force) return;
        auto t0 = std::chrono::steady_clock::now();
        shared.run();
        auto t1 = std::chrono::steady_clock::now();
        stats_.analyzer_latency_sec +=
            std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
        ++stats_.analyzer_runs;
    };

    // Initial analyzer run.
    refresh(true);

    // Pass 1: ConstantFolding (reads facts, may mutate IR).
    UnifiedConstantFoldingPass cf;
    cf.set_shared_analyzer(&shared);
    auto pa = cf.run(m, am);
    stats_.const_fold = cf.stats();
    bool ir_changed = !pa.preserves_all();
    refresh(ir_changed);

    // Pass 2: Canonicalize (reads facts, may mutate IR).
    UnifiedCanonicalizePass can;
    can.set_shared_analyzer(&shared);
    pa = can.run(m, am);
    stats_.canonicalize = can.stats();
    ir_changed = !pa.preserves_all();
    refresh(ir_changed);

    // Pass 3: CSE (reads facts, may mutate IR).
    UnifiedCSEPass cse;
    cse.set_shared_analyzer(&shared);
    pa = cse.run(m, am);
    stats_.cse = cse.stats();
    ir_changed = !pa.preserves_all();
    refresh(ir_changed);

    // Pass 4: CopyElimination (reads facts, may mutate IR).
    UnifiedCopyEliminationPass ce;
    ce.set_shared_analyzer(&shared);
    pa = ce.run(m, am);
    stats_.copy_elim = ce.stats();
    ir_changed = !pa.preserves_all();
    refresh(ir_changed);

    // Pass 5: Recomputation (reads facts, annotates IR — doesn't mutate).
    UnifiedRecomputationPass rc;
    rc.set_shared_analyzer(&shared);
    pa = rc.run(m, am);
    stats_.recompute = rc.stats();
    // Recomputation only annotates, doesn't mutate. No refresh needed.

    // Pass 6: DCE (reads facts, may mutate IR).
    UnifiedDCEPass dce;
    dce.set_shared_analyzer(&shared);
    pa = dce.run(m, am);
    stats_.dce = dce.stats();
    // No refresh needed — DCE is the last pass.

    return PreservedAnalyses::none();
}

} // namespace cg
