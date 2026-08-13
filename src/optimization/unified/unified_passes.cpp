// optimization/unified/unified_passes.cpp - implementations.
//
// Every pass here follows the same pattern:
//   1. Construct a UnifiedAnalyzer on the module + AnalysisManager.
//   2. Run it to fixed point (cheap: ~50-200 us).
//   3. Walk the IR, query the FactStore for each operation.
//   4. Apply rewrites based on Proven facts (never Estimated for legality).
//   5. Return PreservedAnalyses::none() (we mutated the IR).
//
// The KEY design rule: passes do NOT re-implement analysis. They ask the
// analyzer "what's true?" and act on the answer. This is the GCC model.
#include "cg/optimization/unified/unified_passes.hpp"

#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"

#include <iostream>

namespace cg {

namespace {

// Build a fresh analyzer with default propagators + hardware.
UnifiedAnalyzer make_analyzer(Module& m, AnalysisManager& am) {
    UnifiedAnalyzer a(m);
    a.set_numerical_mode(NumericalMode::FastMath);
    a.add_default_propagators();
    return a;
}

// Find the defining op of a value (linear scan; IR is small per function).
Operation* find_defining_op(Function& f, ValueId vid) {
    for (auto& op : *f.entry()) {
        for (auto& r : op.results) {
            if (r.id() == vid) return &op;
        }
    }
    return nullptr;
}

} // namespace

// ===========================================================================
// Pass 1: PropertyDrivenSimplification
// ===========================================================================
PreservedAnalyses PropertyDrivenSimplification::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    auto analyzer = make_analyzer(m, am);
    analyzer.run();
    auto& store = analyzer.store();

    bool changed = false;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty()) continue;
            ValueId result_vid = op.results[0].id();

            // mul(x, Zero) -> Zero  (the result is provably zero)
            if (op.opcode == OP_MUL && op.operands.size() == 2) {
                if (store.is_zero(op.operands[0].id())) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    ++stats_.mul_zero_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
                if (store.is_zero(op.operands[1].id())) {
                    m.replace_all_uses(op.results[0], op.operands[1]);
                    ++stats_.mul_zero_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
            }

            // mul(x, One) -> x
            if (op.opcode == OP_MUL && op.operands.size() == 2) {
                if (store.is_one(op.operands[0].id())) {
                    m.replace_all_uses(op.results[0], op.operands[1]);
                    ++stats_.mul_one_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
                if (store.is_one(op.operands[1].id())) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    ++stats_.mul_one_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
            }

            // add(x, Zero) -> x
            if (op.opcode == OP_ADD && op.operands.size() == 2) {
                if (store.is_zero(op.operands[0].id())) {
                    m.replace_all_uses(op.results[0], op.operands[1]);
                    ++stats_.add_zero_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
                if (store.is_zero(op.operands[1].id())) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    ++stats_.add_zero_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
            }

            // matmul(A, Identity) -> A   ;   matmul(Identity, B) -> B
            // Only valid when the identity has the right shape.
            if (op.opcode == OP_MATMUL && op.operands.size() == 2) {
                if (store.is_identity(op.operands[1].id())) {
                    // matmul(A, I) = A, provided shapes line up.
                    // For a 2D case: A[M,K] x I[K,K] = A[M,K]
                    auto a = op.operands[0].as_tensor();
                    auto ident = op.operands[1].as_tensor();
                    if (a && ident && a->shape.rank() == 2 &&
                        ident->shape.rank() == 2) {
                        // Check K dimensions match.
                        if (a->shape[1]->structurally_equal(*ident->shape[0])) {
                            m.replace_all_uses(op.results[0], op.operands[0]);
                            ++stats_.matmul_identity_eliminated;
                            ++stats_.total_rewrites;
                            changed = true;
                            continue;
                        }
                    }
                }
                if (store.is_identity(op.operands[0].id())) {
                    // matmul(I, B) = B
                    auto b = op.operands[1].as_tensor();
                    auto ident = op.operands[0].as_tensor();
                    if (b && ident && b->shape.rank() == 2 &&
                        ident->shape.rank() == 2) {
                        if (ident->shape[1]->structurally_equal(*b->shape[0])) {
                            m.replace_all_uses(op.results[0], op.operands[1]);
                            ++stats_.matmul_identity_eliminated;
                            ++stats_.total_rewrites;
                            changed = true;
                            continue;
                        }
                    }
                }
            }
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// Pass 2: RangeDrivenStrengthReduction
// ===========================================================================
PreservedAnalyses RangeDrivenStrengthReduction::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    auto analyzer = make_analyzer(m, am);
    analyzer.run();
    const FactStore& store = analyzer.store();

    bool changed = false;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.results.empty() || op.operands.empty()) continue;
            ValueId operand_vid = op.operands[0].id();

            // relu(x) -> x   when x is provably non-negative.
            // LEGALITY: only when the value range is Proven (not Estimated).
            if (op.opcode == OP_RELU) {
                const TensorFacts* tf = store.facts_for(operand_vid);
                if (tf && tf->value_range.known &&
                    tf->value_range.confidence == Confidence::Proven &&
                    tf->value_range.value.is_non_negative()) {
                    m.replace_all_uses(op.results[0], op.operands[0]);
                    ++stats_.relu_eliminated;
                    ++stats_.total_rewrites;
                    changed = true;
                    continue;
                }
            }

            // abs(x) -> x   when x is provably non-negative.
            // (OP_ABS isn't in the opcode list, but we keep this for future
            // use; the test for it is defensive.)
            // Note: the IR doesn't have an OP_ABS yet, so this is a no-op
            // in practice but demonstrates the pattern.
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// Pass 3: CostGuidedFusion
// ===========================================================================
PreservedAnalyses CostGuidedFusion::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    decisions_.clear();
    auto analyzer = make_analyzer(m, am);
    analyzer.set_hardware(HardwareModel::generic_nvidia_gpu());
    analyzer.run();
    auto& store = analyzer.store();

    // Walk the dependence graph: for each (producer, consumer) pair,
    // ask the FactStore "should I fuse these?" and act on the answer.
    bool changed = false;
    auto& edges = store.graph_facts().dependence_edges;
    for (auto& e : edges) {
        ValueId producer = e.producer;
        ValueId consumer = e.consumer;

        if (!store.can_fuse(producer, consumer)) {
            stats_.fusions_rejected_legality++;
            decisions_.push_back({producer, consumer, false,
                                  "can_fuse returned false", 0.0,
                                  Confidence::Proven});
            continue;
        }

        auto benefit = store.fusion_benefit(producer, consumer);
        if (benefit.confidence > min_confidence_) {
            // Confidence too low (higher numeric = less trusted).
            stats_.fusions_rejected_confidence++;
            decisions_.push_back({producer, consumer, false,
                                  "confidence below threshold",
                                  benefit.net_predicted_improvement,
                                  benefit.confidence});
            continue;
        }

        if (benefit.net_predicted_improvement < min_improvement_) {
            stats_.fusions_rejected_cost++;
            decisions_.push_back({producer, consumer, false,
                                  "net improvement below threshold",
                                  benefit.net_predicted_improvement,
                                  benefit.confidence});
            continue;
        }

        // Accept: mark the consumer as fused by attaching an attribute.
        // (In a real pass, we'd actually merge the ops; here we just
        // record the decision and tag the op so downstream passes know.)
        for (auto& f : m.functions()) {
            Operation* consumer_op = find_defining_op(*f, consumer);
            if (consumer_op) {
                consumer_op->attributes.set(
                    "fused_with",
                    Attribute::make_integer(static_cast<i64>(producer)));
                consumer_op->name = "fused_" + consumer_op->name;
                break;
            }
        }

        stats_.fusions_accepted++;
        stats_.total_predicted_improvement += benefit.net_predicted_improvement;
        decisions_.push_back({producer, consumer, true, "accepted",
                              benefit.net_predicted_improvement,
                              benefit.confidence});
        changed = true;
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// Pass 4: LayoutAwareCopyElimination
// ===========================================================================
PreservedAnalyses LayoutAwareCopyElimination::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    auto analyzer = make_analyzer(m, am);
    analyzer.run();

    bool changed = false;
    // transpose(transpose(x)) -> x
    // Walk all ops, find transposes whose operand is also a transpose.
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.opcode != OP_TRANSPOSE || op.operands.empty()) continue;
            ValueId inner_vid = op.operands[0].id();
            Operation* inner = find_defining_op(*f, inner_vid);
            if (!inner || inner->opcode != OP_TRANSPOSE ||
                inner->operands.empty()) continue;

            // Verify the permutations are inverses.
            // For now, we only handle the common 2D case: perm=[1,0] twice.
            auto outer_perm_attr = op.attributes.get("perm");
            auto inner_perm_attr = inner->attributes.get("perm");
            if (!outer_perm_attr || !inner_perm_attr) continue;
            if (outer_perm_attr->kind != AttrKind::IntegerArray ||
                inner_perm_attr->kind != AttrKind::IntegerArray) continue;
            if (outer_perm_attr->ints.size() != 2 ||
                inner_perm_attr->ints.size() != 2) continue;
            // Both [1, 0]?
            if (outer_perm_attr->ints[0] == 1 && outer_perm_attr->ints[1] == 0 &&
                inner_perm_attr->ints[0] == 1 && inner_perm_attr->ints[1] == 0) {
                m.replace_all_uses(op.results[0], inner->operands[0]);
                ++stats_.transpose_transpose_eliminated;
                ++stats_.total_rewrites;
                changed = true;
            }
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// Pass 5: AliasAwareMemoryPlanning
// ===========================================================================
PreservedAnalyses AliasAwareMemoryPlanning::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};
    auto analyzer = make_analyzer(m, am);
    analyzer.run();
    auto& store = analyzer.store();

    // Group values by alias_set_id. Values with the same alias set can
    // share storage.
    std::unordered_map<u32, std::vector<ValueId>> by_alias_set;
    for (auto& [vid, tf] : store) {
        if (tf.alias_class.known &&
            tf.alias_class.value.kind == AliasKind::MustAlias) {
            by_alias_set[tf.alias_class.value.alias_set_id].push_back(vid);
        }
    }

    // For each alias set with >1 member, mark all but the first as
    // "reuses" the first. We do this by emitting OP_REUSE marker ops.
    // (In a real memory planner this would actually share the allocation.)
    bool changed = false;
    for (auto& [set_id, members] : by_alias_set) {
        if (members.size() < 2) continue;
        ValueId canonical = members[0];
        for (usize i = 1; i < members.size(); ++i) {
            // Mark this value as reusing `canonical`'s storage.
            // We don't actually mutate allocations here; we just record
            // the decision via an attribute on the defining op.
            for (auto& f : m.functions()) {
                Operation* def = find_defining_op(*f, members[i]);
                if (def) {
                    def->attributes.set("reuses_storage",
                        Attribute::make_integer(static_cast<i64>(canonical)));
                    ++stats_.buffers_merged;
                    // Approximate bytes saved: assume each merged buffer
                    // is at least 1KB.
                    stats_.bytes_saved += 1024;
                    changed = true;
                    break;
                }
            }
        }
    }

    if (!changed) return PreservedAnalyses::all();
    PreservedAnalyses pa;
    return pa;
}

// ===========================================================================
// UnifiedOptimizationPipeline (runs all five)
// ===========================================================================
PreservedAnalyses UnifiedOptimizationPipeline::run(Module& m, AnalysisManager& am) {
    stats_ = Stats{};

    PropertyDrivenSimplification p1;
    p1.run(m, am);
    stats_.property = p1.stats();

    RangeDrivenStrengthReduction p2;
    p2.run(m, am);
    stats_.range = p2.stats();

    CostGuidedFusion p3;
    p3.run(m, am);
    stats_.fusion = p3.stats();

    LayoutAwareCopyElimination p4;
    p4.run(m, am);
    stats_.layout = p4.stats();

    AliasAwareMemoryPlanning p5;
    p5.run(m, am);
    stats_.alias = p5.stats();

    return PreservedAnalyses::none();
}

} // namespace cg
