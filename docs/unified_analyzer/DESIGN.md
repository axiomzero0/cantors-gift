# VORTEX Unified Tensor Analyzer — Design Document

## What this is

The Unified Tensor Analyzer is a **global abstract-interpretation +
constraint-solving + cost-modeling system** that replaces the pile of
independent analyses with a single shared **Tensor Knowledge Graph**.

Instead of:

```
ShapeAnalysis
LayoutAnalysis
AliasAnalysis
CostAnalysis
LivenessAnalysis
...
```

each computing in isolation, all analyses read from and write to a single
`FactStore`. Analyses feed each other:

```
shape known
   ↓
stride known
   ↓
layout known
   ↓
memory traffic known
   ↓
arithmetic intensity known
   ↓
roofline classification
   ↓
fusion profitability known
```

## Architecture

```
                       Tensor IR
                          │
                          ▼
                 ┌─────────────────┐
                 │   FactStore     │  ← single source of truth
                 └────────┬────────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
       Shape          Layout           Algebra
          │               │                │
          ▼               ▼                ▼
       Stride         Memory          Properties
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                  Dependence Analysis
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        Lifetime                   Parallelism
             │                         │
             └────────────┬────────────┘
                          ▼
                    Cost Analysis
                          │
                          ▼
                   Hardware Model
                          │
                          ▼
                 Global Knowledge
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
       E-Graph          Fusion          Scheduler
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                    Fixed Point
                          │
                          ▼
                       Lowering
```

## The fact lattice

Every SSA value gets a `TensorFacts` object. Each field is an abstract
domain that supports a lattice join (∨), a "top" (unknown), and a
"bottom" (contradiction).

```cpp
struct TensorFacts {
    Fact<std::vector<Dimension>> shape;
    Fact<u32> rank;
    Fact<DType> dtype;
    Fact<DeviceId> device;
    Fact<MemorySpace> memory_space;
    Fact<u32> alignment_bytes;

    Fact<LayoutPtr> layout;
    Fact<std::vector<DimExprPtr>> strides;
    Fact<bool> is_row_major_contiguous;

    Fact<ValueRange> value_range;
    Fact<TensorProperty> properties;
    Fact<double> constant_value;
    Fact<bool> constant_value_known;

    Fact<AliasClass> alias_class;
    Fact<u32> birth_op, death_op, num_users;
    Fact<std::vector<ValueId>> consumers;

    Fact<ReductionInfo> reduction;

    Fact<u64> estimated_flops, estimated_bytes_read, estimated_bytes_written;
    Fact<double> arithmetic_intensity;

    Fact<u32> reuse_distance;
    Fact<double> reuse_factor;
    Fact<CacheBehavior> cache_behavior;

    Fact<u64> independent_items;
    Fact<bool> has_reduction_dim;
    Fact<u64> reduction_length;
};
```

Every fact is wrapped in `Fact<T>` which carries:
- `value` — the actual fact
- `confidence` — `Proven` / `Derived` / `Estimated` / `Profiled` / `Speculative`
- `provenance` — the rule + operand chain that produced it
- `known` — false = "unknown" (top of the lattice)

### Lattice join semantics

When a new fact arrives, we join it with the existing fact:

```
unknown ∨ x      = x
x      ∨ unknown = x
x      ∨ x       = x
x      ∨ y       = (lower confidence wins; on tie, last-writer-wins)
```

Lower numeric `Confidence` value = more trusted
(`Proven=0 < Derived=1 < Estimated=2 < Profiled=3 < Speculative=4`).
A `Proven` fact always overrides an `Estimated` fact. This is what makes
the analyzer safe: **legality uses only Proven/Derived facts; profitability
may use Estimated/Profiled/Speculative facts**.

## Abstract domains

### Dimension

A dimension is NOT just an `int64_t`. It's a triple:

```cpp
struct Dimension {
    Bound bound;  // exact + lower + upper
    std::vector<i64> known_divides;
};
```

A dimension can therefore be:

| State | exact | lower | upper |
|-------|-------|-------|-------|
| `1024` | 1024 | 1024 | 1024 |
| `N` | N | 1 | ∞ |
| `N + 1` | N+1 | 2 | ∞ |
| `[1, ∞)` | unknown | 1 | ∞ |
| `[16, 256]` (tile) | unknown | 16 | 256 |

This lets the compiler answer questions like "is this dimension positive?"
or "is it a multiple of 32?" without generating code first.

### ValueRange

Tracks sign + magnitude information for tensor element values:

```cpp
struct ValueRange {
    std::optional<double> lower, upper;
    bool possibly_negative, possibly_zero, possibly_nan, possibly_inf;
};
```

This lets the analyzer simplify:
- `relu(x) → x` when `x >= 0`
- `abs(x) → x` when `x >= 0`
- `log(x)` is safe when `x > 0`

### TensorProperty (bitset lattice)

A tensor can be `Constant + Diagonal + Sparse` simultaneously. We use a
bitset so all properties compose:

```cpp
enum class TensorProperty : u32 {
    None, Constant, Zero, One, Identity, Diagonal, Symmetric,
    Permutation, BroadcastConst, Sparse, BlockSparse,
    TriangularLower, TriangularUpper, Dense,
};
```

Transfer functions:
- `matmul(A, Identity) → A` (Identity transfers to result)
- `mul(Zero, x) → Zero`
- `mul(One, x) → x`
- `add(Zero, x) → x`
- `transpose(Symmetric) → Symmetric`
- `transpose(Diagonal) → Diagonal`

### AliasClass

```cpp
struct AliasClass {
    AliasKind kind;  // NoAlias / MayAlias / MustAlias
    u32 alias_set_id;
};
```

Views/slices/transposes inherit the parent's alias set (MustAlias). Fresh
allocations get NoAlias. This is critical for memory planning, in-place
operations, and fusion legality.

## Propagators

Each analysis is a `FactPropagator` that reads facts, computes derived
facts, and writes them back with provenance + confidence. All propagators
are **idempotent** — running them twice with the same inputs produces the
same output, which makes fixed-point iteration safe.

| Propagator | Produces |
|------------|----------|
| `ShapePropagator` | shape, rank, dim_bounds |
| `LayoutPropagator` | layout, strides, contiguity |
| `ConstantPropagator` | constant_value, Constant/Zero/One properties |
| `PropertyPropagator` | TensorProperty lattice (matmul×Identity, mul×Zero, etc.) |
| `RangePropagator` | ValueRange (relu≥0, exp>0, sigmoid∈[0,1], tanh∈[-1,1]) |
| `AliasPropagator` | AliasClass (MustAlias for views, NoAlias for fresh allocs) |
| `LifetimePropagator` | birth_op, death_op, num_users |
| `ReductionPropagator` | ReductionInfo (axes, associativity, commutativity, identity) |
| `DependencePropagator` | dependence_edges (Full/Slice/Reduction/Broadcast/LayoutOnly) |
| `CostPropagator` | FLOPs, bytes, arithmetic intensity, CacheBehavior |

## Iterative fixed-point convergence

The `UnifiedAnalyzer` runs all propagators in a worklist until no new
facts are produced:

```cpp
while (changed) {
    for (auto& p : propagators_) {
        if (p->run(store) > 0) changed = true;
    }
    if (!changed) break;
    changed = false;
}
```

Convergence is **guaranteed** because every abstract domain here is finite
(or has finite height in the lattice). In practice the analyzer converges
in **2-4 iterations** for typical tensor programs.

### Convergence metrics

```cpp
struct AnalyzerMetrics {
    u32 iterations;
    u32 facts_discovered;
    u32 worklist_processed;
    double latency_sec;
    u32 contradictions;

    struct PropagatorStats {
        std::string name;
        u32 runs, facts_produced;
        double total_sec;
    };
    std::vector<PropagatorStats> per_propagator;

    double mean_prediction_error;  // |predicted - actual| / actual
    u32 predictions_evaluated;
};
```

These metrics are themselves benchmarked. See
`benchmarks/benchmark_unified_analyzer.py`.

## Provenance + confidence

Every fact carries its derivation chain:

```cpp
struct Provenance {
    std::string rule;       // "ShapeInference", "MulZero", "BayesianCostModel"
    u32 source_op;          // OpId the fact was derived from
    std::vector<ValueId> operands;
    std::string explanation;
};
```

When the compiler makes a terrible decision, you can ask "why?" and get
a proof chain back. The fusion benefit report is the canonical example:

```cpp
struct FusionBenefitReport {
    bool can_fuse;
    std::string legality_reason;

    double saved_bytes;
    double saved_kernel_launches;
    double saved_runtime_sec;

    double added_register_pressure;
    double occupancy_delta_pct;
    double critical_path_delta_pct;

    double net_predicted_improvement;  // 0.178 = 17.8% faster
    Confidence confidence;             // Proven / Estimated / Speculative

    std::vector<Provenance> reasons;   // WHY?
};
```

Example output:

```
Fusion benefit:
  saved_bytes: 262144
  saved_kernel_launches: 1
  added_register_pressure: 256
  occupancy_delta_pct: -1.5625
  net_predicted_improvement: 1526.73
  confidence: estimated
  reason: FusionBenefitAnalysis - saved_bytes from producer.estimated_bytes_written (Proven);
          runtime estimate from hardware model (Estimated)
```

The optimizer can now decide:

```cpp
if (store.can_fuse(producer, consumer) &&
    store.fusion_benefit(producer, consumer).net_predicted_improvement
        > threshold) {
    fuse(producer, consumer);
}
```

rather than every pass implementing its own half-baked version of
"is this probably a good idea?".

## Hardware parameterization

Target-independent facts (shape, layout, properties, alias, lifetime)
are stored WITHOUT hardware context — they describe the program, not the
program's behavior on a specific machine.

Target-dependent predictions (cache hit rate, predicted runtime,
occupancy) are stored with `Confidence::Estimated` and parameterized by
the `HardwareModel`:

```cpp
void FactStore::set_hardware(HardwareModel hw) {
    hw_ = std::move(hw);
    hw_set_ = true;
    invalidate_cost_facts();  // force re-derivation
}
```

The same IR can be analyzed for different machines without re-deriving
the target-independent facts.

## Design rules

1. **The analysis engine should NEVER be an optimization pass.** It is a
   persistent source of truth that passes query. The analyzer never
   mutates the IR.

2. **Optimization legality uses only Proven/Derived facts.** Optimization
   profitability may use Estimated/Profiled/Speculative facts. Never let
   an approximate cost model accidentally become a correctness oracle.

3. **Every fact has provenance.** When the compiler makes a terrible
   decision, you should be able to ask "why?" and get a proof chain back.

4. **`Unknown` is a legitimate state.** Don't force every field to be
   populated. Analyses refine facts lazily.

5. **Propagators are idempotent.** Running the same propagator twice
   with the same inputs produces the same output. This is what makes
   fixed-point iteration safe.

6. **The lattice has finite height.** Convergence is guaranteed.

## Benchmark results

From `benchmarks/benchmark_unified_analyzer.py`:

| Workload | Iters | Facts | Tensors | C++ us |
|----------|-------|-------|---------|--------|
| Small matmul + bias + relu (128×256×512) | 2 | 75 | 6 | 82 |
| Large matmul + bias + relu (1024³) | 2 | 75 | 6 | 28 |
| Elementwise chain (1024-wide, depth 10) | 2 | 154 | 11 | 49 |
| Elementwise chain (1024-wide, depth 50) | 2 | 714 | 51 | 241 |
| Reduction chain (1024×1024, depth 5) | 2 | 84 | 6 | 26 |
| Layout chain (1024×1024, depth 10) | 2 | 121 | 11 | 37 |

Observations:
- All workloads converge in **2 iterations** (fixed point).
- **Zero contradictions** — the lattice joins are consistent.
- Latency scales **linearly** with tensor count.
- ~12 facts discovered per tensor (shape, rank, dtype, layout, strides,
  alias, lifetime, cost, etc.).

## Roadmap: where this goes next

### Tier 1 — Closer to a real semantic engine

- **Incremental dataflow analysis**: currently every propagator re-derives
  everything from scratch. A worklist driver that only re-runs propagators
  whose inputs changed would cut latency by ~10× for large graphs.
- **SMT-backed shape constraints**: the current `Dimension` lattice only
  handles constant bounds. Real symbolic reasoning (M_A == M_B) needs an
  SMT solver or a constraint-graph unification.
- **Profile feedback loop**: `report_actual_runtime` is a stub. Wire it
  to the autotuner's measured runtimes and use it to refine the cost
  model's predicted_runtime fact.

### Tier 2 — Cross-cutting analyses

- **Kernel boundary optimization**: search over all partitions of an op
  chain. The fact store already has the cost building blocks; we need a
  search procedure that uses them.
- **Tile lifetime optimization**: globally optimize tile production /
  consumption / eviction rather than treating each tensor as one allocation.
- **Recomputation vs materialization**: the `ReuseAnalysis` exists but
  needs to be wired to the fact store's `fusion_benefit` query.

### Tier 3 — The really ambitious stuff

- **Algebraic domain switching**: a tensor expression changes representation
  (dense → diagonal → sparse → low-rank) based on downstream cost. The
  `TensorProperty` lattice is the foundation; the next step is choosing
  the representation that minimizes total downstream cost.
- **Error-budget optimization**: define `max_error = ε` and search over
  dtype / reduction order / approximation / algorithm to minimize runtime
  subject to `error ≤ ε`. The `Confidence` system is the foundation; we
  need a constraint solver that reasons about numerical accuracy.
- **Learned rewrite discovery**: train a model to propose new rewrite
  rules from observed performance data. The fact store gives us a safe
  framework for evaluating proposed rewrites: just add them and let the
  cost model pick.

## The big idea

A conventional compiler optimizes a program. **VORTEX should optimize the
mathematical representation of the program before deciding what program
to execute.** The unified Tensor Knowledge Graph is the foundation for
that: it lets the compiler reason about tensor algebra, sparsity,
low-rank structure, and error budgets as first-class objects, not as
patterns to be matched in emitted code.
