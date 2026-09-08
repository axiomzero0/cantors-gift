# Brain-like ML Domain on Cantor's Gift

## TL;DR

The brain-like spatial neural graph ML algorithm has been added **as a new
computational domain on top of the existing Cantor's Gift compiler** —
not as a separate IR architecture. The math is unchanged; the execution
strategy is unchanged. We only added:

1. **Twelve new semantic primitive opcodes** (`OP_NEIGHBOR_QUERY`,
   `OP_PAIRWISE_DIST_SQ`, `OP_GAUSSIAN_KERNEL`, `OP_LATERAL_INHIBITION`,
   `OP_DELAY_FROM_DIST`, `OP_SPIKE`, `OP_SYNAPTIC_ARRIVAL`,
   `OP_ELIGIBILITY_UPDATE`, `OP_CREDIT_PROPAGATE`, `OP_ACTIVE_SET`,
   `OP_REGION_AGGREGATE`, `OP_STRUCTURAL_EPOCH`) registered in
   `OpRegistry` exactly like every other op.
2. **Builder convenience methods** so the front-end doesn't have to
   construct `AttributeDict`s manually.
3. **A high-level `BrainBuilder` facade** (`include/cg/brain/`) that
   emits canonical compositions (the distance-reuse pattern, the
   credit-assignment pattern, the event-driven pattern) with intent.
4. **Four new compile-task kinds** (`brain_distance_reuse`,
   `brain_credit_assignment`, `brain_event_driven`, `brain_full`)
   plumbed through the existing `compile()` API.
5. **Tests** (`tests/brain/`, `tests/python/test_brain.py`) verifying
   that the existing CSE pass collapses redundant distance
   computations and that the existing DCE pass preserves side-effecting
   ops.

No new IR. No new optimizer. No new backend. The whole point is that
Cantor's Gift already has the machinery we'd otherwise have been
inventing.

---

## Why this design

The original math specification describes a hybrid dynamical system on
a mutable directed 3D graph:

$$
\mathfrak M(t) = (\mathcal N, \mathcal E, \mathcal R, \mathcal S,
\mathcal W, \mathcal P, \mathcal C, \mathcal H, \mathcal X,
\mathcal A, \mathcal Q)
$$

with continuous evolution $d\mathfrak M/dt = F(\mathfrak M, t)$
between discrete events $\mathfrak M(t_e^+) = \Phi_e(\mathfrak M(t_e^-))$.

The workload is fundamentally:

> **mutable spatial dynamical graph → sparse event computation**

This is *not* a tensor graph → kernel workload in the strict sense, but
Cantor's Gift's existing separation between **semantic optimization**,
**execution planning**, and **backend codegen** is the right shape. Its
design keeps high-level semantics around until the relevant optimizations
have finished, which is exactly what we need to avoid accidentally
destroying the brain-mathematical meaning during execution planning.

---

## The single biggest optimization target

The naïve implementation computes pairwise distance² as

$$
\forall i, \forall j : \quad d_{ij}^2 = (x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2
$$

which is $O(N^2)$. But the math doesn't require every pair; the
majority of spatial interactions are of the form $d_{ij} < r$, i.e.

$$
\mathcal N_r(i) = \{j \mid \|\mathbf x_i - \mathbf x_j\|^2 < r^2\}
$$

That is captured by `OP_NEIGHBOR_QUERY`, which the compiler may lower
to a grid / hashed grid / octree / SIMD brute force / GPU spatial
partition as the cost model dictates.

More importantly, the same $d_{ij}^2$ value is reused by **five
downstream computations**:

| Consumer | Formula | Op |
|----------|---------|----|
| Affinity kernel | $K_{ij}^x = e^{-d^2/(2\sigma_x^2)}$ | `gaussian_kernel` |
| Lateral inhibition | $g_{ij}^{inh} = g_0 e^{-d^2/(2\sigma_{inh}^2)}$ | `lateral_inhibition` |
| Connection formation | $B_{ij} \supset K_{ij}^x$ | `gaussian_kernel` (same σ) |
| Region assignment | $\mathcal U_{ir} \supset K_x(i,r)$ | downstream of `gaussian_kernel` |
| Synaptic delay | $\tau_{ij}^{delay} = \tau_0 + \sqrt{d^2} / c_{ij}$ | `delay_from_dist` |

A naïve implementation would compute `d²` five times. The optimizer
must recognize `d²` as a **reusable value** and CSE it.

This is exactly what the existing CSE pass does. The CSE pass in
`src/optimization/cse/cse.cpp` dedupes pure operations whose
`(opcode, operands, attributes)` tuples match. Since
`pairwise_dist_sq(x)` has no attributes and one operand, two calls
with the same `x` are automatically deduped to one.

The unit test `BrainBuilder.CSECallsapesDuplicateDistances` in
`tests/brain/brain_tests.cpp` verifies this: it deliberately emits
five `pairwise_dist_sq(positions)` ops, then runs CSE, and asserts
that exactly one survives.

---

## The three-level invariant

The design discussion introduced a clean separation between
**mathematical semantics**, **execution semantics**, and
**optimization**. The brain domain implementation honors it:

### 1. Mathematical semantics

What the system means: $\dot v_i, \dot w_{ij}, \dot c_i, \dot p_i,
\dot \pi_{ir}$ and the event transitions. These are captured by the
twelve new opcodes, which are **pure** (so the optimizer may reorder
them) unless they model genuine side effects:

| Op | Pure? | Why |
|----|-------|-----|
| `pairwise_dist_sq` | yes | pure function of positions |
| `neighbor_query` | yes | pure function of positions + radius |
| `gaussian_kernel` | yes | elementwise pure |
| `lateral_inhibition` | yes | elementwise pure |
| `delay_from_dist` | yes | elementwise pure |
| `synaptic_arrival` | yes | reduction over events |
| `eligibility_update` | yes | elementwise pure |
| `credit_propagate` | yes | reduction over weights |
| `active_set` | yes | top-K selection |
| `region_aggregate` | yes | reduction over regions |
| `spike` | **no** | emits a discrete event (must not be reordered) |
| `structural_epoch` | **no** | barrier across which plasticity may run |

### 2. Execution semantics

How those equations can be evaluated without changing their meaning:

- **Event batching** — multiple `synaptic_arrival` ops with the same
  weight matrix can be fused.
- **Spatial indexing** — `neighbor_query` can lower to grid / hashed
  grid / octree as the cost model selects.
- **State fusion** — multiple `eligibility_update` ops on the same
  tensor with different $(\Delta t, \tau_e)$ can be merged.
- **Lazy decay** — `eligibility_update` can be deferred across
  multiple timesteps if the consumer doesn't read the updated value.
- **Sparse activation** — `active_set` reduces work downstream.

None of these change the math; they only change *when* and *how* it
is computed.

### 3. Optimization

How Cantor's Gift makes those execution semantics fast:

- **CSE** — collapses duplicate `pairwise_dist_sq` calls (the big win).
- **Fusion** — fuses `pairwise_dist_sq` with its elementwise consumers
  (`gaussian_kernel`, `lateral_inhibition`, `delay_from_dist`).
- **Layout** — chooses SoA vs AoS for the positions tensor.
- **Vectorization** — packs the elementwise ops into SIMD lanes.
- **Tiling** — tiles the pairwise kernel for cache locality.
- **Autotuning** — the Bayesian autotuner searches tile sizes.
- **E-graph superoptimization** — discovers algebraic rewrites like
  $g_0 \cdot \exp(-d^2 / (2\sigma_{inh}^2)) = g_0 \cdot K_{inh}(d^2)$,
  i.e. `lateral_inhibition` is `mul(g0, gaussian_kernel(d2, sigma_inh))`.

### Invariant

$$
\boxed{\text{optimization may change execution, but never the mathematical trajectory.}}
$$

This is enforced structurally: every math op is `pure`, so any
reordering produces a mathematically equivalent computation. Every
state-changing op (`spike`, `structural_epoch`) is side-effecting, so
the existing effect system prevents the optimizer from moving or
eliminating it.

---

## Architecture (unchanged from Cantor's Gift)

```
                 Brain Mathematics
                       │
                       ▼
          Mutable Spatial Neural Graph
                       │
                       ▼
              Cantor's Gift
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Graph analysis   Spatial analysis  Temporal analysis
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                Execution planning
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       CPU/event engine      GPU/batched engine
             │                   │
             └─────────┬─────────┘
                       ▼
                    Codegen
```

The neural mathematics remains independent of the optimization
strategy. Aggressive execution optimization cannot change the
learning dynamics because every math op is pure and every
state-changing op is side-effecting.

---

## File map

| File | Purpose |
|------|---------|
| `include/cg/ir/operation.hpp` | New opcode constants `OP_NEIGHBOR_QUERY` … `OP_STRUCTURAL_EPOCH` |
| `src/ir/ops.cpp` | Registration of the new opcodes with `OpRegistry` (traits + type inference) |
| `include/cg/ir/builder.hpp` | Convenience methods on `Builder` |
| `src/ir/builder.cpp` | Convenience method implementations |
| `include/cg/brain/brain_builder.hpp` | High-level patterns (`DistanceReusePattern`, etc.) |
| `src/brain/brain_builder.cpp` | Pattern implementations |
| `include/cg/engine/compile_api.hpp` | Brain fields on `CompileTask` |
| `src/engine/compile_api.cpp` | `build_brain_*_ir()` + dispatch in `build_ir()` |
| `python/bindings.cpp` | Python exposure of brain ops + task fields |
| `tests/brain/brain_tests.cpp` | C++ unit tests (CSE collapse, side-effect preservation, etc.) |
| `tests/python/test_brain.py` | Python integration tests |

---

## Usage

### From Python (high-level compile API)

```python
import cantors_gift as cg

task = cg.CompileTask()
task.kind = "brain_distance_reuse"   # or brain_full / brain_credit_assignment / brain_event_driven
task.N = 1024
task.D = 3
task.sigma_x = 0.5
task.sigma_inh = 0.3
task.g0 = 1.0
task.tau0 = 0.001
task.inv_speed = 0.005
task.dtype = cg.DType.F32
task.hardware = "a100"

result = cg.compile(task, print_ir=True)
print(result.ir_text)
print(result.ops_before, "->", result.ops_after)
```

### From Python (manual IR construction)

```python
import cantors_gift as cg

m = cg.Module()
f = m.create_function(
    "brain_k",
    [cg.tensor_type([256, 3], cg.DType.F32)],
    [cg.tensor_type([256, 256], cg.DType.F32)],
)
b = cg.Builder(f)
positions = f.args[0]

# Emit pairwise_dist_sq ONCE — every downstream op reuses this Value.
d2 = b.pairwise_dist_sq(positions)
kx = b.gaussian_kernel(d2, 0.5)
b.output_tensor(kx)

# Optimize.
am = cg.AnalysisManager(m)
driver = cg.IterativeDriver(am)
driver.set_hardware(cg.HardwareModel.generic_cpu())
driver.run(m)
print(cg.to_string(m))
```

### From C++

```cpp
#include "cg/brain/brain_builder.hpp"
#include "cg/ir/builder.hpp"
#include "cg/optimization/iterative_driver.hpp"

cg::Module m;
auto* f = m.create_function("brain_k",
    {cg::make_tensor_type({256, 3}, cg::DType::F32)},
    {cg::make_tensor_type({256, 256}, cg::DType::F32)});
cg::Builder b(f);
auto positions = f->args()[0];

auto pattern = cg::brain::build_distance_reuse_pattern(
    b, positions, /*sigma_x=*/0.5, /*sigma_inh=*/0.3,
    /*g0=*/1.0, /*tau0=*/0.001, /*inv_speed=*/0.005);
b.output_tensor(pattern.kx);

cg::AnalysisManager am(m);
cg::IterativeDriver driver(am);
driver.set_hardware(cg::HardwareModel::generic_cpu());
driver.run(m);
std::cout << cg::to_string(m) << "\n";
```

---

## What's next

The brain domain is wired end-to-end but does not yet have:

- **Backend lowering** for the brain ops. The PTX / x86 / AMD backends
  will currently emit `OP_PAIRWISE_DIST_SQ` as an unknown op. The right
  next step is to add lowering patterns that translate the brain
  primitives into the existing elementwise / reduction / gather ops
  before codegen.
- **Spatial index selection** in the cost model. The optimizer should
  recognize `neighbor_query` and choose between grid / hashed grid /
  octree based on $N$, $r$, and the hardware.
- **Sparse event queue** integration with the runtime. `OP_SPIKE` is
  currently a marker; the runtime should actually enqueue events.
- **E-graph rewrites** specific to the brain primitives, e.g.
  `lateral_inhibition(d², g0, σ) == mul(g0, gaussian_kernel(d², σ))`.

These are extensions to the existing framework, not new architecture.
