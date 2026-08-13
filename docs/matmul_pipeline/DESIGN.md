# VORTEX Matmul Pipeline — Design & Roadmap

## What this is

The `MatmulPipeline` orchestrator (`src/pipeline/matmul_pipeline.cpp`) is the
end-to-end path that takes a matmul-shaped workload from IR construction to
an autotuned, tensor-core-aware schedule. It exists to make every stage of
the VORTEX compiler inspectable, measurable, and individually testable.

## Pipeline stages

```
            IR construction
                  │
                  ▼
            GTA  (arithmetic intensity analysis with three-level breakdown)
                  │
                  ▼
            E-graph saturation  (TC eligibility, FMA, layout, algebraic rules)
                  │
                  ▼
            Schedule-space generation  (tiles × vector × TC on/off)
                  │
                  ▼
            V2 cost-model pruning  (wave quantization, occupancy, L2 hit rate,
                  │                  bank conflicts, pipeline overlap)
                  ▼
            Bayesian autotuner  (GP + EI on top-k candidates)
                  │
                  ▼
            Codegen  (PTX emission via ptx_mma)
```

The compile-time vs autotuner separation is preserved by design:

- The analytical cost model narrows the schedule space from O(100s) of
  candidates to O(10s), using only static features. This is cheap and
  deterministic.
- The Bayesian autotuner then spends actual benchmark measurements only on
  the top-k promising candidates, using GP+EI to decide where each
  measurement is most informative.
- This is NOT brute force with a fancy hat. The analytical model does the
  heavy lifting of pruning; the autotuner refines.

## Three-level roofline breakdown

The classic roofline model gives one number: arithmetic intensity. In
practice, that single number is misleading because it conflates three
different things:

1. **Graph arithmetic intensity** = total FLOPs / total bytes moved
   between ops (the "naive" intensity, before any fusion or scheduling).
   This is what the user would compute by hand from the source program.

2. **Kernel arithmetic intensity** = FLOPs / bytes-after-fusion. Once
   bias+relu are fused into the matmul epilogue, the intermediate matmul
   output never goes to memory. The kernel intensity reflects this.

3. **Effective arithmetic intensity** = FLOPs / bytes that actually miss
   every cache level. This accounts for:
   - L2 cache reuse (B matrix reused across M/m_tile CTA blocks; if K*N
     fits in L2, hit rate approaches 0.8)
   - Shared-memory reuse (A,B tiles loaded once per K-tile, reused
     K/k_tile times across the reduction)
   - Register reuse (the M_tile × N_tile accumulator lives entirely in
     registers, never touching memory)
   - Coalesced global-memory transactions (vectorized loads reduce
     transaction count)

A graph can look compute-bound at the graph level (high FLOPs/byte) but
become memory-bound at the effective level if the schedule has poor cache
reuse, or vice versa.

For the 1024³ F32 matmul on A100:
- Graph intensity: 170.67 FLOP/byte
- Kernel intensity: 170.67 FLOP/byte (default schedule, no tiling)
- Effective intensity: 365.71 FLOP/byte (L2 reuse doubles it)
- F32 ridge: 9.75 FLOP/byte
- F16 TC ridge: 156.00 FLOP/byte

At the effective level, the kernel is firmly compute-bound — which is
exactly what we want for a matmul.

## SM utilization ≠ occupancy

The most important bug fix in this revision is exposing **SM utilization**
as a first-class metric, separate from **occupancy**.

- **Occupancy** = (warps per SM) / (max warps per SM). This is a per-SM
  metric that tells you whether each active SM has enough warps to hide
  latency.

- **SM utilization** = (SMs that actually receive a block) / (total SMs),
  averaged across waves. This is a GPU-wide metric that tells you whether
  you're using the hardware at all.

A kernel with 64 blocks on a 108-SM A100 can have 100% occupancy (if
each block has enough warps), but only 59.26% SM utilization — 44 SMs
sit completely idle for the duration of the wave.

### Concrete numbers from the test suite

| num_blocks | num_sms | num_waves | sm_utilization_pct | tail_efficiency_pct | idle_sms_in_tail |
|------------|---------|-----------|--------------------|---------------------|------------------|
| 64         | 108     | 1         | 59.26%             | 59.26%              | 44               |
| 65         | 108     | 1         | 60.19%             | 60.19%              | 43               |
| 108        | 108     | 1         | 100.00%            | 100.00%             | 0                |
| 200        | 108     | 2         | 92.59%             | 92.59%              | 16               |
| 216        | 108     | 2         | 100.00%            | 100.00%             | 0                |

The wave-quantization penalty is non-zero whenever there is a partial
tail wave. This includes the case `num_blocks < num_sms` (a single
partial wave), which was previously mislabeled as "perfect utilization".

### Why this matters for scheduling

"One wave" is not automatically better than multiple waves. A schedule
with 200 blocks on 108 SMs (2 waves, 92.59% utilization) may be faster
than a schedule with 64 blocks (1 wave, 59.26% utilization) even though
the latter has fewer waves, because the former keeps the GPU busy.

The autotuner now sees `sm_utilization_pct` as a feature and can prefer
schedules that fill the GPU.

## Schedule model

The v2 cost model computes:

```
T_kernel = max(T_compute, T_memory)            // overlapped
         + T_stall                               // occupancy-induced stalls
         + T_wave_quant                          // tail-effect penalty
         + T_bank_conflict                       // shared-mem bank conflicts
```

Where:
- `T_compute = FLOPs / peak_flops(dtype, uses_tc)`
- `T_memory = (bytes_global + bytes_shared) / bandwidth`
- `T_stall = overlapped × (1 - occupancy_factor) × stall_cycles_per_warp / 4`
- `T_wave_quant = overlapped × wave_quantization_penalty(num_blocks, num_sms)`
- `T_bank_conflict = (shared_bytes / shared_bw) × (conflict_ways - 1)`

The wave-quantization penalty formula:

```
penalty = (1 - remainder/num_sms) / (full_waves + 1)
```

This is non-zero whenever `num_blocks % num_sms != 0`, including the
single-partial-wave case. The penalty is then multiplied by the
overlapped time to get the absolute seconds wasted.

## E-graph rules wired in

The e-graph saturator runs the full rule library, filtered by numerical
mode. Under `FastMath`, the following rule categories are active:

- **Algebraic identities**: `add(x, 0) → x`, `mul(x, 1) → x`,
  `mul(x, 0) → 0`, `sub(x, x) → 0`, `neg(neg(x)) → x`
- **Commutativity**: `add(a,b) ↔ add(b,a)`, same for `mul`, `max`, `min`
- **Associativity** (accuracy-risky): `add(add(a,b),c) ↔ add(a,add(b,c))`
- **FMA formation** (accuracy-risky): `add(mul(a,b),c) → fma(a,b,c)`
- **Cast propagation**: `cast(cast(x,t1),t2) → cast(x,t2)`
- **Layout movement**: `transpose(transpose(x)) → x`,
  `transpose(add(a,b)) → add(transpose(a),transpose(b))`,
  `transpose(relu(x)) → relu(transpose(x))`, etc.
- **Reduction reassociation**: `reduce_sum(add(a,b)) → add(reduce_sum(a), reduce_sum(b))`
- **TC eligibility** (FastMath only): `matmul(A_f32, B_f32) → matmul(cast(A,f16), cast(B,f16))`
  — unlocks 16x throughput on A100 tensor cores (312 TF F16 TC vs 19.5 TF F32)
- **Domain rules**: `relu(x) → max(x, 0)`, `relu(relu(x)) → relu(x)`,
  `add(add(x,a),b) → add(x, add(a,b))` (reassociate under Relaxed)

Under `Strict` mode, only the `Pure` rules fire — no TC eligibility, no
FMA, no reassociation. This gives the user explicit control over the
accuracy/performance tradeoff.

## Tier 1/2/3 optimization roadmap

The following optimizations are not yet implemented but are on the
roadmap. They are organized by impact and complexity.

### Tier 1 — High-impact, foundational

These are the optimizations that would give the biggest wins for the
broadest range of tensor programs.

| Optimization | Description |
|--------------|-------------|
| **Global layout optimization** | Propagate layout (row-major / col-major / blocked) through the graph as metadata. Delay or eliminate physical transposes. The e-graph is well-suited for this: `transpose(transpose(x)) → x` is already wired in; the next step is recognizing `transpose(matmul(A,B)) → matmul(B^T, A^T)` and choosing the cheaper form. |
| **Kernel boundary optimization** | Instead of asking "what can we fuse?", ask "where should kernels exist at all?". For a chain `A → B → C → D → E → F`, search over all partitions `[A B C][D E][F]` vs `[A B][C D E F]` vs `[A B C D E F]`. The optimal boundary depends on register pressure, occupancy, parallelism, memory traffic, launch overhead. |
| **Tensor contraction reordering** | For `A[i,j] B[j,k] C[k,l]`, choose `(A @ B) @ C` or `A @ (B @ C)`. The FLOP count can differ massively. This is query optimization for tensor algebra — the same kind of cost-based search that SQL planners do for joins. |
| **Reduction tree synthesis** | Don't treat `sum(x)` as one primitive. Synthesize the reduction tree structure (linear / tree / warp / block / hierarchical) based on shape, dtype, SIMD width, GPU warp size, cache, parallelism, numerical precision. |
| **Recomputation vs materialization** | For `A → expensive → B → {C, D}`, decide whether to materialize B (compute once, store, load twice) or recompute (compute B twice, never store). The decision depends on compute cost, tensor size, memory bandwidth, cache residency, fanout. |
| **Broadcast elimination** | For `A[M,N] + broadcast(B[N])`, don't materialize the broadcast. Represent it as an indexing transformation. The broadcast becomes zero-cost metadata. |
| **Fusion/splitting** | The inverse of fusion. A huge fused kernel can become slower because of register explosion, occupancy collapse, instruction cache pressure, serialization. The compiler should be able to split a fused kernel at the optimal boundary. |

### Tier 2 — Important for specific patterns

| Optimization | Description |
|--------------|-------------|
| **Symbolic tensor representations** | Represent `ones(M,N)`, `zeros(M,N)`, `IdentityTensor`, `DiagonalTensor`, `BroadcastTensor`, `ViewTensor` symbolically. Don't materialize until necessary. `A @ IdentityTensor → A` should be a zero-cost rewrite. |
| **Partial tensor reuse** | `A[:, :K] @ B` and `A[:, K:] @ B` might share useful subcomputations. This is much deeper than scalar CSE. |
| **Dimension elimination** | `A[M,1,N]` should become `A[M,N]` internally. `reshape[M,1,N]` can become `reshape[M,N]`. Sounds trivial; across a giant graph it isn't. |
| **Shape specialization** | `M=1` means matrix becomes vector. `K=1` collapses GEMM into much simpler ops. Aggressively specialize these cases. |
| **Sparsity/structure propagation** | If analysis proves `A` is 70% zeros, or block-sparse, or triangular, or diagonal — choose a sparse representation/kernel. |
| **Contraction-path optimization** | Take an `einsum(...)` and search for the contraction tree that minimizes FLOPs, temporary memory, communication. This is potentially a major feature — VORTEX could become absurdly good at einsum-style workloads. |

### Tier 3 — Ambitious, research-grade

| Optimization | Description |
|--------------|-------------|
| **Algebraic domain switching** | A tensor expression changes representation based on downstream cost: `dense → diagonal → sparse` or `matrix → low-rank` or `tensor → separable representation`. The optimizer chooses the representation that minimizes total downstream cost. This is much closer to automatic algorithm selection than traditional compiler optimization. |
| **Low-rank recognition** | For `A ≈ U @ V`, recognize or exploit low-rank structure when explicitly represented. `A @ x` becomes `U @ (V @ x)`, potentially reducing complexity dramatically. Requires explicit mathematical/approximation semantics — silently changing numerical accuracy is not OK. |
| **Error-budget optimization** | Define `max_error = ε`. Search over dtype, reduction order, approximation, fusion, algorithm to minimize runtime subject to `error ≤ ε`. This is a constraint optimization problem over tensor programs. Very powerful for scientific computing. |
| **Automatic precision selection** | Analyze where precision is actually needed: `FP32 → FP16 → FP32 accumulation` or `FP32 → BF16` where numerically safe. Controlled by explicit numerical modes (strict / balanced / fast) rather than silently sacrificing accuracy. |
| **Learned tensor rewrite discovery** | Train a model to propose new rewrite rules from observed performance data. The e-graph gives us a safe framework for evaluating proposed rewrites: just add them and let the cost model pick. |

### The big idea

A conventional compiler optimizes a program. **VORTEX should optimize the
mathematical representation of the program before deciding what program
to execute.** That's the tier-3 vision: the compiler reasons about
tensor algebra, sparsity, low-rank structure, and error budgets as
first-class objects, not as patterns to be matched in emitted code.

## Benchmark

The benchmark script `benchmarks/benchmark_matmul_pipeline.py` runs the
full pipeline across the shape sweep:

```
1024×1024×1024   (baseline)
1025×1024×1024   (odd M)
1024×1025×1024   (odd K)
1024×1024×1025   (odd N)
1000×1000×1000
512×4096×512     (skinny K)
4096×512×4096    (fat K)
127×127×127      (small — exposes tail effects)
```

For each shape it reports the full compile-time breakdown, the three-level
roofline breakdown, SM utilization, and the best schedule found.

Run it with:

```bash
python3.13 benchmarks/benchmark_matmul_pipeline.py
```

Results are saved to `benchmarks/matmul_pipeline_results.json`.
