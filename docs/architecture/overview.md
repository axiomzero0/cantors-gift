# Architecture

`cantors-gift` is a multi-level, SSA-based tensor compiler. The central
architectural decision is **never lower information earlier than necessary**:
a `matmul` stays a `matmul` until every optimization that needs to see it as
a matrix multiplication has run.

## The four IRs

| IR             | Answers                                             |
| -------------- | --------------------------------------------------- |
| **Tensor IR**  | *What mathematical computation is being performed?* |
| **Memory IR**  | *Where do tensors live and when?*                   |
| **Schedule IR**| *How should the computation execute?*               |
| **Codegen IR** | *What primitive operations does the backend need?*  |

The most important invariant: **never lower information earlier than
necessary.**

## Data flow

```
                    INPUT GRAPH
                         │
                         ▼
                  ┌─────────────┐
                  │ Tensor IR   │
                  └──────┬──────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Analyze         │
                │ shape/layout/   │
                │ alias/dependence│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Optimize        │
                │ mathematics     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Optimize        │
                │ graph           │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Memory planning │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Schedule search │
                └────────┬────────┘
                         │
                    candidates
                         │
                         ▼
                ┌─────────────────┐
                │ Cost + autotune │
                └────────┬────────┘
                         │
                    best schedule
                         │
                         ▼
                ┌─────────────────┐
                │ Lowering IR     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Backend/codegen │
                └────────┬────────┘
                         │
                         ▼
                     EXECUTABLE
```

## Module dependency graph

```
                    core
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
         IR        Types      Utilities
          │
   ┌──────┼───────┬──────────┐
   ▼      ▼       ▼          ▼
 shape  layout  analysis   effects
   │      │       │          │
   └──────┴───────┴──────────┘
                │
                ▼
           optimization
                │
        ┌───────┼─────────┐
        ▼       ▼         ▼
      graph   egraph    memory
        │       │         │
        └───────┼─────────┘
                ▼
            schedule
                │
          ┌─────┴──────┐
          ▼            ▼
       cost model    autotuner
          │            │
          └─────┬──────┘
                ▼
             lowering
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
       CPU    CUDA      ROCm
        │       │        │
        └───────┼────────┘
                ▼
             runtime
```

The strong dependency rule: **high-level optimization must never depend on a
concrete backend.** Hardware-specific knowledge is exposed through
`TargetInfo` and the cost model.

## What we own, what we delegate

```
                    OUR CODE
════════════════════════════════════════════════════
Tensor IR
Shape solver
Layout algebra
E-graphs
Fusion
Memory planner
Schedule optimizer
Autotuner
Cost model
Codegen IR
Instruction selection
Schedule-level register-pressure estimation
════════════════════════════════════════════════════
                    LIBRARIES
════════════════════════════════════════════════════
AsmJit / Xbyak
PTX assembly/tooling
AMD assembly/tooling
════════════════════════════════════════════════════
                    HARDWARE
════════════════════════════════════════════════════
x86 / ARM / NVIDIA / AMD
```

Vendor libraries (cuBLAS, cuDNN, CUTLASS, rocBLAS, oneDNN, ...) are
**optional kernel candidates**, never a hard dependency.

## No physical register allocator

We deliberately do NOT implement a physical register allocator. The scheduler
estimates register pressure analytically and rejects infeasible schedule
candidates before codegen. The emitter library (Xbyak/AsmJit/PTX) handles
spilling. This is an explicit architectural decision documented in the
project history.

## Global Tensor Analysis (GTA)

GTA is the barrier between high-level optimization and final lowering. It
runs *after* local canonicalization, e-graphs, fusion, memory planning, and
scheduling, and *before* codegen IR lowering.

```
                    HIGH LEVEL
────────────────────────────────────────
Tensor IR
Shape analysis
Layout analysis
E-graphs
Fusion
Memory planning
Scheduling
Autotuning
────────────────────────────────────────
               GLOBAL BARRIER
────────────────────────────────────────
Global Tensor Analysis
Global Cost Model
Global Legality Check
Global Schedule Validation
────────────────────────────────────────
                    LOWERING
────────────────────────────────────────
Codegen IR
Target instructions
Codegen library
```

GTA is **iterative**: passes invalidate analyses, the next phase recomputes
them, and the driver may loop back to earlier phases if a later phase exposes
a new opportunity.

### Analyses bundled in GTA

| Analysis | What it knows |
|----------|---------------|
| `DataflowAnalysis` | use-def / def-use chains, fanout, fanin, critical path, topo order |
| `ShapeAnalysis` | per-value shapes |
| `LayoutAnalysis` | per-value layouts |
| `LifetimeAnalysis` | per-value [start, end] op range, peak live count |
| `ArithmeticIntensityAnalysis` | per-op FLOPs/bytes, bound classification (memory/compute/launch/latency) |
| `ParallelismAnalysis` | per-op independent items, reduction length, parallelism ratio |
| `ReuseAnalysis` | per-value materialize-vs-recompute decision |
| `GlobalAliasAnalysis` | view-of / slice-of / broadcast-of relationships |
| `GlobalCostAnalysis` | decomposed graph cost (execution + memory + launch + sync + spec + code-size) |

### Iterative driver

The `IterativeDriver` runs the pipeline in phases:

1. Canonicalize + Algebraic + SCCP + ConstFold + CSE + DCE
2. ShapeOpt + LayoutOpt
3. Fusion + DCE
4. ReductionOpt
5. CopyElim + MemoryPlanning
6. Specialization
7. **Global Barrier** (GTA + legality + decisions)

Between phases, analyses are invalidated and recomputed. The driver iterates
up to `max_iterations` (default 3) or until convergence (op count stable).
