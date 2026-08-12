# cantors-gift

A multi-level, SSA-based tensor compiler with hard separation between **semantic
optimization**, **execution planning**, and **backend codegen**.

`cantors-gift` is *not* a wrapper around LLVM or MLIR. It owns its entire
optimization stack and delegates only the final act of encoding machine
instructions to thin emitter libraries (Xbyak/AsmJit on x86-64, an internal PTX
emitter on NVIDIA, etc.).

## Architectural skeleton

```
Frontend ─► Tensor IR ─► Analyses ─► Tensor Optimizer ─► Memory IR
        ─► Schedule IR ─► Search / Autotuner ─► Codegen IR ─► Backend ─► Executable
```

### The four IRs

| IR             | Answers                                             |
| -------------- | --------------------------------------------------- |
| **Tensor IR**  | *What mathematical computation is being performed?* |
| **Memory IR**  | *Where do tensors live and when?*                   |
| **Schedule IR**| *How should the computation execute?*               |
| **Codegen IR** | *What primitive operations does the backend need?*  |

The most important invariant: **never lower information earlier than
necessary.** A `matmul` stays a `matmul` until every optimization that needs to
see it as a matrix multiplication has run.

### What we own

```
Tensor semantics           (owned)
Tensor IR                  (owned)
Analyses                   (owned)
Tensor optimizations       (owned)
E-graphs                   (owned)
Fusion / recomputation     (owned)
Memory planning            (owned)
Schedule optimizer         (owned)
Autotuner                  (owned)
Cost model                 (owned)
Codegen IR                 (owned)
Instruction selection      (owned)
Schedule-level register-pressure estimation (owned, no physical allocator)
```

### What we delegate

```
x86-64 instruction encoding      → Xbyak / AsmJit (header-only libraries)
PTX text → SASS binary           → NVIDIA driver / toolchain
AMD code-object emission         → AMD assembler / toolchain
Vendor kernels (cuBLAS, cuDNN, …) → optional kernel candidates, never a dependency
```

We deliberately do **not** implement a physical register allocator. The
scheduler is register-pressure aware via analytical estimation; the emitter
library handles the rest.

## Repository layout

```
.
├── CMakeLists.txt
├── include/cg/        # public headers
├── src/               # implementation
├── tests/             # unit + integration tests (GoogleTest)
├── benchmarks/        # micro / kernel / model / regression benchmarks
├── tools/             # compiler, profiler, tuner CLI entry points
├── docs/              # architecture and design docs
└── third_party/       # vendored header-only libs (fetched on demand)
```

## Status

This is the foundational commit. The following subsystems are implemented and
unit-tested:

- **Core IR**: `Type`, `TensorType`, `Value`, `Operation`, `Block`, `Module`,
  `Builder`, textual `Printer`.
- **Shape system**: symbolic `DimExpr` DAG, `ConstraintSet`, constraint
  `Solver`, expression `Simplifier`, shape inference.
- **Layout system**: `Layout` as a composable index→address function,
  `StridedLayout`, layout equivalence, layout analysis.
- **Effects & Traits**: pure / side-effecting classification, commutative /
  associative / elementwise / reduction traits.
- **Analysis framework**: `AnalysisManager` with caching and automatic
  invalidation via `PreservedAnalyses`.
- **Pass infrastructure**: `Pass`, `PassManager`, `Pipeline`.
- **Real passes**: canonicalization, CSE, constant folding, DCE, shape
  inference, layout inference.
- **Schedule IR**: `IterationDomain`, `Schedule`, schedule transformations
  (split, tile, interchange, vectorize, parallelize, cache), `ScheduleSpace`.
- **Cost model**: `HardwareModel`, analytical `CostEstimator`.
- **Codegen IR**: vector loads/stores, FMA, reductions, barriers, async copies.
- **Backend interface**: `MachineBackend`, `TargetInfo`, `MachineEmitter`.
- **Runtime interface**: `Executable`, `Device`, `Stream`, `Allocator`,
  `KernelCache`.
- **E-graph core**: `ENode`, `EClass`, `EGraph`, rewrites, tensor-aware
  extraction.
- **Standard ops**: `add`, `sub`, `mul`, `matmul`, `relu`, `broadcast`,
  `reshape`, `transpose`, `reduce_sum`, `constant`, plus attributes for
  layouts, dtypes, and effects.

The CPU/CUDA/AMD backends are intentionally interface-only at this stage; the
optimization stack above them is fully real and exercised by the test suite.

## Building

Requires C++20, CMake ≥ 3.20, and Ninja (recommended).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests use GoogleTest, fetched on first configure via `FetchContent`.

## Design principles

1. **Multiple representations, not one giant IR.** Each IR is the smallest
   representation that answers one specific question.
2. **Tensor semantics are preserved for as long as possible.** `matmul` is
   *not* lowered to loops before fusion, layout optimization, and reduction
   optimization have run.
3. **Analyses are cached and invalidated automatically.** A pass returns
   `PreservedAnalyses`; the framework drops caches it can no longer trust.
4. **The optimizer never depends on a concrete backend.** All
   hardware-specific knowledge is exposed through `TargetInfo` and the cost
   model.
5. **Vendor libraries are candidates, not dependencies.** `cuBLAS`, `cuDNN`,
   `oneDNN`, `CUTLASS` may be selected by the optimizer as alternative
   implementations of a tensor op, but the compiler never requires them.
6. **No physical register allocator.** The scheduler estimates register
   pressure analytically and rejects infeasible candidates before codegen.

## License

Apache 2.0. See `LICENSE`.
