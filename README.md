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

This is the third foundational commit. The compiler now has real backends
that produce actual machine code, an e-graph superoptimizer wired into the
pipeline, and a Bayesian autotuner.

### Backends (real code generation)

- **PTX emitter**: generates valid NVIDIA PTX text from Codegen IR
  (register declarations, vector loads/stores, FMA, barriers, thread
  index intrinsics, predicated execution). The NVIDIA driver compiles
  PTX → SASS.
- **x86-64 emitter**: produces real machine code bytes (REX prefixes,
  ModR/M, SIB, 3-byte VEX prefixes for AVX/AVX2). Supports MOV, ADD,
  SUB, IMUL, XOR, PUSH/POP, RET, VMOVAPS, VADDPS, VMULPS, VFMADD231PS,
  MFENCE. The output is a `std::vector<u8>` that can be cast to a
  function pointer and called.
- **NvidiaBackend**: compiles a CGModule to an Executable containing PTX text.
- **CpuBackend**: compiles a CGModule to an Executable containing raw x86-64
  machine code bytes + a hex disassembly.
- **Lowering pass**: Codegen IR → PTX text or x86-64 bytes.

### Optimization passes (14)

Canonicalize, CSE, ConstantFolding, DCE, AlgebraicSimplification, SCCP,
EGraphSuperoptimizer, Fusion (with profitability model),
ShapeOptimization, LayoutOptimization, ReductionOptimization,
CopyElimination, MemoryPlanning (with real alloc/free insertion),
Specialization.

### Global Tensor Analysis (GTA)

DataflowAnalysis, ShapeAnalysis, LayoutAnalysis, LifetimeAnalysis,
ArithmeticIntensityAnalysis (memory/compute/launch/latency bound
classification), ParallelismAnalysis, ReuseAnalysis (materialize vs
recompute), GlobalAliasAnalysis (view-of/slice-of/broadcast-of),
GlobalCostAnalysis (decomposed: execution + memory + launch + sync +
specialization + code-size), GlobalAnalysisManager facade.

### Global Barrier + Iterative Driver

Global Barrier (legality check + schedule validation + final decisions)
+ IterativeDriver (7-phase pipeline with analysis feedback, bounded
iteration, converges to fixpoint, then crosses the barrier).

### Autotuner

Bayesian optimization over ScheduleSpace:
- Gaussian Process with squared-exponential (RBF) kernel
- Expected Improvement acquisition function
- Schedule feature extraction (tile sizes, vector width, unroll factor,
  parallelism, shared memory, tensor core usage)
- Random initial exploration + GP-guided exploitation
- Converges to near-optimal schedule with bounded benchmark count

### Schedule IR, Cost model, Codegen IR, Backend interface, Runtime

(Same as before — Schedule IR with transformations, HardwareModel +
analytical CostEstimator + HardwareProfile, Codegen IR with vector
ops/barriers/async copies, MachineBackend/TargetInfo/MachineEmitter,
Runtime with Device/Stream/Allocator/KernelCache.)

### E-graph core + superoptimizer

ENode/EClass/EGraph with merge/saturate, tensor-aware extraction, and a
superoptimizer pass that extracts pure sub-DAGs, saturates with tensor
rewrite rules (commutativity, identity, associativity), and extracts the
cheapest form.

### Standard ops

`add`, `sub`, `mul`, `div`, `neg`, `matmul`, `relu`, `gelu`, `sigmoid`,
`tanh`, `exp`, `log`, `sqrt`, `broadcast`, `reshape`, `transpose`,
`reduce_sum/max/mean`, `cast`, `copy`, `gather`, `scatter`, `concat`,
`slice`, `softmax`, `layernorm`, `batchnorm`, `conv2d`, `constant`,
`input`, `output`, `return`, `alloc`, `free`.

97 unit tests across 14 test executables; all passing.

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
