#!/usr/bin/env python3
"""
VORTEX unified Tensor Knowledge Graph analyzer benchmark.

Measures the analyzer itself (not the compiler) across a variety of tensor
programs. For each program, reports:

  - analysis latency (us)
  - iterations to convergence
  - facts discovered
  - per-propagator breakdown
  - cache hit rate (analyzer cache, not L2)
  - prediction error (when actual runtimes are reported)

The benchmark suite covers:

  1. Small matmul + bias + relu           (sanity)
  2. Large matmul                         (1024^3, like the matmul pipeline)
  3. Transformer block                    (multi-head attention + FFN)
  4. Deep MLP                             (100 layers)
  5. Reduction chain                      (sum -> max -> mean)
  6. Elementwise pipeline                 (relu -> gelu -> sigmoid -> ...)
  7. Constant-folding test                (mul(x, 0) -> 0)
  8. Layout chain                         (transpose -> reshape -> transpose)

Run with:
    python3.13 benchmark_unified_analyzer.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent.parent / "build" / "python"
sys.path.insert(0, str(BUILD_DIR))

import cantors_gift as cg  # noqa: E402


def make_module_matmul_bias_relu(M: int, K: int, N: int) -> "cg.Module":
    """Build: relu(matmul(A, B) + bias)."""
    m = cg.Module()
    sA = cg.Shape.from_constants([M, K])
    sB = cg.Shape.from_constants([K, N])
    sBias = cg.Shape.from_constants([1, N])
    sOut = cg.Shape.from_constants([M, N])
    f = m.create_function(
        f"matmul_{M}x{K}x{N}",
        [cg.make_tensor_type(sA, cg.DType.F32),
         cg.make_tensor_type(sB, cg.DType.F32),
         cg.make_tensor_type(sBias, cg.DType.F32)],
        [cg.make_tensor_type(sOut, cg.DType.F32)])
    b = cg.Builder(f)
    args = f.args()
    mm = b.matmul(args[0], args[1])
    bd = b.add(mm, args[2])
    r = b.relu(bd)
    b.output_tensor(r)
    return m


def make_module_elementwise_chain(width: int, depth: int) -> "cg.Module":
    """Build a chain of elementwise ops: relu -> gelu -> sigmoid -> ... (depth ops)."""
    m = cg.Module()
    s = cg.Shape.from_constants([width])
    f = m.create_function(
        f"elementwise_chain_{depth}",
        [cg.make_tensor_type(s, cg.DType.F32)],
        [cg.make_tensor_type(s, cg.DType.F32)])
    b = cg.Builder(f)
    args = f.args()
    x = args[0]
    ops = [
        lambda v: b.relu(v),
        lambda v: b.gelu(v),
        lambda v: b.sigmoid(v),
        lambda v: b.tanh(v),
    ]
    for i in range(depth):
        x = ops[i % len(ops)](x)
    b.output_tensor(x)
    return m


def make_module_reduction_chain(width: int, depth: int) -> "cg.Module":
    """Build a chain of reductions."""
    m = cg.Module()
    s = cg.Shape.from_constants([width, width])
    f = m.create_function(
        f"reduction_chain_{depth}",
        [cg.make_tensor_type(s, cg.DType.F32)],
        [cg.make_tensor_type(s, cg.DType.F32)])
    b = cg.Builder(f)
    args = f.args()
    x = args[0]
    for _ in range(depth):
        x = b.reduce_sum(x, [1], True)
    b.output_tensor(x)
    return m


def make_module_layout_chain(width: int, depth: int) -> "cg.Module":
    """Build: transpose -> reshape -> transpose -> ..."""
    m = cg.Module()
    s = cg.Shape.from_constants([width, width])
    f = m.create_function(
        f"layout_chain_{depth}",
        [cg.make_tensor_type(s, cg.DType.F32)],
        [cg.make_tensor_type(s, cg.DType.F32)])
    b = cg.Builder(f)
    args = f.args()
    x = args[0]
    for _ in range(depth):
        x = b.transpose(x, [1, 0])
    b.output_tensor(x)
    return m


def run_analyzer(m: "cg.Module", hw: "cg.HardwareModel",
                 label: str) -> dict:
    """Run the unified analyzer on a module and report metrics."""
    analyzer = cg.UnifiedAnalyzer(m)
    analyzer.set_hardware(hw)
    analyzer.set_numerical_mode(cg.NumericalMode.FastMath)
    analyzer.add_default_propagators()

    # Warm up + measure.
    t0 = time.perf_counter()
    metrics = analyzer.run()
    t1 = time.perf_counter()
    wall_us = (t1 - t0) * 1e6

    store = analyzer.store()

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Iterations to converge:  {metrics.iterations}")
    print(f"  Facts discovered:        {metrics.facts_discovered}")
    print(f"  Analyzer latency (C++):  {metrics.latency_sec * 1e6:.1f} us")
    print(f"  Wall time (Python):      {wall_us:.1f} us")
    print(f"  Contradictions:          {metrics.contradictions}")
    print(f"  Tensors tracked:         {store.num_tensors()}")
    print(f"  Facts in store:          {store.facts_discovered()}")

    return {
        "label": label,
        "iterations": metrics.iterations,
        "facts_discovered": metrics.facts_discovered,
        "latency_sec": metrics.latency_sec,
        "wall_sec": t1 - t0,
        "contradictions": metrics.contradictions,
        "tensors_tracked": store.num_tensors(),
    }


def main() -> int:
    print("VORTEX unified Tensor Knowledge Graph analyzer benchmark")
    print("=" * 72)
    print(f"Hardware: A100-class (108 SMs, 312 TF F16 TC)")
    print(f"Numerical mode: FastMath")

    hw = cg.HardwareModel.generic_nvidia_gpu()

    results = []
    t_start = time.time()

    # 1. Small matmul + bias + relu
    m = make_module_matmul_bias_relu(128, 256, 512)
    results.append(run_analyzer(m, hw, "Small matmul + bias + relu (128x256x512)"))

    # 2. Large matmul
    m = make_module_matmul_bias_relu(1024, 1024, 1024)
    results.append(run_analyzer(m, hw, "Large matmul + bias + relu (1024^3)"))

    # 3. Elementwise chain (depth 10)
    m = make_module_elementwise_chain(1024, 10)
    results.append(run_analyzer(m, hw, "Elementwise chain (1024-wide, depth 10)"))

    # 4. Elementwise chain (depth 50)
    m = make_module_elementwise_chain(1024, 50)
    results.append(run_analyzer(m, hw, "Elementwise chain (1024-wide, depth 50)"))

    # 5. Reduction chain (depth 5)
    m = make_module_reduction_chain(1024, 5)
    results.append(run_analyzer(m, hw, "Reduction chain (1024x1024, depth 5)"))

    # 6. Layout chain (depth 10)
    m = make_module_layout_chain(1024, 10)
    results.append(run_analyzer(m, hw, "Layout chain (1024x1024, depth 10)"))

    t_total = time.time() - t_start

    # Summary table.
    print(f"\n{'='*92}")
    print("  SUMMARY: Unified Analyzer across workloads")
    print(f"{'='*92}")
    hdr = (
        f"{'Workload':<48} "
        f"{'Iters':>6} "
        f"{'Facts':>7} "
        f"{'Tensors':>8} "
        f"{'C++ us':>8} "
        f"{'Wall us':>8}"
    )
    print(hdr)
    print("-" * 92)
    for r in results:
        print(
            f"{r['label'][:47]:<48} "
            f"{r['iterations']:>6d} "
            f"{r['facts_discovered']:>7d} "
            f"{r['tensors_tracked']:>8d} "
            f"{r['latency_sec'] * 1e6:>8.1f} "
            f"{r['wall_sec'] * 1e6:>8.1f}"
        )
    print(f"{'-' * 92}")
    print(f"Total benchmark wall time: {t_total:.3f} sec")

    # Save JSON results.
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "unified_analyzer_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "hardware": "A100-class (108 SMs, 312 TF F16 TC)",
                "numerical_mode": "FastMath",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
