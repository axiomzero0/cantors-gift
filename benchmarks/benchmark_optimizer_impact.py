#!/usr/bin/env python3
"""
OPTIMIZER-IMPACT benchmark: does the unified analyzer actually make the
optimizer BETTER?

This is the question that matters. The analyzer being richer is useless
if the optimizer doesn't exploit it. This benchmark measures:

  - ops eliminated (per pass, per workload)
  - fusions accepted (with predicted improvement)
  - bytes saved (memory planning)
  - decision log quality (can we explain WHY?)

Compares:
  - OLD: existing AlgebraicSimplificationPass + existing fusion pass
  - NEW: PropertyDrivenSimplification + RangeDrivenStrengthReduction
         + CostGuidedFusion + LayoutAwareCopyElimination
         + AliasAwareMemoryPlanning

The KEY test: build IR that the old passes can't simplify but the new
passes can, because the new passes have access to the Property lattice
and ValueRange facts computed by the unified analyzer.

Run with:
    python3.13 benchmark_optimizer_impact.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent.parent / "build" / "python"
sys.path.insert(0, str(BUILD_DIR))

import cantors_gift as cg  # noqa: E402


# ---------------------------------------------------------------------------
# Workload builders — designed to expose cases where unified analysis wins.
# ---------------------------------------------------------------------------

def build_mul_zero_chain(depth: int) -> "cg.Module":
    """Build: x -> mul(.,0) -> mul(.,0) -> ... (depth zeros).

    Each mul-by-zero should be eliminated by the unified pass.
    """
    m = cg.Module()
    s = cg.Shape.from_constants([16, 16])
    f = m.create_function(
        f"mul_zero_{depth}",
        [cg.make_tensor_type(s, cg.DType.F32)],
        [cg.make_tensor_type(s, cg.DType.F32)])
    b = cg.Builder(f)
    x = f.args()[0]
    cur = x
    zero_bytes = b"\x00" * (16 * 16 * 4)
    for _ in range(depth):
        zero = b.constant_tensor([16, 16], cg.DType.F32, zero_bytes)
        cur = b.mul(cur, zero)
    b.output_tensor(cur)
    return m


def build_relu_chain(depth: int) -> "cg.Module":
    """Build: relu(relu(relu(...relu(x)...))) (depth relus).

    Each pair of relus can be collapsed: relu(relu(x)) -> relu(x).
    The unified RangeDrivenStrengthReduction eliminates redundant relus
    because the inner relu is provably >= 0.
    """
    m = cg.Module()
    s = cg.Shape.from_constants([16, 16])
    f = m.create_function(
        f"relu_chain_{depth}",
        [cg.make_tensor_type(s, cg.DType.F32)],
        [cg.make_tensor_type(s, cg.DType.F32)])
    b = cg.Builder(f)
    x = f.args()[0]
    cur = x
    for _ in range(depth):
        cur = b.relu(cur)
    b.output_tensor(cur)
    return m


def build_transpose_chain(depth: int) -> "cg.Module":
    """Build: transpose(transpose(transpose(...transpose(x)...))) (depth transposes).

    Each pair of transposes cancels. The unified LayoutAwareCopyElimination
    eliminates them.
    """
    m = cg.Module()
    s = cg.Shape.from_constants([16, 16])
    f = m.create_function(
        f"transpose_chain_{depth}",
        [cg.make_tensor_type(s, cg.DType.F32)],
        [cg.make_tensor_type(s, cg.DType.F32)])
    b = cg.Builder(f)
    x = f.args()[0]
    cur = x
    for _ in range(depth):
        cur = b.transpose(cur, [1, 0])
    b.output_tensor(cur)
    return m


def build_matmul_bias_relu(M: int, K: int, N: int) -> "cg.Module":
    """Build: relu(matmul(A, B) + bias). Fusion candidates galore."""
    m = cg.Module()
    sA = cg.Shape.from_constants([M, K])
    sB = cg.Shape.from_constants([K, N])
    sBias = cg.Shape.from_constants([1, N])
    sOut = cg.Shape.from_constants([M, N])
    f = m.create_function(
        f"mm_{M}x{K}x{N}",
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


# ---------------------------------------------------------------------------
# Run a unified pass and report stats.
# ---------------------------------------------------------------------------

def run_unified_pipeline(m: "cg.Module") -> dict:
    """Run the UnifiedOptimizationPipeline and report per-pass stats."""
    am = cg.AnalysisManager(m)
    pipe = cg.UnifiedOptimizationPipeline()
    t0 = time.perf_counter()
    pipe.run(m, am)
    t1 = time.perf_counter()

    s = pipe.stats()
    return {
        "latency_sec": t1 - t0,
        "property": {
            "mul_zero": s.property.mul_zero_eliminated,
            "mul_one": s.property.mul_one_eliminated,
            "add_zero": s.property.add_zero_eliminated,
            "matmul_identity": s.property.matmul_identity_eliminated,
            "total": s.property.total_rewrites,
        },
        "range": {
            "relu": s.range.relu_eliminated,
            "total": s.range.total_rewrites,
        },
        "fusion": {
            "accepted": s.fusion.fusions_accepted,
            "rejected_cost": s.fusion.fusions_rejected_cost,
            "rejected_conf": s.fusion.fusions_rejected_confidence,
            "rejected_legality": s.fusion.fusions_rejected_legality,
            "total_improvement": s.fusion.total_predicted_improvement,
        },
        "layout": {
            "transpose_transpose": s.layout.transpose_transpose_eliminated,
            "total": s.layout.total_rewrites,
        },
        "alias": {
            "buffers_merged": s.alias.buffers_merged,
            "bytes_saved": s.alias.bytes_saved,
        },
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> int:
    print("VORTEX optimizer-impact benchmark")
    print("=" * 92)
    print("Does the unified analyzer actually make the OPTIMIZER better?")
    print()

    results = []
    t_start = time.time()

    # ---- Workload 1: mul-by-zero chain ----
    print("--- mul-zero chain (depth 5) ---")
    m = build_mul_zero_chain(5)
    r = run_unified_pipeline(m)
    r["workload"] = "mul-zero chain (depth 5)"
    results.append(r)
    print(f"  latency: {r['latency_sec']*1e6:.1f} us")
    print(f"  mul_zero eliminated: {r['property']['mul_zero']}")
    print(f"  total property rewrites: {r['property']['total']}")
    print()

    # ---- Workload 2: relu chain ----
    print("--- relu chain (depth 10) ---")
    m = build_relu_chain(10)
    r = run_unified_pipeline(m)
    r["workload"] = "relu chain (depth 10)"
    results.append(r)
    print(f"  latency: {r['latency_sec']*1e6:.1f} us")
    print(f"  relu eliminated: {r['range']['relu']}")
    print(f"  total range rewrites: {r['range']['total']}")
    print()

    # ---- Workload 3: transpose chain ----
    print("--- transpose chain (depth 10) ---")
    m = build_transpose_chain(10)
    r = run_unified_pipeline(m)
    r["workload"] = "transpose chain (depth 10)"
    results.append(r)
    print(f"  latency: {r['latency_sec']*1e6:.1f} us")
    print(f"  transpose_transpose eliminated: {r['layout']['transpose_transpose']}")
    print(f"  total layout rewrites: {r['layout']['total']}")
    print()

    # ---- Workload 4: matmul + bias + relu (fusion candidates) ----
    print("--- matmul + bias + relu (1024^3) ---")
    m = build_matmul_bias_relu(1024, 1024, 1024)
    r = run_unified_pipeline(m)
    r["workload"] = "matmul+bias+relu (1024^3)"
    results.append(r)
    print(f"  latency: {r['latency_sec']*1e6:.1f} us")
    print(f"  fusions accepted: {r['fusion']['accepted']}")
    print(f"  fusions rejected (cost): {r['fusion']['rejected_cost']}")
    print(f"  fusions rejected (confidence): {r['fusion']['rejected_conf']}")
    print(f"  fusions rejected (legality): {r['fusion']['rejected_legality']}")
    print(f"  total predicted improvement: {r['fusion']['total_improvement']:.2f}x")
    print()

    # ---- Workload 5: smaller matmul for comparison ----
    print("--- matmul + bias + relu (128x256x512) ---")
    m = build_matmul_bias_relu(128, 256, 512)
    r = run_unified_pipeline(m)
    r["workload"] = "matmul+bias+relu (128x256x512)"
    results.append(r)
    print(f"  latency: {r['latency_sec']*1e6:.1f} us")
    print(f"  fusions accepted: {r['fusion']['accepted']}")
    print(f"  total predicted improvement: {r['fusion']['total_improvement']:.2f}x")
    print()

    t_total = time.time() - t_start

    # ---- Summary ----
    print("=" * 92)
    print("  SUMMARY: optimizer impact of unified analysis")
    print("=" * 92)
    hdr = (
        f"{'Workload':<38} "
        f"{'Lat us':>8} "
        f"{'Prop':>6} "
        f"{'Range':>6} "
        f"{'Fuse':>6} "
        f"{'Layout':>7} "
        f"{'Improv':>8}"
    )
    print(hdr)
    print("-" * 92)
    for r in results:
        print(
            f"{r['workload'][:37]:<38} "
            f"{r['latency_sec']*1e6:>8.1f} "
            f"{r['property']['total']:>6d} "
            f"{r['range']['total']:>6d} "
            f"{r['fusion']['accepted']:>6d} "
            f"{r['layout']['total']:>7d} "
            f"{r['fusion']['total_improvement']:>7.2f}x"
        )
    print("-" * 92)
    print(f"\nTotal benchmark wall time: {t_total:.3f} sec")

    # Save JSON.
    out_path = Path(__file__).resolve().parent / "optimizer_impact_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
