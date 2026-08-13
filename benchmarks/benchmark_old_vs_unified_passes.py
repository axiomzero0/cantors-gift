#!/usr/bin/env python3
"""
A/B benchmark: old optimization passes vs unified-driven passes.

Builds identical IR modules, runs each pipeline, and compares:
  - ops eliminated
  - transform counts
  - final op count (lower = better)
  - latency

The unified passes should produce results AT LEAST AS GOOD as the old
passes, with the bonus of per-pass stats + provenance + confidence.

Run with:
    python3.13 benchmark_old_vs_unified_passes.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent.parent / "build" / "python"
sys.path.insert(0, str(BUILD_DIR))

import cantors_gift as cg  # noqa: E402


def build_mul_zero_chain(depth: int) -> "cg.Module":
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


def build_transpose_chain(depth: int) -> "cg.Module":
    m = cg.Module()
    s = cg.Shape.from_constants([16, 16])
    f = m.create_function(
        f"transpose_{depth}",
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


def run_old_pipeline(m: "cg.Module") -> dict:
    """Run old passes: Canonicalize + CSE + ConstFold + DCE."""
    am = cg.AnalysisManager(m)
    pm = cg.PassManager()
    pm.add(cg.CanonicalizePass())
    pm.add(cg.CSEPass())
    pm.add(cg.ConstantFoldingPass())
    pm.add(cg.DCEPass())
    t0 = time.perf_counter()
    pm.run(m, am)
    t1 = time.perf_counter()
    return {"latency_us": (t1 - t0) * 1e6}


def run_unified_pipeline(m: "cg.Module") -> dict:
    """Run unified passes: UnifiedCanonicalize + UnifiedCSE + UnifiedConstFold + UnifiedDCE."""
    am = cg.AnalysisManager(m)
    pipe = cg.UnifiedPassPipeline()
    t0 = time.perf_counter()
    pipe.run(m, am)
    t1 = time.perf_counter()
    s = pipe.stats()
    return {
        "latency_us": (t1 - t0) * 1e6,
        "cse_duplicates_removed": s.cse.duplicates_removed,
        "dce_ops_removed": s.dce.ops_removed,
        "const_fold_constants_folded": s.const_fold.constants_folded,
        "canonicalize_total_rewrites": s.canonicalize.total_rewrites,
        "canonicalize_mul_zero": s.canonicalize.mul_zero_simplified,
        "canonicalize_transpose_pair": s.canonicalize.transpose_pair_eliminated,
        "copy_elim_total_rewrites": s.copy_elim.total_rewrites,
        "recompute_materialize": s.recompute.materialize_decisions,
        "recompute_recompute": s.recompute.recompute_decisions,
        "recompute_fuse": s.recompute.fuse_decisions,
    }


def count_ops(m: "cg.Module", opcode: int) -> int:
    n = 0
    for f in m.function_ptrs():
        for op in f.entry().ops():
            if op.opcode == opcode:
                n += 1
    return n


def total_ops(m: "cg.Module") -> int:
    return m.num_operations()


def main() -> int:
    print("VORTEX: old passes vs unified-driven passes (A/B benchmark)")
    print("=" * 92)

    results = []
    t_start = time.time()

    workloads = [
        ("mul-zero chain (depth 5)",   lambda: build_mul_zero_chain(5)),
        ("mul-zero chain (depth 10)",  lambda: build_mul_zero_chain(10)),
        ("transpose chain (depth 5)",  lambda: build_transpose_chain(5)),
        ("transpose chain (depth 10)", lambda: build_transpose_chain(10)),
        ("matmul+bias+relu (128x256x512)", lambda: build_matmul_bias_relu(128, 256, 512)),
        ("matmul+bias+relu (1024^3)",      lambda: build_matmul_bias_relu(1024, 1024, 1024)),
    ]

    for label, builder in workloads:
        print(f"\n--- {label} ---")

        # OLD pipeline.
        m_old = builder()
        old_ops_before = total_ops(m_old)
        old_mul_before = count_ops(m_old, 6)  # OP_MUL = 6
        old_transpose_before = count_ops(m_old, 19)  # OP_TRANSPOSE = 19
        old_result = run_old_pipeline(m_old)
        old_ops_after = total_ops(m_old)
        old_mul_after = count_ops(m_old, 6)
        old_transpose_after = count_ops(m_old, 19)

        # UNIFIED pipeline.
        m_new = builder()
        new_ops_before = total_ops(m_new)
        new_mul_before = count_ops(m_new, 6)
        new_transpose_before = count_ops(m_new, 19)
        new_result = run_unified_pipeline(m_new)
        new_ops_after = total_ops(m_new)
        new_mul_after = count_ops(m_new, 6)
        new_transpose_after = count_ops(m_new, 19)

        print(f"  OLD:   {old_ops_before} -> {old_ops_after} ops  ({old_result['latency_us']:.1f} us)")
        print(f"  NEW:   {new_ops_before} -> {new_ops_after} ops  ({new_result['latency_us']:.1f} us)")
        print(f"  NEW stats: cse={new_result['cse_duplicates_removed']}, "
              f"dce={new_result['dce_ops_removed']}, "
              f"const_fold={new_result['const_fold_constants_folded']}, "
              f"canon_rewrites={new_result['canonicalize_total_rewrites']}, "
              f"recompute_fuse={new_result['recompute_fuse']}")

        results.append({
            "workload": label,
            "old": {"ops_before": old_ops_before, "ops_after": old_ops_after,
                    "latency_us": old_result["latency_us"]},
            "new": {"ops_before": new_ops_before, "ops_after": new_ops_after,
                    "latency_us": new_result["latency_us"],
                    **new_result},
        })

    t_total = time.time() - t_start

    # Summary table.
    print(f"\n{'='*100}")
    print("  SUMMARY: old vs unified passes")
    print(f"{'='*100}")
    hdr = (
        f"{'Workload':<38} "
        f"{'Old before':>11} "
        f"{'Old after':>10} "
        f"{'New before':>11} "
        f"{'New after':>10} "
        f"{'Old us':>8} "
        f"{'New us':>8}"
    )
    print(hdr)
    print("-" * 100)
    for r in results:
        print(
            f"{r['workload'][:37]:<38} "
            f"{r['old']['ops_before']:>11d} "
            f"{r['old']['ops_after']:>10d} "
            f"{r['new']['ops_before']:>11d} "
            f"{r['new']['ops_after']:>10d} "
            f"{r['old']['latency_us']:>8.1f} "
            f"{r['new']['latency_us']:>8.1f}"
        )
    print("-" * 100)
    print(f"\nTotal benchmark wall time: {t_total:.3f} sec")

    # Save JSON.
    out_path = Path(__file__).resolve().parent / "old_vs_unified_passes_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
