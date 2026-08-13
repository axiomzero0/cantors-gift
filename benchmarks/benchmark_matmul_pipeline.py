#!/usr/bin/env python3
"""
VORTEX matmul pipeline benchmark.

Captures the full compile-time breakdown and runtime metrics for the
end-to-end matmul pipeline:

    IR construction -> GTA -> e-graph saturation -> fusion ->
    schedule-space generation -> v2 cost-model pruning ->
    Bayesian autotuning -> codegen

For each shape in the sweep, reports:

  Compile-time:
    - IR construction (us)
    - GTA (us)
    - e-graph saturation (us)
    - scheduling (space enumeration + v2 prune) (us)
    - autotuning (us)
    - codegen (us)
    - total compile (us)

  Roofline breakdown:
    - graph arithmetic intensity (FLOP/byte)
    - kernel arithmetic intensity (after fusion)
    - effective arithmetic intensity (after L2/shared/register reuse)
    - F32 ridge, F16 TC ridge

  SM utilization (the bug that was fixed):
    - num_blocks
    - num_sms
    - num_waves
    - sm_utilization_pct  (NOT the same as occupancy!)
    - tail_efficiency_pct
    - idle_sms_in_tail
    - wave_quant_sec

  Best schedule:
    - best_runtime_sec
    - analytical_estimate_sec
    - uses_tensor_core
    - autotune_benchmarks

Usage:
    python benchmark_matmul_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make sure the built Python module is importable.
BUILD_DIR = Path(__file__).resolve().parent.parent / "build" / "python"
sys.path.insert(0, str(BUILD_DIR))

import cantors_gift as cg  # noqa: E402


# ---------------------------------------------------------------------------
# Shape sweep (the user's requested set).
# ---------------------------------------------------------------------------
SHAPE_SWEEP = [
    # (M, K, N, label)
    (1024, 1024, 1024, "1024^3 (baseline)"),
    (1025, 1024, 1024, "odd M"),
    (1024, 1025, 1024, "odd K"),
    (1024, 1024, 1025, "odd N"),
    (1000, 1000, 1000, "1000^3"),
    (512, 4096, 512, "skinny K (512x4096x512)"),
    (4096, 512, 4096, "fat K (4096x512x4096)"),
    (127, 127, 127, "small (127^3)"),
]


def make_config(M: int, K: int, N: int) -> "cg.MatmulPipelineConfig":
    """Build a pipeline config for the given shape."""
    cfg = cg.MatmulPipelineConfig()
    cfg.M = M
    cfg.K = K
    cfg.N = N
    cfg.dtype = cg.DType.F32
    cfg.fuse_bias_relu = True
    cfg.m_tiles = [32, 64, 128]
    cfg.n_tiles = [32, 64, 128]
    cfg.k_tiles = [16, 32, 64]
    cfg.vector_widths = [4, 8, 16]
    cfg.max_benchmarks = 20
    cfg.initial_random = 5
    cfg.top_k_prune = 8
    cfg.mode = cg.NumericalMode.FastMath
    return cfg


def format_us(sec: float) -> str:
    if sec is None:
        return "n/a"
    return f"{sec * 1e6:8.2f}"


def format_pct(p: float) -> str:
    if p is None:
        return "n/a"
    return f"{p:6.2f}%"


def run_one(M: int, K: int, N: int, label: str, hw: "cg.HardwareModel") -> dict:
    """Run the pipeline for one shape and return a result dict."""
    cfg = make_config(M, K, N)
    pipeline = cg.MatmulPipeline(hw, cfg)
    result = pipeline.run_with_analytical_benchmark()

    timing = result.timing
    roof = result.roofline
    cb = result.cost_breakdown

    print(f"\n{'='*72}")
    print(f"  {label}  ({M} x {K} x {N})")
    print(f"{'='*72}")

    print("\n--- Compile-time breakdown (us) ---")
    print(f"  IR construction:   {format_us(timing.ir_construction_sec)}")
    print(f"  GTA:               {format_us(timing.gta_sec)}")
    print(f"  E-graph saturate:  {format_us(timing.egraph_saturation_sec)}")
    print(f"  Scheduling:        {format_us(timing.scheduling_sec)}")
    print(f"  Autotuning:        {format_us(timing.autotuning_sec)}")
    print(f"  Codegen:           {format_us(timing.codegen_sec)}")
    print(f"  Total compile:     {format_us(timing.total_sec)}")

    print("\n--- Roofline breakdown ---")
    print(f"  FLOPs:                  {roof.flops:>15_d}")
    print(f"  Graph bytes:            {roof.graph_bytes:>15_d}")
    print(f"  Kernel bytes:           {roof.kernel_bytes:>15_d}")
    print(f"  Effective bytes:        {roof.effective_bytes:>15_d}")
    print(f"  Graph intensity:        {roof.graph_intensity:>10.2f} FLOP/byte")
    print(f"  Kernel intensity:       {roof.kernel_intensity:>10.2f} FLOP/byte")
    print(f"  Effective intensity:    {roof.effective_intensity:>10.2f} FLOP/byte")
    print(f"  F32 ridge:              {roof.ridge_f32:>10.2f} FLOP/byte")
    print(f"  F16 TC ridge:           {roof.ridge_f16_tc:>10.2f} FLOP/byte")

    print("\n--- SM utilization (the bug that was fixed) ---")
    print(f"  num_blocks:             {cb.num_blocks:>10d}")
    print(f"  num_sms:                {cb.num_sms:>10d}")
    print(f"  num_waves:              {cb.num_waves:>10d}")
    print(f"  sm_utilization_pct:     {format_pct(cb.sm_utilization_pct)}")
    print(f"  tail_efficiency_pct:    {format_pct(cb.tail_efficiency_pct)}")
    print(f"  idle_sms_in_tail:       {cb.idle_sms_in_tail:>10d}")
    print(f"  wave_quant_sec:         {format_us(cb.wave_quant_sec)} us")
    print(f"  estimated_occupancy_pct:{format_pct(cb.estimated_occupancy_pct)}")

    print("\n--- Best schedule ---")
    print(f"  best_runtime:           {format_us(result.best_runtime_sec)} us")
    print(f"  analytical_estimate:    {format_us(result.analytical_estimate_sec)} us")
    print(f"  uses_tensor_core:       {result.uses_tensor_core}")
    print(f"  schedule_space_size:    {result.schedule_space_size}")
    print(f"  pruned_space_size:      {result.pruned_space_size}")
    print(f"  autotune_benchmarks:    {result.autotune_benchmarks}")

    return {
        "shape": {"M": M, "K": K, "N": N, "label": label},
        "timing": {
            "ir_construction_sec": timing.ir_construction_sec,
            "gta_sec": timing.gta_sec,
            "egraph_saturation_sec": timing.egraph_saturation_sec,
            "scheduling_sec": timing.scheduling_sec,
            "autotuning_sec": timing.autotuning_sec,
            "codegen_sec": timing.codegen_sec,
            "total_sec": timing.total_sec,
        },
        "roofline": {
            "flops": roof.flops,
            "graph_bytes": roof.graph_bytes,
            "kernel_bytes": roof.kernel_bytes,
            "effective_bytes": roof.effective_bytes,
            "graph_intensity": roof.graph_intensity,
            "kernel_intensity": roof.kernel_intensity,
            "effective_intensity": roof.effective_intensity,
            "ridge_f32": roof.ridge_f32,
            "ridge_f16_tc": roof.ridge_f16_tc,
        },
        "sm_utilization": {
            "num_blocks": cb.num_blocks,
            "num_sms": cb.num_sms,
            "num_waves": cb.num_waves,
            "sm_utilization_pct": cb.sm_utilization_pct,
            "tail_efficiency_pct": cb.tail_efficiency_pct,
            "idle_sms_in_tail": cb.idle_sms_in_tail,
            "wave_quant_sec": cb.wave_quant_sec,
            "estimated_occupancy_pct": cb.estimated_occupancy_pct,
        },
        "best_schedule": {
            "best_runtime_sec": result.best_runtime_sec,
            "analytical_estimate_sec": result.analytical_estimate_sec,
            "uses_tensor_core": result.uses_tensor_core,
            "schedule_space_size": result.schedule_space_size,
            "pruned_space_size": result.pruned_space_size,
            "autotune_benchmarks": result.autotune_benchmarks,
        },
    }


def print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table comparing all shapes."""
    print(f"\n{'='*108}")
    print("  SUMMARY: VORTEX matmul pipeline across shape sweep")
    print(f"{'='*108}")
    hdr = (
        f"{'Shape':<28} "
        f"{'FLOPs':>14} "
        f"{'Graph AI':>10} "
        f"{'Kernel AI':>10} "
        f"{'Eff AI':>10} "
        f"{'Blocks':>8} "
        f"{'Waves':>6} "
        f"{'SM util':>8} "
        f"{'Best us':>9} "
        f"{'TC':>3}"
    )
    print(hdr)
    print("-" * 108)
    for r in results:
        s = r["shape"]
        ro = r["roofline"]
        sm = r["sm_utilization"]
        bs = r["best_schedule"]
        label = s["label"][:27]
        print(
            f"{label:<28} "
            f"{ro['flops']:>14_d} "
            f"{ro['graph_intensity']:>10.2f} "
            f"{ro['kernel_intensity']:>10.2f} "
            f"{ro['effective_intensity']:>10.2f} "
            f"{sm['num_blocks']:>8d} "
            f"{sm['num_waves']:>6d} "
            f"{sm['sm_utilization_pct']:>7.2f}% "
            f"{bs['best_runtime_sec'] * 1e6:>9.2f} "
            f"{'Y' if bs['uses_tensor_core'] else 'N':>3}"
        )
    print(f"{'='*108}")


def main() -> int:
    print("VORTEX matmul pipeline benchmark")
    print("=" * 72)
    print(f"Hardware: A100-class NVIDIA GPU (108 SMs, 312 TF F16 TC)")
    print(f"Numerical mode: FastMath (TC eligibility enabled)")

    hw = cg.HardwareModel.generic_nvidia_gpu()

    results = []
    t_start = time.time()
    for M, K, N, label in SHAPE_SWEEP:
        r = run_one(M, K, N, label, hw)
        results.append(r)
    t_total = time.time() - t_start

    print_summary_table(results)
    print(f"\nTotal benchmark wall time: {t_total:.2f} sec")

    # Save JSON results.
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "matmul_pipeline_results.json"
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
