#!/usr/bin/env python3
"""Generate benchmark charts from results.json."""
import json
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_results(path):
    with open(path) as f:
        return json.load(f)

def plot_elementwise(results, output_dir):
    """Plot elementwise add and fused relu+mul benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, name in enumerate(["elementwise_add", "fused_relu_mul"]):
        ax = axes[idx]
        bench = [r for r in results if r["name"] == name]
        if not bench:
            continue

        sizes = []
        numpy_times = []
        torch_times = []
        compile_times = []
        cg_times = []

        for r in bench:
            size_str = r["size"]
            if name == "elementwise_add":
                M, N = eval(size_str)
                total = M * N
                sizes.append(f"{M}x{N}")
            else:
                M, N = eval(size_str)
                total = M * N
                sizes.append(f"{M}x{N}")

            numpy_times.append(r.get("numpy_ms") or 0)
            torch_times.append(r.get("torch_ms") or 0)
            compile_times.append(r.get("torch_compile_ms") or 0)
            cg_times.append(r.get("cantors_gift_x86_ms") or 0)

        x = np.arange(len(sizes))
        width = 0.2

        ax.bar(x - 1.5*width, numpy_times, width, label='NumPy', color='#2196F3')
        ax.bar(x - 0.5*width, torch_times, width, label='torch eager', color='#4CAF50')
        ax.bar(x + 0.5*width, compile_times, width, label='torch.compile', color='#FF9800')
        ax.bar(x + 1.5*width, cg_times, width, label='cantors-gift (x86 JIT)', color='#F44336')

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Time (ms)')
        title = "Elementwise Add (C = A + B)" if name == "elementwise_add" else "Fused Relu+Mul (C = relu(A * B))"
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "elementwise_benchmark.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

def plot_matmul(results, output_dir):
    """Plot matmul benchmark."""
    bench = [r for r in results if r["name"] == "matmul"]
    if not bench:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = []
    numpy_times = []
    torch_times = []
    compile_times = []

    for r in bench:
        M, K, N = eval(r["size"])
        sizes.append(f"{M}x{K}x{N}")
        numpy_times.append(r.get("numpy_ms") or 0)
        torch_times.append(r.get("torch_ms") or 0)
        compile_times.append(r.get("torch_compile_ms") or 0)

    x = np.arange(len(sizes))
    width = 0.25

    ax.bar(x - width, numpy_times, width, label='NumPy', color='#2196F3')
    ax.bar(x, torch_times, width, label='torch eager', color='#4CAF50')
    ax.bar(x + width, compile_times, width, label='torch.compile', color='#FF9800')

    ax.set_xlabel('Matmul Size (MxKxN)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Matmul (C = A @ B)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "matmul_benchmark.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

def plot_speedup(results, output_dir):
    """Plot speedup relative to torch.compile."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, name in enumerate(["elementwise_add", "fused_relu_mul"]):
        ax = axes[idx]
        bench = [r for r in results if r["name"] == name]
        if not bench:
            continue

        sizes = []
        speedups = []

        for r in bench:
            M, N = eval(r["size"])
            sizes.append(f"{M}x{N}")
            compile_ms = r.get("torch_compile_ms") or 1
            cg_ms = r.get("cantors_gift_x86_ms")
            if cg_ms and cg_ms > 0:
                speedups.append(compile_ms / cg_ms)
            else:
                speedups.append(0)

        colors = ['#4CAF50' if s >= 1 else '#F44336' for s in speedups]
        ax.bar(sizes, speedups, color=colors)
        ax.axhline(y=1.0, color='black', linestyle='--', label='torch.compile baseline')
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Speedup vs torch.compile (x)')
        title = "Elementwise Add Speedup" if name == "elementwise_add" else "Fused Relu+Mul Speedup"
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "speedup_vs_torch_compile.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results.json"
    output_dir = os.path.dirname(results_path) or "."

    results = load_results(results_path)
    print(f"Loaded {len(results)} benchmark results")

    plot_elementwise(results, output_dir)
    plot_matmul(results, output_dir)
    plot_speedup(results, output_dir)

    print("Done!")

if __name__ == "__main__":
    main()
