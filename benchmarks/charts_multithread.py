#!/usr/bin/env python3
"""Generate multi-threaded benchmark charts."""
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    with open(path) as f: return json.load(f)

def plot(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    names = ["elementwise_add", "fused_relu_mul"]
    titles = ["Elementwise Add (C = A + B)", "Fused Relu+Mul (C = relu(A * B))"]

    for idx, (name, title) in enumerate(zip(names, titles)):
        ax = axes[idx]
        bench = [r for r in results if r["name"] == name]
        if not bench: continue
        sizes = [r["size"] for r in bench]
        x = np.arange(len(sizes))
        w = 0.12
        cols = [("numpy_ms", "NumPy", "#2196F3"),
                ("torch_ms", "torch eager", "#4CAF50"),
                ("torch_compile_ms", "torch.compile", "#FF9800"),
                ("jax_ms", "JAX JIT", "#9C27B0"),
                ("cg_single_ms", "cg single-thread", "#F44336"),
                ("cg_multi_ms", "cg multi-thread", "#795548")]
        for i, (key, label, color) in enumerate(cols):
            vals = [r.get(key) or 0 for r in bench]
            ax.bar(x + (i - 2.5) * w, vals, w, label=label, color=color)
        ax.set_xlabel("Matrix Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=15)
        ax.legend(fontsize=7)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "multithread_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

def plot_speedup(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, name in enumerate(["elementwise_add", "fused_relu_mul"]):
        ax = axes[idx]
        bench = [r for r in results if r["name"] == name]
        sizes = []
        su_tc = []
        su_jax = []
        for r in bench:
            sizes.append(r["size"])
            cgN = r.get("cg_multi_ms")
            tc = r.get("torch_compile_ms")
            jx = r.get("jax_ms")
            su_tc.append(tc/cgN if cgN and tc else 0)
            su_jax.append(jx/cgN if cgN and jx else 0)
        x = np.arange(len(sizes))
        w = 0.35
        ax.bar(x - w/2, su_tc, w, label='vs torch.compile', color='#FF9800')
        ax.bar(x + w/2, su_jax, w, label='vs JAX', color='#9C27B0')
        ax.axhline(y=1.0, color='black', linestyle='--', label='cg multi-thread baseline')
        ax.set_xlabel('Size')
        ax.set_ylabel('Speedup (x)')
        ax.set_title(f'{name} — cg multi-thread speedup')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "multithread_speedup.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_multithread.json"
    out_dir = os.path.dirname(path) or "."
    results = load(path)
    plot(results, out_dir)
    plot_speedup(results, out_dir)
    print("Done!")
