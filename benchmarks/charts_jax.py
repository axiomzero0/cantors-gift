#!/usr/bin/env python3
"""Generate benchmark charts from results_jax.json."""
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    with open(path) as f: return json.load(f)

def plot(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    names = ["elementwise_add", "fused_relu_mul", "matmul", "reduction_sum"]
    titles = ["Elementwise Add (C = A + B)", "Fused Relu+Mul (C = relu(A * B))",
              "Matmul (C = A @ B)", "Reduction (sum over axis=1)"]

    for idx, (name, title) in enumerate(zip(names, titles)):
        ax = axes[idx // 2][idx % 2]
        bench = [r for r in results if r["name"] == name]
        if not bench: continue
        sizes = [r["size"] for r in bench]
        x = np.arange(len(sizes))
        w = 0.15
        cols = [("numpy_ms", "NumPy", "#2196F3"),
                ("torch_ms", "torch eager", "#4CAF50"),
                ("torch_compile_ms", "torch.compile", "#FF9800"),
                ("jax_ms", "JAX JIT", "#9C27B0"),
                ("cg_ms", "cantors-gift", "#F44336")]
        for i, (key, label, color) in enumerate(cols):
            vals = [r.get(key) or 0 for r in bench]
            ax.bar(x + (i - 2) * w, vals, w, label=label, color=color)
        ax.set_xlabel("Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=15)
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "jax_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

def plot_speedup(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, name in enumerate(["elementwise_add", "fused_relu_mul"]):
        ax = axes[idx]
        bench = [r for r in results if r["name"] == name]
        sizes = []
        jax_su = []
        tc_su = []
        for r in bench:
            sizes.append(r["size"])
            cg = r.get("cg_ms")
            jx = r.get("jax_ms")
            tc = r.get("torch_compile_ms")
            jax_su.append(jx/cg if cg and jx else 0)
            tc_su.append(tc/cg if cg and tc else 0)
        x = np.arange(len(sizes))
        w = 0.35
        ax.bar(x - w/2, jax_su, w, label='JAX speedup', color='#9C27B0')
        ax.bar(x + w/2, tc_su, w, label='torch.compile speedup', color='#FF9800')
        ax.axhline(y=1.0, color='black', linestyle='--', label='cg baseline')
        ax.set_xlabel('Size')
        ax.set_ylabel('Speedup vs cantors-gift (x)')
        ax.set_title(f'{name} — speedup of cg vs others')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "jax_speedup.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_jax.json"
    out_dir = os.path.dirname(path) or "."
    results = load(path)
    plot(results, out_dir)
    plot_speedup(results, out_dir)
    print("Done!")
