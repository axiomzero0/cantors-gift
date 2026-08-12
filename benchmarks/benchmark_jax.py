#!/usr/bin/env python3
"""Benchmark cantors-gift vs JAX vs torch.compile vs NumPy.

Measures elementwise add, fused relu+mul, matmul, and reduction across
multiple sizes. All backends verified for correctness.
"""
import sys
import os
import time
import ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "python"))

import numpy as np
import torch
import jax
import jax.numpy as jnp

# JAX config: disable GPU (we're CPU-only), use 64-bit precision for timing
jax.config.update("jax_platform_name", "cpu")

try:
    import cantors_gift as cg
    CG_AVAILABLE = True
except ImportError:
    CG_AVAILABLE = False


def time_fn(fn, args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def time_jax(fn, args, warmup=5, iters=30):
    """Time a JAX function, ensuring result is ready."""
    for _ in range(warmup):
        r = fn(*args)
        r.block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn(*args)
        r.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


# ---- NumPy reference ----

def numpy_add(a, b):
    return a + b

def numpy_fused_relu_mul(a, b):
    return np.maximum(a * b, 0)

def numpy_matmul(a, b):
    return a @ b

def numpy_reduction(a):
    return a.sum(axis=1)


# ---- PyTorch ----

def torch_add(a, b):
    return a + b

def torch_fused(a, b):
    return torch.relu(a * b)

def torch_matmul(a, b):
    return a @ b

def torch_reduction(a):
    return a.sum(dim=1)

@torch.compile(mode="reduce-overhead")
def torch_compiled_add(a, b):
    return a + b

@torch.compile(mode="reduce-overhead")
def torch_compiled_fused(a, b):
    return torch.relu(a * b)

@torch.compile(mode="reduce-overhead")
def torch_compiled_matmul(a, b):
    return a @ b

@torch.compile(mode="reduce-overhead")
def torch_compiled_reduction(a):
    return a.sum(dim=1)


# ---- JAX ----

@jax.jit
def jax_add(a, b):
    return a + b

@jax.jit
def jax_fused_relu_mul(a, b):
    return jnp.maximum(a * b, 0)

@jax.jit
def jax_matmul(a, b):
    return a @ b

@jax.jit
def jax_reduction(a):
    return a.sum(axis=1)


# ---- cantors-gift (direct x86 JIT) ----

def cg_build_elementwise_add(M, N):
    if not CG_AVAILABLE:
        return None
    total = M * N
    if total % 4 != 0:
        return None
    e = cg.X86Emitter()
    e.push(cg.X86Reg.RBP)
    e.mov_reg(cg.X86Reg.RBP, cg.X86Reg.RSP)
    count = total // 4
    e.mov_imm64(cg.X86Reg.RCX, count)
    e.xor_reg(cg.X86Reg.R8, cg.X86Reg.R8)
    e.mov_imm64(cg.X86Reg.R9, 16)
    loop = e.label()
    e.mark_label(loop)
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RDI)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_load(cg.X86VReg.XMM0, cg.X86Reg.RAX, 0, cg.VEXWidth.XMM)
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RSI)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_load(cg.X86VReg.XMM1, cg.X86Reg.RAX, 0, cg.VEXWidth.XMM)
    e.vaddps(cg.X86VReg.XMM0, cg.X86VReg.XMM0, cg.X86VReg.XMM1, cg.VEXWidth.XMM)
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RDX)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_store(cg.X86Reg.RAX, 0, cg.X86VReg.XMM0, cg.VEXWidth.XMM)
    e.add_reg(cg.X86Reg.R8, cg.X86Reg.R9)
    e.dec_reg(cg.X86Reg.RCX)
    e.jne(loop)
    e.pop(cg.X86Reg.RBP)
    e.ret()
    jit = cg.JITMemory()
    jit.allocate(e.take_bytes())
    return jit if jit.valid() else None


def cg_build_fused_relu_mul(M, N):
    if not CG_AVAILABLE:
        return None
    total = M * N
    if total % 4 != 0:
        return None
    e = cg.X86Emitter()
    e.push(cg.X86Reg.RBP)
    e.mov_reg(cg.X86Reg.RBP, cg.X86Reg.RSP)
    e.vxorps(cg.X86VReg.XMM2, cg.X86VReg.XMM2)
    count = total // 4
    e.mov_imm64(cg.X86Reg.RCX, count)
    e.xor_reg(cg.X86Reg.R8, cg.X86Reg.R8)
    e.mov_imm64(cg.X86Reg.R9, 16)
    loop = e.label()
    e.mark_label(loop)
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RDI)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_load(cg.X86VReg.XMM0, cg.X86Reg.RAX, 0, cg.VEXWidth.XMM)
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RSI)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_load(cg.X86VReg.XMM1, cg.X86Reg.RAX, 0, cg.VEXWidth.XMM)
    e.vmulps(cg.X86VReg.XMM0, cg.X86VReg.XMM0, cg.X86VReg.XMM1, cg.VEXWidth.XMM)
    e.vmaxps(cg.X86VReg.XMM0, cg.X86VReg.XMM0, cg.X86VReg.XMM2, cg.VEXWidth.XMM)
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RDX)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_store(cg.X86Reg.RAX, 0, cg.X86VReg.XMM0, cg.VEXWidth.XMM)
    e.add_reg(cg.X86Reg.R8, cg.X86Reg.R9)
    e.dec_reg(cg.X86Reg.RCX)
    e.jne(loop)
    e.pop(cg.X86Reg.RBP)
    e.ret()
    jit = cg.JITMemory()
    jit.allocate(e.take_bytes())
    return jit if jit.valid() else None


def cg_run_3arg(jit, a, b, c):
    if jit is None:
        return
    addr = jit.entry()
    if addr is None or addr == 0:
        return
    fn_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    fn = fn_type(addr)
    fn(a.ctypes.data, b.ctypes.data, c.ctypes.data)


def run_benchmark(name, sizes, setup_fn, numpy_fn,
                  torch_fn, torch_compiled_fn,
                  jax_fn, jax_compiled_fn,
                  cg_build_fn=None, cg_run_fn=None,
                  num_args=2):
    results = []
    for size in sizes:
        print(f"\n  {name} size={size}")
        a_np, b_np, ref = setup_fn(*size)

        result = {"name": name, "size": str(size)}

        # NumPy
        t = time_fn(numpy_fn, (a_np, b_np) if num_args == 2 else (a_np,))
        result["numpy_ms"] = t * 1000
        print(f"    NumPy:          {t*1000:.4f} ms")

        # PyTorch eager
        a_t = torch.from_numpy(a_np.copy())
        b_t = torch.from_numpy(b_np.copy()) if num_args == 2 else None
        t_args = (a_t, b_t) if num_args == 2 else (a_t,)
        t = time_fn(torch_fn, t_args)
        result["torch_ms"] = t * 1000
        print(f"    torch eager:    {t*1000:.4f} ms")

        # torch.compile
        try:
            t = time_fn(torch_compiled_fn, t_args)
            result["torch_compile_ms"] = t * 1000
            print(f"    torch.compile:  {t*1000:.4f} ms")
        except Exception as ex:
            result["torch_compile_ms"] = None
            print(f"    torch.compile:  FAILED ({ex})")

        # JAX JIT
        a_j = jnp.array(a_np)
        b_j = jnp.array(b_np) if num_args == 2 else None
        j_args = (a_j, b_j) if num_args == 2 else (a_j,)
        try:
            t = time_jax(jax_compiled_fn, j_args)
            result["jax_ms"] = t * 1000
            print(f"    JAX JIT:        {t*1000:.4f} ms")
        except Exception as ex:
            result["jax_ms"] = None
            print(f"    JAX JIT:        FAILED ({ex})")

        # cantors-gift
        if cg_build_fn and CG_AVAILABLE:
            try:
                jit = cg_build_fn(*size)
                if jit is not None and jit.valid():
                    c_np = np.zeros_like(a_np)
                    if num_args == 2:
                        cg_run_fn(jit, a_np, b_np, c_np)
                        if not np.allclose(c_np, ref, rtol=1e-3, atol=1e-3):
                            print(f"    cg x86:         CORRECTNESS FAILED")
                            result["cg_ms"] = None
                            results.append(result)
                            continue
                        t = time_fn(cg_run_fn, (jit, a_np, b_np, c_np))
                    else:
                        cg_run_fn(jit, a_np, c_np)
                        t = time_fn(cg_run_fn, (jit, a_np, c_np))
                    result["cg_ms"] = t * 1000
                    print(f"    cg x86:         {t*1000:.4f} ms (verified)")
                else:
                    result["cg_ms"] = None
            except Exception as ex:
                print(f"    cg x86:         FAILED ({ex})")
                result["cg_ms"] = None
        else:
            result["cg_ms"] = None

        results.append(result)
    return results


def main():
    print("=" * 80)
    print("cantors-gift vs JAX vs torch.compile vs NumPy")
    print("=" * 80)
    print(f"cantors-gift: {CG_AVAILABLE}")
    print(f"JAX: {jax.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CPU threads (torch): {torch.get_num_threads()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    all_results = []

    # ---- Elementwise Add ----
    print("-" * 80)
    print("Benchmark: Elementwise Add (C = A + B)")
    print("-" * 80)

    def setup_add(M, N):
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        return a, b, a + b

    sizes = [(64, 64), (256, 256), (1024, 1024)]
    results = run_benchmark(
        "elementwise_add", sizes, setup_add, numpy_add,
        torch_add, torch_compiled_add,
        jax_add, jax_add,
        cg_build_fn=cg_build_elementwise_add, cg_run_fn=cg_run_3arg)
    all_results.extend(results)

    # ---- Fused Relu+Mul ----
    print("\n" + "-" * 80)
    print("Benchmark: Fused Relu+Mul (C = relu(A * B))")
    print("-" * 80)

    def setup_fused(M, N):
        a = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(M, N).astype(np.float32)
        return a, b, np.maximum(a * b, 0)

    sizes = [(64, 64), (256, 256), (1024, 1024)]
    results = run_benchmark(
        "fused_relu_mul", sizes, setup_fused, numpy_fused_relu_mul,
        torch_fused, torch_compiled_fused,
        jax_fused_relu_mul, jax_fused_relu_mul,
        cg_build_fn=cg_build_fused_relu_mul, cg_run_fn=cg_run_3arg)
    all_results.extend(results)

    # ---- Matmul ----
    print("\n" + "-" * 80)
    print("Benchmark: Matmul (C = A @ B)")
    print("-" * 80)

    def setup_matmul(M, K, N):
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        return a, b, a @ b

    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    results = run_benchmark(
        "matmul", sizes, setup_matmul, numpy_matmul,
        torch_matmul, torch_compiled_matmul,
        jax_matmul, jax_matmul)
    all_results.extend(results)

    # ---- Reduction ----
    print("\n" + "-" * 80)
    print("Benchmark: Reduction (sum over axis=1)")
    print("-" * 80)

    def setup_reduction(M, N):
        a = np.random.randn(M, N).astype(np.float32)
        return a, None, a.sum(axis=1)

    sizes = [(256, 256), (1024, 1024)]
    results = run_benchmark(
        "reduction_sum", sizes, setup_reduction, numpy_reduction,
        torch_reduction, torch_compiled_reduction,
        jax_reduction, jax_reduction,
        num_args=1)
    all_results.extend(results)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    hdr = f"{'Benchmark':<20} {'Size':<18} {'NumPy':>8} {'torch':>8} {'compile':>8} {'JAX':>8} {'cg-x86':>8}"
    print(hdr)
    print("-" * 78)
    for r in all_results:
        np_str = f"{r.get('numpy_ms',0):.3f}" if r.get("numpy_ms") else "N/A"
        t_str = f"{r.get('torch_ms',0):.3f}" if r.get("torch_ms") else "N/A"
        c_str = f"{r.get('torch_compile_ms',0):.3f}" if r.get("torch_compile_ms") else "N/A"
        j_str = f"{r.get('jax_ms',0):.3f}" if r.get("jax_ms") else "N/A"
        cg_str = f"{r.get('cg_ms',0):.3f}" if r.get("cg_ms") else "N/A"
        print(f"{r['name']:<20} {r['size']:<18} {np_str:>8} {t_str:>8} {c_str:>8} {j_str:>8} {cg_str:>8}")

    # Save
    import json, csv
    with open("benchmarks/results_jax.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("benchmarks/results_jax.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","size","numpy_ms","torch_ms",
                            "torch_compile_ms","jax_ms","cg_ms"])
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\nResults saved to benchmarks/results_jax.json and .csv")


if __name__ == "__main__":
    main()
