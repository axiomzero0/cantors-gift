#!/usr/bin/env python3
"""Benchmark cantors-gift (multi-threaded) vs JAX vs torch.compile vs NumPy.

Uses ParallelExecutor to split work across N threads.
The x86 kernel accepts a count parameter (4th argument) so it can be
called on chunks of the array.
"""
import sys, os, time, ctypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "python"))

import numpy as np
import torch
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

try:
    import cantors_gift as cg
    CG_AVAILABLE = True
except ImportError:
    CG_AVAILABLE = False


def time_fn(fn, args, warmup=5, iters=30):
    for _ in range(warmup): fn(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def time_jax(fn, args, warmup=5, iters=30):
    for _ in range(warmup):
        r = fn(*args); r.block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn(*args); r.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


# ---- cantors-gift: parametric kernel that accepts count ----

def cg_build_parametric_add():
    """Build a kernel: void fn(float* a, float* b, float* c, int64_t count)
    that processes `count` groups of 4 floats using VADDPS.
    count is passed in RCX (4th argument, System V ABI).
    """
    if not CG_AVAILABLE: return None
    e = cg.X86Emitter()
    e.push(cg.X86Reg.RBP)
    e.mov_reg(cg.X86Reg.RBP, cg.X86Reg.RSP)
    # RDI=a, RSI=b, RDX=c, RCX=count (System V ABI 4th arg)
    e.xor_reg(cg.X86Reg.R8, cg.X86Reg.R8)  # offset = 0
    e.mov_imm64(cg.X86Reg.R9, 16)           # stride = 16 bytes
    loop = e.label()
    e.mark_label(loop)
    # Load A
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RDI)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_load(cg.X86VReg.XMM0, cg.X86Reg.RAX, 0, cg.VEXWidth.XMM)
    # Load B
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RSI)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_load(cg.X86VReg.XMM1, cg.X86Reg.RAX, 0, cg.VEXWidth.XMM)
    # Add
    e.vaddps(cg.X86VReg.XMM0, cg.X86VReg.XMM0, cg.X86VReg.XMM1, cg.VEXWidth.XMM)
    # Store C
    e.mov_reg(cg.X86Reg.RAX, cg.X86Reg.RDX)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.R8)
    e.vmovups_store(cg.X86Reg.RAX, 0, cg.X86VReg.XMM0, cg.VEXWidth.XMM)
    # Advance
    e.add_reg(cg.X86Reg.R8, cg.X86Reg.R9)
    e.dec_reg(cg.X86Reg.RCX)
    e.jne(loop)
    e.pop(cg.X86Reg.RBP)
    e.ret()
    jit = cg.JITMemory()
    jit.allocate(e.take_bytes())
    return jit if jit.valid() else None


def cg_build_parametric_fused():
    """Build: void fn(float* a, float* b, float* c, int64_t count)
    C = relu(A * B) with count in RCX.
    """
    if not CG_AVAILABLE: return None
    e = cg.X86Emitter()
    e.push(cg.X86Reg.RBP)
    e.mov_reg(cg.X86Reg.RBP, cg.X86Reg.RSP)
    e.vxorps(cg.X86VReg.XMM2, cg.X86VReg.XMM2)  # zero for relu
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


# ---- Single-threaded runner (for comparison) ----

def cg_run_single(jit, a, b, c):
    addr = jit.entry()
    fn_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int64)
    fn = fn_type(addr)
    count = a.size // 4
    fn(a.ctypes.data, b.ctypes.data, c.ctypes.data, count)


# ---- Multi-threaded runner via ParallelExecutor ----

def cg_run_parallel(executor, jit, a, b, c):
    executor.execute(jit.entry(), a, b, c, a.size)


# ---- NumPy / torch / JAX ----

def numpy_add(a, b): return a + b
def numpy_fused(a, b): return np.maximum(a * b, 0)

def torch_add(a, b): return a + b
def torch_fused(a, b): return torch.relu(a * b)

@torch.compile(mode="reduce-overhead")
def tc_add(a, b): return a + b

@torch.compile(mode="reduce-overhead")
def tc_fused(a, b): return torch.relu(a * b)

@jax.jit
def jax_add(a, b): return a + b

@jax.jit
def jax_fused(a, b): return jnp.maximum(a * b, 0)


def main():
    print("=" * 85)
    print("cantors-gift (multi-threaded) vs JAX vs torch.compile vs NumPy")
    print("=" * 85)
    print(f"cantors-gift: {CG_AVAILABLE}")
    print(f"JAX: {jax.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CPU threads (torch): {torch.get_num_threads()}")
    nproc = os.cpu_count() or 1
    print(f"CPU cores: {nproc}")

    # Build kernels
    add_kernel = cg_build_parametric_add() if CG_AVAILABLE else None
    fused_kernel = cg_build_parametric_fused() if CG_AVAILABLE else None
    executor = cg.ParallelExecutor(nproc) if CG_AVAILABLE else None
    if executor:
        print(f"ParallelExecutor threads: {executor.num_threads()}")
    print()

    results = []

    for name, kernel, numpy_fn, torch_fn, tc_fn, jax_fn in [
        ("elementwise_add", add_kernel, numpy_add, torch_add, tc_add, jax_add),
        ("fused_relu_mul", fused_kernel, numpy_fused, torch_fused, tc_fused, jax_fused),
    ]:
        print("-" * 85)
        print(f"Benchmark: {name}")
        print("-" * 85)

        for size in [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]:
            M, N = size
            total = M * N
            if total % 4 != 0: continue

            a = np.random.randn(M, N).astype(np.float32)
            b = np.random.randn(M, N).astype(np.float32)
            ref = numpy_fn(a, b)

            print(f"\n  {name} size={size} ({total} elements)")
            result = {"name": name, "size": str(size)}

            # NumPy
            t = time_fn(numpy_fn, (a, b))
            result["numpy_ms"] = t * 1000
            print(f"    NumPy:          {t*1000:.4f} ms")

            # torch eager
            a_t = torch.from_numpy(a.copy()); b_t = torch.from_numpy(b.copy())
            t = time_fn(torch_fn, (a_t, b_t))
            result["torch_ms"] = t * 1000
            print(f"    torch eager:    {t*1000:.4f} ms")

            # torch.compile
            t = time_fn(tc_fn, (a_t, b_t))
            result["torch_compile_ms"] = t * 1000
            print(f"    torch.compile:  {t*1000:.4f} ms")

            # JAX
            a_j = jnp.array(a); b_j = jnp.array(b)
            t = time_jax(jax_fn, (a_j, b_j))
            result["jax_ms"] = t * 1000
            print(f"    JAX JIT:        {t*1000:.4f} ms")

            # cantors-gift single-threaded
            if kernel and executor:
                c = np.zeros_like(a)
                cg_run_single(kernel, a, b, c)
                if not np.allclose(c, ref, rtol=1e-3, atol=1e-3):
                    print(f"    cg single:      CORRECTNESS FAILED")
                    result["cg_single_ms"] = None
                else:
                    t = time_fn(cg_run_single, (kernel, a, b, c))
                    result["cg_single_ms"] = t * 1000
                    print(f"    cg single:      {t*1000:.4f} ms (verified)")

                # cantors-gift multi-threaded
                c2 = np.zeros_like(a)
                cg_run_parallel(executor, kernel, a, b, c2)
                if not np.allclose(c2, ref, rtol=1e-3, atol=1e-3):
                    print(f"    cg multi:       CORRECTNESS FAILED")
                    result["cg_multi_ms"] = None
                else:
                    t = time_fn(cg_run_parallel, (executor, kernel, a, b, c2))
                    result["cg_multi_ms"] = t * 1000
                    print(f"    cg multi ({executor.num_threads()}t):  {t*1000:.4f} ms (verified)")
            else:
                result["cg_single_ms"] = None
                result["cg_multi_ms"] = None

            results.append(result)

    # Summary
    print("\n" + "=" * 85)
    print("Summary")
    print("=" * 85)
    hdr = f"{'Benchmark':<18} {'Size':<14} {'NumPy':>8} {'torch':>8} {'compile':>8} {'JAX':>8} {'cg-1t':>8} {'cg-Nt':>8}"
    print(hdr)
    print("-" * 82)
    for r in results:
        def fmt(k):
            v = r.get(k)
            return f"{v:.3f}" if v else "N/A"
        print(f"{r['name']:<18} {r['size']:<14} {fmt('numpy_ms'):>8} {fmt('torch_ms'):>8} "
              f"{fmt('torch_compile_ms'):>8} {fmt('jax_ms'):>8} {fmt('cg_single_ms'):>8} {fmt('cg_multi_ms'):>8}")

    # Speedup table
    print("\nSpeedup vs torch.compile:")
    print(f"{'Benchmark':<18} {'Size':<14} {'cg-1t vs tc':>12} {'cg-Nt vs tc':>12} {'cg-Nt vs JAX':>13}")
    print("-" * 65)
    for r in results:
        tc = r.get("torch_compile_ms") or 0
        cg1 = r.get("cg_single_ms") or 0
        cgN = r.get("cg_multi_ms") or 0
        jx = r.get("jax_ms") or 0
        s1 = f"{tc/cg1:.2f}x" if cg1 > 0 else "N/A"
        sN = f"{tc/cgN:.2f}x" if cgN > 0 else "N/A"
        sJ = f"{jx/cgN:.2f}x" if cgN > 0 and jx > 0 else "N/A"
        print(f"{r['name']:<18} {r['size']:<14} {s1:>12} {sN:>12} {sJ:>13}")

    import json, csv
    with open("benchmarks/results_multithread.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("benchmarks/results_multithread.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","size","numpy_ms","torch_ms",
                            "torch_compile_ms","jax_ms","cg_single_ms","cg_multi_ms"])
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nResults saved to benchmarks/results_multithread.json and .csv")


if __name__ == "__main__":
    main()
