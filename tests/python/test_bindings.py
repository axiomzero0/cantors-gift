#!/usr/bin/env python3
"""Test suite for the cantors_gift Python bindings.

Run with:
    python3.13 -m pytest tests/python/test_bindings.py
or:
    python3.13 tests/python/test_bindings.py
"""
import sys
import os

# Add the build output to the path so we can import the native module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))

import cantors_gift as cg


def test_dtype_enum():
    assert cg.DType.F32 is not None
    assert cg.dtype_size(cg.DType.F32) == 4
    assert cg.dtype_size(cg.DType.F64) == 8
    assert cg.is_float(cg.DType.F32)
    assert cg.is_int(cg.DType.I32)
    assert cg.promote(cg.DType.F32, cg.DType.F64) == cg.DType.F64


def test_shape_system():
    N = cg.DimExpr.make_symbol(1, "N")
    assert N is not None
    assert N.is_symbol()

    five = cg.DimExpr.make_constant(5)
    assert five.is_constant()
    assert five.is_constant_value(5)

    s = cg.Shape([8, 16, 32])
    assert s.rank() == 3
    assert s.num_elements() == 8 * 16 * 32


def test_constraint_solver():
    M = cg.DimExpr.make_symbol(1, "M")
    N = cg.DimExpr.make_symbol(2, "N")
    K = cg.DimExpr.make_symbol(3, "K")

    cs = cg.ConstraintSet()
    cs.add_eq(M, N)
    cs.add_mod_eq(K, 16, 0)

    solver = cg.Solver(cs)
    assert solver.prove_equal(M, N) == cg.SolverResult.ProvedTrue
    assert solver.prove_divisible(K, 16) == cg.SolverResult.ProvedTrue


def test_shape_inference():
    A = cg.Shape([8, 16])
    B = cg.Shape([16, 32])
    r = cg.infer_matmul(A, B)
    assert r.ok
    assert r.shape.rank() == 2


def test_ir_builder():
    m = cg.Module()
    f = m.create_function("test",
        [cg.tensor_type([16, 32], cg.DType.F32),
         cg.tensor_type([32, 64], cg.DType.F32)],
        [cg.tensor_type([16, 64], cg.DType.F32)])
    b = cg.Builder(f)
    mm = b.matmul(f.args()[0], f.args()[1])
    r = b.relu(mm)
    b.output_tensor(r)

    s = cg.to_string(m)
    assert "matmul" in s
    assert "relu" in s


def test_op_registry():
    reg = cg.OpRegistry.instance()
    assert reg is not None


def test_optimization_pipeline():
    m = cg.Module()
    f = m.create_function("test",
        [cg.tensor_type([64, 64], cg.DType.F32),
         cg.tensor_type([64, 64], cg.DType.F32)],
        [cg.tensor_type([64, 64], cg.DType.F32)])
    b = cg.Builder(f)
    mm = b.matmul(f.args()[0], f.args()[1])
    r = b.relu(mm)
    b.output_tensor(r)

    am = cg.AnalysisManager(m)
    driver = cg.IterativeDriver(am)
    report = driver.run(m)

    assert report.iterations_run >= 1
    assert report.barrier_report.legal


def test_individual_passes():
    m = cg.Module()
    f = m.create_function("test",
        [cg.tensor_type([8, 8], cg.DType.F32)],
        [cg.tensor_type([8, 8], cg.DType.F32)])
    b = cg.Builder(f)
    r1 = b.relu(f.args()[0])
    r2 = b.exp(r1)
    b.output_tensor(r2)

    am = cg.AnalysisManager(m)

    # Run individual passes
    cg.CanonicalizePass().run(m, am)
    cg.CSEPass().run(m, am)
    cg.DCEPass().run(m, am)
    cg.AlgebraicSimplificationPass().run(m, am)
    cg.FusionPass().run(m, am)
    cg.MemoryPlanningPass().run(m, am)

    s = cg.to_string(m)
    assert "relu" in s or "fused" in s


def test_gta_analysis():
    m = cg.Module()
    f = m.create_function("test",
        [cg.tensor_type([16, 32], cg.DType.F32),
         cg.tensor_type([32, 64], cg.DType.F32)],
        [cg.tensor_type([16, 64], cg.DType.F32)])
    b = cg.Builder(f)
    mm = b.matmul(f.args()[0], f.args()[1])
    r = b.relu(mm)
    b.output_tensor(r)

    am = cg.AnalysisManager(m)
    gta = cg.GlobalAnalysisManager(am)

    df = gta.dataflow()
    assert df.critical_path_length() >= 2

    ai = gta.intensity()
    assert ai.total_flops() > 0
    assert ai.total_bytes() > 0


def test_ptx_backend():
    cgm = cg.CGModule()
    fn = cgm.create_function("test_kernel")
    a = fn.allocate(cg.DType.F32, 8)
    b = fn.allocate(cg.DType.F32, 8)
    acc = fn.allocate(cg.DType.F32, 8)
    ptr = fn.allocate(cg.DType.U64)
    off = fn.allocate(cg.DType.I64)
    fn.emit(cg.make_vector_load(a, ptr, off, 8, cg.MemorySpace.Generic))
    fn.emit(cg.make_vector_load(b, ptr, off, 8, cg.MemorySpace.Generic))
    fn.emit(cg.make_fma(acc, a, b))
    fn.emit(cg.make_vector_store(ptr, off, acc, cg.MemorySpace.Generic))
    fn.emit(cg.make_barrier())

    nv = cg.NvidiaBackend()
    exe = nv.compile(cgm)
    assert ".version" in exe.ptx_text
    assert ".target" in exe.ptx_text
    assert "fma.rn" in exe.ptx_text
    assert "bar.sync" in exe.ptx_text


def test_x86_backend():
    cgm = cg.CGModule()
    fn = cgm.create_function("test_kernel")
    a = fn.allocate(cg.DType.F32, 8)
    b = fn.allocate(cg.DType.F32, 8)
    acc = fn.allocate(cg.DType.F32, 8)
    ptr = fn.allocate(cg.DType.U64)
    off = fn.allocate(cg.DType.I64)
    fn.emit(cg.make_vector_load(a, ptr, off, 8, cg.MemorySpace.Generic))
    fn.emit(cg.make_fma(acc, a, b))
    fn.emit(cg.make_vector_store(ptr, off, acc, cg.MemorySpace.Generic))
    fn.emit(cg.make_barrier())

    cpu = cg.CpuBackend()
    exe = cpu.compile(cgm)
    assert len(exe.machine_code) > 0
    # ret instruction
    assert exe.machine_code[-1] == 0xC3
    # push rbp
    assert exe.machine_code[0] == 0x55


def test_x86_emitter():
    e = cg.X86Emitter()
    e.push(cg.X86Reg.RBP)
    e.mov_reg(cg.X86Reg.RBP, cg.X86Reg.RSP)
    e.mov_imm64(cg.X86Reg.RAX, 0x123456789ABCDEF0)
    e.add_reg(cg.X86Reg.RAX, cg.X86Reg.RCX)
    e.ret()
    bytes_out = e.take_bytes()
    assert len(bytes_out) > 0
    assert bytes_out[-1] == 0xC3  # ret


def test_autotuner():
    space = cg.ScheduleSpace.grid_matmul([64, 128], [64, 128], [32], [8, 16])
    assert space.size() > 0

    def benchmark(s):
        f = cg.extract_features(s)
        cost = 1000.0
        cost += abs(f.m_tile - 64) * 2.0
        cost += abs(f.vector_width - 8) * 3.0
        return cost

    result = cg.bayesian_autotune(space, benchmark, 15, 3)
    assert result.total_benchmarks > 0
    assert result.best_runtime < float('inf')
    assert result.best_runtime < 1100.0  # near optimum


def test_hardware_model():
    cpu = cg.HardwareModel.generic_cpu()
    assert cpu.peak_flops(cg.DType.F32) > 0
    assert cpu.device.kind == cg.DeviceKind.CPU

    gpu = cg.HardwareModel.generic_nvidia_gpu()
    assert gpu.peak_flops(cg.DType.F16, True) > 0
    assert gpu.device.kind == cg.DeviceKind.CUDA


def test_cost_estimator():
    hw = cg.HardwareModel.generic_nvidia_gpu()
    est = cg.CostEstimator(hw)
    s = cg.Schedule()
    c = est.estimate_matmul(1024, 1024, 1024, cg.DType.F32, s)
    assert c.flops > 0
    assert c.bytes_global > 0


def test_runtime():
    rt = cg.Runtime()
    assert rt.num_devices() == 0

    key = cg.KernelCache.compute_key("graph", "shapes", "dtypes", "hw")
    assert key != 0


def test_lowering():
    cgm = cg.CGModule()
    fn = cgm.create_function("test")
    ptr = fn.allocate(cg.DType.U64)
    off = fn.allocate(cg.DType.I64)
    a = fn.allocate(cg.DType.F32, 8)
    fn.emit(cg.make_vector_load(a, ptr, off, 8, cg.MemorySpace.Generic))

    ptx = cg.lower_to_ptx(fn, "test_kernel")
    assert ".version" in ptx

    x86_bytes = cg.lower_to_x86(fn)
    assert len(x86_bytes) > 0


def test_schedule():
    s = cg.Schedule()
    s.add(cg.Transform(kind=cg.TransformKind.Tile, dim="m", factor=64))
    s.add(cg.Transform(kind=cg.TransformKind.Vectorize, dim="n_inner", factor=8))
    assert len(s.transforms()) == 2

    space = cg.ScheduleSpace()
    space.add(s)
    assert space.size() == 1


def test_end_to_end():
    """Full end-to-end: build IR -> optimize -> lower -> compile."""
    # 1. Build IR
    m = cg.Module()
    f = m.create_function("mlp",
        [cg.tensor_type([128, 256], cg.DType.F32),
         cg.tensor_type([256, 512], cg.DType.F32),
         cg.tensor_type([128, 512], cg.DType.F32)],
        [cg.tensor_type([128, 512], cg.DType.F32)])
    b = cg.Builder(f)
    mm = b.matmul(f.args()[0], f.args()[1])
    bd = b.add(mm, f.args()[2])
    r = b.relu(bd)
    b.output_tensor(r)

    # 2. Optimize
    am = cg.AnalysisManager(m)
    driver = cg.IterativeDriver(am)
    driver.set_hardware(cg.HardwareModel.generic_nvidia_gpu())
    report = driver.run(m)
    assert report.barrier_report.legal

    # 3. Build codegen IR
    cgm = cg.CGModule()
    fn = cgm.create_function("mlp_kernel")
    a = fn.allocate(cg.DType.F32, 8)
    b_val = fn.allocate(cg.DType.F32, 8)
    acc = fn.allocate(cg.DType.F32, 8)
    ptr = fn.allocate(cg.DType.U64)
    off = fn.allocate(cg.DType.I64)
    fn.emit(cg.make_vector_load(a, ptr, off, 8, cg.MemorySpace.Generic))
    fn.emit(cg.make_vector_load(b_val, ptr, off, 8, cg.MemorySpace.Generic))
    fn.emit(cg.make_fma(acc, a, b_val))
    fn.emit(cg.make_vector_store(ptr, off, acc, cg.MemorySpace.Generic))
    fn.emit(cg.make_barrier())

    # 4. Compile to PTX
    nv = cg.NvidiaBackend("sm_80")
    exe = nv.compile(cgm)
    assert "fma.rn" in exe.ptx_text

    # 5. Compile to x86
    cpu = cg.CpuBackend()
    exe2 = cpu.compile(cgm)
    assert len(exe2.machine_code) > 0

    print("\n=== End-to-end pipeline succeeded ===")
    print(f"PTX: {len(exe.ptx_text)} chars")
    print(f"x86: {len(exe2.machine_code)} bytes")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
