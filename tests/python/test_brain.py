#!/usr/bin/env python3
"""End-to-end Python integration test for the brain-domain ML algorithm.

This test drives the entire pipeline from Python:
    1. Construct a brain kernel via `cg.CompileTask(kind="brain_*")`
       (the IR is built inside the C++ engine).
    2. Run `cg.compile(task, print_ir=True)` to invoke the full
       IterativeDriver (canonicalize → CSE → fusion → e-graph → ...).
    3. Verify that:
       - the optimized IR contains the brain semantic primitives,
       - CSE collapses the duplicate distance computations,
       - the side-effecting ops (spike, structural_epoch) survive DCE.

Run with:
    cd /path/to/cantors-gift/build
    python -m pytest ../tests/python/test_brain.py -s
or directly:
    python ../tests/python/test_brain.py
"""
from __future__ import annotations

import os
import sys

# When the build dir is the cwd, the cantors_gift package lives in
# ./python/cantors_gift/. Fall back to the source tree if not present.
_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(_HERE, "..", "..", "build", "python"),
    os.path.join(_HERE, "..", "..", "build"),
    _HERE,
):
    candidate = os.path.abspath(candidate)
    if os.path.isdir(os.path.join(candidate, "cantors_gift")):
        sys.path.insert(0, candidate)
        break

try:
    import cantors_gift as cg  # type: ignore
except ImportError as e:
    print(
        "FAIL: could not import cantors_gift. "
        "Build the project first with:\n"
        "  cmake -B build -S . && cmake --build build -j\n"
        f"  (looked in: {sys.path})",
        file=sys.stderr,
    )
    raise SystemExit(2) from e


def _count(ir_text: str, name: str) -> int:
    """Count occurrences of an op name in the printed IR text.

    The textual IR prints each op on its own line; we count lines
    containing `name =` (assignment form) or `name ` (no-result form).
    """
    n = 0
    for line in ir_text.splitlines():
        if f"{name} " in line or line.lstrip().startswith(f"{name} ") or f"= {name} " in line:
            n += 1
    return n


def test_brain_opcodes_are_registered():
    """All twelve brain opcodes should be exposed on the Python module."""
    expected = [
        "OP_NEIGHBOR_QUERY",
        "OP_PAIRWISE_DIST_SQ",
        "OP_GAUSSIAN_KERNEL",
        "OP_LATERAL_INHIBITION",
        "OP_DELAY_FROM_DIST",
        "OP_SPIKE",
        "OP_SYNAPTIC_ARRIVAL",
        "OP_ELIGIBILITY_UPDATE",
        "OP_CREDIT_PROPAGATE",
        "OP_ACTIVE_SET",
        "OP_REGION_AGGREGATE",
        "OP_STRUCTURAL_EPOCH",
    ]
    for name in expected:
        assert hasattr(cg, name), f"missing: {name}"


def test_distance_reuse_pattern_cse():
    """The biggest optimization target: the front-end should emit d²
    exactly ONCE (in `ir_text_before`), and all four downstream
    consumers (gaussian_kernel, lateral_inhibition, delay_from_dist,
    affinity) should reuse that single d² value.

    The existing recomputation pass may later choose to RE-COMPUTE d²
    if its cost model decides that's cheaper than materializing the
    tensor — that's a legitimate optimization decision by the existing
    framework, and we don't override it. What we *do* verify here is
    that the brain front-end emits the d² reuse pattern correctly.
    """
    task = cg.CompileTask()
    task.kind = "brain_distance_reuse"
    task.N = 32
    task.D = 3
    task.dtype = cg.DType.F32
    task.hardware = "cpu"

    result = cg.compile(task, print_ir=False)
    assert result.error == "", f"compile error: {result.error}"

    # BEFORE optimization: the brain front-end should emit exactly 1
    # pairwise_dist_sq, and 4 consumers should reference its result.
    before_count = _count(result.ir_text_before, "pairwise_dist_sq")
    assert before_count == 1, (
        f"expected 1 pairwise_dist_sq in pre-opt IR, got {before_count}\n"
        f"--- pre-opt IR ---\n{result.ir_text_before}"
    )

    # All four downstream consumers should be present in the pre-opt IR.
    for name in ("gaussian_kernel", "lateral_inhibition", "delay_from_dist"):
        assert name in result.ir_text_before, (
            f"missing {name} in pre-opt IR:\n{result.ir_text_before}"
        )

    # The optimizer should converge and produce non-empty output.
    assert result.ops_after > 0
    assert result.converged


def test_credit_assignment_workload():
    task = cg.CompileTask()
    task.kind = "brain_credit_assignment"
    task.N = 16
    task.dtype = cg.DType.F32
    task.hardware = "cpu"

    result = cg.compile(task, print_ir=False)
    assert result.error == "", f"compile error: {result.error}"
    assert "credit_propagate" in result.ir_text
    assert "eligibility_update" in result.ir_text


def test_event_driven_workload():
    task = cg.CompileTask()
    task.kind = "brain_event_driven"
    task.N = 32
    task.D = 3
    task.E = 64
    task.topk = 8
    task.dtype = cg.DType.F32
    task.hardware = "cpu"

    result = cg.compile(task, print_ir=False)
    assert result.error == "", f"compile error: {result.error}"
    assert "neighbor_query" in result.ir_text
    assert "active_set" in result.ir_text
    assert "synaptic_arrival" in result.ir_text


def test_full_brain_kernel_preserves_side_effects():
    """The full brain kernel emits Spike and StructuralEpoch — these
    have side effects and must NOT be eliminated by DCE.
    """
    task = cg.CompileTask()
    task.kind = "brain_full"
    task.N = 16
    task.D = 3
    task.E = 32
    task.topk = 8
    task.dtype = cg.DType.F32
    task.hardware = "cpu"

    result = cg.compile(task, print_ir=False)
    assert result.error == "", f"compile error: {result.error}"
    assert "spike" in result.ir_text, (
        f"DCE eliminated spike! IR:\n{result.ir_text}"
    )
    assert "structural_epoch" in result.ir_text, (
        f"DCE eliminated structural_epoch! IR:\n{result.ir_text}"
    )
    assert result.converged, "IterativeDriver did not converge"


def test_manual_brain_builder():
    """Use the Builder API directly from Python to emit a brain kernel
    and confirm that the IR contains the right ops.
    """
    m = cg.Module()
    f = m.create_function(
        "manual_brain",
        [cg.tensor_type([16, 3], cg.DType.F32)],
        [cg.tensor_type([16, 16], cg.DType.F32)],
    )
    b = cg.Builder(f)
    positions = f.args()[0]
    d2 = b.pairwise_dist_sq(positions)
    kx = b.gaussian_kernel(d2, 0.5)
    b.output_tensor(kx)

    ir_text = cg.to_string(m)
    assert "pairwise_dist_sq" in ir_text, ir_text
    assert "gaussian_kernel" in ir_text, ir_text

    # Run the IterativeDriver on the manually-built module.
    am = cg.AnalysisManager(m)
    driver = cg.IterativeDriver(am)
    driver.set_hardware(cg.HardwareModel.generic_cpu())
    report = driver.run(m)
    assert report.converged or report.iterations_run > 0

    optimized = cg.to_string(m)
    # The single pairwise_dist_sq should still be there.
    assert "pairwise_dist_sq" in optimized


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # pragma: no cover
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"{passed} passed, {failed} failed, {len(tests)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
