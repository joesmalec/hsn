"""P2.9 tests: the C header ``kernels/include/hsn_ops.h`` compiles under gcc, clang and nvcc (as C
and C++), its constants equal :mod:`hsn.kernels.spec`, and the spec's tolerance and int32 helpers
behave."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from hsn.kernels import spec

REPO = Path(__file__).resolve().parents[2]
INCLUDE = REPO / "kernels" / "include"
HEADER = INCLUDE / "hsn_ops.h"


# --------------------------------------------------------------------------------------------------
# The header
# --------------------------------------------------------------------------------------------------


def _header_defines() -> dict[str, int]:
    """Every ``#define HSN_<NAME> <integer>`` of the header (the include guard has no value)."""
    pattern = re.compile(r"^#define\s+(HSN_\w+)\s+(-?\d+)\b", re.MULTILINE)
    return {name: int(value) for name, value in pattern.findall(HEADER.read_text())}


def test_p2_9_header_constants_match_spec_py() -> None:
    assert _header_defines() == spec.HEADER_CONSTANTS


def _find_compiler(name: str) -> str | None:
    found = shutil.which(name)
    if found is None and name == "nvcc":
        homes = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"]
        for home in filter(None, homes):
            candidate = Path(home) / "bin" / "nvcc"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return found


#: (compiler, language): C11 and C++17 under gcc and clang, and a CUDA translation unit under nvcc.
COMPILERS = [("gcc", "c"), ("g++", "c++"), ("clang", "c"), ("clang++", "c++"), ("nvcc", "cu")]


@pytest.mark.parametrize(("compiler", "language"), COMPILERS, ids=[c for c, _ in COMPILERS])
def test_p2_9_header_compiles(compiler: str, language: str, tmp_path: Path) -> None:
    """A translation unit that includes the header twice (the include guard) compiles with every
    warning as an error. A compiler that is not installed skips with its name; on hosted CI gcc and
    clang run on ubuntu and Apple clang on macOS (where ``gcc`` is clang), nvcc runs on
    ``profile:rtx3060``."""
    exe = _find_compiler(compiler)
    if exe is None:
        pytest.skip(f"{compiler} is not installed")
    suffix = {"c": ".c", "c++": ".cpp", "cu": ".cu"}[language]
    source = tmp_path / f"hsn_ops_check{suffix}"
    source.write_text('#include "hsn_ops.h"\n#include "hsn_ops.h"\n')
    obj = tmp_path / "hsn_ops_check.o"
    warnings = ["-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    if language == "c":
        cmd = [exe, "-std=c11", *warnings, "-I", str(INCLUDE), "-c", str(source), "-o", str(obj)]
    elif language == "c++":
        cmd = [exe, "-std=c++17", *warnings, "-I", str(INCLUDE), "-c", str(source), "-o", str(obj)]
    else:
        cmd = [
            exe,
            "-std=c++17",
            "-Xcompiler",
            "-Wall,-Wextra,-Werror",
            "-I",
            str(INCLUDE),
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    assert result.returncode == 0, (
        f"{' '.join(cmd)} failed ({version.stdout.strip().splitlines()[:1]}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


# --------------------------------------------------------------------------------------------------
# The spec helpers
# --------------------------------------------------------------------------------------------------


def test_p2_9_spec_tolerance_and_int32_extent() -> None:
    assert spec.tolerance(torch.float32) == spec.Tolerance(atol=1e-5, rtol=1e-5)
    assert spec.tolerance(torch.bfloat16) == spec.Tolerance(atol=1e-2, rtol=1e-2)
    with pytest.raises(ValueError, match="no P2.9 tolerance"):
        spec.tolerance(torch.float64)

    expected = torch.tensor([1.0, -2.0, 100.0])
    close = expected + torch.tensor([9e-6, -2e-5, 1e-3])  # bounds 1.1e-5, 3e-5, 1.01e-3
    report = spec.allclose_report(close, expected, torch.float32)
    assert report.ok and report.n_bad == 0 and report.max_ratio <= 1.0
    far = expected + torch.tensor([0.0, 0.0, 2e-3])
    assert not spec.allclose_report(far, expected, torch.float32).ok
    nan = expected.clone()
    nan[0] = float("nan")
    assert spec.allclose_report(nan, expected, torch.float32).n_bad == 1
    ints = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert spec.allclose_report(ints.to(torch.int64), ints, torch.float32).ok
    assert not spec.allclose_report(ints + 1, ints, torch.float32).ok
    with pytest.raises(ValueError, match="shape mismatch"):
        spec.allclose_report(expected[:2], expected, torch.float32)

    assert spec.check_int32_extent("W1", (8, 256, 512)) == 8 * 256 * 512
    with pytest.raises(OverflowError, match="more than INT32_MAX"):
        spec.check_int32_extent("x", (1 << 16, 1 << 15))
    with pytest.raises(OverflowError, match="exceeds INT32_MAX"):
        spec.check_int32_extent("x", (2**31,))


def test_p2_9_spec_tables_are_consistent() -> None:
    """Every gradient names a float input of the same shape; names are unique per op; every shape
    symbol resolves against the op's dimensions."""
    for op in spec.OPS.values():
        names = [t.name for t in op.tensors]
        assert len(set(names)) == len(names), op.name
        for grad in op.by_role("grad"):
            source = op.tensor(grad.name.removeprefix("grad_"))
            assert source.role == "input" and source.kind == "float"
            assert source.shape == grad.shape
        dims = dict.fromkeys(op.dims, 3)
        for tensor in op.tensors:
            assert all(n >= 3 for n in tensor.resolve(dims)), tensor
    assert spec.OP_NAMES == (
        "gather_concat",
        "grouped_unit_mlp",
        "csr_messages",
        "gated_delta_update",
    )
    assert spec.SIZES["ref"].P == 16
