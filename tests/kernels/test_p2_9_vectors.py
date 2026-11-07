"""P2.9 tests: the C header ``kernels/include/hsn_ops.h`` compiles under gcc, clang and nvcc (as C
and C++) and its constants equal :mod:`hsn.kernels.spec`; the committed test vectors match their
manifest and the spec; and the P2.5 native ops reproduce every vector (all three sizes, fp32) —
``test_p2_9_native_matches_oracle_vectors``, the chunk's cross-check of the PyTorch path against
the Rust CPU oracle."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from hsn.kernels import spec, testvectors

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


# --------------------------------------------------------------------------------------------------
# The vectors
# --------------------------------------------------------------------------------------------------

CASES = [(op, size) for size in spec.SIZES for op in spec.OP_NAMES]
CASE_IDS = [f"{size}-{op}" for op, size in CASES]
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("mps", marks=pytest.mark.mps),
]


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    return torch.empty(0, device=device_type).device


@pytest.mark.parametrize("device_type", DEVICES)
@pytest.mark.parametrize(("op", "size"), CASES, ids=CASE_IDS)
def test_p2_9_native_matches_oracle_vectors(op: str, size: str, device_type: str) -> None:
    """Every forward output and every gradient of the native op agrees with the oracle's vector
    within the fp32 tolerance (``atol = rtol = 1e-5``; ``age_out`` exactly)."""
    device = _device(device_type)
    vector = testvectors.load_vector(op, size, device=device)
    actual = testvectors.run_native(vector)
    compared = {t.name for t in spec.OPS[op].tensors if t.role in ("output", "grad")}
    assert set(actual) == compared - testvectors.NATIVE_UNCOMPARED
    failures = []
    for name in sorted(actual):
        assert actual[name].device == device, name
        report = spec.allclose_report(actual[name], vector.tensors[name], torch.float32)
        if not report.ok:
            failures.append(f"{name}: {report}")
    assert not failures, f"{op} at {size} on {device}: " + "; ".join(failures)


def test_p2_9_manifest_matches_files() -> None:
    """The manifest lists exactly the 12 (size, op) files present, each matching its byte count
    and SHA-256 (checked by ``load_vector``), at this spec version."""
    manifest = testvectors.load_manifest()
    assert manifest.spec_version == spec.SPEC_VERSION
    on_disk = sorted(p.name for p in testvectors.DEFAULT_DIR.glob("*.safetensors"))
    assert on_disk == sorted(entry.path for entry in manifest.files)
    assert len(manifest.files) == len(spec.SIZES) * len(spec.OPS)
    for entry in manifest.files:
        vector = testvectors.load_vector(entry.op, entry.size)
        assert vector.dims == entry.dims


def test_p2_9_vectors_match_spec_shapes() -> None:
    """Every tensor has the spec's dtype and shape at its size, and the vectors cover the cases
    the spec calls out: repeated cells in gather/messages, an empty CSR row among the selected
    cells at ``odd``, padded gated slots holding never-read values, an absent unit at ``odd``."""
    for op, size in CASES:
        vector = testvectors.load_vector(op, size)
        for tensor_spec in spec.OPS[op].tensors:
            tensor = vector.tensors[tensor_spec.name]
            assert tuple(tensor.shape) == tensor_spec.resolve(vector.dims), tensor_spec.name
            expected = torch.float32 if tensor_spec.kind == "float" else torch.int32
            assert tensor.dtype == expected, tensor_spec.name
        t = vector.tensors
        if op in ("gather_concat", "csr_messages"):
            last = t["cell_idx"][-1]
            assert int(last[0]) == int(last[-1]), "the last row repeats a cell"
        if op == "csr_messages" and size == "odd":
            degree = t["indptr"][1:] - t["indptr"][:-1]
            assert bool((degree[t["cell_idx"].long()] == 0).any()), "an empty row is selected"
        if op == "gated_delta_update":
            padded = t["cell_idx"][t["valid"] == 0]
            n_cells = vector.dims["N"]
            if size != "tiny":
                assert bool(((padded < 0) | (padded >= n_cells)).any()), "out-of-range padding"
            assert padded.numel() > 0
        if op == "grouped_unit_mlp" and size == "odd":
            assert 3 not in t["unit_idx"].tolist()
            assert not bool(t["grad_W1"][3].any()) and not bool(t["grad_b2"][3].any())


def test_p2_9_load_refuses_tampered_vector(tmp_path: Path) -> None:
    """A file whose bytes no longer match its manifest entry, or a manifest of another spec
    version, is refused."""
    for path in testvectors.DEFAULT_DIR.iterdir():
        if path.name.startswith("tiny_") or path.name == testvectors.MANIFEST:
            shutil.copy(path, tmp_path / path.name)
    manifest_path = tmp_path / testvectors.MANIFEST
    doc = json.loads(manifest_path.read_text())
    target = tmp_path / testvectors.file_name("gated_delta_update", "tiny")
    data = bytearray(target.read_bytes())
    data[-1] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(testvectors.VectorError, match="does not match MANIFEST.json"):
        testvectors.load_vector("gated_delta_update", "tiny", root=tmp_path)
    # The untouched tiny vectors still load from the copy.
    assert testvectors.load_vector("gather_concat", "tiny", root=tmp_path).size == "tiny"
    doc["spec_version"] = spec.SPEC_VERSION + 1
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(testvectors.VectorError, match="spec version"):
        testvectors.load_vector("gather_concat", "tiny", root=tmp_path)


def test_p2_9_run_native_keeps_global_rng() -> None:
    """Running a vector builds modules but never advances the caller's random state."""
    before = torch.random.get_rng_state()
    for op in spec.OP_NAMES:
        testvectors.run_native(testvectors.load_vector(op, "tiny"))
    assert torch.equal(torch.random.get_rng_state(), before)
