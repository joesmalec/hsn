#!/usr/bin/env python3
"""Build precompiled kernels. Run real builds through scripts/run_when_idle.py.

--dry-run only prints commands; no torch import, compilation, CUDA context or output
directory. Explicit CUDA requests fail when the toolchain is missing; all may skip it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SASS = ("80", "86", "89", "90", "120", "121")
PTX = "120"


def find_nvcc() -> Path | None:
    """Explicit CUDA roots take priority over PATH and the system installation."""
    for name in ("CUDA_HOME", "CUDA_PATH"):
        if root := os.environ.get(name):
            candidate = Path(root) / "bin/nvcc"
            return candidate.resolve() if os.access(candidate, os.X_OK) else None
    if found := shutil.which("nvcc"):
        return Path(found).resolve()
    candidate = Path("/usr/local/cuda/bin/nvcc")
    return candidate.resolve() if os.access(candidate, os.X_OK) else None


def cuda_commands(nvcc: Path, out: Path, jobs: int, fatbin_only: bool) -> list[list[str]]:
    command = [str(nvcc), "--fatbin", "-O3", "--std=c++17", "--fmad=false", "-lineinfo"]
    # Exact erf GELU and fixed fp32 reduction order: no --use_fast_math or 'a' architecture.
    for arch in SASS:
        command += ["--generate-code", f"arch=compute_{arch},code=sm_{arch}"]
    command += ["--generate-code", f"arch=compute_{PTX},code=compute_{PTX}"]
    command += ["-I", str(ROOT / "kernels/include"), str(ROOT / "kernels/cuda/hsn_ops.cu")]
    command += ["-o", str(out / "hsn_ops.fatbin")]
    commands = [command]
    if not fatbin_only:
        commands.append(
            [
                sys.executable,
                str(ROOT / "scripts/build_cuda_extension.py"),
                "--out",
                str(out),
                "--cuda-home",
                str(nvcc.parent.parent),
                "--jobs",
                str(jobs),
            ]
        )
    return commands


def check_targets(nvcc: Path) -> None:
    result = subprocess.run(
        [str(nvcc), "--list-gpu-code"], text=True, capture_output=True, check=True
    )
    missing = {f"sm_{arch}" for arch in SASS} - set(result.stdout.split())
    ptx = subprocess.run([str(nvcc), "--list-gpu-arch"], text=True, capture_output=True, check=True)
    if missing or f"compute_{PTX}" not in ptx.stdout.split():
        raise RuntimeError(f"nvcc lacks required targets: {sorted(missing)}, PTX compute_{PTX}")


def build_cuda(out: Path, *, dry_run: bool, jobs: int, fatbin_only: bool, required: bool) -> int:
    nvcc = find_nvcc()
    if nvcc is None:
        print("cuda: no toolchain (set CUDA_HOME or put nvcc on PATH)", file=sys.stderr)
        return 1 if required else 0
    commands = cuda_commands(nvcc, out, jobs, fatbin_only)
    if dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    check_targets(nvcc)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "cuda_build.json"
    manifest.unlink(missing_ok=True)
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    sources = [
        ROOT / path
        for path in (
            "kernels/cuda/hsn_ops.cu",
            "kernels/cuda/hsn_common.cuh",
            "kernels/include/hsn_cuda_launch.h",
            "kernels/include/hsn_ops.h",
            "python/hsn/kernels/cuda_ext.cpp",
        )
    ]
    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "nvcc": subprocess.check_output([str(nvcc), "--version"], text=True).strip(),
        "sass": list(SASS),
        "ptx": PTX,
        "commands": commands,
        "fatbin_sha256": hashlib.sha256((out / "hsn_ops.fatbin").read_bytes()).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "extension_built": not fatbin_only,
    }
    if not fatbin_only:
        import torch

        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["cuda", "metal", "all", "none"])
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fatbin-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("--out must be inside this worktree")
    try:
        status = 0
        if args.target in ("cuda", "all"):
            status = build_cuda(
                out,
                dry_run=args.dry_run,
                jobs=args.jobs,
                fatbin_only=args.fatbin_only,
                required=args.target == "cuda",
            )
        if args.target in ("metal", "all"):
            print("metal: source not implemented yet (P2.11)")
        return status
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"kernel build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
