#!/usr/bin/env python3
"""P2.10 build/test sequence. Invoke through run_when_idle.py, never at import time."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument(
        "--regressions-only",
        action="store_true",
        help="reuse validated CUDA artifacts and run the full suites and code-quality checks",
    )
    args = parser.parse_args()
    env = os.environ.copy()
    for name, path in {
        "UV_CACHE_DIR": "build/uv-cache",
        "CARGO_HOME": "build/cargo-home",
        "CARGO_TARGET_DIR": "target",
        "TORCHINDUCTOR_CACHE_DIR": "build/inductor",
        "TRITON_CACHE_DIR": "build/triton",
        "TMPDIR": "build/tmp",
        "UV_PROJECT_ENVIRONMENT": ".venv",
    }.items():
        env[name] = str(ROOT / path)
    env.update(
        HSN_REQUIRE_CUDA="1",
        CARGO_BUILD_JOBS="2",
        MAX_JOBS="2",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        UV_CONCURRENT_DOWNLOADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="2",
    )
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", os.defpath)
    python = str(ROOT / ".venv/bin/python")
    commands = []
    if not args.skip_sync:
        commands.append(["uv", "sync", "--locked", "--extra", "dev", "--extra", "cuda"])
        commands.append(
            ["uv", "run", "--no-sync", "maturin", "develop", "--uv", "-m", "rust/hsn-py/Cargo.toml"]
        )
    if not args.regressions_only:
        commands += [
            [python, "scripts/build_kernels.py", "--target", "cuda", "--jobs", "2"],
            [python, "-m", "pytest", "-q", "tests/kernels/test_p2_10_cuda.py"],
            ["cargo", "test", "--locked", "-p", "hsn-kernels", "--features", "cuda"],
        ]
    commands += [
        [python, "-m", "ruff", "check", "."],
        [python, "-m", "ruff", "format", "--check", "."],
        [python, "-m", "mypy", "--strict", "python/hsn"],
        ["cargo", "fmt", "--all", "--", "--check"],
        [
            "cargo",
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--features",
            "hsn-kernels/cuda",
            "--",
            "-D",
            "warnings",
        ],
        [python, "-m", "pytest", "-q"],
        ["cargo", "test", "--locked", "--workspace"],
    ]
    for command in commands:
        print(f"Running: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
