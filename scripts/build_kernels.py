#!/usr/bin/env python3
"""Compile the L2 fused kernels (one CUDA source, one Metal source) into ``build/``.

P0.1 scaffold: the kernel sources do not exist yet (they arrive in P2.10 and P2.11). The script
already probes the toolchains so that developers and CI see exactly one of:

* ``<target>: no toolchain``                        -> exit 0 (nvcc / Metal compiler absent)
* ``<target>: toolchain found, but <source> ...``   -> exit 0 (source not written yet)
* a compile step                                    -> added by P2.10 / P2.11

Usage: ``python scripts/build_kernels.py --target {cuda,metal,all,none} [--out build/]``
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES: dict[str, Path] = {
    "cuda": ROOT / "kernels" / "cuda" / "hsn_ops.cu",
    "metal": ROOT / "kernels" / "metal" / "hsn_ops.metal",
}


def have_cuda_toolchain() -> bool:
    """True when ``nvcc`` is on PATH."""
    return shutil.which("nvcc") is not None


def have_metal_toolchain() -> bool:
    """True when ``xcrun -sdk macosx metal`` runs (Xcode with the Metal toolchain component)."""
    if shutil.which("xcrun") is None:
        return False
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


TOOLCHAIN_PROBES: dict[str, Callable[[], bool]] = {
    "cuda": have_cuda_toolchain,
    "metal": have_metal_toolchain,
}


def build_target(target: str, out: Path) -> int:
    """Build one target; return a process exit status."""
    if not TOOLCHAIN_PROBES[target]():
        print(f"{target}: no toolchain")
        return 0
    source = SOURCES[target]
    if not source.exists():
        print(
            f"{target}: toolchain found, but {source.relative_to(ROOT)} does not exist yet "
            "(added in a later chunk)"
        )
        return 0
    out.mkdir(parents=True, exist_ok=True)
    # Errors over silence: a kernel source without a compile step must never pass quietly.
    print(f"{target}: compile step not implemented in P0.1", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target", required=True, choices=["cuda", "metal", "all", "none"])
    parser.add_argument("--out", type=Path, default=Path("build"))
    args = parser.parse_args(argv)

    if args.target == "none":
        print("none: nothing to do")
        return 0
    targets = ["cuda", "metal"] if args.target == "all" else [args.target]
    status = 0
    for target in targets:
        status = max(status, build_target(target, args.out))
    return status


if __name__ == "__main__":
    sys.exit(main())
