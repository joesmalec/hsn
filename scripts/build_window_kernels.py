#!/usr/bin/env python3
"""Build the P2.10 window kernels: ``build/hsn_window.fatbin`` from ``kernels/cuda/hsn_window.cu``
and its host launcher ``build/hsn_window_py*.so`` (``python/hsn/kernels/window_ext.cpp``).

Independent of ``scripts/build_kernels.py`` (the four P2.9 ops keep their own fatbin, extension
and manifest). ``--arch 86`` builds one SASS target for development; the default is the full
matrix of ``build_kernels.py`` (SASS sm_80, 86, 89, 90, 120, 121 plus PTX compute_120).
``--fatbin-only`` skips the extension (it only changes with ``window_ext.cpp`` or the launch
header). Writes ``build/window_build.json`` with the source and binary hashes that
``hsn.kernels.window`` checks before loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_kernels import PTX, SASS, find_nvcc  # noqa: E402

DEVICE_SOURCES = ("kernels/cuda/hsn_window.cu", "kernels/include/hsn_window_launch.h")
HOST_SOURCES = ("python/hsn/kernels/window_ext.cpp", "kernels/include/hsn_window_launch.h")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arch", action="append", default=[], help="SASS target(s), e.g. 86")
    parser.add_argument("--fatbin-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    nvcc = find_nvcc()
    if nvcc is None:
        print("window kernels: no CUDA toolchain (set CUDA_HOME or put nvcc on PATH)")
        return 1
    out = ROOT / "build"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "window_build.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    manifest.unlink(missing_ok=True)
    sass = tuple(args.arch) or SASS
    command = [str(nvcc), "--fatbin", "-O3", "--std=c++17", "-lineinfo"]
    for arch in sass:
        command += ["--generate-code", f"arch=compute_{arch},code=sm_{arch}"]
    if not args.arch:
        command += ["--generate-code", f"arch=compute_{PTX},code=compute_{PTX}"]
    command += ["-I", str(ROOT / "kernels/include"), str(ROOT / "kernels/cuda/hsn_window.cu")]
    command += ["-o", str(out / "hsn_window.fatbin")]
    subprocess.run(command, cwd=ROOT, check=True)
    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "nvcc": subprocess.check_output([str(nvcc), "--version"], text=True).strip(),
        "sass": list(sass),
        "ptx": None if args.arch else PTX,
        "fatbin_sha256": _sha(out / "hsn_window.fatbin"),
        "device_sources": {name: _sha(ROOT / name) for name in DEVICE_SOURCES},
    }
    if args.fatbin_only:
        for key in ("host_sources", "torch"):
            if key not in previous:
                print("window kernels: no extension built yet; run without --fatbin-only")
                return 1
            report[key] = previous[key]
    else:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_cuda_extension.py"),
                "--out",
                str(out),
                "--cuda-home",
                str(nvcc.parent.parent),
                "--jobs",
                str(args.jobs),
                "--name",
                "hsn_window_py",
                "--source",
                "python/hsn/kernels/window_ext.cpp",
            ],
            cwd=ROOT,
            check=True,
        )
        import torch

        report["host_sources"] = {name: _sha(ROOT / name) for name in HOST_SOURCES}
        report["torch"] = torch.__version__
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
