#!/usr/bin/env python3
"""Build the host extension in the active worktree environment, without a GPU context."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    # P2.10 window kernels build a second, independent extension with the same recipe.
    parser.add_argument("--name", default="hsn_cuda_py")
    parser.add_argument("--source", default="python/hsn/kernels/cuda_ext.cpp")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    out = args.out.resolve()
    if not out.is_relative_to(root) or args.jobs < 1:
        parser.error("output must be in this worktree and jobs must be positive")
    os.environ["CUDA_HOME"] = str(args.cuda_home)
    os.environ["MAX_JOBS"] = str(args.jobs)
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Device code lives only in the shared fatbin, loaded through libcuda.
    sys.argv = [
        sys.argv[0],
        "build_ext",
        "--build-lib",
        str(out),
        "--build-temp",
        str(out / ("extension-temp" if args.name == "hsn_cuda_py" else f"{args.name}-temp")),
    ]
    setup(
        name=args.name,
        # This distribution contains only the extension, not repository packages.
        packages=[],
        py_modules=[],
        ext_modules=[
            CUDAExtension(
                name=args.name,
                sources=[str(root / args.source)],
                include_dirs=[str(root / "kernels/include")],
                libraries=["cuda"],
                extra_compile_args={"cxx": ["-O3", "-std=c++20"]},
            )
        ],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    )


if __name__ == "__main__":
    main()
