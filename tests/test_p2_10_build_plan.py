"""Stdlib-only build planning checks: never import torch or invoke a compiler."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_kernels", ROOT / "scripts/build_kernels.py")
assert SPEC and SPEC.loader
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class BuildPlanTests(unittest.TestCase):
    def test_explicit_cuda_home_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = Path(tmp) / "bin/nvcc"
            nvcc.parent.mkdir()
            nvcc.write_text("unused")
            nvcc.chmod(0o700)
            with (
                patch.dict(os.environ, {"CUDA_HOME": tmp}),
                patch.object(build.shutil, "which") as which,
            ):
                # resolved: a temporary directory may sit behind a symlink (/var on macOS)
                self.assertEqual(build.find_nvcc().resolve(), nvcc.resolve())
                which.assert_not_called()

    def test_broken_explicit_root_does_not_pick_another_toolkit(self):
        with patch.dict(os.environ, {"CUDA_HOME": "/does/not/exist"}):
            self.assertIsNone(build.find_nvcc())

    def test_all_sass_and_ptx_without_runtime_compiler(self):
        commands = build.cuda_commands(Path("/cuda/bin/nvcc"), ROOT / "build", 2, False)
        flags = commands[0]
        for arch in ("80", "86", "89", "90", "120", "121"):
            self.assertIn(f"arch=compute_{arch},code=sm_{arch}", flags)
        self.assertIn("arch=compute_120,code=compute_120", flags)
        self.assertIn("--fatbin", flags)
        self.assertFalse(any("120a" in x or "nvrtc" in x or "fast_math" in x for x in flags))
        self.assertEqual(len(commands), 2)

    def test_dry_run_does_not_spawn_or_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp) / "missing"
            with (
                patch.object(build, "find_nvcc", return_value=Path("/cuda/bin/nvcc")),
                patch.object(build.subprocess, "run") as run,
                patch.object(build.subprocess, "check_output") as check,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    build.main(["--target", "cuda", "--dry-run", "--out", str(out)]), 0
                )
                run.assert_not_called()
                check.assert_not_called()
                self.assertFalse(out.exists())

    def test_explicit_missing_cuda_fails(self):
        with (
            patch.object(build, "find_nvcc", return_value=None),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(build.main(["--target", "cuda"]), 1)
            self.assertEqual(build.main(["--target", "all"]), 0)

    def test_missing_required_target_fails_before_build(self):
        replies = [
            type("R", (), {"stdout": "sm_80 sm_86"})(),
            type("R", (), {"stdout": "compute_120"})(),
        ]
        with (
            patch.object(build.subprocess, "run", side_effect=replies),
            self.assertRaisesRegex(RuntimeError, "lacks required targets"),
        ):
            build.check_targets(Path("nvcc"))


if __name__ == "__main__":
    unittest.main()
