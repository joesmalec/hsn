"""CUDA stack attribution regressions using only stdlib and synthetic frames."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "python/hsn/profiler/attribution.py"
SPEC = importlib.util.spec_from_file_location("hsn_attribution_rules", SOURCE)
assert SPEC and SPEC.loader
attribution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attribution)
ROOT = "/r/hsn/python/hsn/"


class AttributionTests(unittest.TestCase):
    def classify(self, frames):
        return attribution.classify_frames(frames, ROOT)

    def test_cuda_mlp_adapters_and_backward_keep_native_segment(self):
        for prefix in (ROOT, "hsn/"):
            outer = [(prefix + "model/cell.py", 350, "step_sparse")]
            native = outer + [(prefix + "model/units.py", 260, "forward_static")]
            for function in ("grouped_unit_mlp", "mlp_op", "mlp_backward", "_mlp_grad"):
                with self.subTest(prefix=prefix, function=function):
                    cuda = outer + [(prefix + "kernels/ops.py", 140, function)]
                    self.assertEqual(self.classify(cuda), "selected_cells")
                    self.assertEqual(self.classify(cuda), self.classify(native))
                    self.assertEqual(self.classify(cuda[-1:]), "selected_cells")

    def test_cuda_other_ops_remain_messages(self):
        for function in (
            "gather_concat",
            "gather_op",
            "gather_backward",
            "_gather_grad",
            "csr_messages",
            "messages_op",
            "messages_backward",
            "_messages_grad",
            "gated_delta_update",
            "update_op",
            "update_inplace",
            "update_backward",
            "_update_grad",
        ):
            with self.subTest(function=function):
                self.assertEqual(self.classify([("hsn/kernels/ops.py", 1, function)]), "messages")

    def test_shared_helper_and_external_ops_file_do_not_override_mlp(self):
        outer = [
            (ROOT + "model/cell.py", 350, "step_sparse"),
            (ROOT + "kernels/ops.py", 140, "mlp_op"),
        ]
        torch = ("/r/hsn/.venv/lib/python3.12/site-packages/torch/_ops.py", 1, "__call__")
        self.assertEqual(
            self.classify(outer + [(ROOT + "kernels/ops.py", 80, "_run"), torch]), "selected_cells"
        )
        self.assertIsNone(self.classify([torch]))
        lookalike = ("/r/hsn/.venv/lib/python3.12/site-packages/torch/ops.py", 1, "mlp_op")
        self.assertIsNone(self.classify([lookalike]))
        self.assertEqual(self.classify([(ROOT + "kernels/ops.py", 80, "_run")]), "other")

    def test_native_context_rules_are_preserved(self):
        fixtures = [
            ([("model/hsn.py", "_score")], "scorer"),
            ([("model/hsn.py", "_shared_path"), ("model/encoder.py", "forward")], "shared_path"),
            ([("model/router.py", "forward"), ("model/encoder.py", "forward")], "scorer"),
            ([("model/hsn.py", "_window_epilogue"), ("model/encoder.py", "forward")], "decode"),
            (
                [("model/cell.py", "step_sparse"), ("model/graph.py", "messages_selected")],
                "messages",
            ),
            (
                [("model/cell.py", "step_sparse"), ("kernels/native.py", "gather_concat")],
                "messages",
            ),
            ([("train/losses.py", "loss")], "loss"),
        ]
        for stack, expected in fixtures:
            with self.subTest(stack=stack):
                self.assertEqual(self.classify([(ROOT + p, 1, fn) for p, fn in stack]), expected)

    def test_path_normalization_and_package_boundary(self):
        frames = [
            ("hsn\\model\\cell.py", 1, "step_sparse"),
            ("hsn\\kernels\\ops.py", 2, "mlp_backward"),
        ]
        self.assertEqual(self.classify(frames), "selected_cells")
        self.assertIsNone(self.classify([("/r/hsn/python/hsn_rs/ops.py", 1, "mlp_op")]))
        self.assertIsNone(self.classify([]))


if __name__ == "__main__":
    unittest.main()
