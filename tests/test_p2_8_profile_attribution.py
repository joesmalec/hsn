"""P2.8: the frames ``scripts/profile_step.py`` attributes a kernel by.

The segment rules match file names (``graph.py`` → ``messages``), so they read the hsn package's
own frames only (``in_package``): torch has files of the same names (``autograd/graph.py``,
``fx/graph.py``), and a checkout named ``hsn`` has its ``.venv``, torch included, under a
directory named ``hsn``. Frames come in two forms: absolute paths (inductor's provenance stack
traces at L1, the live frames of the MPS classifier) and paths relative to their ``sys.path``
entry (``torch.profiler(with_stack=True)`` at L0).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[1]
#: a checkout named hsn, its .venv inside
CHECKOUT = "/r/hsn"
VENV_TORCH = CHECKOUT + "/.venv/lib/python3.12/site-packages/torch/"

Frame = tuple[str, int, str]


@pytest.fixture(scope="module")
def ps() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "profile_step", REPO / "scripts" / "profile_step.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile_step"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def root(ps: ModuleType, monkeypatch: pytest.MonkeyPatch) -> str:
    """The package root of :data:`CHECKOUT`."""
    package = CHECKOUT + "/python/hsn/"
    monkeypatch.setattr(ps, "PACKAGE_ROOT", package)
    return package


@pytest.mark.cuda
def test_p2_10_compiled_mlp_driver_launches_are_selected_cells(ps: ModuleType) -> None:
    from hsn.kernels import cuda_ops, testvectors

    if os.environ.get("HSN_REQUIRE_CUDA") != "1":
        pytest.skip("custom CUDA artifacts and the idle guard are required")
    cuda_ops.prepare_cuda(0)
    vector = testvectors.load_vector("grouped_unit_mlp", "tiny")
    names = ("v", "unit_idx", "W1", "b1", "W2", "b2")
    inputs = tuple(
        vector.tensors[name].to("cuda").requires_grad_(name != "unit_idx") for name in names
    )
    parameters = tuple(t for t in inputs if t.requires_grad)

    def forward():
        return cuda_ops.mlp_op(*inputs)[0]

    compiled = torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False})
    for _ in range(2):
        out = compiled()
        torch.autograd.grad(out, parameters, torch.ones_like(out))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("STEP::mlp"):
            out = compiled()
            torch.autograd.grad(out, parameters, torch.ones_like(out))
            torch.cuda.synchronize()
    events = list(prof.events())
    custom = [
        e for e in events if e.device_type == ps.DeviceType.CUDA and "hsn_cuda_mlp_" in e.name
    ]
    assert len(custom) == 7  # hidden, out, dz, dv, route grouping, dW1 and dW2
    attributed = ps.attribute(events, {})
    assert attributed["kernels"]["selected_cells"] >= len(custom)
    assert attributed["device_us"]["selected_cells"] >= sum(
        e.time_range.end - e.time_range.start for e in custom
    )
    assert not any("hsn_cuda_mlp_" in key for key in attributed["unattributed_top"])


@pytest.mark.cuda
@pytest.mark.parametrize("p_dense", (0.0, 1.0))
def test_p2_8_attribution_window_disables_graphs_for_all_buckets(
    ps: ModuleType, monkeypatch: pytest.MonkeyPatch, p_dense: float
) -> None:
    from cards import CUDA_PROFILE
    from hsn.train import compile as compile_mod
    from test_p2_8_compile import (
        MIN_MARGIN,
        TOL,
        _case_inputs,
        _cfg,
        _ctx,
        _grad_diff,
        _grads,
        _margin,
        _max_diff,
        _model,
        _run,
    )

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)
    ctx = _ctx(CUDA_PROFILE)
    model = _model(
        _cfg("model.routing=topk", "train.gate_mode=ste", f"train.p_dense={p_dense}"),
        training=True,
        scale=0.3,
    ).to(ctx.device)
    original = compile_mod.CompiledWindow(model, ctx, "reduce-overhead", "inductor")
    plain = ps.ungraphed_window(original)
    assert not plain.cuda_graphs
    # Use the established routed parity fixture: distinct cell states and separated
    # top-K scores keep this cache/lifetime regression independent of routing ties.
    inputs = _case_inputs(model, True)
    expected = _run(lambda: model(*inputs))
    assert _margin(expected[2], model.cfg.k_select) > MIN_MARGIN
    expected_grad = _grads(model, expected)
    tol = TOL[("cuda", "fp32")]
    for _ in range(2):
        actual = _run(lambda: plain(*inputs))
        difference = _max_diff(actual, expected)
        assert difference["exact"] == 0 and difference["values"] <= tol["values"]
        assert _grad_diff(_grads(model, actual), expected_grad) <= tol["grads"]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _grads(model, plain(*inputs))
        torch.cuda.synchronize()
    assert not any("GraphLaunch" in event.name for event in prof.events())
    assert (inputs[0].shape[0] if p_dense else 0) in plain._bucket_blocks


def test_p2_8_torch_files_named_like_hsn_files_do_not_classify(ps: ModuleType, root: str) -> None:
    """torch's ``autograd/graph.py`` and ``fx/graph.py`` under the checkout's ``.venv`` are not HSN
    frames (any path through a directory named ``hsn`` was, and classified as ``messages``); the
    package's own frames decide, innermost first, in both frame forms."""
    units = (root + "model/units.py", 183, "forward_selected")
    torch_frames: list[Frame] = [
        (VENV_TORCH + "autograd/graph.py", 824, "_engine_run_backward"),
        (VENV_TORCH + "fx/graph.py", 1, "python_code"),
    ]
    for torch_frame in torch_frames:
        assert not ps.in_package(torch_frame[0])
        assert ps.classify_frames([torch_frame]) is None
        assert ps.classify_frames([units, torch_frame]) == "selected_cells"
        assert ps.innermost_label([units, torch_frame]) == "units.py:183 forward_selected"
        assert ps.innermost_label([torch_frame]) == "(no hsn frame)"
        assert ps.split([[torch_frame]]) == {"other": 1.0}
        assert ps.split([[units, torch_frame], [torch_frame]]) == {
            "selected_cells": 0.5,
            "other": 0.5,
        }
    # the profiler's form: each path under its sys.path entry
    profiled = [("hsn/model/units.py", 183, "forward_selected"), ("torch/fx/graph.py", 1, "f")]
    assert ps.classify_frames(profiled) == "selected_cells"
    assert ps.classify_frames(profiled[1:]) is None
    # the checkout's other files, and a package whose name starts like hsn's
    for path in (
        CHECKOUT + "/scripts/profile_step.py",
        CHECKOUT + "/tests/test_p2_8_compile.py",
        CHECKOUT + "/python/hsn_rs/__init__.py",
    ):
        assert not ps.in_package(path)
    assert ps.in_package(root + "model/graph.py")
    assert ps.classify_frames([(root + "model/graph.py", 1, "messages_selected")]) == "messages"


def test_p2_8_real_frames_are_package_frames(ps: ModuleType) -> None:
    """Both forms name the package as imported: the absolute file of its modules (what inductor's
    stack traces carry) is in the package, and an op issued by ``MLP2.forward`` under
    ``torch.profiler(with_stack=True)`` classifies by ``encoder.py`` (relative), not by torch's
    ``nn/modules/linear.py`` frame inside it."""
    from hsn.model import encoder, units

    assert ps.in_package(encoder.__file__) and ps.in_package(units.__file__)
    mlp = encoder.MLP2(4, 8, 4, "gelu", init="fan_in")
    x = torch.randn(2, 4)
    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        mlp(x)
    host = next(e for e in prof.events() if e.name == "aten::addmm")
    frames = ps.frames_of_chain(host)
    assert any(path.endswith("nn/modules/linear.py") for path, _, _ in frames)
    assert ps.classify_frames(frames) == "shared_path"
    label = ps.innermost_label(frames)
    assert label.startswith("encoder.py:") and label.endswith(" forward")


@pytest.mark.parametrize("key_address", [False, True])
def test_p1_17_serving_profile_carries_previous_byte(
    ps: ModuleType, monkeypatch: pytest.MonkeyPatch, key_address: bool
) -> None:
    """The measured stream continues the warm-up's last byte; legacy calls stay unchanged."""
    calls: list[tuple[torch.Tensor, dict[str, torch.Tensor | None]]] = []

    class Replica:
        def eval(self) -> Replica:
            return self

        def init_state(self, batch: int) -> None:
            assert batch == 2

    def step(
        tokens: torch.Tensor, context: torch.Tensor, state: None, **kwargs: torch.Tensor | None
    ) -> torch.Tensor:
        calls.append((tokens.clone(), kwargs))
        return torch.zeros(2, 256)

    cfg = SimpleNamespace(
        model=SimpleNamespace(key_address=key_address, vocab_size=256, n_context=8)
    )
    ctx = SimpleNamespace(device=torch.device("cpu"), autocast_dtype=None)
    monkeypatch.setattr(ps.compile_mod, "compile_step", lambda *args: step)
    measured = ps.serving(cfg, ctx, Replica(), "L0", 2, 3)
    assert measured["tokens"] == 3
    assert len(calls) == 11  # eight warm-up calls, then three measured calls
    for i, (_tokens, kwargs) in enumerate(calls):
        if not key_address:
            assert kwargs == {}
        elif i == 0:
            assert kwargs == {"prev_tokens": None}
        else:
            previous = kwargs["prev_tokens"]
            assert isinstance(previous, torch.Tensor)
            assert torch.equal(previous, calls[i - 1][0])
