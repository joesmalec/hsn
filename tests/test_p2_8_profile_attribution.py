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
import sys
from pathlib import Path
from types import ModuleType

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
