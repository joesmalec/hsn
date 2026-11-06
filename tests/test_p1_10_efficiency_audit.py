"""P1.10 (HSN efficiency audit): the launch-attribution logic of ``scripts/eff_profile_window.py``
and the documented cell symmetry of the dense reference.

``test_p1_10_attribution_sums_to_total`` drives the script's attribution on a synthetic event list
(no GPU): every device launch lands in exactly one row, forward launches by their innermost range,
backward launches through their autograd node's ``sequence_nr`` and the engine's accumulation rows.

``test_p1_10_dense_cells_differ_from_zero_state`` records finding 1 of the audit
(``evidence/EFF/AUDIT.md``, local): from the zero state every cell of the dense round receives the
same input, so the 32 cells of ``ref_a1`` stay bitwise identical and the edge logits never receive
a gradient. The test asserts the behaviour a cell graph needs (cells that differ, edge weights that
learn) and is a strict ``xfail`` until a symmetry-breaking amendment lands; it then starts passing
and the marker must go.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from hsn.config import load_config
from hsn.model.hsn import HSN
from torch.autograd import DeviceType  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _Range:
    start: float
    end: float


@dataclass(eq=False)
class _Event:
    """The attributes of ``torch.autograd.profiler_util.FunctionEvent`` the attribution reads."""

    id: int
    name: str
    start: float
    end: float
    parent: _Event | None = None
    device: Any = DeviceType.CPU
    activity_type: str | None = None
    linked_correlation_id: int = 0
    external_id: int = 1
    sequence_nr: int = -1
    is_user_annotation: bool = False
    self_cpu_time_total: float = 1.0
    time_range: _Range = field(init=False)

    def __post_init__(self) -> None:
        self.time_range = _Range(self.start, self.end)

    @property
    def cpu_parent(self) -> _Event | None:
        return self.parent

    @property
    def device_type(self) -> Any:
        return self.device


def _kernel(eid: int, host: _Event | None, t: float, linked: int | None = None) -> _Event:
    return _Event(
        eid,
        "kernel",
        t,
        t + 1,
        device=DeviceType.CUDA,
        activity_type="kernel",
        linked_correlation_id=linked if linked is not None else (host.id if host else 0),
        external_id=0,
    )


def test_p1_10_attribution_sums_to_total() -> None:
    script = _load_script("eff_profile_window")
    step = _Event(1, "STEP::1", 0, 1000, is_user_annotation=True)
    enc = _Event(2, "MOD::encoder", 10, 20, parent=step, is_user_annotation=True)
    addmm = _Event(3, "aten::addmm", 11, 12, parent=enc, sequence_nr=7)
    reset = _Event(4, "FN::state.reset", 30, 40, parent=step, is_user_annotation=True)
    fill = _Event(5, "aten::masked_fill", 31, 32, parent=reset, sequence_nr=8)
    ev_mm = _Event(6, "autograd::engine::evaluate_function: AddmmBackward0", 500, 520)
    node_mm = _Event(7, "AddmmBackward0", 501, 510, parent=ev_mm, sequence_nr=7)
    mm = _Event(8, "aten::mm", 502, 503, parent=node_mm)
    buffer_add = _Event(9, "aten::add_", 511, 512, parent=ev_mm)
    ev_acc = _Event(
        10, "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad", 600, 610
    )
    node_acc = _Event(11, "torch::autograd::AccumulateGrad", 601, 609, parent=ev_acc)
    leaf_add = _Event(12, "aten::add_", 602, 603, parent=node_acc)
    stray = _Event(13, "aten::zero_", 700, 701, parent=step)
    outside = _Event(14, "aten::add", 2000, 2001)
    cpu = [
        step,
        enc,
        addmm,
        reset,
        fill,
        ev_mm,
        node_mm,
        mm,
        buffer_add,
        ev_acc,
        node_acc,
        leaf_add,
    ]
    cpu += [stray, outside]
    kernels = [
        _kernel(101, addmm, 12),
        _kernel(102, fill, 32),
        _kernel(103, fill, 33),
        _kernel(104, mm, 503),
        _kernel(105, buffer_add, 512),
        _kernel(106, leaf_add, 603),
        _kernel(107, stray, 702),
        _kernel(108, None, 800, linked=999),  # no frontend op: attributed by its own time
        _kernel(109, outside, 2002),  # outside every step range
    ]
    report = script.attribute(cpu + kernels, T=1)
    assert report["unattributed_launches"] == 1
    (only,) = report["steps"]
    rows = {r["row"]: r["launches"] for r in only["rows"]}
    assert rows == {
        "MOD::encoder": 1,
        "FN::state.reset": 2,
        "bwd:MOD::encoder": 1,
        "bwd:grad_accumulation(buffer)": 1,
        "bwd:grad_accumulation(leaf)": 1,
        "(main thread, unlabelled)": 1,
        "(no host op)": 1,
    }
    assert only["launches_total"] == sum(rows.values()) == len(kernels) - 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "P1.10 finding 1: from the zero state the dense round gives every cell the same input, so "
        "the cells stay identical and the edge logits get no gradient; a symmetry-breaking "
        "amendment (proposed A3, cell identity) is pending"
    ),
)
def test_p1_10_dense_cells_differ_from_zero_state() -> None:
    torch.manual_seed(0)
    run = load_config(REPO / "configs" / "ref_a1.yaml")
    model = HSN.from_config(run).float()
    B, T = 2, 6
    tokens = torch.randint(0, run.model.vocab_size, (B, T))
    ctx = torch.randint(0, run.model.n_context, (B, T))
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    logits, final, _ = model(tokens, ctx, model.init_state(B), reset)
    logits.logsumexp(-1).mean().backward()  # type: ignore[no-untyped-call]
    spread = (final.h - final.h[:, :1]).abs().amax()
    edge_grad = model.cell.graph.edge_logit.grad
    assert edge_grad is not None
    assert float(spread) > 0.0, "every cell holds the same state"
    assert float(edge_grad.abs().max()) > 0.0, "the edge logits receive no gradient"
