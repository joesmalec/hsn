"""P2.10c, phase 2: the reverse sweep on static buffers (:mod:`hsn.train.state_sweep`) against
the reference sweep of phase 1 (:func:`hsn.train.state_gradient.state_gradient_backward`), on
the CPU in float64 with the glue steps in PyTorch: the same ``dL/dc``, ``dL/db`` and state
cotangents, one step at a time and over whole windows, with the operands' unit blocks written
and cleared as the kernels do it. (The kernels themselves against these steps:
``tests/kernels/test_p2_10c_sweep.py``, on the card.)"""

from __future__ import annotations

import pytest
import torch
from hsn.config import RunConfig
from hsn.train import state_gradient
from hsn.train.compile import fixed_fast_slots
from hsn.train.split import SplitWindow
from hsn.train.state_gradient import cell_constants
from hsn.train.state_sweep import ReverseSweep, out_adjacency, sweep_shape
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import _cfg, _model
from test_p2_10c_state_gradient import BASE, FAST, PINNED, READ, _case, _objective
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

F64 = torch.float64


def sweep_arguments(cfg: RunConfig, monkeypatch: pytest.MonkeyPatch) -> tuple[tuple, int, int]:
    """What a window's recurrence node hands its sweep (the arguments of
    ``state_gradient_backward``), and the model's fixed slots and fast-read cells."""
    model, (tokens, ctx, state, reset), mask = _case(cfg)
    given: list[tuple] = []
    sweep = state_gradient.state_gradient_backward

    def keeping(*args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        given.append(args)
        return sweep(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(state_gradient, "state_gradient_backward", keeping)
    window = SplitWindow(model, cfg, None, state_gradient=True)
    logits, _, info = window(tokens, ctx, state, reset)
    _objective(logits, info, mask).backward()
    monkeypatch.undo()
    n_read = cfg.model.fast_cells if cfg.model.fast_readout else 0
    return given[0], fixed_fast_slots(model), n_read


def buffered(args: tuple, n_fast: int, n_read: int) -> ReverseSweep:
    c, _, _, _, _, _, _, n_cells, *_, record, consts = args
    shape = sweep_shape(c, record, consts, n_cells, n_fast, n_read)
    return ReverseSweep(
        shape, consts, device=c.device, state_dtype=c.dtype, product_dtype=c.dtype, kernels=False
    )


def load(sweep: ReverseSweep, args: tuple, initial: tuple[Tensor, Tensor] | None = None) -> None:
    c, b, h_old, cells, units, valid, reset, _, *cots, record, consts = args
    sweep.load(c, b, h_old, cells, units, valid, reset, *cots, record, consts, initial=initial)


CASES = {
    "routed": (),
    "no_cand_norm": ("model.cand_norm=none",),
    "no_cell_norm": ("model.cell_norm=none",),
    "padded": ("model.budget=0.1",),
    "probes": ("model.age_max=2",),
    "fast_routed_units": FAST,
    "fast_read_routed_units": READ,
    "fast_fixed_units": PINNED,
    "fast_fixed_units_cut": (*PINNED, "model.budget=0.1"),
    "silu": ("model.activation=silu",),
}


def test_out_adjacency_lists_every_edge_by_its_source() -> None:
    model = _model(_cfg(*BASE, "model.random_edges=2"))
    graph = model.cell.graph
    out_tgt, out_edge = out_adjacency(graph._pad_src, graph._pad_mask)
    N, D = graph._pad_src.shape
    seen = set()
    for n in range(N):
        targets = [int(x) for x in out_tgt[n] if int(x) >= 0]
        assert targets == sorted(targets)  # by target (then by the target's slot)
        for j in range(out_tgt.shape[1]):
            target, flat = int(out_tgt[n, j]), int(out_edge[n, j])
            if target < 0:
                continue
            assert flat // D == target and int(graph._pad_src[target, flat % D]) == n
            assert bool(graph._pad_mask[target, flat % D])
            seen.add(flat)
    assert len(seen) == int(graph._pad_mask.sum()) == graph.n_edges
    assert int((out_tgt >= 0).sum(dim=1).max()) == out_tgt.shape[1]


@pytest.mark.parametrize("name", list(CASES))
def test_buffered_sweep_is_the_reference_sweep(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole window: the buffered sweep's ``dL/dc`` and ``dL/db`` are the reference's, and
    after every position its ``A`` and ``R`` are the reference's state cotangent after that
    position's reset."""
    cfg = _cfg(*BASE, *CASES[name])
    args, n_fast, n_read = sweep_arguments(cfg, monkeypatch)
    trace: dict[str, Tensor] = {}
    with torch.no_grad():
        want = state_gradient.state_gradient_backward(*args, trace=trace)
        sweep = buffered(args, n_fast, n_read)
        assert sweep.shape.n_fast == n_fast and sweep.shape.n_read == n_read
        load(sweep, args)
        reset = args[6]
        T = sweep.shape.T
        tol = {"rtol": 1e-11, "atol": 1e-13}
        for t in range(T - 1, -1, -1):
            sweep.step(t)
            keep = ~reset[:, t]
            torch.testing.assert_close(
                sweep.A, torch.where(keep[:, None, None], trace["A"][:, t], 0.0), **tol
            )
            torch.testing.assert_close(
                sweep.R, torch.where(keep[:, None], trace["R"][:, t], 0.0), **tol
            )
            # the operands are zero outside the blocks this position wrote
            blocks = sweep.shape.M
            op1 = sweep.op1.reshape(sweep.op1.shape[0], blocks, -1)
            units = sweep.units[:, t, n_fast:].reshape(-1)
            mine = torch.zeros(op1.shape[:2], dtype=torch.bool)
            mine[torch.arange(units.numel()), units] = True
            assert bool((op1[~mine] == 0).all())
            op2 = sweep.op2[:, : blocks * sweep.shape.H].reshape(sweep.op2.shape[0], blocks, -1)
            assert bool((op2[~mine] == 0).all())
        torch.testing.assert_close(sweep.dc, want[0], **tol)
        torch.testing.assert_close(sweep.db, want[1], **tol)
        # run() is the same loop, and a second load starts clean
        load(sweep, args)
        dc, db = sweep.run()
        torch.testing.assert_close(dc, want[0], **tol)
        torch.testing.assert_close(db, want[1], **tol)
    assert float(want[0].abs().max()) > 1e-6


@pytest.mark.parametrize("name", ["routed", "fast_fixed_units"])
def test_one_buffered_step_from_the_references_cotangent(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single position started from the reference's cotangent of the state after it (the
    per-step test the kernels are held to on the card)."""
    cfg = _cfg(*BASE, *CASES[name])
    args, n_fast, n_read = sweep_arguments(cfg, monkeypatch)
    trace: dict[str, Tensor] = {}
    with torch.no_grad():
        want = state_gradient.state_gradient_backward(*args, trace=trace)
        sweep = buffered(args, n_fast, n_read)
        reset = args[6]
        T = sweep.shape.T
        for t in range(T):
            if t + 1 < T:
                keep = ~reset[:, t + 1]
                after = (
                    torch.where(keep[:, None, None], trace["A"][:, t + 1], 0.0),
                    torch.where(keep[:, None], trace["R"][:, t + 1], 0.0),
                )
            else:
                after = (torch.zeros_like(trace["A"][:, 0]), torch.zeros_like(trace["R"][:, 0]))
            load(sweep, args, initial=after)
            if t + 1 < T:  # the operands as position t + 1 left them: its blocks, to be cleared
                sweep.step(t + 1)
                sweep.A.copy_(after[0])
                sweep.R.copy_(after[1])
            sweep.step(t)
            keep = ~reset[:, t]
            tol = {"rtol": 1e-11, "atol": 1e-13}
            torch.testing.assert_close(sweep.dc[:, t], want[0][:, t], **tol)
            torch.testing.assert_close(sweep.db[:, t], want[1][:, t], **tol)
            torch.testing.assert_close(
                sweep.A, torch.where(keep[:, None, None], trace["A"][:, t], 0.0), **tol
            )


def test_sweep_shape_and_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(*BASE, *PINNED)
    args, n_fast, n_read = sweep_arguments(cfg, monkeypatch)
    sweep = buffered(args, n_fast, n_read)
    s = sweep.shape
    assert (s.B, s.T, s.K, s.N) == (cfg.train.batch, cfg.train.unroll, 4, cfg.model.n_cells)
    assert (s.n_fast, s.n_read, s.in_dim) == (2, 2, 4 * cfg.model.d)
    assert sweep.kernels is False and sweep.routed == 2
    consts = args[-1]
    with pytest.raises(ValueError, match="routed slot"):
        ReverseSweep(
            s._replace(n_fast=s.K), consts, device=torch.device("cpu"), state_dtype=F64,
            product_dtype=F64, kernels=False,
        )  # fmt: skip
    with pytest.raises(ValueError, match="kernels"):
        ReverseSweep(
            s, consts, device=torch.device("cpu"), state_dtype=F64, product_dtype=F64, kernels=True
        )
    with pytest.raises(RuntimeError, match="CUDA"):
        sweep.capture()
    model = _model(cfg)
    with model.cell.window():
        assert cell_constants(model.cell, F64).pad_mask is not None
