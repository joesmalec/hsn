"""P1.10 (the loop fixes of the efficiency audit): ``HSN.forward`` against the per-position
reference, the opt-in per-position capture, and the per-window caches.

P1.10a moves the work that does not depend on the carried state out of the position loop of
:meth:`hsn.model.hsn.HSN.forward` (embeddings, ``E``, the prediction target, the decoder, the
position-independent scorer features, the edge weights, the statistics). :meth:`HSN.step` stays the
per-position reference: the window must equal stepping position by position. The batched matmuls
round differently from the per-position ones (the GEMM row count changes), so the comparison is at
a tolerance in fp64 rather than bit for bit; comparisons between runs of the same shapes (capture
on and off) stay bit-exact.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything
from hsn.model.cell import PDGCell
from hsn.model.graph import CellGraph
from hsn.model.hsn import HSN, StepMode
from hsn.model.router import Selection
from hsn.model.state import CellGraphState
from hsn.train.tbptt import TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"
#: ref_a1 shrunk for the routed modes (the P2.2 test recipe): N 6, M 3, K 2.
SMALL = (
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=6",
    "model.in_degree=2",
    "model.n_units=3",
    "model.k_select=2",
    "model.k_min=1",
    "model.hidden=16",
    "model.scorer_hidden=8",
    "train.unroll=16",
)
#: The forward-vs-step tolerance in fp64 (the chunk's), for the window's values.
ATOL = 1e-12
#: fp32 quantities of a fp64 model (the router's cost accounting is fp32 by design).
ATOL_FP32 = 1e-6
#: Parameter gradients: absolute and relative, since perturbed weights give gradients of O(10).
GRAD_TOL = 1e-10
B, T = 3, 6
RNG_SEED = 1234
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
TRAINER_UNROLL = 32


@dataclass(frozen=True)
class Case:
    name: str
    path: Path
    overrides: tuple[str, ...]
    training: bool = False
    #: Weight perturbation; the full-width ref_a1 takes P2.5's 0.1 (unit outputs of order 3, the
    #: training regime): at 0.3 its logits reach ~700, where 1e-12 absolute is 1e-15 relative.
    scale: float = 0.3


CASES = (
    # dense routing
    Case("dense_tiny", TINY, ()),
    Case("dense_tiny_disabled", TINY, ("model.disabled_cells=[1]",)),
    Case("dense_tiny_exclude", TINY, ("model.scorer_features.exclude=[eps, pos, one]",)),
    Case("dense_tiny_sigmoid_edges", TINY, ("model.edge_norm=sigmoid",)),
    Case("dense_tiny_edgeless", TINY, ("model.in_degree=0",)),
    Case("dense_tiny_mlp_scorer", TINY, ("model.scorer=mlp",)),
    Case("dense_tiny_ema_target", TINY, ("model.predictor_target=ema_encoder",), training=True),
    Case("dense_ref_a1", REF_A1, (), scale=0.1),
    # masked round: the soft warm-up of a routed model in training
    Case("masked_soft_warmup", REF_A1, (*SMALL, "model.routing=topk"), training=True),
    Case(
        "masked_soft_disabled",
        REF_A1,
        (*SMALL, "model.routing=topk", "model.disabled_cells=[2]"),
        training=True,
    ),
    # sparse round
    Case("sparse_topk_eval", REF_A1, (*SMALL, "model.routing=topk")),
    Case(
        "sparse_ste_explore_refresh",
        REF_A1,
        (
            *SMALL,
            "model.routing=topk",
            "train.gate_mode=ste",
            "train.epsilon_start=0.5",
            "train.epsilon_end=0.5",
            "model.age_max=2",
        ),
        training=True,
    ),
    # p_dense 0.3 at RNG_SEED draws rows 0 and 2 dense, row 1 sparse
    Case(
        "sparse_ste_dense_fallback",
        REF_A1,
        (*SMALL, "model.routing=topk", "train.gate_mode=ste", "train.p_dense=0.3"),
        training=True,
    ),
    Case(
        "sparse_reinforce",
        REF_A1,
        (*SMALL, "model.routing=topk", "train.gate_mode=reinforce"),
        training=True,
    ),
    Case("sparse_greedy_budget", REF_A1, (*SMALL, "model.routing=greedy_budget")),
    Case(
        "sparse_random",
        REF_A1,
        (*SMALL, "model.routing=random", "train.gate_mode=ste"),
        training=True,
    ),
    Case(
        "sparse_fixed",
        REF_A1,
        (*SMALL, "model.routing=fixed", "model.fixed_route=[1, 4]", "train.gate_mode=ste"),
        training=True,
    ),
    Case(
        "sparse_topk_disabled",
        REF_A1,
        (*SMALL, "model.routing=topk", "model.disabled_cells=[2]"),
    ),
)
CASE_IDS = [case.name for case in CASES]


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _build(case: Case, seed: int = 0) -> HSN:
    """The case's model in fp64 on the CPU, weights perturbed (non-zero biases, non-uniform edge
    weights) so that the window's outputs depend on everything they read."""
    cfg = load_config(case.path, overrides=list(case.overrides))
    seed_everything(seed)
    model = HSN.from_config(cfg).double()
    generator = torch.Generator(device="cpu").manual_seed(seed + 7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator, dtype=torch.float64) * case.scale)
    model.train(case.training)
    return model


def _inputs(model: HSN, seed: int = 3) -> tuple[Tensor, Tensor, CellGraphState, Tensor]:
    """``tokens``, ``ctx`` ``[B, T]``, a random start state (non-zero ``h``, ``age`` and ``pos``)
    and a reset mask with a reset at ``t = 0`` (row 0), mid-window (row 1) and none (row 2)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m = model.cfg
    tokens = torch.randint(0, m.vocab_size, (B, T), generator=generator)
    ctx = torch.randint(0, m.n_context, (B, T), generator=generator)
    h = torch.rand(B, m.n_cells, m.d, generator=generator, dtype=torch.float64) * 1.6 - 0.8
    state = model.init_state(B).with_dense(h.to(model.dtype))
    for _ in range(3):
        state = state.tick(torch.rand(B, m.n_cells, generator=generator) < 0.5)
    assert bool((state.age > 0).any()) and bool((state.pos > 0).all())
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 0] = True
    reset[1, 3] = True
    return tokens, ctx, state, reset


def _stepwise(
    model: HSN, tokens: Tensor, ctx: Tensor, state: CellGraphState, reset: Tensor
) -> tuple[Tensor, CellGraphState, dict[str, Tensor]]:
    """The pre-P1.10a ``HSN.forward`` (``cdc43c8``), the per-position reference: reset, then
    :meth:`HSN.step` at every position, then the stacks."""
    B_, T_ = tokens.shape
    dense_rows: Tensor | None = None
    if model.cfg.routing != "dense" and model.training:
        p_dense = model.schedule.p_dense(model.train_step)
        if p_dense > 0.0:
            drawn = torch.rand(B_) < p_dense
            if bool(drawn.any()):
                dense_rows = drawn.to(state.device)
    logits: list[Tensor] = []
    stacks: dict[str, list[Tensor]] = {"z": [], "z_hat": [], "z_target": [], "eps": [], "u": []}
    stats: dict[str, list[Tensor]] = {"state_abs_max": [], "eps_abs_mean": [], "alpha_entropy": []}
    costs: list[Tensor] = []
    rounds: list[int] = []
    selections: list[Selection] = []
    with model.cell.graph.step():
        for t in range(T_):
            state = state.reset(reset[:, t])
            out = model.step(tokens[:, t], ctx[:, t], state, dense_rows=dense_rows)
            state = out.state
            logits.append(out.logits)
            for name in stacks:
                stacks[name].append(getattr(out, name))
            for name in stats:
                stats[name].append(out.aux[name])
            costs.append(out.cost)
            rounds.append(out.n_rounds)
            if out.selection is not None:
                selections.append(out.selection)
    info: dict[str, Tensor] = {name: torch.stack(values, dim=1) for name, values in stacks.items()}
    info.update({name: torch.stack(values) for name, values in stats.items()})
    info["cost"] = torch.stack(costs).to(model.dtype)
    info["n_rounds"] = torch.tensor(rounds, dtype=torch.int64, device=state.device)
    if selections:
        assert len(selections) == T_
        info["cell_idx"] = torch.stack([sel.cell_idx for sel in selections], dim=1)
        info["unit_idx"] = torch.stack([sel.unit_idx for sel in selections], dim=1)
        info["valid"] = torch.stack([sel.valid for sel in selections], dim=1)
        info["unit_logits"] = torch.stack([sel.logits for sel in selections], dim=1)
        info["cost_per_row"] = torch.stack([sel.cost.detach() for sel in selections], dim=1)
        info["expected_cost"] = torch.stack([sel.expected_cost.mean() for sel in selections])
        info["route_log_prob"] = torch.stack([sel.log_prob for sel in selections], dim=1).sum(1)
        info["explored"] = torch.stack(
            [sel.explored.any(dim=1).float().mean() for sel in selections]
        )
        info["refreshed"] = torch.stack(
            [sel.refreshed.float().sum(dim=1).mean() for sel in selections]
        )
        info["dense_fallback"] = torch.stack(
            [sel.dense_fallback.float().mean() for sel in selections]
        )
    return torch.stack(logits, dim=1), state, info


Window = tuple[Tensor, CellGraphState, dict[str, Tensor]]


def _run(fn: Callable[[], Window]) -> Window:
    """``fn()`` with the global RNG (the dense-fallback draw, exploration, random routing,
    REINFORCE samples) at a fixed state, so that two runs consume the same draws."""
    torch.manual_seed(RNG_SEED)
    return fn()


def _close(actual: Tensor, expected: Tensor, what: str, atol: float = ATOL) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.0, msg=lambda m: f"{what}: {m}")


def _assert_matches(got: Window, ref: Window, *, exact: bool = False) -> None:
    """Logits, the final state and every ``info`` key of ``got`` equal ``ref``: floats to the
    tolerance (bit for bit with ``exact``), integers and booleans exactly, the same key set."""
    (logits, final, info), (logits_r, final_r, info_r) = got, ref

    def same(a: Tensor, b: Tensor, what: str) -> None:
        if exact or not a.is_floating_point():
            assert a.dtype == b.dtype and torch.equal(a, b), what
        else:
            assert a.dtype == b.dtype, (what, a.dtype, b.dtype)
            _close(a, b, what, ATOL if a.dtype == torch.float64 else ATOL_FP32)

    same(logits, logits_r, "logits")
    for name in ("h", "q", "age", "pos"):
        same(getattr(final, name), getattr(final_r, name), f"state.{name}")
    assert set(info_r) <= set(info), sorted(set(info_r) - set(info))
    for key, value in info_r.items():
        if key == "cost" and "cost_per_row" in info_r:  # fp32 per-row costs, averaged in fp32
            _close(info[key], value, key, ATOL_FP32)
        else:
            same(info[key], value, key)


def _projection_loss(window: Window, seed: int = 11) -> Tensor:
    """A fixed random linear functional of every differentiable output of a window."""
    logits, final, info = window
    generator = torch.Generator(device="cpu").manual_seed(seed)
    terms = [logits, final.h, final.q, info["eps"], info["z_hat"], info["u"], info["z"]]
    if "unit_logits" in info:
        terms.append(info["unit_logits"])
    total = torch.zeros((), dtype=torch.float64)
    for x in terms:
        w = torch.randn(x.shape, generator=generator, dtype=torch.float64).to(x.dtype)
        total = total + (x * w).sum().double()
    if "route_log_prob" in info:
        total = total + info["route_log_prob"].sum().double()
    return total


def _grads(model: HSN, window: Window) -> list[Tensor | None]:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(_projection_loss(window), params, allow_unused=True)
    return list(grads)


# --------------------------------------------------------------------------------------------------
# P1.10a: forward equals stepping
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_p1_10_forward_matches_stepwise_reference(case: Case) -> None:
    """``HSN.forward`` over a window equals the pre-P1.10a loop of ``HSN.step`` (reset, step,
    stack) in every routing mode — dense (with the disabled-cell hook, a scorer ablation, sigmoid
    edges, an edgeless graph, the MLP scorer, the EMA target), the masked round (soft warm-up) and
    the sparse round (top-K in evaluation; STE with exploration, refresh probes and the dense
    fallback; REINFORCE; greedy budget; random; fixed; disabled cells) — from a random state with
    resets at the window start and mid-window: logits, the final state, every ``info`` key, and the
    parameter gradients of a random projection of the outputs. fp64 on the CPU at ``atol`` 1e-12:
    the window batches E and D over ``[B, T]``, which rounds differently from the per-position
    matmuls."""
    model = _build(case)
    tokens, ctx, state, reset = _inputs(model)
    ref = _run(lambda: _stepwise(model, tokens, ctx, state, reset))
    got = _run(lambda: model(tokens, ctx, state, reset))
    _assert_matches(got, ref)
    assert set(got[2]) == set(ref[2])  # no extra keys without capture
    for p, g, g_r in zip(model.parameters(), _grads(model, got), _grads(model, ref), strict=True):
        if g_r is None:
            assert g is None or not bool(g.any())
            continue
        assert g is not None
        torch.testing.assert_close(g, g_r, atol=GRAD_TOL, rtol=GRAD_TOL, msg=str(p.shape))


# --------------------------------------------------------------------------------------------------
# P1.10a: per-position capture
# --------------------------------------------------------------------------------------------------


@contextmanager
def _recording_rounds(model: HSN) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Wrap the instance's ``step_dense`` / ``step_sparse`` (as the diagnostics do) and record
    each call's kind and ``info``; also arm the router's tap to record the selections."""
    calls: list[tuple[str, dict[str, Any]]] = []
    cell = model.cell
    dense, sparse = cell.step_dense, cell.step_sparse

    def step_dense(*args: Any, **kwargs: Any) -> Any:
        h_new, info = dense(*args, **kwargs)
        calls.append(("dense", info))
        return h_new, info

    def step_sparse(*args: Any, **kwargs: Any) -> Any:
        h_new, info = sparse(*args, **kwargs)
        calls.append(("sparse", info))
        return h_new, info

    cell.step_dense, cell.step_sparse = step_dense, step_sparse  # type: ignore[method-assign]
    try:
        yield calls
    finally:
        del cell.step_dense, cell.step_sparse


def _expected_capture(
    model: HSN, calls: list[tuple[str, dict[str, Any]]], selections: list[Selection]
) -> tuple[Tensor, Tensor]:
    """The gate of the written cells (NaN elsewhere) and the written mask, per position, rebuilt
    from the recorded rounds: every enabled cell of a dense round, the ``cell_mask > 0`` cells of
    a masked round, the valid slots of a sparse round, every enabled cell of a dense-fallback
    row."""
    m = model.cfg
    N, d = m.n_cells, m.d
    enabled = torch.ones(N, dtype=torch.bool)
    enabled[list(m.disabled_cells)] = False
    routed = m.routing != "dense"
    soft = model.training and model._gate() == "soft"
    gates: list[Tensor] = []
    written: list[Tensor] = []
    index = 0
    for t in range(T):
        kind, info = calls[index]
        index += 1
        gate = torch.full((B, N, d), float("nan"), dtype=torch.float64)
        mask = torch.zeros(B, N, dtype=torch.bool)
        if kind == "dense" and not routed:
            mask[:] = enabled
            gate[mask] = info["b"].detach()[mask]
        elif kind == "dense":  # the masked round
            assert soft
            sel = selections[t]
            mask = (sel.cell_mask.detach() > 0) & enabled
            gate[mask] = info["b"].detach()[mask]
        else:
            sel = selections[t]
            valid = info["valid"]
            for b in range(B):
                for k in range(sel.k):
                    if bool(valid[b, k]):
                        cell = int(sel.cell_idx[b, k])
                        mask[b, cell] = True
                        gate[b, cell] = info["b"][b, k].detach()
            if index < len(calls) and calls[index][0] == "dense":  # the dense-fallback rows
                _, dense_info = calls[index]
                index += 1
                rows = sel.dense_fallback
                assert bool(rows.any())
                dense_gate = dense_info["b"].detach().masked_fill(~enabled[None, :, None], math.nan)
                gate[rows] = dense_gate[rows]
                mask[rows] = enabled
        gates.append(gate)
        written.append(mask)
    assert index == len(calls)
    return torch.stack(gates, dim=1), torch.stack(written, dim=1)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_p1_10_capture_matches_per_position_states(case: Case) -> None:
    """``forward(..., capture=True)`` changes nothing — logits, the final state and every other
    ``info`` key bit for bit as with ``capture=False`` — and adds the per-position cell states
    (equal to the states of the per-position reference), the gate of every written cell (the
    ``b`` the round's ``step_dense`` / ``step_sparse`` returned; NaN where a cell was not written)
    and the written mask, in every routing mode."""
    model = _build(case)
    tokens, ctx, state, reset = _inputs(model)
    plain = _run(lambda: model(tokens, ctx, state, reset))
    with _recording_rounds(model) as calls, model.router.intercept() as tap:
        captured = _run(lambda: model(tokens, ctx, state, reset, capture=True))
    _assert_matches(captured, plain, exact=True)
    info = captured[2]
    assert set(info) == set(plain[2]) | {"cell_states", "cell_gates", "cell_written"}
    N, d = model.cfg.n_cells, model.cfg.d
    states, gates, written = info["cell_states"], info["cell_gates"], info["cell_written"]
    assert states.shape == gates.shape == (B, T, N, d) and written.shape == (B, T, N)
    assert states.dtype == gates.dtype == model.dtype and written.dtype == torch.bool
    assert not states.requires_grad and not gates.requires_grad
    assert torch.equal(states[:, -1], captured[1].h)

    # the states equal the per-position reference's
    reference_states: list[Tensor] = []
    torch.manual_seed(RNG_SEED)
    current = state
    with torch.no_grad():
        dense_rows = None
        if model.cfg.routing != "dense" and model.training and model.schedule.p_dense(0) > 0:
            drawn = torch.rand(B) < model.schedule.p_dense(model.train_step)
            dense_rows = drawn if bool(drawn.any()) else None
        for t in range(T):
            current = current.reset(reset[:, t])
            current = model.step(tokens[:, t], ctx[:, t], current, dense_rows=dense_rows).state
            reference_states.append(current.h)
    _close(states, torch.stack(reference_states, dim=1), "cell_states")

    # the gates and the written mask follow the recorded rounds
    selections = tap.selections
    assert len(selections) == (T if model.cfg.routing != "dense" else 0)
    expected_gates, expected_written = _expected_capture(model, calls, selections)
    assert torch.equal(written, expected_written)
    assert torch.equal(gates.isnan(), ~written[..., None].expand_as(gates))
    assert torch.equal(gates[written], expected_gates[written])


def test_p1_10_capture_masked_and_sparse_agree_on_written_cells() -> None:
    """The masked round and the sparse round of the same hard selection capture the same written
    cells with the same gates (the masked round computes every cell's gate; only the written ones
    are reported)."""
    model = _build(Case("topk", REF_A1, (*SMALL, "model.routing=topk", "train.gate_mode=ste")))
    model.train()
    tokens, ctx, state, _ = _inputs(model)
    with torch.no_grad():
        q_prev = state.q
        z = model.encoder(model.byte_emb(tokens[:, 0]))
        u = model.shared(torch.cat([z, q_prev, model.ctx_emb(ctx[:, 0])], dim=-1))
        eps = z - model.predictor(q_prev)
        feats = model.scorer.features(u, eps, state, ctx[:, 0], budget_frac=1.0, round_idx=0)
        modes: tuple[StepMode, StepMode] = ("masked", "sparse")
        results = []
        for mode in modes:
            torch.manual_seed(RNG_SEED)
            unit_logits, alpha, selection = model._score(feats, state, mode, None)
            results.append(
                model._update(u, eps, state, mode, alpha, selection, unit_logits, None, None, True)
            )
    (masked, sparse) = results
    assert masked.written is not None and sparse.written is not None
    assert masked.gate is not None and sparse.gate is not None
    assert bool(sparse.written.any()) and not bool(sparse.written.all())
    assert torch.equal(masked.written, sparse.written)
    _close(masked.gate[masked.written], sparse.gate[sparse.written], "gate")
    assert bool(masked.gate[~masked.written].isnan().all())
    _close(masked.state.h, sparse.state.h, "h")


# --------------------------------------------------------------------------------------------------
# P1.10a: the per-window edge weights
# --------------------------------------------------------------------------------------------------


@contextmanager
def _counting_weight_builds(graph: CellGraph) -> Iterator[list[int]]:
    count = [0]
    build = graph._build_padded_weights

    def counted() -> Tensor:
        count[0] += 1
        return build()

    graph._build_padded_weights = counted  # type: ignore[method-assign]
    try:
        yield count
    finally:
        del graph._build_padded_weights


@pytest.mark.parametrize("routing", ["dense", "topk"])
def test_p1_10_edge_weights_cached_per_window(routing: str) -> None:
    """Inside ``forward`` the edge weights are built once for the window (dense and sparse message
    paths), never kept after it; outside a window every message call builds them; the cache
    refuses an edge logit changed in place, a grad-mode switch, and nested windows."""
    case = Case(routing, REF_A1, (*SMALL, f"model.routing={routing}"))
    model = _build(case)
    tokens, ctx, state, reset = _inputs(model)
    graph = model.cell.graph
    with _counting_weight_builds(graph) as built:
        model(tokens, ctx, state, reset)
        assert built[0] == 1
        assert graph._window_cache is None and not graph.in_step
        h = state.h
        graph.messages_dense(h, model.cell.V)
        graph.messages_dense(h, model.cell.V)
        assert built[0] == 3  # outside a window: one build per call

    cell: PDGCell = model.cell
    with cell.window():
        graph.messages_dense(h, cell.V)
        with torch.no_grad():
            graph.edge_logit.add_(0.5)
        with pytest.raises(RuntimeError, match="must not change inside a window"):
            graph.messages_dense(h, cell.V)
    with cell.window():
        graph.messages_dense(h, cell.V)
        with torch.no_grad(), pytest.raises(RuntimeError, match="grad mode"):
            graph.messages_dense(h, cell.V)
        with pytest.raises(RuntimeError, match="do not nest"), graph.window():
            pass
    assert graph._window_cache is None


# --------------------------------------------------------------------------------------------------
# P1.10b: reset only where some row resets; the dense window's counters
# --------------------------------------------------------------------------------------------------

RESET_PATTERNS: dict[str, list[tuple[int, int]]] = {
    "none": [],
    "start_all_rows": [(b, 0) for b in range(B)],  # align_docs: every window starts a document
    "start_some_rows": [(0, 0), (2, 0)],
    "scattered": [(0, 2), (1, 4), (2, 2)],
    "every_column": [(b, t) for b in range(B) for t in range(T)],
    "last_column": [(1, T - 1)],
}
#: Gradients, default vs ``reset_columns=range(T)``: the no-op reset nodes change the order in which
#: autograd sums a parameter's per-position contributions. Bound on ``max|Δg| / max|g|`` per
#: parameter, by dtype, where a gradient below :data:`NOISE_FLOOR` times the model's largest one is
#: held to that floor instead: such a gradient is itself rounding noise (the edge logits when every
#: row starts from zeros, so that the cells are symmetric: about 1e-18 in fp64, 0 against 1e-9 in
#: fp32). Measured 6 Nov 2026 over every case, mask and parameter below, relative to that scale:
#: fp64 2.5e-15 (largest absolute 9.9e-14), fp32 1.1e-6 (largest absolute 5.3e-5, on
#: ``byte_emb.weight`` with gradients up to 69); the bounds are at least 10x those.
REORDER_TOL = {torch.float64: 1e-13, torch.float32: 2e-5}
NOISE_FLOOR = 1e-3
RESET_CASES = [
    case
    for case in CASES
    if case.name
    in (
        "dense_tiny",
        "dense_tiny_disabled",
        "dense_ref_a1",
        "masked_soft_warmup",
        "sparse_topk_eval",
        "sparse_ste_dense_fallback",
    )
]


def _reset_mask(cells: list[tuple[int, int]]) -> Tensor:
    reset = torch.zeros(B, T, dtype=torch.bool)
    for b, t in cells:
        reset[b, t] = True
    return reset


@contextmanager
def _recording_states(model: HSN) -> Iterator[list[CellGraphState]]:
    """The state every round of a window reads (the third argument of ``step_dense`` /
    ``step_sparse``), recorded by wrapping the instance's methods."""
    seen: list[CellGraphState] = []
    cell = model.cell
    dense, sparse = cell.step_dense, cell.step_sparse

    def step_dense(u: Tensor, eps: Tensor, state: CellGraphState, *rest: Any, **kw: Any) -> Any:
        seen.append(state)
        return dense(u, eps, state, *rest, **kw)

    def step_sparse(u: Tensor, eps: Tensor, state: CellGraphState, *rest: Any, **kw: Any) -> Any:
        seen.append(state)
        return sparse(u, eps, state, *rest, **kw)

    cell.step_dense, cell.step_sparse = step_dense, step_sparse  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        del cell.step_dense, cell.step_sparse


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("pattern", list(RESET_PATTERNS))
@pytest.mark.parametrize("case", RESET_CASES, ids=[case.name for case in RESET_CASES])
def test_p1_10_reset_columns_equal_per_position_reset(
    case: Case, pattern: str, dtype: torch.dtype
) -> None:
    """Resetting only at the columns where some row resets (the default, from ``reset``) equals
    resetting at every column (``reset_columns=range(T)``, the per-position reset) bit for bit —
    logits, the final state and every ``info`` key — for reset masks from none to every position,
    in the dense window (whose ``age`` / ``pos`` come from the window counters), the disabled-cell
    hook, the masked round and the sparse round, in fp64 and fp32; the parameter gradients agree
    to :data:`REORDER_TOL` (the no-op reset nodes change the order in which autograd sums a
    parameter's per-position contributions). In fp64 both also equal the per-position reference
    (stepping) to 1e-12, the counters exactly, and every round reads the same state as the
    reference's round does (the window keeps the per-position states truthful); stepping against
    the window is not compared in fp32, where the GEMM row counts' rounding is amplified by the
    recurrence beyond any tight bound."""
    model = _build(case).to(dtype)
    tokens, ctx, state, _ = _inputs(model)
    reset = _reset_mask(RESET_PATTERNS[pattern])
    with _recording_states(model) as seen:
        default = _run(lambda: model(tokens, ctx, state, reset))
    every = _run(lambda: model(tokens, ctx, state, reset, reset_columns=range(T)))
    _assert_matches(default, every, exact=True)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    grads, grads_every = _grads(model, default), _grads(model, every)
    largest = max(float(g.abs().max()) for g in grads_every if g is not None)
    for name, g, g_e in zip(names, grads, grads_every, strict=True):
        assert (g is None) == (g_e is None), name
        if g is not None and g_e is not None:
            scale = max(float(g_e.abs().max()), NOISE_FLOOR * largest)
            worst = float((g - g_e).abs().max())
            assert worst <= REORDER_TOL[dtype] * scale, (name, worst, scale)
    final = default[1]
    assert final.pos.is_contiguous() and final.age.is_contiguous()
    if dtype != torch.float64:
        return
    with _recording_states(model) as seen_ref:
        ref = _run(lambda: _stepwise(model, tokens, ctx, state, reset))
    _assert_matches(default, ref)
    assert len(seen) == len(seen_ref) >= T
    for k, (got, want) in enumerate(zip(seen, seen_ref, strict=True)):
        assert torch.equal(got.age, want.age) and torch.equal(got.pos, want.pos), k
        _close(got.h, want.h, f"round {k} h")
        _close(got.q, want.q, f"round {k} q")


def test_p1_10_reset_columns_validated() -> None:
    """``reset_columns`` must be a strictly increasing sequence of ints in ``[0, T)`` containing
    every column where ``reset`` has a True (checked on the CPU); extra columns are allowed."""
    model = _build(CASES[0])
    tokens, ctx, state, reset = _inputs(model)  # resets at columns 0 and 3
    window = (tokens, ctx, state, reset)
    base = model(*window)
    for columns in ([0, 3], (0, 1, 3, 5), range(T)):
        _assert_matches(model(*window, reset_columns=columns), base, exact=True)
    with pytest.raises(ValueError, match=r"columns \[3\] that reset_columns omits"):
        model(*window, reset_columns=[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        model(*window, reset_columns=[3, 0])
    with pytest.raises(ValueError, match="strictly increasing"):
        model(*window, reset_columns=[0, 0, 3])
    for out_of_range in ([0, 3, T], [-1, 0, 3]):
        with pytest.raises(ValueError, match=r"\[0, T\)"):
            model(*window, reset_columns=out_of_range)
    for not_ints in ([0, 3.0], [True, 3]):
        with pytest.raises(TypeError, match="ints"):
            model(*window, reset_columns=not_ints)  # type: ignore[arg-type]
    for not_a_sequence in (torch.tensor([0, 3]), "03", {0, 3}):
        with pytest.raises(TypeError, match="sequence of ints"):
            model(*window, reset_columns=not_a_sequence)  # type: ignore[arg-type]


def _trainer_docs(length_range: tuple[int, int], n: int) -> list[Example]:
    """kv/in_range documents of ``length_range`` bytes (the P1.10d trainer-test recipe)."""
    task = SPEC.task("kv", "in_range")
    task = TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 64,
            "n_dev": 16,
            "n_test": 16,
            "length_range": list(length_range),
            "dep_len_range": [4, 10],
            "unroll": TRAINER_UNROLL,
        }
    )
    return [generate(task, "train", i) for i in range(n)]  # type: ignore[arg-type]


@pytest.mark.parametrize("align_docs", [True, False])
def test_p1_10_trainer_passes_reset_columns(align_docs: bool, tmp_path: Path) -> None:
    """The TBPTT trainer passes ``reset_columns`` read from the CPU window — exactly the columns of
    the (cut) window that hold a reset — so ``HSN.forward`` never derives them from the device mask:
    aligned documents (every window cut after its documents, a reset at ``t = 0`` only) and packed
    ones (resets anywhere in the window)."""
    cfg = load_config(
        REF_A1,
        overrides=[
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=4",
            "model.in_degree=2",
            "model.n_units=2",
            "model.k_select=2",
            "model.hidden=16",
            "model.scorer_hidden=8",
            f"train.unroll={TRAINER_UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=10",
            "train.eval_every=10",
            "train.ckpt_every=10",
            f"data.align_docs={str(align_docs).lower()}",
        ],
    )
    seed_everything(0)
    model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(model, cfg, resolve(load_hardware_profile("cpu")), run_dir=tmp_path)
    seen: list[tuple[Tensor, Any]] = []
    original = model.forward

    def forward(tokens: Tensor, ctx: Tensor, state: Any, reset: Tensor, **kwargs: Any) -> Any:
        seen.append((reset.clone(), kwargs.get("reset_columns")))
        return original(tokens, ctx, state, reset, **kwargs)

    model.forward = forward  # type: ignore[method-assign]
    trainer.evaluate = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    docs = ByteDataset(_trainer_docs((16, 28), 32))
    trainer.fit(docs, docs, 3)
    assert len(seen) == 3
    for reset, columns in seen:
        assert columns == reset.any(dim=0).nonzero().flatten().tolist()
    if align_docs:
        assert all(columns == [0] and reset.shape[1] < TRAINER_UNROLL for reset, columns in seen)
    else:
        assert any(column > 0 for _, columns in seen for column in columns)


# --------------------------------------------------------------------------------------------------
# P1.10c: fewer ops in the cell
# --------------------------------------------------------------------------------------------------

#: P1.8's recipe for a saturated state: every parameter scaled so that the gates saturate.
SATURATION_SCALE = 50.0
STEPS_10K = 10_000
BOUND_WINDOW = 250


@pytest.mark.parametrize(
    ("dtype", "autocast"),
    [
        pytest.param(torch.float64, False, id="fp64"),
        pytest.param(torch.float32, False, id="fp32"),
        pytest.param(torch.float32, True, id="fp32-bf16-autocast"),
    ],
)
def test_p1_10_lerp_state_bounded_10k_steps(dtype: torch.dtype, autocast: bool) -> None:
    """The ``torch.lerp`` update keeps every coordinate of ``h`` (and of its mean ``q``) in
    ``[−1, 1]`` through 10,000 positions of ``forward`` windows (40 × 250, random inputs, document
    resets at 1 %), with every parameter scaled by P1.8's saturation factor so that the gates and
    candidates saturate: the bound is reached (worst |h| > 0.99), never passed, and ``q`` stays the
    exact mean of the cells. fp64, fp32 and fp32 under CPU bf16 autocast (the update itself runs in
    the state dtype)."""
    cfg = load_config(TINY)
    seed_everything(0)
    model = HSN.from_config(cfg).to(dtype).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SATURATION_SCALE)
    m = cfg.model
    generator = torch.Generator(device="cpu").manual_seed(1)
    Bb = 4
    state = model.init_state(Bb)
    worst_h = worst_q = 0.0
    context = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.no_grad(), context:
        for _ in range(STEPS_10K // BOUND_WINDOW):
            tokens = torch.randint(0, m.vocab_size, (Bb, BOUND_WINDOW), generator=generator)
            ctx = torch.randint(0, m.n_context, (Bb, BOUND_WINDOW), generator=generator)
            reset = torch.rand(Bb, BOUND_WINDOW, generator=generator) < 0.01
            _, state, info = model(tokens, ctx, state, reset, capture=True)
            states = info["cell_states"]
            worst_h = max(worst_h, float(states.abs().max()))
            worst_q = max(worst_q, float(states.mean(dim=2).abs().max()))
            assert torch.isfinite(states).all()
    assert state.h.dtype == dtype
    assert worst_h <= 1.0 and worst_q <= 1.0, (worst_h, worst_q)
    assert worst_h > 0.99, worst_h  # the bound is exercised, not vacuous
    assert torch.equal(state.q, state.recompute_q())


def _previous_step_dense(
    cell: PDGCell, u: Tensor, eps: Tensor, state: CellGraphState, alpha: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """The P1.10b ``PDGCell.step_dense`` (``cdc43c8`` .. ``db209bb``), verbatim: gather messages,
    ``einsum`` units, mixing as product and sum, the update as ``(1 − b)·h + b·c``."""
    B_, N, d = (int(n) for n in state.h.shape)
    h_prev = state.h
    graph = cell.graph
    Vh = cell.V(h_prev)
    weights = graph._padded_weights().to(Vh.dtype)
    m = (Vh[:, graph._pad_src] * weights[None, :, :, None]).sum(dim=2)
    v_raw = torch.cat(
        [u[:, None, :].expand(B_, N, d), eps[:, None, :].expand(B_, N, d), h_prev, m], dim=-1
    )
    v = cell.norm(v_raw)
    units = cell.units
    hidden = torch.einsum("bni,mih->bnmh", v, units.W1)
    hidden = units.act(hidden + units.b1.to(hidden.dtype))
    out = torch.einsum("bnmh,mhd->bnmd", hidden, units.W2)
    unit_out = out + units.b2.to(out.dtype)
    mixed = (alpha.unsqueeze(-1) * unit_out).sum(dim=2)
    c = torch.tanh(cell.W_out(mixed).to(h_prev.dtype))
    b = torch.sigmoid(cell.gate(v).to(h_prev.dtype))
    h_new = (1 - b) * h_prev + b * c
    if cell.disabled_cells:
        h_new = torch.where(cell.disabled[None, :, None], h_prev, h_new)
    return h_new, {"m": m, "v": v, "unit_out": unit_out, "mixed": mixed, "c": c, "b": b}


CELL_CASES = [
    pytest.param(TINY, (), torch.float64, 0.3, id="tiny-fp64"),
    pytest.param(TINY, ("model.disabled_cells=[1]",), torch.float64, 0.3, id="tiny-disabled-fp64"),
    pytest.param(TINY, ("model.edge_norm=sigmoid",), torch.float64, 0.3, id="tiny-sigmoid-fp64"),
    pytest.param(TINY, ("model.in_degree=0",), torch.float64, 0.3, id="tiny-edgeless-fp64"),
    pytest.param(REF_A1, (), torch.float64, 0.1, id="ref-fp64"),
    pytest.param(REF_A1, (), torch.float32, 0.1, id="ref-fp32"),
    pytest.param(REF_A1, SMALL, torch.float32, 0.3, id="small-fp32"),
]
CELL_ATOL = {torch.float64: 1e-12, torch.float32: 1e-5}
#: Gradients, P1.10c cell vs the P1.10b formulation: the same sums in another order. Bound on
#: ``max|Δg| / max(own max|g|, 1e-3 × the largest)`` per input and parameter (gradients reach
#: about 35 here, so an unscaled 1e-5 is not a rounding bound). Measured 6 Nov 2026 on ref fp32:
#: the response to ±1-ulp noise on the reformulated ops' outputs (unit bank, mixing, update) is
#: 6.1e-7 (20 seeds); CI's ubuntu and macOS runners differ by 3.6e-7 (``norm.weight``) and 6.9e-7
#: (``W_out.weight``) on this scale; fp64 1.7e-15. The bounds are at least 10x those.
CELL_GRAD_TOL = {torch.float64: 1e-13, torch.float32: 1e-5}


@pytest.mark.parametrize(("path", "overrides", "dtype", "scale"), CELL_CASES)
def test_p1_10_cell_ops_match_previous_formulation(
    path: Path, overrides: tuple[str, ...], dtype: torch.dtype, scale: float
) -> None:
    """The P1.10c dense round — the unit bank as ``addmm`` / ``baddbmm`` on a per-window ``W1``
    layout, the mixing as one contraction, the update as ``torch.lerp`` (the messages keep the
    slot-order gather) — equals the P1.10b formulation (a verbatim copy) to rounding: ``h_new`` and
    every intermediate at fp64 1e-12 and fp32 1e-5, and the gradients of a random projection with
    respect to the inputs and every cell parameter, each bounded relative to its own scale
    (:data:`CELL_GRAD_TOL`), in and outside a window."""
    cfg = load_config(path, overrides=list(overrides))
    seed_everything(0)
    model = HSN.from_config(cfg).to(dtype)
    generator = torch.Generator(device="cpu").manual_seed(5)
    with torch.no_grad():
        for p in model.cell.parameters():
            p.add_(
                (torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale).to(dtype)
            )
    cell = model.cell
    m = cfg.model
    Bb, N, d, M = 3, m.n_cells, m.d, m.n_units
    h = (torch.rand(Bb, N, d, generator=generator, dtype=torch.float64) * 1.6 - 0.8).to(dtype)
    state = model.init_state(Bb).with_dense(h)
    u = torch.randn(Bb, d, generator=generator, dtype=torch.float64).to(dtype)
    eps = torch.randn(Bb, d, generator=generator, dtype=torch.float64).to(dtype)
    logits = torch.randn(Bb, N, M, generator=generator, dtype=torch.float64).to(dtype)
    atol = CELL_ATOL[dtype]
    params = list(cell.parameters())

    def run(fn: Callable[..., tuple[Tensor, dict[str, Tensor]]]) -> tuple[Any, list[Any]]:
        leaves = [x.detach().requires_grad_(True) for x in (u, eps, state.h, logits)]
        st = CellGraphState(h=leaves[2], q=leaves[2].mean(dim=1), age=state.age, pos=state.pos)
        h_new, info = fn(cell, leaves[0], leaves[1], st, torch.softmax(leaves[3], dim=-1))
        w = torch.Generator(device="cpu").manual_seed(9)
        loss = sum(
            (x * torch.randn(x.shape, generator=w, dtype=torch.float64).to(x.dtype)).sum()
            for x in (h_new, info["m"], info["mixed"], info["c"], info["b"])
        )
        grads = torch.autograd.grad(loss, [*leaves, *params], allow_unused=True)
        return (h_new.detach(), {k: info[k].detach() for k in ("m", "mixed", "c", "b")}), grads

    def current(c: PDGCell, *args: Any) -> tuple[Tensor, dict[str, Tensor]]:
        return c.step_dense(*args)

    (h_ref, info_ref), g_ref = run(_previous_step_dense)
    for in_window in (False, True):
        if in_window:
            with cell.window():
                (h_new, info), g_new = run(current)
        else:
            (h_new, info), g_new = run(current)
        _close(h_new, h_ref, f"h_new (window={in_window})", atol)
        for key, value in info_ref.items():
            _close(info[key], value, f"{key} (window={in_window})", atol)
        present = [g for g in g_ref if g is not None and g.numel() > 0]
        largest = max(float(g.abs().max()) for g in present)
        for i, (g, g_r) in enumerate(zip(g_new, g_ref, strict=True)):
            assert (g is None) == (g_r is None), i
            if g is not None and g_r is not None and g_r.numel() > 0:
                scale = max(float(g_r.abs().max()), 1e-3 * largest)
                worst = float((g - g_r).abs().max())
                assert worst <= CELL_GRAD_TOL[dtype] * scale, (i, worst, scale)


def test_p1_10_cell_ops_bf16_autocast_bound() -> None:
    """Under CPU bf16 autocast the P1.10c round is at least as close to the fp32 round as the P1.10b
    formulation is. ``addmm`` / ``baddbmm`` add the biases inside the fp32 accumulator and round
    once, where ``einsum`` and a separate bf16 add rounded twice; the mixing keeps the product and
    sum there, so the mixing weights are not rounded to bf16. Measured 6 Nov 2026 on ref_a1 at the
    training-regime scale, 10 seeds, |h_new − fp32|: P1.10c max 2.0e-2 / mean 5.5e-4, P1.10b max
    2.7e-2 / mean 5.8e-4 (the two bf16 rounds differ from each other by up to 2.2e-2, mean 3.6e-4;
    the old formulation at B and B + 1 rows is bit-identical on this CPU)."""
    cfg = load_config(REF_A1)
    m = cfg.model
    errors: dict[str, list[Tensor]] = {"new": [], "old": []}
    for seed in range(4):
        seed_everything(0)
        model = HSN.from_config(cfg)
        generator = torch.Generator(device="cpu").manual_seed(5 + seed)
        with torch.no_grad():
            for p in model.cell.parameters():
                p.add_(torch.randn(p.shape, generator=generator) * 0.1)
        cell = model.cell
        h = torch.rand(3, m.n_cells, m.d, generator=generator) * 1.6 - 0.8
        state = model.init_state(3).with_dense(h)
        u = torch.randn(3, m.d, generator=generator)
        eps = torch.randn(3, m.d, generator=generator)
        alpha = torch.softmax(torch.randn(3, m.n_cells, m.n_units, generator=generator), dim=-1)
        with torch.no_grad():
            exact, _ = _previous_step_dense(cell, u, eps, state, alpha)  # fp32, no autocast
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                h_new, _ = cell.step_dense(u, eps, state, alpha)
                h_old, _ = _previous_step_dense(cell, u, eps, state, alpha)
        assert h_new.dtype == h_old.dtype == torch.float32
        errors["new"].append((h_new - exact).abs().flatten())
        errors["old"].append((h_old - exact).abs().flatten())
    new, old = torch.cat(errors["new"]), torch.cat(errors["old"])
    assert float(new.mean()) <= 1.1 * float(old.mean()), (float(new.mean()), float(old.mean()))
    assert float(new.max()) <= 1.5 * float(old.max()), (float(new.max()), float(old.max()))
    assert float(new.max()) < 0.05  # bf16 rounding of a state in [−1, 1], nothing larger


def test_p1_10_cell_layouts_cached_per_window() -> None:
    """Inside a window the dense adjacency and the unit bank's ``W1`` layout are built once and
    reused; outside a window every call builds them; a ``W1`` changed in place inside a window
    raises; nothing is kept after the window."""
    model = _build(CASES[0])
    cell = model.cell
    graph, units = cell.graph, cell.units
    assert graph.dense_adjacency() is not graph.dense_adjacency()
    assert units._w1_flat() is not units._w1_flat()
    with cell.window():
        assert graph.dense_adjacency() is graph.dense_adjacency()
        assert units._w1_flat() is units._w1_flat()
        with torch.no_grad():
            units.W1.add_(0.1)
        with pytest.raises(RuntimeError, match="W1 must not change inside a window"):
            units._w1_flat()
        with pytest.raises(RuntimeError, match="do not nest"), units.window():
            pass
    assert units._window_w1 is None and graph._window_cache is None


# macOS's BLAS (Accelerate) breaks the symmetry at rounding level, and did so before this chunk.
# CI probes of 6 Nov 2026 (draft PRs #30 and #31, closed unmerged), max over positions, as cell
# spread / edge-logit gradient on the macos-arm64-cpu runner:
#
#   case          main cdc43c8         P1.10b b278063       P1.10c 42ba15b
#   fp64 B2 T6    5.6e-17 / 3.1e-24    5.6e-17 / 6.6e-24    3.5e-17 / 6.6e-24
#   fp64 B4 T64   3.9e-16 / 3.4e-22    3.9e-16 / 3.7e-22    2.4e-16 / 2.6e-22
#   fp32 B2 T6    1.1e-7  / 1.6e-14    1.1e-7  / 2.1e-14    6.7e-8  / 2.5e-14
#   fp32 B4 T64   1.6e-7  / 2.2e-13    1.6e-7  / 2.4e-13    1.6e-7  / 3.7e-13
#
# The ubuntu-cpu runner gave 0 / 0 on every line, and the audit measured CUDA exact. The bounds
# are 10x the largest macOS value per dtype (the audit's rule for a measured platform bound).
MACOS_SYMMETRY_BOUND = {torch.float64: (4e-15, 4e-21), torch.float32: (2e-6, 4e-12)}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize(("Bb", "Tt"), [(2, 6), (4, 64)])
def test_p1_10_dense_cells_stay_symmetric_from_zero_state(
    dtype: torch.dtype, Bb: int, Tt: int
) -> None:
    """The loop fixes leave the dense model's cell symmetry exactly as it was (the audit's finding
    1; until amendment A3 breaks it by design): ref_a1 from the zero state, every row reset at
    ``t = 0``, keeps all cells bitwise identical, and the edge logits' gradient is exactly 0. The
    messages sum each row's sources in slot order for this reason; a dense ``A · V(h)`` contraction
    measured 1.6e-22 here in fp64 (P1.10c), which Adam would turn into drifting edge weights.

    Exact on Linux (and on CUDA, the audit's check). On macOS the platform BLAS rounds the cells
    apart by 1 to 2 ulp with or without this chunk, so there the test bounds the spread and the
    gradient by ``MACOS_SYMMETRY_BOUND``."""
    torch.manual_seed(0)
    run = load_config(REF_A1)
    model = HSN.from_config(run).to(dtype)
    tokens = torch.randint(0, run.model.vocab_size, (Bb, Tt))
    ctx = torch.randint(0, run.model.n_context, (Bb, Tt))
    reset = torch.zeros(Bb, Tt, dtype=torch.bool)
    reset[:, 0] = True
    logits, final, info = model(tokens, ctx, model.init_state(Bb), reset, capture=True)
    logits.logsumexp(-1).mean().backward()
    states = info["cell_states"]
    spread = float((states - states[:, :, :1]).abs().max())  # every position
    grad = model.cell.graph.edge_logit.grad
    assert grad is not None
    grad_max = float(grad.abs().max())
    measured = f"cell spread {spread:.3e}, edge grad {grad_max:.3e}"
    if sys.platform == "darwin":
        spread_bound, grad_bound = MACOS_SYMMETRY_BOUND[dtype]
        final_spread = float((final.h - final.h[:, :1]).detach().abs().max())
        assert spread <= spread_bound and final_spread <= spread_bound, measured
        assert grad_max <= grad_bound, measured
        return
    assert torch.equal(states, states[:, :, :1].expand_as(states)), measured  # bit for bit
    assert torch.equal(final.h, final.h[:, :1].expand_as(final.h)), measured
    assert not bool(grad.any()), measured


# --------------------------------------------------------------------------------------------------
# P1.10e: hot-loop Python
# --------------------------------------------------------------------------------------------------


def test_p1_10_window_validates_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``forward`` validates the window once: in the dense window the states are built without
    ``CellGraphState``'s layout check, a constant number of checks per window whatever ``T`` (the
    public constructor and every public method still validate), and the rounds skip their per-call
    input checks inside the window; outside it ``step``, ``step_dense`` and ``messages_dense``
    still refuse bad inputs, and a bad window still raises before anything runs."""
    import hsn.model.state as state_module

    calls = [0]
    check = state_module._check_layout

    def counted(*args: Any, **kwargs: Any) -> None:
        calls[0] += 1
        check(*args, **kwargs)

    monkeypatch.setattr(state_module, "_check_layout", counted)
    model = _build(CASES[0])
    m = model.cfg
    per_window = []
    for length in (4, 12):
        generator = torch.Generator(device="cpu").manual_seed(length)
        tokens = torch.randint(0, m.vocab_size, (B, length), generator=generator)
        ctx = torch.randint(0, m.n_context, (B, length), generator=generator)
        reset = torch.zeros(B, length, dtype=torch.bool)
        reset[:, 0] = True
        state = model.init_state(B)
        calls[0] = 0
        model(tokens, ctx, state, reset)
        per_window.append(calls[0])
    assert per_window[0] == per_window[1], per_window  # independent of T
    assert calls[0] <= 2, per_window

    # outside a window everything is still checked
    state = model.init_state(B)
    with pytest.raises(ValueError, match="layout|shape"):
        CellGraphState(h=state.h, q=state.q[:, :1], age=state.age, pos=state.pos)
    with pytest.raises(ValueError, match="u must have shape"):
        model.cell.step_dense(
            torch.zeros(B, m.d + 1, dtype=model.dtype),
            torch.zeros(B, m.d, dtype=model.dtype),
            state,
            torch.full((B, m.n_cells, m.n_units), 1.0 / m.n_units, dtype=model.dtype),
        )
    with pytest.raises(ValueError, match="h_prev must have shape"):
        model.cell.graph.messages_dense(state.h[:, :1], model.cell.V)
    with pytest.raises(TypeError, match="x_t"):
        model.step(torch.zeros(B), torch.zeros(B, dtype=torch.int64), state)
    with pytest.raises(ValueError, match="reset must have shape"):
        model(
            torch.zeros(B, 3, dtype=torch.int64),
            torch.zeros(B, 3, dtype=torch.int64),
            state,
            torch.zeros(B, 2, dtype=torch.bool),
        )
    assert not model.cell._in_window and model.cell.graph._window_cache is None


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_p1_10_internal_states_pass_layout_check(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every state the window builds without the layout check (``CellGraphState._make``) would pass
    it: the public constructor's ``_check_layout`` is run on each one, in every routing mode."""
    import hsn.model.state as state_module

    made = [0]
    original = CellGraphState._make.__func__  # type: ignore[attr-defined]

    def checked(cls: type[CellGraphState], h: Tensor, q: Tensor, age: Tensor, pos: Tensor) -> Any:
        state_module._check_layout(h, q, age, pos, where="CellGraphState._make")
        made[0] += 1
        return original(cls, h, q, age, pos)

    monkeypatch.setattr(CellGraphState, "_make", classmethod(checked))
    model = _build(case)
    tokens, ctx, state, reset = _inputs(model)
    logits, final, _ = _run(lambda: model(tokens, ctx, state, reset))
    assert made[0] >= T
    state_module._check_layout(final.h, final.q, final.age, final.pos, where="final")


SELECTION_GATES = [
    ("topk", None),
    ("topk", "soft"),
    ("topk", "ste"),
    ("topk", "reinforce"),
    ("greedy_budget", None),
    ("greedy_budget", "ste"),
    ("random", None),
    ("fixed", None),
    ("fixed", "ste"),
    ("dense", None),
]


@pytest.mark.parametrize(("mode", "gate"), SELECTION_GATES)
def test_p1_10_dense_fallback_only_from_dense_rows(mode: str, gate: str | None) -> None:
    """The router flags dense-fallback rows only from ``dense_rows``, never on its own, in every
    selection mode and gate. The sparse round inside a validated window relies on it, since it no
    longer re-checks the flags at every position (P1.10e). A tap's edit that adds such rows is
    refused."""
    overrides = [*SMALL, f"model.routing={mode}"]
    if mode == "fixed":
        overrides.append("model.fixed_route=[1, 4]")
    model = _build(Case(mode, REF_A1, tuple(overrides)))
    router = model.router
    generator = torch.Generator(device="cpu").manual_seed(2)
    N, M = model.cfg.n_cells, model.cfg.n_units
    logits = torch.randn(B, N, M, generator=generator, dtype=torch.float64)
    age = torch.randint(0, 5, (B, N), generator=generator)
    kwargs: dict[str, Any] = {"gate": gate}
    if mode in ("topk", "greedy_budget"):
        kwargs.update(age=age, epsilon=0.3)
    fixed = model._fixed_route if mode == "fixed" else None
    m = model.cfg
    torch.manual_seed(RNG_SEED)
    plain = router.select(logits, m.budget, m.k_select, m.k_min, mode, fixed, **kwargs)
    assert not bool(plain.dense_fallback.any())
    if mode != "dense":
        rows = torch.tensor([True, False, True])
        torch.manual_seed(RNG_SEED)
        flagged = router.select(
            logits, m.budget, m.k_select, m.k_min, mode, fixed, dense_rows=rows, **kwargs
        )
        assert torch.equal(flagged.dense_fallback, rows)

        def add_rows(selection: Selection) -> Selection:
            return replace(selection, dense_fallback=torch.ones_like(selection.dense_fallback))

        with router.intercept(add_rows), pytest.raises(ValueError, match="dense-fallback"):
            router.select(logits, m.budget, m.k_select, m.k_min, mode, fixed, **kwargs)
