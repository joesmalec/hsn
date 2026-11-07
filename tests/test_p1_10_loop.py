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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_config
from hsn.device import seed_everything
from hsn.model.cell import PDGCell
from hsn.model.graph import CellGraph
from hsn.model.hsn import HSN, StepMode
from hsn.model.router import Selection
from hsn.model.state import CellGraphState
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
    state = model.init_state(B).with_dense(h)
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
