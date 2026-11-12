"""``model.ablate_shared_u``: the "remove shared path" ablation of paper §12 and of the P2.13 matrix
(Appendix A.1: "remove u from v").

With the key on, the cell input is ``v_i = RMSNorm([ε_t; h_i; m_i])`` of width ``3d``: the
cell-input norm, the unit MLPs and the gate are built at that width and nothing a cell computes
reads ``u_t``; the scorer's features and the decoder still do. These tests check that it builds
and matches the closed forms, a pure-Python reference of the step, that the dense, masked and
sparse rounds agree (eager, static, compiled, the split window and the fused sweep's products),
that ``u`` has no path into the cell and still reaches the decoder, that it trains, and that the
default model keeps its configuration hashes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import test_p1_5_hsn as p15
import test_p2_8_compile as p28
import test_p2_10_split_window as p210
import torch
from hsn.config import (
    ModelConfig,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
    split_window_scope_error,
)
from hsn.device import DeviceContext, resolve
from hsn.experiments.spec import SpecError, SweepSpec, expand
from hsn.experiments.variants import install_variant
from hsn.kernels import cuda_ops, native
from hsn.kernels import window as window_kernels
from hsn.kernels.costs import cost_table_from_config, profile_dense_costs, write_cost_profile
from hsn.model.cell import WindowConsts
from hsn.model.hsn import HSN, PARAM_REPORT_ROWS
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank
from hsn.train.compile import SweepTwin, sweep_merge_error
from hsn.train.memory_plan import expected_param_count
from hsn.train.split import SplitWindow
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, ALL, ROUTED, _cfg, _model, _start, _window
from test_p2_5_sparse_path import _one_hot_alpha, _selection
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
RECIPE_B095 = REPO / "configs" / "experiments" / "hsn_recipe_b095.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"

ABLATE = "model.ablate_shared_u=true"
#: the small A6 recipe of the split-window tests: A5 routing, the full feedback cut, A4
RECIPE = (*A5, ALL, "model.cand_norm=rms")
F64 = torch.float64

#: Configuration hashes recorded on ``main`` at 981a56d, before the ablation was built (the cell
#: refused the key). An arm id is the hash of its configuration: these must never move.
MAIN_HASHES: tuple[tuple[Path, tuple[str, ...], str], ...] = (
    (REF_A1, (), "0831924010f9c5ecf6f1794c811f0c61ac96cdb5e3fbcc5a3c99b97a8001aa70"),
    (TINY, (), "1afdc30113f9030179f6110f43401686e57b41878eb0c3cb671b5416c9b8b031"),
    (RECIPE_A6, (), "d8b36a25971c9694b25251b4a6f20545c617e6d2796b7e4d1ac544acc3e78e28"),
    (
        RECIPE_A6,
        ("profile=rtx3060", "hardware.compile_mode=reduce-overhead"),
        "4cf87371bb8cf6b243dbb72768fc642f784f1623e572664a6989524f81c8cbaf",
    ),
    (
        RECIPE_A6,
        ("profile=rtx3060", "hardware.compile_mode=reduce-overhead", "hardware.window=fused"),
        "94011da11f52af254420ea3b4403570a75bfdbf400ff900ec6f874396ef732ca",
    ),
    (RECIPE_B095, (), "59e2910be74e204843c72e15bb79f220527e96276a45f0991843064abdd26cc7"),
)


def _close(actual: Tensor, expected: Tensor, atol: float = 1e-12) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.0)


def _round_case(
    cfg: RunConfig, model: HSN, B: int = 4, seed: int = 21
) -> tuple[CellGraphState, Tensor, Tensor, Tensor, torch.Generator]:
    """A random state (distinct cells, mixed ages), ``u``, ``eps`` and scorer logits for one round
    of ``model``, in its dtype and on its device, drawn on the CPU from ``seed``."""
    g = torch.Generator().manual_seed(seed)
    N, M, d = cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    dev, dtype = model.device, model.dtype
    h = (torch.rand(B, N, d, generator=g, dtype=F64) * 1.6 - 0.8).to(dev, dtype)
    state = model.init_state(B).with_dense(h)
    state = state.tick((torch.rand(B, N, generator=g) < 0.5).to(dev))
    u = torch.randn(B, d, generator=g, dtype=F64).to(dev, dtype)
    eps = torch.randn(B, d, generator=g, dtype=F64).to(dev, dtype)
    logits = torch.randn(B, N, M, generator=g, dtype=F64).to(dev, dtype)
    return state, u, eps, logits, g


# --------------------------------------------------------------------------------------------------
# Configuration: opt-in, one key, the default's hashes
# --------------------------------------------------------------------------------------------------


def test_the_key_is_opt_in_and_the_default_keeps_its_hashes() -> None:
    for path, overrides, digest in MAIN_HASHES:
        assert config_hash(load_config(path, overrides=list(overrides))) == digest, path
    assert ModelConfig().ablate_shared_u is False
    base, ablated = load_config(RECIPE_A6), load_config(RECIPE_A6, overrides=[ABLATE])
    assert not base.model.ablate_shared_u and ablated.model.ablate_shared_u
    assert base.model.cell_input_width == 4 * 64 and ablated.model.cell_input_width == 3 * 64
    # exactly one key of the canonical configuration moves ("each exactly one change", P2.13)
    a, b = json.loads(canonical_json(base)), json.loads(canonical_json(ablated))
    assert a["model"]["ablate_shared_u"] is False and b["model"]["ablate_shared_u"] is True
    b["model"]["ablate_shared_u"] = False
    assert a == b and config_hash(ablated) != config_hash(base)
    assert "cell_input_width" not in canonical_json(ablated)  # derived, never hashed
    # the split and the fused window keep their scope with the ablation
    for window in ("split", "fused"):
        cfg = load_config(
            RECIPE_A6,
            overrides=[
                "profile=rtx3060",
                "hardware.compile_mode=reduce-overhead",
                f"hardware.window={window}",
                ABLATE,
            ],
        )
        assert cfg.hardware.window == window and split_window_scope_error(cfg) is None
    # a baseline has no such cell: the key is refused, not ignored
    with pytest.raises(ValueError, match="ablate_shared_u"):
        load_config(GRU_2M, overrides=[ABLATE])


# --------------------------------------------------------------------------------------------------
# It builds: shapes, initialisation, parameter counts against the closed forms
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [TINY, REF_A1])
def test_ablated_model_builds_with_the_3d_cell_input(path: Path) -> None:
    cfg, model = p15._model(path, [ABLATE])
    _, base = p15._model(path)
    m = cfg.model
    d, H, M = m.d, m.hidden, m.n_units
    cell = model.cell
    assert cell.ablate_shared_u and not base.cell.ablate_shared_u
    assert cell.in_dim == cell.units.in_dim == m.cell_input_width == 3 * d
    assert base.cell.in_dim == base.cell.units.in_dim == 4 * d
    assert tuple(cell.units.W1.shape) == (M, 3 * d, H)
    assert tuple(cell.gate.weight.shape) == (d, 3 * d)
    assert tuple(cell.norm.weight.shape) == (3 * d,)
    # nothing else changes shape: the scorer still reads [u; eps; h_i] + scalars, D reads [q; u]
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    base_shapes = {name: tuple(p.shape) for name, p in base.named_parameters()}
    assert list(shapes) == list(base_shapes)
    assert {name for name in shapes if shapes[name] != base_shapes[name]} == {
        "cell.units.W1",
        "cell.gate.weight",
        "cell.norm.weight",
    }
    assert model.scorer.in_features == base.scorer.in_features
    assert model.decoder.fc1.in_features == base.decoder.fc1.in_features == 2 * d

    # the measured counts are the closed forms', row by row
    report, expected = model.param_report(), expected_param_count(m)
    assert list(report) == [*PARAM_REPORT_ROWS, "total", "edges", "total_trainable"]
    assert list(expected) == [*PARAM_REPORT_ROWS, "total"]
    for row, count in expected.items():
        assert report[row] == count, row
    base_report = base.param_report()
    assert {row for row in report if report[row] != base_report[row]} == {
        "units",
        "gate",
        "cell_norm",
        "total",
        "total_trainable",
    }
    assert base_report["total"] - report["total"] == M * d * H + d * d + d
    assert report["total_trainable"] == sum(p.numel() for p in model.parameters())
    assert report["total_trainable"] == report["total"] + report["edges"]
    if path == REF_A1:
        assert report["total"] == 1_554_313 and report["units"] == 1_053_184
        assert report["gate"] == 12_352 and report["cell_norm"] == 192

    # the same seed: every module built before the cell draws the weights of the default model
    base_params = dict(base.named_parameters())
    early = ("byte_emb.", "ctx_emb.", "encoder.", "predictor.", "shared.")
    for name, p in model.named_parameters():
        if name.startswith(early):
            assert torch.equal(p, base_params[name]), name
    # amendment A1's fan-in initialisation follows the width: std 1/sqrt(3d), cut at two std
    bound3, bound4 = 2.0 / math.sqrt(3 * d), 2.0 / math.sqrt(4 * d)
    assert bound4 < float(cell.units.W1.detach().abs().max()) <= bound3 * (1 + 1e-6)
    assert float(cell.gate.weight.detach().abs().max()) <= bound3 * (1 + 1e-6)
    assert float(base.cell.units.W1.detach().abs().max()) <= bound4 * (1 + 1e-6)
    assert float(cell.gate.bias.detach()[0]) == m.gate_bias_init
    assert "ablate_shared_u=True" in repr(cell) and f"in_dim={3 * d}" in repr(cell.units)
    assert "ablate_shared_u" not in repr(base.cell) and "in_dim" not in repr(base.cell.units)

    # a window runs, and the state dict round-trips; the 4d model's weights do not fit
    B, T = 2, 3
    tokens, ctx = p15._ids(cfg, B, T, seed=4)
    reset = torch.zeros(B, T, dtype=torch.bool)
    logits, final, info = model(tokens, ctx, model.init_state(B), reset)
    assert logits.shape == (B, T, m.vocab_size) and bool(torch.isfinite(logits).all())
    assert float(final.h.abs().max()) <= 1.0 and info["u"].shape == (B, T, d)
    _, twin = p15._model(path, [ABLATE])
    twin.load_state_dict(model.state_dict())
    again, _, _ = twin(tokens, ctx, twin.init_state(B), reset)
    assert torch.equal(again, logits)
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(base.state_dict())


def test_unit_bank_input_width() -> None:
    """``UnitBank(in_dim=...)``: every pass runs at the given width and refuses another."""
    torch.manual_seed(3)
    bank = UnitBank(3, 4, 8, "gelu", init="fan_in", in_dim=12).double()
    assert bank.in_dim == 12 and tuple(bank.W1.shape) == (3, 12, 8)
    assert bank.param_count() == 3 * (12 * 8 + 8 + 8 * 4 + 4)
    assert UnitBank(3, 4, 8, "gelu", init="fan_in").in_dim == 16  # the default: 4d
    v = torch.randn(7, 12, dtype=F64)
    idx = torch.tensor([2, 0, 0, 1, 2, 1, 0])
    every = bank.forward_all(v.reshape(1, 7, 12))[0]  # [P, M, d]
    want = every[torch.arange(7), idx]
    _close(bank.forward_selected(v, idx), want)
    _close(bank.forward_grouped(v, idx), want)
    _close(bank.forward_static(v, idx), want)
    chosen, rider = bank.forward_static_chosen(
        v, idx, w1_flat=bank._w1_flat(), w2_flat=bank._w2_flat()
    )
    assert rider is None
    _close(chosen, want)
    for call in (bank.forward_selected, bank.forward_grouped, bank.forward_static):
        with pytest.raises(ValueError, match="12"):
            call(torch.randn(7, 16, dtype=F64), idx)
    with pytest.raises(ValueError, match="in_dim"):
        UnitBank(3, 4, 8, "gelu", init="fan_in", in_dim=0)


# --------------------------------------------------------------------------------------------------
# The step against the pure-Python reference of the P1.5 tests (no torch in the reference)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["model.cell_norm=none"],
        ["model.cand_norm=rms"],
        ["model.scorer=mlp"],
        ["model.edge_norm=sigmoid"],
        ["model.disabled_cells=[1]"],
        ["model.in_degree=0"],
    ],
)
def test_ablated_step_matches_the_pure_python_reference(overrides: list[str]) -> None:
    """The reference writes ``v_i = norm([ε; h_i; m_i])`` out with loops (``_ref_step`` with
    ``ablate_shared_u``); the dense step equals it at every intermediate, and the window
    reproduces the stepped positions."""
    cfg, model = p15._model(TINY, [ABLATE, *overrides])
    assert model.dtype is F64
    p15._perturb(model, seed=12)
    B, T = 2, 3
    state = p15._random_state(model, B, seed=13)
    tokens, ctx = p15._ids(cfg, B, T, seed=14)
    out = p15._check_position(cfg, model, state, tokens[:, 0], ctx[:, 0])
    out = p15._check_position(cfg, model, out.state, tokens[:, 1], ctx[:, 1])
    reset_row = torch.tensor([False, True])
    out = p15._check_position(cfg, model, out.state.reset(reset_row), tokens[:, 2], ctx[:, 2])
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[1, 2] = True
    logits, final, _ = model(tokens, ctx, state, reset)
    p15._close_window(logits[:, 2], out.logits)
    p15._close_window(final.h, out.state.h)
    p15._close_window(final.q, out.state.q)
    assert torch.equal(final.age, out.state.age) and torch.equal(final.pos, out.state.pos)

    # the reference is not blind to the key: the 4d reference does not describe this model
    dims = {**p15._dims(cfg, model), "ablate_shared_u": False}
    with pytest.raises((IndexError, ValueError)):
        p15._ref_step(
            p15._as_lists(model),
            dims,
            state.h[0].tolist(),
            state.q[0].tolist(),
            state.age[0].tolist(),
            int(state.pos[0]),
            int(tokens[0, 0]),
            int(ctx[0, 0]),
        )


# --------------------------------------------------------------------------------------------------
# The dense, the masked and the sparse round agree (as P2.5 / P2.2 / P2.8 check the default)
# --------------------------------------------------------------------------------------------------


def _check_rounds_agree(model: HSN, cfg: RunConfig, atol: float) -> None:
    """Every cell selected, in a random order, with its argmax unit: the sparse round (eager and
    static) equals the dense round with the same one-hot ``α`` slot for slot, and writes the
    dense state; ``v`` is ``3d`` wide on both."""
    cell = model.cell
    N, M, d = cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    state, u, eps, logits, g = _round_case(cfg, model)
    B = state.batch_size
    perm = torch.argsort(torch.rand(B, N, generator=g), dim=1).to(model.device)
    units = logits.argmax(dim=-1).gather(1, perm)
    valid = torch.ones(B, N, dtype=torch.bool, device=model.device)
    sel = _selection(perm, units, valid, logits)
    alpha = _one_hot_alpha(perm, units, N, M, state.dtype)
    with torch.no_grad():
        h_dense, info_d = cell.step_dense(u, eps, state, alpha)
        h_sparse, info_s = cell.step_sparse(u, eps, state, sel)
        with cell.window():
            consts = cell.window_consts()
            h_static, info_t = cell.step_sparse(u, eps, state, sel, static=True, consts=consts)
            h_dense_w, _ = cell.step_dense(u, eps, state, alpha, consts=consts)
    assert info_d["v"].shape == info_d["v_raw"].shape == (B, N, 3 * d)
    index = perm.unsqueeze(-1).expand(B, N, d)
    _close(h_dense_w, h_dense, atol)
    for h_slots, info in ((h_sparse, info_s), (h_static, info_t)):
        v = info["v"]
        assert isinstance(v, Tensor) and v.shape == (B, N, 3 * d)
        _close(h_slots, h_dense.gather(1, index), atol)
        written = info["state"]
        assert isinstance(written, CellGraphState)
        _close(written.h, h_dense, atol)
        _close(written.q, h_dense.mean(dim=1), atol)
        assert bool((written.age == 0).all()) and torch.equal(written.pos, state.pos)
        for name in ("m", "v", "c", "b"):
            dense_t, sparse_t = info_d[name], info[name]
            assert isinstance(sparse_t, Tensor)
            gathered = dense_t.gather(1, perm.unsqueeze(-1).expand(B, N, dense_t.shape[-1]))
            torch.testing.assert_close(sparse_t, gathered, atol=atol, rtol=0.0, msg=name)
        # the slot's old state is the h block of the raw input, [d, 2d) of [eps; h; m]
        h_old = info["h_old"]
        assert isinstance(h_old, Tensor)
        assert torch.equal(h_old, state.h.gather(1, index))


def _check_masked_equals_sparse(model: HSN, cfg: RunConfig, atol: float) -> None:
    """The model's routed step: the masked round (every cell computes, the selected are written)
    and the sparse round take the same decision and write the same state and logits; the cells
    outside the selection keep their state bit for bit."""
    N, K = cfg.model.n_cells, cfg.model.k_select
    assert K < N
    g = torch.Generator().manual_seed(8)
    B = 4
    tokens = torch.randint(0, cfg.model.vocab_size, (B,), generator=g).to(model.device)
    ctx = torch.randint(0, cfg.model.n_context, (B,), generator=g).to(model.device)
    state, *_ = _round_case(cfg, model, B=B, seed=9)
    model.train()
    masked = model.step(tokens, ctx, state, "masked")
    sparse = model.step(tokens, ctx, state, "sparse")
    assert masked.selection is not None and sparse.selection is not None
    assert torch.equal(masked.selection.cell_idx, sparse.selection.cell_idx)
    assert torch.equal(masked.selection.unit_idx, sparse.selection.unit_idx)
    assert bool(sparse.selection.valid.all()) and sparse.selection.k == K
    torch.testing.assert_close(sparse.logits, masked.logits, atol=atol, rtol=atol)
    torch.testing.assert_close(sparse.state.h, masked.state.h, atol=atol, rtol=atol)
    torch.testing.assert_close(sparse.state.q, masked.state.q, atol=atol, rtol=atol)
    assert torch.equal(sparse.state.age, masked.state.age)
    mask = native.selection_mask(sparse.selection.cell_idx, sparse.selection.valid, N)
    assert torch.equal(sparse.state.h[~mask], state.h[~mask])
    assert bool((sparse.state.h[mask] != state.h[mask]).any())
    # the dense reference of the same model runs every cell at the same width
    dense = model.step(tokens, ctx, state, "dense")
    assert bool(torch.isfinite(dense.logits).all()) and bool((dense.state.age == 0).all())


@pytest.mark.parametrize(
    "extra", [(), ("model.cand_norm=rms",), ("model.cell_norm=none",), (*A5[2:], ALL)]
)
def test_dense_masked_and_sparse_rounds_agree(extra: tuple[str, ...]) -> None:
    """fp64 on the CPU, to rounding (1e-12)."""
    cfg = _cfg(*ROUTED, ABLATE, *extra)
    _check_rounds_agree(_model(cfg), cfg, atol=1e-12)
    _check_masked_equals_sparse(_model(cfg), cfg, atol=1e-11)


def test_dense_masked_and_sparse_rounds_agree_fp32() -> None:
    """The small A6 recipe in fp32, to P2.5's 1e-5 (measured on the CPU: below 5e-7)."""
    cfg = _cfg(*RECIPE, ABLATE)
    _check_rounds_agree(_model(cfg, dtype=torch.float32), cfg, atol=1e-5)
    _check_masked_equals_sparse(_model(cfg, dtype=torch.float32), cfg, atol=1e-5)


def test_sparse_round_ignores_padded_slots() -> None:
    """A budget that holds one cell pads the selection: the eager and the static round of the
    ablated cell agree and write the valid slots only."""
    probe = _model(_cfg(*ROUTED, ABLATE)).router
    budget = probe.shared_cost + 1.5 * probe.max_pair_cost  # one cell fits, two do not
    cfg = _cfg(*ROUTED, ABLATE, f"model.budget={budget}")
    model = _model(cfg)
    state, u, eps, logits, _ = _round_case(cfg, model)
    sel = model.router.select(logits, budget=budget, k_max=cfg.model.k_select, k_min=1, mode="topk")
    assert not bool(sel.valid.all()) and bool(sel.valid.any())
    with torch.no_grad():
        h0, info0 = model.cell.step_sparse(u, eps, state, sel)
        h1, info1 = model.cell.step_sparse(u, eps, state, sel, static=True)
    _close(h1, h0)
    s0, s1 = info0["state"], info1["state"]
    assert isinstance(s0, CellGraphState) and isinstance(s1, CellGraphState)
    _close(s1.h, s0.h)
    _close(s1.q, s0.q)
    mask = native.selection_mask(sel.cell_idx, sel.valid, cfg.model.n_cells)
    assert torch.equal(s0.h[~mask], state.h[~mask]) and int(mask.sum()) == int(sel.valid.sum())


def test_masked_training_window_keeps_the_sparse_windows_values() -> None:
    """``train.ste_scope=all`` (the routed training window on the masked round) and the sparse
    window of the same weights: the same decisions, logits and state."""
    cfg = _cfg(*RECIPE, ABLATE)
    sparse_model = _model(cfg)
    masked_model = _model(_cfg(*RECIPE, ABLATE, "train.ste_scope=all"))
    masked_model.load_state_dict(sparse_model.state_dict())
    tokens, ctx, reset = _window(cfg)
    reset[1, 5] = True
    state = _start(sparse_model, tokens.shape[0])
    assert masked_model._window_mode() == "masked" and sparse_model._window_mode() == "sparse"
    logits, final, info = sparse_model(tokens, ctx, state, reset)
    m_logits, m_final, m_info = masked_model(tokens, ctx, state, reset)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(m_info[key], info[key])
    torch.testing.assert_close(m_logits, logits, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(m_final.h, final.h, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(m_final.q, final.q, rtol=1e-9, atol=1e-11)
    # the window is its positions: stepping the sparse round reproduces it
    stepped = state
    sparse_model.train()
    for t in range(tokens.shape[1]):
        out = sparse_model.step(
            tokens[:, t],
            ctx[:, t],
            stepped.reset(reset[:, t]),
            prev_tokens=tokens[:, t - 1] if t else None,
        )
        stepped = out.state
    torch.testing.assert_close(stepped.h, final.h, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(out.logits, logits[:, -1], rtol=1e-9, atol=1e-11)


# --------------------------------------------------------------------------------------------------
# u has no path into the cell, and still reaches the decoder
# --------------------------------------------------------------------------------------------------


def _cell_outputs(model: HSN, cfg: RunConfig) -> tuple[Tensor, Tensor, dict[str, object]]:
    """``(u, another u, {name: fn})``: every cell computation of ``model`` as a function of ``u``
    alone (one fixed state, error, decision and mixing weights), each returning its tensors."""
    cell = model.cell
    state, u, eps, logits, g = _round_case(cfg, model)
    other = torch.randn(u.shape, generator=g, dtype=F64)
    sel = model.router.select(
        logits, cfg.model.budget, cfg.model.k_select, cfg.model.k_min, "topk", gate="ste"
    )
    alpha = torch.softmax(logits, dim=-1)

    def dense(x: Tensor) -> list[Tensor]:
        h_new, info = cell.step_dense(x, eps, state, alpha)
        return [h_new, info["c"], info["b"], info["v"]]

    def sparse(x: Tensor, static: bool = False) -> list[Tensor]:
        h_new, info = cell.step_sparse(x, eps, state, sel, static=static)
        written, c, b = info["state"], info["c"], info["b"]
        assert isinstance(written, CellGraphState)
        assert isinstance(c, Tensor) and isinstance(b, Tensor)
        return [h_new, c, b, written.h, written.q]

    def static(x: Tensor) -> list[Tensor]:
        return sparse(x, static=True)

    def replay(x: Tensor) -> list[Tensor]:
        with cell.window():
            return list(cell.replay_sparse(x, eps, state.h, sel))

    def products(x: Tensor) -> list[Tensor]:
        with torch.no_grad(), cell.window():
            consts = cell.window_consts()
            return list(
                cell.sparse_products(x, eps, state, sel.cell_idx, sel.unit_idx, sel.valid, consts)
            )

    fns = {"dense": dense, "sparse": sparse, "static": static, "replay": replay}
    return u, other, {**fns, "products": products}


@pytest.mark.parametrize("ablate", [True, False])
def test_cell_does_not_read_u(ablate: bool) -> None:
    """Every cell computation of the ablated model — the dense round (which the masked round
    runs), the sparse round eager and static, the split window's replay and the fused sweep's
    products — has no gradient path from ``u`` and returns the same values, bit for bit, for
    another ``u``. The default cell reads it on every one of them."""
    cfg = _cfg(*RECIPE, *((ABLATE,) if ablate else ()))
    u, other, fns = _cell_outputs(_model(cfg), cfg)
    u = u.requires_grad_(True)
    for name, fn in fns.items():
        assert callable(fn)
        outs = fn(u)
        same = all(torch.equal(a, b) for a, b in zip(outs, fn(other), strict=True))
        assert same is ablate, name
        if name == "products":  # forward only (the sweep runs without autograd)
            continue
        total = torch.stack([o.sum() for o in outs]).sum()
        (grad,) = torch.autograd.grad(total, u, allow_unused=True)
        if ablate:
            assert grad is None, name
        else:
            assert grad is not None and float(grad.abs().max()) > 0.0, name


@pytest.mark.parametrize("routing", [(), ROUTED, RECIPE])
def test_u_reaches_the_decoder_and_not_the_cell_state(routing: tuple[str, ...]) -> None:
    """At the model: with the scorer's features detached (A6 ``router``, so the routing decision
    carries no gradient into ``u`` either) the updated cell states of the ablated model have no
    gradient path from ``u``, and the logits still do, through ``D([q; u])``. The default model
    passes ``u`` into the cells. ``S`` keeps its gradient in both."""
    cut = () if ALL in routing else ("train.feedback_detach=[router]",)
    for ablate in (True, False):
        cfg = _cfg(*routing, *cut, *((ABLATE,) if ablate else ()))
        model = _model(cfg)
        g = torch.Generator().manual_seed(3)
        B = 3
        x = torch.randint(0, cfg.model.vocab_size, (B,), generator=g)
        c = torch.randint(0, cfg.model.n_context, (B,), generator=g)
        state = _start(model, B)
        out = model.step(x, c, state)
        (to_logits,) = torch.autograd.grad(out.logits.sum(), out.u, retain_graph=True)
        assert float(to_logits.abs().max()) > 0.0
        (to_state,) = torch.autograd.grad(
            out.state.h.square().sum(), out.u, retain_graph=True, allow_unused=True
        )
        if ablate:
            assert to_state is None
        else:
            assert to_state is not None and float(to_state.abs().max()) > 0.0
        model.zero_grad(set_to_none=True)
        out.logits.square().sum().backward()
        shared_grad = model.shared.fc1.weight.grad
        assert shared_grad is not None and float(shared_grad.abs().max()) > 0.0


def test_grads_reach_every_parameter() -> None:
    """``test_p1_5_grads_reach_every_parameter`` for the ablated model (dense, fp64 tiny): a
    window's backward reaches every parameter — the 3d first layers, the gate, the norm's gain,
    and the shared transform through the scorer and the decoder — and the step passes a
    finite-difference check in the incoming state."""
    cfg, model = p15._model(TINY, [ABLATE])
    p15._perturb(model, seed=22)
    B, T = 2, 3
    tokens, ctx = p15._ids(cfg, B, T, seed=23)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 1] = True
    state = p15._random_state(model, B, seed=24)
    logits, final, info = model(tokens, ctx, state, reset)
    loss = logits.logsumexp(-1).sum() + info["eps"].square().sum() + final.q.sum()
    model.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    for name, p in model.named_parameters():
        if name.startswith("halting."):
            assert p.grad is None, name  # defined, unused while r_max = 1
            continue
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
        assert p.grad.abs().sum().item() > 0, name
    h0 = p15._random_state(model, B, seed=25).h.detach().clone().requires_grad_(True)

    def step_fn(h: Tensor) -> tuple[Tensor, Tensor]:
        out = model.step(tokens[:, 0], ctx[:, 0], model.init_state(B).with_dense(h))
        return out.logits, out.state.h

    assert torch.autograd.gradcheck(step_fn, (h0,), atol=1e-6)


# --------------------------------------------------------------------------------------------------
# The compiled autograd window, the split window and the fused sweep's products
# --------------------------------------------------------------------------------------------------

#: name: (overrides, training); the P2.8 modes that compute the cell differently
COMPILED_MODES: dict[str, tuple[tuple[str, ...], bool]] = {
    "dense": ((), True),
    "topk_ste": (("model.routing=topk", "train.gate_mode=ste"), True),  # the static sparse round
    "topk_soft": (("model.routing=topk",), True),  # the soft warm-up: the masked round
    "topk_eval": (("model.routing=topk",), False),
}


def _check_compiled_equals_eager(ctx: DeviceContext, name: str, calls: int = 1) -> None:
    """P2.8's ``test_p2_8_compiled_equals_eager`` for the ablated model on ``ctx``."""
    overrides, training = COMPILED_MODES[name]
    routed = "model.routing=topk" in overrides
    cfg = p28._cfg(*overrides, ABLATE)
    model = p28._model(cfg, training=training, scale=0.3 if routed else 0.1).to(ctx.device)
    assert model.cell.in_dim == 3 * cfg.model.d
    backend = "inductor" if ctx.device.type == "cuda" else p28.CPU_BACKEND
    window = p28._window(model, ctx, backend=backend)
    tokens, cx, state, reset = p28._case_inputs(model, routed)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    if routed:
        assert p28._margin(ref[2], model.cfg.k_select) > p28.MIN_MARGIN
    for _ in range(calls):  # CUDA Graphs: warm-up, recording, replay
        got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


@pytest.mark.parametrize("name", list(COMPILED_MODES))
@pytest.mark.usefixtures("small_blocks")
def test_ablated_compiled_window_equals_eager(name: str) -> None:
    _check_compiled_equals_eager(p28._ctx(), name)


def _assert_split_grads(ref: dict[str, Tensor | None], got: dict[str, Tensor | None]) -> None:
    """``test_p2_10_split_window._assert_same_grads`` with the router's own precision. Every
    parameter outside the router: 1e-9 relative, as there (fp64). The router's parameters receive
    their gradient through the selection, which the router computes in float32 whatever the
    model's dtype (``Selection.alpha``, the cell mask): the two windows sum the same terms in
    another order before that cast, so an entry can differ by one float32 rounding (measured:
    2**-20 on a slot gradient of 9.1, a refreshed cell of this model with ``age_max=2``). They
    are compared to 1e-6 of the tensor's largest entry."""
    assert {n for n, g in ref.items() if g is None} == {n for n, g in got.items() if g is None}
    live = 0
    for name, want in ref.items():
        mine = got[name]
        if want is None or mine is None:
            continue
        if name.startswith("router."):
            assert float((mine - want).abs().max()) <= 1e-6 * float(want.abs().max()), name
        else:
            torch.testing.assert_close(
                mine, want, rtol=1e-9, atol=1e-12, msg=lambda m, n=name: n + m
            )
        live += int(bool(want.abs().max() > 0))
    assert live >= 20  # the paths are live: this compares real gradients, not zeros


@pytest.mark.parametrize(
    "extra",
    [(), ("model.cand_norm=none",), ("model.lambda_cost=0.05",), ("model.age_max=2",)],
)
def test_ablated_split_window_gradients_equal_autograd(extra: tuple[str, ...]) -> None:
    """``hardware.window=split`` in fp64 on the CPU: the forward sweep, the batched replay
    (:meth:`PDGCell.replay_sparse` on the 3d input) and the state recurrence return
    :meth:`HSN.forward`'s logits, state and gradients."""
    cfg = _cfg(*RECIPE, ABLATE, *extra)
    ref, got, ref_info, s_info = p210._both(cfg)
    _assert_split_grads(ref, got)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])
    assert set(s_info) == set(ref_info)
    if extra == ("model.age_max=2",):
        assert float(ref_info["refreshed"].sum()) > 0
    # the units, the gate and the norm of the 3d input carry real gradients
    for name in ("cell.units.W1", "cell.gate.weight", "cell.norm.weight", "shared.fc1.weight"):
        assert float(ref[name].abs().max()) > 0.0, name


@pytest.mark.parametrize("draw_seed", [0, 1, 2, 3])
def test_ablated_split_window_with_dense_fallback_rows(draw_seed: int) -> None:
    """``train.p_dense``: the rows drawn dense are replayed through the ablated dense step."""
    cfg = _cfg(*RECIPE, ABLATE, "train.p_dense=0.5", "model.lambda_cost=0.05")
    ref, got, ref_info, s_info = p210._both(cfg, draw_seed=draw_seed)
    _assert_split_grads(ref, got)
    torch.testing.assert_close(s_info["dense_fallback"], ref_info["dense_fallback"])


def _split_case(ctx: DeviceContext, *extra: str) -> tuple:  # type: ignore[type-arg]
    """``test_p2_10_split_window._cuda_case`` on ``ctx`` for the ablated model: the small A6
    recipe in fp32, its compiled window and one window of inputs from distinct cell states."""
    cfg = p28._cfg(*p210.CUDA_RECIPE, ABLATE, *extra)
    model = p28._model(cfg, training=True, scale=0.3).to(ctx.device)
    backend = "inductor" if ctx.device.type == "cuda" else p28.CPU_BACKEND
    compiled = p28._window(model, ctx, backend=backend)
    return cfg, model, compiled, p28._case_inputs(model, True)


def _check_split_window_compiled_equals_eager(ctx: DeviceContext, p_dense: float) -> None:
    """``test_split_window_compiled_cuda_equals_eager`` for the ablated model on ``ctx``: the
    split window over a compiled sweep takes the eager window's decisions and returns its
    logits and gradients to the compiled window's tolerance."""
    cfg, model, compiled, inputs = _split_case(ctx, f"train.p_dense={p_dense}")
    tokens, cx, state, reset = inputs
    mask = torch.zeros_like(reset)
    mask[:, 1::3] = True
    params = list(model.parameters())
    torch.manual_seed(5)
    logits, final, info = model(tokens, cx, state, reset)
    if p_dense:
        assert 0.0 < float(info["dense_fallback"].mean()) < 1.0
    assert p28._margin(info, model.cfg.k_select) > p28.MIN_MARGIN
    want = torch.autograd.grad(p210._objective(logits, info, mask), params, allow_unused=True)
    split = SplitWindow(model, cfg, compiled)
    for _ in range(3):  # warm-up, recording, replay
        torch.manual_seed(5)
        s_logits, s_final, s_info = split(tokens, cx, state, reset)
        got = torch.autograd.grad(
            p210._objective(s_logits, s_info, mask), params, allow_unused=True
        )
    tol = p28.TOL[(ctx.device.type, "fp32")]
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], info[key]), key
    assert float((s_logits - logits).abs().max()) <= tol["values"]
    for name in ("h", "q"):
        gap = (getattr(s_final, name) - getattr(final, name)).abs().max()
        assert float(gap) <= tol["values"], name
    assert torch.equal(s_final.age, final.age) and torch.equal(s_final.pos, final.pos)
    for g, w in zip(got, want, strict=True):
        assert (g is None) == (w is None)
        if w is not None:
            assert float((g - w).abs().max()) <= 2e-5 * float(w.abs().max()) + 1e-6
    assert split.calls == 3 and compiled.compile_seconds > 0.0


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("p_dense", [0.0, 0.5])
def test_ablated_split_window_over_a_compiled_sweep_equals_eager(p_dense: float) -> None:
    """fp32 on the CPU: the compiled (inductor) forward sweep under the split window."""
    _check_split_window_compiled_equals_eager(p28._ctx(), p_dense)


@pytest.mark.parametrize("extra", [(), ("model.cand_norm=none",), ("model.budget=0.1",)])
def test_fused_sweep_products_equal_the_static_round(extra: tuple[str, ...]) -> None:
    """The matrix products of the fused sweep (:meth:`PDGCell.sparse_products`, everything of
    ``hardware.window=fused`` that reads the cell input; the kernels around them read its
    outputs) against the static sparse round in fp64: ``σ(gate)`` is its ``b`` and
    ``tanh(cand_norm(y))`` its ``c``, with the separate products and with the merged layouts of
    a fused :class:`SweepTwin` (the gate riding in the unit bank's 3d first layer)."""
    norm = () if extra == ("model.cand_norm=none",) else ("model.cand_norm=rms",)
    cfg = _cfg(*A5, ALL, ABLATE, *norm, *extra)
    model = _model(cfg)
    cell = model.cell
    assert sweep_merge_error(model) is None
    sweep = SweepTwin(model, F64, fused=True)
    assert sweep.merged
    bank = sweep.twin.cell.units
    d, K = cfg.model.d, cfg.model.k_select
    wide = bank.n_units * bank.hidden + d
    assert tuple(bank.get_buffer("sweep_w1_gate").shape) == (3 * d, wide)
    padded = 0
    for seed in range(4):
        state, u, eps, logits, _ = _round_case(cfg, model, seed=30 + seed)
        sel = model.router.select(logits, cfg.model.budget, K, cfg.model.k_min, "topk", gate=None)
        padded += int((~sel.valid).sum())
        with torch.no_grad(), cell.window():
            consts = cell.window_consts()
            _, info = cell.step_sparse(u, eps, state, sel, static=True, consts=consts)
            sweep.refresh()
            w1_gate = bank.get_buffer("sweep_w1_gate")
            merged = WindowConsts(
                edge_weights=sweep.twin.cell.get_buffer("sweep_edge"),
                w1_flat=w1_gate[:, : bank.n_units * bank.hidden],
                w2_flat=bank.get_buffer("sweep_w2_flat"),
                w1_gate=w1_gate,
                b1_gate=bank.get_buffer("sweep_b1_gate"),
            )
            args = (u, eps, state, sel.cell_idx, sel.unit_idx, sel.valid)
            for gate, y in (
                cell.sparse_products(*args, consts),
                sweep.twin.cell.sparse_products(*args, merged),
                sweep.twin.cell.sparse_products(*args, merged, vh=cell.V(state.h)),
            ):
                ok = sel.valid[..., None]
                b, c = info["b"], info["c"]
                assert isinstance(b, Tensor) and isinstance(c, Tensor)
                zero = torch.zeros_like(b)
                _close(torch.where(ok, torch.sigmoid(gate), zero), torch.where(ok, b, zero), 1e-11)
                tanh_y = torch.tanh(cell.cand_norm(y))
                _close(torch.where(ok, tanh_y, zero), torch.where(ok, c, zero), 1e-11)
    if extra and "budget" in extra[0]:
        assert padded > 0
    # the in-kernel fused cell round (not used by the fused window) builds the 4d input
    assert "ablate_shared_u" in (window_kernels.cell_round_error(cell) or "")


def test_cuda_kernel_backend_keeps_the_native_path_for_the_3d_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The P2.9 gather and unit-MLP kernels have the ABI ``[.., 4d]``: with the CUDA backend the
    3d cell input and a 3d unit bank must take the native path. The device check is replaced
    (CPU tensors pass for supported) and the raw kernels raise if they are reached."""

    def raw(*args: object, **kwargs: object) -> Tensor:
        raise AssertionError("the raw CUDA kernel was called")

    monkeypatch.setattr(cuda_ops, "_supported", lambda t: True)
    monkeypatch.setattr(cuda_ops, "gather_op", raw)
    monkeypatch.setattr(cuda_ops, "mlp_op", raw)
    g = torch.Generator().manual_seed(3)
    B, N, K, d, M, H = 3, 7, 4, 8, 3, 16
    u, eps = torch.randn(B, d, generator=g), torch.randn(B, d, generator=g)
    h, m = torch.randn(B, N, d, generator=g), torch.randn(B, K, d, generator=g)
    cells = torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(B)])
    v = cuda_ops.gather_concat(None, eps, h, m, cells)
    assert v.shape == (B, K, 3 * d) and torch.equal(v, native.gather_concat(None, eps, h, m, cells))
    index = cells.unsqueeze(-1).expand(B, K, d)
    assert torch.equal(v[..., :d], eps[:, None].expand(B, K, d))
    assert torch.equal(v[..., d : 2 * d], h.gather(1, index)) and torch.equal(v[..., 2 * d :], m)
    assert torch.equal(native.gather_concat(u, eps, h, m, cells)[..., d:], v)  # the 4d input's tail
    torch.manual_seed(0)
    bank = UnitBank(M, d, H, "gelu", init="fan_in", in_dim=3 * d)
    idx = torch.arange(B * K) % M
    flat = v.reshape(B * K, 3 * d)
    for static in (False, True):
        got = cuda_ops.grouped_unit_mlp(flat, idx, bank, static=static)
        assert torch.equal(got, native.grouped_unit_mlp(flat, idx, bank, static=static))
    # the 4d input and bank do reach the kernels (the guard is the width, nothing else)
    with pytest.raises(AssertionError, match="raw CUDA kernel"):
        cuda_ops.gather_concat(u, eps, h, m, cells)
    with pytest.raises(AssertionError, match="raw CUDA kernel"):
        cuda_ops.grouped_unit_mlp(
            torch.randn(B * K, 4 * d, generator=g),
            idx,
            UnitBank(M, d, H, "gelu", init="fan_in"),
            static=True,
        )
    with pytest.raises(ValueError, match="eps must have shape"):
        native.gather_concat(None, eps[:, :-1], h, m, cells)


# --------------------------------------------------------------------------------------------------
# It trains
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("dense", ()),
        ("recipe", RECIPE),
        ("recipe_split", (*RECIPE, "hardware.window=split")),
    ],
)
def test_ablated_model_trains_and_reduces_the_loss(
    name: str, overrides: tuple[str, ...], tmp_path: Path
) -> None:
    """Thirty TBPTT steps on the tiny CPU configuration of the trainer tests (kv documents, d 8,
    4 cells, 2 units): finite losses that fall, dense and with the small A6 recipe; the split
    window's run follows the autograd window's."""
    steps = 30
    cfg = hsn_config(steps, ABLATE, "train.lr=3e-3", *overrides)
    trainer = hsn_trainer(cfg, tmp_path / name)
    assert trainer.model.cell.in_dim == 3 * cfg.model.d
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    if name == "recipe_split":
        assert trainer._split is not None and trainer._split.calls == steps
        native_run = hsn_trainer(
            hsn_config(steps, ABLATE, "train.lr=3e-3", *RECIPE), tmp_path / "autograd"
        ).fit(*datasets(), steps)
        for mine, theirs in zip(history.steps, native_run.steps, strict=True):
            assert mine.loss == pytest.approx(theirs.loss, rel=1e-4)


def test_standard_recipe_with_the_ablation_trains(tmp_path: Path) -> None:
    """``configs/experiments/hsn_recipe_a6.yaml`` with the one override, shrunk to the trainer
    tests' sizes: every part of the recipe (lazy AdamW, the WSD schedule, the feedback cut, the
    update caps on ``cell.*`` and ``router.*``, the weight average, the route guard) runs on the
    ablated model and the loss falls."""
    from p1_15_support import TINY as SMALL_HSN
    from p1_15_support import UNROLL

    steps = 30
    cfg = load_config(
        RECIPE_A6,
        overrides=[
            ABLATE,
            *SMALL_HSN,
            f"train.unroll={UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.precision=fp32",
            f"train.steps={steps}",
            "train.warmup_steps=2",
            "train.eval_every=1000",
            "train.ckpt_every=1000",
            "train.stats_every=10",
            "train.deterministic=true",
        ],
    )
    assert cfg.model.ablate_shared_u and cfg.train.weight_ema == 0.99
    assert cfg.train.route_guard.enabled and set(cfg.train.update_limit.groups) == {
        "memory",
        "routing",
    }
    trainer = hsn_trainer(cfg, tmp_path)
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.1, losses  # measured: 5.54 to 5.18
    assert trainer.model.cell.in_dim == 3 * cfg.model.d


def _check_compiled_trainer_run(
    window: str, ablate: bool, run_dir: Path, *overrides: str, steps: int = 6
) -> None:
    """A compiled TBPTT run of the small A6 recipe (the P2.8 trainer tests' configuration and
    documents) in window mode ``window``: finite objectives, fp32 master weights, every window
    compiled, and with the split or fused window every training window on the split path."""
    from hsn.data.dataset import ByteDataset
    from hsn.device import seed_everything
    from hsn.train.tbptt import TBPTTTrainer

    cfg = p28._trainer_cfg(
        "reduce-overhead",
        *RECIPE,
        *((ABLATE,) if ablate else ()),
        f"hardware.window={window}",
        f"train.steps={steps}",
        "train.stats_every=3",
        *overrides,
    )
    ctx = resolve(cfg.hardware)
    assert ctx.compile_enabled
    seed_everything(0)
    trainer = TBPTTTrainer(HSN.from_config(cfg), cfg, ctx, run_dir=run_dir)
    history = trainer.fit(ByteDataset(p28._docs(16)), ByteDataset(p28._docs(4, "dev")), steps)
    assert len(history.steps) == steps
    assert all(math.isfinite(r.total) and math.isfinite(r.loss) for r in history.steps)
    assert next(trainer.model.parameters()).dtype is torch.float32
    assert trainer.model.cell.in_dim == (3 if ablate else 4) * cfg.model.d
    assert history.measured["compiled_windows"] >= float(steps)
    assert history.evals and math.isfinite(history.evals[-1]["loss"])
    if window == "autograd":
        assert trainer._split is None
    else:
        assert trainer._split is not None and trainer._split.calls == steps
        assert trainer._split.fused == (window == "fused")


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("window", ["autograd", "split"])
def test_ablated_recipe_trains_compiled(
    window: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer with ``hardware.compile_mode`` on the CPU (dynamo and AOTAutograd, eager
    kernels, as the P2.8 trainer test): the compiled autograd window and the split window over
    the compiled sweep of the ablated model. (The fused window needs the card.)"""
    import functools

    from hsn.train import tbptt
    from hsn.train.compile import compile_window

    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    _check_compiled_trainer_run(window, True, tmp_path)


# --------------------------------------------------------------------------------------------------
# Cost profiles and harness variants
# --------------------------------------------------------------------------------------------------


def test_cost_profiles_bind_the_ablation(tmp_path: Path) -> None:
    """A cost profile records the key (the segmented replica of the dense step is checked against
    ``HSN.step`` while it is measured, so it builds the 3d input), a profile of the 4d cell is
    refused for the ablated model and the reverse, and a profile from before the key reads as
    the 4d cell's."""
    cfg = load_config(TINY, ["model.routing=topk", "train.gate_mode=ste"])
    ablated = cfg.model_copy(
        update={"model": cfg.model.model_copy(update={"ablate_shared_u": True})}
    )
    report = profile_dense_costs(HSN.from_config(cfg), resolve(cfg.hardware), batch=1, positions=1)
    assert report["cost_profile"]["ablate_shared_u"] is False
    path = write_cost_profile(report, tmp_path / "default.json")
    assert cost_table_from_config(cfg.model.model_copy(update={"cost_profile": path}))
    with pytest.raises(ValueError, match=r"ablate_shared_u.*regenerate"):
        cost_table_from_config(ablated.model.model_copy(update={"cost_profile": path}))
    report["cost_profile"].pop("ablate_shared_u")  # a profile written before the key existed
    legacy = write_cost_profile(report, tmp_path / "legacy.json")
    assert cost_table_from_config(cfg.model.model_copy(update={"cost_profile": legacy}))
    with pytest.raises(ValueError, match=r"ablate_shared_u.*legacy default.*regenerate"):
        cost_table_from_config(ablated.model.model_copy(update={"cost_profile": legacy}))

    report = profile_dense_costs(
        HSN.from_config(ablated), resolve(ablated.hardware), batch=2, positions=2
    )
    assert report["cost_profile"]["ablate_shared_u"] is True
    path = write_cost_profile(report, tmp_path / "ablated.json")
    measured = ablated.model.model_copy(update={"cost_profile": path})
    table = cost_table_from_config(measured)
    assert table.n_cells == cfg.model.n_cells and table.dense_cost() == pytest.approx(1.0)
    assert HSN.from_config(ablated.model_copy(update={"model": measured})).router is not None
    with pytest.raises(ValueError, match=r"ablate_shared_u.*regenerate"):
        cost_table_from_config(measured.model_copy(update={"ablate_shared_u": False}))


def test_harness_variants_refuse_the_ablation() -> None:
    """The P1.10 variants are written for the 4d input: refused at install and at expansion."""
    overrides = [ABLATE]
    model = HSN.from_config(load_config(TINY, overrides))
    for variant in ("cell_embed", "gru_cell", "linear_head"):
        with pytest.raises(ValueError, match="ablate_shared_u"):
            install_variant(model, variant, seed=0)
    assert install_variant(model, "pdg", seed=0) == {"variant": "pdg"}
    spec = SweepSpec.model_validate(
        {
            "name": "ablation-unsupported-variant",
            "base": {"config": "configs/tiny_gradcheck.yaml", "overrides": overrides},
            "protocol": {"name": "dry_run"},
            "arms": [{"name": "a", "family": "hsn", "variant": "cell_embed"}],
        }
    )
    with pytest.raises(SpecError, match="ablate_shared_u requires variant pdg"):
        expand(spec)
    spec = SweepSpec.model_validate(
        {
            "name": "ablation-pdg",
            "base": {"config": "configs/tiny_gradcheck.yaml", "overrides": overrides},
            "protocol": {"name": "dry_run"},
            "arms": [{"name": "a", "family": "hsn"}],
        }
    )
    assert len(expand(spec)) >= 1


def test_recipe_arms_with_the_ablation_expand(tmp_path: Path) -> None:
    """How a sweep names the ablation: an arm of the standard recipe with the one override (and
    one more for the fused window). The recipe's own arms keep the stored ids of exps 421-426;
    the ablation's arms are new ones."""
    import yaml
    from p1_11_support import fake_repo
    from test_p1_18_recipe import GUARDED

    raw = yaml.safe_load((REPO / "configs" / "sweeps" / "example_p1_18.yaml").read_text())
    raw["arms"] = [
        {"name": "hsn_recipe", "family": "hsn"},
        {"name": "no_shared_u", "family": "hsn", "overrides": [ABLATE]},
        {
            "name": "no_shared_u_fused",
            "family": "hsn",
            "overrides": [ABLATE, "hardware.window=fused"],
        },
    ]
    arms = expand(SweepSpec.model_validate(raw), repo=fake_repo(tmp_path))
    by_label = {arm.label: arm.arm_id for arm in arms}
    assert len(by_label) == 18 and len(set(by_label.values())) == 18
    assert tuple(by_label[f"hsn_recipe/s{seed}"] for seed in range(6)) == GUARDED
    assert not set(GUARDED) & {v for k, v in by_label.items() if k.startswith("no_shared_u")}


# --------------------------------------------------------------------------------------------------
# CUDA: the same checks on the card (skipped without one)
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
def test_ablated_rounds_agree_cuda() -> None:
    """fp32 on CUDA: the dense, the masked and the sparse round of the ablated model agree to
    the P2.5 tolerance."""
    ctx = p28._cuda_ctx()
    cfg = _cfg(*RECIPE, ABLATE)
    model = _model(cfg, dtype=torch.float32).to(ctx.device)
    _check_rounds_agree(model, cfg, atol=1e-5)
    _check_masked_equals_sparse(model, cfg, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("name", list(COMPILED_MODES))
def test_ablated_compiled_window_equals_eager_cuda(name: str) -> None:
    """The compiled window with CUDA Graphs (warm-up, recording, replay) in fp32."""
    _check_compiled_equals_eager(p28._cuda_ctx(), name, calls=3)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("p_dense", [0.0, 0.5])
def test_ablated_split_window_compiled_cuda_equals_eager(p_dense: float) -> None:
    """``hardware.window=split`` over a compiled, CUDA-graph sweep (the sweep twin) in fp32."""
    _check_split_window_compiled_equals_eager(p28._cuda_ctx(), p_dense)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("extra", [(), ("model.age_max=2",)])
@pytest.mark.parametrize("resets", [False, True])
def test_ablated_fused_sweep_tape_is_the_split_sweeps_cuda(
    extra: tuple[str, ...], resets: bool
) -> None:
    """``hardware.window=fused`` (``test_fused_sweep_tape_is_the_split_sweeps_in_fp32`` for the
    ablated model): the fused sweep (the selection and end-of-round kernels around the 3d
    products, merged layouts) records the tape of the split sweep: decisions, ages and positions
    bit for bit, the state values to P2.8's tolerance."""
    from hsn.train.compile import TAPE_KEYS

    ctx = p28._cuda_ctx()
    cfg, model, compiled, (tokens, cx, state, reset) = _split_case(ctx, *extra)
    if not window_kernels.available():
        pytest.skip("window kernels not built (scripts/build_window_kernels.py)")
    window_kernels.prepare(0)
    if resets:  # blocks of 4 positions: the first of a block, an interior one, the last
        reset = reset.clone()
        reset[0, 0] = True
        reset[1, 2] = True
        reset[0, 3] = True
        reset[1, 4] = True
        reset[0, int(reset.shape[1]) - 1] = True
    tapes = {}
    with torch.no_grad():
        pro = model._window_prologue(
            tokens, cx, state, reset, model._window_mode(), reset_first=True
        )
        for fused in (False, True):
            for _ in range(3):  # warm-up, recording, replay
                tape = compiled.tape(pro, state, reset, fused=fused)
            tapes[fused] = {k: v.clone() for k, v in tape.items()}
    tol = p28.TOL[("cuda", "fp32")]["values"]
    for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
        got, want = tapes[True][key], tapes[False][key]
        if got.dtype.is_floating_point:
            assert float((got - want).abs().max()) <= tol, key
        else:
            assert torch.equal(got, want), key
    for key in ("explored", "refreshed"):
        assert torch.equal(tapes[True][key], tapes[False][key]), key
    if extra:
        assert float(tapes[True]["refreshed"].sum()) > 0.0
    fused_twins = [twin for (_, is_fused), twin in compiled._sweep_twins.items() if is_fused]
    assert fused_twins and all(twin.merged for twin in fused_twins)  # the merged products ran
    assert all(twin.twin.cell.in_dim == 3 * cfg.model.d for twin in fused_twins)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("ablate", [True, False])
@pytest.mark.parametrize("window", ["autograd", "split", "fused"])
def test_ablated_recipe_trains_compiled_cuda(window: str, ablate: bool, tmp_path: Path) -> None:
    """The small A6 recipe under the card's profile (bf16 autocast, compiled windows with CUDA
    Graphs) in each window mode. ``ablate=False`` is the control: the same run of the default
    model, so a failure of both is not the ablation's."""
    from cards import CUDA_PROFILE

    p28._cuda_ctx()
    if window == "fused" and not window_kernels.available():
        pytest.skip("window kernels not built (scripts/build_window_kernels.py)")
    _check_compiled_trainer_run(window, ablate, tmp_path, f"profile={CUDA_PROFILE}")
