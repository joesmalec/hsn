"""Amendment A8 (``docs/amendments/A8_fast_cells.md``): ``model.fast_cells`` and
``model.fast_readout``.

With ``model.fast_cells = F`` the cells ``0 .. F − 1`` are selected at every position and the
router fills the other ``k_select − F`` slots from the remaining cells; with
``model.fast_readout`` the fast group has its own read ``f`` into ``S`` and the decoder and the
selected-cell readout covers the routed slots only. These tests check that both keys are opt-in
and leave the default model and its hashes alone, the selection rule (reserved slots, the
straight-through threshold, the probes), the model's rounds (soft warm-up, hard gates,
evaluation; step against window; eager against compiled), the read-out, the closed forms, the
refusals, and that the recipe trains with them.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config, split_window_scope_error
from hsn.experiments.variants import install_variant
from hsn.model.hsn import HSN
from hsn.model.router import Router
from hsn.train.memory_plan import expected_param_count
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, ALL, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"

#: two fast cells and two routed slots on the six-cell model of the A6 tests
FAST = ("model.k_select=4", "model.fast_cells=2")
READ = (*FAST, "model.fast_readout=true")
F = 2


def _fast_cfg(*extra: str) -> RunConfig:
    return _cfg(*A5, *FAST, *extra)


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_keys_are_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.fast_cells == 0 and base.model.fast_readout is False
    named = load_config(RECIPE_A6, ["model.fast_cells=0", "model.fast_readout=false"])
    assert config_hash(named) == config_hash(base)
    on = load_config(RECIPE_A6, ["model.k_select=8", "model.fast_cells=4"])
    off = load_config(RECIPE_A6, ["model.k_select=8"])
    assert config_hash(on) != config_hash(off)
    read = load_config(
        RECIPE_A6, ["model.k_select=8", "model.fast_cells=4", "model.fast_readout=true"]
    )
    assert len({config_hash(on), config_hash(off), config_hash(read)}) == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["model.fast_cells=2"], "needs model.routing=topk"),  # ref_a1 is dense
        (["model.routing=topk", "model.k_select=4", "model.fast_cells=4"], "must be <"),
        (
            [
                "model.routing=topk",
                "model.k_select=4",
                "model.fast_cells=2",
                "model.disabled_cells=[1]",
            ],
            "lie in the fast group",
        ),
        (["model.routing=topk", "model.fast_readout=true"], "requires model.fast_cells"),
        (
            [
                "model.routing=topk",
                "model.k_select=4",
                "model.fast_cells=2",
                "model.fast_readout=true",
            ],
            "requires model.fast_cells > 0 and model.selected_readout",
        ),
        (
            [
                "model.routing=topk",
                "model.k_select=4",
                "model.fast_cells=2",
                "train.gate_mode=reinforce",
            ],
            "reinforce",
        ),
    ],
)
def test_configuration_refusals(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(REF_A1, overrides)


def test_split_and_fused_windows_refuse_the_fast_group() -> None:
    recipe = ["model.k_select=8", "train.gate_mode=ste", "model.fast_cells=4"]
    cfg = load_config(RECIPE_A6, recipe)
    reason = split_window_scope_error(cfg)
    assert reason is not None and "fast_cells" in reason
    for window in ("split", "fused"):
        with pytest.raises(ValueError, match="fast_cells"):
            load_config(RECIPE_A6, [*recipe, f"hardware.window={window}"])


def test_baselines_and_variants_refuse_the_keys() -> None:
    gru = REPO / "configs" / "baselines" / "gru_2m.yaml"
    with pytest.raises(ValueError, match="HSN-only"):
        load_config(gru, ["model.routing=topk", "model.k_select=4", "model.fast_cells=2"])
    model = HSN.from_config(_cfg("model.routing=topk", "train.gate_mode=ste", *FAST))
    with pytest.raises(ValueError, match="fast_cells"):
        install_variant(model, "gru_cell", {})


# --------------------------------------------------------------------------------------------------
# The selection rule
# --------------------------------------------------------------------------------------------------


def _router(cfg: RunConfig) -> Router:
    return HSN.from_config(cfg).router


def _logits(B: int, N: int, M: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, M, generator=g)


@pytest.mark.parametrize("gate", [None, "ste", "soft"])
def test_reserved_zero_is_the_old_selection_bit_for_bit(gate: str | None) -> None:
    cfg = _cfg(*A5, "model.k_select=4")
    router = _router(cfg)
    logits = _logits(5, 6, 3).requires_grad_(gate is not None)
    kw = dict(budget=1.0, k_max=4, k_min=1, mode="topk", gate=gate)
    old = router.select(logits, **kw)
    new = router.select(logits, **kw, reserved=0)
    for name in ("cell_idx", "unit_idx", "valid", "alpha", "cell_mask", "unit_weights", "cost"):
        assert torch.equal(getattr(old, name), getattr(new, name)), name
    assert torch.equal(old.threshold, new.threshold)


@pytest.mark.parametrize("gate", [None, "ste", "soft"])
def test_reserved_cells_take_the_first_slots_and_the_router_fills_the_rest(
    gate: str | None,
) -> None:
    cfg = _fast_cfg()
    router = _router(cfg)
    B, N, M, K = 7, 6, 3, 4
    logits = _logits(B, N, M, seed=3)
    logits[:, :F] -= 5.0  # the fast cells would never be chosen by score
    logits.requires_grad_(gate is not None)
    sel = router.select(logits, budget=1.0, k_max=K, k_min=1, mode="topk", gate=gate, reserved=F)
    scores = logits.detach().float()
    cell = scores.amax(dim=-1)
    assert bool(sel.valid.all())
    assert torch.equal(sel.cell_idx[:, :F], torch.arange(F).expand(B, F))
    assert torch.equal(sel.unit_idx[:, :F], scores.argmax(dim=-1)[:, :F])
    # the routed slots: the best K - F of the other cells, in score order
    rest = cell[:, F:].sort(dim=1, descending=True, stable=True).indices[:, : K - F] + F
    assert torch.equal(sel.cell_idx[:, F:], rest)
    assert torch.equal(sel.unit_idx[:, F:], scores.argmax(dim=-1).gather(1, rest))
    # the threshold: midway between the lowest routed and the highest unselected cell
    ordered = cell[:, F:].sort(dim=1, descending=True).values
    want = 0.5 * (ordered[:, K - F - 1] + ordered[:, K - F])
    torch.testing.assert_close(sel.threshold, want)
    # the fast cells are fully written and carry no selection gradient
    assert torch.equal(sel.cell_mask.detach()[:, :F], torch.ones(B, F))
    if gate is not None:
        (grad,) = torch.autograd.grad(
            sel.cell_mask[:, :F].sum(), logits, allow_unused=True, retain_graph=True
        )
        assert grad is None or float(grad.abs().max()) == 0.0
        (grad,) = torch.autograd.grad(sel.cell_mask[:, F:].sum(), logits)
        assert float(grad[:, F:].abs().max()) > 0.0
        assert float(grad[:, :F].abs().max()) == 0.0
    if gate != "soft":
        hard = torch.zeros(B, N).scatter_(1, sel.cell_idx, 1.0)
        assert torch.equal(sel.cell_mask.detach(), hard)


def test_probes_never_replace_a_reserved_slot() -> None:
    cfg = _fast_cfg()
    router = _router(cfg)
    B, N, M, K = 64, 6, 3, 4
    logits = _logits(B, N, M, seed=5)
    age = torch.full((B, N), router.age_max + 10, dtype=torch.int64)  # every cell is stale
    g = torch.Generator().manual_seed(0)
    sel = router.select(
        logits,
        budget=1.0,
        k_max=K,
        k_min=1,
        mode="topk",
        gate="ste",
        epsilon=1.0,
        age=age,
        generator=g,
        reserved=F,
    )
    assert torch.equal(sel.cell_idx[:, :F], torch.arange(F).expand(B, F))
    assert not bool(sel.refreshed[:, :F].any()) and not bool(sel.explored[:, :F].any())
    assert bool(sel.refreshed[:, F:].any())  # the probes still act on the routed slots
    assert bool((sel.cell_idx[:, F:] >= F).all())
    # distinct cells in every row
    assert all(len(set(row.tolist())) == K for row in sel.cell_idx)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(mode="random", reserved=2), "needs mode='topk'"),
        (dict(mode="topk", gate="reinforce", reserved=2), "needs mode='topk'"),
        (dict(mode="topk", reserved=4), "must be < k_max"),
        (dict(mode="topk", reserved=-1), "int >= 0"),
    ],
)
def test_select_refuses_bad_reserved(kwargs: dict[str, object], message: str) -> None:
    router = _router(_fast_cfg())
    with pytest.raises(ValueError, match=message):
        router.select(_logits(2, 6, 3), budget=1.0, k_max=4, k_min=1, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [FAST, READ])
def test_param_report_matches_the_closed_form(extra: tuple[str, ...]) -> None:
    for cfg in (
        _cfg(*A5, *extra),
        load_config(RECIPE_A6, ["model.k_select=12", "model.fast_cells=4", *extra[2:]]),
    ):
        model = HSN.from_config(cfg)
        report, expected = model.param_report(), expected_param_count(cfg.model)
        for row, count in expected.items():
            assert report[row] == count, row
    d, h = cfg.model.d, cfg.model.hidden
    plain = expected_param_count(load_config(RECIPE_A6, ["model.k_select=12"]).model)["total"]
    assert expected["total"] - plain == (2 * d * h if "model.fast_readout=true" in extra else 0)


def test_fast_readout_keeps_the_other_parameters_initialisation() -> None:
    """The extra input columns are drawn from their own generator: every parameter the two
    models share starts equal, and the widened layers' old columns too."""
    torch.manual_seed(0)
    plain = HSN.from_config(_cfg(*A5, *FAST))
    torch.manual_seed(0)
    read = HSN.from_config(_cfg(*A5, *READ))
    theirs = dict(read.named_parameters())
    d = plain.cfg.d
    for name, p in plain.named_parameters():
        q = theirs[name]
        if name in ("shared.fc1.weight", "decoder.fc1.weight"):
            assert q.shape[1] == p.shape[1] + d
            assert torch.equal(q[:, : p.shape[1]], p), name
            assert float(q[:, p.shape[1] :].detach().abs().max()) > 0.0
        else:
            assert torch.equal(p, q), name


@pytest.mark.parametrize("extra", [FAST, READ])
@pytest.mark.parametrize("phase", ["soft", "ste", "eval"])
def test_fast_cells_are_written_at_every_position(extra: tuple[str, ...], phase: str) -> None:
    cfg = _cfg(*A5, *extra, "train.soft_until_step=5")
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    reset[1, 5] = True
    if phase == "eval":
        model.eval()
    else:
        model.set_train_step(0 if phase == "soft" else 10)
        assert model._controls().gate == phase
    B = cfg.train.batch
    with torch.set_grad_enabled(phase != "eval"):
        logits, state, info = model(tokens, ctx, _start(model, B), reset, capture=True)
    assert bool(torch.isfinite(logits).all())
    written = info["cell_written"]
    assert bool(written[:, :, :F].all())
    if phase != "soft":
        assert torch.equal(
            info["cell_idx"][:, :, :F], torch.arange(F).expand(B, tokens.shape[1], F)
        )
        assert bool((info["cell_idx"][:, :, F:] >= F).all())
        assert int(written.sum(dim=-1).max()) == cfg.model.k_select
        # a cell that was not written keeps its state exactly
        h = info["cell_states"]
        kept = ~written[:, 1:] & ~reset[:, 1:, None]
        assert torch.equal(h[:, 1:][kept], h[:, :-1][kept])


@pytest.mark.parametrize("extra", [FAST, READ])
def test_window_matches_stepping(extra: tuple[str, ...]) -> None:
    cfg = _cfg(*A5, *extra)
    model = _model(cfg).eval()
    tokens, ctx, reset = _window(cfg)
    reset[2, 4] = True
    B, T = tokens.shape
    with torch.no_grad():
        logits, final, _ = model(tokens, ctx, _start(model, B), reset)
        state, steps, prev = _start(model, B), [], None
        for t in range(T):
            state = state.reset(reset[:, t])
            out = model.step(tokens[:, t], ctx[:, t], state, prev_tokens=prev)
            state, prev = out.state, tokens[:, t]
            steps.append(out.logits)
    torch.testing.assert_close(torch.stack(steps, dim=1), logits, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(state.h, final.h, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("phase", ["soft", "ste", "eval"])
def test_fast_readout_reads_the_routed_slots_and_the_fast_mean(phase: str) -> None:
    cfg = _cfg(*A5, *READ, "train.soft_until_step=5")
    model = _model(cfg)
    if phase == "eval":
        model.eval()
    else:
        model.set_train_step(0 if phase == "soft" else 10)
    B, d = cfg.train.batch, cfg.model.d
    tokens, ctx, _ = _window(cfg)
    state = _start(model, B)
    with torch.no_grad():
        out = model.step(tokens[:, 0], ctx[:, 0], state, mode="sparse")
        sel = out.selection
        assert sel is not None
        readout = model._selected_readout(out.state, sel, model._controls().gate, "sparse")
    assert readout.shape == (B, 2 * d)
    h = out.state.h
    torch.testing.assert_close(readout[:, d:], h[:, :F].mean(dim=1))
    if phase == "soft":
        w = sel.cell_mask.to(h.dtype).clone()
        w[:, :F] = 0
        want = (h * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    else:
        want = torch.stack([h[b, sel.cell_idx[b, F:]].mean(dim=0) for b in range(B)])
    torch.testing.assert_close(readout[:, :d], want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("cut", [(), (ALL,), ("train.feedback_detach=[summary,eps,router]",)])
def test_gradients_reach_the_new_columns(cut: tuple[str, ...]) -> None:
    cfg = _cfg(*A5, *READ, *cut)
    model = _model(cfg)
    model.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, cfg.train.batch), reset)
    logits.logsumexp(dim=-1).sum().backward()
    d = cfg.model.d
    for name in ("shared.fc1.weight", "decoder.fc1.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and float(grad[:, -d:].abs().sum()) > 0.0, name
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


# --------------------------------------------------------------------------------------------------
# The compiled autograd window and training
# --------------------------------------------------------------------------------------------------

COMPILED = {
    "ste": (("train.gate_mode=ste",), True),
    "soft": ((), True),  # the soft warm-up: the masked round
    "eval": ((), False),
}


@pytest.mark.parametrize("extra", [FAST, READ])
@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str, extra: tuple[str, ...]) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *extra,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


@pytest.mark.parametrize(
    "extra", [("model.fast_cells=1",), ("model.fast_cells=1", "model.fast_readout=true")]
)
def test_recipe_with_fast_cells_trains(extra: tuple[str, ...], tmp_path: Path) -> None:
    """Thirty TBPTT steps of the small A6 recipe (soft warm-up, then hard gates) on the tiny CPU
    configuration of the trainer tests: finite losses that fall."""
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        ALL,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=5",
        "train.lr=3e-3",
        *extra,
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)


# --------------------------------------------------------------------------------------------------
# Diagnostics: the gate statistics by group
# --------------------------------------------------------------------------------------------------


def test_trained_state_reports_the_gate_by_group() -> None:
    from hsn.data.dataset import ByteDataset
    from hsn.device import seed_everything
    from hsn.experiments.diagnostics import state_window, window_state_stats
    from hsn.runner import build_model
    from p1_11_support import cpu_ctx, docs, tiny_hsn_config

    routed = ("model.routing=topk", "train.gate_mode=ste", "model.k_select=3")
    plain = tiny_hsn_config(*routed)
    cfg = tiny_hsn_config(*routed, "model.fast_cells=2")
    seed_everything(0)
    model = build_model(cfg)
    window = state_window(cfg, cpu_ctx(), ByteDataset(docs("delayed", "dev", 4)), 4)
    stats = window_state_stats(model, window)
    seed_everything(0)
    before = window_state_stats(build_model(plain), window)
    assert not any("fast" in key or "slow" in key for key in before)
    model.eval()
    with torch.no_grad():
        B = int(window["tokens"].shape[0])
        _, _, info = model(
            window["tokens"], window["context"], model.init_state(B), window["reset"], capture=True
        )
    valid = window["doc"] >= 0
    gates, written = info["cell_gates"].float()[valid], info["cell_written"][valid]
    for tag, cells in (("fast", slice(0, 2)), ("slow", slice(2, None))):
        group = gates[:, cells][written[:, cells]]
        group = group[~torch.isnan(group)]
        assert stats[f"gate_{tag}/mean"] == pytest.approx(float(group.mean()), rel=1e-6)
        assert stats[f"gate_{tag}/write_rate"] == float((group >= 0.5).float().mean())
        assert stats[f"gate_{tag}/p10"] <= stats[f"gate_{tag}/p50"] <= stats[f"gate_{tag}/p90"]
        assert math.isfinite(stats[f"rank_pr_{tag}/valid"])
    # the pooled statistics are still there, unchanged in meaning
    pooled = gates[written]
    pooled = pooled[~torch.isnan(pooled)]
    assert stats["gate/mean"] == pytest.approx(float(pooled.mean()), rel=1e-6)
