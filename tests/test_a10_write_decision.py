"""Amendment A10 (``docs/amendments/A10_write_decision.md``; opt-in): the router's write decision.

``model.write_gate``: the scorer has a second linear output, one write logit per (cell, unit). A
selected routed cell is executed and read as before and written only where the write logit of its
chosen unit is positive (its gate is multiplied by ``w`` in {0, 1}). ``train.lambda_w`` adds the
cost of a write. With the key off nothing changes; with it on and every write logit positive the
model computes exactly what it computed before."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config, split_window_scope_error
from hsn.losses import assemble_terms
from hsn.model.hsn import HSN
from hsn.train.memory_plan import expected_param_count
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, ALL, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
WRITE = ("model.write_gate=true",)
MIXED = (*WRITE, "model.write_init=0.0")  # perturbed weights then give write logits of both signs


def _set_write(model: HSN, bias: float) -> HSN:
    """``model`` with a constant write logit: every routed slot may write (``bias > 0``) or none."""
    net = model.scorer.write_net
    assert net is not None
    with torch.no_grad():
        net.weight.zero_()
        net.bias.fill_(bias)
    return model


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_write_gate_is_opt_in_and_hashed_only_when_set() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.write_gate is False and base.train.lambda_w == 0.0
    same = ["model.write_gate=false", "model.write_init=2.0", "train.lambda_w=0.0"]
    assert config_hash(load_config(RECIPE_A6, same)) == config_hash(base)
    on = load_config(RECIPE_A6, list(WRITE))
    assert config_hash(on) != config_hash(base)
    assert config_hash(load_config(RECIPE_A6, [*WRITE, "train.lambda_w=0.1"])) != config_hash(on)


def test_write_gate_scope() -> None:
    with pytest.raises(ValueError, match="write_gate"):
        load_config(RECIPE_A6, [*WRITE, "model.routing=random"])
    with pytest.raises(ValueError, match="write_gate"):
        load_config(RECIPE_A6, [*WRITE, "model.scorer=mlp"])
    with pytest.raises(ValueError, match="lambda_w"):
        load_config(RECIPE_A6, ["train.lambda_w=0.1"])
    with pytest.raises(ValueError, match="reinforce"):
        load_config(RECIPE_A6, [*WRITE, "train.gate_mode=reinforce"])
    with pytest.raises(ValueError, match="write_gate"):
        load_config(GRU_2M, list(WRITE))
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    assert split_window_scope_error(load_config(RECIPE_A6, recipe)) is None
    reason = split_window_scope_error(load_config(RECIPE_A6, [*recipe, *WRITE]))
    assert reason is not None and "write_gate" in reason


def test_only_the_write_output_is_added() -> None:
    cfg, on = load_config(RECIPE_A6), load_config(RECIPE_A6, list(WRITE))
    torch.manual_seed(0)
    base = HSN.from_config(cfg)
    torch.manual_seed(0)
    model = HSN.from_config(on)
    sd_base, sd = base.state_dict(), model.state_dict()
    assert sorted(set(sd) - set(sd_base)) == [
        "router.scorer.write_net.bias",
        "router.scorer.write_net.weight",
    ]
    for key, value in sd_base.items():  # the same seed gives every other parameter the same start
        if isinstance(value, Tensor):
            assert torch.equal(value, sd[key]), key
    net = model.scorer.write_net
    assert net is not None
    assert bool((net.weight == 0).all()) and bool((net.bias == on.model.write_init).all())
    added = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
    assert added == net.weight.numel() + net.bias.numel()
    assert (
        expected_param_count(on.model)["scorer"] - expected_param_count(cfg.model)["scorer"]
        == added
    )
    assert base.scorer.write_net is None and base.selection_record_keys == tuple(
        k for k in model.selection_record_keys if k != "write_fraction"
    )


# --------------------------------------------------------------------------------------------------
# The selection: decision unchanged, write mask from the chosen unit's write logit
# --------------------------------------------------------------------------------------------------


def _logits(B: int, N: int, M: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, M, generator=g)


@pytest.mark.parametrize("gate", [None, "ste", "soft"])
def test_write_mask_follows_the_chosen_units_write_logit(gate: str | None) -> None:
    cfg = load_config(RECIPE_A6, ["model.k_select=4", *WRITE])
    router = HSN.from_config(cfg).router
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    logits = _logits(3, N, M)
    write = _logits(3, N, M, seed=5).requires_grad_(True)
    args = (cfg.model.budget, K, cfg.model.k_min, "topk")
    plain = router.select(logits, *args, gate=gate)
    sel = router.select(logits, *args, gate=gate, write_logits=write)
    # the decision and its gradient path are those of the selection without the write logits
    for name in ("cell_idx", "unit_idx", "valid", "alpha", "cell_mask", "unit_weights"):
        assert torch.equal(getattr(sel, name), getattr(plain, name)), name
    assert plain.write_mask is None and plain.write_fraction is None
    assert sel.write_mask is not None and sel.write_fraction is not None
    chosen = write.detach().gather(2, logits.argmax(dim=-1)[:, :, None]).squeeze(2)  # [B, N]
    hard = (chosen > 0).float()
    if gate == "soft":
        torch.testing.assert_close(sel.write_mask.detach(), torch.sigmoid(chosen))
    else:
        assert torch.equal(sel.write_mask.detach(), hard)
    # the write fraction: the share of the valid slots that may write
    want = sel.write_mask.detach().gather(1, sel.cell_idx).mean(dim=1)
    torch.testing.assert_close(sel.write_fraction.detach(), want)
    if gate is None:
        assert not sel.write_mask.requires_grad
        return
    # the gradient reaches the chosen unit's write logit of a cell and no other write logit
    sel.write_mask.sum().backward()
    assert write.grad is not None
    onehot = torch.nn.functional.one_hot(logits.argmax(dim=-1), M).bool()
    assert bool((write.grad[~onehot] == 0).all())
    slope = torch.sigmoid(chosen) * (1 - torch.sigmoid(chosen))
    torch.testing.assert_close(write.grad[onehot].reshape(3, N), slope)


def test_fast_cells_are_always_written() -> None:
    fast = ("model.k_select=4", "model.fast_cells=2", *WRITE)
    cfg = load_config(RECIPE_A6, list(fast))
    router = HSN.from_config(cfg).router
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    write = torch.full((2, N, M), -5.0, requires_grad=True)
    sel = router.select(
        _logits(2, N, M),
        cfg.model.budget,
        K,
        cfg.model.k_min,
        "topk",
        gate="ste",
        reserved=2,
        write_logits=write,
    )
    assert sel.write_mask is not None and sel.write_fraction is not None
    assert bool((sel.write_mask[:, :2] == 1).all()) and bool((sel.write_mask[:, 2:] == 0).all())
    assert bool((sel.write_fraction == 0).all())  # counted over the routed slots only
    sel.write_mask[:, :2].sum().backward()
    assert write.grad is not None and bool((write.grad == 0).all())


def test_select_refuses_write_logits_outside_topk() -> None:
    cfg = load_config(RECIPE_A6, ["model.k_select=4", *WRITE])
    router = HSN.from_config(cfg).router
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    logits = _logits(2, N, M)
    with pytest.raises(ValueError, match="write_logits"):
        router.select(logits, cfg.model.budget, K, cfg.model.k_min, "random", write_logits=logits)
    with pytest.raises(ValueError, match="shape"):
        router.select(
            logits, cfg.model.budget, K, cfg.model.k_min, "topk", write_logits=logits[:, :, :-1]
        )


# --------------------------------------------------------------------------------------------------
# The round: h' = (1 - w b) h + w b c
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("training", [False, True])
def test_open_write_gate_is_the_model_without_it(training: bool) -> None:
    """With every write logit positive the model computes what it computed without the key: the
    same logits and the same state, in evaluation and with the straight-through gate."""
    cfg = _cfg(*A5, "train.gate_mode=ste", *WRITE)
    model = _set_write(_model(cfg), 3.0)
    model.train(training)
    model.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    closed_form = copy.deepcopy(model)
    closed_form.scorer.write_net = None  # the scorer of the model without the key
    closed_form.cfg = closed_form.cfg.model_copy(update={"write_gate": False})
    closed_form.selection_record_keys = closed_form.selection_record_keys[:-1]
    B = tokens.shape[0]
    with torch.no_grad():
        got, state, info = model(tokens, ctx, _start(model, B), reset)
        ref, ref_state, ref_info = closed_form(tokens, ctx, _start(closed_form, B), reset)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    torch.testing.assert_close(state.h, ref_state.h, rtol=0, atol=0)
    assert "write_fraction" in info and "write_fraction" not in ref_info
    assert bool((info["write_fraction"] == 1).all())


def test_closed_write_gate_writes_nothing() -> None:
    cfg = _cfg(*A5, *WRITE)
    model = _set_write(_model(cfg), -3.0).eval()
    tokens, ctx, _ = _window(cfg)
    state = _start(model, cfg.train.batch)
    with torch.no_grad():
        out = model.step(tokens[:, 0], ctx[:, 0], state)
    assert out.selection is not None and bool(out.selection.valid.all())  # cells are still read
    torch.testing.assert_close(out.state.h, state.h, rtol=0, atol=0)


def test_round_writes_exactly_the_permitted_slots() -> None:
    """Per selected cell: the state of the model with every write open where ``w = 1``, the
    state before the position where ``w = 0``; a cell that was not selected is unchanged."""
    cfg = _cfg(*A5, *WRITE)
    model = _model(cfg).eval()
    opened = _set_write(copy.deepcopy(model), 3.0)
    tokens, ctx, _ = _window(cfg)
    state = _start(model, cfg.train.batch)
    with torch.no_grad():
        ref = opened.step(tokens[:, 0], ctx[:, 0], state)
    assert ref.selection is not None and bool(ref.selection.valid.all())
    # write logits that permit the first selected cell of every row and no other cell
    B, N, M = cfg.train.batch, cfg.model.n_cells, cfg.model.n_units
    permit = torch.full((B, N, M), -1.0, dtype=state.h.dtype)
    permit.scatter_(1, ref.selection.cell_idx[:, :1, None].expand(B, 1, M), 1.0)
    model.scorer.write_logits = lambda feats: permit  # type: ignore[method-assign]
    with torch.no_grad():
        out = model.step(tokens[:, 0], ctx[:, 0], state)
    sel = out.selection
    assert sel is not None and sel.write_mask is not None
    assert torch.equal(sel.cell_idx, ref.selection.cell_idx)
    assert torch.equal(sel.unit_idx, ref.selection.unit_idx)
    selected = torch.zeros(B, N, dtype=torch.bool).scatter(1, sel.cell_idx, sel.valid)
    writes = selected & (sel.write_mask > 0)
    assert int(writes.sum()) == B and int((selected & ~writes).sum()) == B * (sel.k - 1)
    want = torch.where(writes[:, :, None], ref.state.h, state.h)
    torch.testing.assert_close(out.state.h, want, rtol=1e-12, atol=1e-12)
    assert float((ref.state.h - state.h).abs().amax(dim=-1)[selected].min()) > 1e-6


def test_fast_cells_are_written_with_the_gate_closed() -> None:
    fast = ("model.k_select=4", "model.fast_cells=2", "model.fast_units=fixed")
    cfg = _cfg(*A5, *fast, *WRITE)
    model = _set_write(_model(cfg), -3.0).eval()
    tokens, ctx, _ = _window(cfg)
    state = _start(model, cfg.train.batch)
    with torch.no_grad():
        out = model.step(tokens[:, 0], ctx[:, 0], state)
    assert float((out.state.h[:, :2] - state.h[:, :2]).abs().max()) > 1e-6
    torch.testing.assert_close(out.state.h[:, 2:], state.h[:, 2:], rtol=0, atol=0)


def test_masked_round_equals_the_sparse_round() -> None:
    """The soft warm-up runs the masked round; with hard gates both rounds write the same values."""
    cfg = _cfg(*A5, *MIXED)
    model = _model(cfg).eval()
    tokens, ctx, _ = _window(cfg)
    state = _start(model, cfg.train.batch)
    with torch.no_grad():
        sparse = model.step(tokens[:, 0], ctx[:, 0], state, mode="sparse")
        masked = model.step(tokens[:, 0], ctx[:, 0], state, mode="masked")
    torch.testing.assert_close(sparse.state.h, masked.state.h, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(sparse.logits, masked.logits, rtol=1e-9, atol=1e-11)


def test_window_matches_stepping() -> None:
    cfg = _cfg(*A5, *MIXED)
    model = _model(cfg).eval()
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    with torch.no_grad():
        logits, _, info = model(tokens, ctx, _start(model, B), reset)
        state, steps, prev = _start(model, B), [], None
        for t in range(T):
            state = state.reset(reset[:, t])
            out = model.step(tokens[:, t], ctx[:, t], state, prev_tokens=prev)
            state, prev = out.state, tokens[:, t]
            steps.append(out.logits)
    torch.testing.assert_close(torch.stack(steps, dim=1), logits, rtol=1e-9, atol=1e-11)
    fraction = info["write_fraction"]
    assert tuple(fraction.shape) == (T,) and 0.0 < float(fraction.mean()) < 1.0


def test_write_decision_gets_a_gradient_from_the_task() -> None:
    """Straight-through: the task loss reaches the write output through the written state."""
    cfg = _cfg(*A5, "train.gate_mode=ste", *MIXED)
    model = _model(cfg)
    model.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    logits.square().mean().backward()
    net = model.scorer.write_net
    assert net is not None and net.weight.grad is not None and net.bias.grad is not None
    assert float(net.weight.grad.abs().sum()) > 0.0 and float(net.bias.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------------------
# The compiled autograd window, the cost term and training
# --------------------------------------------------------------------------------------------------

COMPILED = {
    "ste": (("train.gate_mode=ste",), True),
    "soft": ((), True),  # the soft warm-up: the masked round
    "eval": ((), False),
}


@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *MIXED,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    assert "write_fraction" in ref[2] and "write_fraction" in got[2]
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    torch.testing.assert_close(
        got[2]["write_fraction"], ref[2]["write_fraction"], rtol=1e-5, atol=1e-6
    )
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


def test_cost_term_is_the_weighted_mean_write_fraction() -> None:
    cfg = _cfg(*A5, "train.gate_mode=ste", *MIXED, "train.lambda_w=0.25", "train.lambda_p=0.0")
    model = _model(cfg)
    model.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    logits, _, info = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    batch = {
        "targets": tokens,
        "loss_mask": torch.ones_like(tokens, dtype=torch.bool),
        "doc": torch.zeros_like(tokens),
    }
    terms = assemble_terms({"logits": logits, **info}, batch, cfg)
    assert list(terms) == ["task", "write"]
    want = 0.25 * info["write_fraction"].float().mean()
    torch.testing.assert_close(terms["write"], want)
    terms["write"].backward()
    net = model.scorer.write_net
    assert net is not None and net.bias.grad is not None
    assert float(net.bias.grad.abs().sum()) > 0.0  # the cost lowers the write logits


def _trained(tmp_path: Path, *extra: str, steps: int = 30) -> tuple[RunConfig, HSN, list[float]]:
    cfg = hsn_config(
        steps,
        *A5,
        ALL,
        "model.cand_norm=rms",
        "model.n_cells=6",
        "model.k_select=4",
        "train.soft_until_step=5",
        "train.lr=3e-3",
        *WRITE,
        *extra,
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    model = trainer.model
    assert isinstance(model, HSN)
    return cfg, model, [s.loss for s in history.steps]


def test_recipe_with_the_write_gate_trains(tmp_path: Path) -> None:
    _, _, losses = _trained(tmp_path)
    assert len(losses) == 30 and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05


def test_write_cost_lowers_the_write_logits(tmp_path: Path) -> None:
    cfg, model, losses = _trained(tmp_path, "train.lambda_w=5.0")
    assert all(math.isfinite(x) for x in losses)
    net = model.scorer.write_net
    assert net is not None
    assert float(net.bias.mean()) < cfg.model.write_init - 0.01


def test_trained_state_reports_the_permitted_share() -> None:
    from hsn.data.dataset import ByteDataset
    from hsn.device import seed_everything
    from hsn.experiments.diagnostics import state_window, window_state_stats
    from hsn.runner import build_model
    from p1_11_support import cpu_ctx, docs, tiny_hsn_config

    routed = ("model.routing=topk", "train.gate_mode=ste", "model.k_select=3")
    window_cfg = tiny_hsn_config(*routed, *WRITE)
    seed_everything(0)
    model = build_model(window_cfg)
    assert isinstance(model, HSN)
    window = state_window(window_cfg, cpu_ctx(), ByteDataset(docs("delayed", "dev", 4)), 4)
    assert window_state_stats(model, window)["write/permitted"] == 1.0  # the initial bias
    assert window_state_stats(_set_write(model, -1.0), window)["write/permitted"] == 0.0
    seed_everything(0)
    plain = build_model(tiny_hsn_config(*routed))
    assert "write/permitted" not in window_state_stats(plain, window)
