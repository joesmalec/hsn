"""Amendment A12 (``docs/amendments/A12_address_gate.md``; opt-in): address weights that depend on
the position's content. ``model.address_gate``: the scorer adds ``(G u_t)[j, a] * address_a[i]`` to
the logit of cell ``i`` and unit ``j``. ``G`` is zero at the start, so the model starts as the model
without the key; with the key off nothing changes."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import config_hash, load_config, split_window_scope_error
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
REF_A1 = REPO / "configs" / "ref_a1.yaml"
GATE = ("model.address_gate=true",)


def test_address_gate_is_opt_in_and_hashed_only_when_set() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.address_gate is False
    assert config_hash(load_config(RECIPE_A6, ["model.address_gate=false"])) == config_hash(base)
    assert config_hash(load_config(RECIPE_A6, list(GATE))) != config_hash(base)


def test_address_gate_scope() -> None:
    with pytest.raises(ValueError, match="address_gate"):
        load_config(REF_A1, list(GATE))  # no key address
    with pytest.raises(ValueError, match="address_gate"):
        load_config(RECIPE_A6, [*GATE, "model.scorer=mlp"])
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    assert split_window_scope_error(load_config(RECIPE_A6, recipe)) is None
    reason = split_window_scope_error(load_config(RECIPE_A6, [*recipe, *GATE]))
    assert reason is not None and "address_gate" in reason


@pytest.mark.parametrize("ngram", [1, 2])
def test_only_the_gate_is_added(ngram: int) -> None:
    extra = [f"model.address_ngram={ngram}"]
    cfg, on = load_config(RECIPE_A6, extra), load_config(RECIPE_A6, [*extra, *GATE])
    torch.manual_seed(0)
    base = HSN.from_config(cfg)
    torch.manual_seed(0)
    model = HSN.from_config(on)
    sd_base, sd = base.state_dict(), model.state_dict()
    assert sorted(set(sd) - set(sd_base)) == ["router.scorer.address_gate.weight"]
    for key, value in sd_base.items():  # the same seed gives every other parameter the same start
        if isinstance(value, Tensor):
            assert torch.equal(value, sd[key]), key
    gate = model.scorer.address_gate
    assert gate is not None and base.scorer.address_gate is None
    d, units = on.model.d, on.model.n_units
    assert tuple(gate.weight.shape) == (units * 2 * ngram, d)
    assert bool((gate.weight == 0).all())
    added = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
    assert added == gate.weight.numel()
    assert (
        expected_param_count(on.model)["scorer"] - expected_param_count(cfg.model)["scorer"]
        == added
    )


@pytest.mark.parametrize("ngram", [1, 2])
def test_logits_against_a_direct_computation(ngram: int) -> None:
    cfg = load_config(RECIPE_A6, [f"model.address_ngram={ngram}", *GATE])
    scorer = HSN.from_config(cfg).scorer
    gate = scorer.address_gate
    assert gate is not None
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(gate.weight.shape, generator=g))
    B, N, M, d = 3, cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    A = 2 * ngram
    feats = torch.randn(B, N, scorer.in_features, generator=g)
    feats[:, :, :d] = feats[:, :1, :d]  # u is the same in every cell's row
    got = scorer(feats)
    plain = scorer.net(feats)
    u, address = feats[:, 0, :d], feats[:, :, scorer.in_features - A :]
    weights = (u @ gate.weight.T).view(B, M, A)  # the content-dependent address weights
    want = plain.clone()
    for b in range(B):
        for j in range(M):
            want[b, :, j] += address[b] @ weights[b, j]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # a zero gate is the linear scorer, exactly
    with torch.no_grad():
        gate.weight.zero_()
    assert torch.equal(scorer(feats), plain)


def test_a_unit_can_follow_an_address_at_one_content_and_not_at_another() -> None:
    """What the linear scorer cannot express: the same unit weights the current byte's address by
    +w where the content is u1 and by 0 where it is u2."""
    cfg = load_config(RECIPE_A6, list(GATE))
    scorer = HSN.from_config(cfg).scorer
    gate = scorer.address_gate
    assert gate is not None
    N, M, d = cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    a_cur = scorer.feature_slices["a_cur"].start
    u1, u2 = torch.zeros(d), torch.zeros(d)
    u1[0], u2[1] = 1.0, 1.0
    with torch.no_grad():
        scorer.net.weight.zero_()
        scorer.net.bias.zero_()
        gate.weight.zero_()
        gate.weight.view(M, 2, d)[3, 0, 0] = 2.0  # unit 3, feature a_cur, content direction u1
    feats = torch.zeros(2, N, scorer.in_features)
    feats[0, :, :d], feats[1, :, :d] = u1, u2
    feats[:, :, a_cur] = torch.linspace(-1, 1, N)
    logits = scorer(feats)
    torch.testing.assert_close(logits[0, :, 3], 2.0 * torch.linspace(-1, 1, N))
    assert bool((logits[1] == 0).all()) and bool((logits[0, :, :3] == 0).all())


@pytest.mark.parametrize("training", [False, True])
def test_zero_gate_is_the_model_without_it(training: bool) -> None:
    cfg = _cfg(*A5, "train.gate_mode=ste", *GATE)
    plain_cfg = _cfg(*A5, "train.gate_mode=ste")
    model, plain = _model(cfg), _model(plain_cfg)
    gate = model.scorer.address_gate
    assert gate is not None
    with torch.no_grad():
        gate.weight.zero_()
        for (name, p), (_, q) in zip(
            ((n, p) for n, p in model.named_parameters() if "address_gate" not in n),
            plain.named_parameters(),
            strict=True,
        ):
            p.copy_(q)
            assert name
    model.train(training)
    plain.train(training)
    model.set_train_step(10)
    plain.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    B = tokens.shape[0]
    with torch.no_grad():
        got, state, _ = model(tokens, ctx, _start(model, B), reset)
        ref, ref_state, _ = plain(tokens, ctx, _start(plain, B), reset)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    torch.testing.assert_close(state.h, ref_state.h, rtol=0, atol=0)


def test_window_matches_stepping_and_the_masked_round() -> None:
    cfg = _cfg(*A5, *GATE)
    model = _model(cfg).eval()  # the perturbed start gives a non-zero gate
    gate = model.scorer.address_gate
    assert gate is not None and float(gate.weight.abs().sum()) > 0
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    with torch.no_grad():
        logits, _, _ = model(tokens, ctx, _start(model, B), reset)
        state, steps, prev = _start(model, B), [], None
        for t in range(T):
            state = state.reset(reset[:, t])
            out = model.step(tokens[:, t], ctx[:, t], state, prev_tokens=prev)
            state, prev = out.state, tokens[:, t]
            steps.append(out.logits)
        start = _start(model, B)
        sparse = model.step(tokens[:, 0], ctx[:, 0], start, mode="sparse")
        masked = model.step(tokens[:, 0], ctx[:, 0], start, mode="masked")
    torch.testing.assert_close(torch.stack(steps, dim=1), logits, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(sparse.state.h, masked.state.h, rtol=1e-9, atol=1e-11)


def test_gate_gets_a_gradient_and_the_router_cut_holds() -> None:
    """The gate is trained by the task; with the router's read cut (A6) nothing flows from the
    gate into the shared transform that produced ``u``."""
    cfg = _cfg(*A5, "train.gate_mode=ste", ALL, *GATE)
    model = _model(cfg)
    model.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    logits.square().mean().backward()
    gate = model.scorer.address_gate
    assert gate is not None and gate.weight.grad is not None
    assert float(gate.weight.grad.abs().sum()) > 0.0


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
        *GATE,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    gate = model.scorer.address_gate
    assert gate is not None and float(gate.weight.abs().sum()) > 0
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


def test_recipe_with_the_address_gate_trains(tmp_path: Path) -> None:
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        ALL,
        "model.cand_norm=rms",
        "model.n_cells=6",
        "model.k_select=4",
        "train.soft_until_step=5",
        "train.lr=3e-3",
        *GATE,
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
    model = trainer.model
    assert isinstance(model, HSN) and model.scorer.address_gate is not None
    assert float(model.scorer.address_gate.weight.abs().sum()) > 0.0  # it left its zero start
