"""Amendment A11 (``docs/amendments/A11_position_bound.md``; opt-in): the scorer's ``pos`` feature
with a bounded position. ``model.scorer_position``: ``document`` (default, ``pos /
unroll``), ``clamp`` (``min(pos, unroll - 1) / unroll``) or ``wrap`` (``(pos mod unroll) /
unroll``). The three are equal while ``pos < unroll``."""

from __future__ import annotations

from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import config_hash, load_config, split_window_scope_error
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
KEY = "model.scorer_position"


def _later(state: CellGraphState, by: int) -> CellGraphState:
    """``state`` at a later position of its document."""
    return CellGraphState._make(h=state.h, q=state.q, age=state.age, pos=state.pos + by)


def test_position_is_opt_in_and_hashed_only_when_set() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.scorer_position == "document"
    assert config_hash(load_config(RECIPE_A6, [f"{KEY}=document"])) == config_hash(base)
    for mode in ("clamp", "wrap"):
        assert config_hash(load_config(RECIPE_A6, [f"{KEY}={mode}"])) != config_hash(base)
    with pytest.raises(ValueError):
        load_config(RECIPE_A6, [f"{KEY}=window"])


def test_split_and_fused_windows_refuse_a_bounded_position() -> None:
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    assert split_window_scope_error(load_config(RECIPE_A6, recipe)) is None
    reason = split_window_scope_error(load_config(RECIPE_A6, [*recipe, f"{KEY}=wrap"]))
    assert reason is not None and "position" in reason


@pytest.mark.parametrize("mode", ["document", "clamp", "wrap"])
def test_feature_values(mode: str) -> None:
    cfg = load_config(RECIPE_A6, [f"{KEY}={mode}"])
    scorer = HSN.from_config(cfg).scorer
    unroll = cfg.train.unroll
    pos = torch.tensor([[0, 1, unroll - 1, unroll, unroll + 5, 3 * unroll + 7]])
    ctx = torch.zeros_like(pos)
    scalars = scorer.window_scalars(
        ctx,
        pos,
        budget_frac=cfg.model.budget,
        round_idx=0,
        dtype=torch.float64,
        tokens=ctx,
        prev_tokens=ctx,
    )
    got = scalars[
        0, :, 0, scorer.feature_slices["pos"].start - scorer.feature_slices["budget"].start
    ]
    want = {
        "document": pos[0].double() / unroll,
        "clamp": pos[0].clamp(max=unroll - 1).double() / unroll,
        "wrap": (pos[0] % unroll).double() / unroll,
    }[mode]
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert float(got[:3].max()) < 1.0  # the trained range
    if mode != "document":
        assert float(got.max()) < 1.0  # never left


@pytest.mark.parametrize("mode", ["clamp", "wrap"])
def test_same_model_inside_the_window_and_bounded_past_it(mode: str) -> None:
    """A document that fits the window: the bounded model computes what the model without the
    key computes, bit for bit. Past the window the two differ, and the bounded model's scorer
    sees the same position feature as at some position inside the window."""
    cfg, bounded_cfg = _cfg(*A5), _cfg(*A5, f"{KEY}={mode}")
    model, bounded = _model(cfg).eval(), _model(bounded_cfg).eval()
    for (name, p), (_, q) in zip(model.named_parameters(), bounded.named_parameters(), strict=True):
        assert torch.equal(p, q), name
    tokens, ctx, reset = _window(cfg)
    B = tokens.shape[0]
    with torch.no_grad():
        ref, _, _ = model(tokens, ctx, _start(model, B), reset)
        got, _, _ = bounded(tokens, ctx, _start(bounded, B), reset)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)

    # the same window deep inside a long document (no reset): the position is past the unroll
    far = 5 * cfg.train.unroll + 3
    no_reset = torch.zeros_like(reset)
    with torch.no_grad():
        ref_far, _, _ = model(tokens, ctx, _later(_start(model, B), far), no_reset)
        got_far, _, _ = bounded(tokens, ctx, _later(_start(bounded, B), far), no_reset)
    assert float((got_far - ref_far).abs().max()) > 1e-6
    if mode == "wrap":  # the scorer cannot tell position `far + t` from position `3 + t`
        with torch.no_grad():
            near, _, _ = bounded(tokens, ctx, _later(_start(bounded, B), 3), no_reset)
        # the unroll of this tiny configuration is 12: the window wraps once more inside itself
        torch.testing.assert_close(got_far, near, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["clamp", "wrap"])
def test_window_matches_stepping_across_the_bound(mode: str) -> None:
    cfg = _cfg(*A5, f"{KEY}={mode}")
    model = _model(cfg).eval()
    tokens, ctx, _ = _window(cfg)
    B, T = tokens.shape
    reset = torch.zeros(B, T, dtype=torch.bool)
    start = cfg.train.unroll - 4  # the window crosses pos = unroll
    with torch.no_grad():
        logits, _, _ = model(tokens, ctx, _later(_start(model, B), start), reset)
        state, steps, prev = _later(_start(model, B), start), [], None
        for t in range(T):
            out = model.step(tokens[:, t], ctx[:, t], state, prev_tokens=prev)
            state, prev = out.state, tokens[:, t]
            steps.append(out.logits)
    torch.testing.assert_close(torch.stack(steps, dim=1), logits, rtol=1e-9, atol=1e-11)


COMPILED = {
    "ste": (("train.gate_mode=ste",), True),
    "eval": ((), False),
}


@pytest.mark.parametrize("mode", ["clamp", "wrap"])
@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager_across_the_bound(name: str, mode: str) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        f"{KEY}={mode}",
    )
    model = p28._model(cfg, training=training, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    state = _later(state, cfg.train.unroll - 2)  # the window crosses pos = unroll
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]
