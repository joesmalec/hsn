"""Amendment A11 (``docs/amendments/A11_position_bound.md``; opt-in): the scorer's ``pos`` feature
with a bounded position. ``model.scorer_position``: ``document`` (default, ``pos /
unroll``), ``clamp`` (``min(pos, unroll - 1) / unroll``) or ``wrap`` (``(pos mod unroll) /
unroll``). The three are equal while ``pos < unroll``.

The split and fused windows (P2.10, P2.10c) take the key: the feature is one of the scorer's
scalars, which the window's prologue computes and the sweep and the replay read as they are.
Here, on the CPU in float64, the split window against the autograd window on a window that
crosses ``pos = unroll``, under the full cut and with the state gradient; the compiled sweeps'
tapes are compared on the card (``tests/kernels/test_p2_10c_window.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import config_hash, load_config, split_window_scope_error
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.split import SplitWindow
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, ALL, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_p2_10c_state_gradient import (
    BASE,
    PINNED,
    _assert_same_grads,
    _objective,
    _one_gate_sigmoid,
)

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


def test_split_and_fused_windows_take_a_bounded_position() -> None:
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    cut = "train.feedback_detach=[summary,eps,router]"
    for mode in ("document", "clamp", "wrap"):
        for extra in ((), (cut,)):  # the full cut (the recipe's), and the state gradient
            for window in ("split", "fused"):
                cfg = load_config(RECIPE_A6, [*recipe, *extra, f"{KEY}={mode}"])
                assert split_window_scope_error(cfg) is None, (mode, extra)
                # and at load, where hardware.window is checked against the scope
                load_config(
                    RECIPE_A6, [*recipe, *extra, f"{KEY}={mode}", f"hardware.window={window}"]
                )


#: the full cut (P2.10's window) and the state gradient (P2.10c's), the latter also with A8's
#: fast cells at fixed units and their read
WINDOW_CUTS = {
    "full_cut": (*A5, ALL, "model.cand_norm=rms"),
    "state_gradient": BASE,
    "state_gradient_fast_cells": (*BASE, *PINNED),
}


def _across_the_bound(cfg: object, mode: str) -> tuple:
    """A window of the split window's tests whose rows cross ``pos = unroll``: every row
    starts four positions before it, row 1 resets at position 5 (and is back inside the range
    from there on), row 2 resets at position 0 (and never leaves it). Returns the autograd
    window's and the split window's (logits, final state, info, gradients by name)."""
    model = _model(cfg)  # type: ignore[arg-type]
    tokens, ctx, _ = _window(cfg)  # type: ignore[arg-type]
    B, T = tokens.shape
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[1, 5] = True
    reset[2, 0] = True
    state = _later(_start(model, B), cfg.train.unroll - 4)  # type: ignore[attr-defined]
    mask = torch.zeros_like(reset)
    mask[:, 3::4] = True
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    out = []
    for call in (model, SplitWindow(model, cfg, None)):  # type: ignore[arg-type]
        # one sigmoid for the router's float32 gate in both (torch's float32 kernel is not the
        # same function at the two tensor sizes: test_p2_10c_state_gradient._one_gate_sigmoid)
        with _one_gate_sigmoid():
            logits, final, info = call(tokens, ctx, state, reset)
            grads = torch.autograd.grad(_objective(logits, info, mask), params, allow_unused=True)
        out.append((logits.detach(), final, info, dict(zip(names, grads, strict=True))))
    return tuple(out)


@pytest.mark.parametrize("mode", ["clamp", "wrap"])
@pytest.mark.parametrize("cut", list(WINDOW_CUTS))
def test_split_window_equals_autograd_across_the_bound(cut: str, mode: str) -> None:
    """The split window with a bounded position, on a window that crosses ``pos = unroll``:
    the autograd window's decisions, logits, final state and parameter gradients (float64),
    under the full cut and with the state gradient. And the window is sensitive to the key:
    against the document's position the gradients differ, a row that crosses the bound gets
    other logits, and a row that stays inside the range the same ones, bit for bit."""
    cfg = _cfg(*WINDOW_CUTS[cut], f"{KEY}={mode}")
    (logits, final, info, ref), (s_logits, s_final, s_info, got) = _across_the_bound(cfg, mode)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], info[key]), key
    torch.testing.assert_close(s_logits, logits, rtol=1e-10, atol=1e-12)
    for field in ("h", "q"):
        torch.testing.assert_close(
            getattr(s_final, field), getattr(final, field), rtol=1e-10, atol=1e-12
        )
    assert torch.equal(s_final.age, final.age) and torch.equal(s_final.pos, final.pos)
    assert int(final.pos[0]) == 2 * cfg.train.unroll - 4  # the counter itself is not bounded
    _assert_same_grads(ref, got)
    # against the document's position, in the split window
    plain = _cfg(*WINDOW_CUTS[cut])
    _, (d_logits, _, _, d_got) = _across_the_bound(plain, "document")
    if "fast_cells" not in cut:  # (there the decoder reads the fast cells, which always run)
        assert float((s_logits[0] - d_logits[0]).abs().max()) > 1e-6  # crossed the bound
    assert torch.equal(s_logits[2], d_logits[2])  # reset at 0: never left the range
    apart = max(
        float((got[n] - g).abs().max() / g.abs().max())
        for n, g in d_got.items()
        if g is not None and float(g.abs().max()) > 0
    )
    assert apart > 1e-6


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
