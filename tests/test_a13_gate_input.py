"""Amendment A13 (``docs/amendments/A13_gate_input.md``; opt-in): what A12's address gate reads.
``model.address_gate_input``: ``u`` (default, A12 as it is), ``summary`` (the carried summary
``q_{t−1}``, which does not hold the current byte) or ``u_no_fast`` (``S``'s output with the fast
cells' read left out of its input). ``G`` keeps its shape and its zero start; only its input
changes, and with the key at ``u`` nothing changes."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import config_hash, load_config, split_window_scope_error
from hsn.model.hsn import HSN
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
GATE = ("model.address_gate=true",)
SUMMARY = (*GATE, "model.address_gate_input=summary")
FAST = ("model.k_select=4", "model.fast_cells=2", "model.fast_readout=true")
NO_FAST = (*GATE, *FAST, "model.address_gate_input=u_no_fast")
CUT = "train.feedback_detach=[summary,eps,router]"  # the state gradient's cut
VARIANTS = {"summary": SUMMARY, "u_no_fast": NO_FAST}


def _without_gate(variant: str) -> tuple[str, ...]:
    return FAST if variant == "u_no_fast" else ()


def test_gate_input_is_opt_in_hashed_only_when_set_and_scoped() -> None:
    base = load_config(RECIPE_A6, list(GATE))
    assert base.model.address_gate_input == "u"
    assert config_hash(
        load_config(RECIPE_A6, [*GATE, "model.address_gate_input=u"])
    ) == config_hash(base)
    assert config_hash(load_config(RECIPE_A6, list(SUMMARY))) != config_hash(base)
    with pytest.raises(ValueError, match="address_gate_input"):
        load_config(RECIPE_A6, ["model.address_gate_input=summary"])  # no gate to read it
    with pytest.raises(ValueError, match="u_no_fast"):
        load_config(RECIPE_A6, [*GATE, "model.address_gate_input=u_no_fast"])  # no fast read
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    assert split_window_scope_error(load_config(RECIPE_A6, recipe)) is None
    assert split_window_scope_error(load_config(RECIPE_A6, [*recipe, *SUMMARY])) is not None


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_only_the_input_changes(variant: str) -> None:
    """The parameters are A12's: the same names, shapes and initial values, the gate at zero."""
    extra = list(_without_gate(variant))
    torch.manual_seed(0)
    a12 = HSN.from_config(load_config(RECIPE_A6, [*extra, *GATE]))
    torch.manual_seed(0)
    model = HSN.from_config(load_config(RECIPE_A6, list(VARIANTS[variant])))
    ours, theirs = dict(model.named_parameters()), dict(a12.named_parameters())
    assert list(ours) == list(theirs)
    for name, value in ours.items():
        assert torch.equal(value, theirs[name]), name
    gate = model.scorer.address_gate
    assert gate is not None and bool((gate.weight == 0).all())
    assert model.scorer.gate_input == variant and a12.scorer.gate_input == "u"


def test_logits_against_a_direct_computation() -> None:
    cfg = load_config(RECIPE_A6, list(SUMMARY))
    scorer = HSN.from_config(cfg).scorer
    gate = scorer.address_gate
    assert gate is not None
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(gate.weight.shape, generator=g))
    B, N, M, d = 3, cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    A, width = 2, scorer.in_features
    core = torch.randn(B, N, width, generator=g)
    core[:, :, :d] = core[:, :1, :d]  # u is the same in every cell's row
    read = torch.randn(B, d, generator=g)
    feats = torch.cat([core, read[:, None, :].expand(B, N, d)], dim=-1)
    got = scorer(feats)
    plain = scorer.net(core)
    address = core[:, :, width - A :]
    weights = (read @ gate.weight.T).view(B, M, A)  # the address weights the gate adds
    want = plain.clone()
    for b in range(B):
        for j in range(M):
            want[b, :, j] += address[b] @ weights[b, j]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # the gate does not read u: another u moves the logits by the scorer's own map only
    other = core.clone()
    other[:, :, :d] = torch.randn(B, 1, d, generator=g)
    moved = scorer(torch.cat([other, feats[:, :, width:]], dim=-1)) - got
    torch.testing.assert_close(moved, scorer.net(other) - plain, rtol=1e-4, atol=1e-5)
    # the features without the gate's input are refused, and so is assembling without it
    with pytest.raises(ValueError, match="feats must have shape"):
        scorer(core)
    scalars = torch.zeros(B, N, width - 3 * d - 1)
    age = torch.zeros(B, N, dtype=torch.int64)
    with pytest.raises(ValueError, match="gate_in"):
        scorer._assemble(read, read, core[:, :, 2 * d : 3 * d], age, scalars)
    assembled = scorer._assemble(read, read, core[:, :, 2 * d : 3 * d], age, scalars, read)
    assert assembled.shape == (B, N, width + d)
    assert torch.equal(assembled[:, :, width:], read[:, None, :].expand(B, N, d))


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("training", [False, True])
def test_zero_gate_is_the_model_without_it(variant: str, training: bool) -> None:
    cfg = _cfg(*A5, "train.gate_mode=ste", *VARIANTS[variant])
    plain_cfg = _cfg(*A5, "train.gate_mode=ste", *_without_gate(variant))
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


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_window_matches_stepping_and_differs_from_a12(variant: str) -> None:
    cfg = _cfg(*A5, *VARIANTS[variant])
    model = _model(cfg).eval()  # the perturbed start gives a non-zero gate
    gate = model.scorer.address_gate
    assert gate is not None and float(gate.weight.detach().abs().sum()) > 0
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
    torch.testing.assert_close(torch.stack(steps, dim=1), logits, rtol=1e-9, atol=1e-11)
    # the same weights with the gate reading u (A12) give another model
    a12 = _model(_cfg(*A5, *_without_gate(variant), *GATE)).eval()
    with torch.no_grad():
        for (_, p), (_, q) in zip(a12.named_parameters(), model.named_parameters(), strict=True):
            p.copy_(q)
        other, _, _ = a12(tokens, ctx, _start(a12, B), reset)
    assert not torch.allclose(other, logits)


def _position(model: HSN, B: int = 3) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """An encoded byte, the embedded context and a start state's summary and cell states."""
    g = torch.Generator().manual_seed(7)
    state = _start(model, B)
    x = torch.randint(0, model.cfg.vocab_size, (B,), generator=g)
    z = model.encoder(model.byte_emb(x))
    ctx = model.ctx_emb(torch.zeros(B, dtype=torch.int64))
    return z, state.q, ctx, state.h


def test_summary_reads_the_carried_summary_only() -> None:
    model = _model(_cfg(*A5, CUT, *SUMMARY))
    z, q, ctx, h = _position(model)
    read = model._gate_read(z, q, ctx, h, h.dtype)
    assert read is not None and torch.equal(read, q)
    assert not read.requires_grad  # a constant in the backward with the router's cut
    assert torch.equal(model._gate_read(torch.randn_like(z), q, ctx, h, h.dtype), read)
    plain = _model(_cfg(*A5, *GATE))  # A12: nothing rides behind the features
    assert plain._gate_read(z, q, ctx, h, h.dtype) is None


def test_u_no_fast_is_s_without_the_fast_read() -> None:
    model = _model(_cfg(*A5, *NO_FAST))
    fast = model.cfg.fast_cells
    z, q, ctx, h = _position(model)
    with torch.no_grad():
        read = model._gate_read(z, q, ctx, h, h.dtype)
        want = model.shared(torch.cat([z, q, ctx, torch.zeros_like(q)], dim=-1))
        moved = h.clone()
        moved[:, :fast] += 1.0  # another state of the fast cells
        again = model._gate_read(z, q, ctx, moved, h.dtype)
        u, u_moved = (model.shared(model._shared_input(z, q, ctx, s)) for s in (h, moved))
    assert read is not None and again is not None
    torch.testing.assert_close(read, want, rtol=0, atol=0)
    assert torch.equal(again, read)  # the gate's input does not see the fast cells
    assert not torch.allclose(u, u_moved)  # u does


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_gate_gets_a_gradient_under_the_state_gradient_cut(variant: str) -> None:
    cfg = _cfg(*A5, "train.gate_mode=ste", CUT, *VARIANTS[variant])
    model = _model(cfg)
    model.set_train_step(10)
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    logits.square().mean().backward()
    gate = model.scorer.address_gate
    assert gate is not None and gate.weight.grad is not None
    assert float(gate.weight.grad.abs().sum()) > 0.0


COMPILED = {"ste": (("train.gate_mode=ste",), True), "eval": ((), False)}


@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *SUMMARY,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    gate = model.scorer.address_gate
    assert gate is not None and float(gate.weight.detach().abs().sum()) > 0
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


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_recipe_with_the_gate_input_trains(variant: str, tmp_path: Path) -> None:
    steps = 30
    sizes = (
        ("model.n_cells=8", "model.k_select=6")
        if variant == "u_no_fast"
        else (
            "model.n_cells=6",
            "model.k_select=4",
        )
    )
    extra = tuple(o for o in VARIANTS[variant] if not o.startswith("model.k_select"))
    cfg = hsn_config(
        steps,
        *A5,
        CUT,
        "model.cand_norm=rms",
        *sizes,
        "train.soft_until_step=5",
        "train.lr=3e-3",
        *extra,
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
    model = trainer.model
    assert isinstance(model, HSN) and model.scorer.address_gate is not None
    assert float(model.scorer.address_gate.weight.abs().sum()) > 0.0  # it left its zero start
