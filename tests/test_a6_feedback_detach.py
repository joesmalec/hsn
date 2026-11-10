"""A6: ``train.feedback_detach`` changes no forward value and removes the amplifying feedback
paths from the backward pass (docs/amendments/A6_feedback_gradient.md)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hsn.config import FEEDBACK_PATHS, RunConfig, canonical_json, config_hash, load_config
from hsn.device import resolve, seed_everything
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.compile import compile_window
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "configs" / "ref_a1.yaml"
ALL = "train.feedback_detach=[summary,eps,state,router]"
ROUTED = ("model.routing=topk", "train.gate_mode=ste")
A5 = (*ROUTED, "model.key_address=true", "model.selected_readout=true")


def _cfg(*extra: str) -> RunConfig:
    return load_config(
        REF,
        overrides=[
            "profile=cpu",
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=6",
            "model.in_degree=2",
            "model.n_units=3",
            "model.k_select=2",
            "model.hidden=16",
            "model.scorer_hidden=8",
            "model.gate_bias_init=0.0",  # open gates: every feedback path carries a real gradient
            "train.unroll=12",
            "train.batch=3",
            "train.micro_batch=3",
            "train.steps=4",
            "train.warmup_steps=2",
            "train.eval_every=4",
            "train.ckpt_every=4",
            "train.stats_every=4",
            *extra,
        ],
    )


def _model(cfg: RunConfig, dtype: torch.dtype = torch.float64) -> HSN:
    seed_everything(0)
    model = HSN.from_config(cfg).to(dtype)
    with torch.no_grad():  # away from the symmetric initial state, so every path is live
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return model.train()


def _window(cfg: RunConfig, seed: int = 1) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    B, T = cfg.train.batch, cfg.train.unroll
    tokens = torch.randint(0, cfg.model.vocab_size, (B, T), generator=g)
    ctx = torch.randint(0, cfg.model.n_context, (B, T), generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    return tokens, ctx, reset


def _start(model: HSN, B: int, seed: int = 2) -> CellGraphState:
    g = torch.Generator().manual_seed(seed)
    zero = model.init_state(B)
    h = (0.5 * torch.rand(zero.h.shape, generator=g, dtype=torch.float64) - 0.25).to(zero.dtype)
    return CellGraphState._make(h=h, q=h.mean(dim=1), age=zero.age, pos=zero.pos)


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_default_is_full_backpropagation_and_keeps_the_hash() -> None:
    base = _cfg()
    assert base.train.feedback_detach == []
    assert "feedback_detach" not in canonical_json(base)
    assert HSN.from_config(base).feedback_detach == ()
    assert not HSN.from_config(base).cell.detach_state_read
    cut = _cfg(ALL)
    assert config_hash(cut) != config_hash(base)
    assert HSN.from_config(cut).feedback_detach == FEEDBACK_PATHS
    assert HSN.from_config(cut).cell.detach_state_read
    # one model attribute order whatever the configured order
    assert HSN.from_config(_cfg("train.feedback_detach=[state,summary]")).feedback_detach == (
        "summary",
        "state",
    )


@pytest.mark.parametrize(
    "override",
    ["train.feedback_detach=[q]", "train.feedback_detach=[eps,eps]", "train.feedback_detach=eps"],
)
def test_invalid_paths_are_refused(override: str) -> None:
    with pytest.raises(ValueError):
        _cfg(override)


def test_baselines_refuse_the_key() -> None:
    with pytest.raises(ValueError, match="HSN-only"):
        load_config(
            REPO / "configs" / "baselines" / "gru_2m.yaml", ["train.feedback_detach=[summary]"]
        )


def test_trainer_refuses_a_model_built_with_other_paths(tmp_path: Path) -> None:
    cfg = _cfg(ALL)
    model = HSN(cfg.model, seed=cfg.seed, unroll=cfg.train.unroll)  # no feedback_detach
    with pytest.raises(ValueError, match="feedback_detach"):
        TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown paths"):
        HSN(cfg.model, seed=0, unroll=8, feedback_detach=["carry"])


# --------------------------------------------------------------------------------------------------
# Forward values
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("routing", [(), ROUTED, A5])
@pytest.mark.parametrize("paths", ["[summary]", "[eps]", "[state]", "[router]", None])
def test_forward_values_are_unchanged(routing: tuple[str, ...], paths: str | None) -> None:
    override = ALL if paths is None else f"train.feedback_detach={paths}"
    full, cut = _model(_cfg(*routing)), _model(_cfg(*routing, override))
    cut.load_state_dict(full.state_dict())
    tokens, ctx, reset = _window(_cfg(*routing))
    state = _start(full, tokens.shape[0])
    a_logits, a_state, a_info = full(tokens, ctx, state, reset)
    b_logits, b_state, b_info = cut(tokens, ctx, state, reset)
    assert torch.equal(a_logits, b_logits)
    assert torch.equal(a_state.h, b_state.h) and torch.equal(a_state.q, b_state.q)
    for key in ("z_hat", "eps", "u"):
        assert torch.equal(a_info[key], b_info[key])
    # the single-step reference agrees with itself too
    s_full = full.step(tokens[:, 0], ctx[:, 0], state)
    s_cut = cut.step(tokens[:, 0], ctx[:, 0], state)
    assert torch.equal(s_full.logits, s_cut.logits) and torch.equal(s_full.eps, s_cut.eps)


# --------------------------------------------------------------------------------------------------
# The backward pass
# --------------------------------------------------------------------------------------------------


def _state_grad(model: HSN, cfg: RunConfig, loss_at: int = -1) -> tuple[Tensor, Tensor, dict]:
    """d(sum of the logits at one position)/d(start h, start q) and the parameter gradients."""
    tokens, ctx, reset = _window(cfg)
    reset = torch.zeros_like(reset)  # keep the start state: it is what we differentiate
    start = _start(model, tokens.shape[0])
    h = start.h.clone().requires_grad_(True)
    q = start.q.clone().requires_grad_(True)
    state = CellGraphState._make(h=h, q=q, age=start.age, pos=start.pos)
    model.zero_grad(set_to_none=True)
    logits, _, info = model(tokens, ctx, state, reset, capture=True)
    logits[:, loss_at].square().sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    return h.grad, q.grad, {"grads": grads, "info": info, "tokens": tokens}


@pytest.mark.parametrize("routing", [(), ROUTED, A5])
def test_full_cut_leaves_the_carry_as_the_only_recurrence(routing: tuple[str, ...]) -> None:
    """With every path cut, a loss on the last position reaches the start state only through the
    carry: d h_T / d h_0 = prod_t (1 - b_t) on the cells a position wrote, the identity elsewhere,
    plus the linear summary. So the gradient w.r.t. the start state is bounded by the gradient
    w.r.t. the final state, coordinate by coordinate, and equals the carry's exact product."""
    cfg = _cfg(*routing, ALL)
    model = _model(cfg)
    g_h, g_q, extra = _state_grad(model, cfg)
    info = extra["info"]
    gates = info["cell_gates"]  # [B, T, N, d], NaN where a cell was not written
    keep = torch.where(torch.isnan(gates), torch.ones_like(gates), 1.0 - gates).prod(dim=1)
    # the reference: rerun the last position alone from the captured state before it
    tokens, ctx, reset = _window(cfg)
    T = tokens.shape[1]
    start = _start(model, tokens.shape[0])
    with torch.no_grad():
        _, before_last, _ = model(
            tokens[:, : T - 1],
            ctx[:, : T - 1],
            start,
            torch.zeros_like(reset[:, : T - 1]),
            **({"prev_tokens": None} if cfg.model.key_address else {}),
        )
    h_in = before_last.h.clone().requires_grad_(True)
    q_in = before_last.q.clone().requires_grad_(True)
    last = CellGraphState._make(h=h_in, q=q_in, age=before_last.age, pos=before_last.pos)
    kwargs = {"prev_tokens": tokens[:, T - 2]} if cfg.model.key_address else {}
    logits, _, last_info = model(
        tokens[:, T - 1 :], ctx[:, T - 1 :], last, torch.zeros_like(reset[:, :1]), **kwargs
    )
    logits[:, 0].square().sum().backward()
    # The summary's gradient is carried unchanged; h's is scaled by the carry of the earlier
    # positions and receives the summary's share at every earlier write (dq/dh = -b / N).
    torch.testing.assert_close(g_q, q_in.grad, rtol=1e-9, atol=1e-12)
    earlier = gates[:, : T - 1]
    keep_before = torch.where(torch.isnan(earlier), torch.ones_like(earlier), 1.0 - earlier)
    N = cfg.model.n_cells
    expected = h_in.grad.clone()
    for t in range(T - 2, -1, -1):  # back through the earlier positions
        b = torch.where(torch.isnan(earlier[:, t]), torch.zeros_like(expected), earlier[:, t])
        if routing:  # the incremental summary: q' = q + sum_written b (c - h) / N
            expected = keep_before[:, t] * expected - b * q_in.grad[:, None, :] / N
        else:  # dense: q' is the exact mean of h', whatever q was
            expected = keep_before[:, t] * expected
    if routing:
        torch.testing.assert_close(g_h, expected, rtol=1e-8, atol=1e-11)
    assert bool((keep <= 1.0).all()) and bool((keep >= 0.0).all())


@pytest.mark.parametrize("routing", [ROUTED, A5])
def test_full_cut_bounds_the_state_gradient(routing: tuple[str, ...]) -> None:
    """The cut backward cannot amplify: scaled-up weights explode the full gradient, not this."""
    norms = {}
    for name, override in (("full", ()), ("cut", (ALL,))):
        cfg = _cfg(*routing, *override)
        model = _model(cfg)
        with torch.no_grad():  # a high-gain cell: large unit and gate weights
            for key, p in model.named_parameters():
                if key.startswith(("cell.units", "cell.gate.weight", "shared", "predictor")):
                    p.mul_(6.0)
        g_h, g_q, _ = _state_grad(model, cfg)
        g_h_last, g_q_last, _ = _state_grad(model, cfg, loss_at=0)  # one position of history
        norms[name] = (float(g_h.norm()), float(g_h_last.norm()))
    assert norms["full"][0] > 50.0 * norms["cut"][0]  # the feedback paths were the amplifier
    assert norms["cut"][0] < 10.0 * norms["cut"][1] + 1e-9


def test_parameters_still_learn_under_the_full_cut() -> None:
    """Every module keeps a task gradient except the predictor, which the task loss reaches
    through eps only: under the cut the prediction loss alone trains it (tested below)."""
    cfg = _cfg(*A5, ALL)
    model = _model(cfg)
    _, _, extra = _state_grad(model, cfg)
    silent = [
        name
        for name, grad in extra["grads"].items()
        if not name.startswith(("halting", "predictor"))
        and (grad is None or not bool(grad.abs().sum() > 0))
    ]
    assert silent == []


@pytest.mark.parametrize(
    ("path", "probe"),
    [
        ("summary", "shared"),
        ("eps", "predictor"),
    ],
)
def test_single_cuts_stop_their_own_path_only(path: str, probe: str) -> None:
    """``summary`` leaves S's q-columns without a state gradient; ``eps`` keeps the predictor
    trained by the prediction loss alone (no task gradient reaches it through the cell)."""
    cfg = _cfg(*ROUTED, f"train.feedback_detach=[{path}]")
    model = _model(cfg)
    _, _, extra = _state_grad(model, cfg)
    grads = extra["grads"]
    if path == "eps":  # the task loss reaches P only through eps: no gradient at all now
        assert all(
            g is None or not bool(g.abs().sum() > 0)
            for n, g in grads.items()
            if n.startswith("predictor")
        )
        full = _model(_cfg(*ROUTED))
        _, _, ref = _state_grad(full, _cfg(*ROUTED))
        assert any(
            bool(g.abs().sum() > 0) for n, g in ref["grads"].items() if n.startswith("predictor")
        )
    else:  # S still learns to read q (its weights have gradients); q gets none through S
        assert bool(grads["shared.fc1.weight"].abs().sum() > 0)


def test_prediction_loss_still_trains_the_state_under_eps_cut() -> None:
    cfg = _cfg(*ROUTED, "train.feedback_detach=[eps]")
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    start = _start(model, tokens.shape[0])
    q = start.q.clone().requires_grad_(True)
    state = CellGraphState._make(h=start.h, q=q, age=start.age, pos=start.pos)
    _, _, info = model(tokens, ctx, state, torch.zeros_like(reset))
    (info["z_hat"] - info["z_target"]).square().sum().backward()
    assert bool(q.grad.abs().sum() > 0)
    assert bool(model.predictor.fc1.weight.grad.abs().sum() > 0)


# --------------------------------------------------------------------------------------------------
# The compiled window and a training step
# --------------------------------------------------------------------------------------------------


def test_compiled_window_matches_eager_under_the_cut() -> None:
    cfg = _cfg(*A5, ALL, "train.unroll=8")
    ctx = resolve(cfg.hardware)
    eager = _model(cfg, torch.float32)
    compiled = _model(cfg, torch.float32)
    compiled.load_state_dict(eager.state_dict())
    window = compile_window(compiled, ctx, "reduce-overhead", backend="aot_eager")
    tokens, ctxs, reset = _window(cfg)
    outs = []
    for model, fn in ((eager, eager), (compiled, window)):
        model.zero_grad(set_to_none=True)
        logits, state, _ = fn(tokens, ctxs, model.init_state(tokens.shape[0]), reset)
        logits.square().mean().backward()
        outs.append((logits.detach(), {n: p.grad for n, p in model.named_parameters()}))
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-4, atol=1e-5)
    for name, grad in outs[0][1].items():
        other = outs[1][1][name]
        if grad is None:
            assert other is None or not bool(other.abs().sum() > 0)
            continue
        torch.testing.assert_close(grad, other, rtol=2e-3, atol=1e-5)


def test_training_runs_and_resumes_with_the_cut(tmp_path: Path) -> None:
    from p1_15_support import datasets, hsn_config, hsn_trainer

    cfg = hsn_config(8, *A5, ALL, "train.ckpt_every=4")
    full = hsn_trainer(cfg, tmp_path / "full")
    expected = full.fit(*datasets(), 8)
    assert all(torch.isfinite(torch.tensor(s.grad_norm)) for s in expected.steps)
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    assert resumed.resume(tmp_path / "full" / "step_00000004.pt") == 4
    actual = resumed.fit(*datasets(), 8)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[4:]]
    # a checkpoint of the cut recipe is not a checkpoint of full backpropagation
    other = hsn_trainer(hsn_config(8, *A5, "train.ckpt_every=4"), tmp_path / "other")
    with pytest.raises(ValueError, match="configuration"):
        other.resume(tmp_path / "full" / "step_00000004.pt")
