"""P1.6 tests: the prediction loss and its two targets (detached byte embedding, stop-gradient
EMA encoder), the variance floor, the ``λ_p`` terms for ``assemble_loss``, the config keys, and
finite losses and gradients on ``ref_a1`` in both modes."""

import copy
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.config import RunConfig, load_config
from hsn.losses import pred_loss, prediction_terms, variance_floor
from hsn.model.encoder import EMATarget, Encoder
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"
EXAMPLE = REPO / "configs" / "_example.yaml"

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
MODES = ("fixed_embedding", "ema_encoder")
EMA = "model.predictor_target=ema_encoder"


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    return torch.device(device_type)


def _model(path: Path, overrides: Sequence[str] = ()) -> tuple[RunConfig, HSN]:
    cfg = load_config(path, overrides=list(overrides))
    torch.manual_seed(cfg.seed)
    model = HSN.from_config(cfg)
    if cfg.train.precision == "fp64":
        model = model.double()
    return cfg, model


def _ids(cfg: RunConfig, B: int, T: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m = cfg.model
    tokens = torch.randint(0, m.vocab_size, (B, T), generator=generator)
    ctx = torch.randint(0, m.n_context, (B, T), generator=generator)
    return tokens, ctx


def _warm_state(model: HSN, tokens: Tensor, ctx: Tensor, steps: int) -> CellGraphState:
    """A non-trivial detached state: ``steps`` positions processed without gradients."""
    state = model.init_state(tokens.shape[0])
    with torch.no_grad():
        for t in range(steps):
            state = model.step(tokens[:, t], ctx[:, t], state).state
    return state.detach()


def _perturb(module: torch.nn.Module, seed: int, scale: float = 0.5) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
            p.add_(noise.to(dtype=p.dtype, device=p.device))


def _target_tensors(target: EMATarget) -> dict[str, Tensor]:
    return dict(target.module.named_buffers())


def _distance(target: EMATarget, online: torch.nn.Module) -> float:
    tensors = _target_tensors(target)
    return math.sqrt(
        sum(
            float((tensors[name] - p.detach()).square().sum())
            for name, p in online.named_parameters()
        )
    )


# --------------------------------------------------------------------------------------------------
# The fixed-embedding target
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [TINY, REF_A1], ids=["tiny", "ref_a1"])
def test_p1_6_fixed_target_is_detached(path: Path) -> None:
    cfg, model = _model(path)
    assert cfg.model.predictor_target == "fixed_embedding"
    assert model.ema_target is None and model.ema_tau == cfg.train.ema_tau
    B, d = cfg.train.batch, cfg.model.d
    tokens, ctx = _ids(cfg, B, 4, seed=1)
    state = _warm_state(model, tokens, ctx, 2)
    x_t, c_t = tokens[:, 2], ctx[:, 2]
    out = model.step(x_t, c_t, state)

    # the target is the byte embedding of x_t, detached, in the parameters' dtype
    assert out.z_target.shape == (B, d) and out.z_target.dtype == model.dtype
    assert not out.z_target.requires_grad and out.z_target.grad_fn is None
    assert torch.equal(out.z_target, model.byte_emb.weight[x_t].detach())
    assert torch.equal(out.z_target, model.byte_emb(x_t).detach())
    # the prediction comes from q_{t-1} alone and the encoding keeps its graph
    assert out.z_hat.requires_grad and out.z.requires_grad
    assert torch.equal(out.z_hat, model.predictor(state.q))
    assert torch.equal(out.eps, out.z - out.z_hat)

    # the prediction loss trains the predictor only: nothing flows into the embedding table
    # (the target) or the encoder (not on the path)
    loss = pred_loss(out.z_hat, out.z_target)
    assert loss.ndim == 0 and loss.dtype == model.dtype and torch.isfinite(loss)
    expected = ((out.z_hat - out.z_target) ** 2).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.byte_emb.weight.grad is None
    assert all(p.grad is None for p in model.encoder.parameters())
    for name, p in model.predictor.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in model.cell.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())

    # through a window: the stack is [B, T, d], detached, and equal to the embedding lookup
    model.zero_grad(set_to_none=True)
    T = tokens.shape[1]
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 1] = True
    _, _, info = model(tokens, ctx, model.init_state(B), reset)
    assert info["z_target"].shape == (B, T, d) and not info["z_target"].requires_grad
    assert torch.equal(info["z_target"], model.byte_emb.weight[tokens].detach())
    assert info["z_hat"].requires_grad and info["z"].requires_grad

    # errors over silence: an attached target, a shape mismatch, a non-float tensor
    with pytest.raises(ValueError, match="requires grad"):
        pred_loss(out.z_hat, out.z)
    with pytest.raises(ValueError, match="shape"):
        pred_loss(out.z_hat, out.z_target[:, : d - 1])
    with pytest.raises(TypeError):
        pred_loss(out.z_hat, x_t)
    with pytest.raises(TypeError):
        pred_loss(out.z_hat, out.z_target.tolist())  # type: ignore[arg-type]


@pytest.mark.parametrize("path", [TINY, REF_A1], ids=["tiny", "ref_a1"])
def test_p1_6_ema_target_is_detached_and_tracks_x_t(path: Path) -> None:
    cfg, model = _model(path, [EMA])
    assert isinstance(model.ema_target, EMATarget) and model.ema_target.tau == cfg.train.ema_tau
    assert f"ema_tau={cfg.train.ema_tau}" in repr(model)
    B, d = cfg.train.batch, cfg.model.d
    tokens, ctx = _ids(cfg, B, 4, seed=2)
    state = _warm_state(model, tokens, ctx, 2)
    x_t, c_t = tokens[:, 2], ctx[:, 2]
    out = model.step(x_t, c_t, state)

    # the target is the EMA copy applied to the byte embedding: detached, and, before any update,
    # equal to what the online encoder computes (the copy starts equal)
    assert out.z_target.shape == (B, d) and out.z_target.dtype == model.dtype
    assert not out.z_target.requires_grad and out.z_target.grad_fn is None
    emb = model.byte_emb(x_t).detach()
    assert torch.equal(out.z_target, model.ema_target(emb))
    assert torch.equal(out.z_target, model.ema_target.module(emb))
    assert torch.equal(out.z_target, model.encoder(emb).detach())
    assert torch.equal(out.z_hat, model.predictor(state.q))  # q_{t-1} only

    # the prediction loss alone reaches the predictor and nothing else
    pred_loss(out.z_hat, out.z_target).backward()
    assert model.byte_emb.weight.grad is None
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())
    assert list(model.ema_target.parameters()) == []
    assert all(b.grad is None for b in model.ema_target.buffers())

    # once the online encoder moves and the target is updated, the two differ (stop-gradient
    # target, not a live copy) and the window stack follows the target per position
    _perturb(model.encoder, seed=3)
    model.update_target()
    T = tokens.shape[1]
    reset = torch.zeros(B, T, dtype=torch.bool)
    _, _, info = model(tokens, ctx, model.init_state(B), reset)
    assert info["z_target"].shape == (B, T, d) and not info["z_target"].requires_grad
    with torch.no_grad():
        for t in range(T):
            expected = model.ema_target(model.byte_emb(tokens[:, t]))
            assert torch.equal(info["z_target"][:, t], expected)
            assert not torch.equal(
                info["z_target"][:, t], model.encoder(model.byte_emb(tokens[:, t]))
            )


# --------------------------------------------------------------------------------------------------
# EMATarget
# --------------------------------------------------------------------------------------------------


def test_p1_6_ema_moves_toward_online() -> None:
    cfg = load_config(TINY)
    torch.manual_seed(cfg.seed)
    online = Encoder(cfg.model).double()
    tau = 0.9
    target = EMATarget(online, tau)
    assert target.tau == tau and "tau=0.9" in repr(target)

    # a deep copy that starts equal; frozen buffers, no parameters, counted separately
    names = [name for name, _ in online.named_parameters()]
    tensors = _target_tensors(target)
    assert sorted(tensors) == sorted(names)
    for name, p in online.named_parameters():
        assert torch.equal(tensors[name], p.detach()) and tensors[name] is not p
        assert not tensors[name].requires_grad
    assert list(target.parameters()) == []
    assert target.param_count() == online.param_count() == 76
    assert set(target.state_dict()) == {f"module.{name}" for name in names}

    # move the online encoder: one update is exactly tau * old + (1 - tau) * online, closer
    _perturb(online, seed=4, scale=1.0)
    before = {name: t.clone() for name, t in tensors.items()}
    online_before = {name: p.detach().clone() for name, p in online.named_parameters()}
    dist_before = _distance(target, online)
    assert dist_before > 1.0
    target.update()
    after = _target_tensors(target)
    for name, p in online.named_parameters():
        expected = tau * before[name] + (1.0 - tau) * p.detach()
        torch.testing.assert_close(after[name], expected, rtol=0.0, atol=1e-12)
        assert torch.equal(p.detach(), online_before[name])  # the online module is untouched
    dist_after = _distance(target, online)
    assert dist_after < dist_before
    torch.testing.assert_close(dist_after, tau * dist_before, rtol=1e-9, atol=0.0)
    # ... and converges geometrically
    for _ in range(199):
        target.update()
    assert _distance(target, online) < 1e-8 * dist_before

    # forward runs under no_grad: the output is the copy's, without a graph, and the online
    # module's parameters are not part of it
    x = torch.randn(3, cfg.model.d, dtype=torch.float64, requires_grad=True)
    y = target(x)
    assert y.shape == (3, cfg.model.d) and not y.requires_grad and y.grad_fn is None
    assert torch.equal(y, target.module(x).detach())
    assert x.grad is None

    # tau at the ends: 1 freezes the target, 0 copies the online module
    frozen = EMATarget(online, 1.0)
    _perturb(online, seed=5)
    snapshot = {name: t.clone() for name, t in _target_tensors(frozen).items()}
    frozen.update()
    assert all(torch.equal(t, snapshot[name]) for name, t in _target_tensors(frozen).items())
    live = EMATarget(online, 0.0)
    _perturb(online, seed=6)
    live.update()
    assert all(
        torch.equal(t, p.detach())
        for (_, t), p in zip(_target_tensors(live).items(), online.parameters(), strict=True)
    )
    # state dict round trip of the target alone
    fresh = EMATarget(Encoder(cfg.model).double(), tau)
    fresh.load_state_dict(target.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.buffers(), target.buffers(), strict=True))

    # refusals
    for bad in (-0.1, 1.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="ema_tau"):
            EMATarget(online, bad)
    with pytest.raises(TypeError):
        EMATarget(online, "0.5")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EMATarget(online, True)
    with pytest.raises(TypeError):
        EMATarget(online.state_dict(), tau)  # type: ignore[arg-type]
    other = EMATarget(online, tau)
    object.__setattr__(other, "_online", Encoder(cfg.model).double())  # same layout: fine
    other.update()
    object.__setattr__(other, "_online", torch.nn.Linear(cfg.model.d, cfg.model.d).double())
    with pytest.raises(RuntimeError, match="lacks"):
        other.update()

    # the HSN module: the copy exists only in EMA mode, update_target moves it there and is a
    # no-op otherwise; ema_tau comes from train.ema_tau and is required in EMA mode
    cfg_f, fixed = _model(TINY)
    assert fixed.ema_target is None
    weights = {name: p.detach().clone() for name, p in fixed.named_parameters()}
    fixed.update_target()
    assert all(torch.equal(p, weights[name]) for name, p in fixed.named_parameters())
    cfg_e, ema = _model(TINY, [EMA, "train.ema_tau=0.5"])
    assert ema.ema_target is not None and ema.ema_target.tau == 0.5 == cfg_e.train.ema_tau
    assert ema.ema_target._online is ema.encoder
    _perturb(ema.encoder, seed=7)
    d0 = _distance(ema.ema_target, ema.encoder)
    ema.update_target()
    torch.testing.assert_close(
        _distance(ema.ema_target, ema.encoder), 0.5 * d0, rtol=1e-9, atol=0.0
    )
    with pytest.raises(ValueError, match="train.ema_tau"):
        HSN(cfg_e.model, seed=0, unroll=8)
    assert HSN(cfg_f.model, seed=0, unroll=8).ema_target is None  # ignored in fixed mode
    assert HSN(cfg_f.model, seed=0, unroll=8, ema_tau=0.3).ema_target is None

    # parameter accounting is unchanged by the frozen copy; its size is reported on its own
    cfg_r, ref = _model(REF_A1, [EMA])
    report = ref.param_report()
    assert report["total"] == 1_820_361 and report["total_trainable"] == 1_820_489
    assert report["total_trainable"] == sum(p.numel() for p in ref.parameters())
    assert all(p.requires_grad for p in ref.parameters())
    assert (
        ref.ema_target is not None and ref.ema_target.param_count() == 66_112 == report["encoder"]
    )
    # a checkpoint carries the target and restores it; a deep copy keeps the online reference
    _perturb(ref.encoder, seed=8)
    ref.update_target()
    sd = ref.state_dict()
    assert "ema_target.module.fc1.weight" in sd and "ema_target.module.fc2.bias" in sd
    _, again = _model(REF_A1, [EMA])
    again.load_state_dict(sd, strict=True)
    assert again.ema_target is not None
    for (name, a), (_, b) in zip(
        again.ema_target.named_buffers(), ref.ema_target.named_buffers(), strict=True
    ):
        assert torch.equal(a, b), name
    cloned = copy.deepcopy(ref)
    assert cloned.ema_target is not None and cloned.ema_target._online is cloned.encoder


# --------------------------------------------------------------------------------------------------
# The variance floor and the lambda_p terms
# --------------------------------------------------------------------------------------------------


def test_p1_6_variance_floor_penalises_collapse() -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    B, d, gamma, eps = 16, 8, 1.0, 1e-4
    # a collapsed batch (identical rows) pays the full floor
    collapsed = torch.full((B, d), 0.3, dtype=torch.float64)
    floor = variance_floor(collapsed, gamma, eps)
    assert floor.ndim == 0 and floor.dtype == torch.float64
    torch.testing.assert_close(floor, torch.tensor(gamma - math.sqrt(eps), dtype=torch.float64))
    # rows whose per-dimension standard deviation reaches gamma pay nothing
    raw = torch.randn(B, d, generator=generator, dtype=torch.float64)
    spread = (raw - raw.mean(dim=0)) / raw.std(dim=0, correction=1) * 1.5  # std exactly 1.5
    assert variance_floor(spread, gamma, eps).item() == 0.0
    assert variance_floor(spread, gamma=1.5 - 1e-3, eps=1e-12).item() == 0.0
    assert variance_floor(spread, gamma=2.0, eps=1e-12).item() > 0.0
    # the default arguments are the chunk's (gamma 1.0, eps 1e-4)
    assert torch.equal(variance_floor(collapsed), variance_floor(collapsed, 1.0, 1e-4))
    # one dimension collapsed out of d costs (gamma - sqrt(eps)) / d
    partial = spread.clone()
    partial[:, 0] = 0.0
    torch.testing.assert_close(
        variance_floor(partial, gamma, eps),
        torch.tensor((gamma - math.sqrt(eps)) / d, dtype=torch.float64),
    )
    # its gradient pushes a nearly collapsed batch apart
    z = (
        collapsed + 1e-3 * torch.randn(B, d, generator=generator, dtype=torch.float64)
    ).requires_grad_()
    loss = variance_floor(z, gamma, eps)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    stepped = (z - 0.1 * z.grad).detach()
    assert (stepped.std(dim=0) > z.detach().std(dim=0)).all()
    assert variance_floor(stepped, gamma, eps) < loss.detach()
    # gamma = 0 disables the term
    assert variance_floor(collapsed, gamma=0.0).item() == 0.0
    # refusals: one row (an unbiased variance of nothing), wrong rank, bad gamma / eps / dtype
    with pytest.raises(ValueError, match="at least 2 rows"):
        variance_floor(collapsed[:1])
    with pytest.raises(ValueError, match=r"\[B, d\]"):
        variance_floor(collapsed[None])
    with pytest.raises(ValueError, match="gamma"):
        variance_floor(collapsed, gamma=-1.0)
    with pytest.raises(ValueError, match="eps"):
        variance_floor(collapsed, eps=0.0)
    with pytest.raises(TypeError):
        variance_floor(torch.zeros(B, d, dtype=torch.int64))

    # prediction_terms: the fixed target gives the prediction term only; the EMA target adds the
    # floor; both scale with lambda_p; windows flatten; the mask selects positions
    z_hat = torch.randn(B, d, generator=generator, dtype=torch.float64, requires_grad=True)
    z_target = torch.randn(B, d, generator=generator, dtype=torch.float64)
    fixed = prediction_terms(
        z_hat,
        z_target,
        collapsed,
        lambda_p=1.0,
        predictor_target="fixed_embedding",
        variance_floor_gamma=gamma,
    )
    assert list(fixed) == ["pred"]
    torch.testing.assert_close(fixed["pred"], pred_loss(z_hat, z_target))
    ema = prediction_terms(
        z_hat,
        z_target,
        collapsed,
        lambda_p=2.0,
        predictor_target="ema_encoder",
        variance_floor_gamma=gamma,
    )
    assert list(ema) == ["pred", "variance_floor"]
    torch.testing.assert_close(ema["pred"], 2.0 * pred_loss(z_hat, z_target))
    torch.testing.assert_close(ema["variance_floor"], 2.0 * variance_floor(collapsed, gamma))
    assert ema["pred"].requires_grad
    off = prediction_terms(
        z_hat,
        z_target,
        collapsed,
        lambda_p=1.0,
        predictor_target="ema_encoder",
        variance_floor_gamma=0.0,
    )
    assert off["variance_floor"].item() == 0.0
    # a window [B, T, d] is every position as a sample
    T = 3
    w_hat = torch.randn(2, T, d, generator=generator, dtype=torch.float64)
    w_target = torch.randn(2, T, d, generator=generator, dtype=torch.float64)
    w_z = torch.randn(2, T, d, generator=generator, dtype=torch.float64)
    window = prediction_terms(
        w_hat,
        w_target,
        w_z,
        lambda_p=1.0,
        predictor_target="ema_encoder",
        variance_floor_gamma=gamma,
    )
    torch.testing.assert_close(
        window["pred"], pred_loss(w_hat.reshape(-1, d), w_target.reshape(-1, d))
    )
    torch.testing.assert_close(window["variance_floor"], variance_floor(w_z.reshape(-1, d), gamma))
    mask = torch.tensor([[True, False, True], [False, True, True]])
    masked = prediction_terms(
        w_hat,
        w_target,
        w_z,
        lambda_p=1.0,
        predictor_target="ema_encoder",
        variance_floor_gamma=gamma,
        mask=mask,
    )
    torch.testing.assert_close(masked["pred"], pred_loss(w_hat[mask], w_target[mask]))
    torch.testing.assert_close(masked["variance_floor"], variance_floor(w_z[mask], gamma))
    # refusals
    with pytest.raises(ValueError, match="selects no position"):
        prediction_terms(
            w_hat,
            w_target,
            w_z,
            lambda_p=1.0,
            predictor_target="ema_encoder",
            variance_floor_gamma=gamma,
            mask=torch.zeros(2, T, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="mask"):
        prediction_terms(
            w_hat,
            w_target,
            w_z,
            lambda_p=1.0,
            predictor_target="ema_encoder",
            variance_floor_gamma=gamma,
            mask=mask[:, :2],
        )
    with pytest.raises(ValueError, match="predictor_target"):
        prediction_terms(
            z_hat,
            z_target,
            collapsed,
            lambda_p=1.0,
            predictor_target="contrastive",  # type: ignore[arg-type]
            variance_floor_gamma=gamma,
        )
    with pytest.raises(ValueError, match="lambda_p"):
        prediction_terms(
            z_hat,
            z_target,
            collapsed,
            lambda_p=-1.0,
            predictor_target="fixed_embedding",
            variance_floor_gamma=gamma,
        )
    with pytest.raises(ValueError, match="shape"):
        prediction_terms(
            z_hat,
            z_target,
            collapsed[:, :-1],
            lambda_p=1.0,
            predictor_target="fixed_embedding",
            variance_floor_gamma=gamma,
        )
    with pytest.raises(ValueError, match="requires grad"):
        prediction_terms(
            z_hat,
            z_hat,
            collapsed,
            lambda_p=1.0,
            predictor_target="fixed_embedding",
            variance_floor_gamma=gamma,
        )


# --------------------------------------------------------------------------------------------------
# ref_a1 in both modes, and the config keys
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_p1_6_loss_finite_on_ref_a1(mode: str) -> None:
    cfg, model = _model(REF_A1, [f"model.predictor_target={mode}"])
    assert cfg.model.predictor_target == mode
    B, T, V = 4, 6, cfg.model.vocab_size
    tokens, ctx = _ids(cfg, B, T + 1, seed=9)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[1, 2] = True
    state = model.init_state(B)

    def objective() -> tuple[Tensor, dict[str, Tensor]]:
        logits, _, info = model(tokens[:, :T], ctx[:, :T], state, reset)
        task = F.cross_entropy(logits.reshape(-1, V), tokens[:, 1:].reshape(-1))
        terms = prediction_terms(
            info["z_hat"],
            info["z_target"],
            info["z"],
            lambda_p=cfg.train.lambda_p,
            predictor_target=cfg.model.predictor_target,
            variance_floor_gamma=cfg.train.variance_floor_gamma,
        )
        total = task + sum(terms.values())
        return total, {"task": task, **terms}

    total, parts = objective()
    expected_terms = {"task", "pred"} | ({"variance_floor"} if mode == "ema_encoder" else set())
    assert set(parts) == expected_terms
    assert torch.isfinite(total) and all(torch.isfinite(v) for v in parts.values())
    assert parts["pred"].item() > 0.0
    total.backward()
    for name, p in model.named_parameters():
        if name.startswith("halting."):
            continue  # not called while r_max = 1
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if mode == "ema_encoder":
        assert model.ema_target is not None
        assert all(b.grad is None for b in model.ema_target.buffers())
        assert parts["variance_floor"].item() > 0.0  # a fresh encoder has std << gamma
    else:
        assert model.ema_target is None

    # one optimiser step, then the EMA update, then the objective again: still finite
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = (
        {name: b.clone() for name, b in model.ema_target.named_buffers()}
        if model.ema_target is not None
        else {}
    )
    optimizer.step()
    model.update_target()
    if model.ema_target is not None:
        assert all(not torch.equal(b, before[name]) for name, b in model.ema_target.named_buffers())
    model.zero_grad(set_to_none=True)
    total_2, parts_2 = objective()
    assert torch.isfinite(total_2) and all(torch.isfinite(v) for v in parts_2.values())


def test_p1_6_config_keys_validate() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.train.ema_tau == 0.99 and cfg.train.variance_floor_gamma == 1.0
    assert cfg.model.predictor_target == "fixed_embedding"
    for override in ("train.ema_tau=0", "train.ema_tau=1", "train.variance_floor_gamma=0"):
        load_config(EXAMPLE, overrides=[override])
    for override in (
        "train.ema_tau=1.5",
        "train.ema_tau=-0.1",
        "train.variance_floor_gamma=-1",
        "model.predictor_target=contrastive",
    ):
        with pytest.raises(ValueError):
            load_config(EXAMPLE, overrides=[override])
    ema = load_config(
        REF_A1, overrides=[EMA, "train.ema_tau=0.996", "train.variance_floor_gamma=0.5"]
    )
    assert ema.model.predictor_target == "ema_encoder"
    assert ema.train.ema_tau == 0.996 and ema.train.variance_floor_gamma == 0.5


@pytest.mark.parametrize("device_type", [cuda_param, mps_param])
@pytest.mark.parametrize("mode", MODES)
def test_p1_6_runs_on_device(device_type: str, mode: str) -> None:
    device = _device(device_type)
    cfg, model = _model(REF_A1, [f"model.predictor_target={mode}"])
    B, T = 2, 3
    tokens, ctx = _ids(cfg, B, T, seed=10)
    reset = torch.zeros(B, T, dtype=torch.bool)
    _, _, info_cpu = model(tokens, ctx, model.init_state(B), reset)
    on_device = model.to(device)
    _, _, info = on_device(
        tokens.to(device), ctx.to(device), on_device.init_state(B), reset.to(device)
    )
    assert info["z_target"].device.type == device.type and not info["z_target"].requires_grad
    torch.testing.assert_close(info["z_target"].cpu(), info_cpu["z_target"], atol=1e-4, rtol=1e-4)
    terms = prediction_terms(
        info["z_hat"],
        info["z_target"],
        info["z"],
        lambda_p=cfg.train.lambda_p,
        predictor_target=cfg.model.predictor_target,
        variance_floor_gamma=cfg.train.variance_floor_gamma,
    )
    total = sum(terms.values())
    assert isinstance(total, Tensor) and total.device.type == device.type and torch.isfinite(total)
    total.backward()
    _perturb(on_device.encoder, seed=11)
    on_device.update_target()
    if on_device.ema_target is not None:
        assert all(b.device.type == device.type for b in on_device.ema_target.buffers())
