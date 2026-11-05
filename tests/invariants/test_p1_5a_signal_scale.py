"""P1.5a (amendment A1, ``docs/amendments/A1_init_norm.md``), Track T invariant: the write path
preserves signal at initialisation. On a random batch of the Appendix C reference configuration,
the RMS of the embedding, the encoder output, the unit MLP output, the write candidate, the state
and the summary each lie within 0.1x and 10x of the embedding RMS, and the gradient of an answer
logit with respect to the input 32 positions back exceeds 1e-4. The pre-amendment settings
(``trunc_normal_0.02`` on the state path, no cell normalisation, gate bias 0) violate both: that is
the G1 failure of 5 Nov 2026 (``evidence/G1/HSN_GRADIENT_DIAGNOSIS.md``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from hsn.config import RunConfig
from hsn.model.hsn import HSN
from torch import Tensor

ModelFactory = Callable[..., tuple[RunConfig, HSN]]
IdsFactory = Callable[[HSN, int, int, int], tuple[Tensor, Tensor]]

B, T, BACK = 4, 64, 32
LOW, HIGH = 0.1, 10.0
GRAD_MIN = 1e-4
OLD_INIT = ["model.state_init=trunc_normal_0.02", "model.cell_norm=none", "model.gate_bias_init=0"]
STAGES = ("z", "unit_out", "c", "h", "q")


def _rms(t: Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def _probe(model: HSN, tokens: Tensor, ctx: Tensor) -> tuple[dict[str, float], float]:
    """RMS of every stage over the window (mean over positions) and the gradient norm of the
    last position's top logit with respect to the embedding ``BACK`` positions earlier."""
    captured: list[Tensor] = []
    sums: dict[str, list[float]] = {k: [] for k in ("emb", *STAGES)}

    def emb_hook(_m: Any, _i: Any, out: Tensor, _c: list[Tensor] = captured) -> Tensor:
        out = out.clone().requires_grad_(True) if not out.requires_grad else out
        out.retain_grad()
        _c.append(out)
        sums["emb"].append(_rms(out))
        return out

    def enc_hook(_m: Any, _i: Any, out: Tensor) -> None:
        sums["z"].append(_rms(out))

    orig = model.cell.step_dense

    def step_dense(u: Tensor, eps: Tensor, state: Any, alpha: Tensor) -> Any:
        h_new, info = orig(u, eps, state, alpha)
        sums["unit_out"].append(_rms(info["unit_out"]))
        sums["c"].append(_rms(info["c"]))
        sums["h"].append(_rms(h_new))
        sums["q"].append(_rms(h_new.mean(dim=1)))
        return h_new, info

    handles = [
        model.byte_emb.register_forward_hook(emb_hook),
        model.encoder.register_forward_hook(enc_hook),
    ]
    model.cell.step_dense = step_dense  # type: ignore[method-assign]
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        reset = torch.zeros_like(tokens, dtype=torch.bool)
        reset[:, 0] = True
        logits, _state, _info = model(tokens, ctx, model.init_state(B, tokens.device), reset)
        p = T - 1
        top = logits[:, p].max(dim=-1).values.sum()
        top.backward()
        grad = captured[p - BACK].grad
        grad_norm = float(grad.norm()) if grad is not None else 0.0
    finally:
        for h in handles:
            h.remove()
        model.cell.step_dense = orig  # type: ignore[method-assign]
    assert len(captured) == T
    scales = {k: sum(v) / len(v) for k, v in sums.items()}
    return scales, grad_norm


def _ratios(scales: dict[str, float]) -> dict[str, float]:
    return {k: scales[k] / scales["emb"] for k in STAGES}


def test_p1_5a_write_path_scale_at_init(make_model: ModelFactory, ids: IdsFactory) -> None:
    """Every stage of the write path is within 0.1x and 10x of the embedding RMS at init."""
    _cfg, model = make_model("ref")
    tokens, ctx = ids(model, B, T, 0)
    scales, _ = _probe(model, tokens, ctx)
    ratios = _ratios(scales)
    bad = {k: r for k, r in ratios.items() if not (LOW <= r <= HIGH)}
    assert not bad, f"stages outside [{LOW}x, {HIGH}x] of the embedding RMS: {bad}; all: {scales}"


def test_p1_5a_gradient_reaches_32_positions_back(
    make_model: ModelFactory, ids: IdsFactory
) -> None:
    """An answer logit's gradient w.r.t. the input 32 positions back exceeds 1e-4 at init."""
    _cfg, model = make_model("ref")
    tokens, ctx = ids(model, B, T, 1)
    _, grad_norm = _probe(model, tokens, ctx)
    assert grad_norm > GRAD_MIN, grad_norm


def test_p1_5a_old_init_violates_the_invariant(make_model: ModelFactory, ids: IdsFactory) -> None:
    """The pre-amendment settings fail both checks: the write candidate, state and summary are below
    0.1x the embedding RMS and the 32-position gradient is below 1e-4 (the measured G1 failure)."""
    _cfg, model = make_model("ref", overrides=OLD_INIT)
    tokens, ctx = ids(model, B, T, 0)
    scales, grad_norm = _probe(model, tokens, ctx)
    ratios = _ratios(scales)
    assert ratios["c"] < LOW and ratios["h"] < LOW and ratios["q"] < LOW, ratios
    assert grad_norm < GRAD_MIN, grad_norm
