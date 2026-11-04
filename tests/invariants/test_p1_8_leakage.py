"""P1.8: the prediction ``ẑ_t`` is computed from the previous summary before ``x_t`` is encoded
(plan §2.2, global rule 2): perturbing ``x_t``, ``ctx_t`` or ``x_{t+1}`` never changes it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from hsn.config import RunConfig
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from torch import Tensor

ModelFactory = Callable[..., tuple[RunConfig, HSN]]
IdsFactory = Callable[[HSN, int, int, int], tuple[Tensor, Tensor]]
PerturbFn = Callable[..., None]
StateFactory = Callable[..., CellGraphState]

T = 8


def _other(ids: Tensor, n: int) -> Tensor:
    """Ids that differ from ``ids`` at every position (the next symbol, wrapping)."""
    return (ids + 1) % n


def test_p1_8_prediction_no_leakage(
    device: torch.device,
    make_model: ModelFactory,
    ref_model: HSN,
    perturb: PerturbFn,
    ids: IdsFactory,
    random_state: StateFactory,
    expect_same: Any,
    synthetic_batch: dict[str, Tensor],
) -> None:
    """For every position ``t`` of a window: changing ``x_t`` alone or ``ctx_t`` alone leaves
    ``ẑ_0..ẑ_t`` (and the logits before ``t``) unchanged; changing ``x_{t+1}`` alone leaves
    ``ẑ_0..ẑ_{t+1}``, ``ε_0..ε_t`` and the targets ``z*_0..z*_t`` unchanged. The checks are not
    vacuous: changing ``x_t`` does change ``ε_t``, the logits at ``t`` and ``ẑ_{t+1}``. Directly,
    ``ẑ_t`` is the predictor applied to ``q_{t−1}``. The tiny model runs on random ids, the
    reference model (CPU) on a window of real kv bytes."""
    cases: list[tuple[str, HSN, dict[str, Tensor] | None]] = [
        ("tiny", make_model("tiny", device=device)[1], None)
    ]
    if device.type == "cpu":
        cases.append(("ref", ref_model, synthetic_batch))
    for name, model, batch in cases:
        perturb(model, seed=3)
        m = model.cfg
        if batch is None:
            tokens, ctx = ids(model, 3, T, seed=5)
        else:
            tokens, ctx = batch["tokens"][:, :T].to(device), batch["context"][:, :T].to(device)
        B = int(tokens.shape[0])
        other_tokens, other_ctx = _other(tokens, m.vocab_size), _other(ctx, m.n_context)
        assert (other_tokens != tokens).all() and (other_ctx != ctx).all()
        state = random_state(model, B, seed=4)
        reset = torch.zeros(B, T, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, _, info = model(tokens, ctx, state, reset)
            z_hat, z_target, eps = info["z_hat"], info["z_target"], info["eps"]
            for t in range(T):
                # x_t alone
                perturbed = tokens.clone()
                perturbed[:, t] = other_tokens[:, t]
                logits_x, _, info_x = model(perturbed, ctx, state, reset)
                expect_same(
                    info_x["z_hat"][:, : t + 1], z_hat[:, : t + 1], device, what="z_hat|x_t"
                )
                expect_same(logits_x[:, :t], logits[:, :t], device, what="logits|x_t")
                expect_same(info_x["eps"][:, :t], eps[:, :t], device, what="eps|x_t")
                expect_same(info_x["z_target"][:, :t], z_target[:, :t], device, what="z*|x_t")
                assert not torch.equal(info_x["eps"][:, t], eps[:, t]), (name, t)
                assert not torch.equal(info_x["z_target"][:, t], z_target[:, t]), (name, t)
                assert not torch.equal(logits_x[:, t], logits[:, t]), (name, t)
                if t + 1 < T:
                    assert not torch.equal(info_x["z_hat"][:, t + 1], z_hat[:, t + 1]), (name, t)
                # ctx_t alone
                perturbed_ctx = ctx.clone()
                perturbed_ctx[:, t] = other_ctx[:, t]
                logits_c, _, info_c = model(tokens, perturbed_ctx, state, reset)
                expect_same(
                    info_c["z_hat"][:, : t + 1], z_hat[:, : t + 1], device, what="z_hat|ctx_t"
                )
                expect_same(info_c["eps"][:, : t + 1], eps[:, : t + 1], device, what="eps|ctx_t")
                expect_same(logits_c[:, :t], logits[:, :t], device, what="logits|ctx_t")
                assert not torch.equal(logits_c[:, t], logits[:, t]), (name, t)
                # x_{t+1} alone
                if t + 1 < T:
                    perturbed = tokens.clone()
                    perturbed[:, t + 1] = other_tokens[:, t + 1]
                    logits_n, _, info_n = model(perturbed, ctx, state, reset)
                    expect_same(
                        info_n["z_hat"][:, : t + 2], z_hat[:, : t + 2], device, what="z_hat|x_t+1"
                    )
                    expect_same(
                        info_n["eps"][:, : t + 1], eps[:, : t + 1], device, what="eps|x_t+1"
                    )
                    expect_same(
                        info_n["z_target"][:, : t + 1],
                        z_target[:, : t + 1],
                        device,
                        what="z*|x_t+1",
                    )
                    expect_same(
                        logits_n[:, : t + 1], logits[:, : t + 1], device, what="logits|x_t+1"
                    )

            # ẑ_t is P(q_{t−1}) at every position, before x_t is encoded
            current = state
            for t in range(T):
                out = model.step(tokens[:, t], ctx[:, t], current)
                expect_same(out.z_hat, model.predictor(current.q), device, what="z_hat = P(q)")
                expect_same(out.z_hat, z_hat[:, t], device, what="z_hat step vs forward")
                current = out.state
    print(f"leakage on {device.type}: {expect_same.report()}")
