"""Read-only eager diagnostics on a fixed window, independent of training decisions.

The perturbation gain is one directional finite difference over a short horizon, not a
Jacobian spectral norm or a measurement of the compiled backward. No parameter gradients,
training state, RNG advancement, or document filtering are retained.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.checkpoint import capture_rng_state, restore_rng_state


def route_sets(info: dict[str, Tensor], n_cells: int) -> Tensor | None:
    if "cell_idx" not in info:
        return None
    indices, valid = info["cell_idx"], info["valid"]
    counts = torch.zeros(*indices.shape[:2], n_cells, device=indices.device, dtype=torch.int64)
    counts.scatter_add_(2, indices.masked_fill(~valid, 0), valid.long())
    return counts > 0


class StabilityObserver:
    def __init__(self, epsilon: float, position: int, horizon: int) -> None:
        self.epsilon, self.position, self.horizon = epsilon, position, horizon
        self.previous: Tensor | None = None
        self.previous_step: int | None = None

    @torch.no_grad()
    def observe(self, model: HSN, w: dict[str, Tensor], step: int) -> dict[str, Any]:
        rng, training = capture_rng_state(), model.training
        model.eval()
        try:
            # Explicitly disable any surrounding autocast: this is a separate eager FP32 probe.
            with torch.autocast(device_type=model.device.type, enabled=False):
                return self._observe(model, w, step)
        finally:
            model.train(training)
            restore_rng_state(rng)

    def _observe(self, model: HSN, w: dict[str, Tensor], step: int) -> dict[str, Any]:
        b, length = w["tokens"].shape
        out: dict[str, Any] = {
            "stability/execution": "eager_no_autocast",
            "stability/reference_step": self.previous_step,
        }
        _, _, info = model(w["tokens"], w["context"], model.init_state(b), w["reset"])
        routes = route_sets(info, model.cfg.n_cells)
        valid = w["doc"] >= 0
        out["stability/route_churn"] = None
        if routes is not None:
            if self.previous is not None and bool(valid.any()):
                out["stability/route_churn"] = float(
                    (routes != self.previous).any(-1)[valid].float().mean()
                )
            self.previous = routes.detach().clone()
            self.previous_step = step
        start, end = self.position, min(length, self.position + self.horizon)
        out.update(
            {
                "stability/epsilon": self.epsilon,
                "stability/position": start,
                "stability/horizon": max(0, end - start),
                "stability/eligible_rows": 0,
                "stability/state_gain_max": None,
                "stability/logit_gain_max": None,
            }
        )
        if start >= length or end <= start:
            return out
        eligible = valid[:, start:end].all(1) & ~w["reset"][:, start:end].any(1)
        if not bool(eligible.any()):
            return out
        state = model.init_state(b)
        if start:
            _, state, _ = model(
                w["tokens"][:, :start], w["context"][:, :start], state, w["reset"][:, :start]
            )
        # A fixed unit direction per row avoids consuming any random draws.
        direction = torch.arange(state.h[0].numel(), device=state.h.device).remainder(2)
        direction = (direction.to(state.h.dtype) * 2 - 1).reshape_as(state.h[:1])
        direction = direction / direction.double().norm().to(state.h.dtype)
        h = (state.h + self.epsilon * direction).clamp(-1, 1)
        delta = h - state.h
        perturbed = CellGraphState(h, state.q + delta.mean(1), state.age.clone(), state.pos.clone())
        prev = w["tokens"][:, start - 1] if start else None
        args = (w["tokens"][:, start:end], w["context"][:, start:end])
        logits, reference, _ = model(*args, state, w["reset"][:, start:end], prev_tokens=prev)
        shifted, changed, _ = model(*args, perturbed, w["reset"][:, start:end], prev_tokens=prev)
        denominator = delta.double().flatten(1).norm(dim=1)
        eligible &= denominator > 0
        out["stability/eligible_rows"] = int(eligible.sum())
        if bool(eligible.any()):
            state_gain = (changed.h - reference.h).double().flatten(1).norm(dim=1)[eligible]
            logit_gain = (shifted[:, -1] - logits[:, -1]).double().norm(dim=1)[eligible]
            for label, numerator in (("state", state_gain), ("logit", logit_gain)):
                gain = numerator / denominator[eligible]
                out[f"stability/{label}_gain_max"] = (
                    float(gain.max()) if bool(torch.isfinite(gain).all()) else None
                )
                out[f"stability/{label}_gain_nonfinite"] = int((~torch.isfinite(gain)).sum())
        return out
