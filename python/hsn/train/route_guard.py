"""``train.route_guard``: act when an established minority route is being taken over (A6, opt-in).

**The event.** In a routed HSN one router unit usually serves almost every position and a few
units serve one kind of position each (on the delayed task: the positions where a fact's value is
written, which is what binds a value to its key). The dominant unit's scores keep growing with
training; when they overtake a specialised unit's at its positions, top-K hands those positions to
the dominant unit, the specialised unit stops receiving a gradient there and cannot win them back,
and the accuracy the route carried is lost (exps 375-377: 10 to 21 points; the binding overlap
falls from 0.44-0.51 to 0.16-0.19). The takeover is abrupt for a position and not tied to a step
count: it happened at steps 2,000 to 4,000 at one setting and at 2,700 or 3,500 at others.

**The measurement** needs no knowledge of the task and no extra forward pass. From every training
window the guard counts, per input byte ``b`` and unit ``u``, the selection slots ``u`` won at
positions whose input is ``b``. Every ``interval`` applied steps it reads the counts of the
interval::

    dominant      = the unit with the most slots
    minority[b]   = share of byte b's slots that are not the dominant unit's
    established   = bytes whose reference minority share is at least ``specialised``
    lost          = sum_b n[b] * max(0, reference[b] - minority[b]) / sum_b n[b] * reference[b]
                    over the established bytes

``reference[b]`` is the largest minority share byte ``b`` has had since the last event, and a
byte is established once its minority share has been at least ``specialised`` at ``persistence``
consecutive readings. A change of the dominant unit starts a new reference.

At the start of a run the dominant unit absorbs most positions while the loss falls: that is
learning. So a reading is an event only if training is no longer improving: the interval's mean
training loss is above the lowest interval mean so far by ``loss_rise``.

**The response** is on the unit that is taking the positions over. The dominant unit's weight
on the current byte's address grows at a steady pace for a whole run, whatever the loss is (exp
389: 0.20 to 0.62), while a specialised unit stops growing once its route is learned; a learning
rate cut does not touch this, because the router's proposed step is about ten times its update
cap. So when ``lost >= lost_fraction`` the guard multiplies the *applied* step of the dominant
unit's rows of the scorer's output layer by ``reduce_factor`` (never below ``min_scale``): from
then on every update of those rows is ``old + scale * (new - old)``, after the optimiser step and
the caps. The other units keep their full step, so an established route can go on strengthening.
With ``restore`` the dominant unit's rows are also put back to their values at the last check at
which less than half of ``lost_fraction`` was lost, which undoes the takeover instead of only
slowing the next one (every other weight is kept: a rejected step of one router row, not a
rollback of the run). The reference then restarts from the current usage (or stays, with
``restore``).

Nothing is skipped and no document is dropped; a run in which no event fires trains exactly as
without the guard. Every event is recorded (:meth:`RouteGuard.report`) and the state is part of
the checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import RouteGuardConfig

__all__ = ["RouteGuard", "lost_usage"]


def lost_usage(
    counts: Tensor, reference: Tensor, specialised: float, settled: Tensor | None = None
) -> tuple[float, float, Tensor, int]:
    """``(lost, usage, minority, dominant)`` of one interval's ``counts [V, U]`` against
    ``reference [V]`` (module docstring). ``usage`` is the minority share of all slots; ``lost``
    is 0 when no byte is established. ``settled`` (bool ``[V]``) restricts the established bytes
    to those whose minority share has persisted."""
    counts = counts.to(torch.float64)
    per_byte = counts.sum(dim=1)
    total = float(per_byte.sum())
    dominant = int(counts.sum(dim=0).argmax())
    minority = torch.where(
        per_byte > 0,
        1.0 - counts[:, dominant] / per_byte.clamp_min(1.0),
        torch.zeros_like(per_byte),
    )
    usage = float((per_byte * minority).sum() / total) if total > 0 else 0.0
    established = (reference >= specialised) & (per_byte > 0)
    if settled is not None:
        established = established & settled
    base = float((per_byte * reference)[established].sum())
    if base <= 0.0:
        return 0.0, usage, minority, dominant
    gone = float((per_byte * (reference - minority).clamp_min(0.0))[established].sum())
    return gone / base, usage, minority, dominant


class RouteGuard:
    """The route-takeover guard of one trainer (module docstring)."""

    def __init__(
        self, router: nn.Module, cfg: RouteGuardConfig, *, n_units: int, vocab: int
    ) -> None:
        self.cfg = cfg
        self.n_units = n_units
        self.vocab = vocab
        # the scorer's output layer: one row per unit (the linear scorer itself, or the second
        # layer of the two-layer scorer)
        net = getattr(getattr(router, "scorer", None), "net", None)
        out = net if isinstance(net, nn.Linear) else getattr(net, "fc2", None)
        if not isinstance(out, nn.Linear) or out.out_features != n_units:
            raise NotImplementedError("train.route_guard needs a scorer with one output per unit")
        self.params = [p for p in (out.weight, out.bias) if p is not None and p.requires_grad]
        if not self.params:
            raise ValueError("train.route_guard: the scorer's output layer is not trainable")
        device = self.params[0].device
        self.counts = torch.zeros(vocab * n_units, dtype=torch.int64, device=device)
        #: the applied-step scale of every unit's rows (1: untouched)
        self.scale = torch.ones(n_units, dtype=self.params[0].dtype, device=device)
        self.reference: Tensor | None = None  # [V] float64, CPU
        #: per byte, the consecutive readings with a minority share of at least `specialised`
        self.streak = torch.zeros(vocab, dtype=torch.int64)
        self.best_loss: float | None = None  # the lowest interval mean of the training loss
        self._loss_sum = 0.0
        self._loss_n = 0
        self.dominant: int | None = None
        self.checks = 0
        self.events: list[dict[str, Any]] = []
        self.last: dict[str, Any] | None = None
        self._old: list[Tensor] | None = None
        self._anchor = [p.detach().clone() for p in self.params]
        self._anchor_step = 0

    # every window ----------------------------------------------------------------------------

    @torch.no_grad()
    def observe(self, tokens: Tensor, info: Mapping[str, Tensor]) -> None:
        """Count a window's selection slots per (input byte, unit); a dense window has none."""
        if "unit_idx" not in info:
            return
        units, valid = info["unit_idx"], info["valid"]
        index = tokens.to(torch.int64).unsqueeze(-1) * self.n_units + units.to(torch.int64)
        self.counts.index_add_(0, index.reshape(-1), valid.reshape(-1).to(torch.int64))

    def note_loss(self, loss: float) -> None:
        """A step's mean training loss (a non-finite one is left out)."""
        if loss == loss and abs(loss) != float("inf"):
            self._loss_sum += float(loss)
            self._loss_n += 1

    # every optimiser step --------------------------------------------------------------------

    @torch.no_grad()
    def capture(self) -> None:
        """Before the optimiser step: the unit rows, while some unit's step is being scaled."""
        if bool((self.scale < 1.0).any()):
            if self._old is None:
                self._old = [p.detach().clone() for p in self.params]
            else:
                torch._foreach_copy_(self._old, [p.detach() for p in self.params])

    @torch.no_grad()
    def apply(self) -> None:
        """After the optimiser step and the caps: ``old + scale[u] * (new - old)`` on unit
        ``u``'s rows."""
        if self._old is not None and bool((self.scale < 1.0).any()):
            for p, old in zip(self.params, self._old, strict=True):
                scale = self.scale.view(-1, *([1] * (p.ndim - 1)))
                p.copy_(old + scale * (p - old))

    # every interval --------------------------------------------------------------------------

    @torch.no_grad()
    def check(self, step: int) -> dict[str, Any] | None:
        """Read the interval's counts (one host copy) and decide; the event's record, or None."""
        # a copy: on the CPU .cpu() would alias the counts that are zeroed next
        counts = self.counts.view(self.vocab, self.n_units).to(device="cpu", copy=True)
        self.counts.zero_()
        self.checks += 1
        loss = self._loss_sum / self._loss_n if self._loss_n else None
        self._loss_sum, self._loss_n = 0.0, 0
        worse = (
            loss is not None
            and self.best_loss is not None
            and loss > self.best_loss * (1.0 + self.cfg.loss_rise)
        )
        if loss is not None:
            self.best_loss = loss if self.best_loss is None else min(self.best_loss, loss)
        if int(counts.sum()) == 0:
            return None
        if self.reference is None:
            self.reference = torch.zeros(self.vocab, dtype=torch.float64)
        settled = self.streak >= self.cfg.persistence
        lost, usage, minority, dominant = lost_usage(
            counts, self.reference, self.cfg.specialised, settled
        )
        self.last = {
            "step": step,
            "lost": lost,
            "minority_usage": usage,
            "dominant": dominant,
            "loss": loss,
            "worse": worse,
        }
        if self.dominant is None or dominant != self.dominant:
            # another unit serves most positions now: what "minority" means has changed
            self.dominant = dominant
            self.reference = minority.clone()
            self.streak.zero_()
            self._set_anchor(step)
            return None
        self.streak = torch.where(
            minority >= self.cfg.specialised, self.streak + 1, torch.zeros_like(self.streak)
        )
        if worse and lost >= self.cfg.lost_fraction and usage >= self.cfg.min_usage:
            before = float(self.scale[dominant])
            after = max(before * self.cfg.reduce_factor, self.cfg.min_scale)
            self.scale[dominant] = after
            event = {
                "step": step,
                "unit": dominant,
                "lost": lost,
                "minority_usage": usage,
                "loss": loss,
                "best_loss": self.best_loss,
                "scale_before": before,
                "scale": after,
                "restored_from": self._anchor_step if self.cfg.restore else None,
            }
            if self.cfg.restore:
                for p, anchor in zip(self.params, self._anchor, strict=True):
                    p[dominant].copy_(anchor[dominant])
            else:
                self.reference = minority.clone()
                self._set_anchor(step)
            self.events.append(event)
            return event
        if lost < 0.5 * self.cfg.lost_fraction:
            self._set_anchor(step)
        self.reference = torch.maximum(self.reference, minority)
        return None

    def _set_anchor(self, step: int) -> None:
        torch._foreach_copy_(self._anchor, [p.detach() for p in self.params])
        self._anchor_step = step

    # records and checkpoints -----------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """The guard's block of a result record."""
        return {
            "scale": [float(x) for x in self.scale],
            "checks": self.checks,
            "events": [dict(e) for e in self.events],
            "last": dict(self.last) if self.last is not None else None,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts.detach().to(device="cpu", copy=True),
            "scale": self.scale.detach().to(device="cpu", copy=True),
            "reference": None if self.reference is None else self.reference.clone(),
            "streak": self.streak.clone(),
            "best_loss": self.best_loss,
            "loss_sum": self._loss_sum,
            "loss_n": self._loss_n,
            "dominant": self.dominant,
            "checks": self.checks,
            "events": [dict(e) for e in self.events],
            "last": dict(self.last) if self.last is not None else None,
            "anchor": [a.detach().to(device="cpu", copy=True) for a in self._anchor],
            "anchor_step": self._anchor_step,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Any, path: str = "checkpoint") -> None:
        if not isinstance(state, Mapping):
            raise ValueError(f"{path}: route_guard checkpoint state is missing")
        counts, anchor, scale = state.get("counts"), state.get("anchor"), state.get("scale")
        if (
            not isinstance(counts, Tensor)
            or tuple(counts.shape) != tuple(self.counts.shape)
            or not isinstance(scale, Tensor)
            or tuple(scale.shape) != tuple(self.scale.shape)
            or not isinstance(anchor, list)
            or len(anchor) != len(self._anchor)
            or any(
                tuple(a.shape) != tuple(b.shape) for a, b in zip(anchor, self._anchor, strict=True)
            )
        ):
            raise ValueError(f"{path}: route_guard state does not fit this model")
        self.counts.copy_(counts.to(self.counts.device))
        self.scale.copy_(scale.to(device=self.scale.device, dtype=self.scale.dtype))
        reference = state.get("reference")
        self.reference = None if reference is None else reference.to(torch.float64).clone()
        streak = state.get("streak")
        if not isinstance(streak, Tensor) or tuple(streak.shape) != tuple(self.streak.shape):
            raise ValueError(f"{path}: route_guard state does not fit this model")
        self.streak = streak.to(torch.int64).clone()
        best = state.get("best_loss")
        self.best_loss = None if best is None else float(best)
        self._loss_sum = float(state.get("loss_sum", 0.0))
        self._loss_n = int(state.get("loss_n", 0))
        self.dominant = state.get("dominant")
        self.checks = int(state["checks"])
        self.events = [dict(e) for e in state.get("events", [])]
        last = state.get("last")
        self.last = dict(last) if last is not None else None
        for mine, saved in zip(self._anchor, anchor, strict=True):
            mine.copy_(saved.to(device=mine.device, dtype=mine.dtype))
        self._anchor_step = int(state.get("anchor_step", 0))
        self._old = None
