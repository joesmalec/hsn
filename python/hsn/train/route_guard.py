"""``train.route_guard``: act when an established route is being taken over (A6, opt-in).

**The event.** In a routed HSN the router's units divide the positions between them: one or two
serve the filler, one serves the positions where a query's key is read, and a few serve the
positions where a fact's value is written, which is what binds a value to its key. The unit that
reads the query keys keeps growing its weight on the current byte's address for a whole run,
whatever the loss or the learning rate (exp 389: 0.13 to 0.62; its proposed step is about ten
times the update cap, so a rate cut does not reach it), while the value units' weights level off
once their route is learned. When its scores overtake theirs at the value positions, top-K hands
it those positions, the value units stop receiving a gradient there and cannot win them back, and
the accuracy the route carried is lost (exps 375-377: 10 to 21 points; binding overlap 0.44-0.51
to 0.16-0.19). The takeover is not tied to a step count: it began at step 1,500, 2,500 or 3,100
depending on the settings and the run.

**The measurement** needs no knowledge of the task beyond the loss mask and no extra forward
pass. From every training window the guard counts, per input byte ``b`` and unit ``u``, the
selection slots ``u`` won at positions whose input is ``b``, and per unit the slots it won at the
supervised positions. Every ``interval`` applied steps it reads the interval's shares ``p[b, u]``
(unit ``u``'s share of byte ``b``'s slots)::

    readers          = units with at least ``specialised`` of the slots at supervised positions
                       (the units whose routes feed the answers; in every run inspected the
                       taker is one of them)
    reference[b, u]  = the largest share since the last event
    established      = routes (b, u) whose share was at least ``specialised`` at ``persistence``
                       consecutive readings (bytes with fewer than ``min_slots`` slots are left out)
    taken bytes      = bytes on which a reader's share is above its own reference
    lost[b]          = sum_u max(0, reference - p) / sum_u reference    (other units' established
                       routes)
    lost             = the mean of lost[b] over the quarter of the taken bytes that lost most
    gain[u]          = sum over those bytes of n[b] * max(0, p[b, u] - p_previous[b, u]), readers

(a kind of position is a group of input bytes: the values are a quarter to a third of the bytes
in use and two per cent of the slots, so a slot-weighted total would not see their routes go; and
routes that move between non-reader units, as the filler's do, are not this event: a guard that
counted them slowed the wrong units, exps 402 to 404).

At the start of a run the units sort the positions out while the loss falls: that is learning.
So a reading is an event only if training is no longer improving: the interval's mean training
loss is above the lowest interval mean so far by ``loss_rise``.

**The response** is on the reader that gained the most of what the established routes lost. Its
*applied* step on its rows of the scorer's output layer is multiplied by ``reduce_factor`` (never
below ``min_scale``): from then on every update of those rows is ``old + scale * (new - old)``,
after the optimiser step and the caps. The other units keep their full step, so an established
route can go on strengthening. No weight is put back: an option that also rewound the reader's
rows by one reading was worse in its one run (exp 400) and was removed. After an event the
reference restarts from the current shares.

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
    counts: Tensor,
    reference: Tensor,
    established: Tensor,
    readers: Tensor,
    min_slots: int = 1,
    previous: Tensor | None = None,
) -> tuple[float, float, Tensor, int | None]:
    """``(lost, usage, shares, gaining reader)`` of one interval's ``counts [V, U]`` against
    ``reference [V, U]`` (module docstring): over the bytes with at least ``min_slots`` slots on
    which a reader unit (``readers``, bool ``[U]``) holds a larger share than it ever has, the
    share of the other units' ``established [V, U]`` routes that is gone, averaged over the
    quarter of those bytes that lost most. ``usage`` is the other units' established routes'
    reference share of all slots; ``lost`` is 0 and the unit None when no reader is taking an
    established route. The gaining reader is the one whose share of those bytes grew most since
    ``previous`` (the last reading's shares; the reference when not given)."""
    counts = counts.to(torch.float64)
    per_byte = counts.sum(dim=1)
    total = float(per_byte.sum())
    shares = torch.where(
        per_byte[:, None] > 0, counts / per_byte[:, None].clamp_min(1.0), torch.zeros_like(counts)
    )
    others = established & ~readers[None, :] & (per_byte[:, None] >= min_slots)
    base = (reference * others).sum(dim=1)  # per byte: the other units' established share
    usage = float((per_byte * base).sum() / total) if total > 0 else 0.0
    rising = ((shares - reference) * readers[None, :]).clamp_min(0.0).sum(dim=1) > 0
    taken = (base > 0) & rising  # bytes where a reader is above its own record
    n_taken = int(taken.sum())
    if n_taken == 0:
        return 0.0, usage, shares, None
    drop = ((reference - shares).clamp_min(0.0) * others).sum(dim=1)
    relative = torch.where(taken, drop / base.clamp_min(1e-12), torch.zeros_like(drop))
    worst = relative.topk(max(1, -(-n_taken // 4)))  # the quarter of those bytes that lost most
    lost = float(worst.values.mean())
    if lost <= 0.0:
        return 0.0, usage, shares, None
    rows = worst.indices[worst.values > 0]
    before = reference if previous is None else previous
    gain = (per_byte[rows, None] * (shares[rows] - before[rows]).clamp_min(0.0)).sum(dim=0)
    gain = gain * readers
    unit = int(gain.argmax()) if float(gain.max()) > 0.0 else None
    return lost, usage, shares, unit


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
        #: per unit, the slots won at supervised positions in the interval
        self.read_counts = torch.zeros(n_units, dtype=torch.int64, device=device)
        #: the applied-step scale of every unit's rows (1: untouched)
        self.scale = torch.ones(n_units, dtype=self.params[0].dtype, device=device)
        self.reference = torch.zeros(vocab, n_units, dtype=torch.float64)  # CPU
        self.previous = torch.zeros(vocab, n_units, dtype=torch.float64)  # the last reading
        #: per (byte, unit), the consecutive readings with a share of at least `specialised`
        self.streak = torch.zeros(vocab, n_units, dtype=torch.int64)
        self.best_loss: float | None = None  # the lowest interval mean of the training loss
        self._loss_sum = 0.0
        self._loss_n = 0
        self.checks = 0
        self.events: list[dict[str, Any]] = []
        self.last: dict[str, Any] | None = None
        self._old: list[Tensor] | None = None

    # every window ----------------------------------------------------------------------------

    @torch.no_grad()
    def observe(self, tokens: Tensor, loss_mask: Tensor, info: Mapping[str, Tensor]) -> None:
        """Count a window's selection slots per (input byte, unit) and, at the supervised
        positions (``loss_mask [B, T]``), per unit; a dense window has none."""
        if "unit_idx" not in info:
            return
        units, valid = info["unit_idx"].to(torch.int64), info["valid"].to(torch.int64)
        index = tokens.to(torch.int64).unsqueeze(-1) * self.n_units + units
        self.counts.index_add_(0, index.reshape(-1), valid.reshape(-1))
        supervised = valid * loss_mask.to(torch.int64).unsqueeze(-1)
        self.read_counts.index_add_(0, units.reshape(-1), supervised.reshape(-1))

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
        read = self.read_counts.to(device="cpu", copy=True).to(torch.float64)
        self.counts.zero_()
        self.read_counts.zero_()
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
        established = (self.streak >= self.cfg.persistence) & (
            self.reference >= self.cfg.specialised
        )
        readers = read >= self.cfg.specialised * read.sum().clamp_min(1.0)
        lost, usage, shares, unit = lost_usage(
            counts, self.reference, established, readers, self.cfg.min_slots, self.previous
        )
        self.previous = shares.clone()
        self.last = {
            "step": step,
            "lost": lost,
            "established_usage": usage,
            "gaining_unit": unit,
            "readers": [u for u in range(self.n_units) if bool(readers[u])],
            "loss": loss,
            "worse": worse,
        }
        self.streak = torch.where(
            shares >= self.cfg.specialised, self.streak + 1, torch.zeros_like(self.streak)
        )
        if (
            worse
            and unit is not None
            and lost >= self.cfg.lost_fraction
            and usage >= self.cfg.min_usage
        ):
            before = float(self.scale[unit])
            after = max(before * self.cfg.reduce_factor, self.cfg.min_scale)
            self.scale[unit] = after
            event = {
                "step": step,
                "unit": unit,
                "lost": lost,
                "established_usage": usage,
                "loss": loss,
                "best_loss": self.best_loss,
                "scale_before": before,
                "scale": after,
            }
            self.reference = shares.clone()
            self.events.append(event)
            return event
        self.reference = torch.maximum(self.reference, shares)
        return None

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
            "read_counts": self.read_counts.detach().to(device="cpu", copy=True),
            "scale": self.scale.detach().to(device="cpu", copy=True),
            "reference": self.reference.clone(),
            "previous": self.previous.clone(),
            "streak": self.streak.clone(),
            "best_loss": self.best_loss,
            "loss_sum": self._loss_sum,
            "loss_n": self._loss_n,
            "checks": self.checks,
            "events": [dict(e) for e in self.events],
            "last": dict(self.last) if self.last is not None else None,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Any, path: str = "checkpoint") -> None:
        """A checkpoint written while the ``restore`` option existed also holds ``anchor`` and
        ``anchor_step`` (the rows a restore went back to); they are not read."""
        if not isinstance(state, Mapping):
            raise ValueError(f"{path}: route_guard checkpoint state is missing")
        counts, scale = state.get("counts"), state.get("scale")
        if (
            not isinstance(counts, Tensor)
            or tuple(counts.shape) != tuple(self.counts.shape)
            or not isinstance(scale, Tensor)
            or tuple(scale.shape) != tuple(self.scale.shape)
        ):
            raise ValueError(f"{path}: route_guard state does not fit this model")
        self.counts.copy_(counts.to(self.counts.device))
        read = state.get("read_counts")
        if not isinstance(read, Tensor) or tuple(read.shape) != tuple(self.read_counts.shape):
            raise ValueError(f"{path}: route_guard state does not fit this model")
        self.read_counts.copy_(read.to(self.read_counts.device))
        self.scale.copy_(scale.to(device=self.scale.device, dtype=self.scale.dtype))
        reference, streak = state.get("reference"), state.get("streak")
        if (
            not isinstance(reference, Tensor)
            or tuple(reference.shape) != tuple(self.reference.shape)
            or not isinstance(streak, Tensor)
            or tuple(streak.shape) != tuple(self.streak.shape)
        ):
            raise ValueError(f"{path}: route_guard state does not fit this model")
        self.reference = reference.to(torch.float64).clone()
        previous = state.get("previous")
        if not isinstance(previous, Tensor) or tuple(previous.shape) != tuple(reference.shape):
            raise ValueError(f"{path}: route_guard state does not fit this model")
        self.previous = previous.to(torch.float64).clone()
        self.streak = streak.to(torch.int64).clone()
        best = state.get("best_loss")
        self.best_loss = None if best is None else float(best)
        self._loss_sum = float(state.get("loss_sum", 0.0))
        self._loss_n = int(state.get("loss_n", 0))
        self.checks = int(state["checks"])
        self.events = [dict(e) for e in state.get("events", [])]
        last = state.get("last")
        self.last = dict(last) if last is not None else None
        self._old = None
