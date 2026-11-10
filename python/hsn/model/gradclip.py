"""A6: per-row clipping of the gradient a position passes back into the recurrent state.

:func:`clip_row_grad` is the identity in the forward pass. In the backward pass it scales each
batch row of the incoming gradient to at most ``limit`` in L2 norm: a row under the limit is
returned as it came (no scaling), a row above it keeps its direction. :class:`~hsn.model.hsn.HSN`
applies it to ``h`` and ``q`` at the entry of every position when ``train.state_grad_clip`` is set
(``docs/amendments/A6_feedback_gradient.md``), so the gradient that reaches position ``t − 1``
from position ``t`` and everything after it is bounded row by row, whatever happened later in the
window.

Why per row and per position, when the trainer already clips the global norm: the A5 bursts are
single documents whose state gradient jumps 7x to 22x at one write or read (a hard-switching gate
or candidate in transition), several times in a row. The global clip then rescales the whole
batch by the exploded norm: the other rows' gradients vanish and the step follows that one
document's switching direction at full size. Here the row is limited where the jump happens, and
the other rows are untouched.

Two limits, either or both (the smaller applies). The **absolute** limit is one number for the
whole run. Healthy per-row gradients grow about twentyfold over a run of the A5 recipe (the largest
``dL/dq`` row of a window: 0.13 at step 500, 0.35 at 1,000, 1.2 at 2,000, 2.9 from 3,000 on), so a
number that never touches a healthy row late is far above an exploding row early. The **relative**
limit needs no scale: a row may be at most ``ratio`` times the batch's typical live row at the same
position (:func:`typical_row`: the 90th percentile of the rows that carry a gradient there). In
healthy windows the largest row is 1.2 to 2.1 times that percentile, at every stage measured.

The norm is computed as ``peak * ||g / peak||`` (``peak`` the row's largest magnitude), which
cannot overflow for finite entries (exp 316's finite gradient had an fp32 norm of infinity). A row
holding a non-finite entry is zeroed: its direction is unknown, and the non-finite-gradient guard
of the trainer still sees whatever reached the parameters directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["TYPICAL_QUANTILE", "clip_row_grad", "row_norm", "typical_row"]


def row_norm(grad: Tensor) -> Tensor:
    """The L2 norm of each batch row of ``grad [B, ...]``, ``[B, 1]`` in ``grad``'s dtype, without
    overflow for finite entries (``inf`` or ``nan`` for a row that holds one)."""
    flat = grad.flatten(1)
    peak = flat.abs().amax(dim=1, keepdim=True)
    safe = peak.clamp_min(torch.finfo(flat.dtype).tiny)
    norm: Tensor = peak * torch.linalg.vector_norm(flat / safe, dim=1, keepdim=True)
    return norm


#: The quantile of the batch's live rows that a relative limit multiplies (module docstring).
TYPICAL_QUANTILE = 0.9


def typical_row(norm: Tensor) -> Tensor:
    """The :data:`TYPICAL_QUANTILE` quantile of the live rows of ``norm [B, 1]`` (finite and
    non-zero: a row whose document has no later loss carries no gradient and says nothing about
    scale), a 0-dim tensor; ``inf`` when no row is live. The ``ceil(q * n)``-th smallest of the
    ``n`` live rows, so with fewer than ten live rows it is their largest (no row is over it)."""
    rows = norm.squeeze(1)
    live = torch.isfinite(rows) & (rows > 0)
    count = live.sum()
    ordered = torch.sort(torch.where(live, rows, torch.full_like(rows, float("inf")))).values
    index = (torch.ceil(TYPICAL_QUANTILE * count.to(rows.dtype)).to(torch.int64) - 1).clamp_min(0)
    # a gather, not ``ordered[index]``: indexing by a 0-dim tensor reads it on the host, which a
    # compiled window cannot capture
    typical: Tensor = ordered.gather(0, index.reshape(1)).reshape(())
    return typical


class _RowGradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, limit: float | None, ratio: float | None) -> Tensor:
        ctx.limit = limit
        ctx.ratio = ratio
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None]:
        flat = grad.flatten(1)
        norm = row_norm(grad)
        finite = torch.isfinite(norm)
        bound = torch.full_like(norm, float("inf"))
        if ctx.limit is not None:
            bound = bound.clamp_max(ctx.limit)
        if ctx.ratio is not None:
            bound = torch.minimum(bound, ctx.ratio * typical_row(norm))
        over = finite & (norm > bound)
        # an over row has norm > bound >= 0, so the division is safe; the others are not scaled
        scale = torch.where(over, bound / norm.clamp_min(torch.finfo(flat.dtype).tiny), 1.0)
        # under the limit: the gradient itself (no multiplication by 1.0 that could round)
        out = torch.where(over, flat * scale, flat)
        out = torch.where(finite, out, torch.zeros_like(out))
        return out.view_as(grad), None, None


def clip_row_grad(x: Tensor, limit: float | None = None, *, ratio: float | None = None) -> Tensor:
    """``x`` (``[B, ...]``) unchanged; its gradient limited per row (module docstring) to
    ``limit`` (an absolute L2 norm), to ``ratio`` times the batch's typical live row
    (:func:`typical_row`), or to the smaller of the two. At least one must be given; each is a
    finite number > 0."""
    for name, value in (("limit", limit), ("ratio", ratio)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not value > 0.0
        ):
            raise ValueError(f"{name} must be None or a finite number > 0, got {value!r}")
    if limit is None and ratio is None:
        raise ValueError("clip_row_grad needs a limit, a ratio or both")
    out: Tensor = _RowGradClip.apply(
        x, None if limit is None else float(limit), None if ratio is None else float(ratio)
    )
    return out
