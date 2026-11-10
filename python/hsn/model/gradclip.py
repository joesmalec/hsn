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

The norm is computed as ``peak * ||g / peak||`` (``peak`` the row's largest magnitude), which
cannot overflow for finite entries (exp 316's finite gradient had an fp32 norm of infinity). A row
holding a non-finite entry is zeroed: its direction is unknown, and the non-finite-gradient guard
of the trainer still sees whatever reached the parameters directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["clip_row_grad", "row_norm"]


def row_norm(grad: Tensor) -> Tensor:
    """The L2 norm of each batch row of ``grad [B, ...]``, ``[B, 1]`` in ``grad``'s dtype, without
    overflow for finite entries (``inf`` or ``nan`` for a row that holds one)."""
    flat = grad.flatten(1)
    peak = flat.abs().amax(dim=1, keepdim=True)
    safe = peak.clamp_min(torch.finfo(flat.dtype).tiny)
    norm: Tensor = peak * torch.linalg.vector_norm(flat / safe, dim=1, keepdim=True)
    return norm


class _RowGradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, limit: float) -> Tensor:
        ctx.limit = limit
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        flat = grad.flatten(1)
        norm = row_norm(grad)
        finite = torch.isfinite(norm)
        over = finite & (norm > ctx.limit)
        scaled = flat * (ctx.limit / norm.clamp_min(ctx.limit))
        # under the limit: the gradient itself (no multiplication by 1.0 that could round)
        out = torch.where(over, scaled, flat)
        out = torch.where(finite, out, torch.zeros_like(out))
        return out.view_as(grad), None


def clip_row_grad(x: Tensor, limit: float) -> Tensor:
    """``x`` (``[B, ...]``) unchanged; its gradient limited per row to ``limit`` (module
    docstring). ``limit`` is a finite number > 0."""
    if isinstance(limit, bool) or not isinstance(limit, int | float) or not limit > 0.0:
        raise ValueError(f"limit must be a finite number > 0, got {limit!r}")
    out: Tensor = _RowGradClip.apply(x, float(limit))
    return out
