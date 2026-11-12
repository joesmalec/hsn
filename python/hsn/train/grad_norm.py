"""Opt-in float64 global gradient norms and clipping outside the captured backward.

The native branch delegates directly to PyTorch. The float64 branch prevents finite
float32 gradients from overflowing the L2 reduction, and multiplies in float64 too:
casting a very small clipping coefficient to the gradient dtype can erase its direction.
MPS reductions and scaling use the CPU because MPS does not support float64.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from hsn.config import GradNormPrecision


def _wide(gradient: Tensor) -> Tensor:
    device = torch.device("cpu") if gradient.device.type == "mps" else gradient.device
    return gradient.detach().to(device=device, dtype=torch.float64)


@torch.no_grad()
def norm_float64(gradients: Sequence[Tensor]) -> Tensor:
    """L2 norm without narrowing the gradients; absent gradients contribute nothing."""
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    norms = [torch.linalg.vector_norm(_wide(g)) for g in gradients]
    result: Tensor = torch.linalg.vector_norm(torch.stack([n.to(norms[0].device) for n in norms]))
    return result


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[Tensor], max_norm: float, *, precision: GradNormPrecision = "native"
) -> Tensor:
    """Clip in place, preserving gradient storage; return the unscaled pre-clip norm.

    True nonfinite gradients are left intact for the existing guard / GradScaler and
    divergence diagnostics. No finite batch is skipped here. The 1e-6 denominator
    epsilon is PyTorch's existing clipping definition, not a new trigger or threshold.
    """
    if precision == "native":
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    # P2.10: the float64 copies are made once (the norm and the scaling read the same ones),
    # and the one host read decides both early returns. The values are those of the two-pass
    # version, bit for bit (tests/test_p2_10_trainer_speed.py).
    wides = [_wide(g) for g in gradients]
    norms = [torch.linalg.vector_norm(w) for w in wides]
    total: Tensor = torch.linalg.vector_norm(torch.stack([n.to(norms[0].device) for n in norms]))
    value = float(total)
    if not math.isfinite(value):
        return total
    if max_norm / (value + 1e-6) >= 1.0:  # the coefficient clamps to exactly 1: nothing changes
        return total
    coefficient = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for gradient, wide in zip(gradients, wides, strict=True):
        # Do not cast coefficient to the original dtype before multiplying.
        product = wide.mul_(coefficient.to(wide.device))
        if wide.device != gradient.device:
            # Cast on the computation device first, so MPS never receives a float64 tensor.
            product = product.to(dtype=gradient.dtype)
        gradient.copy_(product)
    return total
