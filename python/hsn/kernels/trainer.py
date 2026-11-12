"""P2.10b: the trainer's per-step passes after ``backward()`` as single kernel launches.

The optimiser step (:mod:`hsn.train.lazy_adam`), the float64 clip's scaling
(:mod:`hsn.train.grad_norm`), the update caps (:mod:`hsn.train.update_limit`) and the weight
average (:mod:`hsn.train.weight_ema`) each walk a list of tensors with a handful of elementwise
operations: one small kernel per tensor and operation, a few hundred per step, issued by a host
that the card is waiting for. The kernels of ``kernels/cuda/hsn_trainer.cu`` do one such pass
over the whole list in one launch (they live in the window kernels' fatbin and use its launcher,
:mod:`hsn.kernels.window`; ``scripts/build_window_kernels.py`` builds both).

Every function here returns ``True`` when it ran and ``False`` when it did not, having touched
nothing: the kernels are not built or could not be loaded, the tensors are not contiguous fp32
tensors on one CUDA device, or the kernels are switched off (:func:`set_enabled`, or
``HSN_TRAINER_KERNELS=0`` in the environment). The caller then runs its PyTorch path, which is
the reference: each kernel computes the same values bit for bit
(``tests/test_p2_10b_trainer_kernels.py``). Import never initialises CUDA.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch
from torch import Tensor

from hsn.kernels import window

__all__ = [
    "MAX_UNITS",
    "adamw",
    "adamw_units",
    "copy",
    "enabled",
    "lerp",
    "limit_project",
    "limit_squares",
    "ready",
    "scale64",
    "set_enabled",
]

logger = logging.getLogger("hsn.kernels.trainer")

#: Threads per block of every trainer kernel (one element per thread).
_BLOCK = 256
#: The most units a stacked unit tensor may have (``HSN_TR_UNITS`` of the launch descriptor).
MAX_UNITS = 64

_enabled = os.environ.get("HSN_TRAINER_KERNELS", "1") != "0"
_failed = False


def set_enabled(flag: bool) -> bool:
    """Switch the trainer kernels on or off for this process; returns the previous setting."""
    global _enabled
    previous = _enabled
    _enabled = bool(flag)
    return previous


def enabled() -> bool:
    return _enabled


def ready(device: torch.device) -> bool:
    """Whether the trainer kernels can run on ``device``: a CUDA device, the kernels built and
    loaded (loaded here on first use). A build that cannot be loaded (stale sources, another
    PyTorch) is logged once and counts as absent."""
    global _failed
    if not _enabled or device.type != "cuda":
        return False
    if window.prepared(device):
        return True
    if _failed or not window.available():
        return False
    try:
        window.prepare(device.index or 0)
    except RuntimeError as exc:
        _failed = True
        logger.warning("trainer kernels unavailable, the PyTorch passes run instead: %s", exc)
        return False
    return True


def _launch(
    name: str,
    rows: Sequence[Sequence[Tensor]],
    elementwise: int,
    *,
    ints: Sequence[int] = (),
    floats: Sequence[float] = (),
    doubles: Sequence[float] = (),
    units: Sequence[Sequence[float]] = (),
) -> bool:
    first = rows[0]
    if not first or not ready(first[0].device):
        return False
    launched: bool = window._extension().launch_table(
        name,
        _BLOCK,
        [list(row) for row in rows],
        elementwise,
        [int(v) for v in ints],
        [float(v) for v in floats],
        [float(v) for v in doubles],
        [[float(v) for v in row] for row in units],
    )
    return launched


def adamw(
    params: Sequence[Tensor],
    grads: Sequence[Tensor],
    exp_avgs: Sequence[Tensor],
    exp_avg_sqs: Sequence[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> bool:
    """:meth:`hsn.train.lazy_adam.LazyAdamW._dense_steps` as one launch: the AdamW step of
    ordinary tensors at the step count ``step`` (already advanced)."""
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    step_size = lr / bias_correction1
    bias_correction2_sqrt = bias_correction2**0.5
    return _launch(
        "hsn_tr_adamw",
        (params, grads, exp_avgs, exp_avg_sqs),
        4,
        ints=(int(weight_decay != 0),),
        floats=(
            1 - lr * weight_decay,
            1 - beta1,
            beta2,
            1 - beta2,
            1.0 / bias_correction2_sqrt,  # torch divides a tensor list by a scalar this way
            eps,
            -step_size,
        ),
    )


def adamw_units(
    params: Sequence[Tensor],
    grads: Sequence[Tensor],
    exp_avgs: Sequence[Tensor],
    exp_avg_sqs: Sequence[Tensor],
    *,
    n_units: int,
    active: Sequence[bool],
    decay: Sequence[float],
    lerp_weight: Sequence[float],
    beta2_factor: Sequence[float],
    bias_correction2_sqrt: Sequence[float],
    step_size: Sequence[float],
    beta2: float,
    eps: float,
    weight_decay: float,
) -> bool:
    """:meth:`hsn.train.lazy_adam.LazyAdamW._unit_step` as one launch for stacked unit tensors
    ``[n_units, ...]`` that share their per-unit coefficients (each a sequence of ``n_units``
    values of the parameters' dtype: the coefficient tensors of ``_unit_step``)."""
    if n_units > MAX_UNITS or any(p.shape[0] != n_units for p in params):
        return False
    return _launch(
        "hsn_tr_adamw_units",
        (params, grads, exp_avgs, exp_avg_sqs),
        4,
        ints=(int(weight_decay != 0), n_units),
        floats=(0.0, 0.0, 0.0, 1 - beta2, 0.0, eps),
        units=(
            [1.0 if flag else 0.0 for flag in active],
            decay,
            lerp_weight,
            beta2_factor,
            bias_correction2_sqrt,
            step_size,
        ),
    )


def lerp(selfs: Sequence[Tensor], ends: Sequence[Tensor], weight: float) -> bool:
    """``torch._foreach_lerp_(selfs, ends, weight)`` as one launch."""
    return _launch("hsn_tr_lerp", (selfs, ends), 2, floats=(weight,))


def scale64(grads: Sequence[Tensor], coefficient: Tensor) -> bool:
    """``grad.copy_((grad.double() * coefficient).float())`` for every gradient as one launch;
    ``coefficient`` is a 0-dim float64 tensor on the gradients' device."""
    if coefficient.dtype != torch.float64 or coefficient.numel() != 1:
        return False
    return _launch("hsn_tr_scale64", (grads, [coefficient] * len(grads)), 1)


def copy(dsts: Sequence[Tensor], srcs: Sequence[Tensor]) -> bool:
    """``torch._foreach_copy_(dsts, srcs)`` for fp32 tensors as one launch."""
    return _launch("hsn_tr_copy", (dsts, srcs), 2)


def limit_squares(
    params: Sequence[Tensor],
    olds: Sequence[Tensor],
    old_squares: Sequence[Tensor],
    delta_squares: Sequence[Tensor],
) -> bool:
    """The squares :meth:`hsn.train.update_limit.UpdateLimiter.apply` sums for its two norms:
    ``old ** 2`` and ``(param - old) ** 2`` of every tensor, into the given buffers."""
    return _launch("hsn_tr_limit_squares", (params, olds, old_squares, delta_squares), 4)


def limit_project(
    params: Sequence[Tensor],
    olds: Sequence[Tensor],
    applied_squares: Sequence[Tensor],
    groups: Sequence[Tensor],
) -> bool:
    """The projection of :meth:`~hsn.train.update_limit.UpdateLimiter.apply`: ``param = old +
    (param - old) * scale`` on the tensors of a clipped group (``groups[j]`` holds tensor
    ``j``'s group as two fp32 values on the device, ``[scale, clipped]``), and the squares of
    the applied step ``(param - old) ** 2`` into ``applied_squares``."""
    if any(g.dtype != torch.float32 or g.numel() != 2 for g in groups):
        return False
    return _launch("hsn_tr_limit_project", (params, olds, applied_squares, groups), 3)
