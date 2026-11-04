"""Loss terms of the training objective (plan §2.6). P1.6 contributes the prediction part:
``L_pred`` with collapse protection. P1.7 adds ``assemble_loss``; later chunks add their terms.

The predictor ``P`` must not be able to minimise its loss by collapsing the encoder ``E`` (every
input mapped to one point makes ``ẑ = z`` trivial). Two targets, selected by
``model.predictor_target``:

* ``fixed_embedding``: the target is the byte embedding of ``x_t``, detached. The embedding table
  is trained by every other path but receives no gradient from ``L_pred``, so the prediction loss
  cannot pull it together; collapse is avoided trivially. The first experiment.
* ``ema_encoder``: the target is the output of a slowly updated copy of ``E``
  (:class:`hsn.model.encoder.EMATarget`, BYOL-style stop-gradient) and a VICReg-style variance
  floor keeps the *online* encodings spread out across the batch.

Equations (``z_hat``, ``z_target``, ``z`` are ``[B, d]``, or ``[B, T, d]`` over a window)::

    pred_loss(z_hat, target)        = mean over every element of (z_hat − target)²
                                      (the ‖ε‖² of §2.6 averaged over the batch and divided by d;
                                      the normalisation is absorbed by λ_p)
    variance_floor(z, γ, eps)       = mean_j relu(γ − sqrt(Var_batch(z[:, j]) + eps))
    prediction_terms(...)   fixed:  {"pred": λ_p · pred_loss(z_hat, z_target)}
                            ema:    {"pred": ..., "variance_floor": λ_p · variance_floor(z, γ)}

The target is detached in both modes; :func:`pred_loss` refuses one that requires grad rather
than letting the gradient leak into the target path (errors over silence). The variance floor is
part of ``L_pred``'s collapse protection and therefore scales with ``train.lambda_p``;
``train.variance_floor_gamma = 0`` switches it off (``relu(0 − sqrt(·)) = 0``). It is applied to
the online ``z`` (the branch that receives gradients and that the EMA target tracks), never to the
detached target, where it would have no effect. The batch variance is the unbiased one
(``correction=1``, the VICReg convention), so a batch of one row is refused: its unbiased variance
is NaN, which would silently poison the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hsn.config import PredictorTarget

__all__ = ["pred_loss", "prediction_terms", "variance_floor"]

_MODES: tuple[str, ...] = ("fixed_embedding", "ema_encoder")


def pred_loss(z_hat: Tensor, target: Tensor) -> Tensor:
    """Mean squared error between the prediction ``z_hat [B, d]`` and the detached ``target``.

    Both tensors must have the same shape (any leading shape, ``[..., d]``) and ``target`` must not
    require grad. Returns a 0-dim tensor in ``z_hat``'s dtype.
    """
    _check_tensor("z_hat", z_hat)
    _check_tensor("target", target)
    if z_hat.shape != target.shape:
        raise ValueError(
            f"pred_loss: z_hat has shape {tuple(z_hat.shape)} but target {tuple(target.shape)}"
        )
    if target.requires_grad:
        raise ValueError(
            "pred_loss: the target requires grad; the prediction target must be detached in both "
            "modes (the fixed embedding with .detach(), the EMA encoder under no_grad)"
        )
    return F.mse_loss(z_hat, target)


def variance_floor(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """``mean_j relu(gamma − sqrt(Var_batch(z[:, j]) + eps))`` over the rows of ``z [B, d]``.

    Zero when every dimension's standard deviation across the batch is at least ``gamma``; positive
    (up to ``gamma − sqrt(eps)``) as the batch collapses. ``B`` must be at least 2 (the variance is
    the unbiased estimate). ``gamma = 0`` disables the term.
    """
    _check_tensor("z", z)
    if z.ndim != 2:
        raise ValueError(f"variance_floor expects z [B, d], got shape {tuple(z.shape)}")
    if z.shape[0] < 2:
        raise ValueError(
            f"variance_floor needs at least 2 rows to estimate a batch variance, got {z.shape[0]}"
        )
    if gamma < 0.0:
        raise ValueError(f"variance_floor: gamma must be >= 0, got {gamma}")
    if eps <= 0.0:
        raise ValueError(f"variance_floor: eps must be > 0, got {eps}")
    std = torch.sqrt(z.var(dim=0, correction=1) + eps)  # [d]
    return F.relu(gamma - std).mean()


def prediction_terms(
    z_hat: Tensor,
    z_target: Tensor,
    z: Tensor,
    *,
    lambda_p: float,
    predictor_target: PredictorTarget,
    variance_floor_gamma: float,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """The ``λ_p`` part of the objective, one entry per term, for ``assemble_loss`` (P1.7).

    ``z_hat``, ``z_target`` and ``z`` are ``[B, d]`` (one :meth:`~hsn.model.hsn.HSN.step`) or
    ``[B, T, d]`` (the ``info`` stacks of a window); leading dimensions are flattened, so over a
    window every position is one sample of the batch variance. ``mask`` (bool, the leading shape,
    e.g. the non-padding positions) selects the samples; ``None`` takes all of them. Returns
    ``{"pred": λ_p · pred_loss(z_hat, z_target)}`` for the fixed-embedding target and additionally
    ``"variance_floor": λ_p · variance_floor(z, γ)`` for the EMA target
    (``train.lambda_p``, ``model.predictor_target``, ``train.variance_floor_gamma``). The total is
    the sum of the values; each entry is logged separately so its gradient can be inspected (§2.6).
    """
    if predictor_target not in _MODES:
        raise ValueError(
            f"model.predictor_target must be one of {_MODES}, got {predictor_target!r}"
        )
    if lambda_p < 0.0:
        raise ValueError(f"train.lambda_p must be >= 0, got {lambda_p}")
    for name, x in (("z_hat", z_hat), ("z_target", z_target), ("z", z)):
        _check_tensor(name, x)
        if x.ndim < 2:
            raise ValueError(f"prediction_terms: {name} must be [..., d], got {tuple(x.shape)}")
        if x.shape != z_hat.shape:
            raise ValueError(
                f"prediction_terms: {name} has shape {tuple(x.shape)} but z_hat "
                f"{tuple(z_hat.shape)}"
            )
    d = z_hat.shape[-1]
    flat_hat, flat_target, flat_z = (x.reshape(-1, d) for x in (z_hat, z_target, z))
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != z_hat.shape[:-1]:
            raise ValueError(
                f"prediction_terms: mask must be a bool tensor of shape {tuple(z_hat.shape[:-1])}, "
                f"got {mask.dtype} {tuple(mask.shape)}"
            )
        rows = mask.reshape(-1)
        flat_hat, flat_target, flat_z = flat_hat[rows], flat_target[rows], flat_z[rows]
        if flat_hat.shape[0] == 0:
            raise ValueError("prediction_terms: the mask selects no position")
    terms = {"pred": lambda_p * pred_loss(flat_hat, flat_target)}
    if predictor_target == "ema_encoder":
        terms["variance_floor"] = lambda_p * variance_floor(flat_z, variance_floor_gamma)
    return terms


def _check_tensor(name: str, x: object) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must be a floating-point tensor, got dtype {x.dtype}")
