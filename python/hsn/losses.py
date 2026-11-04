"""Loss terms of the training objective (plan §2.6). P1.6 contributes the prediction part:
``L_pred`` with collapse protection; P1.7 adds :func:`assemble_loss`, the whole objective; later
chunks add the terms they compute (P2.4, P3.1-P3.3).

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

P1.7: :func:`assemble_loss` builds the objective of §2.6 from one window of model output::

    L = L_task + λ_p·L_pred + λ_r·L_replay + λ_d·L_distill + λ_e·L_EWC
                + λ_c·L_cost + λ_b·L_balance + λ_g·L_graph

``task`` (next-symbol cross-entropy over the ``loss_mask`` positions, weight 1) is always on;
``pred`` (:func:`prediction_terms` over the non-padding positions ``doc >= 0``) is on whenever
``train.lambda_p > 0``. Both are computed here from the model's window output. Every other term is
computed by a later chunk (replay P3.1, distill P3.2, ewc P3.3, cost / balance / graph P2.4) and
reaches the assembly as a raw 0-dim tensor in ``extras``, which the assembly multiplies by its λ.
A continual-learning term (replay, distill, ewc) is on when its λ is positive. An auxiliary term
(cost, balance, graph: :data:`hsn.config.AUX_TERMS`) is on when it is named in
``train.aux_enabled`` or its λ is positive, and the two switches must agree. **At most one
auxiliary term may be on** unless ``train.allow_multiple_aux`` carries a written justification
(global rule 4: penalties are added one at a time); the config layer refuses such a file at load
time and the assembly refuses it again, because a configuration can be built in code. A term that
is on without a value, a value for a term that is off, and an unknown name in ``extras`` all raise
(errors over silence). Loss inputs are upcast to fp32 first (losses stay in fp32 whatever autocast
produced). ``parts`` holds every included term as a λ-weighted float (the P1.6 convention) in
formula order, so each can be logged and its gradient inspected.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from hsn.config import AUX_TERMS, PredictorTarget, RunConfig

__all__ = ["TERMS", "assemble_loss", "pred_loss", "prediction_terms", "variance_floor"]

#: The terms of the objective (§2.6) in formula order.
TERMS: tuple[str, ...] = ("task", "pred", "replay", "distill", "ewc", "cost", "balance", "graph")
#: ``train.*`` weight of every term but ``task`` (whose weight is 1).
_LAMBDA: dict[str, str] = {
    "pred": "lambda_p",
    "replay": "lambda_r",
    "distill": "lambda_d",
    "ewc": "lambda_e",
    "cost": "lambda_c",
    "balance": "lambda_b",
    "graph": "lambda_g",
}
#: The chunk that computes each term the assembly does not compute itself.
_SOURCE: dict[str, str] = {
    "replay": "P3.1",
    "distill": "P3.2",
    "ewc": "P3.3",
    "cost": "P2.4",
    "balance": "P2.4",
    "graph": "P2.4",
}
_CONTINUAL_TERMS: tuple[str, ...] = ("replay", "distill", "ewc")
_EXTRA_TERMS: tuple[str, ...] = _CONTINUAL_TERMS + AUX_TERMS

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


# --------------------------------------------------------------------------------------------------
# P1.7: the whole objective
# --------------------------------------------------------------------------------------------------


def assemble_loss(
    out: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: RunConfig,
    extras: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """The §2.6 objective of one window: ``(total, parts)`` (module docstring).

    ``out`` is the model's window output: ``logits [B, T, V]`` plus the ``info`` of
    :meth:`hsn.model.hsn.HSN.forward` (``z_hat``, ``z_target``, ``z`` ``[B, T, d]`` for the
    prediction term). ``batch`` is the loader window: ``targets`` (int64 ``[B, T]``), ``loss_mask``
    (bool ``[B, T]``) and ``doc`` (int64 ``[B, T]``, ``-1`` on padding). ``cfg`` is the run
    configuration (``train.lambda_*``, ``train.aux_enabled``, ``train.allow_multiple_aux``,
    ``train.variance_floor_gamma`` and ``model.predictor_target``). ``extras`` maps the name of
    every enabled replay / distill / ewc / cost / balance / graph term to its raw (unweighted) 0-dim
    value. ``total`` is a 0-dim fp32 tensor with the graph; ``parts`` the λ-weighted value of every
    included term as a float, in formula order (``variance_floor`` follows ``pred`` in EMA mode).
    A window without loss positions contributes a zero task term with a zero gradient.
    """
    if not isinstance(cfg, RunConfig):
        raise TypeError(f"assemble_loss needs the RunConfig, got {type(cfg).__name__}")
    for name, mapping in (("out", out), ("batch", batch)):
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"assemble_loss: {name} must be a mapping of tensors, got {type(mapping).__name__}"
            )
    given = _check_extras(extras)
    train = cfg.train
    terms: dict[str, Tensor] = {}

    # L_task: next-symbol cross-entropy over the loss positions
    logits = _field(out, "logits", "out")
    targets = _field(batch, "targets", "batch")
    loss_mask = _field(batch, "loss_mask", "batch")
    if logits.ndim != 3 or not torch.is_floating_point(logits):
        raise ValueError(
            f"assemble_loss: logits must be a floating [B, T, V] tensor, got {logits.dtype} "
            f"{tuple(logits.shape)}"
        )
    shape = tuple(logits.shape[:2])
    if loss_mask.dtype != torch.bool or tuple(loss_mask.shape) != shape:
        raise ValueError(
            f"assemble_loss: loss_mask must be a bool tensor of shape {shape}, got "
            f"{loss_mask.dtype} {tuple(loss_mask.shape)}"
        )
    if targets.dtype != torch.int64 or tuple(targets.shape) != shape:
        raise ValueError(
            f"assemble_loss: targets must be an int64 tensor of shape {shape}, got "
            f"{targets.dtype} {tuple(targets.shape)}"
        )
    if bool(loss_mask.any()):
        terms["task"] = F.cross_entropy(logits[loss_mask].float(), targets[loss_mask])
    else:  # padding only: a zero term that still carries a (zero) gradient
        terms["task"] = logits.float().sum() * 0.0

    # λ_p · L_pred over the non-padding positions (P1.6)
    if train.lambda_p < 0.0:
        raise ValueError(f"train.lambda_p must be >= 0, got {train.lambda_p}")
    if train.lambda_p > 0.0:
        doc = _field(batch, "doc", "batch")
        if tuple(doc.shape) != shape:
            raise ValueError(f"assemble_loss: doc must have shape {shape}, got {tuple(doc.shape)}")
        z_hat, z_target, z = (_field(out, key, "out").float() for key in ("z_hat", "z_target", "z"))
        terms.update(
            prediction_terms(
                z_hat,
                z_target,
                z,
                lambda_p=train.lambda_p,
                predictor_target=cfg.model.predictor_target,
                variance_floor_gamma=train.variance_floor_gamma,
                mask=doc >= 0,
            )
        )

    # continual-learning terms (P3.1-P3.3): on when their weight is positive
    for name in _CONTINUAL_TERMS:
        lam = _weight(train, name)
        if lam > 0.0:
            terms[name] = lam * _take_extra(name, given)

    # auxiliary terms (P2.4): one at a time, switched by train.aux_enabled and the weight together
    on = [name for name in AUX_TERMS if name in train.aux_enabled or _weight(train, name) > 0.0]
    if len(on) > 1 and not train.allow_multiple_aux.strip():
        raise ValueError(
            f"auxiliary terms {on} are enabled together; penalties are added one at a time "
            "(global rule 4). Set train.allow_multiple_aux to a written justification to override."
        )
    for name in on:
        key, lam = _LAMBDA[name], _weight(train, name)
        if name not in train.aux_enabled:
            raise ValueError(
                f"train.{key}={lam} but {name!r} is not in train.aux_enabled="
                f"{train.aux_enabled}; name the term there or set its weight to 0"
            )
        if lam <= 0.0:
            raise ValueError(
                f"{name!r} is in train.aux_enabled but train.{key}={lam}; a weight of 0 would "
                "silently drop the term"
            )
        terms[name] = lam * _take_extra(name, given)
    if given:
        raise ValueError(
            f"extras carries values for {sorted(given)}, which are not enabled by the "
            "configuration (set the term's weight, and for cost / balance / graph name it in "
            "train.aux_enabled)"
        )

    total = terms["task"]
    for name, value in terms.items():
        if name != "task":
            total = total + value
    parts = {name: float(value.detach()) for name, value in terms.items()}
    return total, parts


def _field(mapping: Mapping[str, Tensor], key: str, where: str) -> Tensor:
    if key not in mapping:
        raise KeyError(f"assemble_loss: {where} lacks {key!r}; has {sorted(mapping)}")
    value = mapping[key]
    if not isinstance(value, Tensor):
        raise TypeError(
            f"assemble_loss: {where}[{key!r}] must be a tensor, got {type(value).__name__}"
        )
    return value


def _weight(train: object, name: str) -> float:
    key = _LAMBDA[name]
    lam = float(getattr(train, key))
    if lam < 0.0:
        raise ValueError(f"train.{key} must be >= 0, got {lam}")
    return lam


def _check_extras(extras: Mapping[str, Tensor] | None) -> dict[str, Tensor]:
    """``extras`` as a dict of fp32 0-dim tensors; names must be terms a later chunk computes."""
    if extras is None:
        return {}
    if not isinstance(extras, Mapping):
        raise TypeError(f"assemble_loss: extras must be a mapping, got {type(extras).__name__}")
    checked: dict[str, Tensor] = {}
    for name, value in extras.items():
        if name not in _EXTRA_TERMS:
            computed = " (computed here, not taken from extras)" if name in TERMS else ""
            raise ValueError(
                f"assemble_loss: extras names {name!r}{computed}; it accepts {_EXTRA_TERMS}"
            )
        if not isinstance(value, Tensor) or not torch.is_floating_point(value) or value.ndim != 0:
            got = (
                f"{value.dtype} of shape {tuple(value.shape)}"
                if isinstance(value, Tensor)
                else type(value).__name__
            )
            raise ValueError(
                f"assemble_loss: extras[{name!r}] must be a 0-dim floating tensor, got {got}"
            )
        checked[name] = value.float()
    return checked


def _take_extra(name: str, given: dict[str, Tensor]) -> Tensor:
    if name not in given:
        raise ValueError(
            f"{name!r} is enabled (train.{_LAMBDA[name]} > 0) but extras carries no value for it; "
            f"its computation arrives with chunk {_SOURCE[name]}"
        )
    return given.pop(name)
