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
formula order, so each can be logged and its gradient inspected. :func:`assemble_terms` is the same
assembly returning the λ-weighted term *tensors* (formula order) and :func:`total_and_parts` turns
them into ``(total, parts)``; :func:`assemble_loss` is their composition.

P2.4: the auxiliary terms of §2.6 and the per-term gradient report, under the one-at-a-time rule::

    cost_loss(selection, reference)  = mean_b selection.expected_cost[b] / reference
    balance_loss(logits, unit_idx, valid)
                                     = M · Σ_j f_j · p̄_j
                                       f_j = valid slots routed to unit j / valid slots
                                       p̄_j = mean over every (row, cell) of softmax(logits)[..., j]
    graph_loss(graph)                = Σ_(i,j) |a_ij| = graph.l1()

``cost_loss`` reads :attr:`~hsn.model.router.Selection.expected_cost`, the differentiable expected
cost of P2.2, not the detached accounting ``cost``: the two are equal in the hard modes and only the
former carries the gate gradient. Every profiled cost table is uniform over units
(:mod:`hsn.kernels.costs`), so that gradient flows through the cell mask alone (the soft warm-up
and the straight-through mask); under REINFORCE the mask is hard and the penalty reaches the sampled
cells only through the reward (the trainer adds the λ-weighted cost part to it); with ``random``
routing the value carries no gradient and with ``fixed`` a zero one (the controls take no routing
gradient, and the report below shows 0). In the soft warm-up the penalty is the relaxed mask's cost
while the charged cost (``route_cost`` in the trainer's statistics) stays dense; dense-fallback rows
are the constant dense cost. ``reference_cost`` is the dense step in table units
(:meth:`~hsn.kernels.costs.CostTable.dense_cost`, 1.0 by construction). ``balance_loss`` is the
Switch-Transformer load-balancing term: ``f`` counts valid (non-padded) slots only — a padded slot
holds unit 0 and would bias the count — and ``p̄`` averages the unit softmax of every scored cell
(the signature carries no ``cell_idx``; every cell has a live unit softmax: the warm-up and the
dense-fallback rows execute all of them). It is exactly 1 when both ``f`` and ``p̄`` are uniform
and rises towards ``M`` as both concentrate on one unit; the gradient reaches the scorer through
``p̄``. ``graph_loss`` is the §2.6 sum as written: every edge's gradient is then ``λ_g · σ'(l_e)``
whatever the edge count, a per-edge L1 prior, so ``λ_g`` transfers across ``N`` and ``D_g`` and
survives SAN changing the edge count (a per-edge mean would silently weaken the penalty as the
graph grows); the sigmoid bounds every ``|a_ij| ≤ 1``. Under ``model.edge_norm = softmax`` the
sum is a constant with no gradient, and an edgeless graph has nothing to shrink: both are refused.

:func:`auxiliary_terms` is the window-level bridge the trainer calls: from the ``info`` of
:meth:`hsn.model.hsn.HSN.forward` it computes the raw value of every term named in
``train.aux_enabled`` — ``cost`` as the mean over positions of ``info["expected_cost"]`` (the
``(1/T) Σ_t`` of §2.6) over the reference cost, ``balance`` from ``info["unit_logits"]``,
``info["unit_idx"]`` and ``info["valid"]`` over the whole window, ``graph`` from the model's
:class:`~hsn.model.graph.CellGraph` — and returns exactly the ``extras`` the assembly accepts. A
term that needs a selection (cost, balance) under dense routing raises.

:func:`grad_norm_report` is the inspection the normalisation is for: for every term tensor it takes
``torch.autograd.grad`` over the model's trainable parameters (``retain_graph``, never touching
``.grad``) and reports the global L2 norm as ``term_grad_norm/<term>``; a term without a gradient
path (a constant, or one that reaches no parameter) reports 0.0 so the keys are predictable.
``scale`` is the fp16 loss scale: the term is scaled before the backward and the norm divided back,
because an unscaled fp16 backward underflows (:mod:`hsn.train.amp`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import AUX_TERMS, PredictorTarget, RunConfig
from hsn.model.graph import CellGraph
from hsn.model.router import Selection

__all__ = [
    "TERM_GRAD_NORM_PREFIX",
    "TERMS",
    "assemble_loss",
    "assemble_terms",
    "auxiliary_terms",
    "balance_loss",
    "cost_loss",
    "grad_norm_report",
    "graph_loss",
    "pred_loss",
    "prediction_terms",
    "total_and_parts",
    "variance_floor",
]

#: The key prefix of the per-term gradient norms of :func:`grad_norm_report` (P2.4); distinct from
#: the trainer's per-parameter-group ``grad_norm/<group>``.
TERM_GRAD_NORM_PREFIX = "term_grad_norm/"

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


def assemble_terms(
    out: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: RunConfig,
    extras: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """The λ-weighted terms of the §2.6 objective of one window, in formula order (module
    docstring); :func:`assemble_loss` sums them.

    ``out`` is the model's window output: ``logits [B, T, V]`` plus the ``info`` of
    :meth:`hsn.model.hsn.HSN.forward` (``z_hat``, ``z_target``, ``z`` ``[B, T, d]`` for the
    prediction term). ``batch`` is the loader window: ``targets`` (int64 ``[B, T]``), ``loss_mask``
    (bool ``[B, T]``) and ``doc`` (int64 ``[B, T]``, ``-1`` on padding). ``cfg`` is the run
    configuration (``train.lambda_*``, ``train.aux_enabled``, ``train.allow_multiple_aux``,
    ``train.variance_floor_gamma`` and ``model.predictor_target``). ``extras`` maps the name of
    every enabled replay / distill / ewc / cost / balance / graph term to its raw (unweighted) 0-dim
    value (:func:`auxiliary_terms` builds the P2.4 ones). Every value is a 0-dim fp32 tensor with
    its graph (``variance_floor`` follows ``pred`` in EMA mode). A window without loss positions
    contributes a zero task term with a zero gradient.
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
    return terms


def total_and_parts(terms: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
    """``(total, parts)`` of λ-weighted term tensors: ``total`` is their sum in the given order
    starting from ``task`` (a 0-dim tensor with the graph), ``parts`` every term as a float."""
    if not isinstance(terms, Mapping) or "task" not in terms:
        raise ValueError("total_and_parts: terms must be a mapping that holds the 'task' term")
    for name, value in terms.items():
        if not isinstance(value, Tensor) or not torch.is_floating_point(value) or value.ndim != 0:
            raise ValueError(f"total_and_parts: terms[{name!r}] must be a 0-dim floating tensor")
    total = terms["task"]
    for name, value in terms.items():
        if name != "task":
            total = total + value
    parts = {name: float(value.detach()) for name, value in terms.items()}
    return total, parts


def assemble_loss(
    out: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: RunConfig,
    extras: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """The §2.6 objective of one window: ``(total, parts)`` (module docstring), i.e.
    :func:`total_and_parts` of :func:`assemble_terms` with the same arguments. ``total`` is a 0-dim
    fp32 tensor with the graph; ``parts`` the λ-weighted value of every included term as a float,
    in formula order."""
    return total_and_parts(assemble_terms(out, batch, cfg, extras))


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


# --------------------------------------------------------------------------------------------------
# P2.4: the auxiliary terms and the per-term gradient report
# --------------------------------------------------------------------------------------------------


def cost_loss(selection: Selection, reference_cost: float = 1.0) -> Tensor:
    """``L_cost`` of one round (module docstring): the batch mean of the selection's differentiable
    ``expected_cost`` over ``reference_cost`` (the dense step in table units), a 0-dim fp32 tensor.
    ``reference_cost`` must be a finite number > 0."""
    if not isinstance(selection, Selection):
        raise TypeError(f"cost_loss needs a Selection, got {type(selection).__name__}")
    reference = _check_reference_cost(reference_cost)
    return selection.expected_cost.float().mean() / reference


def balance_loss(logits: Tensor, unit_idx: Tensor, valid: Tensor) -> Tensor:
    """``L_balance = M · Σ_j f_j · p̄_j`` (module docstring) of the scorer ``logits [..., N, M]``
    and the routed slots ``unit_idx`` (int64 ``[..., K]``) masked by ``valid`` (bool ``[..., K]``),
    with the same leading shape (``[B]`` for one round, ``[B, T]`` for a window). ``f`` counts the
    valid slots only; ``p̄`` is the mean unit softmax (fp32) over every leading index and cell.
    Returns a 0-dim fp32 tensor. Refuses a window without a valid slot, an index outside
    ``[0, M)``, and mismatched shapes, dtypes or devices."""
    _check_tensor("logits", logits)
    if logits.ndim < 3:
        raise ValueError(
            f"balance_loss: logits must be [..., N, M] with at least one leading dimension, got "
            f"shape {tuple(logits.shape)}"
        )
    for name, x, dtype in (("unit_idx", unit_idx, torch.int64), ("valid", valid, torch.bool)):
        if not isinstance(x, Tensor):
            raise TypeError(f"balance_loss: {name} must be a torch.Tensor, got {type(x).__name__}")
        if x.dtype != dtype:
            raise ValueError(f"balance_loss: {name} must have dtype {dtype}, got {x.dtype}")
        if x.device != logits.device:
            raise ValueError(
                f"balance_loss: {name} lives on {x.device} but logits on {logits.device}"
            )
    lead = tuple(logits.shape[:-2])
    M = int(logits.shape[-1])
    if unit_idx.ndim != len(lead) + 1 or tuple(unit_idx.shape[:-1]) != lead:
        raise ValueError(
            f"balance_loss: unit_idx must be [{', '.join(str(n) for n in lead)}, K] to match "
            f"logits {tuple(logits.shape)}, got {tuple(unit_idx.shape)}"
        )
    if valid.shape != unit_idx.shape:
        raise ValueError(
            f"balance_loss: valid must have unit_idx's shape {tuple(unit_idx.shape)}, got "
            f"{tuple(valid.shape)}"
        )
    routed = unit_idx[valid]  # the units of the valid slots, [n_valid]
    n_valid = int(routed.numel())
    if n_valid == 0:
        raise ValueError("balance_loss: no valid slot; the balance of nothing is undefined")
    lo, hi = int(routed.min()), int(routed.max())
    if lo < 0 or hi >= M:
        raise ValueError(
            f"balance_loss: unit_idx holds {lo if lo < 0 else hi}, outside [0, {M}) for M = {M}"
        )
    fraction = torch.bincount(routed, minlength=M).to(torch.float32) / n_valid  # f_j [M]
    probability = torch.softmax(logits.float(), dim=-1).reshape(-1, M).mean(dim=0)  # p̄_j [M]
    return M * (fraction * probability).sum()


def graph_loss(graph: CellGraph) -> Tensor:
    """``L_graph = Σ_(i,j) |a_ij|`` over the learned edge weights of ``graph`` (module docstring),
    a 0-dim fp32 tensor. Refuses a graph whose ``edge_norm`` is not ``sigmoid`` (under ``softmax``
    the sum is a constant without gradient) and an edgeless graph (nothing to shrink)."""
    if not isinstance(graph, CellGraph):
        raise TypeError(f"graph_loss needs a CellGraph, got {type(graph).__name__}")
    if graph.edge_norm != "sigmoid":
        raise ValueError(
            f"graph_loss: the graph normalises its edges with {graph.edge_norm!r}, under which "
            "Σ|a_ij| is the constant number of non-empty rows and carries no gradient; set "
            "model.edge_norm=sigmoid to use the graph penalty"
        )
    if graph.n_edges == 0:
        raise ValueError(
            "graph_loss: the graph has no edge (model.in_degree + model.random_edges = 0), so "
            "the penalty has nothing to shrink; drop 'graph' from train.aux_enabled"
        )
    return graph.l1().float()


def auxiliary_terms(
    info: Mapping[str, Tensor],
    cfg: RunConfig,
    graph: CellGraph | None,
    *,
    reference_cost: float = 1.0,
) -> dict[str, Tensor]:
    """The raw values of the P2.4 terms named in ``cfg.train.aux_enabled`` for one window (module
    docstring), from the ``info`` of :meth:`hsn.model.hsn.HSN.forward` and the model's ``graph``:
    exactly the ``extras`` :func:`assemble_terms` accepts (``{}`` when none is enabled). ``cost``
    and ``balance`` need the routed keys of ``info`` (``expected_cost``, ``unit_logits``,
    ``unit_idx``, ``valid``) and raise under dense routing; ``graph`` needs ``graph``.
    ``reference_cost`` is the dense step in table units."""
    if not isinstance(cfg, RunConfig):
        raise TypeError(f"auxiliary_terms needs the RunConfig, got {type(cfg).__name__}")
    if not isinstance(info, Mapping):
        raise TypeError(f"auxiliary_terms: info must be a mapping, got {type(info).__name__}")
    enabled = cfg.train.aux_enabled
    terms: dict[str, Tensor] = {}
    for name in AUX_TERMS:  # formula order
        if name not in enabled:
            continue
        if name == "cost":
            expected = _routed_field(info, "expected_cost", name)
            terms[name] = expected.float().mean() / _check_reference_cost(reference_cost)
        elif name == "balance":
            terms[name] = balance_loss(
                _routed_field(info, "unit_logits", name),
                _routed_field(info, "unit_idx", name),
                _routed_field(info, "valid", name),
            )
        else:  # graph
            if graph is None:
                raise ValueError(
                    "auxiliary_terms: 'graph' is in train.aux_enabled but no CellGraph was given"
                )
            terms[name] = graph_loss(graph)
    return terms


def grad_norm_report(
    model: nn.Module, parts: Mapping[str, Tensor], *, scale: float = 1.0
) -> dict[str, float]:
    """``{"term_grad_norm/<term>": ‖∂term/∂θ‖₂}`` over the trainable parameters of ``model`` for
    every 0-dim term tensor in ``parts`` (module docstring). Each norm is taken with
    ``torch.autograd.grad`` on ``scale · term`` (``retain_graph=True``, so the graph stays usable
    for the training backward; ``.grad`` is never touched) and divided by ``scale``, the fp16 loss
    scale (1.0 otherwise). A term that requires no grad or reaches no parameter reports 0.0; a
    non-finite norm is reported as it is."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"grad_norm_report needs an nn.Module, got {type(model).__name__}")
    if not isinstance(parts, Mapping):
        raise TypeError(f"grad_norm_report: parts must be a mapping, got {type(parts).__name__}")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, int | float)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(f"grad_norm_report: scale must be a finite number > 0, got {scale!r}")
    params = [p for p in model.parameters() if p.requires_grad]
    report: dict[str, float] = {}
    for name, term in parts.items():
        if not isinstance(term, Tensor) or not torch.is_floating_point(term) or term.ndim != 0:
            raise ValueError(f"grad_norm_report: parts[{name!r}] must be a 0-dim floating tensor")
        key = f"{TERM_GRAD_NORM_PREFIX}{name}"
        if not term.requires_grad or not params:
            report[key] = 0.0
            continue
        scaled = term * scale if scale != 1.0 else term
        grads = torch.autograd.grad(scaled, params, retain_graph=True, allow_unused=True)
        norms = [torch.linalg.vector_norm(g.detach().float()) for g in grads if g is not None]
        if not norms:
            report[key] = 0.0
            continue
        report[key] = float(torch.linalg.vector_norm(torch.stack(norms))) / scale
    return report


def _check_reference_cost(reference_cost: float) -> float:
    if (
        isinstance(reference_cost, bool)
        or not isinstance(reference_cost, int | float)
        or not math.isfinite(reference_cost)
        or reference_cost <= 0.0
    ):
        raise ValueError(f"reference_cost must be a finite number > 0, got {reference_cost!r}")
    return float(reference_cost)


def _routed_field(info: Mapping[str, Tensor], key: str, term: str) -> Tensor:
    if key not in info:
        raise ValueError(
            f"{term!r} is in train.aux_enabled but the window info has no {key!r}: only a routed "
            "model.routing (topk, greedy_budget, random, fixed) produces a selection to penalise; "
            f"choose one or drop {term!r}"
        )
    value = info[key]
    if not isinstance(value, Tensor):
        raise TypeError(
            f"auxiliary_terms: info[{key!r}] must be a tensor, got {type(value).__name__}"
        )
    return value
