"""Per-unit lazy AdamW (P2.6).

:class:`LazyAdamW` is AdamW with decoupled weight decay (the semantics and the arithmetic of
``torch.optim.AdamW``) with one addition: a parameter group may declare its tensors as *stacked unit
tensors* ``[M, ...]`` by carrying ``{"unit_dim": 0, "n_units": M}``, and every :meth:`step` then
takes an ``active_units`` mask (bool ``[M]``). A unit the mask leaves inactive receives no update at
all — its weights, its first and second moments and its own step counter stay as they were — while
an active unit takes exactly the AdamW step it would take on its own, with the bias correction of
*its* step count and the weight decay applied at *its* step. Groups without the two keys are plain
AdamW. With every unit active (``active_units=None``) the optimiser is AdamW.

**Why.** Standard Adam applies a step whenever a parameter has a gradient tensor, even one of zeros:
the moments decay, the bias correction advances and the stale first moment keeps moving the weights.
The unit bank of a sparsely routed model (P2.5) always has such a gradient — :meth:`UnitBank.
forward_selected <hsn.model.units.UnitBank.forward_selected>` unbinds the stacked weights, so the
rows of the units that did not execute are exact zeros, not ``None`` — and AdamW would therefore
keep moving units that never ran. That is a correctness problem for sparse execution and it blocks
P4.4's unit offloading, which must know exactly which units changed. The mask is the only source of
truth: the optimiser never infers activity from ``grad is None`` or from a zero gradient, and a
gradient a mask marks inactive is ignored (``where(m, grad, 0)``).

**The update** of one stacked tensor (``m`` the mask broadcast to ``[M, 1, ...]``, every coefficient
in the parameter's dtype)::

    step[j]    += m[j]                                            int64 [M], hosted on the CPU
    p          *= where(m, 1 - lr·wd, 1)                          decay only the active units
    g           = where(m, grad, 0)
    exp_avg     = lerp(exp_avg, g, where(m, 1 - β1, 0))
    exp_avg_sq  = exp_avg_sq · where(m, β2, 1) + (1 - β2) · g²
    bc1, bc2    = 1 - β1^step, 1 - β2^step                        float64, per unit
    p          -= where(m, lr / bc1, 0) · exp_avg / (sqrt(exp_avg_sq) / where(m, sqrt(bc2), 1) + ε)

Inactive units see a decay factor of 1, a lerp weight of 0, a moment factor of 1 and a step size of
0, so their weights and step are bit-identical before and after and their moments equal (``torch.
equal``; a ``-0.0`` first moment can become ``+0.0``). The ``where`` on ``bc2`` keeps the step-0
division finite for a unit that has never been active. The op order of the active path is the
single-tensor path of ``torch.optim.AdamW`` (decay, ``lerp_``, ``mul_().addcmul_()``, float64 bias
corrections cast to the parameter dtype, ``sqrt() / bc2_sqrt + eps``, ``addcdiv_``), which makes an
all-active step bit-exact with it on the CPU; on CUDA the ``addcdiv_`` kernel rounds ``alpha·(a/b)``
instead of ``(alpha·a)/b`` and the two agree to one ulp of the update (measured: 3e-8 at |p| < 0.5).
The step counters live on the CPU exactly as torch hosts its own ``step`` state, so the bias
corrections are float64 on every backend (MPS has no float64) and the counters survive a checkpoint
loaded with ``map_location="cpu"`` untouched.

**Active units.** :func:`active_units_from_indices` turns the router's ``unit_idx`` / ``valid``
stacks of any leading shape into the mask (the union over the valid slots; padded slots hold unit 0
and are never counted; a dense-fallback row ran every unit) by a comparison against ``arange(M)`` —
no scatter, so deterministic mode is happy — and one device-to-host transfer that also checks every
valid index lies in ``[0, M)``. :meth:`LazyAdamW.active_mask_from_selection` is the same for one
:class:`~hsn.model.router.Selection`, :meth:`LazyAdamW.active_mask_from_selections` for a window of
them. The trainer (:mod:`hsn.train.tbptt`) builds the mask per window, ORs it over the windows of a
step and hands the CPU tensor to :meth:`step`; a soft-gated (masked) round or a dense-fallback row
executes every unit, which the trainer accounts for before calling in here.

**Groups.** :func:`build_lazy_adamw` builds the groups the P0.5 trainer builds for AdamW (weight
decay on tensors with ``ndim >= 2``, none on the rest) and marks the unit bank's four stacked
tensors (:class:`~hsn.model.units.UnitBank`: ``W1 [M, 4d, H]``, ``b1 [M, H]``, ``W2 [M, H, d]``,
``b2 [M, d]``) as unit groups; the shared parameters (E, P, S, D, V, W_out, the gate B, the scorer,
the edge logits, the embeddings) are never lazy (chunk "Do not"). A model without a unit bank is
refused: there is nothing lazy to do, and the configuration asked for it. **P1.13:** with
``cell_lr_mult != 1`` or a ``cell_weight_decay`` (``train.*``) the parameters of the HSN cell
(:func:`cell_parameters`; the unit bank is part of it) get their own groups at ``lr *
cell_lr_mult`` with ``cell_weight_decay`` on their ``ndim >= 2`` tensors; with the defaults the
groups are exactly the ones above.

**State and checkpoints.** Per tensor: ``exp_avg`` and ``exp_avg_sq`` like the parameter and
``step`` — int64 ``[M]`` on the CPU for a unit group, a 0-dim int64 CPU tensor otherwise — all
created at the first step like torch does. ``state_dict()`` / ``load_state_dict()`` are the base
class's (the per-unit steps travel inside the P0.5 checkpoint unchanged); the load additionally
refuses a state whose steps do not have the shapes of this optimiser's groups (a plain AdamW state,
for instance) or whose saved ``n_units`` disagrees with the parameters (plan §4.3: errors over
silence).

Later: P4.4 keeps cold units' state on the host (``HostAdamShard``) and updates resident units by
index; this version touches the whole stacked tensor with masks, which is right while ``M`` is tens.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from hsn.model.cell import PDGCell
from hsn.model.router import Selection
from hsn.model.units import UnitBank

__all__ = [
    "LazyAdamW",
    "active_units_from_indices",
    "build_lazy_adamw",
    "cell_parameters",
    "find_unit_bank",
]

_UNIT_KEYS = ("unit_dim", "n_units")
_INDEX_DTYPES = (torch.int64, torch.int32)


# --------------------------------------------------------------------------------------------------
# The active-unit mask
# --------------------------------------------------------------------------------------------------


def active_units_from_indices(
    unit_idx: Tensor,
    valid: Tensor,
    n_units: int,
    *,
    dense_rows: Tensor | None = None,
) -> Tensor:
    """The units that executed, bool ``[n_units]`` on the CPU: the union of ``unit_idx`` over the
    slots where ``valid`` is True, plus every unit when any entry of ``dense_rows`` is True.

    ``unit_idx`` (int64 or int32) and ``valid`` (bool) have the same shape of any rank (``[B, K]``
    for one selection, ``[B, T, K]`` for a window); ``dense_rows`` (bool, any shape) flags rows or
    positions that ran the dense step, which executes every unit. One device-to-host transfer brings
    the mask back together with a range check: a valid slot whose index lies outside
    ``[0, n_units)`` raises ``IndexError`` instead of being dropped.
    """
    if not isinstance(unit_idx, Tensor) or unit_idx.dtype not in _INDEX_DTYPES:
        got = unit_idx.dtype if isinstance(unit_idx, Tensor) else type(unit_idx).__name__
        raise TypeError(f"unit_idx must be an int64 or int32 tensor, got {got}")
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
        got = valid.dtype if isinstance(valid, Tensor) else type(valid).__name__
        raise TypeError(f"valid must be a bool tensor, got {got}")
    if valid.shape != unit_idx.shape:
        raise ValueError(
            f"valid must have the shape of unit_idx {tuple(unit_idx.shape)}, got "
            f"{tuple(valid.shape)}"
        )
    if isinstance(n_units, bool) or not isinstance(n_units, int) or n_units < 1:
        raise ValueError(f"n_units must be an int >= 1, got {n_units!r}")
    if dense_rows is not None and (
        not isinstance(dense_rows, Tensor) or dense_rows.dtype != torch.bool
    ):
        got = dense_rows.dtype if isinstance(dense_rows, Tensor) else type(dense_rows).__name__
        raise TypeError(f"dense_rows must be a bool tensor or None, got {got}")
    units = torch.arange(n_units, device=unit_idx.device, dtype=unit_idx.dtype)
    hits = (unit_idx.unsqueeze(-1) == units) & valid.unsqueeze(-1)  # [..., M]
    flat = hits.reshape(-1, n_units)
    mask = flat.any(dim=0)
    # every valid slot hit exactly one unit; a valid slot with an index outside [0, M) hits none
    in_range = (flat.any(dim=1) == valid.reshape(-1)).all()
    if dense_rows is not None:
        mask = mask | dense_rows.any().to(mask.device)
    packed = torch.cat([mask, in_range.reshape(1)]).cpu()
    if not bool(packed[-1]):
        raise IndexError(f"unit_idx has a valid slot outside [0, {n_units})")
    return packed[:-1]


# --------------------------------------------------------------------------------------------------
# The optimiser
# --------------------------------------------------------------------------------------------------


def _check_number(name: str, value: Any, *, low: float, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < low or (high is not None and value >= high):
        bound = f">= {low}" if high is None else f"in [{low}, {high})"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return float(value)


class LazyAdamW(torch.optim.Optimizer):
    """AdamW whose stacked unit tensors update only the active units (module docstring).

    ``params`` are tensors or parameter groups as for every torch optimiser; a group with
    ``{"unit_dim": 0, "n_units": M}`` holds stacked unit tensors ``[M, ...]`` (only ``unit_dim=0``
    exists: the bank stacks on dimension 0). Hyperparameters and their defaults are
    ``torch.optim.AdamW``'s: ``lr`` 1e-3, ``betas`` (0.9, 0.999), ``eps`` 1e-8, ``weight_decay``
    1e-2 (decoupled). No ``amsgrad``, ``fused`` or ``capturable`` variants.

    P2.10: the ordinary (non-unit) tensors of a group are stepped together through torch's
    multi-tensor ops (:meth:`_dense_steps`): the same operations per element as the
    single-tensor step (:meth:`_dense_step`, kept as the reference), bit for bit, without one
    Python round trip per tensor and operation. ``foreach = False`` restores the per-tensor loop.
    """

    #: Step the ordinary tensors of a group through the multi-tensor ops (class docstring).
    foreach: bool = True

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": tuple(betas),
            "eps": eps,
            "weight_decay": weight_decay,
        }
        _check_hyperparameters(defaults)
        super().__init__(params, defaults)

    # -- groups -----------------------------------------------------------------------------------

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a group (the base class fills the defaults); then check its hyperparameters and,
        for a unit group, ``unit_dim == 0``, ``n_units`` and every tensor's leading dimension."""
        super().add_param_group(param_group)
        try:
            self._check_groups()
        except ValueError:
            del self.param_groups[-1]
            raise

    def _check_groups(self) -> None:
        seen: set[int] = set()
        for index, group in enumerate(self.param_groups):
            _check_hyperparameters(group)
            present = [key for key in _UNIT_KEYS if key in group]
            if not present:
                continue
            if len(present) != len(_UNIT_KEYS):
                raise ValueError(
                    f"param group {index} carries {present} but a unit group needs both "
                    f"{list(_UNIT_KEYS)}"
                )
            unit_dim, n_units = group["unit_dim"], group["n_units"]
            if isinstance(unit_dim, bool) or not isinstance(unit_dim, int) or unit_dim != 0:
                raise ValueError(
                    f"param group {index}: unit_dim must be 0 (stacked unit tensors are [M, ...]), "
                    f"got {unit_dim!r}"
                )
            if isinstance(n_units, bool) or not isinstance(n_units, int) or n_units < 1:
                raise ValueError(
                    f"param group {index}: n_units must be an int >= 1, got {n_units!r}"
                )
            for p in group["params"]:
                if p.ndim < 1 or p.shape[0] != n_units:
                    raise ValueError(
                        f"param group {index}: every tensor of a unit group must be "
                        f"[n_units={n_units}, ...], got shape {tuple(p.shape)}"
                    )
            seen.add(n_units)
        if len(seen) > 1:
            raise ValueError(f"unit groups disagree on n_units: {sorted(seen)}")

    @property
    def n_units(self) -> int | None:
        """``M`` of the unit groups, or ``None`` when no group is a unit group."""
        values = {group["n_units"] for group in self.param_groups if "n_units" in group}
        if not values:
            return None
        (value,) = values
        return int(value)

    def _require_n_units(self) -> int:
        n_units = self.n_units
        if n_units is None:
            raise ValueError(
                "this LazyAdamW has no unit group ({'unit_dim': 0, 'n_units': M}); there are no "
                "units to build a mask for"
            )
        return n_units

    # -- masks ------------------------------------------------------------------------------------

    def active_mask_from_selection(self, sel: Selection) -> Tensor:
        """The units ``sel`` executed, bool ``[M]`` on the CPU: the union of ``sel.unit_idx`` over
        the valid slots (padded slots are never counted), every unit if a row of
        ``sel.dense_fallback`` ran the dense step."""
        if not isinstance(sel, Selection):
            raise TypeError(f"sel must be a Selection, got {type(sel).__name__}")
        return active_units_from_indices(
            sel.unit_idx, sel.valid, self._require_n_units(), dense_rows=sel.dense_fallback
        )

    def active_mask_from_selections(self, sels: Iterable[Selection]) -> Tensor:
        """The union of :meth:`active_mask_from_selection` over the selections of a window (any
        number of positions and rounds; ``K`` may differ between them)."""
        selections = list(sels)
        if not selections:
            raise ValueError("active_mask_from_selections needs at least one Selection")
        for sel in selections:
            if not isinstance(sel, Selection):
                raise TypeError(f"every item must be a Selection, got {type(sel).__name__}")
        return active_units_from_indices(
            torch.cat([sel.unit_idx.reshape(-1) for sel in selections]),
            torch.cat([sel.valid.reshape(-1) for sel in selections]),
            self._require_n_units(),
            dense_rows=torch.cat([sel.dense_fallback.reshape(-1) for sel in selections]),
        )

    def _check_mask(self, active_units: Tensor | None) -> Tensor | None:
        """``active_units`` validated and moved to the CPU, or ``None`` (every unit active)."""
        if active_units is None:
            return None
        n_units = self._require_n_units()
        if not isinstance(active_units, Tensor) or active_units.dtype != torch.bool:
            got = (
                active_units.dtype
                if isinstance(active_units, Tensor)
                else type(active_units).__name__
            )
            raise TypeError(f"active_units must be a bool tensor [M={n_units}] or None, got {got}")
        if tuple(active_units.shape) != (n_units,):
            raise ValueError(
                f"active_units must have shape [M] = ({n_units},), got {tuple(active_units.shape)}"
            )
        return active_units.detach().to("cpu")

    # -- state ------------------------------------------------------------------------------------

    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if "n_units" in group:
            state["step"] = torch.zeros(int(group["n_units"]), dtype=torch.int64)
        else:
            state["step"] = torch.zeros((), dtype=torch.int64)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """The base class's load, then a check that the loaded state is this optimiser's: the
        saved group keys (``unit_dim``, ``n_units``) fit the parameters and every ``step`` has the
        shape of its group (int64 ``[M]`` for a unit group, 0-dim otherwise); the moments must have
        the parameter's shape. A plain AdamW state (float32 scalar steps) is refused."""
        super().load_state_dict(state_dict)
        self._check_groups()
        for index, group in enumerate(self.param_groups):
            expected = (int(group["n_units"]),) if "n_units" in group else ()
            for p in group["params"]:
                if p not in self.state:
                    continue
                state = self.state[p]
                step = state.get("step")
                if (
                    not isinstance(step, Tensor)
                    or step.dtype != torch.int64
                    or tuple(step.shape) != expected
                ):
                    got_step = (
                        f"{step.dtype} {tuple(step.shape)}"
                        if isinstance(step, Tensor)
                        else repr(step)
                    )
                    raise ValueError(
                        f"param group {index}: the loaded step state must be int64 "
                        f"{list(expected)} for this group, got {got_step}; this is not a "
                        "LazyAdamW state of the same groups"
                    )
                if step.device.type != "cpu":
                    state["step"] = step.to("cpu")
                for key in ("exp_avg", "exp_avg_sq"):
                    moment = state.get(key)
                    if not isinstance(moment, Tensor) or moment.shape != p.shape:
                        got_moment = (
                            tuple(moment.shape) if isinstance(moment, Tensor) else repr(moment)
                        )
                        raise ValueError(
                            f"param group {index}: the loaded {key} must have the parameter's "
                            f"shape {tuple(p.shape)}, got {got_moment}"
                        )

    # -- the step ---------------------------------------------------------------------------------

    @torch.no_grad()
    def step(  # type: ignore[override]  # the spec's order; torch types step(closure)
        self,
        active_units: Tensor | None = None,
        closure: Callable[[], float] | None = None,
    ) -> None:
        """One optimiser step. ``active_units`` (bool ``[M]``, any device) marks the units of the
        unit groups that executed; ``None`` means every unit. ``closure`` is evaluated with
        gradients enabled before the step, for API compatibility; its value is not returned."""
        if closure is not None:
            with torch.enable_grad():
                closure()
        active_cpu = self._check_mask(active_units)
        masks: dict[torch.device, Tensor] = {}
        scalars: dict[tuple[float, torch.dtype, torch.device], Tensor] = {}
        for group in self.param_groups:
            lazy = "n_units" in group
            # the group's ordinary tensors by (step count, device, dtype): one multi-tensor step
            together: dict[tuple[int, torch.device, torch.dtype], list[list[Tensor]]] = {}
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("LazyAdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    self._init_state(p, state, group)
                if lazy:
                    self._unit_step(p, grad, state, group, active_cpu, masks, scalars)
                elif not self.foreach:
                    self._dense_step(p, grad, state, group)
                else:
                    state["step"] += 1
                    lists = together.setdefault(
                        (int(state["step"]), p.device, p.dtype), [[], [], [], []]
                    )
                    for members, tensor in zip(
                        lists, (p, grad, state["exp_avg"], state["exp_avg_sq"]), strict=True
                    ):
                        members.append(tensor)
            for (step, _, _), (params, grads, exp_avgs, exp_avg_sqs) in together.items():
                self._dense_steps(params, grads, exp_avgs, exp_avg_sqs, step, group)

    @staticmethod
    def _dense_steps(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        step: int,
        group: dict[str, Any],
    ) -> None:
        """:meth:`_dense_step` for several tensors at the same step count, through the
        multi-tensor ops: the same operation per element, in the same order."""
        lr, (beta1, beta2) = group["lr"], group["betas"]
        eps, weight_decay = group["eps"], group["weight_decay"]
        if weight_decay != 0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5
        denoms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denoms, bias_correction2_sqrt)
        torch._foreach_add_(denoms, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-step_size)

    @staticmethod
    def _dense_step(p: Tensor, grad: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """The single-tensor AdamW step of torch, op for op (an ordinary tensor)."""
        lr, (beta1, beta2) = group["lr"], group["betas"]
        eps, weight_decay = group["eps"], group["weight_decay"]
        exp_avg, exp_avg_sq, step_t = state["exp_avg"], state["exp_avg_sq"], state["step"]
        step_t += 1
        step = int(step_t)
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-step_size)

    @staticmethod
    def _unit_step(
        p: Tensor,
        grad: Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        active_cpu: Tensor | None,
        masks: dict[torch.device, Tensor],
        scalars: dict[tuple[float, torch.dtype, torch.device], Tensor] | None = None,
    ) -> None:
        """The masked, per-unit vectorised step of a stacked unit tensor (module docstring).
        ``scalars`` caches the 0-dim device constants of one :meth:`step` (each is otherwise a
        host-to-device copy per use)."""
        if scalars is None:
            scalars = {}
        n_units = int(group["n_units"])
        lr, (beta1, beta2) = group["lr"], group["betas"]
        eps, weight_decay = group["eps"], group["weight_decay"]
        exp_avg, exp_avg_sq, step_cpu = state["exp_avg"], state["exp_avg_sq"], state["step"]
        if tuple(step_cpu.shape) != (n_units,):
            raise RuntimeError(
                f"the step state of a [{n_units}, ...] unit tensor must be [{n_units}], got "
                f"{tuple(step_cpu.shape)}"
            )
        if active_cpu is None:
            active_cpu = torch.ones(n_units, dtype=torch.bool)
        step_cpu.add_(active_cpu.to(torch.int64))
        device, dtype = p.device, p.dtype
        act = masks.get(device)
        if act is None:
            act = masks[device] = active_cpu.to(device)
        view = (n_units,) + (1,) * (p.ndim - 1)
        actv = act.view(view)

        def const(value: float, kind: torch.dtype) -> Tensor:
            key = (float(value), kind, device)
            found = scalars.get(key)
            if found is None:
                found = scalars[key] = torch.tensor(value, dtype=kind, device=device)
            return found

        def coef(on: float, off: float) -> Tensor:
            return torch.where(actv, const(on, dtype), const(off, dtype))

        g = torch.where(actv, grad, const(0.0, grad.dtype))
        if weight_decay != 0:
            p.mul_(coef(1 - lr * weight_decay, 1.0))
        exp_avg.lerp_(g, coef(1 - beta1, 0.0))
        exp_avg_sq.mul_(coef(beta2, 1.0)).addcmul_(g, g, value=1 - beta2)
        # per-unit bias corrections in float64 on the host, like torch's Python-float ones
        steps = step_cpu.to(torch.float64)
        bias_correction1 = 1 - torch.pow(beta1, steps)
        bias_correction2 = 1 - torch.pow(beta2, steps)
        zero = torch.zeros((), dtype=torch.float64)
        one = torch.ones((), dtype=torch.float64)
        step_size = torch.where(active_cpu, lr / bias_correction1, zero).to(dtype).to(device)
        bias_correction2_sqrt = (
            torch.where(active_cpu, bias_correction2.pow(0.5), one).to(dtype).to(device)
        )
        denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt.view(view)).add_(eps)
        p.addcdiv_(exp_avg.mul(step_size.view(view)), denom, value=-1.0)


def _check_hyperparameters(group: dict[str, Any]) -> None:
    _check_number("lr", group["lr"], low=0.0)
    betas = group["betas"]
    if not isinstance(betas, tuple | list) or len(betas) != 2:
        raise ValueError(f"betas must be a pair, got {betas!r}")
    _check_number("betas[0]", betas[0], low=0.0, high=1.0)
    _check_number("betas[1]", betas[1], low=0.0, high=1.0)
    group["betas"] = (float(betas[0]), float(betas[1]))
    _check_number("eps", group["eps"], low=0.0)
    _check_number("weight_decay", group["weight_decay"], low=0.0)


# --------------------------------------------------------------------------------------------------
# Building the optimiser for a model
# --------------------------------------------------------------------------------------------------


def find_unit_bank(model: nn.Module) -> UnitBank:
    """The one :class:`~hsn.model.units.UnitBank` in ``model.modules()``; none or several raise."""
    banks = [module for module in model.modules() if isinstance(module, UnitBank)]
    if len(banks) != 1:
        raise ValueError(
            f"train.optimizer=lazy_adamw needs a model with exactly one UnitBank (the HSN unit "
            f"bank); {type(model).__name__} has {len(banks)}. Use train.optimizer=adamw for "
            "models without stacked unit tensors."
        )
    return banks[0]


def cell_parameters(model: nn.Module) -> list[nn.Parameter]:
    """P1.13: the trainable parameters of the one :class:`~hsn.model.cell.PDGCell` in ``model``
    (``HSN.cell``): ``V``, ``W_out``, the gate, the input and candidate norms, the unit bank, the
    graph's edge logits and anything a P1.11 variant installs under the cell, in module order. The
    set is exactly the ``cell`` group of the P1.11 diagnostics (``grad_share/cell``). A model
    without one PDGCell (the baselines) raises: ``train.cell_lr_mult`` and
    ``train.cell_weight_decay`` act on the HSN cell only."""
    cells = [module for module in model.modules() if isinstance(module, PDGCell)]
    if len(cells) != 1:
        raise ValueError(
            f"train.cell_lr_mult / train.cell_weight_decay act on the HSN cell (one PDGCell); "
            f"{type(model).__name__} has {len(cells)}. Leave them at their defaults (1.0 / null)."
        )
    return [p for p in cells[0].parameters() if p.requires_grad]


def build_lazy_adamw(
    model: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    cell_lr_mult: float = 1.0,
    cell_weight_decay: float | None = None,
) -> LazyAdamW:
    """The :class:`LazyAdamW` of ``model`` with the P0.5 groups (``weight_decay`` on tensors with
    ``ndim >= 2``, none on the rest), the unit bank's tensors in unit groups
    (``unit_dim=0, n_units=M``) and every other parameter in ordinary groups. Group order:
    shared decay, shared no-decay, unit decay, unit no-decay (empty groups are left out).

    P1.13: with ``cell_lr_mult != 1`` or ``cell_weight_decay`` given, the cell's parameters
    (:func:`cell_parameters`) that are not units get groups of their own between the shared and
    the unit groups, and the cell and unit groups train at ``lr * cell_lr_mult`` with
    ``cell_weight_decay`` (``None``: ``weight_decay``) on their ``ndim >= 2`` tensors. Group order:
    shared decay, shared no-decay, cell decay, cell no-decay, unit decay, unit no-decay."""
    bank = find_unit_bank(model)
    unit_ids = {id(p) for p in bank.parameters()}
    split = cell_lr_mult != 1.0 or cell_weight_decay is not None
    cell_ids = {id(p) for p in cell_parameters(model)} if split else set()
    # bucket key: (kind, decays), kind 0 shared, 1 cell (not units), 2 units
    buckets: dict[tuple[int, bool], list[Tensor]] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        kind = 2 if id(p) in unit_ids else 1 if id(p) in cell_ids else 0
        buckets.setdefault((kind, p.ndim >= 2), []).append(p)
    if not buckets:
        raise ValueError("the model has no trainable parameters")
    if not any(kind == 2 for kind, _ in buckets):
        raise ValueError(
            "train.optimizer=lazy_adamw: the unit bank has no trainable parameters, so there is "
            "nothing lazy to update; use train.optimizer=adamw"
        )
    cell_decay = weight_decay if cell_weight_decay is None else cell_weight_decay
    groups: list[dict[str, Any]] = []
    for kind in (0, 1, 2):
        for decays in (True, False):
            params = buckets.get((kind, decays))
            if not params:
                continue
            group: dict[str, Any] = {
                "params": params,
                "weight_decay": (cell_decay if kind and split else weight_decay) if decays else 0.0,
            }
            if kind and split:  # P1.13: the cell's own learning rate
                group["lr"] = lr * cell_lr_mult
            if kind == 2:
                group["unit_dim"] = 0
                group["n_units"] = bank.n_units
            groups.append(group)
    return LazyAdamW(groups, lr=lr, betas=betas, weight_decay=weight_decay)
