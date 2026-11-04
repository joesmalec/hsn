"""The dense two-layer MLPs of the HSN step (P1.2): encoder E, predictor P, shared transform S,
decoder D, all instances of :class:`MLP2` with hidden width ``H = model.hidden`` (512 in A.1).

Plan §2.1 / §2.2 and Appendix C.1:

* ``E``: ``z_t = E(emb(x_t))``, ``d -> H -> d`` (:class:`Encoder`).
* ``P``: ``ẑ_t = P(q_{t-1})``, ``d -> H -> d`` (:class:`Predictor`), computed *before* ``x_t`` is
  encoded (global rule 2; the ordering is P1.5's job, this module only provides the map).
* ``S``: ``u_t = S([z_t; q_{t-1}; ctx_t])``, ``(2d + context_dim) -> H -> d``
  (:class:`SharedTransform`). The chunk writes the input width as ``3d``, which is the A.1 value
  (``context_dim = d``); using ``2d + context_dim`` keeps the module equal to the P1.1 closed form
  ``shared`` for every configuration.
* ``D``: ``logits_t = D([q_t; u_t])``, ``2d -> H -> V`` (:class:`Decoder`).

The activation is ``model.activation`` (``gelu`` in A.1, taken as the exact erf form; the P2.9
kernels must compute the same function) and the weights follow ``model.init`` through
:func:`hsn.model.embeddings.init_weight_`; biases are zero. Both values are required arguments
rather than code defaults, so they always come from the config, and unknown names raise (errors
over silence). ``MLP2`` applies to any leading shape, ``[..., in_dim] -> [..., out_dim]``, and
raises when the last dimension is not ``in_dim`` rather than letting the matmul error speak.

P1.6 adds :class:`EMATarget`, the stop-gradient target encoder of ``model.predictor_target =
ema_encoder``: a frozen deep copy of ``E`` that :meth:`EMATarget.update` moves toward the online
``E`` by ``p_t ← τ·p_t + (1 − τ)·p_online`` after each optimiser step (``train.ema_tau``).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.embeddings import init_weight_

__all__ = [
    "ACTIVATIONS",
    "MLP2",
    "Decoder",
    "EMATarget",
    "Encoder",
    "Predictor",
    "SharedTransform",
    "make_activation",
]

#: ``model.activation`` names and the module each builds. ``gelu`` is the exact (erf) GELU.
ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def make_activation(name: str) -> nn.Module:
    """The activation module ``model.activation`` names; unknown names raise ``ValueError``."""
    try:
        factory = ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"model.activation={name!r} is not supported; known: {sorted(ACTIVATIONS)}"
        ) from None
    return factory()


class MLP2(nn.Module):
    """Two-layer MLP ``in_dim -> hidden -> out_dim`` with ``activation`` between the layers.

    ``forward(x[..., in_dim]) -> [..., out_dim]``. ``activation`` and ``init`` are the config's
    ``model.activation`` / ``model.init`` (no defaults); both biases start at zero. The parameter
    count is ``in·hidden + hidden + hidden·out + out``, the ``_mlp`` closed form of
    :mod:`hsn.train.memory_plan`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        activation: str,
        *,
        init: str,
    ) -> None:
        super().__init__()
        for name, value in (("in_dim", in_dim), ("hidden", hidden), ("out_dim", out_dim)):
            if value < 1:
                raise ValueError(f"{type(self).__name__} needs {name} >= 1, got {value}")
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.activation_name = activation
        self.init = init
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = make_activation(activation)
        self.fc2 = nn.Linear(hidden, out_dim)
        for linear in (self.fc1, self.fc2):
            init_weight_(linear.weight, init)
            nn.init.zeros_(linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"{type(self).__name__} expects input [..., {self.in_dim}], got shape "
                f"{tuple(x.shape)}"
            )
        out: Tensor = self.fc2(self.act(self.fc1(x)))
        return out

    def param_count(self) -> int:
        """Measured parameter count (weights and biases of both layers)."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, hidden={self.hidden}, out_dim={self.out_dim}, "
            f"activation={self.activation_name!r}, init={self.init!r}"
        )


class Encoder(MLP2):
    """``E``: encoded input ``z_t = E(emb(x_t))``, ``d -> H -> d`` (C.1 row ``encoder``)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg.d, cfg.hidden, cfg.d, cfg.activation, init=cfg.init)


class Predictor(MLP2):
    """``P``: predicted representation ``ẑ_t = P(q_{t-1})``, ``d -> H -> d`` (C.1 ``predictor``)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg.d, cfg.hidden, cfg.d, cfg.activation, init=cfg.init)


class SharedTransform(MLP2):
    """``S``: shared features ``u_t = S([z_t; q_{t-1}; ctx_t])``, ``(2d + context_dim) -> H -> d``
    (C.1 row ``shared``; ``3d`` in A.1 where ``context_dim = d``)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(
            2 * cfg.d + cfg.context_dim, cfg.hidden, cfg.d, cfg.activation, init=cfg.init
        )


class Decoder(MLP2):
    """``D``: next-symbol logits ``D([q_t; u_t])``, ``2d -> H -> V`` (C.1 row ``decoder``)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(2 * cfg.d, cfg.hidden, cfg.vocab_size, cfg.activation, init=cfg.init)


class EMATarget(nn.Module):
    """A frozen copy of ``online`` whose tensors follow the online module by an exponential moving
    average (P1.6; BYOL-style stop-gradient target).

    The copy is a deep copy taken at construction, so the target starts equal to the online
    module. Its parameters are re-registered as **buffers**: they are not trainable, never reach an
    optimiser or ``parameters()`` (so :meth:`hsn.model.hsn.HSN.param_report` counts only trainable
    parameters, as before), yet they are part of ``state_dict()`` and move with ``.to()`` /
    ``.double()``, so a checkpoint restores the target exactly. :meth:`forward` runs the copy under
    ``torch.no_grad()`` (its output never requires grad) and :meth:`update` applies
    ``p_t ← τ·p_t + (1 − τ)·p_online`` to every parameter under ``no_grad``; buffers of the online
    module are copied. The online module is held by reference, not as a submodule (it is saved
    once, by its owner). ``τ`` (``train.ema_tau``) lies in ``[0, 1]``: 1 freezes the target, 0
    copies the online module at every update. Call :meth:`update` after each optimiser step.
    """

    _online: nn.Module

    def __init__(self, online: nn.Module, tau: float) -> None:
        super().__init__()
        if not isinstance(online, nn.Module):
            raise TypeError(f"EMATarget needs an nn.Module to copy, got {type(online).__name__}")
        if isinstance(tau, bool) or not isinstance(tau, int | float):
            raise TypeError(f"tau must be a float in [0, 1], got {type(tau).__name__}")
        if not (math.isfinite(tau) and 0.0 <= tau <= 1.0):
            raise ValueError(f"train.ema_tau must lie in [0, 1], got {tau}")
        self.tau = float(tau)
        self.module = copy.deepcopy(online)
        self._param_names: tuple[str, ...] = tuple(name for name, _ in online.named_parameters())
        for module in self.module.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                delattr(module, name)
                module.register_buffer(name, param.detach().clone())
        object.__setattr__(self, "_online", online)  # a reference, not a registered submodule

    def forward(self, x: Tensor) -> Tensor:
        """The copy applied to ``x`` under ``no_grad``: the output does not require grad."""
        with torch.no_grad():
            out: Tensor = self.module(x)
        return out

    def update(self) -> None:
        """``p_t ← τ·p_t + (1 − τ)·p_online`` for every parameter name of the online module, and
        a copy of its buffers; under ``no_grad``. Raises when the two no longer line up."""
        targets = dict(self.module.named_buffers())
        with torch.no_grad():
            for name, p_online in self._online.named_parameters():
                p_t = self._target_tensor(targets, name, p_online)
                p_t.mul_(self.tau).add_(p_online.detach(), alpha=1.0 - self.tau)
            for name, b_online in self._online.named_buffers():
                self._target_tensor(targets, name, b_online).copy_(b_online)

    def param_count(self) -> int:
        """Stored (non-trainable) values of the copy that mirror the online parameters."""
        targets = dict(self.module.named_buffers())
        return sum(targets[name].numel() for name in self._param_names)

    def extra_repr(self) -> str:
        return f"tau={self.tau}, online={type(self._online).__name__}"

    def _target_tensor(self, targets: dict[str, Tensor], name: str, online: Tensor) -> Tensor:
        try:
            target = targets[name]
        except KeyError:
            raise RuntimeError(
                f"EMATarget: the online module has a tensor {name!r} that the target copy lacks; "
                "the two modules no longer share a layout"
            ) from None
        if target.shape != online.shape:
            raise RuntimeError(
                f"EMATarget: {name!r} is {tuple(online.shape)} online but "
                f"{tuple(target.shape)} in the target copy"
            )
        if target.dtype != online.dtype or target.device != online.device:
            raise RuntimeError(
                f"EMATarget: {name!r} is {online.dtype} on {online.device} online but "
                f"{target.dtype} on {target.device} in the target copy; move the owner module as "
                "a whole"
            )
        return target
