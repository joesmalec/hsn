"""Byte and context embeddings (P1.2), and the weight initialiser that ``model.init`` names.

:class:`ByteEmbedding` turns byte ids ``x_t`` into width-``d`` vectors that the encoder ``E``
consumes (plan §2.1, ``z_t = E(x_t)``); :class:`ContextEmbedding` turns context symbol ids
``context_t`` (language, engine, version ... 16 symbols in A.1) into the ``ctx_t`` slice of the
shared-transform input ``[z_t; q_{t-1}; ctx_t]``. Both are lookup tables whose sizes come from
:class:`~hsn.config.ModelConfig` (``vocab_size × d`` and ``n_context × context_dim``), the two
Appendix C.1 rows ``byte_emb`` and ``ctx_emb``.

The paper fixes neither the activation nor the initialisation of the dense modules; P1.2 uses
``trunc_normal(std=0.02)`` with zero biases and records the choice in ``model.init`` (Appendix A.1,
default ``trunc_normal_0.02``). :func:`init_weight_` is the one place that interprets that string,
for every P1.2 module. ``trunc_normal_<std>`` means N(0, std²) truncated to ``[-2·std, 2·std]`` (the
usual meaning of a truncated normal, as in TensorFlow's ``truncated_normal``). PyTorch's
``nn.init.trunc_normal_`` defaults its bounds to ±2 in *absolute* units, which at std 0.02 would
truncate nothing, so the bounds are passed explicitly. Truncation at ±2σ lowers the effective
standard deviation to ≈ 0.88·std (0.0176 for ``trunc_normal_0.02``); the P1.2 tests measure it.
Any other ``model.init`` string raises (errors over silence); new schemes are added here when a
config needs them.
"""

from __future__ import annotations

import math
import re
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig

__all__ = ["ByteEmbedding", "ContextEmbedding", "init_weight_", "parse_init", "replace_content"]

_TRUNC_NORMAL_RE = re.compile(r"^trunc_normal_(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")
_INDEX_DTYPES = (torch.int64, torch.int32)


def parse_init(init: str) -> float:
    """Standard deviation named by a ``model.init`` string ``trunc_normal_<std>``.

    Raises ``ValueError`` for any other form or a non-positive std.
    """
    match = _TRUNC_NORMAL_RE.match(init)
    if match is None:
        raise ValueError(
            f"model.init={init!r} is not supported; expected 'trunc_normal_<std>' such as "
            "'trunc_normal_0.02' (Appendix A.1)"
        )
    std = float(match.group(1))
    if not (math.isfinite(std) and std > 0.0):
        raise ValueError(
            f"model.init={init!r} names a standard deviation that is not finite and > 0"
        )
    return std


def init_weight_(
    weight: Tensor,
    init: str,
    *,
    fan_in: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Initialise ``weight`` in place as ``model.init`` / ``model.state_init`` asks and return it.

    ``trunc_normal_<std>``: N(0, std²) truncated to ``[-2·std, 2·std]`` (see the module docstring
    for why the bounds are explicit). ``fan_in`` (P1.5a, amendment A1): the same truncated normal
    with std ``1/sqrt(fan_in)``, the fan-in being ``weight.shape[1]`` unless given: the input width
    of an ``nn.Linear`` weight ``[out, in]`` and of a stacked unit weight ``[M, in, out]``. Biases
    are not initialised here: every P1.2 module zeroes them (the cell sets its gate's bias).
    ``generator``: draw from it instead of the global random stream (``None``, the default: the
    global stream, as every caller before 15 Nov 2026).
    """
    if init == "fan_in":
        fan = int(weight.shape[1]) if fan_in is None else int(fan_in)
        if fan < 1:
            raise ValueError(f"init 'fan_in' needs a fan-in >= 1, got {fan}")
        std = 1.0 / math.sqrt(fan)
    else:
        std = parse_init(init)
    if generator is None:
        return nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    return nn.init.trunc_normal_(
        weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std, generator=generator
    )


def _check_index_tensor(name: str, ids: Tensor) -> None:
    if ids.dtype not in _INDEX_DTYPES:
        raise TypeError(f"{name} must be an int64 or int32 tensor of ids, got dtype {ids.dtype}")


class _IdEmbedding(nn.Module):
    """A ``[num_ids, width]`` lookup table, P1.2-initialised: ids ``[...] -> [..., width]``."""

    def __init__(self, num_ids: int, width: int, *, init: str, id_name: str) -> None:
        super().__init__()
        if num_ids < 1 or width < 1:
            raise ValueError(
                f"{type(self).__name__} needs num_ids >= 1 and width >= 1, got {num_ids} and "
                f"{width}"
            )
        self.num_ids = num_ids
        self.width = width
        self.init = init
        self._id_name = id_name
        self.weight = nn.Parameter(torch.empty(num_ids, width))
        init_weight_(self.weight, init)

    def forward(self, ids: Tensor) -> Tensor:
        """Look up ``ids [...]`` (int64 or int32, values in ``[0, num_ids)``) -> ``[..., width]``.

        The id dtype is checked here. An id outside the table is an error of the backend's lookup
        on every profile (``IndexError`` on the CPU, ``AcceleratorError`` on MPS, a device-side
        assert on CUDA), never re-read as a different id; no host round trip is spent on it.
        """
        _check_index_tensor(self._id_name, ids)
        return F.embedding(ids, self.weight)

    def param_count(self) -> int:
        """Measured parameter count (``num_ids × width``)."""
        return self.weight.numel()

    def extra_repr(self) -> str:
        return f"num_ids={self.num_ids}, width={self.width}, init={self.init!r}"


class ByteEmbedding(_IdEmbedding):
    """Byte ids ``tokens [B] -> [B, d]``: the ``vocab_size × d`` table ``byte_emb`` of C.1."""

    def __init__(self, vocab_size: int, d: int, *, init: str) -> None:
        super().__init__(vocab_size, d, init=init, id_name="tokens")
        self.vocab_size = vocab_size
        self.d = d

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Self:
        return cls(cfg.vocab_size, cfg.d, init=cfg.state_init)

    def add_unknown_row(self, generator: torch.Generator) -> nn.Parameter:
        """``train.content_dropout_target = unknown`` (``docs/key_content.md``): register and
        return the learned "unknown" row ``unknown`` ``[1, d]``, drawn as the table's rows are
        (:func:`init_weight_` with the table's ``init``) from ``generator``, so the global
        random stream is not read and no other parameter's start value changes. It is shaped
        as one row of the table: a ``fan_in`` init reads the table's fan-in ``d``, and the
        optimiser's weight decay (on tensors with two dimensions) treats it as it treats the
        table. :meth:`forward` does not read it: the HSN puts it in place of a byte's row
        (:func:`replace_content`). Without this call the module has no such parameter."""
        if "unknown" in self._parameters:
            raise ValueError("this byte embedding already has its unknown row")
        row = torch.empty(1, self.width, dtype=self.weight.dtype, device=self.weight.device)
        init_weight_(row, self.init, generator=generator)
        self.unknown = nn.Parameter(row)
        return self.unknown


def replace_content(
    rows: Tensor,
    zero: Tensor | None = None,
    unknown: Tensor | None = None,
    unknown_row: Tensor | None = None,
) -> Tensor:
    """The byte-content options (``docs/key_content.md``): ``rows [..., w]`` (a content
    table's lookup) with a zero row at the positions where ``zero`` (bool ``[...]``) is True
    and ``unknown_row [1, w]`` (the table's learned unknown row) where ``unknown`` is True.
    The two masks never overlap (``HSN._content_masks``). With both ``None`` this is ``rows``
    itself, the lookup unchanged."""
    if zero is not None:
        rows = rows.masked_fill(zero[..., None], 0.0)
    if unknown is not None:
        if unknown_row is None:
            raise ValueError("positions read the unknown row, but none was given")
        rows = torch.where(unknown[..., None], unknown_row, rows)
    return rows


class ContextEmbedding(_IdEmbedding):
    """Context symbol ids ``ctx [B] -> [B, context_dim]``: the ``n_context × context_dim`` table
    ``ctx_emb`` of Appendix C.1 (``context_dim = d`` in A.1)."""

    def __init__(self, n_context: int, context_dim: int, *, init: str) -> None:
        super().__init__(n_context, context_dim, init=init, id_name="ctx")
        self.n_context = n_context
        self.context_dim = context_dim

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Self:
        return cls(cfg.n_context, cfg.context_dim, init=cfg.state_init)
