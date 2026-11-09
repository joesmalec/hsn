"""Model variants of the P1.10 efficiency audit, installed on a built HSN instance by name.

Moved from ``scripts/eff_overfit.py`` (branch ``hsn-efficiency``, ``f8884ec``) with its semantics
unchanged. A variant patches the built model (no repository code path changes and no model
configuration key: variants are harness-level, P1.11 brief) and returns the record that goes into
the run's result.

* ``pdg``: the HSN as configured (the default for model family ``hsn``).
* ``gru_cell``: the PDG update replaced by one ``nn.GRUCell(4d, d)`` shared by every cell, on the
  normalised cell input ``norm([u; eps; h_i; m_i])`` (the A1 wrapper), with PyTorch's default GRU
  initialisation. The unit bank, ``W_out`` and the gate stay in place but receive no gradient; the
  round reports a NaN gate so that ``forward(capture=True)`` works (the diagnostics drop NaN gates).
* ``gru_cell_linear_head`` (W1): ``gru_cell`` with HSN's decoder ``D([q; u])`` replaced by the GRU
  baseline's readout ``Linear(LayerNorm(q))`` (weights ``N(0, 0.02^2)``, bias 0; ``u`` not read).
* ``linear_head`` (V1): the PDG model with that readout.
* ``decoder_fc1_fan_in`` (V2): the decoder's first layer at the A1 fan-in initialisation.
* ``cell_offset``: a fixed per-cell vector ``o_i ~ N(0, offset_std^2)`` of width ``4d`` added to the
  normalised cell input of the dense round (a buffer), so that the cells are not interchangeable.
* ``cell_embed``: the same vector as a learned parameter (``N * 4d`` parameters).
* ``cell_embed_linear_head``: ``cell_embed`` with the ``linear_head`` readout.

Random draws, as at ``f8884ec``: the offsets come from their own generator seeded with the arm's
seed; ``nn.GRUCell``, the linear head and the fan-in initialisation draw from the global generator
right after the model is built. Retired variants raise with the reason (:data:`RETIRED`). The
non-HSN model families take the variant ``none`` only.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from hsn.experiments.variant_names import (
    DEFAULT_OFFSET_STD,
    OFFSET_VARIANTS,
    RETIRED,
    VARIANTS,
    variant_params,
)
from hsn.model.cell import WindowConsts
from hsn.model.embeddings import init_weight_
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState

__all__ = [
    "DEFAULT_OFFSET_STD",
    "OFFSET_VARIANTS",
    "RETIRED",
    "VARIANTS",
    "CellIdentity",
    "LinearHead",
    "install_variant",
    "variant_params",
]


class CellIdentity(nn.Module):
    """``norm(v) + o[cell]`` on the dense round's ``[B, N, 4d]`` input; ``o`` fixed or learned."""

    offset: Tensor

    def __init__(self, norm: nn.Module, offset: Tensor, learned: bool) -> None:
        super().__init__()
        self.norm = norm
        self.n_cells = int(offset.shape[0])
        if learned:
            self.offset = nn.Parameter(offset.clone())
        else:
            self.register_buffer("offset", offset.clone())

    def forward(self, v: Tensor) -> Tensor:
        out: Tensor = self.norm(v)
        if v.ndim != 3 or v.shape[1] != self.n_cells:
            raise ValueError(
                f"CellIdentity applies to the dense round's [B, N={self.n_cells}, 4d] input, got "
                f"{tuple(v.shape)}"
            )
        return out + self.offset.to(out.dtype)[None]


class LinearHead(nn.Module):
    """The GRU baseline's readout on HSN's summary: ``Linear(LayerNorm(q))`` applied to the decoder
    input ``[q; u]`` (``u`` is not read). The linear layer is named ``fc1`` because
    :attr:`HSN.device` and :attr:`HSN.dtype` read ``decoder.fc1.weight``."""

    def __init__(self, d: int, vocab: int) -> None:
        super().__init__()
        self.d = d
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, vocab)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fc1(self.norm(x[..., : self.d]))
        return out


def _param_report_with(extra_rows: dict[str, nn.Module]) -> Callable[[HSN], dict[str, int]]:
    """``HSN.param_report`` with the variant's extra rows (modules outside every Appendix C row)
    added to its honesty check (``total_trainable = total + edges + extras``)."""

    def param_report(self: HSN) -> dict[str, int]:
        rows: dict[str, nn.Module] = {
            "byte_emb": self.byte_emb,
            "ctx_emb": self.ctx_emb,
            "encoder": self.encoder,
            "predictor": self.predictor,
            "shared": self.shared,
            "units": self.cell.units,
            "w_out": self.cell.W_out,
            "v_msg": self.cell.V,
            "gate": self.cell.gate,
            "cell_norm": self.cell.norm,
            "scorer": self.scorer,
            "halting": self.halting,
            "decoder": self.decoder,
        }
        report = {name: sum(p.numel() for p in m.parameters()) for name, m in rows.items()}
        report["total"] = sum(report.values())
        report["edges"] = self.cell.graph.edge_logit.numel()
        extra = 0
        for name, module in extra_rows.items():
            report[name] = sum(p.numel() for p in module.parameters())
            extra += report[name]
        report["total_trainable"] = sum(p.numel() for p in self.parameters())
        if report["total_trainable"] != report["total"] + report["edges"] + extra:
            raise RuntimeError(f"param_report (variant): unaccounted parameters in {report}")
        return report

    return param_report


def install_variant(
    model: nn.Module, variant: str, seed: int, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Install ``variant`` on ``model`` (an :class:`HSN` unless ``none``/``pdg``) and return its
    record. ``params`` are the variant's (``offset_std`` for the offset variants)."""
    resolved = variant_params(variant, params)
    if variant == "none":
        if isinstance(model, HSN):
            raise ValueError("variant 'none' is for the baselines; an HSN takes 'pdg' or a variant")
        return {"variant": "none"}
    if not isinstance(model, HSN):
        raise TypeError(f"variant {variant} needs an HSN model, got {type(model).__name__}")
    if variant == "pdg":
        return {"variant": "pdg"}
    if model.cfg.key_address or model.cfg.selected_readout:
        raise ValueError(
            f"variant {variant!r} with A5 is unsupported: use variant pdg and the "
            "key_address / selected_readout flags for the independent A5 ablations"
        )
    if model.cfg.cand_norm != "none":  # P1.13 hook (agreed with the coordinator, 7 Nov 2026)
        raise ValueError(
            f"variant {variant!r} with model.cand_norm={model.cfg.cand_norm} is refused: the "
            "variants' rounds and parameter reports predate the candidate norm; use variant pdg"
        )
    cell = model.cell
    d, N = cell.d, cell.n_cells
    if variant == "linear_head":
        _linear_head(model)
        return {"variant": variant, "decoder": "LayerNorm(q) -> Linear(d, V)"}
    if variant == "decoder_fc1_fan_in":
        init_weight_(model.decoder.fc1.weight, "fan_in")
        return {"variant": variant, "decoder_fc1_init": "fan_in"}
    if variant in OFFSET_VARIANTS:
        offset_std = float(resolved["offset_std"])
        learned = variant != "cell_offset"
        gen = torch.Generator().manual_seed(seed)
        offset = torch.randn(N, 4 * d, generator=gen) * offset_std
        # step_dense reads self.norm(v_raw); the cell_norm row of HSN.param_report now holds the
        # RMSNorm gain plus, for the learned variants, the per-cell vectors, so its check balances
        cell.norm = CellIdentity(cell.norm, offset, learned=learned)
        record: dict[str, Any] = {
            "variant": variant,
            "offset_std": offset_std,
            "extra_params": N * 4 * d if learned else 0,
        }
        if variant == "cell_embed_linear_head":
            _linear_head(model)
            record["decoder"] = "LayerNorm(q) -> Linear(d, V)"
        return record
    if variant in ("gru_cell", "gru_cell_linear_head"):
        return _install_gru_cell(model, variant)
    raise AssertionError(f"variant {variant!r} is registered but not installed")  # pragma: no cover


def _linear_head(model: HSN) -> None:
    """Replace the decoder by :class:`LinearHead` (``nn.Module.__setattr__`` registers it)."""
    setattr(model, "decoder", LinearHead(model.cell.d, model.cfg.vocab_size))  # noqa: B010


def _install_gru_cell(model: HSN, variant: str) -> dict[str, Any]:
    cell = model.cell
    d = cell.d
    gru = nn.GRUCell(4 * d, d)  # PyTorch's default initialisation (as nn.GRU in gru_2m)
    cell.gru_update = gru

    def step_dense(
        u: Tensor,
        eps: Tensor,
        state: CellGraphState,
        alpha: Tensor,
        *,
        consts: WindowConsts | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if consts is not None:  # P2.8's compiled window: variants run eager only
            raise NotImplementedError(
                f"variant {variant}: P2.8's compiled window is not supported (variants run eager "
                "only: hardware.compile_mode=none)"
            )
        B, NN, dd = cell._check_inputs(u, eps, state, alpha)
        with cell.graph.step():
            h_prev = state.h
            m = cell.graph.messages_dense(h_prev, cell.V)
            v_raw = torch.cat(
                [
                    u[:, None, :].expand(B, NN, dd),
                    eps[:, None, :].expand(B, NN, dd),
                    h_prev,
                    m,
                ],
                dim=-1,
            )
            v = cell.norm(v_raw)
            info: dict[str, Tensor] = {"m": m, "v": v, "v_raw": v_raw, "h_prev": h_prev}
            width = 4 * dd
            with torch.autocast(device_type=h_prev.device.type, enabled=False):
                h_new = gru(
                    v.reshape(B * NN, width).to(h_prev.dtype), h_prev.reshape(B * NN, dd)
                ).reshape(B, NN, dd)
        # the GRU has no write gate b: NaN, so forward(capture=True) works and the diagnostics
        # report no gate quantiles
        info["b"] = torch.full_like(h_new, math.nan)
        return h_new, info

    cell.step_dense = step_dense  # type: ignore[method-assign]
    extra: dict[str, nn.Module] = {"gru_update": gru}
    record: dict[str, Any] = {"variant": variant}
    if variant == "gru_cell_linear_head":
        _linear_head(model)
        record["decoder"] = "LayerNorm(q) -> Linear(d, V)"
    model.param_report = types.MethodType(_param_report_with(extra), model)  # type: ignore[method-assign]
    record["extra_params"] = sum(p.numel() for p in gru.parameters())
    return record
