"""The variant registry's names and parameters (torch-free, so that a spec expands without torch);
the variants themselves are :mod:`hsn.experiments.variants`."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["DEFAULT_OFFSET_STD", "OFFSET_VARIANTS", "RETIRED", "VARIANTS", "variant_params"]

VARIANTS: tuple[str, ...] = (
    "none",
    "pdg",
    "gru_cell",
    "gru_cell_linear_head",
    "cell_offset",
    "cell_embed",
    "cell_embed_linear_head",
    "linear_head",
    "decoder_fc1_fan_in",
)
OFFSET_VARIANTS: tuple[str, ...] = ("cell_offset", "cell_embed", "cell_embed_linear_head")
DEFAULT_OFFSET_STD = 0.5
RETIRED: dict[str, str] = {
    "gru_cell_direct_input": (
        "retired with P1.10a: it patched HSN._shared_path, which HSN.forward no longer calls per "
        "position, so installed now it would silently do nothing (W2 result in the audit log)"
    ),
}


def variant_params(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """The variant's parameters with their defaults filled in (what an arm id hashes); unknown
    names, retired variants and parameters a variant does not take raise."""
    if name in RETIRED:
        raise NotImplementedError(f"variant {name!r} is retired: {RETIRED[name]}")
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; known: {VARIANTS}")
    given = dict(params or {})
    if name in OFFSET_VARIANTS:
        std = float(given.pop("offset_std", DEFAULT_OFFSET_STD))
        if not (std > 0.0 and math.isfinite(std)):
            raise ValueError(f"variant {name}: offset_std must be positive, got {std}")
        out: dict[str, Any] = {"offset_std": std}
    else:
        out = {}
    if given:
        raise ValueError(f"variant {name!r} takes no parameters {sorted(given)}")
    return out
