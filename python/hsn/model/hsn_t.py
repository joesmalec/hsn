"""HSN-T (``model_name: hsn_t``, ``docs/hsn_t.md``): the HSN's memory core behind a reader.

A model family of its own beside the HSN, named by the user on 14 Nov 2026. The memory is the
HSN's, unchanged: the cells and their carried state, the fixed addresses, the router, the
address gate, the units, the convex tanh gate, the graph and the summary ``q``. What reads the
current window is a small causal Transformer (:class:`hsn.model.reader.Reader`), whose output
``c_t`` enters where the HSN's fast cells are read: ``u_t = S([z_t ; q_{t−1} ; ctx_t ; c_t])`` and
``D([q_t ; u_t ; r_t ; c_t])``.

The two families are kept apart by construction: an :class:`~hsn.model.hsn.HSN` refuses
``model.reader`` (the HSN has no sequence attention), and an :class:`HSNT` refuses a model
without one. The cell arithmetic is shared code (:mod:`hsn.model.hsn`), so every opt-in key of
the HSN's memory (fast cells, address n-grams, the address gate and its inputs, the write
decision, the feedback cuts) means the same in both.
"""

from __future__ import annotations

from typing import Any

from hsn.config import ModelConfig
from hsn.model.hsn import HSN

__all__ = ["HSNT"]


class HSNT(HSN):
    """The HSN's memory core with a window reader (module docstring). Built like an
    :class:`~hsn.model.hsn.HSN`; ``cfg.reader`` must name a reader."""

    _takes_reader = True

    def __init__(self, cfg: ModelConfig, **kwargs: Any) -> None:
        if cfg.reader == "none":
            raise ValueError(
                "HSN-T is the HSN's memory core behind a reader: set model.reader=transformer "
                "(without one the model is hsn.model.hsn.HSN, model_name: hsn)"
            )
        super().__init__(cfg, **kwargs)
