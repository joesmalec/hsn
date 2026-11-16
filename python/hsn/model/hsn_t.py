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

``model.memory = false`` (``docs/hsn_t_no_memory.md``; opt-in) is HSN-T with its memory
switched off, :class:`HSNTNoMemory`: the parts of HSN-T that are not the memory, over all
positions of a window at once. It is a model of its own and not an :class:`HSNT`: it has no
state, no position loop and none of the memory's modules, and the generic trainer trains it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig, RunConfig
from hsn.model.embeddings import ByteEmbedding, ContextEmbedding
from hsn.model.encoder import Decoder, Encoder, SharedTransform
from hsn.model.hsn import HSN
from hsn.model.reader import Reader

__all__ = ["MEMORY_OFF_MODULES", "HSNT", "HSNTNoMemory"]

#: The modules of an HSN-T that are not its memory, under the names they have there (so the
#: state-dict keys are the same): what :class:`HSNTNoMemory` is made of.
MEMORY_OFF_MODULES: tuple[str, ...] = (
    "byte_emb",
    "ctx_emb",
    "encoder",
    "shared",
    "decoder",
    "reader",
)


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


class HSNTNoMemory(nn.Module):
    """HSN-T without its memory (``model.memory = false``, ``docs/hsn_t_no_memory.md``)::

        z      = E(emb(x))                        [B, T, d]
        c      = Reader(x, reset)                 [B, T, w]
        u      = S([z ; 0 ; ctx ; c])             0 where HSN-T reads q_{t−1}
        logits = D([0 ; u ; 0 ; c])               0 where HSN-T reads q_t and r_t

    Every line is computed for all positions of the window at once: there is no loop over
    positions, no state and nothing carried between windows. No cell, router, unit, graph,
    predictor or halting head is built, so none counts as a parameter. ``S`` and ``D`` are
    HSN-T's own modules at HSN-T's input widths; the columns of ``q`` and ``r`` stay and read
    zeros. That is what lets a trained HSN-T's weights of these six modules load unchanged
    (:meth:`load_memory_on_state`): the model is HSN-T with every read of the state held at
    zero. It costs arithmetic that produces nothing: at the default sizes 20 % of ``S``'s
    multiply-accumulates and 25 % of ``D``'s (the document has the count).

    Initialisation: the six modules are drawn as HSN-T draws them, in HSN-T's order. The reader
    and the input columns that A5 and the reader add to ``S`` and ``D`` (``r`` and ``c``) come
    from their own generators, and the embeddings and ``E`` are the first draws of the global
    stream, so for one seed they start equal to HSN-T's. The other weights of ``S`` and ``D``
    do not: HSN-T draws the predictor, the cell, the router and the halting head before them.

    Stateless under :class:`hsn.train.loop.SequenceModel`, like the reader alone
    (:class:`hsn.baselines.reader_lm.ReaderLM`); unlike it, the context symbol is read. The
    forward holds no data-dependent branch, shape or host read, so it compiles as one graph.

    The model keeps no ``cfg`` attribute on purpose: the evaluation loop passes the carried
    address bytes (``prev_tokens``) to any model whose ``cfg.key_address`` is set.
    """

    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        super().__init__()
        if cfg.memory:
            raise ValueError(
                "HSNTNoMemory is HSN-T with model.memory=false; with its memory the model is "
                "hsn.model.hsn_t.HSNT"
            )
        if not cfg.selected_readout:
            raise ValueError(
                "model.memory=false keeps HSN-T's decoder, which reads [q; u; r; c]: set "
                "model.selected_readout=true (the columns of q and r read zeros)"
            )
        self.vocab_size = cfg.vocab_size
        self.d = cfg.d
        self.seed = seed
        self.byte_emb = ByteEmbedding.from_config(cfg)
        self.ctx_emb = ContextEmbedding.from_config(cfg)
        self.encoder = Encoder(cfg)
        self.shared = SharedTransform(cfg, seed=seed)
        self.decoder = Decoder(cfg, seed=seed)
        self.reader = Reader(cfg, seed=seed)

    @classmethod
    def from_config(cls, run: RunConfig) -> Self:
        """``HSNTNoMemory(run.model, seed=run.seed)``."""
        if not isinstance(run, RunConfig):
            raise TypeError(f"from_config needs a RunConfig, got {type(run).__name__}")
        return cls(run.model, seed=run.seed)

    def load_memory_on_state(self, state: Mapping[str, Tensor]) -> None:
        """Load the weights of an HSN-T with its memory (``HSNT.state_dict()``, or a
        checkpoint's ``model``): the tensors of the six modules this model has, which must all
        be there with their shapes. The memory's tensors (``predictor.*``, ``cell.*``,
        ``router.*``, ``halting.*``) are left out."""
        own = self.state_dict()
        self.load_state_dict({key: value for key, value in state.items() if key in own})

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> None:
        """Stateless: there is nothing to carry between windows."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        return None

    def reset_state(self, state: None, mask: Tensor) -> None:
        if state is not None:
            raise ValueError(f"HSN-T without memory carries no state, got {type(state).__name__}")
        if mask.dtype is not torch.bool or mask.dim() != 1:
            raise ValueError(f"mask must be a bool [B] tensor, got {tuple(mask.shape)}")
        return None

    def forward(
        self, tokens: Tensor, context: Tensor, state: None, reset: Tensor
    ) -> tuple[Tensor, None, dict[str, Any]]:
        """``tokens`` / ``context`` / ``reset`` are ``[B, T]``; ``reset`` starts a document for
        the reader, which reads nothing before it. Returns ``logits [B, T, V]``, ``None`` and
        an empty ``info``."""
        if state is not None:
            raise ValueError(f"HSN-T without memory carries no state, got {type(state).__name__}")
        if context.shape != tokens.shape:
            raise ValueError(
                f"context must have shape {tuple(tokens.shape)}, got {tuple(context.shape)}"
            )
        c = self.reader(tokens, reset)  # [B, T, w]; checks tokens and reset
        z = self.encoder(self.byte_emb(tokens))  # [B, T, d]
        nothing = torch.zeros_like(z)  # the columns of q (and of r), both of width d
        u = self.shared(torch.cat([z, nothing, self.ctx_emb(context), c], dim=-1))
        logits: Tensor = self.decoder(torch.cat([nothing, u, nothing, c], dim=-1))
        return logits, None, {}

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts: one row per module (:data:`MEMORY_OFF_MODULES`, the
        closed form of :func:`hsn.train.memory_plan.memory_off_param_count`), then ``stored``
        == ``active`` (the model is dense) and ``trainable``."""
        rows = {
            name: sum(p.numel() for p in getattr(self, name).parameters())
            for name in MEMORY_OFF_MODULES
        }
        total = sum(p.numel() for p in self.parameters())
        if total != sum(rows.values()):
            raise RuntimeError(
                f"param_report: {total} parameters, but the rows give {sum(rows.values())}; a "
                "submodule carries unaccounted parameters"
            )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {**rows, "stored": total, "active": total, "trainable": trainable}

    def extra_repr(self) -> str:
        return f"seed={self.seed}, d={self.d}"
