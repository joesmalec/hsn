"""Gated DeltaNet byte-level language model: the fast-weight associative-memory control.

Gated DeltaNet (Yang, Kautz and Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta
Rule", 2024, arXiv:2412.06464) keeps, per head, a matrix ``S`` that maps keys to values and
writes it with a gated delta rule: decay everything a little, then correct what the matrix
returns for the current key towards the current value. It is among the strongest efficient
models on associative recall, which makes it the control for the HSN's keyed recall:
``GatedDeltaNetBaseline`` under ``model_name: gated_deltanet`` is ``n_layers`` blocks of
[Gated DeltaNet mixer, gated MLP] between a byte embedding plus the context-symbol embedding and
a linear head. ``docs/gated_deltanet_baseline.md`` has the sizes, the mapping to the paper, the
measurements and how to run it.

**One block** (pre-norm, residual; the reference block of flash-linear-attention,
``fla.models.gated_deltanet`` with ``fla.layers.GatedDeltaNet``, version 0.6.0), per head with
key width ``K = head_dim`` and value width ``V = expand_v × K``::

    q, k, v = SiLU(causal depthwise conv over d_conv positions of q_proj(x), k_proj(x), v_proj(x))
    q, k    = q / |q|, k / |k|                                 # unit length
    α_t     = exp(-exp(A_log) · softplus(a_proj(x_t) + dt_bias))   # decay in (0, 1), per head
    β_t     = sigmoid(b_proj(x_t))                             # write strength in (0, 1), per head
    S_t     = α_t S_{t-1} + β_t (v_t - α_t S_{t-1} k_t) k_tᵀ   # the gated delta rule, S is [V, K]
    o_t     = S_t q_t / sqrt(K)
    y       = o_proj(RMSNorm(o) ⊙ SiLU(g_proj(x)))             # the output gate
    x       = x + y;  x = x + down_proj(SiLU(gate_proj(n)) ⊙ up_proj(n)),  n = RMSNorm(x)

Initialisation follows the reference model: every projection, the convolution and the
embeddings are normal with standard deviation 0.02, ``exp(A_log)`` is uniform in (0, 16) and
``softplus(dt_bias)`` log-uniform in [0.001, 0.1].

**State.** The model is recurrent under :class:`hsn.train.loop.SequenceModel`: the state is the
pair ``(S, conv)`` with ``S [n_layers, B, num_heads, V, K]`` the fast weights of every block
and ``conv [n_layers, B, d_conv - 1, 2·H·K + H·V]`` the last inputs of every block's
convolution, both float32. The trainer carries it between windows and detaches it at every
window start (truncated BPTT), as for the GRU. ``reset[b, t]`` clears row ``b`` before position
``t``: the fast weights are zeroed there and the convolution does not read positions before the
reset, so a document never sees the one before it.

**Kernels** (``GatedDeltaNetBaseline.kernel``):

* ``chunk``: the delta rule of a whole window in parallel form through
  ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``, flash-linear-attention's Triton kernels
  (CUDA only), which take the incoming ``S`` and return the final one. A window in which a row
  starts a document after its first position goes through the same kernels in their
  variable-length form: the batch as one stream cut at every row start and every reset, a zero
  initial state for the segments that start a document. One kernel call per block either way.
  Under the trainer's autocast the kernels read ``q``, ``k``, ``v`` in the autocast dtype and
  compute and keep ``S`` in float32, as the reference does. Their matrix products run on the
  card's tensor cores in TF32 on Ampere and newer, so in float32 they agree with the
  recurrence to about three digits, not to rounding (the document has the measurements).
* ``recurrent``: the recurrence above written out in PyTorch, one position at a time, float32.
  What runs without a CUDA device (the CPU tests and the CPU scorers), and what ``chunk`` is
  tested against.
* ``reference_chunk``: the parallel form in PyTorch
  (``fla.ops.gated_delta_rule.naive_chunk_gated_delta_rule``), for tests on the CPU; it takes no
  reset after a window's first position.
* ``auto`` (default): ``chunk`` on CUDA, ``recurrent`` elsewhere.

Everything around the delta rule (projections, the convolution with its carried inputs and
resets, the gated norm, the MLP) is plain PyTorch on every device. ``hardware.baseline_compile``
is refused for this family at load: the kernels are fla's own and are marked
``torch.compiler.disable``, so the model runs eager with them.

**Dependency.** The kernels are not a dependency of this package and the shared environment
does not carry them: importing this module without ``fla`` (pip ``fla-core``, pinned to
:data:`FLA_VERSION`) raises ``ImportError`` with what to do. ``hsn.runner`` imports the module
only when a ``gated_deltanet`` model is built, so everything else runs without it.

Sizes come from ``baseline.gated_deltanet.*``. With the defaults (d_model 176, 4 blocks, 3 heads
of key width 44 and value width 88, d_conv 4, MLP width 512, V=256, 16 context symbols) the
model has 1,932,584 parameters as measured by :meth:`GatedDeltaNetBaseline.param_report`.
``baseline.address_input`` adds the HSN's fixed byte codes to the input
(:mod:`hsn.baselines.address`).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.baselines.address import AddressInput
from hsn.config import GatedDeltaNetBaselineConfig, RunConfig

#: The flash-linear-attention release this module is written and tested against (pip
#: ``fla-core``; the kernels' arguments changed between releases, so another one is refused).
FLA_VERSION = "0.6.0"

try:
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, naive_chunk_gated_delta_rule
except ImportError as exc:
    raise ImportError(
        "hsn.baselines.gated_deltanet (model_name: gated_deltanet) needs the Gated DeltaNet "
        f"kernels of flash-linear-attention (pip package fla-core=={FLA_VERSION}, module "
        "'fla'), and this Python environment does not have them. They are not a dependency of "
        "hsn and are not installed in the shared environment: run this baseline with the "
        "interpreter of an environment that has them (scripts/gated_deltanet_env.sh builds one; "
        f"docs/gated_deltanet_baseline.md, 'Environment'). The import failed with: {exc}"
    ) from exc
if fla.__version__ != FLA_VERSION:
    raise ImportError(
        f"hsn.baselines.gated_deltanet is written and tested against fla-core=={FLA_VERSION}, "
        f"and this environment has {fla.__version__}: the kernels' arguments differ between "
        "releases (scripts/gated_deltanet_env.sh builds the pinned environment)"
    )

__all__ = ["FLA_VERSION", "GatedDeltaNetBaseline", "Kernel"]

State = tuple[Tensor, Tensor]
State3 = tuple[Tensor, Tensor, Tensor]
Kernel = Literal["auto", "chunk", "recurrent", "reference_chunk"]
KERNELS: tuple[str, ...] = ("auto", "chunk", "recurrent", "reference_chunk")
#: Row segments of a window for the variable-length kernels: the cumulative segment bounds on
#: the device and on the host, and the index of every row's first and last segment.
Segments = tuple[Tensor, Tensor, Tensor, Tensor]
#: The epsilon of the kernels' key and query normalisation (``fla.modules.l2norm``).
L2_EPS = 1e-6


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def _linear(d_in: int, d_out: int) -> nn.Linear:
    """A projection without bias, initialised as the reference model initialises it."""
    proj = nn.Linear(d_in, d_out, bias=False)
    nn.init.normal_(proj.weight, std=0.02)
    return proj


def _unit(x: Tensor) -> Tensor:
    """``x`` at unit length along its last dimension, as the kernels normalise keys and queries."""
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + L2_EPS)


class _Mixer(nn.Module):
    """The Gated DeltaNet layer of one block; see the module docstring for the recurrence.
    Parameters carry the names of ``fla.layers.GatedDeltaNet``; the three convolutions are one
    tensor (``conv_weight``: the channels of q, k and v in that order, tap ``j`` reading the
    input ``j`` positions back)."""

    def __init__(self, spec: GatedDeltaNetBaselineConfig) -> None:
        super().__init__()
        d_model, heads = spec.d_model, spec.num_heads
        self.num_heads = heads
        self.head_k = spec.head_dim
        self.head_v = spec.expand_v * spec.head_dim
        self.key_dim = heads * self.head_k
        self.value_dim = heads * self.head_v
        self.d_conv = spec.d_conv
        self.eps = spec.norm_eps
        self.q_proj = _linear(d_model, self.key_dim)
        self.k_proj = _linear(d_model, self.key_dim)
        self.v_proj = _linear(d_model, self.value_dim)
        self.a_proj = _linear(d_model, heads)
        self.b_proj = _linear(d_model, heads)
        self.g_proj = _linear(d_model, self.value_dim)
        self.o_proj = _linear(self.value_dim, d_model)
        self.conv_weight = nn.Parameter(
            torch.empty(2 * self.key_dim + self.value_dim, spec.d_conv).normal_(std=0.02)
        )
        # the decay's rate and step, as the reference draws them (no weight decay: one-dimensional)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(0, 16)))
        dt = torch.exp(
            torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse of softplus
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v))

    def forward(
        self,
        x: Tensor,
        S: Tensor,
        conv: Tensor,
        reset: Tensor,
        same_doc: list[Tensor],
        tail_doc: Tensor,
        kernel: str,
        segments: Segments | None,
        reset_cols: list[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``x [B, T, d_model]`` (normed), ``S [B, H, V, K]``, ``conv [B, d_conv - 1, C]``.
        ``same_doc[j - 1] [B, T, 1]``: the input ``j`` positions back belongs to the position's
        document; ``tail_doc [B, d_conv - 1, 1]``: likewise for the inputs carried on. Returns
        the layer's output, the new ``S`` and the new carried inputs."""
        B, T, _ = x.shape
        H, K, V = self.num_heads, self.head_k, self.head_v
        # ---- causal depthwise convolution with the carried inputs on the left ----
        stream = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        first = self.d_conv - 1  # position t of the window is index t + first of `padded`
        padded = torch.cat([conv.to(stream.dtype), stream], dim=1)
        u = padded[:, first:] * self.conv_weight[:, 0]
        for j in range(1, self.d_conv):
            tap = padded[:, first - j : first - j + T]
            u = u + (tap * same_doc[j - 1]) * self.conv_weight[:, j]
        new_conv = (padded[:, T:] * tail_doc).float()
        widths = [self.key_dim, self.key_dim, self.value_dim]
        q, k, v = torch.split(F.silu(u).to(stream.dtype), widths, dim=-1)
        q, k, v = q.reshape(B, T, H, K), k.reshape(B, T, H, K), v.reshape(B, T, H, V)
        a, b = self.a_proj(x), self.b_proj(x)  # [B, T, H]: the decay's and the write's inputs
        # ---- the gated delta rule ----
        if kernel == "chunk":
            o, S = self._chunk(q, k, v, a, b, S, reset, segments)
        elif kernel == "recurrent":
            o, S = self._recurrent(q, k, v, a, b, S, reset, reset_cols)
        else:
            o, S = self._reference_chunk(q, k, v, a, b, S, reset)
        # ---- output gate: RMS norm over a head's values, times SiLU of the gate ----
        o = o.float()
        gate = self.g_proj(x).reshape(B, T, H, V).float()
        o = o * torch.rsqrt(o.square().mean(dim=-1, keepdim=True) + self.eps)
        o = o * self.o_norm_weight * F.silu(gate)
        out: Tensor = self.o_proj(o.reshape(B, T, self.value_dim))
        return out, S, new_conv

    # the three forms of the delta rule: (q, k, v, a, b, S, reset, ...) -> (o [B, T, H, V], S)

    def _chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        S: Tensor,
        reset: Tensor,
        segments: Segments | None,
    ) -> tuple[Tensor, Tensor]:
        """fla's Triton kernels (the parallel form). The kernels normalise q and k, apply the
        sigmoid of the write strength and the decay's activation themselves, and keep the state
        value-first (``[V, K]``), as ``fla.layers.GatedDeltaNet`` calls them."""
        if q.device.type != "cuda":
            raise RuntimeError(
                "GatedDeltaNetBaseline.kernel='chunk' runs flash-linear-attention's Triton "
                f"kernels, which need a CUDA device (the window is on {q.device}); "
                "kernel='recurrent' (what 'auto' takes off CUDA) runs anywhere"
            )
        B, T, H, V = v.shape
        options: dict[str, Any] = {
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "use_gate_in_kernel": True,
            "state_v_first": True,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
        }
        S = S.float().masked_fill(reset[:, 0].view(B, 1, 1, 1), 0.0)
        if segments is None:  # no row starts a document after the first position
            o, S = chunk_gated_delta_rule(q, k, v, a, b, initial_state=S, **options)
            return o, S
        bounds, bounds_host, row_first, row_last = segments
        initial = S.new_zeros(bounds.numel() - 1, *S.shape[1:]).index_copy(0, row_first, S)
        q, k, v, a, b = (z.reshape(1, B * T, *z.shape[2:]) for z in (q, k, v, a, b))
        options.update(initial_state=initial, cu_seqlens=bounds, cu_seqlens_cpu=bounds_host)
        o, finals = chunk_gated_delta_rule(q, k, v, a, b, **options)
        return o.reshape(B, T, H, V), finals.index_select(0, row_last)

    def _gates(self, q: Tensor, k: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, ...]:
        """What the Triton kernels compute inside, in float32: unit q and k, the log of the
        decay and the write strength."""
        log_decay = -torch.exp(self.A_log) * F.softplus(a.float() + self.dt_bias)
        return _unit(q.float()), _unit(k.float()), log_decay, torch.sigmoid(b.float())

    def _recurrent(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        S: Tensor,
        reset: Tensor,
        reset_cols: list[int],
    ) -> tuple[Tensor, Tensor]:
        """The recurrence of the module docstring, one position at a time (float32)."""
        B, T = reset.shape
        resets = set(reset_cols)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, log_decay, beta = self._gates(q, k, a, b)
            decay = torch.exp(log_decay)
            v, S = v.float(), S.float()
            outputs: list[Tensor] = []
            for t in range(T):
                if t in resets:
                    S = S.masked_fill(reset[:, t].view(B, 1, 1, 1), 0.0)
                S = S * decay[:, t, :, None, None]
                key = k[:, t].unsqueeze(-1)  # [B, H, K, 1]
                error = (v[:, t] - (S @ key).squeeze(-1)) * beta[:, t, :, None]  # [B, H, V]
                S = S + error.unsqueeze(-1) * key.transpose(-1, -2)
                outputs.append((S @ q[:, t].unsqueeze(-1)).squeeze(-1))
            o = torch.stack(outputs, dim=1) * self.head_k**-0.5
        return o, S

    def _reference_chunk(
        self, q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, S: Tensor, reset: Tensor
    ) -> tuple[Tensor, Tensor]:
        """fla's parallel form in PyTorch (float32; its state is key-first, ``[K, V]``)."""
        if bool(reset[:, 1:].any()):
            raise ValueError(
                "GatedDeltaNetBaseline.kernel='reference_chunk' takes no document start after "
                "a window's first position; use 'recurrent' (or 'chunk' on CUDA)"
            )
        B = reset.shape[0]
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, log_decay, beta = self._gates(q, k, a, b)
            S = S.float().masked_fill(reset[:, 0].view(B, 1, 1, 1), 0.0)
            o, S = naive_chunk_gated_delta_rule(
                q,
                k,
                v.float(),
                log_decay,
                beta,
                initial_state=S.transpose(-1, -2),
                output_final_state=True,
            )
        return o, S.transpose(-1, -2)


class _Block(nn.Module):
    """``x + mixer(norm(x))``, then ``x + gated MLP(norm(x))``."""

    def __init__(self, spec: GatedDeltaNetBaselineConfig) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(spec.d_model, spec.norm_eps)
        self.attn = _Mixer(spec)
        self.mlp_norm = _RMSNorm(spec.d_model, spec.norm_eps)
        self.gate_proj = _linear(spec.d_model, spec.mlp_width())
        self.up_proj = _linear(spec.d_model, spec.mlp_width())
        self.down_proj = _linear(spec.mlp_width(), spec.d_model)

    def forward(self, x: Tensor, S: Tensor, conv: Tensor, *window: Any) -> State3:
        y, S, conv = self.attn(self.attn_norm(x), S, conv, *window)
        x = x + y
        n = self.mlp_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(n)) * self.up_proj(n))
        return x, S, conv


class GatedDeltaNetBaseline(nn.Module):
    """Byte + context embeddings → ``n_layers`` Gated DeltaNet blocks → RMSNorm → linear head."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        spec = cfg.baseline.gated_deltanet
        self.vocab_size = cfg.model.vocab_size
        self.n_context = cfg.model.n_context
        self.d_model = spec.d_model
        self.n_layers = spec.n_layers
        self.num_heads = spec.num_heads
        self.head_k = spec.head_dim
        self.head_v = spec.expand_v * spec.head_dim
        self.d_conv = spec.d_conv
        self.conv_channels = spec.num_heads * (2 * self.head_k + self.head_v)
        #: which form of the delta rule runs (module docstring); not a configuration key: the
        #: forms compute the same function, and a run takes ``auto``
        self.kernel: Kernel = "auto"
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.ctx_emb = nn.Embedding(self.n_context, self.d_model)
        self.address = AddressInput(cfg, self.d_model) if cfg.baseline.address_input else None
        self.blocks = nn.ModuleList(_Block(spec) for _ in range(self.n_layers))
        self.norm = _RMSNorm(self.d_model, spec.norm_eps)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.ctx_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> State:
        """Zeros ``(S, conv)`` in the shapes the module docstring gives."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        S = torch.zeros(self.n_layers, B, self.num_heads, self.head_v, self.head_k, device=device)
        conv = torch.zeros(self.n_layers, B, self.d_conv - 1, self.conv_channels, device=device)
        return S, conv

    def reset_state(self, state: State, mask: Tensor) -> State:
        """Zero the rows of both state tensors where ``mask`` (``[B]`` bool) is True."""
        S, conv = self._check_state(state, mask.shape[0])
        if mask.shape != (S.shape[1],) or mask.dtype is not torch.bool:
            raise ValueError(
                f"mask must be a bool [B={S.shape[1]}] tensor, got {tuple(mask.shape)}"
            )
        rows = mask.to(S.device)
        return S.masked_fill(rows.view(1, -1, 1, 1, 1), 0.0), conv.masked_fill(
            rows.view(1, -1, 1, 1), 0.0
        )

    def state_size(self) -> dict[str, int]:
        """Numbers of state per row: the fast weights, the carried convolution inputs, both."""
        fast = self.n_layers * self.num_heads * self.head_v * self.head_k
        conv = self.n_layers * (self.d_conv - 1) * self.conv_channels
        return {"fast_weights": fast, "conv_inputs": conv, "total": fast + conv}

    def resolved_kernel(self, device: torch.device) -> str:
        """The form of the delta rule a window on ``device`` runs: ``kernel``, with ``auto``
        resolved (``chunk`` on CUDA, ``recurrent`` elsewhere)."""
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {self.kernel!r}")
        if self.kernel == "auto":
            return "chunk" if device.type == "cuda" else "recurrent"
        return self.kernel

    def forward(
        self, tokens: Tensor, context: Tensor, state: State, reset: Tensor
    ) -> tuple[Tensor, State, dict[str, Any]]:
        """``tokens``/``context``/``reset`` are ``[B, T]``; returns logits ``[B, T, V]``, state."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got shape {tuple(tokens.shape)}")
        B, T = tokens.shape
        for name, value in (("context", context), ("reset", reset)):
            if value.shape != (B, T):
                raise ValueError(f"{name} must have shape {(B, T)}, got {tuple(value.shape)}")
        if reset.dtype is not torch.bool:
            raise TypeError(f"reset must be a bool tensor, got {reset.dtype}")
        S, conv = self._check_state(state, B)
        kernel = self.resolved_kernel(tokens.device)
        # what the window's document starts mean for the delta rule (one host sync per window)
        segments: Segments | None = None
        reset_cols: list[int] = []
        if kernel == "recurrent":
            reset_cols = reset.any(dim=0).nonzero().flatten().tolist()
        elif kernel == "chunk" and bool(reset[:, 1:].any()):
            segments = _segments(reset)
        # what they mean for the convolution: tap j of a position is read only when the input
        # j positions back is of the same document (carried inputs count as position < 0)
        positions = torch.arange(T, device=tokens.device)
        marks = torch.where(reset, positions, torch.full_like(positions, -self.d_conv))
        last_reset = marks.cummax(dim=1).values  # [B, T]: the latest reset at or before t
        same_doc = [((positions - j) >= last_reset).unsqueeze(-1) for j in range(1, self.d_conv)]
        tail = torch.arange(T - (self.d_conv - 1), T, device=tokens.device)  # may be negative
        tail_doc = (tail >= last_reset[:, -1:]).unsqueeze(-1)

        x = self.tok_emb(tokens) + self.ctx_emb(context)
        if self.address is not None:
            x = x + self.address(tokens, reset)
        new_S: list[Tensor] = []
        new_conv: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            x, S_i, conv_i = block(
                x, S[i], conv[i], reset, same_doc, tail_doc, kernel, segments, reset_cols
            )
            new_S.append(S_i)
            new_conv.append(conv_i)
        logits = self.head(self.norm(x))
        out_S = torch.stack(new_S)
        info = {"state_abs_max": out_S.detach().abs().max()}  # a tensor: no host sync
        return logits, (out_S, torch.stack(new_conv)), info

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts; ``active`` == ``stored`` for this dense model."""
        parts = {
            "byte_emb": _count(self.tok_emb),
            "ctx_emb": _count(self.ctx_emb),
            **({"address": _count(self.address)} if self.address is not None else {}),
            "blocks": _count(self.blocks),
            "norm": _count(self.norm),
            "head": _count(self.head),
        }
        stored = _count(self)
        if sum(parts.values()) != stored:
            raise RuntimeError(
                "GatedDeltaNetBaseline.param_report: component counts do not sum to the total"
            )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {**parts, "stored": stored, "active": stored, "trainable": trainable}

    # helpers ---------------------------------------------------------------------------------

    def _check_state(self, state: Any, B: int) -> State:
        shapes = (
            (self.n_layers, B, self.num_heads, self.head_v, self.head_k),
            (self.n_layers, B, self.d_conv - 1, self.conv_channels),
        )
        ok = (
            isinstance(state, tuple)
            and len(state) == 2
            and all(isinstance(s, Tensor) for s in state)
            and tuple(tuple(s.shape) for s in state) == shapes
        )
        if not ok:
            raise ValueError(f"GatedDeltaNetBaseline state must be two tensors of shapes {shapes}")
        return state[0], state[1]


def _segments(reset: Tensor) -> Segments:
    """The window as one stream of ``B × T`` positions cut at every row start and every reset
    (the kernels' variable-length form): the cumulative bounds ``[N + 1]`` on the device and on
    the host, and the indices ``[B]`` of every row's first and last segment."""
    B, T = reset.shape
    starts = reset.clone()
    starts[:, 0] = True  # a row is a segment of its own whether or not it starts a document
    flat = starts.flatten().nonzero().flatten()  # ascending positions b·T + t
    bounds = torch.cat([flat, flat.new_tensor([B * T])])
    row_first = (flat % T == 0).nonzero().flatten()
    row_last = torch.cat([row_first[1:] - 1, row_first.new_tensor([flat.numel() - 1])])
    return bounds, bounds.cpu(), row_first, row_last


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
