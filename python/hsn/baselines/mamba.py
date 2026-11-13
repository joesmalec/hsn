"""Mamba-style selective state-space byte-level language model: the selective-memory control.

The paper names "an optimized selective-state baseline such as Mamba" as the recurrent control the
HSN's persistent-state claim has to beat (§2 "Selective recurrent memory", §12; plan P2.13).
``MambaBaseline`` is that model under ``model_name: selective_ssm``: Mamba blocks (Gu and Dao,
2023) between a byte embedding plus the context-symbol embedding and a linear head.

**One block** (pre-norm, residual), with ``d_inner = expand × d_model``::

    x, z   = split(in_proj(RMSNorm(input)))                    # two streams of width d_inner
    u      = SiLU(causal depthwise conv over d_conv positions of x)
    Δ, B, C = softplus(dt_proj(·)), ·, ·   from x_proj(u)      # input-dependent ("selective")
    h_t    = exp(Δ_t A) ⊙ h_{t-1} + Δ_t B_t u_t                # A = -exp(A_log), per channel
    y_t    = h_t · C_t + D ⊙ u_t
    output = input + out_proj(y ⊙ SiLU(z))

Initialisation follows the reference implementation (``A_log = log(1..d_state)``, the step-size
bias drawn so that ``softplus(bias)`` is log-uniform in [0.001, 0.1], ``D = 1``).

**State.** The model is recurrent under :class:`hsn.train.loop.SequenceModel`: the state is the
pair ``(h, conv)`` with ``h [n_layers, B, d_inner, d_state]`` the scan's state and
``conv [n_layers, B, d_conv - 1, d_inner]`` the last inputs of every block's convolution. The
trainer carries it between windows and detaches it at every window start (truncated BPTT), as for
the GRU. ``reset[b, t]`` clears row ``b`` before position ``t``: the scan state is zeroed there
and the convolution does not read positions before the reset, so a document never sees the one
before it.

**What this implementation is and is not.** It is the published recurrence, exact, in float32
(autocast is switched off inside ``forward``: the decay ``exp(Δ A)`` sits close to 1 for a long
memory and bfloat16 cannot represent that). The scan is a Python loop over the window with its
backward written out (:class:`_SelectiveScan`), not the reference's fused CUDA scan, so its step
time says nothing about Mamba's speed; it is here for accuracy at a matched parameter count.
Training memory grows with ``B × T × d_inner × d_state`` per block (the decays and states the
backward reads).

Sizes come from ``baseline.selective_ssm.*``. With the defaults (d_model 256, 4 blocks, d_state
16, d_conv 4, expand 2, V=256, 16 context symbols) the model has 1,887,744 parameters as measured
by :meth:`MambaBaseline.param_report`. ``baseline.address_input`` adds the HSN's fixed byte codes
to the input (:mod:`hsn.baselines.address`).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.baselines.address import AddressInput
from hsn.config import RunConfig

__all__ = ["MambaBaseline"]

State = tuple[Tensor, Tensor]


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _SelectiveScan(torch.autograd.Function):
    """The scan of one block as a single autograd node with its backward written out.

    ``y_t = h_t · C_t`` with ``h_t = exp(Δ_t A) ⊙ h_{t-1} + drive_t B_t``, ``h`` zeroed before the
    positions ``reset`` marks. Autograd over the position loop is exact too, but it builds a few
    nodes per position and block and spends three quarters of a step walking them; here the
    forward keeps the decays and states (``[B, T, d_inner, d_state]`` each) and the backward is
    one reverse pass over positions plus a few whole-window products. First-order only.
    """

    @staticmethod
    def forward(
        ctx: Any,
        delta: Tensor,  # [B, T, C] step sizes (positive)
        A: Tensor,  # [C, S] (negative)
        drive: Tensor,  # [B, T, C] Δ_t ⊙ u_t
        b_in: Tensor,  # [B, T, S]
        c_in: Tensor,  # [B, T, S]
        h0: Tensor,  # [B, C, S]
        reset: Tensor,  # [B, T] bool
        reset_cols: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        B, T, _ = delta.shape
        decay = torch.exp_(delta.unsqueeze(-1) * A)  # [B, T, C, S]
        hs = drive.unsqueeze(-1) * b_in.unsqueeze(2)  # starts as the driven term, becomes h
        resets = set(reset_cols)
        previous = h0
        for t in range(T):
            if t in resets:
                previous = previous.masked_fill(reset[:, t].view(B, 1, 1), 0.0)
            hs[:, t].addcmul_(decay[:, t], previous)
            previous = hs[:, t]
        y = torch.matmul(hs, c_in.unsqueeze(-1)).squeeze(-1)  # [B, T, C]
        ctx.save_for_backward(delta, A, drive, b_in, c_in, h0, reset)
        ctx.decay, ctx.hs, ctx.resets = decay, hs, resets
        return y, hs[:, -1].clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, grad_y: Tensor | None, grad_last: Tensor | None
    ) -> tuple[Tensor | None, ...]:
        delta, A, drive, b_in, c_in, h0, reset = ctx.saved_tensors
        decay, hs, resets = ctx.decay, ctx.hs, ctx.resets
        B, T, _ = delta.shape
        if grad_y is None:
            grad_y = torch.zeros_like(delta)
        # G[:, t] = dL/dh_t: the read-out term, then what position t + 1 passes back
        G = grad_y.unsqueeze(-1) * c_in.unsqueeze(2)
        if grad_last is not None:
            G[:, -1] += grad_last
        for t in range(T - 2, -1, -1):
            later = G[:, t + 1]
            if t + 1 in resets:  # a row that starts a document at t + 1 passes nothing back
                later = later.masked_fill(reset[:, t + 1].view(B, 1, 1), 0.0)
            G[:, t].addcmul_(decay[:, t + 1], later)
        grad_c = torch.matmul(hs.transpose(2, 3), grad_y.unsqueeze(-1)).squeeze(-1)
        grad_drive = torch.matmul(G, b_in.unsqueeze(-1)).squeeze(-1)
        grad_b = torch.matmul(G.transpose(2, 3), drive.unsqueeze(-1)).squeeze(-1)
        first = G[:, 0]
        if 0 in resets:
            first = first.masked_fill(reset[:, 0].view(B, 1, 1), 0.0)
        grad_h0 = decay[:, 0] * first
        # dL/d(decay_t) = G_t ⊙ h_{t-1}, and decay = exp(Δ A), so dL/d(Δ A) = that ⊙ decay
        G[:, 1:].mul_(hs[:, :-1])
        G[:, 0].mul_(h0)
        for t in resets:  # h_{t-1} was zeroed for the rows that reset at t
            G[:, t].masked_fill_(reset[:, t].view(B, 1, 1), 0.0)
        G.mul_(decay)
        grad_delta = torch.einsum("btcs,cs->btc", G, A)
        grad_a = torch.einsum("btcs,btc->cs", G, delta)
        return grad_delta, grad_a, grad_drive, grad_b, grad_c, grad_h0, None, None


class _Block(nn.Module):
    """One Mamba block; see the module docstring for the recurrence."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dt_rank: int) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.norm = _RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # depthwise causal convolution, tap k reads the input k positions back
        bound = 1.0 / math.sqrt(d_conv)
        self.conv_weight = nn.Parameter(torch.empty(self.d_inner, d_conv).uniform_(-bound, bound))
        self.conv_bias = nn.Parameter(torch.empty(self.d_inner).uniform_(-bound, bound))
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        dt_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse of softplus
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        conv: Tensor,
        reset: Tensor,
        reset_cols: list[int],
        last_reset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``x [B, T, d_model]``, ``h [B, d_inner, d_state]``, ``conv [B, d_conv - 1, d_inner]``;
        ``last_reset [B, T]`` is the position of the latest reset at or before each position
        (``-d_conv`` when the window has none so far)."""
        B, T, _ = x.shape
        K = self.d_conv
        stream, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        # ---- causal depthwise convolution with the carried inputs on the left ----
        padded = torch.cat([conv, stream], dim=1)  # position t of the window is index t + K - 1
        positions = torch.arange(T, device=x.device)
        u = self.conv_bias + padded[:, K - 1 :] * self.conv_weight[:, 0]
        for k in range(1, K):
            tap = padded[:, K - 1 - k : K - 1 - k + T]
            same_doc = (positions - k) >= last_reset  # [B, T]: the tap is not before a reset
            u = u + tap * same_doc.unsqueeze(-1) * self.conv_weight[:, k]
        tail = torch.arange(T - (K - 1), T, device=x.device)  # may be negative: carried inputs
        new_conv = padded[:, T:] * (tail >= last_reset[:, -1:]).unsqueeze(-1)
        u = F.silu(u)
        # ---- selective scan ----
        delta_in, b_in, c_in = self.x_proj(u).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_in))  # [B, T, d_inner]
        y, h = _SelectiveScan.apply(
            delta, -torch.exp(self.A_log), delta * u, b_in, c_in, h, reset, tuple(reset_cols)
        )
        y = y + u * self.D
        return x + self.out_proj(y * F.silu(z)), h, new_conv


class MambaBaseline(nn.Module):
    """Byte + context embeddings → ``n_layers`` Mamba blocks → RMSNorm → linear head."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        spec = cfg.baseline.selective_ssm
        self.vocab_size = cfg.model.vocab_size
        self.n_context = cfg.model.n_context
        self.d_model = spec.d_model
        self.n_layers = spec.n_layers
        self.d_inner = spec.expand * spec.d_model
        self.d_state = spec.d_state
        self.d_conv = spec.d_conv
        dt_rank = spec.dt_rank if spec.dt_rank is not None else math.ceil(spec.d_model / 16)
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.ctx_emb = nn.Embedding(self.n_context, self.d_model)
        self.address = AddressInput(cfg, self.d_model) if cfg.baseline.address_input else None
        self.blocks = nn.ModuleList(
            _Block(self.d_model, spec.d_state, spec.d_conv, spec.expand, dt_rank)
            for _ in range(self.n_layers)
        )
        self.norm = _RMSNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.ctx_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> State:
        """Zeros ``(h, conv)`` in the shapes the module docstring gives."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        h = torch.zeros(self.n_layers, B, self.d_inner, self.d_state, device=device)
        conv = torch.zeros(self.n_layers, B, self.d_conv - 1, self.d_inner, device=device)
        return h, conv

    def reset_state(self, state: State, mask: Tensor) -> State:
        """Zero the rows of both state tensors where ``mask`` (``[B]`` bool) is True."""
        h, conv = self._check_state(state, mask.shape[0])
        if mask.shape != (h.shape[1],) or mask.dtype is not torch.bool:
            raise ValueError(
                f"mask must be a bool [B={h.shape[1]}] tensor, got {tuple(mask.shape)}"
            )
        rows = mask.view(1, -1, 1, 1).to(h.device)
        return h.masked_fill(rows, 0.0), conv.masked_fill(rows, 0.0)

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
        h, conv = self._check_state(state, B)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = self.tok_emb(tokens) + self.ctx_emb(context)
            if self.address is not None:
                x = x + self.address(tokens, reset)
            # one host sync per window: the columns where any row starts a document
            reset_cols = reset.any(dim=0).nonzero().flatten().tolist()
            positions = torch.arange(T, device=tokens.device)
            marks = torch.where(reset, positions, torch.full_like(positions, -self.d_conv))
            last_reset = marks.cummax(dim=1).values
            new_h: list[Tensor] = []
            new_conv: list[Tensor] = []
            for i, block in enumerate(self.blocks):
                x, h_i, conv_i = block(
                    x, h[i].float(), conv[i].float(), reset, reset_cols, last_reset
                )
                new_h.append(h_i)
                new_conv.append(conv_i)
            logits = self.head(self.norm(x))
            out_h = torch.stack(new_h)
            out_conv = torch.stack(new_conv)
        return logits, (out_h, out_conv), {"state_abs_max": out_h.detach().abs().max()}

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
                "MambaBaseline.param_report: component counts do not sum to the total"
            )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {**parts, "stored": stored, "active": stored, "trainable": trainable}

    # helpers ---------------------------------------------------------------------------------

    def _check_state(self, state: Any, B: int) -> State:
        shapes = (
            (self.n_layers, B, self.d_inner, self.d_state),
            (self.n_layers, B, self.d_conv - 1, self.d_inner),
        )
        ok = (
            isinstance(state, tuple)
            and len(state) == 2
            and all(isinstance(s, Tensor) for s in state)
            and tuple(tuple(s.shape) for s in state) == shapes
        )
        if not ok:
            raise ValueError(f"MambaBaseline state must be two tensors of shapes {shapes}")
        return state[0], state[1]


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
