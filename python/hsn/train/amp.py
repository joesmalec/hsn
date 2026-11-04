"""Automatic mixed precision (P0.5): autocast per the device context, a GradScaler only for fp16.

The dtype policy is plan §3.7 and is *decided* by :func:`hsn.device.resolve` from the hardware
profile: :attr:`~hsn.device.DeviceContext.autocast_dtype` is ``torch.bfloat16`` on CUDA
(Ampere and newer), ``torch.float16`` on MPS and ``None`` on the CPU, which runs fp32 without
autocast. :func:`autocast_ctx` turns that into a context manager and :func:`make_grad_scaler`
returns a :class:`torch.amp.GradScaler` only for fp16, whose range overflows without loss
scaling (P1.7 pitfall); bf16 shares fp32's exponent range and needs none, and a scaler under
fp32 would only add work. Master weights stay in :attr:`~hsn.device.DeviceContext.master_dtype`.

``train.precision`` (Appendix A.3) is the value the memory planner (P1.1) uses for bytes per
activation; the device context is authoritative for what actually runs, so a profile and a
config that disagree are resolved in favour of the profile here.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager

import torch

from hsn.device import DeviceContext

__all__ = ["autocast_ctx", "autocast_name", "make_grad_scaler"]

_DTYPE_NAMES: dict[torch.dtype, str] = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def autocast_ctx(ctx: DeviceContext) -> AbstractContextManager[object]:
    """``torch.autocast`` for the context's device and dtype, or a no-op for the fp32 policy."""
    if ctx.autocast_dtype is None:
        return contextlib.nullcontext()
    if ctx.autocast_dtype not in _DTYPE_NAMES:
        raise ValueError(
            f"profile {ctx.profile!r} resolved to autocast dtype {ctx.autocast_dtype}, which is "
            f"not an autocast dtype ({sorted(map(str, _DTYPE_NAMES))})"
        )
    return torch.autocast(device_type=ctx.device.type, dtype=ctx.autocast_dtype)


def autocast_name(ctx: DeviceContext) -> str:
    """``"bf16"``, ``"fp16"`` or ``"none"`` for logs and manifests."""
    if ctx.autocast_dtype is None:
        return "none"
    return _DTYPE_NAMES.get(ctx.autocast_dtype, str(ctx.autocast_dtype))


def make_grad_scaler(ctx: DeviceContext) -> torch.amp.GradScaler | None:
    """A loss scaler when the policy is fp16 autocast; ``None`` for bf16 and fp32 (§3.7)."""
    if ctx.autocast_dtype is not torch.float16:
        return None
    return torch.amp.GradScaler(ctx.device.type)
