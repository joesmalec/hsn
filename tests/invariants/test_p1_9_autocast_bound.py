"""P1.9 invariant (Track T, "state coordinates stay in [-1, 1]") under the training precision.

The P1.8 bound test runs in fp32. The G1 run on ``profile:rtx3060`` trains under bf16 autocast, and
there the P1.5 cell let |h| compound past 1: ``W_out`` and the gate emitted bf16, so ``1 - b`` was
rounded and the "convex" combination had weights summing to more than one (the pre-fix G1 training
logs show ``state_abs_max`` 1.06 to 1.37; a pre-fix overfit checkpoint reaches 1.13 over one
256-position window of its own training documents). The cell now runs tanh, the gate and the
combination in the state's dtype; this test pins the bound under the device profile's autocast.

Random saturating weights do not expose the defect: their gates round to exactly 0 or 1 in bf16 and
the pre-fix cell also stays at 1.0. The regime that does is a gate that opens fully at ``h = 0``
and nearly closes (``b`` about 1e-3) once ``|h|`` is about 1, with a candidate ``c`` about +1: in
bf16 ``1 - b`` rounds to exactly 1 and every position adds ``b * c``. :func:`growth_regime` sets it
up from the cell's parameters; measured on rtx3060 after 256 positions: pre-fix cell 1.106, fixed
cell 0.998, fp32 reference 0.998.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import RunConfig, load_hardware_profile
from hsn.device import resolve
from hsn.model.hsn import HSN
from hsn.train.amp import autocast_ctx
from torch import Tensor

ModelFactory = Callable[..., tuple[RunConfig, HSN]]
IdsFactory = Callable[[HSN, int, int, int], tuple[Tensor, Tensor]]
PerturbFn = Callable[..., None]

cuda_param = pytest.param(CUDA_PROFILE, marks=pytest.mark.cuda)
mps_param = pytest.param("m_series", marks=pytest.mark.mps)


def growth_regime(
    model: HSN, *, bias: float = 6.0, slope: float = 13.0, unit_out: float = 5.0
) -> None:
    """The cell's parameters set so that ``b = sigmoid(bias - slope * h)`` coordinatewise (the gate
    reads its own coordinate of ``h_prev``, the third block of ``v = [u; eps; h; m]``) and ``c =
    tanh(unit_out)`` (constant unit outputs through an identity ``W_out``): about 0.998 at
    ``h = 0``, about 1e-3 at ``h = 1``, so ``h`` jumps to ``c`` and then creeps towards it."""
    cell = model.cell
    d = cell.d
    with torch.no_grad():
        cell.gate.weight.zero_()
        cell.gate.weight[:, 2 * d : 3 * d] = -slope * torch.eye(d, device=cell.gate.weight.device)
        cell.gate.bias.fill_(bias)
        cell.W_out.weight.copy_(torch.eye(d, device=cell.W_out.weight.device))
        cell.units.W1.zero_()
        cell.units.b1.zero_()
        cell.units.W2.zero_()
        cell.units.b2.fill_(unit_out)


@pytest.mark.parametrize("profile", [cuda_param, mps_param])
def test_p1_9_state_bound_under_autocast(
    profile: str, make_model: ModelFactory, perturb: PerturbFn, ids: IdsFactory
) -> None:
    """|h| <= 1 after a 256-position window of ``ref_a1`` in the growth regime under the profile's
    autocast dtype (bf16 on rtx3060, fp16 on m_series); the window's own ``state_abs_max``
    statistic and the summary ``q`` agree, and the fp32 run of the same window matches (the casts
    are no-ops without autocast)."""
    hardware = load_hardware_profile(profile)
    if hardware.device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if hardware.device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    ctx = resolve(hardware)
    _cfg, model = make_model("ref", device=ctx.device, overrides=[f"profile={profile}"])
    perturb(model, 1, 3.0)  # everything outside the cell saturated (the P1.8 recipe)
    growth_regime(model)
    B, T = 8, 256
    tokens, ctx_ids = ids(model, B, T, 2)
    reset = torch.zeros(B, T, dtype=torch.bool, device=ctx.device)
    reset[:, 0] = True
    with torch.no_grad(), autocast_ctx(ctx):
        _, state, info = model(tokens, ctx_ids, model.init_state(B, ctx.device), reset)
    assert state.h.dtype is torch.float32, state.h.dtype
    worst = float(state.h.abs().max())
    assert worst <= 1.0, f"|h| = {worst} after {T} positions under {ctx.autocast_dtype}"
    assert float(info["state_abs_max"].max()) <= 1.0
    assert float(state.q.abs().max()) <= 1.0
    assert worst > 0.99  # the regime was reached: h sits just below the bound, not far from it
    with torch.no_grad():
        _, reference, _ = model(tokens, ctx_ids, model.init_state(B, ctx.device), reset)
    assert float(reference.h.abs().max()) <= 1.0
    torch.testing.assert_close(state.h, reference.h, atol=1e-2, rtol=0.0)
