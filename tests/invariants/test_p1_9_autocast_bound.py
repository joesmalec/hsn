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

Two problems with the original setup (measured 8 Nov 2026 on an M4 Pro, torch 2.15):

* P1.5a's RMSNorm over the cell input (``model.cell_norm = rms``, which became the ref_a1 default
  after this test was written) feeds the gate ``v / rms(v)``, and the RMS was dominated by the
  shared features ``u``, so the gate read no ``h``: ``b = sigmoid(6) = 0.9975`` at every
  position and the pre-fix cell measured exactly 1.0 under bf16, inside the bound. The test had
  stopped exposing the defect. The window now runs with ``model.cell_norm=none``, the cell the
  regime was built for; the norm sits before the gate and does not touch the combination this test
  pins.
* ``perturb(model, 1, 3.0)`` drove the layers outside the cell beyond fp16's range (fp32 window:
  ``shared.fc1`` up to 3.9e5, the decoder logits 2.4e10; fp16 stops at 65,504). Under the m_series
  fp16 autocast ``S`` overflowed at position 0 (1,574 infinities in ``fc1``, then NaN after the
  GELU) and NaN entered the cell through ``u``, the scorer's ``alpha`` and the norm, so ``h`` was
  NaN from the first position, not after a drift. CPU fp16 autocast gives the same infinities, so
  this is fp16's range, not an MPS fault, and not the cell's. With ``cell_norm = none`` nothing
  outside the cell changes ``h`` here (the gate reads only its ``h`` block and the units are
  constant), so the perturbation is the default scale 0.3: the fp32 window peaks at 2,568
  (the decoder logits), and the cell's inputs at 364 (the scorer).

Measured after 256 positions with this setup. CPU and MPS autocast give the same numbers in each
dtype, and bf16 reproduces the rtx3060 numbers above (CUDA was not rerun):

=====  ===========  ==================  ============  ==================
dtype  fixed |h|    fixed vs fp32       pre-fix |h|   pre-fix vs fp32
=====  ===========  ==================  ============  ==================
bf16   0.99798      1.5e-5              1.10623       1.1e-1
fp16   0.99796      4.2e-7              0.99516       2.8e-3
=====  ===========  ==================  ============  ==================

In fp16 the pre-fix cell's ``1 - b`` (one ulp is 2^-11 just below 1) rounds down in this regime, so
its |h| stays under 1. The comparison with the fp32 window (``atol`` 1e-4) catches it there.
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

#: The cell the growth regime is built for: the gate reads ``h`` itself (module docstring).
REGIME_OVERRIDES = ("model.cell_norm=none",)
#: The autocast window against the fp32 window of the fixed cell: measured 1.5e-5 (bf16) and 4.2e-7
#: (fp16); the pre-fix cell is 1.1e-1 (bf16) and 2.8e-3 (fp16) away.
FP32_ATOL = 1e-4


def growth_regime(
    model: HSN, *, bias: float = 6.0, slope: float = 13.0, unit_out: float = 5.0
) -> None:
    """The cell's parameters set so that ``b = sigmoid(bias - slope * h)`` coordinatewise (the gate
    reads its own coordinate of ``h_prev``, the third block of ``v = [u; eps; h; m]``, which needs
    ``model.cell_norm = none``) and ``c = tanh(unit_out)`` (constant unit outputs through an
    identity ``W_out``): about 0.998 at ``h = 0``, about 1e-3 at ``h = 1``, so ``h`` jumps to ``c``
    and then creeps towards it."""
    cell = model.cell
    if cell.cell_norm != "none":
        raise ValueError(
            f"growth_regime needs model.cell_norm=none (got {cell.cell_norm!r}): a norm over v "
            "rescales the gate's h block by the RMS of u, eps and m and the gate never closes"
        )
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
    statistic and the summary ``q`` agree, and the fp32 run of the same window matches to
    :data:`FP32_ATOL` (the casts are no-ops without autocast). The cell's inputs must be finite and
    the gate must open at ``h = 0`` and nearly close by the end: otherwise the setup, not the cell,
    is what failed, and a pass would prove nothing."""
    hardware = load_hardware_profile(profile)
    if hardware.device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if hardware.device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    ctx = resolve(hardware)
    _cfg, model = make_model(
        "ref", device=ctx.device, overrides=[f"profile={profile}", *REGIME_OVERRIDES]
    )
    # non-trivial weights outside the cell, inside fp16's range (module docstring)
    perturb(model, 1)
    growth_regime(model)
    B, T = 8, 256
    tokens, ctx_ids = ids(model, B, T, 2)
    reset = torch.zeros(B, T, dtype=torch.bool, device=ctx.device)
    reset[:, 0] = True
    with torch.no_grad(), autocast_ctx(ctx):
        _, state, info = model(
            tokens, ctx_ids, model.init_state(B, ctx.device), reset, capture=True
        )
    for key in ("u", "eps"):  # an overflow outside the cell is a setup failure, not a bound one
        assert bool(torch.isfinite(info[key]).all()), (
            f"{key} is not finite under {ctx.autocast_dtype}: the setup overflows the dtype"
        )
    gates = info["cell_gates"]  # [B, T, N, d]; the dense round writes every cell
    assert float(gates[:, 0].min()) > 0.99, "the regime's gate is not open at h = 0"
    assert float(gates[:, -1].max()) < 1e-2, "the regime's gate did not close once |h| is near 1"
    assert state.h.dtype is torch.float32, state.h.dtype
    worst = float(state.h.abs().max())
    assert worst <= 1.0, f"|h| = {worst} after {T} positions under {ctx.autocast_dtype}"
    assert float(info["state_abs_max"].max()) <= 1.0
    assert float(state.q.abs().max()) <= 1.0
    assert worst > 0.99  # the regime was reached: h sits just below the bound, not far from it
    with torch.no_grad():
        _, reference, _ = model(tokens, ctx_ids, model.init_state(B, ctx.device), reset)
    assert float(reference.h.abs().max()) <= 1.0
    torch.testing.assert_close(state.h, reference.h, atol=FP32_ATOL, rtol=0.0)
