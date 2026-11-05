"""P1.8: the finite-difference gradient check of ``HSN.forward`` on the tiny fp64 configuration."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from torch import Tensor
from torch.func import functional_call

IdsFactory = Callable[[HSN, int, int, int], tuple[Tensor, Tensor]]
PerturbFn = Callable[..., None]
StateFactory = Callable[..., CellGraphState]

B, T = 2, 3
#: The chunk's "including edge logits and the gate": these must be among the checked inputs.
REQUIRED_INPUTS = ("cell.graph.edge_logit", "cell.gate.weight", "cell.gate.bias")


def test_p1_8_gradcheck_tiny_fp64(
    tiny_model: HSN,
    perturb: PerturbFn,
    ids: IdsFactory,
    random_state: StateFactory,
    deterministic_kernels: None,
) -> None:
    """``torch.autograd.gradcheck`` of the window function over 3 positions (a document reset at
    position 1 of row 0) with respect to the incoming state ``h_0`` and every parameter tensor of
    the model, the edge logits and the gate among them, in double precision on the CPU with
    nondeterministic kernels disabled; every element of the Jacobian is compared (slow mode).
    The same scalar's backward then shows that every gradient is finite and non-zero, except the
    halting head's, which is unused while ``r_max = 1`` (``None``)."""
    model = tiny_model
    assert model.dtype is torch.float64 and model.device.type == "cpu"
    assert torch.are_deterministic_algorithms_enabled()
    perturb(model, seed=3)
    tokens, ctx = ids(model, B, T, seed=4)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 1] = True

    names = [name for name, _ in model.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for _, p in model.named_parameters())
    h0 = random_state(model, B, seed=5).h.detach().clone().requires_grad_(True)
    assert h0.dtype is torch.float64 and all(p.dtype is torch.float64 for p in params)
    for required in REQUIRED_INPUTS:
        assert required in names, required
    assert any(name.startswith("halting.") for name in names)
    report = model.param_report()
    n_inputs = h0.numel() + sum(p.numel() for p in params)
    # 1,083 + the 16 of amendment A1's cell_norm gain (4d = 16 for the tiny configuration)
    assert report["total"] == 1_099 and report["edges"] == model.cell.graph.n_edges == 6
    assert n_inputs == report["total_trainable"] + B * model.cfg.n_cells * model.cfg.d == 1_129

    def window(h: Tensor, *values: Tensor) -> tuple[Tensor, ...]:
        """Every differentiable output of one window from the state ``h`` under the parameters
        ``values`` (bound by name; the int/bool buffers stay the module's own)."""
        state = model.init_state(B).with_dense(h)
        bound = dict(zip(names, values, strict=True))
        logits, final, info = functional_call(model, bound, (tokens, ctx, state, reset))
        return logits, final.h, final.q, info["z"], info["z_hat"], info["eps"], info["u"]

    outputs = window(h0, *params)
    n_outputs = sum(int(o.numel()) for o in outputs)
    assert all(o.requires_grad and o.dtype is torch.float64 for o in outputs)
    started = time.perf_counter()
    assert torch.autograd.gradcheck(window, (h0, *params), atol=1e-6, rtol=1e-4)
    seconds = time.perf_counter() - started

    # every input the outputs depend on receives a finite, non-zero gradient; the halting head is
    # defined but not called while r_max = 1, so autograd reports no gradient for it
    scalar = sum(o.square().sum() for o in outputs)
    assert isinstance(scalar, Tensor)
    grads = torch.autograd.grad(scalar, (h0, *params), allow_unused=True)
    for name, grad in zip(["h0", *names], grads, strict=True):
        if name.startswith("halting."):
            assert grad is None, name
            continue
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0, name
    print(
        f"gradcheck: {n_inputs} inputs x {n_outputs} outputs over {T} positions, fp64, "
        f"{seconds:.2f} s"
    )
