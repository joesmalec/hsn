"""DO NOT MERGE: a temporary probe (P1.10c review). Measures, on the CI runner's BLAS, whether the
dense ref_a1 cells stay bitwise identical from the zero state, and the edge-logit gradient after a
forward and backward. Reports the numbers as a warning (shown in the log even when it passes) and
fails only on a crash."""

import platform
import warnings
from pathlib import Path

import torch
from hsn.config import load_config
from hsn.model.hsn import HSN

REF = Path(__file__).resolve().parents[1] / "configs" / "ref_a1.yaml"


def _measure(dtype: torch.dtype, B: int, T: int) -> str:
    torch.manual_seed(0)
    run = load_config(REF)
    model = HSN.from_config(run).to(dtype)
    tokens = torch.randint(0, run.model.vocab_size, (B, T))
    ctx = torch.randint(0, run.model.n_context, (B, T))
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    logits, final, _ = model(tokens, ctx, model.init_state(B), reset)
    logits.logsumexp(-1).mean().backward()
    grad = model.cell.graph.edge_logit.grad
    final_spread = float((final.h - final.h[:, :1]).abs().amax())
    step_spread = 0.0
    with torch.no_grad():
        state = model.init_state(B)
        for t in range(T):
            state = model.step(tokens[:, t], ctx[:, t], state.reset(reset[:, t])).state
            step_spread = max(step_spread, float((state.h - state.h[:, :1]).abs().amax()))
    edge = float(grad.abs().max()) if grad is not None else float("nan")
    return (
        f"{str(dtype).split('.')[-1]} B{B} T{T}: forward final spread {final_spread:.3e}, "
        f"step max spread {step_spread:.3e}, edge grad {edge:.3e}"
    )


def test_zz_macos_symmetry_probe() -> None:
    rows = [_measure(d, B, T) for d in (torch.float64, torch.float32) for B, T in ((2, 6), (4, 64))]
    head = f"SYMMETRY PROBE on {platform.system()} {platform.machine()}, torch {torch.__version__}"
    warnings.warn(head + "\n" + "\n".join(rows), stacklevel=1)
