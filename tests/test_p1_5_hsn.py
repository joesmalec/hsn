"""P1.5 tests: the dense PDG cell step, the scorer features, the HSN module, the parameter count
against the P1.1 closed form, and a pure-Python reference of one step in fp64."""

import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig, load_config
from hsn.model.halting import HaltingHead
from hsn.model.hsn import HSN, PARAM_REPORT_ROWS, StepOutput
from hsn.model.router import SCORER_FEATURE_NAMES, EMAFeatures, Scorer
from hsn.model.state import CellGraphState
from hsn.train.memory_plan import SCORER_SCALAR_FEATURES, expected_param_count
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
DEVICES = ["cpu", cuda_param, mps_param]

AUX_KEYS = ("state_abs_max", "eps_abs_mean", "alpha_entropy")
#: P1.10a: ``HSN.forward`` runs E and D over the window ``[B, T]``, ``HSN.step`` over ``[B]``; the
#: GEMM rounds differently with the row count, so forward-vs-step and cross-T comparisons hold to
#: these tolerances (fp64, fp32; P2.5's parity tolerance) instead of bit for bit. Same-shape
#: comparisons stay bit-exact.
WINDOW_ATOL = {torch.float64: 1e-12, torch.float32: 1e-5}


def _close_window(actual: Tensor, expected: Tensor) -> None:
    """``actual`` equals ``expected`` to the forward-vs-step tolerance of their dtype."""
    atol = WINDOW_ATOL[expected.dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.0)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    # The device as tensors report it (``cuda:0`` / ``mps:0``, with the index): a bare
    # ``torch.device("mps")`` never compares equal to a tensor's device (found on MPS, P1.7).
    return torch.empty(0, device=device_type).device


def _model(path: Path, overrides: Sequence[str] = ()) -> tuple[RunConfig, HSN]:
    """A model from a config file (fp64 when the config says so), weights seeded by ``run.seed``."""
    cfg = load_config(path, overrides=list(overrides))
    torch.manual_seed(cfg.seed)
    model = HSN.from_config(cfg)
    if cfg.train.precision == "fp64":
        model = model.double()
    return cfg, model


def _perturb(model: HSN, seed: int, scale: float = 0.3) -> None:
    """Add seeded noise to every parameter: non-zero biases, non-uniform edge weights, so that a
    wrong bias row, a wrong edge or a swapped operand would show in the outputs."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
            p.add_(noise.to(dtype=p.dtype, device=p.device))


def _ids(cfg: RunConfig, B: int, T: int, seed: int) -> tuple[Tensor, Tensor]:
    """Tokens and context ids ``[B, T]`` drawn on the CPU from ``seed``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m = cfg.model
    tokens = torch.randint(0, m.vocab_size, (B, T), generator=generator)
    ctx = torch.randint(0, m.n_context, (B, T), generator=generator)
    return tokens, ctx


def _random_state(model: HSN, B: int, seed: int, ticks: int = 2) -> CellGraphState:
    """A state with non-trivial ``h`` (|h| < 1), ``q``, ``age`` and ``pos``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    N, d = model.cfg.n_cells, model.cfg.d
    h = torch.rand(B, N, d, generator=generator, dtype=torch.float64) * 1.6 - 0.8
    state = model.init_state(B).with_dense(h.to(dtype=model.dtype, device=model.device))
    for _ in range(ticks):
        mask = torch.rand(B, N, generator=generator) < 0.5
        state = state.tick(mask.to(model.device))
    return state


# --------------------------------------------------------------------------------------------------
# The pure-Python reference: one input step written out with loops and the math module. It reads
# the weights as nested lists and never calls torch, so it checks the wiring (operand order,
# layouts, the softmax over units, the P1.5a RMSNorm on the cell input, tanh on the mixed output,
# the updated q in the decoder) against plan §2.2 / §2.3 / §2.4 and amendment A1 rather than
# against torch itself.
# --------------------------------------------------------------------------------------------------


def _as_lists(model: HSN) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, Tensor):
            out[key] = value.detach().cpu().double().tolist()
    return out


def _dims(cfg: RunConfig, model: HSN) -> dict[str, Any]:
    m = cfg.model
    graph = model.cell.graph
    return {
        "d": m.d,
        "N": m.n_cells,
        "M": m.n_units,
        "V": m.vocab_size,
        "n_ctx": m.n_context,
        "age_max": m.age_max,
        "r_max": m.r_max,
        "unroll": cfg.train.unroll,
        "budget": m.budget,
        "scorer": m.scorer,
        "exclude": set(m.scorer_features.exclude),
        "edge_norm": m.edge_norm,
        "cell_norm": m.cell_norm,  # P1.5a: RMSNorm on the cell input, or none
        "cand_norm": m.cand_norm,  # P1.13: RMSNorm before the candidate's tanh, or none
        "ablate_shared_u": m.ablate_shared_u,  # the cell input without u (test_ablate_shared_u)
        "disabled": set(m.disabled_cells),
        "indptr": graph.indptr.tolist(),
        "indices": graph.indices.tolist(),
    }


def _linear(W: list[list[float]], b: list[float] | None, x: list[float]) -> list[float]:
    """``nn.Linear`` layout: ``W[out][in]``."""
    return [
        sum(W[o][i] * x[i] for i in range(len(x))) + (b[o] if b is not None else 0.0)
        for o in range(len(W))
    ]


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rmsnorm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """``nn.RMSNorm`` (P1.5a, amendment A1): ``x / sqrt(mean(x²) + eps) * weight``."""
    scale = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * scale * w for v, w in zip(x, weight, strict=True)]


def _softmax(xs: list[float]) -> list[float]:
    top = max(xs)
    exps = [math.exp(x - top) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _mlp2(p: dict[str, Any], prefix: str, x: list[float]) -> list[float]:
    hidden = [_gelu(v) for v in _linear(p[f"{prefix}.fc1.weight"], p[f"{prefix}.fc1.bias"], x)]
    return _linear(p[f"{prefix}.fc2.weight"], p[f"{prefix}.fc2.bias"], hidden)


def _unit(p: dict[str, Any], j: int, v: list[float]) -> list[float]:
    """``F_j(v)`` with the bank's ``[in, out]`` layout: ``x @ W1[j] + b1[j]``."""
    W1, b1, W2, b2 = (
        p["cell.units.W1"][j],
        p["cell.units.b1"][j],
        p["cell.units.W2"][j],
        p["cell.units.b2"][j],
    )
    hidden = [_gelu(sum(v[i] * W1[i][o] for i in range(len(v))) + b1[o]) for o in range(len(b1))]
    return [sum(hidden[k] * W2[k][o] for k in range(len(hidden))) + b2[o] for o in range(len(b2))]


def _ref_features(
    dims: dict[str, Any],
    u: list[float],
    eps: list[float],
    h: list[list[float]],
    age: list[int],
    pos: int,
    ctx: int,
    budget: float,
    round_idx: int,
    ema_u: list[float],
    ema_s: list[float],
) -> list[list[float]]:
    feats = []
    d = dims["d"]
    for i in range(dims["N"]):
        scalars = [
            age[i] / dims["age_max"],
            budget,
            round_idx / dims["r_max"],
            ctx / dims["n_ctx"],
            ema_u[i],
            ema_s[i],
            pos / dims["unroll"],
            1.0,
        ]
        blocks = {"u": u, "eps": eps, "h": h[i]}
        row: list[float] = []
        for name in SCORER_FEATURE_NAMES[:3]:
            block = blocks[name]
            row += [0.0] * d if name in dims["exclude"] else block
        for name, value in zip(SCORER_FEATURE_NAMES[3:], scalars, strict=True):
            row.append(0.0 if name in dims["exclude"] else value)
        feats.append(row)
    return feats


def _ref_step(
    p: dict[str, Any],
    dims: dict[str, Any],
    h: list[list[float]],
    q: list[float],
    age: list[int],
    pos: int,
    x: int,
    ctx: int,
) -> dict[str, Any]:
    """One position for one batch row; every intermediate of plan §2.2–§2.4 as lists."""
    d, N, M = dims["d"], dims["N"], dims["M"]
    z_hat = _mlp2(p, "predictor", q)
    z = _mlp2(p, "encoder", p["byte_emb.weight"][x])
    eps = [z[i] - z_hat[i] for i in range(d)]
    u = _mlp2(p, "shared", z + q + p["ctx_emb.weight"][ctx])

    feats = _ref_features(dims, u, eps, h, age, pos, ctx, dims["budget"], 0, [0.0] * N, [0.0] * N)
    if dims["scorer"] == "linear":
        unit_logits = [
            _linear(p["router.scorer.net.weight"], p["router.scorer.net.bias"], f) for f in feats
        ]
    else:
        unit_logits = [_mlp2(p, "router.scorer.net", f) for f in feats]
    alpha = [_softmax(row) for row in unit_logits]

    Vh = [_linear(p["cell.V.weight"], None, h[j]) for j in range(N)]
    indptr, indices, logit = dims["indptr"], dims["indices"], p["cell.graph.edge_logit"]
    m = []
    for i in range(N):
        lo, hi = indptr[i], indptr[i + 1]
        row_logits = logit[lo:hi]
        if dims["edge_norm"] == "softmax":
            weights = _softmax(row_logits) if row_logits else []
        else:
            weights = [_sigmoid(v) for v in row_logits]
        m_i = [0.0] * d
        for k, w in enumerate(weights):
            src = indices[lo + k]
            for c in range(d):
                m_i[c] += w * Vh[src][c]
        m.append(m_i)

    v, c_out, b_out, h_new, unit_out = [], [], [], [], []
    for i in range(N):
        # the cell input [u; eps; h_i; m_i], or [eps; h_i; m_i] with model.ablate_shared_u
        v_raw = (eps + h[i] + m[i]) if dims["ablate_shared_u"] else (u + eps + h[i] + m[i])
        # P1.5a: the units and the gate read the normalised input; the gated update reads h[i]
        v_i = _rmsnorm(v_raw, p["cell.norm.weight"]) if dims["cell_norm"] == "rms" else v_raw
        outs = [_unit(p, j, v_i) for j in range(M)]
        mixed = [sum(alpha[i][j] * outs[j][c] for j in range(M)) for c in range(d)]
        pre = _linear(p["cell.W_out.weight"], None, mixed)
        if dims["cand_norm"] == "rms":  # P1.13 (amendment A4)
            pre = _rmsnorm(pre, p["cell.cand_norm.weight"])
        c_i = [math.tanh(val) for val in pre]
        b_i = [_sigmoid(val) for val in _linear(p["cell.gate.weight"], p["cell.gate.bias"], v_i)]
        if i in dims["disabled"]:
            h_i = list(h[i])
        else:
            h_i = [(1.0 - b_i[c]) * h[i][c] + b_i[c] * c_i[c] for c in range(d)]
        v.append(v_i)
        unit_out.append(outs)
        c_out.append(c_i)
        b_out.append(b_i)
        h_new.append(h_i)
    q_new = [sum(h_new[i][c] for i in range(N)) / N for c in range(d)]
    logits = _mlp2(p, "decoder", q_new + u)
    age_new = [age[i] + 1 if i in dims["disabled"] else 0 for i in range(N)]
    return {
        "z_hat": z_hat,
        "z": z,
        "eps": eps,
        "u": u,
        "feats": feats,
        "unit_logits": unit_logits,
        "alpha": alpha,
        "m": m,
        "v": v,
        "unit_out": unit_out,
        "c": c_out,
        "b": b_out,
        "h": h_new,
        "q": q_new,
        "logits": logits,
        "age": age_new,
        "pos": pos + 1,
    }


def _close(actual: Tensor, expected: Any, atol: float = 1e-9) -> None:
    ref = torch.tensor(expected, dtype=torch.float64)
    assert actual.shape == ref.shape, (tuple(actual.shape), tuple(ref.shape))
    diff = (actual.detach().cpu().double() - ref).abs().max().item()
    assert diff <= atol, f"max abs diff {diff} > {atol}"


def _check_position(
    cfg: RunConfig, model: HSN, state: CellGraphState, x: Tensor, ctx: Tensor
) -> StepOutput:
    """Run ``model.step`` from ``state`` and compare every intermediate against the reference."""
    p, dims = _as_lists(model), _dims(cfg, model)
    out = model.step(x, ctx, state)
    feats = model.scorer.features(
        out.u, out.eps, state, ctx, budget_frac=cfg.model.budget, round_idx=0, ema=None
    )
    unit_logits = model.scorer(feats)
    alpha = torch.softmax(unit_logits, dim=-1)
    h_new, info = model.cell.step_dense(out.u, out.eps, state, alpha)
    assert torch.equal(h_new, out.state.h)
    for b in range(state.batch_size):
        ref = _ref_step(
            p,
            dims,
            state.h[b].tolist(),
            state.q[b].tolist(),
            state.age[b].tolist(),
            int(state.pos[b]),
            int(x[b]),
            int(ctx[b]),
        )
        _close(out.z_hat[b], ref["z_hat"])
        _close(out.z[b], ref["z"])
        _close(out.eps[b], ref["eps"])
        _close(out.u[b], ref["u"])
        _close(feats[b], ref["feats"])
        _close(unit_logits[b], ref["unit_logits"])
        _close(alpha[b], ref["alpha"])
        _close(info["m"][b], ref["m"])
        _close(info["v"][b], ref["v"])
        _close(info["unit_out"][b], ref["unit_out"])
        _close(info["c"][b], ref["c"])
        _close(info["b"][b], ref["b"])
        _close(out.state.h[b], ref["h"])
        _close(out.state.q[b], ref["q"])
        _close(out.logits[b], ref["logits"])
        assert out.state.age[b].tolist() == ref["age"]
        assert int(out.state.pos[b]) == ref["pos"]
    return out


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p1_5_param_count_is_1_820_617_and_per_component() -> None:
    """Appendix C.1's 1,820,361 plus the 256 of amendment A1's ``cell_norm`` row (P1.5a)."""
    cfg, model = _model(REF_A1)
    expected = expected_param_count(cfg.model)
    report = model.param_report()
    assert list(report) == [*PARAM_REPORT_ROWS, "total", "edges", "total_trainable"]
    assert list(expected) == [*PARAM_REPORT_ROWS, "total"]  # the P1.1 table order
    for component, count in expected.items():
        assert report[component] == count, component
    assert report["total"] == 1_820_617
    # Appendix C.1 to the digit, measured from the modules
    assert report["scorer"] == 1_608 == model.scorer.param_count()
    assert model.scorer.in_features == 3 * cfg.model.d + SCORER_SCALAR_FEATURES == 200
    assert report["halting"] == 129 == model.halting.param_count()
    assert report["gate"] == 16_448 and report["w_out"] == report["v_msg"] == 4_096
    assert report["cell_norm"] == 256 == model.cell.norm.weight.numel()  # A1: the 4d gain
    assert report["units"] == 1_315_328 == model.cell.units.param_count()
    assert report["decoder"] == 197_376 and report["shared"] == 131_648
    # the P1.4 edge logits are trainable but outside Appendix C; the run manifest records them
    assert report["edges"] == cfg.model.n_cells * cfg.model.in_degree == 128
    assert report["edges"] == model.cell.graph.edge_logit.numel()
    measured = sum(p.numel() for p in model.parameters())
    assert report["total_trainable"] == measured == 1_820_617 + 128 == 1_820_745
    assert all(p.requires_grad for p in model.parameters())
    # every parameter belongs to exactly one row (or is the edge logit)
    owners = {
        "byte_emb": "byte_emb.",
        "ctx_emb": "ctx_emb.",
        "encoder": "encoder.",
        "predictor": "predictor.",
        "shared": "shared.",
        "units": "cell.units.",
        "w_out": "cell.W_out.",
        "v_msg": "cell.V.",
        "gate": "cell.gate.",
        "cell_norm": "cell.norm.",
        "scorer": "router.scorer.",
        "halting": "halting.",
        "decoder": "decoder.",
    }
    for name, _ in model.named_parameters():
        rows = [row for row, prefix in owners.items() if name.startswith(prefix)]
        assert len(rows) == 1 or name == "cell.graph.edge_logit", name

    # the tiny configuration (1,083 + 16 for the A1 gain) and three overrides, row by row
    cfg_t, tiny = _model(TINY)
    report_t = tiny.param_report()
    expected_t = expected_param_count(cfg_t.model)
    assert all(report_t[k] == v for k, v in expected_t.items())
    assert report_t["total"] == 1_099 and report_t["edges"] == 3 * 2
    assert report_t["total_trainable"] == sum(p.numel() for p in tiny.parameters()) == 1_105

    cfg_m, mlp = _model(REF_A1, ["model.scorer=mlp"])
    report_m = mlp.param_report()
    expected_m = expected_param_count(cfg_m.model)
    assert all(report_m[k] == v for k, v in expected_m.items())
    assert report_m["scorer"] == 26_760 and report_m["total"] == 1_820_617 - 1_608 + 26_760

    cfg_r, random_edges = _model(REF_A1, ["model.random_edges=2"])
    report_r = random_edges.param_report()
    assert report_r["total"] == 1_820_617 == expected_param_count(cfg_r.model)["total"]
    assert report_r["edges"] == 32 * 6 == 192
    assert report_r["total_trainable"] == 1_820_617 + 192

    # model.cell_norm=none drops the row and reproduces the pre-amendment total of 1,820,361
    cfg_n, no_norm = _model(REF_A1, ["model.cell_norm=none"])
    report_n = no_norm.param_report()
    assert report_n["cell_norm"] == 0 == expected_param_count(cfg_n.model)["cell_norm"]
    assert report_n["total"] == 1_820_361 and isinstance(no_norm.cell.norm, torch.nn.Identity)


def test_p1_5_state_bounded_after_1000_steps() -> None:
    cfg, model = _model(REF_A1)
    with torch.no_grad():  # saturating weights: tanh and sigmoid near their limits, large messages
        for p in model.parameters():
            p.mul_(20.0)
    B, steps = 2, 1000
    m = cfg.model
    generator = torch.Generator(device="cpu").manual_seed(1)
    state = model.init_state(B)
    worst_h = torch.zeros(())
    worst_q = torch.zeros(())
    worst_aux = torch.zeros(())
    n_resets = torch.zeros(B, dtype=torch.int64)
    with torch.no_grad():
        for _ in range(steps):
            x = torch.randint(0, m.vocab_size, (B,), generator=generator)
            ctx = torch.randint(0, m.n_context, (B,), generator=generator)
            reset = torch.rand(B, generator=generator) < 0.01
            n_resets += reset.to(torch.int64)
            state = state.reset(reset)
            out = model.step(x, ctx, state)
            state = out.state
            worst_h = torch.maximum(worst_h, state.h.abs().amax())
            worst_q = torch.maximum(worst_q, state.q.abs().amax())
            worst_aux = torch.maximum(worst_aux, out.aux["state_abs_max"])
    assert worst_h.item() <= 1.0 and worst_q.item() <= 1.0 and worst_aux.item() <= 1.0
    assert worst_h.item() > 0.5  # the bound is exercised, not vacuous
    assert torch.isfinite(state.h).all() and torch.isfinite(state.q).all()
    assert torch.isfinite(out.logits).all()
    assert torch.equal(state.q, state.recompute_q())  # dense mode: the exact mean, no drift
    assert (state.age == 0).all()  # every cell updated every step
    # pos counts positions since the last reset
    assert n_resets.sum().item() > 0 or (state.pos == steps).all()
    assert (state.pos <= steps).all() and (state.pos >= 1).all()


def test_p1_5_prediction_independent_of_x_t() -> None:
    for path in (TINY, REF_A1):
        cfg, model = _model(path)
        _perturb(model, seed=3)
        B, T = 3, 6
        state = _random_state(model, B, seed=4)
        tokens, ctx = _ids(cfg, B, T, seed=5)
        other_tokens, other_ctx = _ids(cfg, B, T, seed=6)
        assert not torch.equal(tokens, other_tokens) and not torch.equal(ctx, other_ctx)

        # the same state, another x_t and ctx_t: ẑ_t is bit for bit the same
        out = model.step(tokens[:, 0], ctx[:, 0], state)
        other = model.step(other_tokens[:, 0], other_ctx[:, 0], state)
        assert torch.equal(out.z_hat, other.z_hat)
        assert torch.equal(out.z_hat, model.predictor(state.q))
        assert not torch.equal(out.z, other.z) and not torch.equal(out.u, other.u)
        assert not torch.equal(out.logits, other.logits)

        # causality through a window: changing the inputs from position 3 on leaves everything
        # up to position 2 untouched, and ẑ_3 too (it is computed from the state before x_3)
        reset = torch.zeros(B, T, dtype=torch.bool)
        logits_a, _, info_a = model(tokens, ctx, state, reset)
        mixed_tokens = torch.cat([tokens[:, :3], other_tokens[:, 3:]], dim=1)
        mixed_ctx = torch.cat([ctx[:, :3], other_ctx[:, 3:]], dim=1)
        logits_b, _, info_b = model(mixed_tokens, mixed_ctx, state, reset)
        assert torch.equal(logits_a[:, :3], logits_b[:, :3])
        assert torch.equal(info_a["z_hat"][:, :4], info_b["z_hat"][:, :4])
        assert torch.equal(info_a["u"][:, :3], info_b["u"][:, :3])
        assert not torch.equal(logits_a[:, 3:], logits_b[:, 3:])
        assert not torch.equal(info_a["z_hat"][:, 4:], info_b["z_hat"][:, 4:])
        # perturbing x_t alone never changes ẑ_t (only ẑ_{t+1} onwards)
        for t in range(T):
            perturbed = tokens.clone()
            perturbed[:, t] = other_tokens[:, t]
            _, _, info_c = model(perturbed, ctx, state, reset)
            assert torch.equal(info_c["z_hat"][:, : t + 1], info_a["z_hat"][:, : t + 1])


def test_p1_5_disabled_cell_never_changes() -> None:
    for path, disabled in ((TINY, [1]), (TINY, [0, 2]), (REF_A1, [0, 2]), (REF_A1, [31])):
        cfg, model = _model(path, [f"model.disabled_cells={disabled}"])
        _perturb(model, seed=7)
        assert model.cell.disabled_cells == tuple(disabled)
        m = cfg.model
        enabled = [i for i in range(m.n_cells) if i not in disabled]
        B, T = 3, 25
        state = _random_state(model, B, seed=8, ticks=0)
        initial = state.h.clone()
        tokens, ctx = _ids(cfg, B, T, seed=9)
        reset = torch.zeros(B, T, dtype=torch.bool)

        logits, state, _ = model(tokens, ctx, state, reset)
        assert torch.isfinite(logits).all()
        assert torch.equal(state.h[:, disabled], initial[:, disabled])  # bit for bit
        for i in enabled:
            assert not torch.equal(state.h[:, i], initial[:, i]), f"cell {i} did not change"
        assert (state.age[:, disabled] == T).all() and (state.age[:, enabled] == 0).all()
        assert (state.pos == T).all()
        assert torch.allclose(state.q, state.recompute_q(), atol=0, rtol=0)

        # a reset zeroes the disabled cells of the reset session only; they then stay at zero
        reset2 = torch.zeros(B, T, dtype=torch.bool)
        reset2[0, 4] = True
        _, state2, _ = model(tokens, ctx, state, reset2)
        assert (state2.h[0, disabled] == 0).all()
        assert torch.equal(state2.h[1:, disabled], initial[1:, disabled])
        assert (state2.age[0, disabled] == T - 4).all()  # reset before position 4
        assert (state2.age[1:, disabled] == 2 * T).all()
        assert int(state2.pos[0]) == T - 4 and (state2.pos[1:] == 2 * T).all()

        # step_dense computes the candidate for every cell but carries the disabled rows through
        out = model.step(tokens[:, 0], ctx[:, 0], state)
        alpha = torch.full((B, m.n_cells, m.n_units), 1.0 / m.n_units, dtype=model.dtype)
        h_new, info = model.cell.step_dense(out.u, out.eps, state, alpha)
        assert torch.equal(h_new[:, disabled], state.h[:, disabled])
        assert torch.isfinite(info["c"][:, disabled]).all()
        assert (info["b"] >= 0).all() and (info["b"] <= 1).all()  # fp32 sigmoid may saturate
        assert not torch.equal(h_new[:, enabled], state.h[:, enabled])

        # the gradient of the carried rows never comes from the discarded candidate: with
        # everything but the disabled rows' candidates perturbed, the disabled rows stay put
        h0 = state.h.detach().clone().requires_grad_(True)
        tracked = model.init_state(B).with_dense(h0)
        out2 = model.step(tokens[:, 1], ctx[:, 1], tracked)
        (grad,) = torch.autograd.grad(out2.state.h[:, disabled].sum(), h0)
        expected = torch.zeros_like(h0)
        expected[:, disabled] = 1.0
        assert torch.equal(grad, expected)

    # the config refuses cells outside [0, N)
    with pytest.raises(ValueError, match="disabled_cells"):
        load_config(TINY, overrides=["model.disabled_cells=[3]"])


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_5_shapes(device_type: str) -> None:
    device = _device(device_type)
    for path in (TINY, REF_A1):
        cfg, model = _model(path)
        if cfg.train.precision == "fp64" and device.type == "mps":
            continue  # MPS has no float64
        model = model.to(device)
        m = cfg.model
        B, T = cfg.train.batch, 5
        d, N, M, V = m.d, m.n_cells, m.n_units, m.vocab_size
        F = 3 * d + SCORER_SCALAR_FEATURES
        dtype = model.dtype
        state = model.init_state(B, device)
        assert state.device == device and state.dtype == dtype
        assert (state.batch_size, state.n_cells, state.width) == (B, N, d)
        assert model.init_state(B).device == device  # the model's device by default
        tokens, ctx = (t.to(device) for t in _ids(cfg, B, T, seed=10))

        out = model.step(tokens[:, 0], ctx[:, 0], state)
        assert isinstance(out, StepOutput)
        assert out.logits.shape == (B, V)
        for name in ("z", "z_hat", "eps", "u"):
            assert getattr(out, name).shape == (B, d), name
            assert getattr(out, name).dtype == dtype, name
        assert out.state.h.shape == (B, N, d) and out.state.q.shape == (B, d)
        assert out.state.age.shape == (B, N) and out.state.pos.shape == (B,)
        assert (out.state.pos == 1).all() and (out.state.age == 0).all()
        assert out.state.dtype == dtype and out.state.device == device
        assert float(out.cost) == 1.0 and out.n_rounds == 1  # P2.2: a 0-dim tensor
        assert set(out.aux) == set(AUX_KEYS)
        for key in AUX_KEYS:
            assert out.aux[key].shape == () and not out.aux[key].requires_grad, key
            assert torch.isfinite(out.aux[key]).item(), key
        assert 0.0 <= out.aux["alpha_entropy"].item() <= math.log(M) + 1e-6
        assert out.aux["state_abs_max"].item() <= 1.0
        assert out.logits.requires_grad and out.state.h.requires_grad and out.state.q.requires_grad
        assert torch.isfinite(out.logits).all()

        feats = model.scorer.features(
            out.u, out.eps, state, ctx[:, 0], budget_frac=1.0, round_idx=0
        )
        assert feats.shape == (B, N, F) and feats.dtype == dtype
        if path == REF_A1:
            assert F == 200
        unit_logits = model.scorer(feats)
        assert unit_logits.shape == (B, N, M)
        alpha = torch.softmax(unit_logits, dim=-1)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(B, N, dtype=dtype, device=device))
        h_new, info = model.cell.step_dense(out.u, out.eps, state, alpha)
        assert h_new.shape == (B, N, d) and torch.equal(h_new, out.state.h)
        assert info["m"].shape == (B, N, d) and info["v"].shape == (B, N, 4 * d)
        assert info["unit_out"].shape == (B, N, M, d) and info["mixed"].shape == (B, N, d)
        assert info["c"].shape == info["b"].shape == (B, N, d)
        assert (info["c"].abs() <= 1).all() and (info["b"] >= 0).all() and (info["b"] <= 1).all()
        assert torch.equal(info["h_prev"], state.h)
        p_halt = model.halting(out.state.q, out.u)
        assert p_halt.shape == (B,) and (p_halt > 0).all() and (p_halt < 1).all()

        reset = torch.zeros(B, T, dtype=torch.bool, device=device)
        logits, final, info_w = model(tokens, ctx, state, reset)
        assert logits.shape == (B, T, V) and logits.dtype == dtype
        _close_window(logits[:, 0], out.logits)  # the window starts with the same step
        assert (final.pos == T).all() and final.h.shape == (B, N, d)
        for name in ("z", "z_hat", "eps", "u"):
            assert info_w[name].shape == (B, T, d), name
        for key in AUX_KEYS:
            assert info_w[key].shape == (T,), key
        assert info_w["cost"].shape == (T,) and info_w["cost"].dtype == dtype
        assert (info_w["cost"] == 1.0).all()
        assert info_w["n_rounds"].shape == (T,) and info_w["n_rounds"].dtype == torch.int64
        assert (info_w["n_rounds"] == 1).all()
        assert set(info_w) == {"z", "z_hat", "z_target", "eps", "u", *AUX_KEYS, "cost", "n_rounds"}
        assert all(t.device.type == device.type for t in info_w.values())

    # P1.5 scope and input validation: errors over silence
    cfg, model = _model(TINY)
    B = 2
    state = model.init_state(B)
    tokens, ctx = _ids(cfg, B, 3, seed=11)
    x, c = tokens[:, 0], ctx[:, 0]
    with pytest.raises(ValueError, match="model.routing=dense"):  # P2.5: needs a selection mode
        model.step(x, c, state, mode="sparse")
    with pytest.raises(ValueError, match="mode"):
        model.step(x, c, state, mode="soft")  # type: ignore[arg-type]
    # the "remove shared path" ablation builds, with the 3d cell input (test_ablate_shared_u.py)
    ablated = HSN.from_config(load_config(TINY, overrides=["model.ablate_shared_u=true"]))
    assert ablated.cell.in_dim == 3 * cfg.model.d == ablated.cell.gate.in_features
    assert model.cell.in_dim == 4 * cfg.model.d and not model.cell.ablate_shared_u
    with pytest.raises(ValueError, match="scorer_features.exclude"):
        HSN.from_config(load_config(TINY, overrides=["model.scorer_features.exclude=[eps, nope]"]))
    with pytest.raises(TypeError):
        HSN(cfg.model)  # type: ignore[call-arg]  # seed and unroll are required
    with pytest.raises(ValueError, match="unroll"):
        HSN(cfg.model, seed=0, unroll=0)
    with pytest.raises(ValueError, match="seed"):
        HSN(cfg.model, seed=-1, unroll=8)
    with pytest.raises(TypeError, match="RunConfig"):
        HSN.from_config(cfg.model)  # type: ignore[arg-type]
    out = model.step(x, c, state)
    scorer = model.scorer
    good = dict(budget_frac=1.0, round_idx=0)
    with pytest.raises(ValueError, match="round_idx"):
        scorer.features(out.u, out.eps, state, c, budget_frac=1.0, round_idx=1)
    with pytest.raises(ValueError, match="round_idx"):
        scorer.features(out.u, out.eps, state, c, budget_frac=1.0, round_idx=-1)
    with pytest.raises(TypeError, match="round_idx"):
        scorer.features(out.u, out.eps, state, c, budget_frac=1.0, round_idx=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget_frac"):
        scorer.features(out.u, out.eps, state, c, budget_frac=-0.1, round_idx=0)
    with pytest.raises(ValueError, match="budget_frac"):
        scorer.features(out.u, out.eps, state, c, budget_frac=math.inf, round_idx=0)
    with pytest.raises(ValueError, match=r"u must have shape"):
        scorer.features(out.u[:1], out.eps, state, c, **good)
    with pytest.raises(ValueError, match="eps must have dtype"):
        scorer.features(out.u, out.eps.float(), state, c, **good)
    with pytest.raises(TypeError, match="ctx"):
        scorer.features(out.u, out.eps, state, c.float(), **good)
    with pytest.raises(ValueError, match="ctx must have shape"):
        scorer.features(out.u, out.eps, state, c[:1], **good)
    bad_ema = EMAFeatures(utility=torch.zeros(B + 1, cfg.model.n_cells), selected=torch.zeros(3))
    with pytest.raises(ValueError, match="EMAFeatures.utility"):
        scorer.features(out.u, out.eps, state, c, ema=bad_ema, **good)
    with pytest.raises(ValueError, match="EMAFeatures.selected"):
        scorer.features(
            out.u, out.eps, state, c, ema=EMAFeatures(torch.zeros(3), torch.zeros(4)), **good
        )
    with pytest.raises(ValueError, match="EMAFeatures.utility"):
        scorer.features(
            out.u, out.eps, state, c, ema=EMAFeatures(torch.zeros(3).long(), torch.zeros(3)), **good
        )
    with pytest.raises(ValueError, match=r"\[B, N, 20\]"):
        scorer(torch.zeros(B, cfg.model.n_cells, 19, dtype=torch.float64))
    with pytest.raises(ValueError, match="alpha"):
        model.cell.step_dense(out.u, out.eps, state, torch.zeros(B, 3, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="alpha must have dtype"):
        model.cell.step_dense(out.u, out.eps, state, torch.zeros(B, 3, 2))
    with pytest.raises(ValueError, match="state has N"):
        model.cell.step_dense(
            out.u,
            out.eps,
            CellGraphState.zeros(B, 4, 4, "cpu", torch.float64),
            torch.zeros(B, 4, 2),
        )
    with pytest.raises(ValueError, match="x_t must have shape"):
        model.step(x[:1], c, state)
    with pytest.raises(TypeError, match="x_t"):
        model.step(x.float(), c, state)
    with pytest.raises(ValueError, match="ctx_t must have shape"):
        model.step(x, c[:, None], state)
    with pytest.raises(ValueError, match="state has dtype"):
        model.step(x, c, CellGraphState.zeros(B, 3, 4, "cpu", torch.float32))
    with pytest.raises(ValueError, match="state has N"):
        model.step(x, c, CellGraphState.zeros(B, 2, 4, "cpu", torch.float64))
    with pytest.raises(TypeError, match="CellGraphState"):
        model.step(x, c, (state.h, state.q))  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        model.step(x + cfg.model.vocab_size, c, state)  # an id outside the table is never re-read
    reset = torch.zeros(B, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="tokens must be"):
        model(tokens[:, 0], ctx, state, reset)
    with pytest.raises(ValueError, match="at least one position"):
        model(tokens[:, :0], ctx[:, :0], state, reset[:, :0])
    with pytest.raises(ValueError, match="ctx must have shape"):
        model(tokens, ctx[:, :2], state, reset)
    with pytest.raises(ValueError, match="reset must have dtype"):
        model(tokens, ctx, state, reset.long())
    with pytest.raises(ValueError, match="reset must have shape"):
        model(tokens, ctx, state, reset[:, :2])
    with pytest.raises(TypeError, match="reset"):
        model(tokens, ctx, state, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="q must have shape"):
        HaltingHead(cfg.model)(state.q[:, :2], out.u)
    with pytest.raises(TypeError):
        Scorer(cfg.model)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unroll"):
        Scorer(cfg.model, unroll=True)


def test_p1_5_matches_pure_python_reference_tiny_fp64() -> None:
    variants: list[list[str]] = [
        [],
        ["model.scorer=mlp"],
        ["model.scorer_features.exclude=[eps, pos, one]"],
        ["model.edge_norm=sigmoid"],
        ["model.disabled_cells=[1]"],
        ["model.in_degree=0"],
    ]
    for overrides in variants:
        cfg, model = _model(TINY, overrides)
        assert cfg.model.activation == "gelu" and model.dtype is torch.float64
        _perturb(model, seed=12)
        B, T = 2, 3
        state = _random_state(model, B, seed=13)
        assert (state.age > 0).any() and (state.pos == 2).all()
        tokens, ctx = _ids(cfg, B, T, seed=14)

        # position 0 from the random state, position 1 from the carried state
        out = _check_position(cfg, model, state, tokens[:, 0], ctx[:, 0])
        out = _check_position(cfg, model, out.state, tokens[:, 1], ctx[:, 1])
        # a partial reset before position 2 (row 1 restarts from zeros)
        reset_row = torch.tensor([False, True])
        out = _check_position(cfg, model, out.state.reset(reset_row), tokens[:, 2], ctx[:, 2])

        # the window loop reproduces the same positions (reset applied before the position)
        reset = torch.zeros(B, T, dtype=torch.bool)
        reset[1, 2] = True
        logits, final, info = model(tokens, ctx, state, reset)
        _close_window(logits[:, 2], out.logits)
        _close_window(final.h, out.state.h)
        _close_window(final.q, out.state.q)
        assert torch.equal(final.age, out.state.age) and torch.equal(final.pos, out.state.pos)
        _close_window(info["u"][:, 2], out.u)

    # the feature builder alone, with the P2.3 averages, a spent budget and the age/pos counters
    cfg, model = _model(TINY)
    _perturb(model, seed=15)
    dims = _dims(cfg, model)
    B = 2
    state = _random_state(model, B, seed=16, ticks=5)
    tokens, ctx = _ids(cfg, B, 1, seed=17)
    out = model.step(tokens[:, 0], ctx[:, 0], state)
    generator = torch.Generator(device="cpu").manual_seed(18)
    utility = torch.randn(B, cfg.model.n_cells, generator=generator, dtype=torch.float64)
    selected = torch.rand(cfg.model.n_cells, generator=generator, dtype=torch.float64)
    feats = model.scorer.features(
        out.u,
        out.eps,
        state,
        ctx[:, 0],
        budget_frac=0.37,
        round_idx=0,
        ema=EMAFeatures(utility=utility, selected=selected),
    )
    for b in range(B):
        ref = _ref_features(
            dims,
            out.u[b].tolist(),
            out.eps[b].tolist(),
            state.h[b].tolist(),
            state.age[b].tolist(),
            int(state.pos[b]),
            int(ctx[b, 0]),
            0.37,
            0,
            utility[b].tolist(),
            selected.tolist(),
        )
        _close(feats[b], ref)
    # the layout, slot by slot
    s = model.scorer.feature_slices
    d = cfg.model.d
    assert [s[name] for name in SCORER_FEATURE_NAMES] == [
        slice(0, d),
        slice(d, 2 * d),
        slice(2 * d, 3 * d),
        *(slice(3 * d + k, 3 * d + k + 1) for k in range(SCORER_SCALAR_FEATURES)),
    ]
    assert torch.equal(feats[..., s["u"]], out.u[:, None].expand(B, 3, d))
    assert torch.equal(feats[..., s["h"]], state.h)
    assert torch.equal(feats[..., s["age"]].squeeze(-1), state.age.double() / cfg.model.age_max)
    assert (feats[..., s["budget"]] == 0.37).all() and (feats[..., s["round"]] == 0).all()
    assert (feats[..., s["one"]] == 1).all()
    assert torch.equal(feats[..., s["ema_utility"]].squeeze(-1), utility)
    assert torch.equal(feats[..., s["ema_selected"]].squeeze(-1), selected[None].expand(B, 3))
    assert torch.equal(
        feats[..., s["pos"]].squeeze(-1),
        (state.pos.double() / cfg.train.unroll)[:, None].expand(B, 3),
    )
    # exclusion zeroes blocks without changing the width or the count
    _, excluded = _model(TINY, ["model.scorer_features.exclude=[u, age, one, u]"])
    assert excluded.scorer.exclude == ("u", "age", "one")
    assert excluded.scorer.in_features == model.scorer.in_features
    assert excluded.param_report() == model.param_report()
    ex_feats = excluded.scorer.features(
        out.u, out.eps, state, ctx[:, 0], budget_frac=1.0, round_idx=0
    )
    assert (ex_feats[..., s["u"]] == 0).all() and (ex_feats[..., s["age"]] == 0).all()
    assert (ex_feats[..., s["one"]] == 0).all() and torch.equal(ex_feats[..., s["h"]], state.h)


# --------------------------------------------------------------------------------------------------
# Additional coverage
# --------------------------------------------------------------------------------------------------


def test_p1_5_reset_restarts_state() -> None:
    for path in (TINY, REF_A1):
        cfg, model = _model(path)
        _perturb(model, seed=19)
        B, T = 3, 4
        tokens, ctx = _ids(cfg, B, T, seed=20)
        dirty = _random_state(model, B, seed=21)
        fresh = model.init_state(B)

        # a reset at position 0 for every session equals a fresh state, bit for bit
        reset_all = torch.zeros(B, T, dtype=torch.bool)
        reset_all[:, 0] = True
        logits_reset, state_reset, _ = model(tokens, ctx, dirty, reset_all)
        logits_fresh, state_fresh, _ = model(tokens, ctx, fresh, torch.zeros_like(reset_all))
        assert torch.equal(logits_reset, logits_fresh)
        assert torch.equal(state_reset.h, state_fresh.h) and torch.equal(
            state_reset.q, state_fresh.q
        )
        assert torch.equal(state_reset.pos, state_fresh.pos)
        # ... and a dirty state gives different outputs
        logits_dirty, _, _ = model(tokens, ctx, dirty, torch.zeros_like(reset_all))
        assert not torch.equal(logits_dirty, logits_fresh)

        # a reset of one session at position 2 restarts that session only: from there on its
        # outputs equal a fresh session fed the same inputs; the other sessions are unaffected
        reset_one = torch.zeros(B, T, dtype=torch.bool)
        reset_one[1, 2] = True
        logits_one, state_one, _ = model(tokens, ctx, dirty, reset_one)
        assert torch.equal(logits_one[[0, 2]], logits_dirty[[0, 2]])
        assert torch.equal(logits_one[1, :2], logits_dirty[1, :2])
        # (P1.10a: forward runs E and D over [B, T], so windows of 4 and 2 positions round their
        # GEMMs differently: equal to the window tolerance, not bit for bit, and compared in fp64
        # only. The perturbed ref model in fp32 moves these values by up to 2.3e-3 (logits up to
        # 2480) under another CPU's one-ulp differences, measured 6 Nov 2026; its fp32 run keeps
        # the exact reset properties above)
        logits_tail, state_tail, _ = model(
            tokens[:, 2:], ctx[:, 2:], model.init_state(B), torch.zeros(B, 2, dtype=torch.bool)
        )
        if model.dtype == torch.float64:
            _close_window(logits_one[1, 2:], logits_tail[1])
            _close_window(state_one.h[1], state_tail.h[1])
            _close_window(state_one.q[1], state_tail.q[1])
        assert int(state_one.pos[1]) == 2 and (state_one.pos[[0, 2]] == dirty.pos[[0, 2]] + T).all()


def test_p1_5_grads_reach_every_parameter() -> None:
    cfg, model = _model(TINY)
    _perturb(model, seed=22)
    B, T = 2, 3
    tokens, ctx = _ids(cfg, B, T, seed=23)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 1] = True
    state = _random_state(model, B, seed=24)
    logits, final, info = model(tokens, ctx, state, reset)
    loss = logits.logsumexp(-1).sum() + info["eps"].square().sum() + final.q.sum()
    model.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    for name, p in model.named_parameters():
        if name.startswith("halting."):
            assert p.grad is None, name  # defined, unused while r_max = 1
            continue
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum().item() > 0, name
    # the state carried out of the window still backpropagates into the window's inputs
    assert final.h.requires_grad and final.q.grad_fn is not None
    carried = final.detach()
    assert not carried.h.requires_grad

    # finite differences on one step in fp64: the step is a smooth function of the incoming
    # state (the P1.8 suite does the full window and every parameter)
    h0 = _random_state(model, B, seed=25).h.detach().clone().requires_grad_(True)

    def step_fn(h: Tensor) -> tuple[Tensor, Tensor]:
        out = model.step(tokens[:, 0], ctx[:, 0], model.init_state(B).with_dense(h))
        return out.logits, out.state.h

    assert torch.autograd.gradcheck(step_fn, (h0,), atol=1e-6)


def test_p1_5_state_dict_round_trip_and_repr() -> None:
    cfg, source = _model(TINY)
    _perturb(source, seed=26)
    torch.manual_seed(99)
    target = HSN.from_config(cfg).double()
    B = 2
    tokens, ctx = _ids(cfg, B, 2, seed=27)
    state = _random_state(source, B, seed=28)
    assert not torch.equal(
        source.step(*_first(tokens, ctx), state).logits,
        target.step(*_first(tokens, ctx), state).logits,
    )
    target.load_state_dict(source.state_dict())
    assert torch.equal(
        source.step(*_first(tokens, ctx), state).logits,
        target.step(*_first(tokens, ctx), state).logits,
    )
    assert torch.equal(source.cell.graph.indices, target.cell.graph.indices)
    # the seed and the unroll are in the repr for logs and manifests
    text = repr(source)
    assert "seed=0" in text and f"unroll={cfg.train.unroll}" in text
    assert "kind='linear'" in text and "in_features=20" in text
    assert "disabled_cells=[]" in text


def _first(tokens: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
    return tokens[:, 0], ctx[:, 0]


def _to_device(state: CellGraphState, device: torch.device) -> CellGraphState:
    return CellGraphState(
        h=state.h.to(device),
        q=state.q.to(device),
        age=state.age.to(device),
        pos=state.pos.to(device),
    )


@pytest.mark.parametrize("device_type", [cuda_param, mps_param])
def test_p1_5_runs_on_device(device_type: str) -> None:
    """The forward on the device agrees with the CPU.

    Two tolerances (P1.7, measured on MPS with the weights perturbed to std 0.3): every single
    position, started from the *same* input state on both sides, must agree to 1e-3 (the device's
    one-step numerics: MPS gives 3.2e-4 absolute on 3 of 1,024 logits of a 512-wide decoder,
    fp32 accumulation order; a broken kernel shows here as an order-one error). Over the whole
    window the state is carried on each side and that rounding compounds through six gated
    recurrent steps: MPS reached 0.0145 absolute on the logits, so the carried comparison allows
    5e-2, which is still far below what a wrong operation produces.

    The new state ``h`` / ``q`` is checked against an fp64 reference instead of against the CPU:
    the device must be no more than 4x as far from it as the CPU's own fp32 result (plus 1e-6).
    ``h`` is bounded by 1 but gated from pre-activations in the hundreds under these perturbed
    weights (``u`` reaches ~160), so ordinary fp32 rounding is ~1e-3 absolute there on *both*
    backends, often on opposite sides of the exact value. Measured on CUDA (RTX 3060, torch
    2.15): CPU and CUDA each 6e-4 from fp64 on one ``h`` element, 1.22e-3 apart; the device/CPU
    error ratio was at most 1.73 on these seeds and up to 2.97 on five other seed triples. At the
    unperturbed initialisation every output is within 1e-7 of fp64 on the CPU.
    """
    device = _device(device_type)
    cfg, model = _model(REF_A1)
    _perturb(model, seed=29)
    B, T = cfg.train.batch, 6
    tokens, ctx = _ids(cfg, B, T, seed=30)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 3] = True
    state = _random_state(model, B, seed=31)
    logits_cpu, state_cpu, info_cpu = model(tokens, ctx, state, reset)
    cpu_model = copy.deepcopy(model)
    exact_model = copy.deepcopy(model).double()  # fp64 reference for the new state

    on_device = model.to(device)
    state_dev = _to_device(state, device)
    logits_dev, state_out, info_dev = on_device(
        tokens.to(device), ctx.to(device), state_dev, reset.to(device)
    )
    assert logits_dev.device.type == device.type and state_out.device.type == device.type
    # one step at a time from the same input state: the device's own numerics
    s_cpu = state
    with torch.no_grad():
        for t in range(T):
            s_cpu = s_cpu.reset(reset[:, t])
            out_cpu = cpu_model.step(tokens[:, t], ctx[:, t], s_cpu)
            out_dev = on_device.step(
                tokens[:, t].to(device), ctx[:, t].to(device), _to_device(s_cpu, device)
            )
            for name in ("z", "z_hat", "eps", "u", "logits"):
                torch.testing.assert_close(
                    getattr(out_dev, name).cpu(),
                    getattr(out_cpu, name),
                    atol=1e-3,
                    rtol=1e-3,
                    msg=lambda m, t=t, name=name: f"position {t}, {name}: {m}",
                )
            # the new state: at least as accurate as the CPU, measured against fp64
            s_exact = CellGraphState(
                h=s_cpu.h.double(), q=s_cpu.q.double(), age=s_cpu.age, pos=s_cpu.pos
            )
            out_exact = exact_model.step(tokens[:, t], ctx[:, t], s_exact)
            for name in ("h", "q"):
                exact = getattr(out_exact.state, name)
                err_cpu = (getattr(out_cpu.state, name).double() - exact).abs().max().item()
                err_dev = (getattr(out_dev.state, name).cpu().double() - exact).abs().max().item()
                assert err_dev <= 4 * err_cpu + 1e-6, (
                    f"position {t}, state.{name}: {device} is {err_dev:.2e} from fp64, "
                    f"the CPU {err_cpu:.2e}"
                )
            s_cpu = out_cpu.state
    # the whole window with the state carried on each side: accumulated fp32 rounding only
    torch.testing.assert_close(logits_dev.cpu(), logits_cpu, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(state_out.h.cpu(), state_cpu.h, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(state_out.q.cpu(), state_cpu.q, atol=5e-2, rtol=5e-2)
    assert torch.equal(state_out.age.cpu(), state_cpu.age)
    assert torch.equal(state_out.pos.cpu(), state_cpu.pos)
    for key in ("z", "z_hat", "eps", "u", *AUX_KEYS):
        torch.testing.assert_close(info_dev[key].cpu(), info_cpu[key], atol=5e-2, rtol=5e-2)
    assert on_device.param_report()["total"] == 1_820_617
    # backward on the device reaches the edge logits and the gate
    logits_dev.square().mean().backward()  # type: ignore[no-untyped-call]
    assert on_device.cell.graph.edge_logit.grad is not None
    assert on_device.cell.gate.weight.grad is not None
    assert torch.isfinite(on_device.cell.gate.weight.grad).all()
