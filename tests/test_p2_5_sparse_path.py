"""P2.5 tests: the four native kernel ops, ``PDGCell.step_sparse``, the mode wiring of ``HSN.step``
(sparse equals dense when every cell is selected, unselected cells untouched, incremental ``q``,
budgets, padded slots, every backend, deterministic mode, autocast, profiler segments) and the
backend switch of ``hsn.kernels``."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from hsn import kernels
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything, set_deterministic
from hsn.kernels.costs import CostTable, profile_dense_costs, write_cost_profile
from hsn.kernels.native import (
    csr_messages,
    gated_delta_update,
    gather_concat,
    grouped_unit_mlp,
    selection_mask,
)
from hsn.model.hsn import HSN, StepOutput
from hsn.model.router import BUDGET_SLACK, Selection
from hsn.model.state import CellGraphState
from hsn.profiler import Profiler
from hsn.train.tbptt import TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
DEVICES = ["cpu", cuda_param, mps_param]
SPARSE_MODES = ("topk", "greedy_budget", "random", "fixed")
#: The chunk's fp32 agreement tolerance (P2.9 uses the same for the native path vs the oracle).
ATOL = 1e-5


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    return torch.empty(0, device=device_type).device  # with the index, as tensors report it


def _cpu_ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _model(
    path: Path,
    overrides: Sequence[str] = (),
    *,
    perturb: int | None = None,
    scale: float = 0.3,
    seed: int = 0,
) -> tuple[RunConfig, HSN]:
    """A model from a config file (fp64 when it says so), weights seeded by ``seed``; ``perturb``
    adds seeded noise of std ``scale`` to every parameter (the P1.5 recipe at 0.3: states of order
    0.1 rather than 1e-5, so wrong operands show)."""
    # P2.2: the default gate (soft) relaxes every cell in training mode; this chunk's tests are
    # about hard decisions, so they train with the straight-through gate (hard forward values)
    cfg = load_config(path, overrides=["train.gate_mode=ste", *overrides])
    torch.manual_seed(seed)
    model = HSN.from_config(cfg)
    if cfg.train.precision == "fp64":
        model = model.double()
    if perturb is not None:
        generator = torch.Generator(device="cpu").manual_seed(perturb)
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
                p.add_(noise.to(dtype=p.dtype, device=p.device))
    return cfg, model


def _ids(cfg: RunConfig, B: int, T: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m = cfg.model
    tokens = torch.randint(0, m.vocab_size, (B, T), generator=generator)
    ctx = torch.randint(0, m.n_context, (B, T), generator=generator)
    return tokens, ctx


def _random_state(model: HSN, B: int, seed: int, ticks: int = 2) -> CellGraphState:
    """A state with non-trivial ``h`` (|h| < 1), an exact ``q``, mixed ``age`` and ``pos``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    N, d = model.cfg.n_cells, model.cfg.d
    h = torch.rand(B, N, d, generator=generator, dtype=torch.float64) * 1.6 - 0.8
    state = model.init_state(B).with_dense(h.to(dtype=model.dtype, device=model.device))
    for _ in range(ticks):
        mask = torch.rand(B, N, generator=generator) < 0.5
        state = state.tick(mask.to(model.device))
    return state


def _to_device(state: CellGraphState, device: torch.device) -> CellGraphState:
    return CellGraphState(
        h=state.h.to(device),
        q=state.q.to(device),
        age=state.age.to(device),
        pos=state.pos.to(device),
    )


def _prefix(
    model: HSN, x: Tensor, c: Tensor, state: CellGraphState
) -> tuple[Tensor, Tensor, Tensor]:
    """``(u, eps, unit_logits)`` of one position in the state dtype, as ``HSN.step`` computes them
    before the cell round."""
    q_prev = state.q
    z_hat = model.predictor(q_prev)
    z = model.encoder(model.byte_emb(x))
    eps = z - z_hat
    u = model.shared(torch.cat([z, q_prev, model.ctx_emb(c)], dim=-1))
    u, eps = u.to(state.dtype), eps.to(state.dtype)
    feats = model.scorer.features(u, eps, state, c, budget_frac=model.cfg.budget, round_idx=0)
    return u, eps, model.router(feats).to(state.dtype)


def _selection(
    cell_idx: Tensor,
    unit_idx: Tensor,
    valid: Tensor,
    logits: Tensor,
    *,
    alpha: Tensor | None = None,
    cost: Tensor | None = None,
) -> Selection:
    """A hand-built P2.1 ``Selection`` (hard α = 1 on valid slots unless given); the P2.2 fields
    (P2.2 landed beside this chunk) hold the hard, flag-free values of the slots."""
    device = logits.device
    B, N, M = (int(s) for s in logits.shape)
    K = int(cell_idx.shape[1])
    cells = cell_idx.to(device=device, dtype=torch.int64)
    units = unit_idx.to(device=device, dtype=torch.int64)
    valid = valid.to(device)
    cost_t = (torch.zeros(B) if cost is None else cost).to(device)
    # padded slots may hold anything (a sentinel >= N): only the valid slots build the masks
    safe_cells = torch.where(valid, cells, torch.zeros_like(cells))
    safe_units = torch.where(valid, units, torch.zeros_like(units))
    cell_mask = torch.zeros(B, N, dtype=torch.float32, device=device).scatter_add(
        1, safe_cells, valid.to(torch.float32)
    )
    unit_weights = torch.zeros(B, N, M, dtype=torch.float32, device=device)
    rows = torch.arange(B, device=device)[:, None].expand(B, K)
    unit_weights[rows[valid], safe_cells[valid], safe_units[valid]] = 1.0
    return Selection(
        cell_idx=cells,
        unit_idx=units,
        alpha=(valid.to(torch.float32) if alpha is None else alpha).to(device),
        valid=valid,
        logits=logits,
        cost=cost_t,
        scorer_cost=0.0,
        n_rounds_used=1,
        cell_mask=cell_mask,
        unit_weights=unit_weights,
        expected_cost=cost_t.clone(),
        log_prob=torch.zeros(B, dtype=torch.float32, device=device),
        explored=torch.zeros(B, K, dtype=torch.bool, device=device),
        refreshed=torch.zeros(B, K, dtype=torch.bool, device=device),
        dense_fallback=torch.zeros(B, dtype=torch.bool, device=device),
    )


def _one_hot_alpha(
    cell_idx: Tensor, unit_idx: Tensor, N: int, M: int, dtype: torch.dtype
) -> Tensor:
    """``alpha [B, N, M]`` with a 1 at ``(cell_idx[b, k], unit_idx[b, k])`` for every slot."""
    B, K = cell_idx.shape
    alpha = torch.zeros(B, N, M, dtype=dtype, device=cell_idx.device)
    rows = torch.arange(B, device=cell_idx.device)[:, None].expand(B, K)
    alpha[rows.reshape(-1), cell_idx.reshape(-1), unit_idx.reshape(-1)] = 1.0
    return alpha


def _grad_sum(model: HSN) -> dict[str, float]:
    cell = model.cell
    named = {
        "V": cell.V.weight,
        "W_out": cell.W_out.weight,
        "gate": cell.gate.weight,
        "W1": cell.units.W1,
        "W2": cell.units.W2,
        "edge_logit": cell.graph.edge_logit,
        "decoder": model.decoder.fc1.weight,
        "encoder": model.encoder.fc1.weight,
    }
    return {
        k: float(p.grad.abs().sum()) if p.grad is not None else math.nan for k, p in named.items()
    }


# --------------------------------------------------------------------------------------------------
# Fixtures for the training tests (the P1.7 recipe: tiny kv in-range documents)
# --------------------------------------------------------------------------------------------------


def _tiny_task() -> TaskSpec:
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 64,
            "n_dev": 16,
            "n_test": 16,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


@pytest.fixture(scope="module")
def train_docs() -> list[Example]:
    task = _tiny_task()
    return [generate(task, "train", i) for i in range(64)]  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def dev_docs() -> list[Example]:
    task = _tiny_task()
    return [generate(task, "dev", i) for i in range(16)]  # type: ignore[arg-type]


def _train_config(*overrides: str) -> RunConfig:
    return load_config(
        REF_A1,
        overrides=[
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=4",
            "model.in_degree=2",
            "model.n_units=2",
            "model.k_select=2",
            "model.hidden=16",
            "model.scorer_hidden=8",
            "model.routing=topk",
            "train.gate_mode=ste",  # P2.2: hard forward decisions with the straight-through gate
            f"train.unroll={UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=100",
            "train.eval_every=50",
            "train.ckpt_every=100",
            "train.stats_every=1",
            *overrides,
        ],
    )


# --------------------------------------------------------------------------------------------------
# The tests named by the chunk
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", DEVICES)
def test_p2_5_sparse_equals_dense_when_all_selected(device_type: str) -> None:
    """K = N with one-hot α: the sparse round equals the dense one to the chunk's 1e-5 in fp32 —
    at the cell level against ``step_dense`` with the same one-hot α (the cells in a random
    order), at the kernel level against ``CellGraphState.with_delta`` (bit for bit) and
    ``with_dense``'s exact mean, and at the model level (``routing=topk``, ``k_select = N``,
    budget 1.0) against a dense reference assembled from the same argmax units.

    The weights are perturbed at std 0.1, not the P1.5 recipe's 0.3: the messages, inputs and gate
    are bit-identical at any scale, but 0.3 drives the unit outputs to magnitude ~1.5e3, where the
    fp32 accumulation order of the einsum against the grouped ``addmm`` alone is 3e-4 absolute
    (2e-7 relative); at 0.1 the unit outputs are of order 3 (the training regime) and the round
    agrees to 8e-7 on the cpu.
    """
    device = _device(device_type)
    cfg, model = _model(REF_A1, ["model.routing=topk", "model.k_select=32"], perturb=29, scale=0.1)
    model = model.to(device)
    cell = model.cell
    B, N, M, d = 4, cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    state = _random_state(model, B, seed=31)
    tokens, ctx = _ids(cfg, B, 1, seed=30)
    x, c = tokens[:, 0].to(device), ctx[:, 0].to(device)
    with torch.no_grad():
        u, eps, logits = _prefix(model, x, c, state)
        # every cell, in a random per-row order, with its argmax unit
        generator = torch.Generator(device="cpu").manual_seed(5)
        perm = torch.argsort(torch.rand(B, N, generator=generator), dim=1).to(device)
        units = logits.argmax(dim=-1).gather(1, perm)
        valid = torch.ones(B, N, dtype=torch.bool, device=device)
        sel = _selection(perm, units, valid, logits)
        alpha = _one_hot_alpha(perm, units, N, M, state.dtype)

        h_dense, info_d = cell.step_dense(u, eps, state, alpha)
        h_sparse, info_s = cell.step_sparse(u, eps, state, sel)
        index = perm.unsqueeze(-1).expand(B, N, d)
        torch.testing.assert_close(h_sparse, h_dense.gather(1, index), atol=ATOL, rtol=0)
        written = info_s["state"]
        assert isinstance(written, CellGraphState)
        torch.testing.assert_close(written.h, h_dense, atol=ATOL, rtol=0)
        torch.testing.assert_close(written.q, h_dense.mean(dim=1), atol=ATOL, rtol=0)
        assert bool((written.age == 0).all()) and torch.equal(written.pos, state.pos)
        selected = info_s["selected"]
        assert isinstance(selected, Tensor) and bool(selected.all())
        for name in ("m", "v", "c", "b"):
            dense_t, sparse_t = info_d[name], info_s[name]
            assert isinstance(sparse_t, Tensor)
            gathered = dense_t.gather(1, perm.unsqueeze(-1).expand(B, N, dense_t.shape[-1]))
            torch.testing.assert_close(sparse_t, gathered, atol=ATOL, rtol=0, msg=name)
        # the kernel against P1.3's functional write of the same values
        c_t, b_t = info_s["c"], info_s["b"]
        assert isinstance(c_t, Tensor) and isinstance(b_t, Tensor)
        direct = gated_delta_update(state, perm, c_t, b_t, valid)
        h_old = state.h.gather(1, index)
        via_state = state.with_delta(perm, (1 - b_t) * h_old + b_t * c_t)
        assert torch.equal(direct.h, via_state.h) and torch.equal(direct.age, via_state.age)
        torch.testing.assert_close(direct.q, via_state.q, atol=1e-6, rtol=0)
        torch.testing.assert_close(direct.q, state.with_dense(direct.h).q, atol=ATOL, rtol=0)

        # the model: topk with k_select = N and budget 1.0 selects every cell
        out = model.step(x, c, state)
        assert out.selection is not None and bool(out.selection.valid.all())
        assert out.selection.k == N and torch.equal(out.state.age, torch.zeros_like(state.age))
        alpha_ref = _one_hot_alpha(
            out.selection.cell_idx, out.selection.unit_idx, N, M, state.dtype
        )
        h_ref, _ = cell.step_dense(u, eps, state, alpha_ref)
        ref_state = model._write_dense(state, h_ref)
        logits_ref = model.decoder(torch.cat([ref_state.q, u], dim=-1))
        torch.testing.assert_close(out.state.h, ref_state.h, atol=ATOL, rtol=0)
        torch.testing.assert_close(out.state.q, ref_state.q, atol=ATOL, rtol=0)
        torch.testing.assert_close(out.logits, logits_ref, atol=ATOL, rtol=ATOL)
        assert torch.equal(out.state.pos, state.pos + 1)
        # every cell counts, so the executed cost is the whole table (1/M of the dense step under
        # the uniform placeholder: N cells at 1/(N·M) each, no shared or per-cell cost)
        assert float(out.cost) == pytest.approx(1.0 / M, rel=1e-6)


def test_p2_5_unselected_cells_unchanged() -> None:
    """K < N: the cells outside the selection keep ``h`` bit for bit and age by one, the selected
    cells are written and reset to age 0, and ``q`` moves by the selected deltas only."""
    cfg, model = _model(REF_A1, ["model.routing=topk"], perturb=3)
    B, N = 4, cfg.model.n_cells
    state = _random_state(model, B, seed=7, ticks=3)
    tokens, ctx = _ids(cfg, B, 1, seed=8)
    out = model.step(tokens[:, 0], ctx[:, 0], state)
    sel = out.selection
    assert sel is not None and sel.k == cfg.model.k_select == 4
    mask = selection_mask(sel.cell_idx, sel.valid, N)
    assert torch.equal(mask.sum(dim=1), sel.n_valid()) and bool((sel.n_valid() == 4).all())
    assert torch.equal(out.state.h[~mask], state.h[~mask])
    changed = (out.state.h != state.h).any(dim=-1)
    assert bool((changed <= mask).all()) and bool(changed.any())
    assert torch.equal(out.state.age[~mask], state.age[~mask] + 1)
    assert bool((out.state.age[mask] == 0).all())
    assert torch.equal(out.state.pos, state.pos + 1)
    delta = (out.state.h - state.h).sum(dim=1) / N
    torch.testing.assert_close(out.state.q, state.q + delta, atol=1e-6, rtol=0)
    assert out.n_rounds == 1 and out.cost.shape == () and not out.cost.requires_grad
    assert float(out.cost) == pytest.approx(4.0 / (N * cfg.model.n_units))
    assert out.logits.requires_grad and out.state.h.requires_grad and out.state.q.requires_grad


def test_p2_5_incremental_q_consistent() -> None:
    """1,024 sparse positions over four carried windows with document resets and no resync: the
    incremental ``q`` stays within 1e-5 of the exact mean (P1.8's bound; P1.3 measured 2e-6
    after 10,000 rounds)."""
    cfg, model = _model(REF_A1, ["model.routing=topk"], perturb=11)
    B, T = cfg.train.batch, 256
    state = model.init_state(B)
    generator = torch.Generator(device="cpu").manual_seed(12)
    worst = 0.0
    with torch.no_grad():
        for window in range(4):
            tokens, ctx = _ids(cfg, B, T, seed=100 + window)
            reset = torch.rand(B, T, generator=generator) < 0.01
            _, state, info = model(tokens, ctx, state, reset)
            assert info["valid"].shape == (B, T, cfg.model.k_select)
            state = state.detach()
            worst = max(worst, float((state.q - state.recompute_q()).abs().max()))
    assert 0.0 < worst <= 1e-5, worst
    assert bool((state.pos <= 4 * T).all()) and bool((state.pos > 0).any())  # resets zero pos


def test_p2_5_cost_within_budget(tmp_path: Path) -> None:
    """The executed cost of every position respects ``model.budget`` with a profiled table (from
    ``profile_dense_costs``) and with the uniform placeholder; ``StepOutput.cost`` is the batch mean
    of ``Selection.cost``; exactly the valid slots are written; an unaffordable ``k_min`` is refused
    when the model is built."""
    _, dense = _model(TINY)
    profile = write_cost_profile(
        profile_dense_costs(dense, _cpu_ctx(), batch=2, positions=3), tmp_path / "tiny_costs.json"
    )
    table = CostTable.from_profile(profile, dense.cfg.n_units)
    assert table.shared > 0.0 and table.cell_overhead > 0.0
    budget = table.shared + 1.5 * table.max_pair_cost()  # one cell fits, a second never does
    cfg, model = _model(
        TINY,
        [
            f"model.cost_profile={profile}",
            "model.routing=topk",
            "model.k_select=3",
            "model.k_min=1",
            f"model.budget={budget}",
        ],
    )
    assert model.router.cost_source == f"profile:{profile}"
    B = 2
    state = model.init_state(B)
    tokens, ctx = _ids(cfg, B, 3, seed=1)
    for t in range(3):
        out = model.step(tokens[:, t], ctx[:, t], state)
        sel = out.selection
        assert sel is not None
        assert bool((sel.n_valid() == 1).all()) and sel.k == 3
        assert bool((sel.cost <= budget * (1.0 + BUDGET_SLACK)).all())
        assert float(out.cost) == pytest.approx(float(sel.cost.mean()))
        assert torch.equal((out.state.age == 0).sum(dim=1), sel.n_valid())
        state = out.state
    with pytest.raises(ValueError, match="cannot hold"):
        _model(TINY, [f"model.cost_profile={profile}", "model.routing=topk", "model.budget=0.01"])

    # the uniform placeholder on ref_a1: units cost 1/(N·M) each, so a budget of 0.01 admits two
    cfg, model = _model(REF_A1, ["model.routing=topk", "model.budget=0.01"])
    N, M = cfg.model.n_cells, cfg.model.n_units
    assert 2 / (N * M) <= 0.01 < 3 / (N * M)
    state = _random_state(model, 3, seed=2)
    tokens, ctx = _ids(cfg, 3, 1, seed=3)
    out = model.step(tokens[:, 0], ctx[:, 0], state)
    sel = out.selection
    assert sel is not None and bool((sel.n_valid() == 2).all())
    assert bool((~sel.valid[:, 2:]).all()) and bool((sel.cell_idx[:, 2:] == 0).all())
    assert float(out.cost) == pytest.approx(2 / (N * M)) and float(out.cost) <= 0.01
    assert torch.equal((out.state.age == 0).sum(dim=1), sel.n_valid())


@pytest.mark.parametrize("device_type", DEVICES)
def test_p2_5_runs_on_all_backends(device_type: str) -> None:
    """Every selection mode runs forward and backward through a window on the device with finite
    outputs and gradients that reach the cell, the units and the edge logits (not the scorer: hard
    selection has no gradient until P2.2); one ``topk`` position agrees with the cpu to 1e-3 (the
    P1.5 device tolerance) and makes the same selection; the backend switch refuses the pending
    backends by chunk name."""
    device = _device(device_type)
    B, T = 4, 6
    for routing in SPARSE_MODES:
        overrides = [f"model.routing={routing}"]
        if routing == "fixed":
            overrides.append("model.fixed_route=[1, 5, 9]")
        cfg, model = _model(REF_A1, overrides, perturb=17)
        model = model.to(device)
        tokens, ctx = _ids(cfg, B, T, seed=18)
        reset = torch.zeros(B, T, dtype=torch.bool)
        reset[1, 2] = True
        state = _random_state(model, B, seed=19)
        seed_everything(0)  # the random mode draws from the global RNG
        logits, final, info = model(tokens.to(device), ctx.to(device), state, reset.to(device))
        assert logits.shape == (B, T, cfg.model.vocab_size) and torch.isfinite(logits).all()
        assert final.device.type == device.type and bool(final.pos[1] == T - 2)
        K = len(cfg.model.fixed_route) if routing == "fixed" else cfg.model.k_select
        assert info["cell_idx"].shape == (B, T, cfg.model.k_select)
        assert info["unit_idx"].shape == (B, T, cfg.model.k_select)
        assert (
            info["valid"].shape == (B, T, cfg.model.k_select) and info["valid"].dtype is torch.bool
        )
        assert bool((info["valid"].sum(dim=-1) == K).all())
        assert info["cost_per_row"].shape == (B, T) and info["cost"].shape == (T,)
        torch.testing.assert_close(info["cost_per_row"].mean(dim=0), info["cost"].float())
        assert not info["cost_per_row"].requires_grad
        logits.square().mean().backward()  # type: ignore[no-untyped-call]
        grads = _grad_sum(model)
        for name, value in grads.items():
            assert math.isfinite(value) and value > 0.0, (routing, name, value)
        scorer_grad = model.scorer.net.weight.grad  # type: ignore[union-attr]
        if routing == "random":  # nothing the random control decides depends on the scorer
            assert scorer_grad is None, routing
        else:  # P2.2: the straight-through gradient reaches the scorer through alpha
            assert scorer_grad is not None and bool(torch.isfinite(scorer_grad).all()), routing
        model.zero_grad(set_to_none=True)

    # one topk position from the same state on both sides
    cfg, model = _model(REF_A1, ["model.routing=topk"], perturb=23)
    cpu_model = copy.deepcopy(model)
    on_device = model.to(device)
    state = _random_state(cpu_model, B, seed=24)
    tokens, ctx = _ids(cfg, B, 1, seed=25)
    x, c = tokens[:, 0], ctx[:, 0]
    with torch.no_grad():
        out_cpu = cpu_model.step(x, c, state)
        out_dev = on_device.step(x.to(device), c.to(device), _to_device(state, device))
    assert out_cpu.selection is not None and out_dev.selection is not None
    assert torch.equal(out_dev.selection.cell_idx.cpu(), out_cpu.selection.cell_idx)
    assert torch.equal(out_dev.selection.unit_idx.cpu(), out_cpu.selection.unit_idx)
    assert torch.equal(out_dev.selection.valid.cpu(), out_cpu.selection.valid)
    for name in ("logits", "u", "eps"):
        torch.testing.assert_close(
            getattr(out_dev, name).cpu(), getattr(out_cpu, name), atol=1e-3, rtol=1e-3, msg=name
        )
    torch.testing.assert_close(out_dev.state.h.cpu(), out_cpu.state.h, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_dev.state.q.cpu(), out_cpu.state.q, atol=1e-3, rtol=1e-3)
    assert torch.equal(out_dev.state.age.cpu(), out_cpu.state.age)
    assert float(out_dev.cost) == float(out_cpu.cost)

    # the backend switch
    assert kernels.current_backend() == "native" and kernels.BACKENDS == ("native", "cuda", "metal")
    from hsn.kernels import native

    assert kernels.ops() is native
    with pytest.raises(NotImplementedError, match="P2.10"):
        kernels.set_backend("cuda")
    with pytest.raises(NotImplementedError, match="P2.11"):
        kernels.set_backend("metal")
    with pytest.raises(ValueError, match="unknown kernel backend"):
        kernels.set_backend("vulkan")  # type: ignore[arg-type]
    kernels.set_backend("native")
    assert kernels.current_backend() == "native"


def test_p2_5_padded_slots_ignored() -> None:
    """Padded slots (``valid=False``) change nothing: the state equals the unpadded selection's
    whatever the padded slot holds (cell 0 beside a real cell 0, another cell, a unit, an α), they
    carry no gradient, and the validation ignores them while refusing duplicated or out-of-range
    valid slots. The written values are compared within fp32 rounding (``atol`` 1e-5), not bit for
    bit: a padded pair joins its unit's block in the grouped matmul, and on the hosted ubuntu
    runner's BLAS a block of another height rounds the same rows differently (exact on Apple
    silicon); the ages, the tick mask and the gradients stay exact."""
    cfg, model = _model(REF_A1, ["model.routing=topk"], perturb=41)
    cell = model.cell
    B, N = 4, cfg.model.n_cells
    state = _random_state(model, B, seed=42)
    tokens, ctx = _ids(cfg, B, 1, seed=43)
    u, eps, logits = _prefix(model, tokens[:, 0], ctx[:, 0], state)
    # rows 0 and 2 select cell 0 for real; rows 1 and 3 do not
    cells = torch.tensor([[3, 0], [7, 9], [0, 1], [5, 6]])
    units = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2]])
    valid = torch.ones(B, 2, dtype=torch.bool)
    base = _selection(cells, units, valid, logits)
    h_base, info_base = cell.step_sparse(u, eps, state, base)
    base_state = info_base["state"]
    assert isinstance(base_state, CellGraphState)

    def padded(pad_cell: int, pad_unit: int, pad_alpha: float) -> Selection:
        cells_p = torch.cat([cells, torch.full((B, 2), pad_cell)], dim=1)
        units_p = torch.cat([units, torch.full((B, 2), pad_unit)], dim=1)
        valid_p = torch.cat([valid, torch.zeros(B, 2, dtype=torch.bool)], dim=1)
        alpha_p = torch.cat([torch.ones(B, 2), torch.full((B, 2), pad_alpha)], dim=1)
        return _selection(cells_p, units_p, valid_p, logits, alpha=alpha_p)

    for pad_cell, pad_unit, pad_alpha in ((0, 0, 0.0), (5, 7, 0.7), (31, 7, 1.0)):
        sel = padded(pad_cell, pad_unit, pad_alpha)
        h_pad, info_pad = cell.step_sparse(u, eps, state, sel)
        pad_state = info_pad["state"]
        assert isinstance(pad_state, CellGraphState)
        torch.testing.assert_close(pad_state.h, base_state.h, atol=1e-5, rtol=0)
        torch.testing.assert_close(pad_state.q, base_state.q, atol=1e-5, rtol=0)
        assert torch.equal(pad_state.age, base_state.age)
        torch.testing.assert_close(h_pad[:, :2], h_base, atol=1e-5, rtol=0)
        assert torch.equal(h_pad[:, 2:], state.h[:, 0:1].expand(B, 2, cfg.model.d))  # cell 0, as is
        selected = info_pad["selected"]
        assert isinstance(selected, Tensor) and torch.equal(selected, info_base["selected"])
        assert torch.equal(pad_state.age[:, pad_cell] == 0, base_state.age[:, pad_cell] == 0)
    # the unpadded cells 0 of rows 0 and 2 were written, cell 0 of rows 1 and 3 was not
    assert torch.equal(base_state.age[:, 0] == 0, torch.tensor([True, False, True, False]))
    assert torch.equal(base_state.h[1, 0], state.h[1, 0]) and torch.equal(
        base_state.h[3, 0], state.h[3, 0]
    )

    # out-of-range padding (a P2.8 sentinel cell, a unit >= M) with a disabled cell among the valid
    # slots: still bit-equal to the base, and the disabled cell is dropped without being read
    cfg_dis, model_dis = _model(
        REF_A1, ["model.routing=topk", "model.disabled_cells=[9]"], perturb=41
    )
    model_dis.load_state_dict(model.state_dict())
    cells_s = torch.cat([cells, torch.full((B, 2), N + 7)], dim=1)
    units_s = torch.cat([units, torch.full((B, 2), cfg.model.n_units + 3)], dim=1)
    valid_s = torch.cat([valid, torch.zeros(B, 2, dtype=torch.bool)], dim=1)
    h_s, info_s = model_dis.cell.step_sparse(
        u, eps, state, _selection(cells_s, units_s, valid_s, logits)
    )
    state_s = info_s["state"]
    assert isinstance(state_s, CellGraphState)
    torch.testing.assert_close(state_s.h[[0, 2, 3]], base_state.h[[0, 2, 3]], atol=1e-5, rtol=0)
    assert torch.equal(state_s.h[1, 9], state.h[1, 9])  # the disabled cell is never written
    torch.testing.assert_close(state_s.h[1, 7], base_state.h[1, 7], atol=1e-5, rtol=0)
    assert bool(state_s.age[1, 9] == state.age[1, 9])
    torch.testing.assert_close(h_s[:, :2][[0, 2, 3]], h_base[[0, 2, 3]], atol=1e-5, rtol=0)
    valid_eff = info_s["valid"]
    assert isinstance(valid_eff, Tensor) and not bool(valid_eff[1, 1]) and bool(valid_eff[1, 0])

    # unit 7 appears only in padded slots: its parameters get no gradient at all
    model.zero_grad(set_to_none=True)
    sel = padded(2, 7, 1.0)
    _, info = cell.step_sparse(u, eps, state, sel)
    written = info["state"]
    assert isinstance(written, CellGraphState)
    (written.h.square().sum() + written.q.square().sum()).backward()  # type: ignore[no-untyped-call]
    W1, b2 = cell.units.W1.grad, cell.units.b2.grad
    assert W1 is not None and b2 is not None
    assert float(W1[7].abs().sum()) == 0.0 and float(b2[7].abs().sum()) == 0.0
    assert float(W1[[1, 2, 3, 4, 5, 6]].abs().sum()) > 0.0

    # the kernel alone: all slots padded is a no-op; validation sees valid slots only
    c_t = torch.randn(B, 2, cfg.model.d)
    b_t = torch.rand(B, 2, cfg.model.d)
    none = gated_delta_update(state, cells, c_t, b_t, torch.zeros(B, 2, dtype=torch.bool))
    assert torch.equal(none.h, state.h) and torch.equal(none.q, state.q)
    assert torch.equal(none.age, state.age) and torch.equal(none.pos, state.pos)
    sentinel = torch.tensor([[3, N + 5], [7, 9], [0, 1], [5, 6]])  # a P2.8-style sentinel
    ok = torch.tensor([[True, False], [True, True], [True, True], [True, True]])
    gated_delta_update(state, sentinel, c_t, b_t, ok)
    with pytest.raises(IndexError, match="valid cell_idx"):
        gated_delta_update(state, sentinel, c_t, b_t, torch.ones(B, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="distinct"):
        gated_delta_update(state, torch.tensor([[3, 3], [7, 9], [0, 1], [5, 6]]), c_t, b_t, valid)
    dup_padded = torch.tensor([[0, 0], [7, 9], [0, 1], [5, 6]])
    gated_delta_update(state, dup_padded, c_t, b_t, torch.tensor([[True, False]] * 4))
    assert bool(selection_mask(dup_padded, ok, N)[0, 0])
    with pytest.raises(ValueError, match="shape"):
        gated_delta_update(state, cells, c_t[:, :1], b_t, valid)
    with pytest.raises(ValueError, match="int64"):
        gated_delta_update(state, cells.to(torch.int32), c_t, b_t, valid)


# --------------------------------------------------------------------------------------------------
# Further properties
# --------------------------------------------------------------------------------------------------


def test_p2_5_mode_from_config() -> None:
    """``HSN.step`` follows ``model.routing`` unless ``mode`` says otherwise; ``mode="dense"`` on a
    sparse-routed model is the dense reference bit for bit; the refusals."""
    cfg_d, dense = _model(REF_A1, perturb=51)
    cfg_s, sparse = _model(REF_A1, ["model.routing=topk"], perturb=51)
    assert cfg_d.model.routing == "dense" and cfg_s.model.routing == "topk"
    assert dense.state_dict().keys() == sparse.state_dict().keys()
    assert all(k.startswith("scorer.") for k in dense.router.state_dict())
    assert {k for k in dense.state_dict() if k.startswith("router.")} == {
        f"router.{k}" for k in dense.router.state_dict()
    }
    assert dense.scorer is dense.router.scorer and dense.param_report()["scorer"] == 1_608
    B = 3
    state = _random_state(dense, B, seed=52)
    tokens, ctx = _ids(cfg_d, B, 1, seed=53)
    x, c = tokens[:, 0], ctx[:, 0]
    out_d = dense.step(x, c, state)
    assert out_d.selection is None and float(out_d.cost) == 1.0 and out_d.cost.dtype is dense.dtype
    assert isinstance(out_d, StepOutput)
    out_explicit = dense.step(x, c, state, mode="dense")
    assert torch.equal(out_explicit.logits, out_d.logits) and out_explicit.selection is None
    forced = sparse.step(x, c, state, mode="dense")
    assert forced.selection is None and torch.equal(forced.logits, out_d.logits)
    assert torch.equal(forced.state.h, out_d.state.h) and torch.equal(forced.state.q, out_d.state.q)
    out_s = sparse.step(x, c, state)
    assert out_s.selection is not None and out_s.selection.k == cfg_s.model.k_select
    assert torch.equal(sparse.step(x, c, state, mode="sparse").logits, out_s.logits)
    assert not torch.equal(out_s.logits, out_d.logits)
    with pytest.raises(ValueError, match="model.routing=dense"):
        dense.step(x, c, state, mode="sparse")
    with pytest.raises(ValueError, match="mode must be"):
        dense.step(x, c, state, mode="soft")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="profiler"):
        dense.step(x, c, state, profiler=object())  # type: ignore[arg-type]
    reset = torch.zeros(B, 2, dtype=torch.bool)
    tokens, ctx = _ids(cfg_d, B, 2, seed=54)
    _, _, info_d = dense(tokens, ctx, state, reset)
    _, _, info_s = sparse(tokens, ctx, state, reset)
    assert "cell_idx" not in info_d and bool((info_d["cost"] == 1.0).all())
    assert set(info_s) == set(info_d) | {
        "cell_idx",
        "unit_idx",
        "valid",
        "cost_per_row",
        "expected_cost",  # P2.2's routed statistics
        "route_log_prob",
        "explored",
        "refreshed",
        "dense_fallback",
    }
    assert info_s["cost"].dtype is sparse.dtype and bool((info_s["cost"] < 1.0).all())


def test_p2_5_disabled_cells_never_written() -> None:
    """A selected disabled cell (the P1.8 hook) is not written and keeps ageing, the other
    selected cells are, and the selection still charges it."""
    cfg, model = _model(
        REF_A1,
        [
            "model.routing=fixed",
            "model.fixed_route=[5, 1, 9]",
            "model.k_select=3",
            "model.disabled_cells=[5]",
        ],
        perturb=61,
    )
    B, N, M = 2, cfg.model.n_cells, cfg.model.n_units
    state = _random_state(model, B, seed=62)
    tokens, ctx = _ids(cfg, B, 1, seed=63)
    out = model.step(tokens[:, 0], ctx[:, 0], state)
    sel = out.selection
    assert (
        sel is not None
        and bool(sel.valid.all())
        and torch.equal(sel.cell_idx[0], torch.tensor([5, 1, 9]))
    )
    assert torch.equal(out.state.h[:, 5], state.h[:, 5])
    assert torch.equal(out.state.age[:, 5], state.age[:, 5] + 1)
    assert bool((out.state.age[:, [1, 9]] == 0).all())
    assert bool((out.state.h[:, [1, 9]] != state.h[:, [1, 9]]).any())
    assert float(out.cost) == pytest.approx(3 / (N * M))
    torch.testing.assert_close(
        out.state.q, state.q + (out.state.h - state.h).sum(dim=1) / N, atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("low", [torch.float16, torch.bfloat16])
def test_p2_5_mixed_dtype_boundaries_cpu(low: torch.dtype) -> None:
    """What autocast hands the kernels, on the cpu where hosted CI runs it: fp16 / bf16 messages
    into ``gather_concat`` and fp16 / bf16 gate and candidate into ``gated_delta_update`` against an
    fp32 state. The state keeps its dtype, the gated value is the dense path's own expression on
    the same operands bit for bit, and it is within the low dtype's rounding of the fp32 formula."""
    B, N, d, K = 3, 6, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(7)
    h = torch.rand(B, N, d, generator=generator) * 2 - 1
    state = CellGraphState.zeros(B, N, d, "cpu", torch.float32).with_dense(h)
    u = torch.randn(B, d, generator=generator)
    eps = torch.randn(B, d, generator=generator)
    m = torch.randn(B, K, d, generator=generator).to(low)
    cell_idx = torch.tensor([[0, 3, 5, 0], [1, 2, 4, 0], [5, 4, 3, 2]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False], [True] * 4])
    v = gather_concat(u, eps, state.h, m, cell_idx)
    assert v.dtype is torch.float32 and v.shape == (B, K, 4 * d)
    assert torch.equal(v[..., 3 * d :], m.float())
    index = cell_idx.unsqueeze(-1).expand(B, K, d)
    assert torch.equal(v[..., 2 * d : 3 * d], state.h.gather(1, index))
    c = torch.tanh(torch.randn(B, K, d, generator=generator)).to(low)
    b = torch.sigmoid(torch.randn(B, K, d, generator=generator)).to(low)
    new = gated_delta_update(state, cell_idx, c, b, valid)
    assert new.h.dtype is torch.float32 and new.q.dtype is torch.float32
    h_old = state.h.gather(1, index)
    dense_expr = (1 - b) * h_old + b * c  # the dense path's operands and expression
    assert dense_expr.dtype is torch.float32
    written = new.h.gather(1, index)
    assert torch.equal(written[valid], dense_expr[valid])
    exact = (1 - b.float()) * h_old + b.float() * c.float()
    tol = 2e-3 if low is torch.float16 else 1.6e-2  # one ulp of (1 − b) in the low dtype
    torch.testing.assert_close(written[valid], exact[valid], atol=tol, rtol=0)
    delta = torch.where(valid.unsqueeze(-1), dense_expr - h_old, torch.zeros_like(h_old))
    torch.testing.assert_close(new.q, state.q + delta.sum(dim=1) / N, atol=1e-6, rtol=0)
    assert torch.equal(
        new.h[~selection_mask(cell_idx, valid, N)], state.h[~selection_mask(cell_idx, valid, N)]
    )


def test_p2_5_deterministic_mode_cpu() -> None:
    """The sparse round (forward and backward) runs under ``torch.use_deterministic_algorithms``
    on the cpu, where the trainer sets it from ``train.deterministic``; the flag is restored."""
    before = torch.are_deterministic_algorithms_enabled()
    set_deterministic(True)
    try:
        cfg, model = _model(REF_A1, ["model.routing=topk"], perturb=71)
        B, T = 2, 4
        tokens, ctx = _ids(cfg, B, T, seed=72)
        state = _random_state(model, B, seed=73)
        reset = torch.zeros(B, T, dtype=torch.bool)
        logits, final, _ = model(tokens, ctx, state, reset)
        logits.square().mean().backward()  # type: ignore[no-untyped-call]
        assert all(math.isfinite(v) and v > 0 for v in _grad_sum(model).values())
        assert torch.isfinite(final.h).all()
    finally:
        set_deterministic(before)


def test_p2_5_gradcheck_tiny_fp64() -> None:
    """Finite differences in fp64 on the tiny shapes for ``gather_concat`` and
    ``gated_delta_update`` (with a padded slot), with respect to every floating input."""
    cfg, model = _model(TINY, ["model.routing=topk"], perturb=81)
    B, N, d, K = 2, cfg.model.n_cells, cfg.model.d, cfg.model.k_select
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(82)

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype).requires_grad_()

    cell_idx = torch.tensor([[2, 0], [1, 0]])
    valid = torch.tensor([[True, True], [True, False]])  # row 1: a padded cell 0
    u, eps, h, m = rand(B, d), rand(B, d), rand(B, N, d), rand(B, K, d)
    assert torch.autograd.gradcheck(
        lambda u, eps, h, m: gather_concat(u, eps, h, m, cell_idx), (u, eps, h, m)
    )

    def update(h: Tensor, q: Tensor, c: Tensor, pre_b: Tensor) -> tuple[Tensor, Tensor]:
        state = CellGraphState(
            h=h,
            q=q,
            age=torch.zeros(B, N, dtype=torch.int64),
            pos=torch.zeros(B, dtype=torch.int64),
        )
        new = gated_delta_update(state, cell_idx, torch.tanh(c), torch.sigmoid(pre_b), valid)
        return new.h, new.q

    assert torch.autograd.gradcheck(
        update, (rand(B, N, d), rand(B, d), rand(B, K, d), rand(B, K, d))
    )
    # the composed round of the tiny fp64 model, with respect to the state
    state = _random_state(model, B, seed=83)
    tokens, ctx = _ids(cfg, B, 1, seed=84)
    u_s, eps_s, logits = _prefix(model, tokens[:, 0], ctx[:, 0], state)
    sel = _selection(cell_idx, torch.tensor([[1, 0], [0, 1]]), valid, logits)

    def round_(h: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        st = CellGraphState(h=h, q=q, age=state.age, pos=state.pos)
        _, info = model.cell.step_sparse(u_s.detach(), eps_s.detach(), st, sel)
        new = info["state"]
        assert isinstance(new, CellGraphState)
        return new.h, new.q

    assert torch.autograd.gradcheck(
        round_, (state.h.clone().requires_grad_(), state.q.clone().requires_grad_())
    )
    # the other two ops are the P1.2 / P1.4 passes, checked here on the same inputs
    v = gather_concat(u_s, eps_s, state.h, m.detach().to(dtype), cell_idx).reshape(B * K, 4 * d)
    out = grouped_unit_mlp(v, sel.unit_idx.reshape(-1), model.cell.units)
    assert out.shape == (B * K, d) and out.dtype is dtype
    msgs = csr_messages(state.h, model.cell.graph, model.cell.V, cell_idx)
    expected = model.cell.graph.messages_dense(state.h, model.cell.V).gather(
        1, cell_idx.unsqueeze(-1).expand(B, K, d)
    )
    torch.testing.assert_close(msgs, expected, atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="UnitBank"):
        grouped_unit_mlp(v, sel.unit_idx.reshape(-1), model.cell.graph)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CellGraph"):
        csr_messages(state.h, model.cell.units, model.cell.V, cell_idx)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype"):
        gather_concat(u_s.float(), eps_s, state.h, m.detach().to(dtype), cell_idx)


@pytest.mark.parametrize("device_type", DEVICES)
def test_p2_5_profiler_segments(device_type: str) -> None:
    """A profiler passed to ``HSN.step`` times ``shared_path``, ``scorer``, ``messages`` (two
    entries per position), ``selected_cells`` and ``decode`` on the sparse round, and the three
    shared segments on the dense path; the ``scorer`` segment never nests in itself; the P2.1
    dense cost profile runs on the same device (it compares ``mps`` with ``mps:0``)."""
    device = _device(device_type)
    profile_name = {"cpu": "cpu", "cuda": "rtx3060", "mps": "m_series"}[device_type]
    ctx = resolve(load_hardware_profile(profile_name))
    B, T = 3, 4
    _, dense_model = _model(REF_A1)
    costs = profile_dense_costs(dense_model.to(device), ctx, batch=B, positions=2)
    assert costs["segments"]["selected_cells"]["tokens"] == 2 * B
    for routing, expected in (
        ("topk", ["shared_path", "scorer", "messages", "selected_cells", "decode"]),
        ("dense", ["shared_path", "scorer", "decode"]),
    ):
        cfg, model = _model(REF_A1, [f"model.routing={routing}"])
        model = model.to(device)
        state = model.init_state(B)
        tokens, ctx_ids = _ids(cfg, B, T, seed=91)
        tokens, ctx_ids = tokens.to(device), ctx_ids.to(device)
        profiler = Profiler(ctx)
        with torch.no_grad():
            for t in range(T):
                state = model.step(tokens[:, t], ctx_ids[:, t], state, profiler=profiler).state
        report = profiler.report()
        assert list(report.segments) == expected, routing
        for name, row in report.segments.items():
            entries = 2 * T if name == "messages" else T  # the second per-cell entry adds no tokens
            assert row.n == entries and row.tokens == T * B, (routing, name)
            assert row.device_ms > 0.0 and row.wall_ms_p50 > 0.0, (routing, name)


def test_p2_5_trains_with_tbptt(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Three TBPTT steps with ``routing=topk`` on the cpu: finite objective, the resync counters,
    tokens/s measured, the selection stacks in the window ``info``; the router group carries the
    straight-through gradient (P2.2) and appears in the per-group norms."""
    cfg = _train_config(
        "train.steps=3", "train.eval_every=3", "train.ckpt_every=3", "train.q_resync_every=2"
    )
    seed_everything(0)
    model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(model, cfg, _cpu_ctx(), run_dir=tmp_path)
    history = trainer.fit(ByteDataset(train_docs), ByteDataset(dev_docs), 3)
    assert len(history.steps) == 3
    assert np.isfinite([r.total for r in history.steps]).all()
    assert history.steps[-1].resyncs == 1 and history.steps[-1].windows == 3
    assert history.measured["tokens_per_s"] > 0.0 and history.evals
    groups = {k for k in history.steps[-1].stats if k.startswith("grad_norm/")}
    assert "grad_norm/cell" in groups and "grad_norm/router" in groups  # P2.2: STE flows
    carried = trainer._carried_state
    assert isinstance(carried, CellGraphState) and not carried.h.requires_grad
    torch.testing.assert_close(carried.q, carried.recompute_q(), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param("rtx3060", marks=pytest.mark.cuda),
        pytest.param("m_series", marks=pytest.mark.mps),
    ],
)
def test_p2_5_amp_sparse_runs(
    profile: str, train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Four sparse TBPTT steps under the profile's autocast (bf16 on CUDA, fp16 with the loss
    scaler on MPS): the fp16 messages, gate, candidate and unit outputs meet the fp32 state in the
    kernels; finite objective, fp32 master weights and carried state."""
    hardware = load_hardware_profile(profile)
    available = (
        torch.cuda.is_available()
        if hardware.device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"no {hardware.device} device on this machine")
    ctx = resolve(hardware)
    assert ctx.autocast_dtype is not None
    cfg = _train_config(
        f"profile={profile}",
        "train.steps=4",
        "train.eval_every=2",
        "train.ckpt_every=4",
        "train.stats_every=2",
    )
    seed_everything(0)
    model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(model, cfg, ctx, run_dir=tmp_path)
    history = trainer.fit(ByteDataset(train_docs), ByteDataset(dev_docs), 4)
    assert np.isfinite([r.total for r in history.steps]).all()
    assert next(model.parameters()).dtype is torch.float32
    carried = trainer._carried_state
    assert isinstance(carried, CellGraphState)
    assert carried.dtype is torch.float32 and carried.device.type == ctx.device.type
    assert history.evals and math.isfinite(history.evals[-1]["loss"])
