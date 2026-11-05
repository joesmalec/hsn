"""P2.4 tests: the auxiliary terms ``L_cost``, ``L_balance`` and ``L_graph`` (values, gradients,
refusals), the window bridge ``auxiliary_terms``, the one-at-a-time rule in the configuration and
the assembly, the per-term gradient-norm report, the REINFORCE reward with the cost term on, and
every term enabled alone from config through the trainer with its gradient norm in the logs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.kernels.costs import CostTable
from hsn.losses import (
    TERM_GRAD_NORM_PREFIX,
    assemble_loss,
    assemble_terms,
    auxiliary_terms,
    balance_loss,
    cost_loss,
    grad_norm_report,
    graph_loss,
    total_and_parts,
)
from hsn.model.graph import CellGraph
from hsn.model.hsn import HSN
from hsn.model.router import Router, Scorer
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32  # the kv fixture's window (documents fit one window)
WINDOW = 16  # the synthetic windows of the unit tests
N_DOCS = 64

#: The tiny model every test uses (the P2.2 fixture): N = 6 cells, M = 3 units, K = 2.
TINY_OVERRIDES = (
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=6",
    "model.in_degree=2",
    "model.n_units=3",
    "model.k_select=2",
    "model.k_min=1",
    "model.hidden=16",
    "model.scorer_hidden=8",
    f"train.unroll={UNROLL}",
    "train.batch=4",
    "train.micro_batch=4",
)
N, M, K = 6, 3, 2
PREFIX = TERM_GRAD_NORM_PREFIX


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _config(*overrides: str) -> RunConfig:
    return load_config(REF_A1, overrides=[*TINY_OVERRIDES, *overrides])


def _sparse_config(*overrides: str) -> RunConfig:
    return _config("model.routing=topk", "train.gate_mode=ste", *overrides)


def _train_config(*overrides: str) -> RunConfig:
    return _sparse_config(
        "train.steps=40",
        "train.eval_every=40",
        "train.ckpt_every=40",
        "train.stats_every=2",
        *overrides,
    )


def _with_train(cfg: RunConfig, **changes: Any) -> RunConfig:
    """``cfg`` with ``train.*`` fields replaced without re-validation (what the loss checks)."""
    return cfg.model_copy(update={"train": cfg.train.model_copy(update=changes)})


def _build(cfg: RunConfig, seed: int = 0, perturb: int | None = 7) -> HSN:
    seed_everything(seed)
    model = HSN.from_config(cfg)
    if perturb is not None:
        _perturb(model, perturb)
    return model


def _perturb(model: HSN, seed: int, scale: float = 0.3) -> None:
    """Seeded noise on every parameter (the P1.5 recipe): a freshly initialised model's outputs
    barely depend on the state, so every gradient is ~1e-13 whatever the path."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
            p.add_(noise.to(dtype=p.dtype, device=p.device))


def _batch(cfg: RunConfig, B: int = 4, T: int = WINDOW, seed: int = 1) -> dict[str, Tensor]:
    """A synthetic window: random bytes and context ids, every position a loss position."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, cfg.model.vocab_size, (B, T + 1), generator=generator)
    ctx = torch.randint(0, cfg.model.n_context, (B, T), generator=generator)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    return {
        "tokens": tokens[:, :T],
        "context": ctx,
        "reset": reset,
        "targets": tokens[:, 1:],
        "loss_mask": torch.ones(B, T, dtype=torch.bool),
        "doc": torch.zeros(B, T, dtype=torch.int64),
    }


def _window(cfg: RunConfig, model: HSN, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """``{"logits": ..., **info}`` of one training window (the gate and the schedules apply)."""
    model.train()
    model.set_train_step(0)
    B = int(batch["tokens"].shape[0])
    logits, _, info = model(batch["tokens"], batch["context"], model.init_state(B), batch["reset"])
    return {"logits": logits, **info}


def _terms(cfg: RunConfig, model: HSN, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    out = _window(cfg, model, batch)
    extras = auxiliary_terms(out, cfg, model.cell.graph)
    return assemble_terms(out, batch, cfg, extras)


def _scorer(cfg: RunConfig, seed: int = 0) -> Scorer:
    torch.manual_seed(seed)
    return Scorer(cfg.model, unroll=cfg.train.unroll)


def _router(cfg: RunConfig, table: CostTable | None = None) -> Router:
    table = table if table is not None else CostTable.uniform(M, N)
    return Router(_scorer(cfg), table, lambda_cost=0.0)


def _logits(B: int, seed: int, n: int = N, m: int = M) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(B, n, m, generator=generator)


def _cpu() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


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


def _docs(task: TaskSpec, split: str, n: int) -> list[Example]:
    return [generate(task, split, i) for i in range(n)]  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def train_docs() -> list[Example]:
    return _docs(_tiny_task(), "train", N_DOCS)


@pytest.fixture(scope="module")
def dev_docs() -> list[Example]:
    return _docs(_tiny_task(), "dev", 16)


# --------------------------------------------------------------------------------------------------
# The chunk's tests
# --------------------------------------------------------------------------------------------------


def test_p2_4_balance_minimised_at_uniform() -> None:
    """``M · Σ_j f_j p̄_j`` is exactly 1 when both the routed fractions and the mean unit
    probabilities are uniform, rises towards ``M`` as both concentrate on one unit, counts valid
    slots only, is linear in ``p̄``, pushes an over-used unit's probability down, accepts a window's
    leading shape, and refuses degenerate inputs."""
    B = 3  # B·K = 6 slots: a round robin over the 3 units is exactly uniform
    uniform = torch.zeros(B, N, M)
    round_robin = torch.arange(B * K).remainder(M).reshape(B, K)
    valid = torch.ones(B, K, dtype=torch.bool)
    value = balance_loss(uniform, round_robin, valid)
    assert value.dtype is torch.float32 and value.ndim == 0
    assert float(value) == pytest.approx(1.0, abs=1e-6)

    # both concentrated on unit 0: f = (1, 0, 0), p̄ -> (1, 0, 0), the value -> M
    peaked = torch.zeros(B, N, M)
    peaked[..., 0] = 20.0
    all_zero = torch.zeros(B, K, dtype=torch.int64)
    assert float(balance_loss(peaked, all_zero, valid)) == pytest.approx(M, abs=1e-3)
    # linear in p̄: a skewed f with uniform p̄ is still 1 (Σ_j f_j · (1/M) · M = Σ_j f_j)
    assert float(balance_loss(uniform, all_zero, valid)) == pytest.approx(1.0, abs=1e-6)
    # in between: f concentrated, p̄ uniform except one row -> strictly between 1 and M
    mixed = balance_loss(peaked * 0.1, all_zero, valid)
    assert 1.0 < float(mixed) < M

    # padded slots (unit 0, valid = False) do not count
    padded_idx = torch.tensor([[1, 0], [2, 0], [1, 0]])
    padded_valid = torch.tensor([[True, False], [True, False], [True, False]])
    logits = _logits(B, seed=3)
    with_padding = balance_loss(logits, padded_idx, padded_valid)
    without = balance_loss(logits, padded_idx[:, :1], padded_valid[:, :1])
    torch.testing.assert_close(with_padding, without)
    counted = balance_loss(logits, padded_idx, torch.ones_like(padded_valid))
    assert not torch.isclose(counted, without)

    # the gradient raises the penalty with the over-used unit's probability
    grad_logits = torch.zeros(B, N, M, requires_grad=True)
    balance_loss(grad_logits, all_zero, valid).backward()
    assert grad_logits.grad is not None
    assert bool((grad_logits.grad[..., 0] > 0).all()) and bool(
        (grad_logits.grad[..., 1:] < 0).all()
    )

    # a window [B, T, N, M] / [B, T, K] equals its flattening over B·T
    T = 5
    window_logits = _logits(B * T, seed=4).reshape(B, T, N, M)
    generator = torch.Generator(device="cpu").manual_seed(5)
    window_idx = torch.randint(0, M, (B, T, K), generator=generator)
    window_valid = torch.rand(B, T, K, generator=generator) < 0.7
    torch.testing.assert_close(
        balance_loss(window_logits, window_idx, window_valid),
        balance_loss(
            window_logits.reshape(B * T, N, M),
            window_idx.reshape(B * T, K),
            window_valid.reshape(B * T, K),
        ),
    )

    with pytest.raises(ValueError, match="no valid slot"):
        balance_loss(uniform, round_robin, torch.zeros_like(valid))
    with pytest.raises(ValueError, match="outside"):
        balance_loss(uniform, torch.full_like(round_robin, M), valid)
    with pytest.raises(ValueError, match="outside"):
        balance_loss(uniform, torch.full_like(round_robin, -1), valid)
    with pytest.raises(ValueError, match="unit_idx must be"):
        balance_loss(uniform, round_robin[:2], valid[:2])
    with pytest.raises(ValueError, match="valid must have"):
        balance_loss(uniform, round_robin, valid[:, :1])
    with pytest.raises(ValueError, match="dtype"):
        balance_loss(uniform, round_robin, valid.to(torch.int64))
    with pytest.raises(ValueError, match="dtype"):
        balance_loss(uniform, round_robin.to(torch.int32), valid)
    with pytest.raises(ValueError, match="leading"):
        balance_loss(uniform[0], round_robin[0], valid[0])
    with pytest.raises(TypeError):
        balance_loss(uniform, round_robin.tolist(), valid)  # type: ignore[arg-type]


def test_p2_4_cost_scales_with_selection() -> None:
    """``L_cost`` is the selection's expected cost over the reference: it grows with the number of
    selected cells, equals the charged cost in the hard modes, carries the gate gradient under STE,
    is the relaxed mask's cost (below the charged dense cost) in the soft warm-up, has no gradient
    under random routing, scales with the reference, and ``auxiliary_terms`` gives the window
    mean."""
    cfg = _sparse_config()
    table = CostTable.uniform(M, N)  # dense_cost = 1.0; every (cell, unit) costs 1 / (N·M)
    router = _router(cfg, table)
    logits = _logits(4, seed=11)
    per_pair = 1.0 / (N * M)

    hard = {
        k: router.select(logits, 1.0, k, 1, "topk", gate="ste", temperature=1.0) for k in (1, 2)
    }
    for k, sel in hard.items():
        value = cost_loss(sel, table.dense_cost())
        assert value.dtype is torch.float32 and value.ndim == 0
        assert float(value) == pytest.approx(k * per_pair, rel=1e-5)
        torch.testing.assert_close(value, sel.cost.mean())  # hard modes: expected == charged
    assert float(cost_loss(hard[1])) < float(cost_loss(hard[2])) < 1.0
    dense = router.select(logits, 1.0, N, 1, "dense")
    assert float(cost_loss(dense)) == pytest.approx(1.0, rel=1e-6)

    # straight-through: the penalty reaches the scorer logits through the soft cell mask
    grad_logits = logits.clone().requires_grad_(True)
    sel = router.select(grad_logits, 1.0, K, 1, "topk", gate="ste", temperature=1.0)
    value = cost_loss(sel)
    assert value.requires_grad
    value.backward()
    assert grad_logits.grad is not None and float(grad_logits.grad.abs().sum()) > 0.0

    # soft warm-up: the relaxed mask's cost, below the dense cost that is charged
    soft_logits = logits.clone().requires_grad_(True)
    soft = router.select(soft_logits, 1.0, K, 1, "topk", gate="soft", temperature=1.0)
    assert bool((soft.cost == 1.0).all())
    assert float(cost_loss(soft)) < 1.0 and cost_loss(soft).requires_grad

    # a control: the value without a gradient path
    generator = torch.Generator(device="cpu").manual_seed(0)
    random = router.select(logits, 1.0, K, 1, "random", generator=generator)
    assert float(cost_loss(random)) == pytest.approx(K * per_pair, rel=1e-5)
    assert not cost_loss(random).requires_grad

    # the reference
    torch.testing.assert_close(cost_loss(hard[2], 0.5), 2.0 * cost_loss(hard[2], 1.0))
    for bad in (0.0, -1.0, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="reference_cost"):
            cost_loss(hard[2], bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cost_loss(logits)  # type: ignore[arg-type]

    # the window bridge: (1/T) Σ_t of the batch-mean expected cost, = the charged cost under STE
    cost_cfg = _sparse_config("train.aux_enabled=[cost]", "train.lambda_c=0.1")
    model = _build(cost_cfg)
    out = _window(cost_cfg, model, _batch(cost_cfg))
    extras = auxiliary_terms(out, cost_cfg, model.cell.graph, reference_cost=1.0)
    assert list(extras) == ["cost"]
    torch.testing.assert_close(extras["cost"], out["expected_cost"].float().mean())
    torch.testing.assert_close(extras["cost"], out["cost_per_row"].float().mean())
    assert extras["cost"].requires_grad
    torch.testing.assert_close(
        auxiliary_terms(out, cost_cfg, model.cell.graph, reference_cost=0.25)["cost"],
        4.0 * extras["cost"],
    )


def test_p2_4_graph_loss_shrinks_edges() -> None:
    """``L_graph = Σ|a_ij|`` on sigmoid edges: gradient descent on it alone lowers every edge
    weight; the value is the graph's ``l1`` (E / 2 at init) and every edge's gradient is ``σ'(l)``
    whatever the edge count; softmax edges and an edgeless graph are refused."""
    graph = CellGraph(N, 2, seed=0, edge_norm="sigmoid")
    E = graph.n_edges
    assert E == N * 2
    first = graph_loss(graph)
    assert first.dtype is torch.float32 and first.ndim == 0
    torch.testing.assert_close(first, graph.l1().float())
    assert float(first) == pytest.approx(E / 2, rel=1e-6)  # sigmoid(0) = 0.5 on every edge

    before = graph.edge_weights().detach().clone()
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.5)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = graph_loss(graph)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    after = graph.edge_weights().detach()
    assert bool((after < before).all())
    assert losses[-1] < losses[0] and all(
        b <= a for a, b in zip(losses[:-1], losses[1:], strict=True)
    )

    # a per-edge L1 prior: the gradient of every edge logit is σ'(0) = 0.25 for N = 6 and N = 12
    for n_cells in (N, 2 * N):
        fresh = CellGraph(n_cells, 2, seed=1, edge_norm="sigmoid")
        graph_loss(fresh).backward()
        assert fresh.edge_logit.grad is not None
        torch.testing.assert_close(fresh.edge_logit.grad, torch.full((fresh.n_edges,), 0.25))

    with pytest.raises(ValueError, match="sigmoid"):
        graph_loss(CellGraph(N, 2, seed=0, edge_norm="softmax"))
    with pytest.raises(ValueError, match="no edge"):
        graph_loss(CellGraph(N, 0, seed=0, edge_norm="sigmoid"))
    with pytest.raises(TypeError):
        graph_loss(torch.zeros(3))  # type: ignore[arg-type]


def test_p2_4_one_at_a_time_rule() -> None:
    """Global rule 4 at both layers: the configuration refuses two auxiliary terms without a
    written justification and refuses a term that could not act (cost / balance under dense
    routing, graph under softmax edges or without edges); the assembly refuses two terms again on
    a configuration built in code; ``auxiliary_terms`` returns nothing when nothing is enabled and
    names the missing selection otherwise."""
    with pytest.raises(ValueError, match="one at a time"):
        _sparse_config("train.aux_enabled=[cost, balance]")
    two = _sparse_config(
        "train.aux_enabled=[cost, balance]",
        "train.lambda_c=0.1",
        "train.lambda_b=0.01",
        "train.allow_multiple_aux=ablation: cost and balance together",
    )
    assert two.train.aux_enabled == ["cost", "balance"]
    for term in ("cost", "balance"):
        with pytest.raises(ValueError, match="model.routing=dense"):
            _config(f"train.aux_enabled=[{term}]")
    with pytest.raises(ValueError, match="sigmoid"):
        _config("train.aux_enabled=[graph]")
    with pytest.raises(ValueError, match="edgeless"):
        _config("train.aux_enabled=[graph]", "model.edge_norm=sigmoid", "model.in_degree=0")
    assert _config("train.aux_enabled=[graph]", "model.edge_norm=sigmoid").train.aux_enabled == [
        "graph"
    ]

    # the assembly's own check, on a configuration that skipped validation
    model = _build(two)
    batch = _batch(two)
    out = _window(two, model, batch)
    extras = auxiliary_terms(out, two, model.cell.graph)
    assert list(extras) == ["cost", "balance"]
    unjustified = _with_train(two, allow_multiple_aux="")
    with pytest.raises(ValueError, match="one at a time"):
        assemble_loss(out, batch, unjustified, extras=extras)
    total, parts = assemble_loss(out, batch, two, extras=extras)
    assert list(parts) == ["task", "pred", "cost", "balance"] and math.isfinite(float(total))

    # the bridge
    assert auxiliary_terms(out, _sparse_config(), model.cell.graph) == {}
    dense_cfg = _config()
    dense_out = _window(dense_cfg, _build(dense_cfg), batch)
    cost_cfg = _sparse_config("train.aux_enabled=[cost]", "train.lambda_c=0.1")
    with pytest.raises(ValueError, match="routed"):
        auxiliary_terms(dense_out, cost_cfg, model.cell.graph)
    graph_cfg = _sparse_config(
        "train.aux_enabled=[graph]", "train.lambda_g=0.1", "model.edge_norm=sigmoid"
    )
    with pytest.raises(ValueError, match="CellGraph"):
        auxiliary_terms(out, graph_cfg, None)
    with pytest.raises(TypeError):
        auxiliary_terms(out, two.train, model.cell.graph)  # type: ignore[arg-type]


def test_p2_4_grad_norm_report_keys() -> None:
    """The report carries ``term_grad_norm/<term>`` for every term, finite, equal to the norm of
    the gradients a backward of the term alone accumulates, unchanged by the loss scale, 0.0 for a
    term without a gradient path, and leaves ``.grad`` and the graph untouched."""
    cfg = _sparse_config("train.aux_enabled=[cost]", "train.lambda_c=0.1")
    model = _build(cfg)
    terms = _terms(cfg, model, _batch(cfg))
    assert list(terms) == ["task", "pred", "cost"]
    report = grad_norm_report(model, terms)
    assert set(report) == {f"{PREFIX}task", f"{PREFIX}pred", f"{PREFIX}cost"}
    assert all(math.isfinite(v) and v > 0.0 for v in report.values()), report
    assert all(p.grad is None for p in model.parameters())
    scaled = grad_norm_report(model, terms, scale=1024.0)
    for key in report:
        assert scaled[key] == pytest.approx(report[key], rel=1e-5), key

    terms["task"].backward(retain_graph=True)
    norm = math.sqrt(
        sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None)
    )
    assert report[f"{PREFIX}task"] == pytest.approx(norm, rel=1e-5)
    total, _ = total_and_parts(terms)
    total.backward()  # the graph survived the report and the term backward

    # no gradient path: a constant, and the cost term of a control
    assert grad_norm_report(model, {"const": torch.tensor(1.0)}) == {f"{PREFIX}const": 0.0}
    random_cfg = _config("model.routing=random", "train.aux_enabled=[cost]", "train.lambda_c=0.1")
    random_model = _build(random_cfg)
    random_terms = _terms(random_cfg, random_model, _batch(random_cfg))
    assert not random_terms["cost"].requires_grad
    random_report = grad_norm_report(random_model, random_terms)
    assert random_report[f"{PREFIX}cost"] == 0.0 and random_report[f"{PREFIX}task"] > 0.0

    for bad in (0.0, -1.0, math.inf, True):
        with pytest.raises(ValueError, match="scale"):
            grad_norm_report(model, terms, scale=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="0-dim"):
        grad_norm_report(model, {"task": torch.ones(2, requires_grad=True)})
    with pytest.raises(TypeError):
        grad_norm_report(cfg, terms)  # type: ignore[arg-type]


def test_p2_4_reinforce_reward_includes_cost(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Under REINFORCE the cell mask is hard, so the cost penalty reaches the sampled cells only
    through the reward: with the cost term on the baseline after the first step is
    ``−(task + cost)`` of that window, without it ``−task`` as in P2.2."""
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    runs = {
        "plain": ["train.gate_mode=reinforce"],
        "cost": ["train.gate_mode=reinforce", "train.aux_enabled=[cost]", "train.lambda_c=0.5"],
    }
    for name, overrides in runs.items():
        cfg = _train_config(*overrides)
        trainer = TBPTTTrainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path / name)
        history = trainer.fit(train_ds, dev_ds, 1)
        record = history.steps[-1]
        assert isinstance(record, TBPTTStepRecord)
        parts = record.parts  # one window per step: the window's parts
        assert "reinforce" in parts and parts["reinforce"] == 0.0  # the first advantage is 0
        expected = -(parts["task"] + (parts["cost"] if name == "cost" else 0.0))
        assert trainer.reinforce_baseline == pytest.approx(expected, rel=1e-6), name
        if name == "cost":
            assert parts["cost"] > 0.0 and list(parts) == ["task", "pred", "cost", "reinforce"]


def test_p2_4_each_term_trains_alone(
    train_docs: list[Example],
    dev_docs: list[Example],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Definition of done: each term enabled alone from config trains end to end on the tiny
    hard-routed model; its λ-weighted part is logged in formula order, its gradient norm appears
    on stats steps (and in the log line) and nowhere else, and the parameter-group norms are
    untouched."""
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    runs = {
        "cost": ["train.aux_enabled=[cost]", "train.lambda_c=0.1"],
        "balance": ["train.aux_enabled=[balance]", "train.lambda_b=0.01"],
        "graph": ["train.aux_enabled=[graph]", "train.lambda_g=0.01", "model.edge_norm=sigmoid"],
    }
    steps = 6
    for term, overrides in runs.items():
        cfg = _train_config(*overrides)
        trainer = TBPTTTrainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path / term)
        with caplog.at_level(logging.INFO, logger="hsn.train"):
            history = trainer.fit(train_ds, dev_ds, steps)
        records = [r for r in history.steps if isinstance(r, TBPTTStepRecord)]
        assert len(records) == steps
        for record in records:
            assert list(record.parts) == ["task", "pred", term], term
            assert record.parts[term] > 0.0 and math.isfinite(record.parts[term]), term
            assert math.isfinite(record.total) and math.isfinite(record.grad_norm)
            term_keys = {k for k in record.stats if k.startswith(PREFIX)}
            group_keys = {k for k in record.stats if k.startswith("grad_norm/")}
            if record.step % cfg.train.stats_every == 0 or record.step == steps:
                assert term_keys == {f"{PREFIX}task", f"{PREFIX}pred", f"{PREFIX}{term}"}, term
                assert all(math.isfinite(record.stats[k]) for k in term_keys)
                assert record.stats[f"{PREFIX}{term}"] > 0.0, term
                assert {"grad_norm/router", "grad_norm/cell"} <= group_keys
            else:
                assert not term_keys and not group_keys
            if term == "cost":
                assert record.stats["route_cost"] == pytest.approx(K / (N * M), abs=1e-6)
        lines = [
            rec.getMessage()
            for rec in caplog.records
            if rec.getMessage().startswith(f"step {steps}/{steps}")
        ]
        assert lines and f"{PREFIX}{term} " in lines[-1], (term, lines)
        caplog.clear()


def test_p2_4_assemble_terms_matches_assemble_loss() -> None:
    """``assemble_terms`` + ``total_and_parts`` is ``assemble_loss`` bit for bit, the term tensors
    come in formula order, and ``total_and_parts`` refuses a mapping without the task term or
    with a non-scalar value."""
    cfg = _config()
    batch = _batch(cfg)
    out = _window(cfg, _build(cfg), batch)
    terms = assemble_terms(out, batch, cfg)
    assert list(terms) == ["task", "pred"]
    assert all(t.dtype is torch.float32 and t.ndim == 0 and t.requires_grad for t in terms.values())
    total, parts = total_and_parts(terms)
    total_ref, parts_ref = assemble_loss(out, batch, cfg)
    assert torch.equal(total, total_ref) and parts == parts_ref
    assert parts == {name: float(t.detach()) for name, t in terms.items()}
    with pytest.raises(ValueError, match="task"):
        total_and_parts({"pred": terms["pred"]})
    with pytest.raises(ValueError, match="0-dim"):
        total_and_parts({"task": torch.ones(2)})
