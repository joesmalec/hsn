"""P2.3: counterfactual utility targets — probes, the utility regression and the EMA features.

Every test runs a tiny ``ref_a1`` (N = 6 cells, M = 3 units, K = 2) on the CPU unless marked; the
kv fixture is the P2.2 one (documents fit one 32-position window).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything, set_deterministic
from hsn.losses import utility_loss
from hsn.model.hsn import HSN
from hsn.model.router import (
    CANDIDATE_ADD,
    CANDIDATE_NONE,
    CANDIDATE_REMOVE,
    EMAFeatures,
    Selection,
)
from hsn.model.state import CellGraphState
from hsn.profiler import SEGMENT_NAMES, Profiler
from hsn.runner import profile_run
from hsn.train.checkpoint import capture_rng_state, restore_rng_state
from hsn.train.probes import (
    ProbeResult,
    cell_utility,
    choose_probe_position,
    classify_candidates,
    measure_utility_correlation,
    probe_window,
    run_probe,
    sample_candidates,
    spearman,
    utility_predictions,
    window_split,
)
from hsn.train.tbptt import TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32

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
    "train.unroll=16",
    "train.batch=4",
    "train.micro_batch=4",
)
N, M, K, B, T = 6, 3, 2, 4, 16
ROUTED = ("model.routing=topk", "train.gate_mode=ste")


@pytest.fixture(scope="module", autouse=True)
def _restore_deterministic_flag() -> Iterator[None]:
    """``Trainer`` sets ``torch.use_deterministic_algorithms`` process-wide."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _config(*overrides: str) -> RunConfig:
    return load_config(REF_A1, overrides=[*TINY_OVERRIDES, *overrides])


def _build(cfg: RunConfig, seed: int = 0, perturb: int | None = 7) -> HSN:
    seed_everything(seed)
    model = HSN.from_config(cfg)
    if perturb is not None:  # a fresh model barely depends on its state (the P1.5 recipe)
        generator = torch.Generator().manual_seed(perturb)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=generator, dtype=torch.float64).float() * 0.3)
    model.train()
    return model


def _window(seed: int = 1, *, loss_rate: float = 0.4, split: str = "train") -> dict[str, Any]:
    """A random training window with every row a document from position 0."""
    generator = torch.Generator().manual_seed(seed)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    return {
        "tokens": torch.randint(0, 256, (B, T), generator=generator),
        "targets": torch.randint(0, 256, (B, T), generator=generator),
        "context": torch.randint(0, 16, (B, T), generator=generator),
        "loss_mask": torch.rand(B, T, generator=generator) < loss_rate,
        "reset": reset,
        "split": split,
    }


def _random_state(model: HSN, seed: int = 2) -> CellGraphState:
    generator = torch.Generator().manual_seed(seed)
    h = torch.rand(B, N, model.cfg.d, generator=generator) * 2 - 1
    return CellGraphState(
        h=h,
        q=h.mean(dim=1),
        age=torch.randint(0, 80, (B, N), generator=generator),
        pos=torch.randint(0, 50, (B,), generator=generator),
    )


def _selection_at(model: HSN, state: CellGraphState, win: dict[str, Any], t: int) -> Selection:
    """The policy's selection at ``t`` from the state before ``t`` (RNG left as found)."""
    saved = capture_rng_state()
    try:
        with torch.no_grad(), model.router.intercept() as tap:
            model.step(win["tokens"][:, t], win["context"][:, t], state.reset(win["reset"][:, t]))
    finally:
        restore_rng_state(saved)
    return tap.selections[0]


def _all_candidates(sel: Selection) -> Tensor:
    """Every unselected (cell, unit) as an add and every valid slot as a remove, per row, padded
    with (−1, −1): ``[B, C, 2]``."""
    selected = sel.selected_cells()
    rows: list[list[tuple[int, int]]] = []
    for b in range(B):
        row = [(i, j) for i in range(N) if not bool(selected[b, i]) for j in range(M)]
        row += [
            (int(sel.cell_idx[b, k]), int(sel.unit_idx[b, k]))
            for k in range(sel.k)
            if bool(sel.valid[b, k]) and not bool(sel.explored[b, k] | sel.refreshed[b, k])
        ]
        rows.append(row)
    width = max(len(row) for row in rows)
    out = torch.full((B, width, 2), -1, dtype=torch.int64)
    for b, row in enumerate(rows):
        if row:
            out[b, : len(row)] = torch.tensor(row)
    return out


def _copy(value: Any) -> Any:
    return value.clone() if isinstance(value, Tensor) else copy.deepcopy(value)


def _same(a: Any, b: Any) -> bool:
    return torch.equal(a, b) if isinstance(a, Tensor) else bool(a == b)


def _snapshot(model: HSN, state: CellGraphState) -> dict[str, Any]:
    return {
        "state": [t.clone() for t in (state.h, state.q, state.age, state.pos)],
        "model": {k: _copy(v) for k, v in model.state_dict().items()},
        "rng": capture_rng_state(),
        "training": model.training,
        "step": model.train_step,
    }


def _assert_unchanged(model: HSN, state: CellGraphState, before: dict[str, Any]) -> None:
    for old, new in zip(before["state"], (state.h, state.q, state.age, state.pos), strict=True):
        assert torch.equal(old, new)
    for key, value in model.state_dict().items():
        assert _same(before["model"][key], value), key
    assert all(p.grad is None for p in model.parameters())
    rng = capture_rng_state()
    assert torch.equal(before["rng"]["torch"], rng["torch"])
    assert before["rng"]["numpy"] == rng["numpy"] and before["rng"]["python"] == rng["python"]
    assert model.training == before["training"] and model.train_step == before["step"]


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


@pytest.fixture(scope="module")
def train_docs() -> list[Example]:
    return [generate(_tiny_task(), "train", i) for i in range(64)]


@pytest.fixture(scope="module")
def dev_docs() -> list[Example]:
    return [generate(_tiny_task(), "dev", i) for i in range(16)]


def _train_config(*overrides: str) -> RunConfig:
    return _config(
        *ROUTED,
        f"train.unroll={UNROLL}",
        "train.steps=6",
        "train.eval_every=6",
        "train.ckpt_every=6",
        "train.stats_every=3",
        "train.lambda_u=1.0",
        "train.probe_fraction=1.0",
        "train.n_probe=2",
        "train.probe_horizon=8",
        *overrides,
    )


class _RecordingProfiler:
    """A duck-typed profiler (the trainer's segments are not all P0.6 names)."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, int]] = []

    @contextmanager
    def segment(self, name: str, tokens: int = 0) -> Iterator[None]:
        self.entries.append((name, tokens))
        yield


# --------------------------------------------------------------------------------------------------
# The chunk's tests
# --------------------------------------------------------------------------------------------------


def test_p2_3_probe_only_on_train_split() -> None:
    """``run_probe`` and ``probe_window`` refuse every split but train (and a window without one);
    the evaluation refuses the test split."""
    model = _build(_config(*ROUTED))
    state = model.init_state(B)
    none = torch.full((B, 1, 2), -1, dtype=torch.int64)
    generator = torch.Generator().manual_seed(0)
    win = _window()
    for split in ("dev", "test", "validation", None):
        bad = {key: value for key, value in win.items() if key != "split"}
        if split is not None:
            bad["split"] = split
        with pytest.raises(ValueError, match="split"):
            run_probe(model, state, bad, 0, none, 4)
        with pytest.raises(ValueError, match="split"):
            probe_window(model, state, bad, 0, n_probe=2, horizon=4, generator=generator)
    result = run_probe(model, state, win, 0, none, 4)
    assert result.n_pairs == 0 and result.rollouts == 1
    assert isinstance(
        probe_window(model, state, win, 3, n_probe=2, horizon=4, generator=generator), ProbeResult
    )
    with pytest.raises(ValueError, match="split"):
        measure_utility_correlation(model, [_window(split="test")], n_probe=2, horizon=4)
    # a window's split comes from its documents' meta
    docs = [generate(_tiny_task(), "train", 0), generate(_tiny_task(), "dev", 0)]
    ds = ByteDataset(docs)
    assert window_split(ds, torch.tensor([[0, 0, -1]])) == "train"
    assert window_split(ds, torch.tensor([[1, -1, -1]])) == "dev"
    with pytest.raises(ValueError, match="one split"):
        window_split(ds, torch.tensor([[0, 1, -1]]))


def test_p2_3_probe_does_not_mutate_state() -> None:
    """A probe leaves the caller's state, every weight and buffer, the gradients, the mode, the
    step and every RNG exactly as they were — with exploration on, which draws in every rollout."""
    cfg = _config(*ROUTED, "train.epsilon_start=0.5", "train.epsilon_end=0.5", "model.age_max=3")
    model = _build(cfg)
    state = _random_state(model)
    win = _window(loss_rate=1.0)
    torch.manual_seed(11)
    before = _snapshot(model, state)
    result = probe_window(
        model, state, win, 5, n_probe=3, horizon=6, generator=torch.Generator().manual_seed(4)
    )
    assert (
        result.n_pairs > 0 and result.rollouts > 1 and result.positions == 5 + 6 * result.rollouts
    )
    _assert_unchanged(model, state, before)
    sel = _selection_at(model, state, win, 5)
    cand = sample_candidates(sel, 2, torch.Generator().manual_seed(5))
    result = run_probe(model, state, win, 5, cand, 6)
    assert result.n_pairs > 0
    _assert_unchanged(model, state, before)
    model.eval()  # and in evaluation mode
    before = _snapshot(model, state)
    run_probe(
        model,
        state,
        win,
        5,
        sample_candidates(_selection_at(model, state, win, 5), 2, torch.Generator()),
        6,
    )
    _assert_unchanged(model, state, before)


def test_p2_3_target_sign_convention() -> None:
    """Every target equals ``L_without − L_with`` measured by an independent rollout on the masked
    (dense) path, so a candidate that lowers the loss gets a positive target: an added candidate
    that helps and a selected one whose removal hurts are both positive."""
    model = _build(_config(*ROUTED))
    state = _random_state(model)
    win = _window(loss_rate=1.0)
    t, H = 4, 5
    sel = _selection_at(model, state, win, t)
    cand = _all_candidates(sel)
    result = run_probe(model, state, win, t, cand, H)
    kind, inconsistent = classify_candidates(sel, cand)
    assert not bool(inconsistent.any())
    assert torch.equal(result.selected, kind == CANDIDATE_REMOVE)
    assert torch.equal(result.valid, kind != CANDIDATE_NONE)  # every row has loss positions

    def masked_loss(column: int | None) -> Tensor:
        """The horizon loss with column ``column``'s candidates (``None``: the baseline), every
        position on the masked path."""
        edit = None
        if column is not None:
            k = kind[:, column]
            edit = lambda s: model.router.with_candidate(  # noqa: E731
                sel, cand[:, column, 0], cand[:, column, 1], k
            )
        loss = torch.zeros(B)
        with torch.no_grad():
            current = state
            for s in range(t, t + H):
                current = current.reset(win["reset"][:, s])
                with model.router.intercept(edit if s == t else None):
                    out = model.step(
                        win["tokens"][:, s], win["context"][:, s], current, mode="masked"
                    )
                current = out.state
                ce = torch.nn.functional.cross_entropy(
                    out.logits, win["targets"][:, s], reduction="none"
                )
                loss += torch.where(win["loss_mask"][:, s], ce, torch.zeros_like(ce))
        return loss

    base = masked_loss(None)
    assert torch.allclose(result.loss_selected, base, atol=1e-5)
    for c in range(cand.shape[1]):
        with_c = masked_loss(c)
        expected = torch.where(kind[:, c] == CANDIDATE_ADD, base - with_c, with_c - base)
        valid = result.valid[:, c]
        assert torch.allclose(result.delta[:, c][valid], expected[valid], atol=1e-5), c
    added, removed = result.valid & ~result.selected, result.valid & result.selected
    assert bool((result.delta[added] > 0).any()) and bool((result.delta[added] < 0).any())
    assert bool((result.delta[removed] > 0).any())


def test_p2_3_ema_features_update(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """The EMA step, the scorer reading its buffers, and the trainer updating them once per window
    when ``λ_u > 0`` (and never otherwise)."""
    ema = EMAFeatures(utility=torch.tensor([0.5, 0.0, -1.0]), selected=torch.zeros(3))
    selection = torch.tensor([[True, False, False], [True, True, False]])
    out = ema.update(
        selection, torch.tensor([1.0, 2.0, 3.0]), 0.9, measured=torch.tensor([True, False, True])
    )
    assert torch.allclose(out.ema_selected, torch.tensor([0.1, 0.05, 0.0]))
    assert torch.allclose(out.ema_utility, torch.tensor([0.55, 0.0, -0.6]))
    assert torch.equal(ema.utility, torch.tensor([0.5, 0.0, -1.0]))  # pure
    assert torch.equal(ema.update(selection, None, 1.0).selected, ema.selected)
    assert torch.allclose(ema.update(selection, torch.ones(3), 0.0).utility, torch.ones(3))
    with pytest.raises(ValueError, match=r"\[N\]"):
        EMAFeatures(torch.zeros(2, 3), torch.zeros(2, 3)).update(selection, None, 0.9)
    with pytest.raises(ValueError, match="selection"):
        ema.update(torch.zeros(2, 4, dtype=torch.bool), None, 0.9)
    with pytest.raises(ValueError, match="measured"):
        ema.update(selection, None, 0.9, measured=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="decay"):
        ema.update(selection, None, 1.5)

    # the scorer reads its buffers when no averages are passed
    model = _build(_config(*ROUTED))
    scorer = model.scorer
    assert torch.equal(scorer.ema_utility, torch.zeros(N)) and torch.equal(
        scorer.ema_selected, torch.zeros(N)
    )
    values = EMAFeatures(
        utility=torch.arange(N, dtype=torch.float32) / 10, selected=torch.full((N,), 0.25)
    )
    scorer.load_ema(values)
    state = _random_state(model)
    u = torch.randn(B, model.cfg.d)
    feats = scorer.features(u, u, state, torch.zeros(B, dtype=torch.int64), 1.0, 0)
    slices = scorer.feature_slices
    assert torch.equal(feats[..., slices["ema_utility"]].squeeze(-1), values.utility.expand(B, N))
    assert torch.equal(feats[..., slices["ema_selected"]].squeeze(-1), values.selected.expand(B, N))
    with pytest.raises(ValueError, match="non-finite"):
        scorer.load_ema(EMAFeatures(torch.full((N,), math.nan), torch.zeros(N)))
    with pytest.raises(ValueError, match=r"\[N\]"):
        scorer.load_ema(EMAFeatures(torch.zeros(N + 1), torch.zeros(N + 1)))

    # the trainer: a probing fit moves both averages; a fit with lambda_u = 0 leaves them at zero
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    for lam in (1.0, 0.0):
        cfg = _train_config(f"train.lambda_u={lam}")
        model = _build(cfg)
        history = TBPTTTrainer(model, cfg, _cpu(), run_dir=tmp_path / f"l{lam}").fit(
            train_ds, dev_ds, 3
        )
        if lam > 0:
            assert float(model.scorer.ema_selected.sum()) > 0.0
            assert bool((model.scorer.ema_utility != 0).any())
            assert all("utility" in r.parts for r in history.steps)
            assert sum(r.stats["probe_windows"] for r in history.steps) >= 1
            assert all(r.stats["probe_pairs"] >= 0 for r in history.steps)
        else:
            assert torch.equal(model.scorer.ema_selected, torch.zeros(N))
            assert torch.equal(model.scorer.ema_utility, torch.zeros(N))
            assert all(
                "utility" not in r.parts and "probe_windows" not in r.stats for r in history.steps
            )


def test_p2_3_probe_cost_in_profiler(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Probes are charged to the ``probe`` segment: once per probe window in the trainer (with the
    window's tokens), in a real P0.6 profiler, and in the runner's profiling pass when
    ``λ_u > 0`` only."""
    assert "probe" in SEGMENT_NAMES
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    cfg = _train_config()
    model = _build(cfg)
    recorder = _RecordingProfiler()
    trainer = TBPTTTrainer(model, cfg, _cpu(), recorder, run_dir=tmp_path / "fit")
    history = trainer.fit(train_ds, dev_ds, 3)
    probes = [tokens for name, tokens in recorder.entries if name == "probe"]
    assert len(probes) == sum(int(r.stats["probe_windows"]) for r in history.steps) >= 1
    assert all(tokens == cfg.train.batch * UNROLL for tokens in probes)
    assert sum(r.stats["probe_positions"] for r in history.steps) > 0

    profiler = Profiler(_cpu())
    win = _window(loss_rate=1.0)
    with profiler.segment("probe", tokens=B * T):
        probe_window(
            _build(_config(*ROUTED)),
            _build(_config(*ROUTED)).init_state(B),
            win,
            2,
            n_probe=2,
            horizon=4,
            generator=torch.Generator(),
        )
    report = profiler.report()
    assert report.segments["probe"].n == 1 and report.segments["probe"].wall_ms_p50 > 0.0

    for lam, expected in ((1.0, True), (0.0, False)):
        run = _train_config(f"train.lambda_u={lam}")
        train_report, _ = profile_run(_build(run), run, _cpu(), train_ds, dev_ds, steps=2)
        assert ("probe" in train_report.segments) is expected
        if expected:
            assert train_report.segments["probe"].n >= 1


# --------------------------------------------------------------------------------------------------
# Further properties
# --------------------------------------------------------------------------------------------------


def test_p2_3_rollouts_share_randomness() -> None:
    """Every rollout starts from the same RNG state: under heavy exploration a candidate that
    changes nothing (a disabled cell, never written) has a target of exactly zero, while real
    candidates do not."""
    cfg = _config(
        *ROUTED, "train.epsilon_start=0.8", "train.epsilon_end=0.8", "model.disabled_cells=[5]"
    )
    model = _build(cfg)
    state = _random_state(model)
    win = _window(loss_rate=1.0)
    sel = _selection_at(model, state, win, 3)
    free = ~sel.selected_cells()[:, 5]
    assert bool(free.any())
    cand = sample_candidates(sel, 3, torch.Generator().manual_seed(1))
    noop = torch.full((B, 1, 2), -1, dtype=torch.int64)
    noop[free, 0] = torch.tensor([5, 0])
    result = run_probe(model, state, win, 3, torch.cat([noop, cand], dim=1), 8)
    assert torch.equal(result.delta[:, 0], torch.zeros(B))
    assert bool(result.valid[:, 0].any())
    assert bool((result.delta[:, 1:][result.valid[:, 1:]] != 0).any())


def test_p2_3_candidates_exclude_flagged_slots() -> None:
    """Alternatives are distinct unselected cells; explored, refreshed and padded slots and
    dense-fallback rows give no candidate; the classification follows."""
    cfg = _config(*ROUTED, "model.age_max=2")
    router = _build(cfg).router
    logits = torch.randn(B, N, M, generator=torch.Generator().manual_seed(3))
    age = torch.tensor([[0, 0, 9, 0, 0, 0]] * B)  # cell 2 is stale: a refresh probe
    dense = torch.tensor([False, False, False, True])
    sel = router.select(
        logits,
        1.0,
        K,
        1,
        "topk",
        epsilon=1.0,
        age=age,
        dense_rows=dense,
        generator=torch.Generator().manual_seed(2),
        gate="ste",
    )
    assert bool(sel.explored.any()) and bool(sel.refreshed.any())
    cand = sample_candidates(sel, 3, torch.Generator().manual_seed(0))
    assert cand.shape == (B, 3 + K, 2)
    selected = sel.selected_cells()
    for b in range(B):
        alt = cand[b, :3, 0].tolist()
        if bool(dense[b]):
            assert cand[b].eq(-1).all()
            continue
        assert len(set(alt)) == 3 and not any(bool(selected[b, i]) for i in alt)
        for k in range(K):
            flagged = bool(sel.explored[b, k] | sel.refreshed[b, k]) or not bool(sel.valid[b, k])
            pair = cand[b, 3 + k].tolist()
            assert pair == (
                [-1, -1] if flagged else [int(sel.cell_idx[b, k]), int(sel.unit_idx[b, k])]
            )
    kind, inconsistent = classify_candidates(sel, cand)
    assert not bool(inconsistent.any())
    assert bool((kind[:, :3][~dense] == CANDIDATE_ADD).all())
    explored_rows = sel.explored.any(dim=1)
    flagged_pair = torch.full((B, 1, 2), -1, dtype=torch.int64)
    for b in explored_rows.nonzero().flatten().tolist():
        k = int(sel.explored[b].nonzero()[0])
        flagged_pair[b, 0] = torch.tensor([int(sel.cell_idx[b, k]), int(sel.unit_idx[b, k])])
    _, bad = classify_candidates(sel, flagged_pair)
    assert torch.equal(bad[:, 0], explored_rows)
    n_free = min(int((~selected[0]).sum()), 3)
    many = sample_candidates(sel, N + 2, torch.Generator())
    assert int((many[0, : N + 2, 0] >= 0).sum()) == int((~selected[0]).sum()) >= n_free


def test_p2_3_with_candidate_selection() -> None:
    """The edit adds a candidate in a free slot (one padded column appended) or drops a slot, keeps
    the mask, unit weights, alpha and cost consistent, and refuses what is not a counterfactual;
    the tap records and edits what ``select`` returns."""
    cfg = _config(*ROUTED)
    router = _build(cfg).router
    logits = torch.randn(B, N, M, generator=torch.Generator().manual_seed(6))
    sel = router.select(logits, 1.0, K, 1, "topk")
    selected = sel.selected_cells()
    add_cell = int((~selected[0]).nonzero()[0])
    cells = torch.tensor([add_cell, int(sel.cell_idx[1, 0]), 0, 0])
    units = torch.tensor([2, int(sel.unit_idx[1, 0]), 0, 0])
    kind = torch.tensor([CANDIDATE_ADD, CANDIDATE_REMOVE, CANDIDATE_NONE, CANDIDATE_NONE])
    out = router.with_candidate(sel, cells, units, kind)
    assert out.k == K + 1
    assert out.selected_cells()[0, add_cell] and int(out.n_valid()[0]) == int(sel.n_valid()[0]) + 1
    slot = int(((out.cell_idx[0] == add_cell) & out.valid[0]).nonzero()[0])
    assert int(out.unit_idx[0, slot]) == 2 and float(out.alpha[0, slot]) == 1.0
    assert float(out.cell_mask[0, add_cell]) == 1.0
    assert torch.equal(out.unit_weights[0, add_cell], torch.tensor([0.0, 0.0, 1.0]))
    assert not out.selected_cells()[1, int(sel.cell_idx[1, 0])]
    assert float(out.cell_mask[1, int(sel.cell_idx[1, 0])]) == 0.0
    per_cell = router.cell_overhead + float(router.cost_table[0])  # the uniform placeholder table
    for b in range(2):
        assert float(out.cost[b]) == pytest.approx(
            router.shared_cost + int(out.n_valid()[b]) * per_cell
        )
    for name in ("cell_mask", "unit_weights", "cost"):
        assert torch.equal(getattr(out, name)[2:], getattr(sel, name)[2:]), name
    with pytest.raises(ValueError, match="already selected"):
        router.with_candidate(
            sel, sel.cell_idx[:, 0].clone(), units, torch.full((B,), CANDIDATE_ADD)
        )
    with pytest.raises(ValueError, match="not a valid slot"):
        router.with_candidate(sel, cells, (units + 1) % M, kind)
    with pytest.raises(ValueError, match="CANDIDATE_NONE"):
        router.with_candidate(sel, cells, units, torch.full((B,), 7))
    soft = router.select(logits, 1.0, K, 1, "topk", gate="soft", temperature=1.0)
    with pytest.raises(ValueError, match="soft"):
        router.with_candidate(soft, cells, units, kind)
    # K = N: no column to append; an add fills a padded slot
    full = router.select(logits, 1.0, N, 1, "topk")
    assert router.with_candidate(full, cells, units, torch.zeros(B, dtype=torch.int64)).k == N
    # the tap
    with router.intercept(lambda s: router.with_candidate(s, cells, units, kind)) as tap:
        router(torch.zeros(B, N, router.scorer.in_features))
        returned = router.select(logits, 1.0, K, 1, "topk")
        with pytest.raises(RuntimeError, match="nest"):
            with router.intercept():
                pass
    assert returned.k == K + 1 and tap.selections == [returned] and len(tap.features) == 1
    assert router.select(logits, 1.0, K, 1, "topk").k == K  # disarmed


def test_p2_3_utility_loss_reaches_scorer_only() -> None:
    """``utility_loss`` is the MSE of the pairs and refuses malformed input; the regression's
    gradient reaches the scorer and nothing else; the per-cell utilities average the targets."""
    pred = torch.tensor([[1.0], [2.0], [0.0]], requires_grad=True)
    target = torch.tensor([[0.0], [2.0], [3.0]])
    loss = utility_loss(pred, target)
    assert float(loss.detach()) == pytest.approx((1.0 + 0.0 + 9.0) / 3)
    loss.backward()
    assert torch.allclose(pred.grad, 2 * (pred.detach() - target) / 3)
    with pytest.raises(ValueError, match="requires grad"):
        utility_loss(pred.detach(), target.requires_grad_())
    with pytest.raises(ValueError, match="shape"):
        utility_loss(torch.zeros(3, 1), torch.zeros(2, 1))
    with pytest.raises(ValueError, match=r"\[\*, 1\]"):
        utility_loss(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="no probe pair"):
        utility_loss(torch.zeros(0, 1), torch.zeros(0, 1))

    model = _build(_config(*ROUTED))
    result = probe_window(
        model,
        _random_state(model),
        _window(loss_rate=1.0),
        2,
        n_probe=3,
        horizon=6,
        generator=torch.Generator().manual_seed(0),
    )
    u_pred, u_target = utility_predictions(model.router, result)
    assert u_pred.shape == u_target.shape == (result.n_pairs, 1) and not u_target.requires_grad
    utility_loss(u_pred, u_target).backward()
    for name, p in model.named_parameters():
        has_grad = p.grad is not None and bool(p.grad.abs().sum() > 0)
        assert has_grad == name.startswith("router.scorer."), name
    mean, measured = cell_utility(result, N)
    rows, cells, _, target, _ = result.pairs()
    for i in range(N):
        on_cell = cells == i
        assert bool(measured[i]) == bool(on_cell.any())
        if bool(on_cell.any()):
            assert float(mean[i]) == pytest.approx(float(target[on_cell].mean()), abs=1e-6)


def test_p2_3_config_validation() -> None:
    """``train.lambda_u > 0`` needs a routed model, probes, a horizon inside the window and a hard
    gate at some point; otherwise the P2.3 keys are inert."""
    assert _config().train.lambda_u == 0.0  # off by default, whatever probe_fraction says
    _config(*ROUTED, "train.lambda_u=1.0")
    _config("model.routing=random", "train.lambda_u=1.0")  # a control stays hard in the warm-up
    _config(*ROUTED, "train.lambda_u=1.0", "train.soft_until_step=10")
    for overrides, message in (
        (("train.lambda_u=1.0",), "dense"),
        ((*ROUTED, "train.lambda_u=1.0", "train.probe_fraction=0.0"), "probe_fraction"),
        ((*ROUTED, "train.lambda_u=1.0", "train.probe_horizon=17"), "probe_horizon"),
        (("model.routing=topk", "train.gate_mode=soft", "train.lambda_u=1.0"), "soft"),
        ((*ROUTED, "train.lambda_u=-1.0"), "lambda_u"),
    ):
        with pytest.raises(ValueError, match=message):
            _config(*overrides)


def test_p2_3_resume_bit_exact_with_probes(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Probe draws come from a per-window generator and the EMA lives in the model: a resumed
    probing run continues bit for bit."""
    cfg = _train_config(
        "train.steps=4",
        "train.ckpt_every=2",
        "train.eval_every=4",
        "train.probe_fraction=0.5",
        "train.epsilon_start=0.3",
        "train.epsilon_end=0.3",
        "train.deterministic=true",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    model_a = _build(cfg)
    history_a = TBPTTTrainer(model_a, cfg, _cpu(), run_dir=tmp_path / "a").fit(train_ds, dev_ds, 4)
    assert sum(r.stats["probe_windows"] for r in history_a.steps) >= 1
    model_b = _build(cfg, seed=1)
    trainer_b = TBPTTTrainer(model_b, cfg, _cpu(), run_dir=tmp_path / "b")
    assert trainer_b.resume(tmp_path / "a" / "step_00000002.pt") == 2
    history_b = trainer_b.fit(train_ds, dev_ds, 4)
    for rec_a, rec_b in zip(history_a.steps[2:], history_b.steps, strict=True):
        assert rec_a.total == rec_b.total and rec_a.parts == rec_b.parts
        assert rec_a.stats == rec_b.stats
    for key, value in model_a.state_dict().items():
        assert _same(value, model_b.state_dict()[key]), key


def test_p2_3_loads_pre_p2_3_state_dict() -> None:
    """A state dict written before the EMA buffers (Scorer version 1) loads with zeros; a version-2
    one without them is refused."""
    model = _build(_config(*ROUTED))
    model.scorer.load_ema(EMAFeatures(torch.ones(N), torch.ones(N)))
    sd = model.state_dict()
    assert sd._metadata["router.scorer"]["version"] == 2  # type: ignore[attr-defined]
    keys = ("router.scorer.ema_utility", "router.scorer.ema_selected")
    old = type(sd)((k, v) for k, v in sd.items() if k not in keys)
    old._metadata = {k: dict(v) for k, v in sd._metadata.items()}  # type: ignore[attr-defined]
    old._metadata["router.scorer"]["version"] = 1  # type: ignore[attr-defined]
    fresh = _build(_config(*ROUTED), perturb=None)
    fresh.scorer.load_ema(EMAFeatures(torch.full((N,), 3.0), torch.full((N,), 3.0)))
    fresh.load_state_dict(old, strict=True)
    assert torch.equal(fresh.scorer.ema_utility, torch.zeros(N))
    assert torch.equal(fresh.scorer.ema_selected, torch.zeros(N))
    old._metadata["router.scorer"]["version"] = 2  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="ema_utility"):
        fresh.load_state_dict(old, strict=True)


def test_p2_3_spearman() -> None:
    """Average-rank Spearman against hand-computed values."""
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    rx, ry = np.array([1.0, 2.5, 2.5, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(float(np.corrcoef(rx, ry)[0, 1]))
    assert spearman([3, 1, 2], [1, 3, 2]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3])) and math.isnan(spearman([1], [2]))
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2, 3])


def test_p2_3_measure_utility_correlation() -> None:
    """The evaluation pools (U, delta) pairs over windows, is deterministic for a seed, follows
    the to-the-end horizon when asked, and writes nothing."""
    model = _build(_config(*ROUTED))
    model.eval()
    windows = [_window(seed=s, split="dev") for s in (1, 2, 3)]
    before = _snapshot(model, model.init_state(B))
    first = measure_utility_correlation(
        model, windows, n_probe=2, horizon=4, seed=0, n_bootstrap=50
    )
    again = measure_utility_correlation(
        model, windows, n_probe=2, horizon=4, seed=0, n_bootstrap=50
    )
    assert first == again
    assert first.n_windows == 3 and first.n_probed_windows >= 1 and first.n_pairs > 0
    assert first.n_pairs == first.n_added + first.n_selected
    assert -1.0 <= first.spearman <= 1.0 and first.ci_low <= first.ci_high
    to_end = measure_utility_correlation(model, windows, n_probe=2, horizon=None, n_bootstrap=10)
    assert to_end.horizon is None and to_end.n_pairs > 0
    assert set(first.to_dict()) >= {"spearman", "ci_low", "ci_high", "n_pairs", "horizon"}
    _assert_unchanged(model, model.init_state(B), before)


def test_p2_3_choose_probe_position() -> None:
    """``t`` is drawn among the positions whose horizon holds a loss position of some row."""
    mask = torch.zeros(2, 10, dtype=torch.bool)
    assert choose_probe_position(mask, 3, torch.Generator()) is None
    mask[1, 6] = True
    seen = {choose_probe_position(mask, 3, torch.Generator().manual_seed(s)) for s in range(50)}
    assert seen == {4, 5, 6}
    seen = {choose_probe_position(mask, None, torch.Generator().manual_seed(s)) for s in range(200)}
    assert seen == set(range(7))
    with pytest.raises(ValueError, match="horizon"):
        choose_probe_position(mask, 11, torch.Generator())


@pytest.mark.cuda
def test_p2_3_probe_on_cuda() -> None:
    """The probe's targets on CUDA match the CPU's within fp32 rounding."""
    if not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    model = _build(_config(*ROUTED))
    state = _random_state(model)
    win = _window(loss_rate=1.0)
    sel = _selection_at(model, state, win, 3)
    cand = sample_candidates(sel, 3, torch.Generator().manual_seed(0))
    cpu = run_probe(model, state, win, 3, cand, 6)
    device = torch.device("cuda")
    model_cuda = model.to(device)
    state_cuda = replace(
        state,
        h=state.h.to(device),
        q=state.q.to(device),
        age=state.age.to(device),
        pos=state.pos.to(device),
    )
    win_cuda = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in win.items()}
    gpu = run_probe(model_cuda, state_cuda, win_cuda, 3, cand.to(device), 6)
    assert torch.equal(gpu.valid.cpu(), cpu.valid)
    assert torch.allclose(gpu.delta.cpu(), cpu.delta, atol=1e-4)
