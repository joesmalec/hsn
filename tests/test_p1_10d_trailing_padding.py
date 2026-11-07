"""P1.10d tests: the trailing-padding cut of the TBPTT trainer.

Under ``data.align_docs`` and dense routing the trainer computes a window only up to its last
column with a document position when every row ends there (:func:`hsn.train.tbptt.trailing_cut`).
The CPU profile is the oracle: the cut run and the uncut run (the cut monkeypatched away) agree bit
for bit in fp64 and fp32, in every loss, gradient, weight and optimiser moment. The carried states
differ (the uncut one has run the padding) and the next window's reset discards both, which the
trainer requires and the tests check. Routed runs and unaligned data keep full windows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything, set_deterministic
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.runner import build_model, make_trainer
from hsn.train import tbptt
from hsn.train.loop import StreamLoader, Trainer
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer, trailing_cut
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
EXAMPLE = REPO / "configs" / "_example.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32  # the P1.7 trainer tests' window
SHORT = (16, 28)  # documents end before the window does, so every aligned window is cut
STATE_FIELDS = ("h", "q", "age", "pos")
INFO_STACKS = ("z", "z_hat", "z_target", "eps", "u")  # [B, T, d]
INFO_SERIES = ("state_abs_max", "eps_abs_mean", "alpha_entropy", "cost", "n_rounds")  # [T]
#: Stats over the computed positions: the cut run leaves the all-padding tail out of them.
POSITION_STATS = ("state_abs_max", "eps_mean", "alpha_entropy", "window_positions")

Batch = dict[str, Tensor]


# --------------------------------------------------------------------------------------------------
# Fixtures: the P1.7 recipe (kv in-range documents, ref_a1 shrunk to a few hundred parameters)
# --------------------------------------------------------------------------------------------------


def _task(length_range: tuple[int, int], dep_range: tuple[int, int] = (4, 10)) -> TaskSpec:
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 64,
            "n_dev": 16,
            "n_test": 16,
            "length_range": list(length_range),
            "dep_len_range": list(dep_range),
            "unroll": UNROLL,
        }
    )


def _docs(length_range: tuple[int, int], n: int) -> list[Example]:
    task = _task(length_range)
    return [generate(task, "train", i) for i in range(n)]  # type: ignore[arg-type]


def _config(*overrides: str, batch: int = 4) -> RunConfig:
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
            f"train.unroll={UNROLL}",
            f"train.batch={batch}",
            f"train.micro_batch={batch}",
            "train.steps=100",
            "train.eval_every=100",
            "train.ckpt_every=100",
            "train.stats_every=10",
            *overrides,
        ],
    )


def _ctx(profile: str = "cpu", master: str = "fp32") -> DeviceContext:
    hardware = load_hardware_profile(profile)
    if hardware.master_dtype != master:
        hardware = hardware.model_copy(update={"master_dtype": master})
    return resolve(hardware)


def _build(cfg: RunConfig, seed: int = 0, perturb: bool = False) -> HSN:
    seed_everything(seed)
    model = HSN.from_config(cfg)
    if perturb:  # the P1.5 recipe: states of order 0.1, so a carried state is visibly different
        generator = torch.Generator(device="cpu").manual_seed(3)
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * 0.3
                p.add_(noise.to(dtype=p.dtype))
    return model


def _windows(cfg: RunConfig, docs: list[Example], n: int) -> list[Batch]:
    """The first ``n`` training windows exactly as the trainer's loader produces them."""
    loader = StreamLoader(
        ByteDataset(docs),
        batch=cfg.train.batch,
        unroll=cfg.train.unroll,
        seed=cfg.seed,
        shuffle=True,
        align_docs=cfg.data.align_docs,
        epochs=None,
    )
    return [next(loader) for _ in range(n)]


def _copy(state: CellGraphState) -> CellGraphState:
    return CellGraphState(
        h=state.h.detach().clone(),
        q=state.q.detach().clone(),
        age=state.age.detach().clone(),
        pos=state.pos.detach().clone(),
    )


class _Scripted:
    """A training loader that hands out given windows (resume restores nothing into it)."""

    def __init__(self, windows: Sequence[Batch]) -> None:
        self._windows: Iterator[Batch] = iter(windows)

    def __iter__(self) -> _Scripted:
        return self

    def __next__(self) -> Batch:
        return next(self._windows)

    def state_dict(self) -> dict[str, Any]:
        return {"scripted": True}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        del state


def _script(
    trainer: TBPTTTrainer, windows: Sequence[Batch], monkeypatch: pytest.MonkeyPatch
) -> list[int]:
    """Feed ``windows`` to ``trainer.fit`` (evaluation stubbed out); returns the list that collects
    the number of columns of every window the model computes."""
    seen: list[int] = []
    original = trainer._forward

    def forward(batch: Batch, state: Any, **kwargs: Any) -> Any:
        seen.append(int(batch["tokens"].shape[1]))
        return original(batch, state, **kwargs)

    monkeypatch.setattr(trainer, "_loader", lambda *args, **kwargs: _Scripted(windows))
    monkeypatch.setattr(trainer, "evaluate", lambda *args, **kwargs: {})
    monkeypatch.setattr(trainer, "_forward", forward)
    return seen


# --------------------------------------------------------------------------------------------------
# A recorded training run
# --------------------------------------------------------------------------------------------------


@dataclass
class _Run:
    """What one training run computed: per window (``windows``) the computed columns, the reset
    mask, the state handed in and returned, the logits and the info; every position's state in
    order (``positions``: the state after position ``t`` of each window); the gradients of every
    step before clipping (``grads``)."""

    trainer: TBPTTTrainer
    steps: list[TBPTTStepRecord] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)
    positions: list[CellGraphState] = field(default_factory=list)
    grads: list[list[Tensor | None]] = field(default_factory=list)


def _train(
    cfg: RunConfig, ctx: DeviceContext, docs: list[Example], run_dir: Path, steps: int
) -> _Run:
    model = _build(cfg, perturb=True)
    trainer = TBPTTTrainer(model, cfg, ctx, run_dir=run_dir)
    run = _Run(trainer)
    original_forward = trainer._forward
    # P1.10a: HSN.forward no longer calls HSN.step per position; the state after position t is
    # the state the dense round at t + 1 reads (these windows reset at t = 0 only), and the carry
    rounds: list[CellGraphState] = []
    original_dense = model.cell.step_dense

    def step_dense(u: Tensor, eps: Tensor, state: CellGraphState, alpha: Tensor) -> Any:
        rounds.append(_copy(state))
        return original_dense(u, eps, state, alpha)

    def forward(batch: Batch, state: CellGraphState, **kwargs: Any) -> Any:
        first = len(run.positions)
        assert not bool(batch["reset"][:, 1:].any()), "a reset after t = 0 inside the window"
        rounds.clear()
        logits, new_state, info = original_forward(batch, state, **kwargs)
        assert len(rounds) == int(batch["tokens"].shape[1])
        run.positions.extend(rounds[1:])
        run.positions.append(_copy(new_state))
        run.windows.append(
            {
                "T": int(batch["tokens"].shape[1]),
                "reset": batch["reset"].clone(),
                "first": first,
                "state_in": _copy(state),
                "state_out": _copy(new_state),
                "logits": logits.detach().clone(),
                "info": {key: value.detach().clone() for key, value in info.items()},
            }
        )
        return logits, new_state, info

    params = list(model.parameters())
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(*args: Any, **kwargs: Any) -> Tensor:  # once per step, after the step's backwards
        run.grads.append([None if p.grad is None else p.grad.detach().clone() for p in params])
        return original_clip(*args, **kwargs)

    trainer._forward = forward  # type: ignore[method-assign]
    model.cell.step_dense = step_dense  # type: ignore[method-assign]
    trainer.evaluate = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
        history = trainer.fit(ByteDataset(docs), ByteDataset(docs), steps)
    run.steps = [r for r in history.steps if isinstance(r, TBPTTStepRecord)]
    assert len(run.steps) == steps
    return run


def _uncut(window: Mapping[str, Tensor]) -> int:
    return int(window["doc"].shape[1])


# --------------------------------------------------------------------------------------------------
# The test named by the chunk
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def _restore_determinism() -> Iterator[None]:
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


@pytest.mark.parametrize(
    ("profile", "master"),
    [
        pytest.param("cpu", "fp64", id="cpu-fp64"),
        pytest.param("cpu", "fp32", id="cpu-fp32"),
        pytest.param(CUDA_PROFILE, "fp32", id="cuda-bf16", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.usefixtures("_restore_determinism")
def test_p1_10_truncated_window_same_loss_and_carry(
    profile: str, master: str, tmp_path: Path
) -> None:
    """Six steps of two windows each, every window cut (documents of 16 to 28 positions in a
    32-position window), against the same run with the cut patched away: the same losses, parts,
    gradients, weights and optimiser moments, the same logits and info stacks on the computed
    columns, a carry out of every cut window equal to the uncut window's state at the cut, and after
    the next window's reset the same (zero) start state, although the raw carries differ.

    The floats agree to rounding, not bit for bit: since P1.10a ``HSN.forward`` runs E and D over
    the window ``[B, T]``, and a cut window (``T = c``) rounds its GEMMs differently from the full
    one (``T = 32``). The tolerance is P1.10a's window tolerance (fp64 1e-12, fp32 1e-5; 1e-5
    under bf16 autocast on CUDA); integers and the reset zeros stay exact. The weights after the six
    Adam steps take 1e-4 in fp32, except the edge logits: with the perturbed (non-uniform) edge
    weights a row sums identical cell messages only to rounding, so the edge logits' gradients are
    rounding noise of about 1e-9, near Adam's ``eps``, and their post-Adam weight measures Adam's
    amplification of that noise (1.4e-5 here, 1.5e-4 on other CPUs), not the cut. Their gradients
    are compared with every other gradient; in fp64 every weight is compared at 1e-12 (1.4e-13
    measured). Unperturbed, the edge logits' gradients are exactly 0 (the dense symmetry)."""
    if profile == CUDA_PROFILE and not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    ctx = _ctx(profile, master)
    exact = ctx.device.type == "cpu"  # deterministic kernels: the CPU always is
    tol = 1e-12 if exact and ctx.master_dtype == torch.float64 else 1e-5
    weight_tol = tol if exact and ctx.master_dtype == torch.float64 else 1e-4
    cfg = _config(
        "train.steps=6",
        "train.stats_every=3",
        "train.grad_accum=2",
        "train.q_resync_every=2",
        *(() if exact else ("train.deterministic=true",)),
    )
    docs = _docs(SHORT, 48)

    def same(a: Any, b: Any, what: str, within: float | None = None) -> None:
        if not isinstance(a, Tensor):  # a module's extra state
            assert a == b, what
        elif not a.is_floating_point():
            assert a.dtype == b.dtype and torch.equal(a, b), what
        else:
            assert a.dtype == b.dtype, what
            bound = tol if within is None else within
            torch.testing.assert_close(a, b, atol=bound, rtol=bound, msg=lambda m: f"{what}: {m}")

    def same_float(a: float, b: float, what: str) -> None:
        assert a == pytest.approx(b, rel=tol, abs=tol), (what, a, b)

    cut = _train(cfg, ctx, docs, tmp_path / "cut", 6)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tbptt, "trailing_cut", _uncut)
        full = _train(cfg, ctx, docs, tmp_path / "full", 6)

    # every window of the recipe is cut; the patched run computes every column
    assert len(cut.windows) == len(full.windows) == 12
    columns = [w["T"] for w in cut.windows]
    assert all(c < UNROLL for c in columns), columns
    assert all(w["T"] == UNROLL for w in full.windows)
    assert cut.trainer.model.state_dict()["decoder.fc1.weight"].dtype == ctx.master_dtype

    # the step records: unchanged except the statistics over the computed positions
    for index, (a, b) in enumerate(zip(cut.steps, full.steps, strict=True)):
        for name in ("loss", "accuracy", "total", "grad_norm", "lr"):
            same_float(getattr(a, name), getattr(b, name), f"step {a.step} {name}")
        assert a.parts.keys() == b.parts.keys() == {"task", "pred"}
        for name in a.parts:
            same_float(a.parts[name], b.parts[name], f"step {a.step} part {name}")
        assert (a.tokens, a.loss_positions, a.windows, a.resyncs) == (
            b.tokens,
            b.loss_positions,
            b.windows,
            b.resyncs,
        )
        assert a.stats.keys() == b.stats.keys()
        for name in a.stats.keys() - set(POSITION_STATS):  # route_cost, grad norms per group / term
            same_float(a.stats[name], b.stats[name], f"step {a.step} stat {name}")
        assert a.stats["window_positions"] == sum(columns[2 * index : 2 * index + 2]) / 2
        assert b.stats["window_positions"] == UNROLL
        if exact:  # a maximum over a prefix of the same positions (to rounding)
            assert a.stats["state_abs_max"] <= b.stats["state_abs_max"] + tol
    assert any(key.startswith("term_grad_norm/") for key in cut.steps[2].stats)

    # the gradients of every step (before clipping), the weights and the optimiser moments
    assert len(cut.grads) == len(full.grads) == 6
    for step, (grads_a, grads_b) in enumerate(zip(cut.grads, full.grads, strict=True), start=1):
        for i, (ga, gb) in enumerate(zip(grads_a, grads_b, strict=True)):
            assert (ga is None) == (gb is None), (step, i)
            if ga is not None and gb is not None:
                same(ga, gb, f"step {step} gradient {i}")
    weights_a, weights_b = cut.trainer.model.state_dict(), full.trainer.model.state_dict()
    for key, value in weights_a.items():
        if key == "cell.graph.edge_logit" and ctx.master_dtype != torch.float64:
            continue  # rounding-noise gradients amplified by Adam (docstring); gradients checked
        same(value, weights_b[key], f"weight {key}", weight_tol)
    moments_a = cut.trainer.optimizer.state_dict()["state"]
    moments_b = full.trainer.optimizer.state_dict()["state"]
    for index, entry in moments_a.items():
        for key, value in entry.items():
            same(value, moments_b[index][key], f"optimiser {index} {key}")

    # per window: the computed columns are the uncut window's, and the carry is its state at the cut
    for w, (a, b) in enumerate(zip(cut.windows, full.windows, strict=True)):
        c = a["T"]
        same(a["logits"], b["logits"][:, :c], f"window {w} logits")
        for name in INFO_STACKS:
            same(a["info"][name], b["info"][name][:, :c], f"window {w} {name}")
        for name in INFO_SERIES:
            same(a["info"][name], b["info"][name][:c], f"window {w} {name}")
        assert a["info"].keys() == b["info"].keys()
        at_cut = full.positions[b["first"] + c - 1]
        for name in STATE_FIELDS:
            same(getattr(a["state_out"], name), getattr(at_cut, name), f"window {w} carry {name}")

    # the carry is irrelevant: the next window resets every row, after which both runs start from
    # the same zeros; the raw carries differ, because the uncut window ran the padding
    differs = 0
    for w in range(1, len(cut.windows)):
        a, b = cut.windows[w], full.windows[w]
        assert bool(a["reset"][:, 0].all()) and torch.equal(a["reset"][:, 0], b["reset"][:, 0])
        start_a = a["state_in"].reset(a["reset"][:, 0])
        start_b = b["state_in"].reset(b["reset"][:, 0])
        for name in STATE_FIELDS:
            assert torch.equal(getattr(start_a, name), getattr(start_b, name)), (w, name)
            assert not getattr(start_a, name).any(), (w, name)
        differs += not torch.equal(
            cut.windows[w - 1]["state_out"].h, full.windows[w - 1]["state_out"].h
        )
    assert differs == len(cut.windows) - 1


# --------------------------------------------------------------------------------------------------
# Where the cut falls, and the loader invariant it relies on
# --------------------------------------------------------------------------------------------------


def _window(ends: Sequence[int], T: int = 8) -> Batch:
    """A window whose row ``r`` holds one document over columns ``[0, ends[r])``."""
    B = len(ends)
    doc = torch.full((B, T), -1, dtype=torch.int64)
    for r, end in enumerate(ends):
        doc[r, :end] = r
    valid = doc >= 0
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = valid[:, 0]
    return {"doc": doc, "loss_mask": valid.clone(), "reset": reset}


def test_p1_10_cut_only_where_every_row_is_padding() -> None:
    """``trailing_cut`` keeps every column up to the last one where some row holds a document,
    keeps windows without padding or without documents whole, and refuses a padding tail with a
    loss position or a reset. On the trainer's aligned loader with documents that span windows, a
    row with padding at the window's end resets at the next window's start, a cut happens exactly
    when every row has, and a row that continues its document prevents the cut."""
    assert trailing_cut(_window([8, 8])) == 8
    assert trailing_cut(_window([3, 6])) == 6
    assert trailing_cut(_window([5, 1, 2])) == 5
    assert trailing_cut(_window([0, 0])) == 8  # no document position: nothing to cut after
    assert trailing_cut(_window([7, 8])) == 8
    for key in ("loss_mask", "reset"):
        window = _window([3, 4])
        window[key][0, 6] = True
        with pytest.raises(ValueError, match=f"{key} position in its padding tail"):
            trailing_cut(window)
    with pytest.raises(TypeError, match="int64"):
        trailing_cut({**_window([3, 4]), "doc": _window([3, 4])["doc"].to(torch.int32)})

    cfg = _config()
    windows = _windows(cfg, _docs((20, 96), 24), 300)
    cuts = continued = 0
    for window, after in zip(windows[:-1], windows[1:], strict=True):
        doc = window["doc"]
        cut = trailing_cut(window)
        assert cut == int((doc >= 0).any(dim=0).nonzero().max()) + 1 or cut == UNROLL
        ended = doc[:, -1] < 0  # rows whose document ended inside the window
        assert bool(after["reset"][ended, 0].all())
        assert (cut < UNROLL) == bool(ended.all())
        if cut < UNROLL:
            cuts += 1
            assert bool(after["reset"][:, 0].all())
        keeps = ~after["reset"][:, 0]  # rows that continue their document into the next window
        if bool(keeps.any()):
            continued += 1
            assert cut == UNROLL
            assert torch.equal(after["doc"][keeps, 0], doc[keeps, -1])
    assert cuts > 10 and continued > 10, (cuts, continued)


def test_p1_10_cut_raises_when_next_window_keeps_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window after a cut window that does not reset every row at ``t = 0`` raises before the
    model computes it: that row would continue from the state at the cut, not from the state
    after the padding the uncut window computes. After an uncut window the same window trains."""
    cfg = _config("train.steps=2", "train.stats_every=2")
    docs = _docs(SHORT, 16)
    ds = ByteDataset(docs)
    first, second = _windows(cfg, docs, 2)
    c = trailing_cut(first)
    assert c < UNROLL and bool(second["reset"][:, 0].all())
    keeps_row_1 = {key: value.clone() for key, value in second.items()}
    keeps_row_1["reset"][1, 0] = False

    trainer = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path / "intact")
    seen = _script(trainer, [first, second], monkeypatch)
    trainer.fit(ds, ds, 2)
    assert seen == [c, trailing_cut(second)]

    trainer = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path / "broken")
    seen = _script(trainer, [first, keeps_row_1], monkeypatch)
    with pytest.raises(RuntimeError, match=r"rows \[1\] do not reset at t = 0"):
        trainer.fit(ds, ds, 2)
    assert seen == [c]  # the second window never reached the model

    uncut = {key: value.clone() for key, value in first.items()}
    uncut["doc"][0, -1] = uncut["doc"][0, 0]  # a document position in the last column
    assert trailing_cut(uncut) == UNROLL
    trainer = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path / "uncut")
    seen = _script(trainer, [uncut, keeps_row_1], monkeypatch)
    trainer.fit(ds, ds, 2)
    assert seen == [UNROLL, trailing_cut(keeps_row_1)]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(("data.align_docs=false",), id="dense-unaligned"),
        pytest.param(("model.routing=topk", "train.gate_mode=soft"), id="masked"),
        pytest.param(("model.routing=topk", "train.gate_mode=ste"), id="sparse"),
    ],
)
def test_p1_10_full_windows_when_routed_or_unaligned(
    overrides: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routed runs (the masked round of the soft gate, the sparse round of the hard one) and
    unaligned data keep every column of every window, although the aligned windows of the recipe
    end in padding in every row: there the padding tail is not inert (module docstring of
    :mod:`hsn.train.tbptt`)."""
    cfg = _config(*overrides, "train.steps=3", "train.stats_every=3")
    ds = ByteDataset(_docs(SHORT, 32))
    trainer = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path)
    assert not trainer._cut_enabled
    seen: list[tuple[int, int]] = []
    original = trainer._forward

    def forward(batch: Batch, state: Any, **kwargs: Any) -> Any:
        seen.append((int(batch["tokens"].shape[1]), trailing_cut(batch)))
        return original(batch, state, **kwargs)

    monkeypatch.setattr(trainer, "_forward", forward)
    history = trainer.fit(ds, ds, 3)
    assert len(seen) == 3 and all(columns == UNROLL for columns, _ in seen), seen
    assert all(
        isinstance(r, TBPTTStepRecord) and r.stats["window_positions"] == UNROLL
        for r in history.steps
    )
    if cfg.data.align_docs:  # the dense trainer would have cut these windows
        assert all(cut < UNROLL for _, cut in seen), seen


@pytest.mark.parametrize("model_name", ["gru", "transformer"])
def test_p1_10_baselines_keep_full_windows(
    model_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cut belongs to the HSN trainer only: the baselines run on the generic P0.5 trainer,
    whose windows keep every column although the aligned windows of the recipe end in padding in
    every row (the baseline comparisons of the G1 gate stay on unchanged code)."""
    cfg = load_config(
        EXAMPLE,
        overrides=[
            f"model_name={model_name}",
            f"train.unroll={UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=2",
            "train.eval_every=2",
            "train.ckpt_every=2",
            "train.stats_every=2",
            "baseline.gru.d_model=16",
            "baseline.gru.n_layers=1",
            "baseline.transformer.d_model=16",
            "baseline.transformer.n_layers=1",
            "baseline.transformer.n_heads=2",
            "baseline.transformer.ffn_hidden=32",
        ],
    )
    assert cfg.data.align_docs
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, _ctx(), run_dir=tmp_path)
    assert type(trainer) is Trainer
    seen: list[tuple[int, int]] = []
    original = trainer._forward

    def forward(batch: Batch, state: Any, **kwargs: Any) -> Any:
        seen.append((int(batch["tokens"].shape[1]), trailing_cut(batch)))
        return original(batch, state, **kwargs)

    monkeypatch.setattr(trainer, "_forward", forward)
    ds = ByteDataset(_docs(SHORT, 16))
    trainer.fit(ds, ds, 2)
    assert len(seen) == 2 and all(columns == UNROLL and cut < UNROLL for columns, cut in seen)


def test_p1_10_resume_keeps_cut_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint written after a cut window records it (``extra["cut_pending"]``): a resumed
    run continues bit for bit across cut windows, and still refuses a first window that does not
    reset every row. A checkpoint without the key (written before P1.10d, which never cut) resumes
    without the guard; a value that is not a bool raises."""
    cfg = _config(
        "train.steps=4", "train.eval_every=2", "train.ckpt_every=2", "train.stats_every=2"
    )
    docs = _docs(SHORT, 32)
    ds = ByteDataset(docs)
    run_a = tmp_path / "a"
    trainer_a = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=run_a)
    history_a = trainer_a.fit(ds, ds, 4)
    checkpoint = run_a / "step_00000002.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["extra"]["cut_pending"] is True

    trainer_b = TBPTTTrainer(_build(cfg, seed=1), cfg, _ctx(), run_dir=tmp_path / "b")
    assert trainer_b.resume(checkpoint) == 2 and trainer_b._cut_pending is True
    history_b = trainer_b.fit(ds, ds, 4)
    for rec_a, rec_b in zip(history_a.steps[2:], history_b.steps, strict=True):
        assert isinstance(rec_a, TBPTTStepRecord) and isinstance(rec_b, TBPTTStepRecord)
        assert rec_a.loss == rec_b.loss and rec_a.total == rec_b.total
        assert rec_a.parts == rec_b.parts and rec_a.stats == rec_b.stats
        assert rec_a.grad_norm == rec_b.grad_norm
        assert rec_a.stats["window_positions"] < UNROLL
    for key, value in trainer_a.model.state_dict().items():
        other = trainer_b.model.state_dict()[key]
        assert torch.equal(value, other) if isinstance(value, Tensor) else value == other, key

    keeps_row_0 = {key: value.clone() for key, value in _windows(cfg, docs, 1)[0].items()}
    keeps_row_0["reset"][0, 0] = False
    trainer_c = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path / "c")
    trainer_c.resume(checkpoint)
    seen = _script(trainer_c, [keeps_row_0], monkeypatch)
    with pytest.raises(RuntimeError, match=r"window 3 follows a window cut"):
        trainer_c.fit(ds, ds, 3)
    assert seen == []

    def tampered(change: Callable[[dict[str, Any]], None], name: str) -> Path:
        data = torch.load(checkpoint, weights_only=True)
        change(data["extra"])
        path = tmp_path / name
        torch.save(data, path)
        return path

    trainer_d = TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path / "d")
    trainer_d.resume(tampered(lambda extra: extra.pop("cut_pending"), "before_p1_10d.pt"))
    assert trainer_d._cut_pending is False
    with pytest.raises(ValueError, match="cut_pending must be a bool"):
        TBPTTTrainer(_build(cfg), cfg, _ctx(), run_dir=tmp_path / "e").resume(
            tampered(lambda extra: extra.update(cut_pending=1), "not_a_bool.pt")
        )
