"""``train.tbptt_windows`` (``docs/tbptt_windows.md``): gradient segments of several windows.

The TBPTT trainer detaches the carried state at every window (P1.7), so no window's loss trains
what an earlier window wrote. With ``train.tbptt_windows = K`` the ``train.grad_accum`` windows of
an optimiser step are cut into segments of ``K`` consecutive windows: the state is detached at a
segment's first window only, every window keeps its own call of the model over its own columns,
and the segment's objectives take one backward pass. These tests check that the key is opt-in and
refused outside its scope; that at ``K = 1`` the trainer is the per-window reference; that at
``K > 1`` a later window's loss reaches an earlier window's state (and nothing at ``K = 1``), that
the trainer's gradients, boundary-state gradients and updated parameters equal an explicit
reference (``K`` separate model calls on attached state, the same masked losses, one backward
pass), with several segments per step, a ``q`` resync inside a segment, the trailing cut and
padding; that a document reset cuts only its own row; and that a resumed run continues bit for
bit. The compiled window (``hardware.compile_mode=default``) is checked against the eager one.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import TaskSpec, generate
from hsn.device import resolve, seed_everything
from hsn.losses import assemble_terms, auxiliary_terms, terms_total
from hsn.model.hsn import HSN, address_history
from hsn.model.hsn_t import HSNT as HSNTModel
from hsn.model.state import CellGraphState
from hsn.runner import build_model
from hsn.train import tbptt
from hsn.train.amp import autocast_ctx
from hsn.train.compile import check_compile_scope, compile_window
from hsn.train.grad_norm import clip_grad_norm_
from hsn.train.tbptt import TBPTTTrainer, _columns, trailing_cut
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import GRU_2M, SPEC, UNROLL, datasets, hsn_config
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs" / "experiments" / "hsn_t_memory.yaml"
#: The HSN-T memory recipe's form on the tiny CPU model of the trainer tests: a routed (top-K,
#: straight-through) memory behind a reader, A5's addresses and readout, A9's two-byte addresses
#: carried across windows, A12's summary-gated address weights, A11's wrapped position, cut C
#: and executed credit.
HSNT = (
    "model_name=hsn_t",
    "model.reader=transformer",
    "model.reader_width=8",
    "model.reader_heads=2",
    "model.routing=topk",
    "model.k_select=3",
    "model.key_address=true",
    "model.selected_readout=true",
    "model.address_ngram=2",
    "model.address_history=ngram",
    "model.address_gate=true",
    "model.address_gate_input=summary",
    "model.scorer_position=wrap",
    "model.gate_bias_init=0.0",  # open gates: the carried state is not ~zero
    "train.feedback_detach=[summary,eps,router]",
    "train.unit_credit=executed",
    "train.gate_mode=ste",
    "train.soft_until_step=0",
    "train.optimizer=adamw",  # the reference steps the optimiser without the active-unit mask
)
#: documents run across window boundaries (what carries a gradient from window to window)
STREAM = "data.align_docs=false"


def _cfg(steps: int, *extra: str) -> RunConfig:
    return hsn_config(steps, *HSNT, *extra)


def _long_datasets() -> tuple[ByteDataset, ByteDataset]:
    """kv documents of 48 to 96 bytes in windows of 32 (P1.7's unaligned recipe, longer): a
    document runs over two to four windows, so a row's state carries from window to window."""
    task = TaskSpec.model_validate(
        {
            **SPEC.task("kv", "in_range").model_dump(),
            "n_train": 32,
            "n_dev": 8,
            "n_test": 8,
            "length_range": [48, 96],
            "dep_len_range": [4, 10],
            "unroll": UNROLL,
        }
    )

    def docs(split: str, n: int) -> ByteDataset:
        return ByteDataset([generate(task, split, i) for i in range(n)])  # type: ignore[arg-type]

    return docs("train", 32), docs("dev", 8)


def _trainer(cfg: RunConfig, run_dir: Path, seed: int = 0) -> TBPTTTrainer:
    seed_everything(seed)
    return TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)


def _params(model: torch.nn.Module) -> list[Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


class _Windows:
    """The trainer's training loader with the windows changed by ``change(index, window)``
    (``index`` counts the windows the loader gave); state and position are the loader's."""

    def __init__(self, loader: Any, change: Callable[[int, dict[str, Tensor]], dict[str, Tensor]]):
        self.loader = loader
        self.change = change
        self.count = 0

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        return self

    def __next__(self) -> dict[str, Tensor]:
        window = self.change(self.count, dict(next(self.loader)))
        self.count += 1
        return window

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def _change_windows(
    trainer: TBPTTTrainer, change: Callable[[int, dict[str, Tensor]], dict[str, Tensor]]
) -> None:
    """Route the trainer's training windows (not evaluation's) through ``change``."""
    original = trainer._loader

    def loader(ds: ByteDataset, *, shuffle: bool, **kwargs: Any) -> Any:
        made = original(ds, shuffle=shuffle, **kwargs)
        return _Windows(made, change) if shuffle else made

    trainer._loader = loader  # type: ignore[method-assign]


def _loss_in_last_window_only(every: int) -> Callable[[int, dict[str, Tensor]], dict[str, Tensor]]:
    """No supervised position except in the last of every ``every`` windows."""

    def change(index: int, window: dict[str, Tensor]) -> dict[str, Tensor]:
        if index % every != every - 1:
            window["loss_mask"] = torch.zeros_like(window["loss_mask"])
        return window

    return change


def _retain_outputs(trainer: TBPTTTrainer) -> list[tuple[CellGraphState, CellGraphState]]:
    """Spy on the trainer's windows: every window's (start state, final state), the final
    state's ``h`` and ``q`` retaining their gradients."""
    seen: list[tuple[CellGraphState, CellGraphState]] = []
    original = trainer._forward

    def spy(batch: dict[str, Tensor], state: CellGraphState, **kwargs: Any) -> Any:
        logits, new_state, info = original(batch, state, **kwargs)
        if new_state.h.requires_grad:
            new_state.h.retain_grad()
            new_state.q.retain_grad()
        seen.append((state, new_state))
        return logits, new_state, info

    trainer._forward = spy  # type: ignore[method-assign]
    return seen


def _capture_grads(trainer: TBPTTTrainer) -> list[list[Tensor | None]]:
    """The raw gradient of every step (P1.15's hook: unscaled, before clipping)."""
    grads: list[list[Tensor | None]] = []
    trainer.grad_hook = lambda t: grads.append(
        [None if p.grad is None else p.grad.detach().clone() for p in _params(t.model)]
    )
    return grads


def _capture_params(trainer: TBPTTTrainer) -> list[list[Tensor]]:
    """The parameters after every applied optimiser step."""
    after: list[list[Tensor]] = []
    trainer.update_post_hooks.append(
        lambda t: after.append([p.detach().clone() for p in _params(t.model)])
    )
    return after


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_keeps_the_default_hash() -> None:
    base = load_config(RECIPE)
    assert base.train.tbptt_windows == 1
    assert config_hash(load_config(RECIPE, ["train.tbptt_windows=1"])) == config_hash(base)
    accum = load_config(RECIPE, ["train.grad_accum=2"])
    joined = load_config(RECIPE, ["train.grad_accum=2", "train.tbptt_windows=2"])
    assert len({config_hash(base), config_hash(accum), config_hash(joined)}) == 3
    # compiled without CUDA graphs (on the card's profile too) and eager are in scope
    for mode in ("none", "default"):
        cfg = load_config(
            RECIPE,
            [
                "profile=rtx3060",
                f"hardware.compile_mode={mode}",
                "train.grad_accum=4",
                "train.tbptt_windows=2",
            ],
        )
        assert cfg.train.tbptt_windows == 2 and cfg.hardware.compile_mode == mode


K2 = ("train.tbptt_windows=2", "train.grad_accum=2")


@pytest.mark.parametrize(
    ("config", "overrides", "message"),
    [
        (RECIPE, ["train.tbptt_windows=2"], "not a multiple"),
        (RECIPE, ["train.tbptt_windows=2", "train.grad_accum=3"], "not a multiple"),
        (RECIPE, ["train.tbptt_windows=0"], "greater than or equal to 1"),
        (RECIPE, [*K2, "profile=rtx3060", "hardware.compile_mode=reduce-overhead"], "CUDA graphs"),
        (RECIPE, [*K2, "hardware.compile_mode=max-autotune"], "CUDA graphs"),
        (RECIPE, [*K2, "hardware.window=split"], "split window"),
        (RECIPE, [*K2, "model.memory=false"], "generic trainer"),
        (RECIPE, [*K2, "train.lambda_u=0.1"], "lambda_u"),
        (GRU_2M, [*K2], "generic trainer"),
    ],
)
def test_refused_outside_its_scope(config: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(config, overrides)


def test_memory_key_holds_the_windows_only_when_set() -> None:
    """The sweep's memory key (stored step-time and memory estimates) gains ``grad_accum`` and
    ``tbptt_windows`` when they are not 1, and stays what it was at their defaults."""
    from hsn.experiments.spec import memory_key

    protocol = {"name": "train_eval", "eval_batch": 128}
    base = memory_key(load_config(RECIPE), "pdg", protocol)
    named = load_config(RECIPE, ["train.grad_accum=1", "train.tbptt_windows=1"])
    assert memory_key(named, "pdg", protocol) == base
    keys = {
        memory_key(load_config(RECIPE, overrides), "pdg", protocol)
        for overrides in (["train.grad_accum=2"], ["train.grad_accum=2", "train.tbptt_windows=2"])
    }
    assert len(keys | {base}) == 3


def test_compiled_scope_accumulates_without_cuda_graphs_only() -> None:
    """The compiled window refuses ``train.grad_accum > 1`` in the CUDA-graph modes (on every
    device: one configuration means one thing) and takes it in ``default``."""
    model = build_model(_cfg(1))
    assert isinstance(model, HSN)
    check_compile_scope(model, grad_accum=4, mode="default")
    for mode in ("reduce-overhead", "max-autotune"):
        with pytest.raises(NotImplementedError, match="grad_accum"):
            check_compile_scope(model, grad_accum=2, mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grad_accum"):
        check_compile_scope(model, grad_accum=0, mode="default")


def test_trainer_refuses_a_hand_built_configuration(tmp_path: Path) -> None:
    cfg = _cfg(1, "train.grad_accum=3")
    broken = cfg.model_copy(update={"train": cfg.train.model_copy(update={"tbptt_windows": 2})})
    trainer = TBPTTTrainer(build_model(cfg), broken, resolve(cfg.hardware), run_dir=tmp_path)
    with pytest.raises(ValueError, match="multiple"):
        trainer.fit(*datasets(), 1)


# --------------------------------------------------------------------------------------------------
# The gradient crosses the window boundary at K > 1, and only there
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("windows", [1, 2, 3])
def test_loss_in_the_last_window_reaches_the_earlier_windows(windows: int, tmp_path: Path) -> None:
    """Long documents, supervision only in the step's last window, no prediction term. The
    final state of an earlier window (what its writes left in the cells) receives a gradient
    exactly when it shares the supervised window's segment: with ``K = 3`` (three windows) both
    earlier windows, with ``K = 2`` (four windows) the third, with ``K = 1`` none; every window
    but a segment's first starts from an attached state."""
    accum = 4 if windows == 2 else 3  # segments of whole steps: four windows for K = 2
    cfg = _cfg(
        1,
        STREAM,
        "train.lambda_p=0",
        f"train.grad_accum={accum}",
        f"train.tbptt_windows={windows}",
    )
    trainer = _trainer(cfg, tmp_path)
    _change_windows(trainer, _loss_in_last_window_only(cfg.train.grad_accum))
    seen = _retain_outputs(trainer)
    grads = _capture_grads(trainer)
    trainer.fit(*_long_datasets(), 1)
    n = cfg.train.grad_accum
    assert len(seen) == n and len(grads) == 1
    for index, (start, final) in enumerate(seen[:-1]):
        assert start.h.requires_grad == (index % windows != 0), index
        reached = final.h.grad is not None and bool(final.h.grad.any())
        expected = index // windows == (n - 1) // windows  # the supervised window's segment
        assert reached == expected, (index, windows)
        if not expected:  # nothing at all: zeros from the zero-weighted task, or no gradient
            assert final.q.grad is None or not bool(final.q.grad.any())
    present = [g for g in grads[0] if g is not None]
    assert present and all(bool(torch.isfinite(g).all()) for g in present)


def test_cross_window_credit_changes_the_parameter_gradient(tmp_path: Path) -> None:
    """The same windows and weights, loss in the second window only: the gradients of the
    cell, the reader and the router differ between ``K = 1`` and ``K = 2``: the second
    window's loss reaches the first window's computation only at ``K = 2``."""
    grads = {}
    for windows in (1, 2):
        cfg = _cfg(
            1, STREAM, "train.lambda_p=0", "train.grad_accum=2", f"train.tbptt_windows={windows}"
        )
        trainer = _trainer(cfg, tmp_path / str(windows))
        _change_windows(trainer, _loss_in_last_window_only(2))
        captured = _capture_grads(trainer)
        names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
        trainer.fit(*_long_datasets(), 1)
        grads[windows] = dict(zip(names, captured[0], strict=True))
    for prefix in ("cell.", "reader.", "router."):
        gaps = [
            float((grads[1][k] - grads[2][k]).abs().max())
            for k in grads[1]
            if k.startswith(prefix) and grads[1][k] is not None and grads[2][k] is not None
        ]
        assert gaps and max(gaps) > 0.0, prefix


@pytest.mark.parametrize("column", [0, 5])
def test_document_reset_cuts_only_its_row(column: int, tmp_path: Path) -> None:
    """A document starting in row 0 of the second window (at column 0, or at column 5 with no
    supervision before it in that row) leaves row 0 of the first window's final state without
    any gradient, while a continuing row keeps one."""
    cfg = _cfg(1, STREAM, "train.lambda_p=0", "train.grad_accum=2", "train.tbptt_windows=2")
    trainer = _trainer(cfg, tmp_path)
    last_only = _loss_in_last_window_only(2)

    def change(index: int, window: dict[str, Tensor]) -> dict[str, Tensor]:
        window = last_only(index, window)
        if index == 1:
            reset, loss = window["reset"].clone(), window["loss_mask"].clone()
            reset[0, column] = True
            loss[0, :column] = False
            window["reset"], window["loss_mask"] = reset, loss
        return window

    _change_windows(trainer, change)
    seen = _retain_outputs(trainer)
    trainer.fit(*_long_datasets(), 1)
    (_, first), (start, _) = seen
    assert start.h.requires_grad and first.h.grad is not None and first.q.grad is not None
    assert not bool(first.h.grad[0].any()) and not bool(first.q.grad[0].any())
    continuing = [b for b in range(1, cfg.train.batch) if bool(first.h.grad[b].any())]
    assert continuing, "no continuing row carried a gradient"


# --------------------------------------------------------------------------------------------------
# The explicit reference: K separate model calls on attached state, one backward per segment
# --------------------------------------------------------------------------------------------------


def _reference(
    cfg: RunConfig, steps: int, run_dir: Path, train_ds: ByteDataset
) -> tuple[list[list[Tensor | None]], list[list[Tensor]], list[Tensor | None]]:
    """``steps`` optimiser steps written out: every window its own call of the model (the
    trainer's windows from a loader made like the trainer's), the state detached at a segment's
    first window only, ``q`` recomputed from the attached ``h`` every ``train.q_resync_every``
    windows, the task term weighted by the step's supervised positions, the segment's objectives
    summed and one backward pass, then clipping and the optimiser. Returns the raw gradients,
    the parameters after each step and the final states' ``h`` gradients, window by window."""
    trainer = _trainer(cfg, run_dir)  # its loader, device copies, optimiser and schedule only
    model = trainer.hsn
    train = cfg.train
    ctx = trainer.ctx
    loader = trainer._loader(train_ds, shuffle=True, epochs=None)
    params = _params(model)
    state = model.init_state(train.batch, ctx.device)
    prev: Tensor | None = None
    grads: list[list[Tensor | None]] = []
    after: list[list[Tensor]] = []
    boundary: list[Tensor] = []
    windows_done, since_resync = 0, 0
    reference_cost = model.router.costs.dense_cost()
    G, K = train.grad_accum, train.tbptt_windows
    for step in range(steps):
        windows = [next(loader) for _ in range(G)]
        counts = [int(w["loss_mask"].sum()) for w in windows]
        n_loss = sum(counts)
        trainer.optimizer.zero_grad(set_to_none=True)
        for start in range(0, G, K):
            state = state.detach()
            loss: Tensor | None = None
            for window, count in zip(
                windows[start : start + K], counts[start : start + K], strict=True
            ):
                windows_done += 1
                since_resync += 1
                if since_resync >= train.q_resync_every:
                    state = CellGraphState._make(
                        h=state.h, q=state.h.mean(dim=1), age=state.age, pos=state.pos
                    )
                    since_resync = 0
                cut = trailing_cut(window) if trainer._cut_enabled else window["doc"].shape[1]
                batch = trainer._to_device(window)
                fwd = batch if cut == window["doc"].shape[1] else _columns(batch, cut)
                columns = window["reset"][:, :cut].any(dim=0).nonzero().flatten().tolist()
                model.set_train_step(step)
                address = {"prev_tokens": prev} if cfg.model.key_address else {}
                with autocast_ctx(ctx):
                    logits, state, info = model(
                        fwd["tokens"],
                        fwd["context"],
                        state,
                        fwd["reset"],
                        reset_columns=columns,
                        **address,
                    )
                state.h.retain_grad()
                boundary.append(state.h)
                if cfg.model.key_address:
                    prev = address_history(cfg.model, fwd["tokens"], prev)
                extras = auxiliary_terms(info, cfg, model.cell.graph, reference_cost=reference_cost)
                terms = assemble_terms({"logits": logits, **info}, fwd, cfg, extras)
                if G > 1:
                    weight = G * count / n_loss if n_loss else 0
                    terms = {**terms, "task": terms["task"] * weight}
                scaled = terms_total(terms) / G
                loss = scaled if loss is None else loss + scaled
            assert loss is not None
            loss.backward()  # type: ignore[no-untyped-call]
        grads.append([None if p.grad is None else p.grad.detach().clone() for p in params])
        max_norm = train.grad_clip if train.grad_clip > 0.0 else math.inf
        clip_grad_norm_(params, max_norm, precision=train.grad_norm_precision)
        trainer.optimizer.step()
        trainer.scheduler.step()
        after.append([p.detach().clone() for p in params])
    return grads, after, [None if h.grad is None else h.grad.clone() for h in boundary]


def _assert_equal(a: Any, b: Any, where: str) -> None:
    """``a == b``, tensors bit for bit, through dicts and sequences."""
    if isinstance(a, Tensor):
        assert isinstance(b, Tensor) and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for key in a:
            _assert_equal(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list | tuple):
        assert isinstance(b, list | tuple) and len(a) == len(b), where
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_equal(x, y, f"{where}[{index}]")
    else:
        assert a == b, where


def _assert_same(got: list[Any], ref: list[Any], what: str) -> None:
    assert len(got) == len(ref), what
    for index, (a, b) in enumerate(zip(got, ref, strict=True)):
        if a is None or b is None:
            assert a is None and b is None, (what, index)
        else:
            assert torch.equal(a, b), (what, index, float((a - b).abs().max()))


#: The dense HSN of the trainer tests (ref_a1, tiny): under ``data.align_docs`` its windows are
#: cut at their trailing padding (P1.10d).
DENSE = ("train.optimizer=adamw",)


@pytest.mark.parametrize(
    ("windows", "accum", "base", "long"),
    [
        pytest.param(1, 2, (*HSNT, STREAM), True, id="K1-G2"),  # P1.7: a backward per window
        pytest.param(2, 2, (*HSNT, STREAM), True, id="K2-G2"),
        pytest.param(2, 4, (*HSNT, STREAM), True, id="K2-G4-two-segments"),
        pytest.param(3, 3, (*HSNT, STREAM), True, id="K3-G3"),
        pytest.param(2, 4, (*HSNT, STREAM), False, id="K2-G4-short-docs"),
        # aligned documents: padding, mixed row lengths, windows without a supervised position,
        # a reset in every row at every window start (the dense HSN adds the trailing cut)
        pytest.param(2, 4, HSNT, False, id="K2-G4-aligned"),
        pytest.param(2, 2, DENSE, False, id="K2-G2-aligned-dense-cut"),
    ],
)
def test_trainer_equals_the_explicit_reference(
    windows: int,
    accum: int,
    base: tuple[str, ...],
    long: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two optimiser steps: the trainer's raw gradients, the gradients that reach every
    window's final state and the parameters after each step equal the reference's bit for bit.
    ``q`` is resynced every third window, inside a segment for every ``K > 1`` here. (The last
    step's per-term gradient norms are switched off: ``torch.autograd.grad`` fills the
    gradients the spy retains on the final states, which nothing in training reads.)"""
    monkeypatch.setattr(tbptt, "grad_norm_report", lambda *args, **kwargs: {})
    steps = 2
    cfg = hsn_config(
        steps,
        *base,
        f"train.grad_accum={accum}",
        f"train.tbptt_windows={windows}",
        "train.q_resync_every=3",
        "train.grad_clip=0.05",  # the clip acts
    )
    train_ds, dev_ds = _long_datasets() if long else datasets()
    ref_grads, ref_after, ref_boundary = _reference(cfg, steps, tmp_path / "ref", train_ds)
    trainer = _trainer(cfg, tmp_path / "run")
    seen = _retain_outputs(trainer)
    grads = _capture_grads(trainer)
    after = _capture_params(trainer)
    history = trainer.fit(train_ds, dev_ds, steps)
    assert trainer.windows == steps * accum and trainer.resyncs == steps * accum // 3
    assert all(math.isfinite(r.loss) for r in history.steps)
    for step in range(steps):
        _assert_same(grads[step], ref_grads[step], f"gradients of step {step + 1}")
        _assert_same(after[step], ref_after[step], f"parameters after step {step + 1}")
    _assert_same(
        [None if final.h.grad is None else final.h.grad for _, final in seen],
        ref_boundary,
        "final-state gradients",
    )
    assert trainer._cut_enabled == (base == DENSE)
    if long and windows > 1:  # the gradient crossed a window boundary
        assert any(
            final.h.grad is not None and bool(final.h.grad.any())
            for index, (_, final) in enumerate(seen)
            if index % windows != windows - 1
        )


# --------------------------------------------------------------------------------------------------
# Resume and the records of a segmented run
# --------------------------------------------------------------------------------------------------


def test_resume_continues_bit_for_bit(tmp_path: Path) -> None:
    """``K = 2`` with two segments per step: a run resumed from its step-2 checkpoint (other
    initial weights) continues the uninterrupted run's records, weights, optimiser moments,
    carried state and address bytes bit for bit; the checkpoint's state has no autograd
    history."""
    cfg = _cfg(
        4,
        STREAM,
        "train.grad_accum=4",
        "train.tbptt_windows=2",
        "train.q_resync_every=3",
        "train.ckpt_every=2",
        "train.stats_every=2",
    )
    train_ds, dev_ds = _long_datasets()
    trainer_a = _trainer(cfg, tmp_path / "a")
    history_a = trainer_a.fit(train_ds, dev_ds, 4)
    checkpoint = tmp_path / "a" / "step_00000002.pt"
    payload = torch.load(checkpoint, weights_only=True)["extra"]
    assert not payload["carried_state"]["h"].requires_grad
    assert payload["q_resync"] == {"windows": 8, "since_resync": 2, "resyncs": 2}
    trainer_b = _trainer(cfg, tmp_path / "b", seed=1)
    assert trainer_b.resume(checkpoint) == 2
    history_b = trainer_b.fit(train_ds, dev_ds, 4)
    for a, b in zip(history_a.steps[2:], history_b.steps, strict=True):
        assert (a.step, a.loss, a.total, a.parts, a.grad_norm, a.windows, a.resyncs) == (
            b.step,
            b.loss,
            b.total,
            b.parts,
            b.grad_norm,
            b.windows,
            b.resyncs,
        )
        assert {k: v for k, v in a.stats.items()} == {k: v for k, v in b.stats.items()}
    _assert_equal(trainer_a.model.state_dict(), trainer_b.model.state_dict(), "weights")
    _assert_equal(trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict(), "optimiser")
    state_a, state_b = trainer_a._carried_state, trainer_b._carried_state
    assert isinstance(state_a, CellGraphState) and isinstance(state_b, CellGraphState)
    for name in ("h", "q", "age", "pos"):
        assert torch.equal(getattr(state_a, name), getattr(state_b, name)), name
    assert state_a.h.grad_fn is None
    assert trainer_a._prev_tokens is not None and torch.equal(
        trainer_a._prev_tokens, trainer_b._prev_tokens
    )


def test_records_of_a_segmented_run(tmp_path: Path) -> None:
    """``K = 3``, ``grad_accum = 6``: three steps of six windows, finite records with the
    statistics of every window (per-term gradient norms of the step's first window on its
    statistics steps), the counters of 18 windows."""
    cfg = _cfg(3, STREAM, "train.grad_accum=6", "train.tbptt_windows=3", "train.stats_every=1")
    trainer = _trainer(cfg, tmp_path)
    history = trainer.fit(*datasets(), 3)
    assert trainer.windows == 18 and [r.windows for r in history.steps] == [6, 12, 18]
    for record in history.steps:
        assert math.isfinite(record.loss) and math.isfinite(record.grad_norm)
        assert record.tokens == 6 * cfg.train.batch * cfg.train.unroll
        assert record.stats["term_grad_norm/task"] > 0.0
        assert record.stats["window_positions"] == float(cfg.train.unroll)
    assert isinstance(trainer._carried_state, CellGraphState)
    assert trainer._carried_state.h.grad_fn is None


# --------------------------------------------------------------------------------------------------
# The compiled window without CUDA graphs
# --------------------------------------------------------------------------------------------------


def _compiled_case() -> tuple[RunConfig, HSNTModel]:
    """The P2.8 compile tests' small top-K model (ref_a1 shrunk, ``T = 8``) as HSN-T with cut C,
    away from its symmetric start (as ``test_hsn_t``'s compiled case)."""
    cfg = p28._cfg(
        "model_name=hsn_t",
        "model.reader=transformer",
        "model.reader_width=8",
        "model.reader_heads=2",
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        "train.feedback_detach=[summary,eps,router]",
    )
    seed_everything(0)
    model = HSNTModel.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    return cfg, model.train()


@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_carries_the_gradient_across_windows() -> None:
    """Two windows of the compiled window without CUDA graphs (``hardware.compile_mode=default``;
    inductor on Linux), the first one's final state carried attached into the second, a loss
    on the second window alone: the values, every parameter's gradient and the gradient into
    the first window's start state (a leaf) equal the eager chain's to the compiled-vs-eager
    tolerance, and that start-state gradient is not zero (the compiled blocks pass the
    gradient back to their inputs)."""
    _, model = _compiled_case()
    ctx = p28._ctx()
    window = compile_window(model, ctx, "default", backend=p28.CPU_BACKEND, grad_accum=2)
    assert isinstance(window, p28.CompiledWindow) and not window.cuda_graphs
    tokens, cx, state, reset = p28._case_inputs(model, True)  # routed: distinct cell states
    tokens_2, cx_2, _, _ = p28._inputs(model, seed=5)
    reset_2 = torch.zeros_like(reset)
    params = _params(model)

    def chain(fn: Callable[..., Any]) -> tuple[Any, list[Tensor | None], Any]:
        h0 = state.h.detach().clone().requires_grad_(True)
        q0 = state.q.detach().clone().requires_grad_(True)
        start = CellGraphState._make(h=h0, q=q0, age=state.age, pos=state.pos)
        torch.manual_seed(1234)
        first = fn(tokens, cx, start, reset)
        second = fn(tokens_2, cx_2, first[1], reset_2)
        grads = torch.autograd.grad(p28._loss(second), [*params, h0, q0], allow_unused=True)
        return second, list(grads), first

    ref, ref_grads, ref_first = chain(model)
    got, got_grads, _ = chain(window)
    for info in (ref_first[2], ref[2]):
        assert p28._margin(info, model.cfg.k_select) > p28.MIN_MARGIN
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    assert p28._grad_diff(got_grads, ref_grads) <= tol["grads"]
    h0_grad = ref_grads[-2]
    assert h0_grad is not None and float(h0_grad.abs().max()) > 1e-3


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize(("windows", "accum"), [(1, 2), (2, 2), (2, 4)])
def test_compiled_trainer_equals_eager(
    windows: int, accum: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer over two optimiser steps with ``hardware.compile_mode=default`` (``aot_eager``
    on the CPU: dynamo and AOTAutograd with eager kernels, as the other compiled-trainer tests)
    and with ``none``: long documents, every step a statistics step. The losses, the raw
    gradients of both steps (at ``K = 2`` they hold the second window's credit to the first)
    and the per-term gradient norms (the task term weighted under accumulation as eager
    weights it) agree to the compiled-vs-eager tolerance; every training window ran compiled,
    without CUDA graphs, and the second step compiled nothing new (the attached start state
    of a segment's later windows reuses the first window's graph)."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    runs: dict[str, dict[str, Any]] = {}
    for mode in ("none", "default"):
        cfg = _cfg(
            2,
            STREAM,
            f"train.grad_accum={accum}",
            f"train.tbptt_windows={windows}",
            "train.stats_every=1",
            f"hardware.compile_mode={mode}",
        )
        trainer = _trainer(cfg, tmp_path / mode)
        graphs: list[int] = []
        grads: list[list[Tensor | None]] = []

        def hook(
            t: Any, grads: list[list[Tensor | None]] = grads, graphs: list[int] = graphs
        ) -> None:
            grads.append([None if p.grad is None else p.grad.clone() for p in _params(t.model)])
            graphs.append(int(counters["stats"]["unique_graphs"]))

        trainer.grad_hook = hook
        history = trainer.fit(*_long_datasets(), 2)
        runs[mode] = {"history": history, "grads": grads}
        if mode == "default":
            compiled = trainer.compiled_window
            assert compiled is not None and not compiled.cuda_graphs
            assert history.measured["compiled_windows"] >= 2 * accum
            assert graphs[1] == graphs[0], graphs
    tol = p28.TOL[("cpu", "fp32")]
    eager, comp = runs["none"], runs["default"]
    for a, b in zip(eager["history"].steps, comp["history"].steps, strict=True):
        assert abs(a.loss - b.loss) <= 1e-5 * max(1.0, abs(a.loss)), (a.loss, b.loss)
        norms = {k: v for k, v in a.stats.items() if k.startswith("term_grad_norm/")}
        assert norms and list(norms) == [k for k in b.stats if k.startswith("term_grad_norm/")]
        for key, value in norms.items():
            assert abs(b.stats[key] - value) <= 1e-4 * max(1.0, abs(value)), key
    assert len(comp["grads"]) == len(eager["grads"]) == 2
    for got, ref in zip(comp["grads"], eager["grads"], strict=True):
        assert p28._grad_diff(got, ref) <= tol["grads"]
