"""Reach dropout (``train.reach_dropout``, ``train.reach_dropout_span``; HSN-T with its memory,
training only; ``docs/reach_dropout.md``).

In a training window each batch row reads, with probability ``p``, through the reader with the
per-block span ``S`` (the mask of ``model.reader_span = S``) and the other rows through the whole
window; evaluation reads as configured. These tests check that the keys are opt-in (hash, memory
key) and refused outside their scope; that the reader's mask without a short row is the
whole-window mask bit for bit and a short row's is the span reader's; that inside the model a
short row equals the span-``S`` model on that row and every other row the whole-window model (per
byte and word events, with and without C-read), in training only; that ``p = 1`` trains exactly
as the span-``S`` configuration while evaluation reads the whole window; that the rows come from
the run seed and the window counter (reruns repeat them, the global random stream is untouched);
and that the compiled window equals the eager one with short rows and compiles nothing new when
they change.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
    reach_dropout_scope_error,
)
from hsn.device import seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.hsn_t import HSNT
from hsn.model.reader import Reader
from hsn.train import tbptt
from hsn.train.compile import compile_window
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets
from test_a6_feedback_detach import A5, _cfg, _start
from test_hsn_t import READER, _gap, _model
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_tbptt_windows import _cfg as _trainer_cfg
from test_tbptt_windows import _params, _trainer
from test_write_events import _carry, _text
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs" / "experiments" / "hsn_t_memory.yaml"
WORDS = REPO / "configs" / "experiments" / "hsn_t_memory_words.yaml"
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
#: the whole-window (top0) reader with C-read: the E1 arms' control
TOP0 = [
    "model.reader_width=160",
    "model.reader_layers=4",
    "model.reader_rotary_pairs=[20,20,20,0]",
    "model.reader_span=null",
    "model.cell_attention_heads=4",
]
E1 = ["train.reach_dropout=0.5", "train.reach_dropout_span=16"]
S = 4  # the short span of the small models (windows of 8, 12 and 32 bytes)
ON = ("train.reach_dropout=0.5", f"train.reach_dropout_span={S}")
CREAD = "model.cell_attention_heads=2"
WORD = "model.write_events=word"
#: the memory's forms on the six-cell model: per byte or word events, without or with C-read
FORMS: dict[str, tuple[str, ...]] = {
    "byte": (),
    "byte_cread": (CREAD,),
    "word": (WORD,),
    "word_cread": (WORD, CREAD),
}
ROWS = (True, False, True)  # the short rows of a batch of three
#: float64: a row's outputs against the same row in another batch (test below)
ROUNDING = 1e-12


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_keys_are_opt_in_and_keep_the_hash_and_the_memory_key() -> None:
    base = load_config(RECIPE, TOP0)
    named = load_config(RECIPE, [*TOP0, "train.reach_dropout=0.0", "train.reach_dropout_span=16"])
    assert config_hash(named) == config_hash(base)
    assert "reach_dropout" not in canonical_json(base)
    assert {"train.reach_dropout", "train.reach_dropout_span"} <= set(HASH_OMIT_AT_DEFAULT)
    on = load_config(RECIPE, [*TOP0, *E1])
    eight = load_config(RECIPE, [*TOP0, "train.reach_dropout=0.5", "train.reach_dropout_span=8"])
    assert len({config_hash(base), config_hash(on), config_hash(eight)}) == 3
    # the model, its step time and its memory are the reader's: the memory key stays
    protocol = {"name": "train_eval", "eval_batch": 128}
    assert memory_key(on, "pdg", protocol) == memory_key(base, "pdg", protocol)
    assert on.model == base.model
    words = load_config(WORDS, [*TOP0, *E1])  # the word-event twin of the E1 arm
    assert (words.model.write_events, words.train.reach_dropout) == ("word", 0.5)


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE, ["train.reach_dropout=0.5"], "reader_span=32 already limits every row"),
        (RECIPE, [*TOP0, *E1, "train.reach_dropout_span=256"], "not shorter than the window"),
        (RECIPE, [*TOP0, *E1, "train.lambda_u=0.1"], "lambda_u"),
        # docs/reader_causal_flash.md: the causal mask has no mask tensor for the span to narrow
        (RECIPE, [*TOP0, *E1, "model.reader_mask=causal"], "short rows' span mask could not act"),
        (RECIPE_A6, ["train.reach_dropout=0.5"], "no memory behind a reader"),
        (NOMEM, ["train.reach_dropout=0.5"], "train.reach_dropout=0.5"),
        (RECIPE, [*TOP0, "train.reach_dropout=1.5"], "less than or equal"),
        (RECIPE, [*TOP0, *E1, "train.reach_dropout_span=0"], "greater than or equal"),
    ],
)
def test_configuration_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_causal_mask_and_carried_reader_refuse_short_rows() -> None:
    """``model.reader_mask=causal`` (SDPA's ``is_causal``, no mask tensor) and a carried reader
    (``Reader.read``'s own mask) take no short rows: refused at configuration load, and the
    causal reader refuses them itself."""
    causal = load_config(RECIPE, [*TOP0, "model.reader_mask=causal"])
    assert causal.data.align_docs and causal.train.reach_dropout == 0.0  # allowed without it
    on = load_config(RECIPE, [*TOP0, *E1])
    assert reach_dropout_scope_error(on) is None
    # model.reader_carry with the memory is refused before this check on main (17 Nov 2026);
    # the scope check refuses it on its own, whichever refusal a later main keeps
    carried = on.model_copy(update={"model": on.model.model_copy(update={"reader_carry": 8})})
    reason = reach_dropout_scope_error(carried)
    assert reason is not None and "model.reader_carry=8" in reason
    cfg = _cfg(*A5, *READER, "model.reader_mask=causal")
    reader = Reader(cfg.model, seed=0).double()
    assert reader.causal
    tokens = torch.randint(0, 256, (3, 8))
    reset = torch.zeros(3, 8, dtype=torch.bool)
    reset[:, 0] = True
    reader(tokens, reset)  # no short rows: the causal reader as on main
    with pytest.raises(ValueError, match="reader_mask=causal"):
        reader(tokens, reset, (torch.tensor(ROWS), S))


# --------------------------------------------------------------------------------------------------
# The reader and the model
# --------------------------------------------------------------------------------------------------


def _pair(*extra: str) -> tuple[RunConfig, HSNT, HSNT]:
    """The six-cell HSN-T with the whole-window reader and the same model with
    ``model.reader_span = S``: the same parameters (a span is a mask, not a parameter)."""
    cfg = _cfg(*A5, *READER, *extra)
    whole, short = _model(cfg), _model(_cfg(*A5, *READER, *extra, f"model.reader_span={S}"))
    for (name, a), (_, b) in zip(whole.named_parameters(), short.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    return cfg, whole, short


def test_reader_short_rows_are_the_span_readers_mask() -> None:
    cfg, whole_model, short_model = _pair()
    whole, short = whole_model.reader, short_model.reader
    assert whole is not None and short is not None
    tokens, _, reset, _ = _text(cfg)  # documents start mid-row in rows 0 and 1
    rows = torch.tensor(ROWS)
    mask = whole.attention_mask(reset, (rows, S))
    assert torch.equal(mask[rows], short.attention_mask(reset)[rows])
    assert torch.equal(mask[~rows], whole.attention_mask(reset)[~rows])
    out, plain, ref = whole(tokens, reset, (rows, S)), whole(tokens, reset), short(tokens, reset)
    assert torch.equal(out[rows], ref[rows]) and torch.equal(out[~rows], plain[~rows])
    assert _gap(ref[rows], plain[rows]) > 1e-6  # the span changes those rows
    none = torch.zeros(3, dtype=torch.bool)
    assert torch.equal(whole(tokens, reset, (none, S)), plain)  # no short row: bit for bit
    with pytest.raises(ValueError, match="whole-window reader"):
        short(tokens, reset, (rows, S))
    for bad, message in (
        ((rows.long(), S), "bool"),
        ((rows[:2], S), "bool"),
        ((rows, 0), "int >= 1"),
    ):
        with pytest.raises(ValueError, match=message):
            whole(tokens, reset, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bool"):
        whole_model.set_reach_rows(rows.long(), S)
    with pytest.raises(ValueError, match="int >= 1"):
        whole_model.set_reach_rows(rows, 0)


@pytest.mark.parametrize("form", list(FORMS))
def test_short_rows_are_the_span_model_in_training_only(form: str) -> None:
    """A training window with short rows: a short row's logits and final state are the
    span-``S`` model's, every other row's the whole-window model's (per byte and word events,
    with and without C-read), in float64 to rounding: C-read's batched products round by the
    batch's other rows (2.4e-15 measured), against a gap of 2 to 4 between the two readers. In
    evaluation the rows change nothing, bit for bit."""
    cfg, whole, short = _pair(*FORMS[form])
    tokens, ctx, reset, prev = _text(cfg)
    state = _start(whole, 3)
    rows = torch.tensor(ROWS)

    def run(model: HSNT, reach: Tensor | None = None) -> tuple[Tensor, Tensor]:
        model.set_reach_rows(reach, S)
        try:
            logits, final, _ = model(tokens, ctx, state, reset, prev_tokens=_carry(cfg, prev))
        finally:
            model.set_reach_rows(None)
        return logits, final.h

    mixed, plain, ref = run(whole, rows), run(whole), run(short)
    for got, want in zip(mixed, ref, strict=True):
        assert _gap(got[rows], want[rows]) <= ROUNDING
    for got, want in zip(mixed, plain, strict=True):
        assert _gap(got[~rows], want[~rows]) <= ROUNDING
    assert _gap(ref[0][rows], plain[0][rows]) > 0.1
    whole.eval()
    short.eval()
    evaluated = run(whole)
    assert all(torch.equal(a, b) for a, b in zip(run(whole, rows), evaluated, strict=True))
    assert _gap(evaluated[0], run(short)[0]) > 1e-6
    assert whole._reach is None


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


def _grads_hook(grads: list[list[Tensor | None]]) -> Callable[[Any], None]:
    def hook(trainer: Any) -> None:
        grads.append([None if p.grad is None else p.grad.clone() for p in _params(trainer.model)])

    return hook


def _spy_rows(trainer: TBPTTTrainer) -> list[Tensor]:
    """The rows the trainer hands the model, window by window."""
    seen: list[Tensor] = []
    original = trainer.hsn.set_reach_rows

    def spy(rows: Tensor | None, span: int = 0) -> None:
        if rows is not None:
            assert span == trainer.cfg.train.reach_dropout_span
            seen.append(rows.clone())
        original(rows, span)

    trainer.hsn.set_reach_rows = spy  # type: ignore[method-assign]
    return seen


def test_p1_trains_as_the_span_reader_and_evaluates_the_whole_window(tmp_path: Path) -> None:
    """``p = 1``: every row of every training window is short, so training (losses, per-term
    gradient norms of the statistics steps, raw gradients, weights) is the span-``S``
    configuration's bit for bit, while evaluation reads the whole window: it is the
    whole-window configuration's evaluation of the same weights."""
    steps = 3
    dev = datasets()[1]
    runs: dict[str, tuple[TBPTTTrainer, Any, list[list[Tensor | None]]]] = {}
    for name, extra in (
        ("dropout", ("train.reach_dropout=1.0", f"train.reach_dropout_span={S}")),
        ("span", (f"model.reader_span={S}",)),
    ):
        trainer = _trainer(_trainer_cfg(steps, "train.stats_every=1", *extra), tmp_path / name)
        grads: list[list[Tensor | None]] = []
        trainer.grad_hook = _grads_hook(grads)
        runs[name] = (trainer, trainer.fit(*datasets(), steps), grads)
    (a, history_a, grads_a), (b, history_b, grads_b) = runs["dropout"], runs["span"]
    assert [s.loss for s in history_a.steps] == [s.loss for s in history_b.steps]
    for x, y in zip(history_a.steps, history_b.steps, strict=True):
        norms = {k: v for k, v in x.stats.items() if k.startswith("term_grad_norm/")}
        assert norms and norms == {k: y.stats[k] for k in norms}
    for got, want in zip(grads_a, grads_b, strict=True):
        assert all(
            (g is None and w is None) or (g is not None and w is not None and torch.equal(g, w))
            for g, w in zip(got, want, strict=True)
        )
    assert all(torch.equal(p, q) for p, q in zip(_params(a.model), _params(b.model), strict=True))
    assert a.hsn._reach is None  # cleared after every window
    loss_a, loss_b = a.evaluate(dev)["loss"], b.evaluate(dev)["loss"]
    assert loss_a != loss_b
    whole = _trainer(_trainer_cfg(steps), tmp_path / "whole")
    whole.model.load_state_dict(a.model.state_dict())
    assert whole.evaluate(dev)["loss"] == loss_a


def test_rows_come_from_the_run_seed_and_the_window_counter(tmp_path: Path) -> None:
    steps = 4
    cfg = _trainer_cfg(steps, *ON)

    def run(cfg: RunConfig, name: str) -> tuple[list[Tensor], list[float]]:
        trainer = _trainer(cfg, tmp_path / name)
        seen = _spy_rows(trainer)
        history = trainer.fit(*datasets(), steps)
        return seen, [s.loss for s in history.steps]

    rows, losses = run(cfg, "a")
    again, losses_again = run(cfg, "b")
    assert len(rows) == steps and losses == losses_again
    assert all(torch.equal(x, y) for x, y in zip(rows, again, strict=True))
    stacked = torch.stack(rows)
    assert bool(stacked.any()) and not bool(stacked.all())
    assert len({tuple(r.tolist()) for r in rows}) > 1  # new rows every window
    other, _ = run(_trainer_cfg(steps, *ON, "seed=1"), "seed1")
    assert any(not torch.equal(x, y) for x, y in zip(rows, other, strict=True))
    off, _ = run(_trainer_cfg(steps), "off")  # the default: nothing handed to the model
    assert off == []
    trainer = _trainer(cfg, tmp_path / "rng")
    before = torch.get_rng_state()
    drawn = trainer._reach_rows(4)
    assert drawn is not None and drawn.dtype is torch.bool and drawn.shape == (4,)
    assert torch.equal(torch.get_rng_state(), before)  # the global random stream is untouched


# --------------------------------------------------------------------------------------------------
# The compiled window
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["byte_cread", "word_cread"])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager_and_new_rows_compile_nothing(
    form: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader runs in the eager prologue, so the short rows reach the compiled blocks as
    values of ``c_t`` only: the compiled training window (``aot_eager``) equals the eager one with
    short rows, and other rows run on the same graphs (no recompile)."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        *READER,
        *FORMS[form],
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train()
    window = p28._window(model, p28._ctx(), backend="aot_eager")
    tokens, cx, state, reset = p28._case_inputs(model, True)
    tokens[:, 0::3], tokens[:, 1::3], tokens[:, 2::3] = 97, 120, 32  # a word end every third byte
    tol = p28.TOL[("cpu", "fp32")]
    calls = graphs = seconds = 0
    for index, rows in enumerate((ROWS, (False, True, True))):
        model.set_reach_rows(torch.tensor(rows), 3)
        ref = p28._run(lambda: model(tokens, cx, state, reset))
        if index:
            calls, seconds = window.calls, window.compile_seconds
            graphs = counters["stats"]["unique_graphs"]
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(lambda: window(tokens, cx, state, reset))
            assert window.calls == calls + 1 and window.compile_seconds == seconds
            assert counters["stats"]["unique_graphs"] == graphs
        else:
            got = p28._run(lambda: window(tokens, cx, state, reset))
        model.set_reach_rows(None)
        diff = p28._max_diff(got, ref)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_trainer_equals_eager_without_new_graphs(
    events: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three optimiser steps with ``hardware.compile_mode=reduce-overhead`` (the HSN-T arms'
    mode; ``aot_eager`` on the CPU, where there are no CUDA graphs) and with ``none``, short
    rows on, C-read on, every step a statistics step: the losses and raw gradients agree to the
    compiled-vs-eager tolerance, the rows change from step to step, and steps 2 and 3 compile
    nothing new. ``events``: the loop over events with the exact word address."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    steps = 3
    runs: dict[str, tuple[Any, list[list[Tensor | None]], list[Tensor]]] = {}
    for mode in ("none", "reduce-overhead"):
        cfg = _trainer_cfg(
            steps,
            *ON,
            CREAD,
            "train.stats_every=1",
            f"hardware.compile_mode={mode}",
            *((WORD, "model.word_address=exact") if events else ()),
        )
        trainer = _trainer(cfg, tmp_path / mode)
        rows = _spy_rows(trainer)
        grads: list[list[Tensor | None]] = []
        graphs: list[int] = []
        grad_hook = _grads_hook(grads)

        def hook(t: Any, grad_hook: Any = grad_hook, graphs: list[int] = graphs) -> None:
            grad_hook(t)
            graphs.append(int(counters["stats"]["unique_graphs"]))

        trainer.grad_hook = hook
        history = trainer.fit(*datasets(), steps)
        runs[mode] = (history, grads, rows)
        if mode != "none":
            compiled = trainer.compiled_window
            assert compiled is not None and not compiled.cuda_graphs
            assert history.measured["compiled_windows"] >= steps
            assert graphs[1:] == [graphs[0]] * (steps - 1), graphs
    (eager, eager_grads, eager_rows), (comp, comp_grads, comp_rows) = runs.values()
    assert len(comp_rows) >= steps and len({tuple(r.tolist()) for r in comp_rows}) > 1
    assert all(torch.equal(x, y) for x, y in zip(eager_rows, comp_rows, strict=True))
    for a, b in zip(eager.steps, comp.steps, strict=True):
        assert abs(a.loss - b.loss) <= 1e-5 * max(1.0, abs(a.loss)), (a.loss, b.loss)
    for got, ref in zip(comp_grads, eager_grads, strict=True):
        assert p28._grad_diff(got, ref) <= p28.TOL[("cpu", "fp32")]["grads"]
