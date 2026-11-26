"""``hardware.reader_compile`` (``docs/wcells_speed.md``, :mod:`hsn.train.reader_compile`): the
carried reader's read of a training window of an HSN-T with its memory as one compiled graph.

The key is opt-in: at ``none`` the configuration keeps its hash, its arm and its memory key, the
trainer builds no compiled reader and the compiled window hands the model none. Outside its scope
it is refused with the reason. Set, the trainer's training windows read through the compiled
function (``aot_eager`` on the CPU, as the other compiled-trainer tests) and equal the eager
reader's to rounding, with one window a gradient segment and with two; evaluation and every read
without a gradient stay eager. On the card (``-m cuda``): bf16 steps of the recipe's HSN write
with the fused event block and three windows a segment.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import pytest
import test_wev_fused_trainer as fused_tests
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    canonical_json,
    config_hash,
    load_config,
    reader_compile_scope_error,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, memory_key
from hsn.model.hsn import HSN
from hsn.model.hsn_t import HSNT
from hsn.runner import build_model
from hsn.train import reader_compile as rc
from hsn.train import tbptt
from hsn.train.compile import CompiledWindow, compile_window
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from test_p2_8_compile import _release_compiled_state  # noqa: F401
from test_word_cells import ARM, KEYS, ON, RECIPE, _text_datasets

pytestmark = pytest.mark.usefixtures("restore_determinism")

KEY = "hardware.reader_compile=default"
#: the recipe's CUDA arm as the HSN write trains (the fused block, three windows a segment)
CARD = (
    "profile=rtx3060",
    "hardware.compile_mode=default",
    "hardware.graph_alias_index=true",
    "hardware.message_gradients=state",
    "hardware.round_products=merged",
    "hardware.event_kernel=fused",
    "train.grad_accum=3",
    "train.tbptt_windows=3",
    ON,
    "model.word_cells_value=word",
)


# --------------------------------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_keeps_the_default_hash() -> None:
    base = load_config(RECIPE, list(CARD))
    assert base.hardware.reader_compile == "none"
    assert "hardware.reader_compile" in HASH_OMIT_AT_DEFAULT
    named = load_config(RECIPE, [*CARD, "hardware.reader_compile=none"])
    assert config_hash(named) == config_hash(base)
    assert "reader_compile" not in canonical_json(base)
    compiled = load_config(RECIPE, [*CARD, KEY])
    assert compiled.hardware.reader_compile == "default"
    assert config_hash(compiled) != config_hash(base)
    assert '"reader_compile":"default"' in canonical_json(compiled)
    # nothing else of the configuration moves with the key
    a, b = compiled.model_dump(mode="json"), base.model_dump(mode="json")
    assert a["hardware"].pop("reader_compile") == "default"
    assert b["hardware"].pop("reader_compile") == "none"
    assert a == b
    assert reader_compile_scope_error(compiled) is None


def test_arm_id_and_memory_key_hold_the_key_only_when_set() -> None:
    protocol = {"name": "train_eval", "eval_batch": 32}
    base = load_config(RECIPE, list(CARD))
    named = load_config(RECIPE, [*CARD, "hardware.reader_compile=none"])
    compiled = load_config(RECIPE, [*CARD, KEY])
    assert config_view(named) == config_view(base)
    assert "reader_compile" not in config_view(base)["hardware"]
    assert config_view(compiled)["hardware"]["reader_compile"] == "default"
    assert memory_key(named, "pdg", protocol) == memory_key(base, "pdg", protocol)
    assert memory_key(compiled, "pdg", protocol) != memory_key(base, "pdg", protocol)


REFUSED = [
    (("train.reach_dropout=0.25",), "reach_dropout"),
    (("train.content_dropout=0.1", "train.tbptt_windows=1"), "byte-content"),
    (("model.unseen_bytes_as=zero",), "byte-content"),
]


@pytest.mark.parametrize(("overrides", "reason"), REFUSED)
def test_refused_outside_its_scope(overrides: tuple[str, ...], reason: str) -> None:
    without = [
        line for line in CARD if line.split("=")[0] not in {o.split("=")[0] for o in overrides}
    ]
    assert load_config(RECIPE, [*without, *overrides]).hardware.reader_compile == "none"
    with pytest.raises(ValueError, match="hardware.reader_compile=default is not available") as e:
        load_config(RECIPE, [*without, *overrides, KEY])
    assert reason in str(e.value)


def test_refused_without_the_memory_a_carried_reader_or_a_compiled_window() -> None:
    """An HSN and an HSN-T without its memory have no carried reader in an eager prologue
    (their own compile is ``hardware.baseline_compile``); the key needs the compiled window
    that hands the model its reader, and the autograd window: the scope names each."""
    cfg = load_config(RECIPE, list(CARD))
    for update, reason in (
        ({"model_name": "hsn"}, "model_name='hsn'"),
        ({"model": cfg.model.model_copy(update={"memory": False})}, "model.memory=false"),
        ({"model": cfg.model.model_copy(update={"reader_carry": 0})}, "reader_carry=0"),
        ({"hardware": cfg.hardware.model_copy(update={"compile_mode": "none"})}, "compile_mode"),
        ({"hardware": cfg.hardware.model_copy(update={"window": "split"})}, "window=split"),
        ({"hardware": cfg.hardware.model_copy(update={"device": "mps"})}, "device=mps"),
    ):
        other = cfg.model_copy(update=update)
        assert reason in str(reader_compile_scope_error(other))


# --------------------------------------------------------------------------------------------------
# The compiled reader and the trainer, on the CPU (aot_eager, float64)
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def aot_reader(monkeypatch: pytest.MonkeyPatch, reference_block: None) -> None:
    """The trainer's compiled reader without inductor (``aot_eager``, as its compiled window
    in these tests), beside the fused block on its PyTorch reference."""
    monkeypatch.setattr(tbptt, "reader_of", functools.partial(rc.reader_of, backend="aot_eager"))


reference_block = fused_tests.reference_block


def _small(*extra: str) -> Any:
    keys = (*KEYS, *ARM["word"], *fused_tests.FP64, "hardware.compile_mode=default", *extra)
    return fused_tests._small(*keys)


def test_no_compiled_reader_without_the_key(tmp_path: Path, reference_block: None) -> None:
    cfg = _small()
    model = build_model(cfg)
    assert isinstance(model, HSN)
    ctx = resolve(cfg.hardware)
    assert rc.reader_of(model, ctx, "none") is None
    with pytest.raises(ValueError, match="none or one of"):
        rc.reader_of(model, ctx, "reduce-overhead")
    trainer = TBPTTTrainer(model, cfg, ctx, run_dir=tmp_path)
    window = trainer.compiled_window
    assert trainer._reader_compiled is None
    assert isinstance(window, CompiledWindow) and window.reader is None
    history = trainer.fit(*_text_datasets(), 2)
    assert "reader_compiled_windows" not in history.measured
    assert "read" not in vars(model.reader)  # nothing is installed on the model
    assert isinstance(model, HSNT) and model.reader is not None
    reader = rc.CompiledReader(model.reader, ctx, "default", backend="aot_eager")
    with pytest.raises(ValueError, match="needs a compiled window"):
        compile_window(model, ctx, "none", reader=reader)


def test_the_compiled_read_equals_the_eager_read_and_stays_eager_without_a_gradient() -> None:
    """One window through :class:`CompiledReader`: with a gradient the output, the next
    cache and the gradients of the reader's parameters equal the eager read's; without one,
    and in evaluation, the eager reader runs (no compiled call); short rows are refused."""
    cfg = _small()
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSNT) and model.reader is not None
    reader = model.reader
    compiled = rc.CompiledReader(reader, resolve(cfg.hardware), "default", backend="aot_eager")
    B, T = 3, 12
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 256, (B, T), generator=g)
    reset = torch.rand(B, T, generator=g) < 0.1
    cache = reader.init_cache(B, "cpu")
    assert cache is not None
    _, cache = reader.read(tokens, reset, cache)  # a cache that holds a window
    params = [p for p in reader.parameters() if p.requires_grad]
    outs = []
    for read in (reader.read, compiled.read):
        out, kept = read(tokens, reset.roll(1, 1), cache)
        grads = torch.autograd.grad(out.square().sum(), params, allow_unused=True)
        outs.append((out.detach(), kept, grads))
    assert compiled.calls == 1 and compiled.compile_seconds > 0
    (a, ka, ga), (b, kb, gb) = outs
    assert float((a - b).abs().max()) <= 1e-12
    for x, y in zip(ka, kb, strict=True):
        assert x.dtype == y.dtype and float((x.double() - y.double()).abs().max()) <= 1e-12
        assert not y.requires_grad
    for x, y in zip(ga, gb, strict=True):
        assert (x is None) == (y is None)
        assert x is None or float((x - y).abs().max()) <= 1e-10 * max(1.0, float(x.abs().max()))
    with torch.no_grad():
        out, _ = compiled.read(tokens, reset, cache)
    assert compiled.calls == 1 and torch.equal(out, reader.read(tokens, reset, cache)[0].detach())
    reader.eval()
    compiled.read(tokens, reset, cache)
    assert compiled.calls == 1
    reader.train()
    with pytest.raises(ValueError, match="no short rows"):
        compiled.read(tokens, reset, cache, (torch.ones(B, dtype=torch.bool), 2))
    compiled.read(tokens, reset, cache)
    assert compiled.calls == 2  # the second read of the variant: no compile time added


@pytest.mark.usefixtures("aot_reader")
@pytest.mark.parametrize("segments", [1, 2])
def test_trainer_with_the_key_equals_the_eager_reader(
    segments: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four optimiser steps in float64 on the fused block's reference with the keyed cells'
    HSN write (lazy AdamW, the balance term, a statistics step every second step, an
    evaluation at the end): with the key every training window and every term pass reads
    through the compiled function and the run equals the run without it; the evaluation reads
    eager. With two windows a gradient segment the compiled reads of a segment share one
    backward pass."""
    monkeypatch.setattr(fused_tests, "_long_datasets", _text_datasets)
    accumulate = (f"train.grad_accum={segments}", f"train.tbptt_windows={segments}")
    ref = fused_tests._fit(_small(*accumulate), tmp_path / "eager", 4)
    got = fused_tests._fit(_small(*accumulate, KEY), tmp_path / "compiled", 4)
    trainer = got["trainer"]
    compiled, window = trainer._reader_compiled, trainer.compiled_window
    assert isinstance(compiled, rc.CompiledReader) and isinstance(window, CompiledWindow)
    assert window.reader is compiled and ref["trainer"]._reader_compiled is None
    fused_tests._assert_same_run(got, ref, loss=1e-9, grad=1e-8, param=1e-9, stats=1e-8)
    measured = got["history"].measured
    # the training windows and the two statistics steps' term passes (three terms each)
    assert measured["reader_compiled_windows"] == compiled.calls >= 4 * segments + 2
    assert measured["reader_compile_seconds"] == compiled.compile_seconds > 0
    assert "reader_compiled_windows" not in ref["history"].measured
    calls = compiled.calls
    trainer.evaluate(_text_datasets()[1])
    assert compiled.calls == calls  # evaluation keeps the eager reader
    assert "read" not in vars(trainer.model.reader)


# --------------------------------------------------------------------------------------------------
# The card
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
def test_bf16_steps_of_the_hsn_write_with_the_key_cuda(tmp_path: Path) -> None:
    """hsn_t_memory_words_carry_bal_qk.yaml's CUDA arm as the HSN write trains (the fused
    event block, three windows a gradient segment, bf16) with the reader compiled by
    inductor: four steps with finite losses within 5 % of the same steps without the key
    (the two round differently), every training window read compiled, one compile."""
    import test_p2_8_compile as p28
    from cards import CUDA_PROFILE
    from hsn.kernels import window as window_kernels
    from test_word_cells import CARD_STEPS, QUIET

    p28._cuda_ctx()
    if not window_kernels.available():
        pytest.skip("needs the window kernels (scripts/build_window_kernels.py)")
    card = [f"profile={CUDA_PROFILE}", *CARD[1:], *CARD_STEPS, *QUIET]
    runs = {}
    for name, extra in (("eager", []), ("compiled", [KEY])):
        cfg = load_config(RECIPE, [*card, *extra])
        seed_everything(0)
        model = build_model(cfg)
        trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / name)
        runs[name] = (trainer, trainer.fit(*_text_datasets(64, 900), cfg.train.steps))
    (_, ref), (trainer, got) = runs["eager"], runs["compiled"]
    assert len(got.steps) == len(ref.steps) == 4
    for a, b in zip(got.steps, ref.steps, strict=True):
        assert math.isfinite(a.loss) and abs(a.loss - b.loss) <= 0.05 * abs(b.loss), (a.step,)
    compiled = trainer._reader_compiled
    assert compiled is not None and compiled.calls >= 4 * 3
    assert len(compiled._variants) == 1 and compiled.compile_seconds > 0
    assert got.measured["reader_compiled_windows"] == compiled.calls
    assert runs["eager"][0]._reader_compiled is None
    fused = trainer._fused_events
    assert fused is not None and fused.calls == 4 * 3
