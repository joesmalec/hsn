"""The byte-content options (``docs/key_content.md``): ``train.content_dropout`` and
``model.zero_unseen_bytes``, for the HSN and for HSN-T.

Both act on what the encoder ``E`` and HSN-T's reader read of a byte: at some positions a zero
row in place of the byte's row of the two content tables. The tests hold them to that: the
defaults are the model of f3f271a, the blanked positions are exactly the drawn ones (training
only) and exactly the bytes no training window held, nothing else of the window changes, and
the trainer, the compiled window and the split window run with them.

``train.content_dropout_target = unknown`` and ``model.unseen_bytes_as`` (the last section):
the same positions read a learned unknown row of each table instead of zero; the rows are
drawn like their tables from generators of their own, get the gradient of exactly the
positions that read them, and train, resume and run in the compiled and the split window.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, RunConfig, canonical_json, config_hash, load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import arm_identity, memory_key
from hsn.model.embeddings import init_weight_, replace_content
from hsn.model.hsn import HSN
from hsn.runner import build_model, load_datasets, warm_start
from hsn.train import tbptt
from hsn.train.checkpoint import load_checkpoint
from hsn.train.compile import compile_window
from hsn.train.split import SplitWindow
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5, ALL, _cfg, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
NO_MEMORY = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"

DROP = "train.content_dropout=0.3"
ZERO = "model.zero_unseen_bytes=true"
#: the learned unknown rows: dropout to them, and the never-trained bytes read them too
TO_UNKNOWN = "train.content_dropout_target=unknown"
UNSEEN_UNKNOWN = "model.unseen_bytes_as=unknown"
#: the two option sets of the windows below: zero rows, and the learned unknown rows
BLANKS: dict[str, tuple[str, ...]] = {
    "zero": (DROP, ZERO),
    "unknown": (DROP, TO_UNKNOWN, UNSEEN_UNKNOWN),
}
#: the learned unknown rows' parameter names
UNKNOWN_ROWS = ("byte_emb.unknown", "reader.emb_unknown")
#: the candidate's cut: the cell's read of the state keeps its gradient
CUT_C = "train.feedback_detach=[summary,eps,router]"
#: the reader of the six-cell model of the A6 tests (d = 8), short-sighted
READER = (
    "model_name=hsn_t",
    "model.reader=transformer",
    "model.reader_width=8",
    "model.reader_heads=2",
    "model.reader_span=4",
)
#: the memory core alone (the control runs) and behind the reader
FAMILIES: dict[str, tuple[str, ...]] = {"hsn": (*A5, CUT_C), "hsn_t": (*A5, CUT_C, *READER)}

#: The 32-byte reader on main and the memory core alone, as the hold-out arms set them.
RD32 = (
    "model_name=hsn_t",
    "model.reader=transformer",
    "model.reader_span=32",
    "model.k_select=12",
    "model.n_cells=64",
    "model.address_gate=true",
    "model.address_gate_input=summary",
    "model.address_ngram=2",
    "model.address_history=ngram",
    "model.scorer_position=wrap",
    "train.feedback_detach=[summary,eps,router]",
    "train.unit_credit=executed",
    "train.soft_until_step=0",
)
CORE = tuple(o for o in RD32 if "reader" not in o and o != "model_name=hsn_t")
PROTOCOL = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
#: (configuration hash, memory key, arm id) of ``hsn_recipe_a6.yaml`` with these overrides, as
#: the code of f3f271a (the commit before the keys) computes them.
F3F271A = {
    (): (
        "d8b36a25971c9694b25251b4a6f20545c617e6d2796b7e4d1ac544acc3e78e28",
        "ed70cf96715a05ef",
        "c42f0b97c13e4c55",
    ),
    CORE: (
        "cf7d38877c55a51cf11bb9305458fd9188aad4bac7f8f15d6d7676da04309116",
        "76516e058e93db17",
        "86dd2e7408f95143",
    ),
    RD32: (
        "6c3b4fcd943f012cfc21967ebdf915a7bae6f6b1622bd84b0e7f7bd7a76d2d20",
        "14c19ee8511588e7",
        "e49978fd43c88a9a",
    ),
}


def _arm_id(cfg: RunConfig) -> str:
    return arm_identity(
        cfg,
        family=cfg.model_name,
        variant="pdg",
        variant_params={},
        protocol={**PROTOCOL, "version": 1},
        early_stop=[],
        data="x",
    )[0]


def _build(cfg: RunConfig, dtype: torch.dtype = torch.float64) -> HSN:
    """The model of ``cfg`` away from its symmetric start, in training mode. Two
    configurations that differ in the options only give the same weights."""
    seed_everything(0)
    model = build_model(cfg).to(dtype)
    assert isinstance(model, HSN)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    return model.train()


class _Reads:
    """What ``E`` and the reader's first block read in the model's last window, and with
    ``blank`` set the same reads with those positions zeroed from outside: the reference a
    model with an option on is compared with."""

    def __init__(self, model: HSN, blank: Tensor | None = None) -> None:
        self.encoder: Tensor | None = None
        self.reader: Tensor | None = None
        self.blank = blank
        self._handles = [model.encoder.register_forward_pre_hook(self._encoder)]
        if model.reader is not None:
            self._handles.append(model.reader.blocks[0].register_forward_pre_hook(self._reader))

    def _zeroed(self, x: Tensor) -> Tensor:
        return x if self.blank is None else x.masked_fill(self.blank[..., None], 0.0)

    def _encoder(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = self._zeroed(args[0])
        self.encoder = x.detach().clone()
        return (x, *args[1:])

    def _reader(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = self._zeroed(args[0])
        self.reader = x.detach().clone()
        return (x, *args[1:])

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


@contextmanager
def _reads(model: HSN, blank: Tensor | None = None) -> Iterator[_Reads]:
    reads = _Reads(model, blank)
    try:
        yield reads
    finally:
        reads.close()


def _assert_blanked(
    model: HSN, reads: _Reads, tokens: Tensor, blank: Tensor, *, unknown: bool = False
) -> None:
    """``E`` and the reader read the tables' rows, zero exactly where ``blank`` is True (with
    ``unknown``: their learned unknown row there)."""
    assert 0 < int(blank.sum()) < blank.numel()  # the case holds both kinds of position
    byte_row, reader_row = model.unknown_rows()
    tables = [(reads.encoder, model.byte_emb(tokens).detach(), byte_row)]
    if model.reader is not None:
        tables.append((reads.reader, model.reader.emb(tokens).detach(), reader_row))
    for read, rows, row in tables:
        assert read is not None
        if unknown:
            assert row is not None
            assert torch.equal(read[blank], row.detach().expand(int(blank.sum()), -1))
        else:
            assert bool((read[blank] == 0.0).all())
        assert torch.equal(read[~blank], rows[~blank]) and bool((rows[blank] != 0.0).any())


def _same(a: Any, b: Any) -> bool:
    """Two state-dict values are equal bit for bit (a value may be a mapping of tensors)."""
    if isinstance(a, Tensor):
        return isinstance(b, Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


def _assert_same_state(model: HSN, keyed: HSN) -> None:
    """``keyed`` holds ``byte_seen`` and otherwise the tensors of ``model``, bit for bit."""
    state, other = model.state_dict(), keyed.state_dict()
    assert set(other) - set(state) == {"byte_seen"} and set(state) <= set(other)
    assert all(_same(other[key], value) for key, value in state.items())


def _assert_same_window(got: p28.Window, ref: p28.Window) -> None:
    """Two windows are the same bit for bit: logits, state and every entry of ``info``."""
    (logits, final, info), (logits_r, final_r, info_r) = got, ref
    assert torch.equal(logits, logits_r)
    for name in ("h", "q", "age", "pos"):
        assert torch.equal(getattr(final, name), getattr(final_r, name)), name
    assert set(info) == set(info_r)
    for key, value in info_r.items():
        assert torch.equal(info[key], value), key


def _assert_same_gradients(model: HSN, got: p28.Window, other: HSN, ref: p28.Window) -> None:
    live = 0
    grads, grads_r = p28._grads(model, got), p28._grads(other, ref)
    for g, r in zip(grads, grads_r, strict=True):
        assert (g is None) == (r is None)
        if g is not None and r is not None:
            assert torch.equal(g, r)
            live += int(bool(g.abs().max() > 0))
    assert live >= 20  # real gradients, not zeros


# --------------------------------------------------------------------------------------------------
# Configuration: opt-in, and the defaults are f3f271a's
# --------------------------------------------------------------------------------------------------


def test_keys_are_opt_in_and_the_defaults_keep_every_identity() -> None:
    base = load_config(RECIPE_A6)
    assert base.train.content_dropout == 0.0 and base.model.zero_unseen_bytes is False
    dump = json.loads(canonical_json(base))
    assert "content_dropout" not in dump["train"] and "zero_unseen_bytes" not in dump["model"]
    assert base.train.content_dropout_target == "zero" and base.model.unseen_bytes_as == "own"
    assert "content_dropout_target" not in dump["train"] and "unseen_bytes_as" not in dump["model"]
    assert {
        "train.content_dropout",
        "model.zero_unseen_bytes",
        "train.content_dropout_target",
        "model.unseen_bytes_as",
    } <= set(HASH_OMIT_AT_DEFAULT)
    for overrides, (hashed, memory, arm) in F3F271A.items():
        cfg = load_config(RECIPE_A6, list(overrides))
        named = load_config(
            RECIPE_A6,
            [
                *overrides,
                "train.content_dropout=0.0",
                "model.zero_unseen_bytes=false",
                "train.content_dropout_target=zero",
                "model.unseen_bytes_as=own",
            ],
        )
        for config in (cfg, named):
            assert config_hash(config) == hashed
            assert memory_key(config, "pdg", PROTOCOL) == memory
            assert _arm_id(config) == arm
    # on: each key is hashed and is in the arm id; only the model key is in the memory key
    plain = load_config(RECIPE_A6, list(RD32))
    drop = load_config(RECIPE_A6, [*RD32, "train.content_dropout=0.1"])
    zero = load_config(RECIPE_A6, [*RD32, ZERO])
    both = load_config(RECIPE_A6, [*RD32, "train.content_dropout=0.1", ZERO])
    assert len({config_hash(c) for c in (plain, drop, zero, both)}) == 4
    assert len({_arm_id(c) for c in (plain, drop, zero, both)}) == 4
    assert memory_key(drop, "pdg", PROTOCOL) == memory_key(plain, "pdg", PROTOCOL)
    assert memory_key(zero, "pdg", PROTOCOL) != memory_key(plain, "pdg", PROTOCOL)
    assert json.loads(canonical_json(both))["train"]["content_dropout"] == 0.1
    assert json.loads(canonical_json(both))["model"]["zero_unseen_bytes"] is True


@pytest.mark.parametrize("family", list(FAMILIES))
def test_defaults_build_the_model_without_the_options(family: str) -> None:
    """No buffer, no state-dict key, no mask: the model of the commit before the keys."""
    off = _build(_cfg(*FAMILIES[family]))
    assert off.content_dropout == 0.0 and off.byte_seen is None
    assert "byte_seen" not in off.state_dict() and not list(off.buffers(recurse=False))
    tokens, _, _ = _window(_cfg(*FAMILIES[family]))
    assert off._content_masks(tokens) == (None, None)  # training: nothing is drawn or computed
    off.mark_bytes_seen(tokens)  # a no-op with the key off
    assert off.byte_seen is None
    # the options leave the initialisation alone: every tensor the models share starts equal
    on = _build(_cfg(*FAMILIES[family], DROP, ZERO))
    _assert_same_state(off, on)
    assert on.content_dropout == 0.3 and on.byte_seen is not None
    assert on.byte_seen.dtype is torch.bool and not bool(on.byte_seen.any())
    assert tuple(on.byte_seen.shape) == (on.cfg.vocab_size,)
    assert "content_dropout=0.3" in on.extra_repr() and "zero_unseen_bytes" in on.extra_repr()
    assert "content_dropout" not in off.extra_repr() and "unseen" not in off.extra_repr()


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (GRU_2M, ["train.content_dropout=0.1"], "generic trainer"),
        (GRU_2M, [ZERO], "generic trainer"),
        (READER_LM, ["train.content_dropout=0.1"], "generic trainer"),
        (READER_LM, [ZERO], "generic trainer"),
        (NO_MEMORY, ["train.content_dropout=0.1"], "model.memory=false"),
        (NO_MEMORY, [ZERO], "model.memory=false"),
        (RECIPE_A6, ["train.content_dropout=1.0"], "less than 1"),
        (RECIPE_A6, ["train.content_dropout=-0.1"], "greater than or equal to 0"),
        (RECIPE_A6, ["train.content_dropout=nan"], "content_dropout"),
    ],
)
def test_configuration_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_fused_window_and_probes_are_refused_and_the_split_window_is_not() -> None:
    recipe = (*A5, ALL, "model.cand_norm=rms")
    for option in (DROP, ZERO):
        with pytest.raises(ValueError, match="hardware.window=fused is untested"):
            _cfg(*recipe, "hardware.window=fused", option)
        assert _cfg(*recipe, "hardware.window=split", option).hardware.window == "split"
    assert _cfg(*recipe, "hardware.window=fused").hardware.window == "fused"
    probes = ("train.lambda_u=0.1", "train.probe_horizon=4")
    with pytest.raises(ValueError, match="counterfactual probes"):
        _cfg(*A5, *probes, DROP)
    assert _cfg(*A5, *probes, ZERO).model.zero_unseen_bytes


def test_model_and_trainer_refuse_a_value_that_is_not_the_configurations(tmp_path: Path) -> None:
    cfg = hsn_config(4, *A5)
    for bad in (-0.1, 1.0, True, "0.1"):
        with pytest.raises(ValueError, match="content_dropout"):
            HSN(cfg.model, seed=0, unroll=cfg.train.unroll, content_dropout=bad)  # type: ignore[arg-type]
    ctx = resolve(cfg.hardware)
    with pytest.raises(ValueError, match="train.content_dropout"):
        TBPTTTrainer(build_model(hsn_config(4, *A5, DROP)), cfg, ctx, run_dir=tmp_path)
    with pytest.raises(ValueError, match="model.zero_unseen_bytes"):
        TBPTTTrainer(build_model(hsn_config(4, *A5, ZERO)), cfg, ctx, run_dir=tmp_path)


# --------------------------------------------------------------------------------------------------
# (a) train.content_dropout: exactly the drawn positions, in training only
# --------------------------------------------------------------------------------------------------


def test_drawn_positions_are_a_function_of_seed_step_and_shape() -> None:
    cfg = _cfg(*FAMILIES["hsn"], "train.content_dropout=0.1")
    model = build_model(cfg)
    assert isinstance(model, HSN)
    model.set_train_step(5)
    mask = model.content_drop_mask(64, 256)
    assert mask.dtype is torch.bool and tuple(mask.shape) == (64, 256)
    assert mask.device.type == "cpu"
    assert abs(float(mask.float().mean()) - 0.1) < 0.01  # 16,384 positions: sd 0.0023
    assert torch.equal(model.content_drop_mask(64, 256), mask)  # no state: the same again
    # a window cut at its trailing padding (P1.10d) draws the uncut window's positions
    for cut in (1, 7, 100, 255):
        assert torch.equal(model.content_drop_mask(64, cut), mask[:, :cut])
    model.set_train_step(6)
    later = model.content_drop_mask(64, 256)
    assert not torch.equal(later, mask)
    model.set_train_step(5)
    assert torch.equal(model.content_drop_mask(64, 256), mask)  # a replayed step
    other = build_model(_cfg(*FAMILIES["hsn"], "train.content_dropout=0.1", "seed=1"))
    assert isinstance(other, HSN)
    other.set_train_step(5)
    assert not torch.equal(other.content_drop_mask(64, 256), mask)
    # a second model of the same seed draws the same positions, whatever the global stream did
    torch.manual_seed(99)
    torch.rand(17)
    twin = build_model(cfg)
    assert isinstance(twin, HSN)
    twin.set_train_step(5)
    assert torch.equal(twin.content_drop_mask(64, 256), mask)
    # the draw does not read the global stream
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    model.content_drop_mask(64, 256)
    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_content_dropout_blanks_exactly_the_drawn_positions_in_training(family: str) -> None:
    cfg = _cfg(*FAMILIES[family], DROP)
    model = _build(cfg)
    model.set_train_step(7)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    mask = model.content_drop_mask(B, T)
    state = _start(model, B)
    with _reads(model) as reads:
        stream = torch.get_rng_state()
        _, _, info = model(tokens, ctx, state, reset)
        assert torch.equal(torch.get_rng_state(), stream)  # the global stream is not read
        _assert_blanked(model, reads, tokens, mask)
        # the prediction loss keeps its target: the byte's own row at every position
        assert torch.equal(info["z_target"], model.byte_emb(tokens).detach())
        # another step, other positions
        model.set_train_step(8)
        model(tokens, ctx, state, reset)
        other = model.content_drop_mask(B, T)
        assert not torch.equal(other, mask)
        _assert_blanked(model, reads, tokens, other)
        # evaluation: every row is read
        model.eval()
        with torch.no_grad():
            model(tokens, ctx, state, reset)
        assert reads.encoder is not None
        assert torch.equal(reads.encoder, model.byte_emb(tokens).detach())
        if model.reader is not None:
            assert reads.reader is not None
            assert torch.equal(reads.reader, model.reader.emb(tokens).detach())


@pytest.mark.parametrize("family", list(FAMILIES))
def test_content_dropout_changes_nothing_but_those_reads(family: str) -> None:
    """The training window with the key on is, bit for bit and in every gradient, the window
    of the model without it whose two reads are zeroed at the drawn positions from outside:
    the address features, the context, the cells, the router and the decoder are computed from
    the same bytes. In evaluation the two models are the same without any change."""
    cfg, cfg_off = _cfg(*FAMILIES[family], DROP), _cfg(*FAMILIES[family])
    on, off = _build(cfg), _build(cfg_off)
    on.set_train_step(3)
    off.set_train_step(3)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    mask = on.content_drop_mask(B, T)
    got = on(tokens, ctx, _start(on, B), reset)
    with _reads(off, mask):
        ref = off(tokens, ctx, _start(off, B), reset)
    _assert_same_window(got, ref)
    _assert_same_gradients(on, got, off, ref)
    plain = off(tokens, ctx, _start(off, B), reset)
    assert not torch.equal(plain[0], got[0])  # the positions matter to this window
    on.eval()
    off.eval()
    with torch.no_grad():
        _assert_same_window(
            on(tokens, ctx, _start(on, B), reset), off(tokens, ctx, _start(off, B), reset)
        )


def test_content_dropout_refuses_a_single_training_position() -> None:
    cfg = _cfg(*FAMILIES["hsn"], DROP)
    on, off = _build(cfg), _build(_cfg(*FAMILIES["hsn"]))
    tokens, ctx, _ = _window(cfg)
    with pytest.raises(ValueError, match="draws its positions for a window"):
        on.step(tokens[:, 0], ctx[:, 0], _start(on, tokens.shape[0]))
    on.eval()
    off.eval()
    with torch.no_grad():
        got = on.step(tokens[:, 0], ctx[:, 0], _start(on, tokens.shape[0]))
        ref = off.step(tokens[:, 0], ctx[:, 0], _start(off, tokens.shape[0]))
    assert torch.equal(got.logits, ref.logits) and torch.equal(got.state.h, ref.state.h)


# --------------------------------------------------------------------------------------------------
# (b) model.zero_unseen_bytes: exactly the bytes no training window held
# --------------------------------------------------------------------------------------------------


def _half_seen(model: HSN, tokens: Tensor) -> Tensor:
    """Mark every second byte value of ``tokens`` as seen; the positions that stay unseen."""
    seen = tokens.unique()[::2]
    model.mark_bytes_seen(seen)
    assert model.byte_seen is not None
    assert model.byte_seen.nonzero().flatten().tolist() == seen.tolist()
    return ~torch.isin(tokens, seen)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_zero_unseen_bytes_blanks_exactly_the_unseen_bytes(family: str, training: bool) -> None:
    cfg, cfg_off = _cfg(*FAMILIES[family], ZERO), _cfg(*FAMILIES[family])
    on, off = _build(cfg).train(training), _build(cfg_off).train(training)
    tokens, ctx, reset = _window(cfg)
    B = tokens.shape[0]
    # a new model has seen nothing: every position reads a zero row
    with _reads(on) as reads, torch.no_grad():
        on(tokens, ctx, _start(on, B), reset)
        assert reads.encoder is not None and not bool(reads.encoder.any())
    unseen = _half_seen(on, tokens)
    assert on.byte_seen is not None
    marked = on.byte_seen.clone()
    with torch.set_grad_enabled(training):
        with _reads(on) as reads:
            got = on(tokens, ctx, _start(on, B), reset)
            _assert_blanked(on, reads, tokens, unseen)
        assert torch.equal(on.byte_seen, marked)  # a window marks nothing, in either mode
        # nothing else changes: the model without the key, its two reads zeroed from outside
        with _reads(off, unseen):
            ref = off(tokens, ctx, _start(off, B), reset)
        _assert_same_window(got, ref)
        if training:
            _assert_same_gradients(on, got, off, ref)
        # with every byte of the window seen the key changes nothing at all
        on.mark_bytes_seen(tokens)
        got = on(tokens, ctx, _start(on, B), reset)
        ref = off(tokens, ctx, _start(off, B), reset)
        _assert_same_window(got, ref)
        if training:
            _assert_same_gradients(on, got, off, ref)


def test_zero_unseen_bytes_in_a_single_position() -> None:
    """:meth:`HSN.step` (the chat session, the serving step) reads the same zero rows."""
    cfg = _cfg(*FAMILIES["hsn"], ZERO)
    on, off = _build(cfg).eval(), _build(_cfg(*FAMILIES["hsn"])).eval()
    tokens, ctx, _ = _window(cfg)
    x, c = tokens[:, 0], ctx[:, 0]
    on.mark_bytes_seen(x[:1])
    unseen = x != x[0]
    assert bool(unseen.any()) and not bool(unseen.all())
    with torch.no_grad():
        got = on.step(x, c, _start(on, x.shape[0]))
        with _reads(off, unseen):
            ref = off.step(x, c, _start(off, x.shape[0]))
        plain = off.step(x, c, _start(off, x.shape[0]))
    assert torch.equal(got.logits, ref.logits) and torch.equal(got.state.h, ref.state.h)
    assert torch.equal(got.z_target, plain.z_target)  # the target keeps the byte's row
    assert not torch.equal(got.logits, plain.logits)


def test_both_options_blank_the_union() -> None:
    cfg = _cfg(*FAMILIES["hsn_t"], DROP, ZERO)
    model = _build(cfg)
    model.set_train_step(2)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    unseen = _half_seen(model, tokens)
    drop = model.content_drop_mask(B, T)
    assert bool((drop & ~unseen).any()) and bool((unseen & ~drop).any())
    with _reads(model) as reads:
        model(tokens, ctx, _start(model, B), reset)
        _assert_blanked(model, reads, tokens, unseen | drop)
        model.eval()
        with torch.no_grad():
            model(tokens, ctx, _start(model, B), reset)
        _assert_blanked(model, reads, tokens, unseen)


@pytest.mark.parametrize("blanks", list(BLANKS))
def test_window_under_bf16_autocast_with_both_options(blanks: str) -> None:
    """The card's precision (float32 masters under bf16 autocast): the rows are blanked where
    the tables are read, before any cast, and the window and its gradients are finite."""
    cfg = _cfg(*FAMILIES["hsn_t"], "train.gate_mode=ste", *BLANKS[blanks])
    model = _build(cfg, torch.float32)
    model.set_train_step(2)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    blank = _half_seen(model, tokens) | model.content_drop_mask(B, T)
    state = _start(model, B)
    with _reads(model) as reads, torch.autocast("cpu", dtype=torch.bfloat16):
        logits, new_state, _ = model(tokens, ctx, state, reset)
        _assert_blanked(model, reads, tokens, blank, unknown=blanks == "unknown")
    assert bool(torch.isfinite(logits).all()) and new_state.h.dtype == torch.float32
    logits.float().square().mean().backward()
    assert model.reader is not None
    for table in (model.byte_emb.weight, model.reader.emb.weight):
        assert table.grad is not None and bool(torch.isfinite(table.grad).all())
        # a row is trained only where a position read it: an unseen byte's row gets nothing
        read = torch.zeros(model.cfg.vocab_size, dtype=torch.bool)
        read[tokens[~blank]] = True
        assert not bool(table.grad[~read].any()) and bool(table.grad[read].any())
    for row in model.unknown_rows():  # (None, None) with zero rows
        assert (row is None) == (blanks == "zero")
        if row is not None:
            assert row.grad is not None and row.grad.dtype == torch.float32
            assert bool(torch.isfinite(row.grad).all()) and bool(row.grad.any())


def test_seen_bytes_are_saved_with_the_model_only_with_the_key_on() -> None:
    cfg = _cfg(*FAMILIES["hsn_t"], ZERO)
    on, off = _build(cfg), _build(_cfg(*FAMILIES["hsn_t"]))
    on.mark_bytes_seen(torch.tensor([[3, 200], [3, 77]]))
    assert on.byte_seen is not None
    assert on.byte_seen.nonzero().flatten().tolist() == [3, 77, 200]
    twin = _build(cfg)
    twin.load_state_dict(on.state_dict())  # strict
    assert twin.byte_seen is not None and torch.equal(twin.byte_seen, on.byte_seen)
    assert twin.float().byte_seen is not None and twin.byte_seen.dtype is torch.bool
    # the two kinds of checkpoint do not load into each other
    with pytest.raises(RuntimeError, match="byte_seen"):
        twin.load_state_dict(off.state_dict())
    with pytest.raises(RuntimeError, match="byte_seen"):
        off.load_state_dict(on.state_dict())


# --------------------------------------------------------------------------------------------------
# The compiled window and the split window
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "blanks"), [("hsn", "zero"), ("hsn_t", "zero"), ("hsn_t", "unknown")]
)
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager_with_both_options(family: str, blanks: str) -> None:
    """The options act in the window's prologue, which the compiled window runs eagerly: the
    compiled training window and the compiled evaluation window equal the eager ones."""
    reader = READER if family == "hsn_t" else ()
    unknown = blanks == "unknown"
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        *reader,
        *BLANKS[blanks],
    )
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSN)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=True, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train()
    model.set_train_step(4)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    unseen = _half_seen(model, tokens)
    blank = unseen | model.content_drop_mask(*tokens.shape)
    tol = p28.TOL[(ctx.device.type, "fp32")]
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    with _reads(model) as reads:
        got = p28._run(lambda: window(tokens, cx, state, reset))
        _assert_blanked(model, reads, tokens, blank, unknown=unknown)
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    grads, grads_ref = p28._grads(model, got), p28._grads(model, ref)
    assert p28._grad_diff(grads, grads_ref) <= tol["grads"]
    if unknown:  # the unknown rows are trained in the compiled window too
        named = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in UNKNOWN_ROWS if reader else UNKNOWN_ROWS[:1]:
            grad = grads[named.index(name)]
            assert grad is not None and bool(grad.any())
    model.eval()
    with torch.no_grad():
        ref = p28._run(lambda: model(tokens, cx, state, reset))
        with _reads(model) as reads:
            got = p28._run(lambda: window(tokens, cx, state, reset))
            _assert_blanked(model, reads, tokens, unseen, unknown=unknown)
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff


def _terms(logits: Tensor, info: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
    """The gradient-carrying outputs of a window, one term each (the terms of the split
    window's own tests, ``tests/test_p2_10_split_window.py``)."""
    return {
        "task": logits[mask].square().sum(),
        "pred": (info["z_hat"] - info["z_target"]).square().mean()
        + 0.1 * info["z"].square().mean(),
        "cost": info["expected_cost"].square().sum(),
        "balance": info["unit_logits"].square().mean(),
    }


@pytest.mark.parametrize("blanks", list(BLANKS))
def test_split_window_equals_the_autograd_window_with_both_options(blanks: str) -> None:
    """``hardware.window=split`` (the HSN's): the sweep and the replay start from the same
    prologue, so the split window's values and gradients are the autograd window's.

    The values agree to float64 rounding. The gradients are compared term by term, each
    tensor to 1e-6 of its largest entry: the router's cost path is float32 in both windows
    (``expected_cost``), the scorer's gradient is rounded there, and the two windows add it
    up in different orders. Measured on 15 Nov 2026 with this model, with the options and
    with the same reads zeroed from outside in the model without them: gaps up to 2e-9 of
    the largest entry in one term and 7e-9 in a summed objective, the largest of them in
    the scorer's linear layer. A window that blanked other positions than the other would
    differ by its whole size. With the unknown rows, their gradients are among those
    compared."""
    cfg = _cfg(*A5, ALL, "model.cand_norm=rms", *BLANKS[blanks])
    model = _build(cfg)
    model.set_train_step(1)
    tokens, ctx, reset = _window(cfg)
    reset[0, 0] = False  # row 0 carries its start state into the window
    reset[1, 5] = True
    B, T = tokens.shape
    blank = _half_seen(model, tokens) | model.content_drop_mask(B, T)
    state = _start(model, B)
    mask = torch.zeros_like(reset)
    mask[:, 3::4] = True
    params = [p for _, p in model.named_parameters()]
    names = [n for n, _ in model.named_parameters()]
    assert ("byte_emb.unknown" in names) == (blanks == "unknown")
    logits, ref_state, ref_info = model(tokens, ctx, state, reset)
    with _reads(model) as reads:
        s_logits, s_state, s_info = SplitWindow(model, cfg, None)(tokens, ctx, state, reset)
        _assert_blanked(model, reads, tokens, blank, unknown=blanks == "unknown")
    torch.testing.assert_close(s_logits, logits, rtol=1e-10, atol=1e-12)
    for field in ("h", "q"):
        torch.testing.assert_close(
            getattr(s_state, field), getattr(ref_state, field), rtol=1e-10, atol=1e-12
        )
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])
    ref_terms, got_terms = _terms(logits, ref_info, mask), _terms(s_logits, s_info, mask)
    live, row_live = 0, 0
    for name, term in ref_terms.items():
        ref = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        got = torch.autograd.grad(got_terms[name], params, retain_graph=True, allow_unused=True)
        for param, want, have in zip(names, ref, got, strict=True):
            assert (want is None) == (have is None), name
            if want is not None and have is not None:
                largest = float(want.abs().max())
                assert float((have - want).abs().max()) <= 1e-6 * largest + 1e-12, name
                live += int(largest > 0)
                row_live += int(param in UNKNOWN_ROWS and largest > 0)
    assert live >= 20  # real gradients, not zeros
    assert (row_live > 0) == (blanks == "unknown")


def test_sweep_twin_holds_the_seen_bytes_and_sweeps_the_same_tape() -> None:
    """The split window's sweep on the card runs a twin of the model
    (:class:`hsn.train.compile.SweepTwin`) that holds the model's buffers by reference:
    ``byte_seen``, at the model's top level, among them. The twin's sweep of a window with
    blank positions records the model's own tape (float64, CPU, uncompiled)."""
    from hsn.train.compile import TAPE_KEYS, SweepTwin, _tape_block, run_tape

    cfg = _cfg(*A5, ALL, "model.cand_norm=rms", DROP, ZERO)
    model = _build(cfg)
    model.set_train_step(1)
    sweep = SweepTwin(model, torch.float64)
    assert sweep.twin.byte_seen is model.byte_seen
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    blank = _half_seen(model, tokens) | model.content_drop_mask(B, T)
    assert sweep.twin.byte_seen is model.byte_seen and bool(sweep.twin.byte_seen.any())
    state = _start(model, B)
    with _reads(model) as reads:
        pro = model._window_prologue(
            tokens, ctx, state, reset, model._window_mode(), reset_first=True
        )
        _assert_blanked(model, reads, tokens, blank)
    with torch.no_grad():
        plain = run_tape(_tape_block, model, pro, state, reset, T)
        twinned = run_tape(_tape_block, model, pro, state, reset, T, twin=sweep)
    for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
        if plain[key].is_floating_point():
            torch.testing.assert_close(twinned[key], plain[key], rtol=1e-10, atol=1e-12)
        else:
            assert torch.equal(twinned[key], plain[key]), key


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------

#: the small A6 recipe of the trainer tests with the candidate's cut (tests/test_hsn_t.py)
TRAINED = (
    *A5,
    CUT_C,
    "model.cand_norm=rms",
    "model.k_select=3",
    "train.soft_until_step=0",
    "train.gate_mode=ste",
    "train.lr=3e-3",
)
TRAINED_FAMILIES: dict[str, tuple[str, ...]] = {"hsn": TRAINED, "hsn_t": (*TRAINED, *READER)}


def _trainer(cfg: RunConfig, run_dir: Path) -> TBPTTTrainer:
    seed_everything(0)
    return TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)


@pytest.mark.parametrize(
    "options",
    [
        ("train.content_dropout=0.1",),
        (ZERO,),
        ("train.content_dropout=0.1", ZERO),
        ("train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_UNKNOWN),
    ],
)
@pytest.mark.parametrize("family", list(TRAINED_FAMILIES))
def test_recipe_with_the_options_trains(
    family: str, options: tuple[str, ...], tmp_path: Path
) -> None:
    """Thirty TBPTT steps on the tiny CPU configuration of the trainer tests: finite losses
    that fall, and a final evaluation; the unknown rows are trained."""
    steps = 30
    cfg = hsn_config(steps, *TRAINED_FAMILIES[family], *options)
    trainer = _trainer(cfg, tmp_path / "run")
    start = [None if r is None else r.detach().clone() for r in trainer.hsn.unknown_rows()]
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    assert history.evals and math.isfinite(history.evals[-1]["loss"])
    rows = trainer.hsn.unknown_rows()
    expected = 2 if TO_UNKNOWN in options and family == "hsn_t" else int(TO_UNKNOWN in options)
    assert sum(r is not None for r in rows) == expected
    for row, before in zip(rows, start, strict=True):
        if row is not None and before is not None:
            assert bool(torch.isfinite(row).all()) and not torch.equal(row.detach(), before)


def _log_blank_rows(model: HSN) -> list[tuple[int, Tensor]]:
    """Record, for every training window the model runs from now on, its optimiser step and
    the positions at which ``E`` read a zero row (bool ``[B, T]``)."""
    log: list[tuple[int, Tensor]] = []

    def hook(module: Any, args: tuple[Any, ...]) -> None:
        if model.training and args[0].ndim == 3:
            log.append((model.train_step, (args[0] == 0.0).all(dim=-1).detach().clone()))

    model.encoder.register_forward_pre_hook(hook)
    return log


@pytest.mark.usefixtures("small_blocks")
def test_compiled_trainer_draws_the_eager_trainers_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer's compiled run (``tests/test_p2_8_compile.py``'s, dense routing) with both
    keys on. The eager run cuts its windows at their trailing padding and the compiled run
    does not, and a statistics step runs one more compiled pass per loss term: every one of
    these windows blanks the positions of its optimiser step, so the compiled run's losses,
    evaluation and per-term gradient norms are the eager run's to rounding."""
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    train, dev = ByteDataset(p28._docs(16)), ByteDataset(p28._docs(4, "dev"))
    steps = 5
    histories, logs, models = {}, {}, {}
    for mode in ("none", "reduce-overhead"):
        cfg = p28._trainer_cfg(mode, "train.content_dropout=0.2", ZERO)
        ctx = resolve(cfg.hardware)
        assert ctx.compile_enabled == (mode != "none")
        trainer = _trainer(cfg, tmp_path / mode)
        models[mode], logs[mode] = trainer.hsn, _log_blank_rows(trainer.hsn)
        histories[mode] = trainer.fit(train, dev, steps)
    eager, comp = histories["none"], histories["reduce-overhead"]
    assert comp.measured["compiled_windows"] >= 9.0 and eager.measured["compiled_windows"] == 0.0
    assert any(r.stats["window_positions"] < 32.0 for r in eager.steps)  # the eager run cuts
    assert all(r.stats["window_positions"] == 32.0 for r in comp.steps)
    # the positions: one eager window per step; the compiled run's windows of a step (its own
    # and a statistics step's passes) all blank what the step draws, and the eager window the
    # same positions on the columns it computes
    assert [step for step, _ in logs["none"]] == list(range(steps))
    assert len(logs["reduce-overhead"]) > steps  # the statistics steps' extra passes
    for step, blank in logs["none"]:
        models["none"].set_train_step(step)
        drawn = models["none"].content_drop_mask(4, 32)
        assert 0 < int(drawn.sum()) < drawn.numel()
        assert blank.shape[1] <= 32 and torch.equal(blank, drawn[:, : blank.shape[1]])
        passes = [rows for at, rows in logs["reduce-overhead"] if at == step]
        assert passes and all(torch.equal(rows, drawn) for rows in passes)
    for a, b in zip(eager.steps, comp.steps, strict=True):
        assert abs(a.total - b.total) <= 1e-4 * max(1.0, abs(a.total))
        norms = {k: v for k, v in a.stats.items() if k.startswith("term_grad_norm/")}
        assert list(norms) == [k for k in b.stats if k.startswith("term_grad_norm/")]
        for key, value in norms.items():
            assert abs(b.stats[key] - value) <= 1e-4 * max(1.0, abs(value)), key
    assert sum(1 for k, v in comp.steps[2].stats.items() if "term_grad_norm/" in k and v > 0) >= 2
    for key, value in eager.evals[0].items():
        if isinstance(value, float):
            assert abs(comp.evals[0][key] - value) <= 1e-4 * max(1.0, abs(value)), key


def test_trainer_marks_the_bytes_of_its_training_windows_and_no_others(tmp_path: Path) -> None:
    steps = 3
    cfg = hsn_config(steps, *TRAINED_FAMILIES["hsn_t"], ZERO, "train.eval_every=1")
    trainer = _trainer(cfg, tmp_path / "run")
    train_ds, dev_ds = datasets()
    history = trainer.fit(train_ds, dev_ds, steps)
    assert len(history.evals) == steps  # the evaluations ran between the steps
    # the windows the trainer was given, from a loader of its own
    loader = _trainer(cfg, tmp_path / "other")._loader(train_ds, shuffle=True, epochs=None)
    held = torch.cat([next(loader)["tokens"].flatten() for _ in range(steps)]).unique()
    seen = trainer.hsn.byte_seen
    assert seen is not None
    assert seen.nonzero().flatten().tolist() == held.tolist()
    # the development documents hold bytes that no training window of these steps held, and
    # the evaluations did not mark them
    dev_bytes = torch.cat([dev_ds[i]["tokens"] for i in range(len(dev_ds))]).unique()
    missing = dev_bytes[~torch.isin(dev_bytes, held)].to(torch.int64)
    assert missing.numel() > 0 and not bool(seen[missing].any())
    # the checkpoint holds the buffer, and a warm start (train.init_from) takes it over; a
    # model without the key has no place for it
    checkpoint = history.checkpoints[-1]
    assert torch.equal(load_checkpoint(checkpoint, build_model(cfg))["model"]["byte_seen"], seen)
    warm = build_model(cfg)
    assert isinstance(warm, HSN) and warm_start(warm, checkpoint)["step"] == steps
    assert warm.byte_seen is not None and torch.equal(warm.byte_seen, seen)
    with pytest.raises(ValueError, match="byte_seen"):  # the options to set are named
        warm_start(build_model(hsn_config(steps, *TRAINED_FAMILIES["hsn_t"])), checkpoint)


@pytest.mark.parametrize("family", list(TRAINED_FAMILIES))
def test_zero_unseen_bytes_alone_takes_the_same_training_steps(family: str, tmp_path: Path) -> None:
    """A training window's bytes are marked before the window runs, so no training lookup is
    ever blanked: the run with the key takes the steps of the run without it, bit for bit.
    Only what is scored differs."""
    steps = 8
    train_ds, dev_ds = datasets()
    plain = _trainer(hsn_config(steps, *TRAINED_FAMILIES[family]), tmp_path / "plain")
    expected = plain.fit(train_ds, dev_ds, steps)
    keyed = _trainer(hsn_config(steps, *TRAINED_FAMILIES[family], ZERO), tmp_path / "keyed")
    actual = keyed.fit(train_ds, dev_ds, steps)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps]
    assert [s.grad_norm for s in actual.steps] == [s.grad_norm for s in expected.steps]
    _assert_same_state(plain.hsn, keyed.hsn)


def test_resume_draws_the_same_positions_and_keeps_the_seen_bytes(tmp_path: Path) -> None:
    steps = 8
    options = ("train.content_dropout=0.2", ZERO, "train.ckpt_every=4", "train.eval_every=4")
    cfg = hsn_config(steps, *TRAINED_FAMILIES["hsn_t"], *options)
    train_ds, dev_ds = datasets()
    full = _trainer(cfg, tmp_path / "full")
    expected = full.fit(train_ds, dev_ds, steps)
    resumed = _trainer(cfg, tmp_path / "resumed")
    assert resumed.resume(tmp_path / "full" / "step_00000004.pt") == 4
    partial = resumed.hsn.byte_seen
    assert partial is not None and 0 < int(partial.sum())
    actual = resumed.fit(train_ds, dev_ds, steps)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[4:]]
    assert [e["loss"] for e in actual.evals] == [e["loss"] for e in expected.evals[1:]]
    state, other = full.model.state_dict(), resumed.model.state_dict()
    assert state.keys() == other.keys() and all(_same(other[k], v) for k, v in state.items())
    # the run is not the run without the dropout
    plain = _trainer(
        hsn_config(steps, *TRAINED_FAMILIES["hsn_t"], *options[1:]), tmp_path / "plain"
    )
    assert [s.loss for s in plain.fit(train_ds, dev_ds, steps).steps] != [
        s.loss for s in expected.steps
    ]
    with pytest.raises(ValueError, match="configuration"):
        plain.resume(tmp_path / "full" / "step_00000004.pt")


# --------------------------------------------------------------------------------------------------
# train.content_dropout_target = unknown and model.unseen_bytes_as: the learned unknown rows
# --------------------------------------------------------------------------------------------------

#: the seed tags of the two rows' generators (b"KCUB", b"KCUR")
UNKNOWN_TAGS = {"byte_emb.unknown": 0x4B435542, "reader.emb_unknown": 0x4B435552}
UNSEEN_ZERO = "model.unseen_bytes_as=zero"


def _rows_of(family: str) -> tuple[str, ...]:
    return UNKNOWN_ROWS if family == "hsn_t" else UNKNOWN_ROWS[:1]


def _build_unknown(cfg: RunConfig, dtype: torch.dtype = torch.float64) -> HSN:
    """:func:`_build` for a model with the unknown rows: the shared parameters get the offsets
    :func:`_build` gives the model without them (the rows are moved after every other one), so
    the two models hold the same weights."""
    seed_everything(0)
    model = build_model(cfg).to(dtype)
    assert isinstance(model, HSN)
    generator = torch.Generator().manual_seed(7)
    named = list(model.named_parameters())
    order = [p for n, p in named if n not in UNKNOWN_ROWS] + [
        p for n, p in named if n in UNKNOWN_ROWS
    ]
    with torch.no_grad():
        for p in order:
            p.add_(0.3 * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    return model.train()


class _Swapped(_Reads):
    """The reads of a model without the unknown rows, with ``rows`` (byte row, reader row) put
    in from outside where ``unknown`` is True and zero where ``zero`` is: the reference that a
    model with the rows is compared with."""

    def __init__(
        self,
        model: HSN,
        zero: Tensor | None,
        unknown: Tensor | None,
        rows: tuple[Tensor, Tensor | None],
    ) -> None:
        super().__init__(model)
        self.zero, self.unknown, self.rows = zero, unknown, rows

    def __enter__(self) -> _Swapped:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _encoder(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = replace_content(args[0], self.zero, self.unknown, self.rows[0])
        self.encoder = x.detach().clone()
        return (x, *args[1:])

    def _reader(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = replace_content(args[0], self.zero, self.unknown, self.rows[1])
        self.reader = x.detach().clone()
        return (x, *args[1:])


def _outside_rows(model: HSN) -> tuple[Tensor, Tensor | None]:
    """Leaf copies of ``model``'s unknown rows, to put into a model without them."""
    byte_row, reader_row = model.unknown_rows()
    assert byte_row is not None
    outside = byte_row.detach().clone().requires_grad_()
    return outside, None if reader_row is None else reader_row.detach().clone().requires_grad_()


def _assert_same_window_and_row_gradients(
    on: HSN, got: p28.Window, off: HSN, ref: p28.Window, outside: tuple[Tensor, Tensor | None]
) -> None:
    """The window of ``on`` (with the rows) is the window of ``off`` (rows put in from outside)
    bit for bit, every shared parameter has the same gradient, and each unknown row has the
    gradient of its outside copy: what the positions that read it send back, and nothing
    else."""
    _assert_same_window(got, ref)
    names = [n for n, p in on.named_parameters() if p.requires_grad]
    params = [p for n, p in on.named_parameters() if p.requires_grad]
    grads = dict(
        zip(names, torch.autograd.grad(p28._loss(got), params, allow_unused=True), strict=True)
    )
    off_names = [n for n, p in off.named_parameters() if p.requires_grad]
    off_params = [p for n, p in off.named_parameters() if p.requires_grad]
    extra = [r for r in outside if r is not None]
    off_grads = torch.autograd.grad(p28._loss(ref), [*off_params, *extra], allow_unused=True)
    assert set(names) - set(off_names) == set(UNKNOWN_ROWS[: len(extra)])
    live = 0
    for name, g in zip(off_names, off_grads[: len(off_names)], strict=True):
        assert (g is None) == (grads[name] is None), name
        if g is not None:
            assert torch.equal(grads[name], g), name
            live += int(bool(g.abs().max() > 0))
    assert live >= 20
    for name, g in zip(UNKNOWN_ROWS, off_grads[len(off_names) :], strict=False):
        assert g is not None and bool(g.any()), name  # the row is trained
        assert torch.equal(grads[name], g), name


@pytest.mark.parametrize("family", list(FAMILIES))
def test_unknown_rows_exist_only_with_the_target_and_are_drawn_like_their_tables(
    family: str,
) -> None:
    plain_cfg = _cfg(*FAMILIES[family], DROP)
    cfg = _cfg(*FAMILIES[family], DROP, TO_UNKNOWN)
    seed_everything(0)
    off = build_model(plain_cfg)
    stream_off = torch.get_rng_state()
    seed_everything(0)
    on = build_model(cfg)
    stream_on = torch.get_rng_state()
    assert isinstance(on, HSN) and isinstance(off, HSN)
    assert torch.equal(stream_on, stream_off)  # the rows do not read the global stream
    rows = _rows_of(family)
    on_state, off_state = on.state_dict(), off.state_dict()
    assert set(on_state) - set(off_state) == set(rows) and set(off_state) <= set(on_state)
    assert all(_same(on_state[key], value) for key, value in off_state.items())
    assert off.unknown_rows() == (None, None) and off.content_dropout_target == "zero"
    # drawn as their tables' rows are, from generators seeded with (run seed, tag) alone
    drawn = dict(zip(UNKNOWN_ROWS, on.unknown_rows(), strict=True))
    for name in UNKNOWN_ROWS:
        row = drawn[name]
        if name not in rows:
            assert row is None
            continue
        state = np.random.SeedSequence([cfg.seed, UNKNOWN_TAGS[name]]).generate_state(1)[0]
        generator = torch.Generator().manual_seed(int(state))
        if name == "byte_emb.unknown":
            expected = init_weight_(
                torch.empty(1, cfg.model.d), cfg.model.state_init, generator=generator
            )
        else:
            expected = torch.nn.init.normal_(
                torch.empty(1, cfg.model.reader_width), std=0.02, generator=generator
            )
        assert row is not None and row.ndim == 2 and row.requires_grad
        assert torch.equal(row.detach(), expected), name
        assert row is on.get_parameter(name)
    # parameters of their tables' modules: counted in the tables' report rows
    report_on, report_off = on.param_report(), off.param_report()
    assert report_on["byte_emb"] - report_off["byte_emb"] == cfg.model.d
    width = cfg.model.reader_width if family == "hsn_t" else 0
    assert report_on["total_trainable"] - report_off["total_trainable"] == cfg.model.d + width
    if family == "hsn_t":
        assert report_on["reader"] - report_off["reader"] == width
    # another seed draws other rows
    other = build_model(_cfg(*FAMILIES[family], DROP, TO_UNKNOWN, "seed=1"))
    assert isinstance(other, HSN)
    other_row, byte_row = other.unknown_rows()[0], on.unknown_rows()[0]
    assert other_row is not None and byte_row is not None
    assert not torch.equal(other_row, byte_row)
    assert "content_dropout_target='unknown'" in on.extra_repr()
    assert "content_dropout_target" not in off.extra_repr()
    # a checkpoint with the rows does not load into a model without them, nor the reverse
    with pytest.raises(RuntimeError, match="unknown"):
        off.load_state_dict(on.state_dict())
    with pytest.raises(RuntimeError, match="unknown"):
        on.load_state_dict(off.state_dict())


@pytest.mark.parametrize("family", list(FAMILIES))
def test_unknown_target_puts_the_learned_row_at_exactly_the_dropped_positions(
    family: str,
) -> None:
    """In training the dropped positions read the unknown row and every other position its
    byte's row; the window is, bit for bit and in every gradient, the window of the model
    without the keys whose two reads get the row from outside at those positions, and the
    row gets exactly the gradient of the positions that read it. In evaluation nothing
    changes (the bytes keep their own rows here)."""
    cfg = _cfg(*FAMILIES[family], DROP, TO_UNKNOWN)
    on, off = _build_unknown(cfg), _build(_cfg(*FAMILIES[family]))
    on.set_train_step(3)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    drop = on.content_drop_mask(B, T)
    with _reads(on) as reads:
        got = on(tokens, ctx, _start(on, B), reset)
        _assert_blanked(on, reads, tokens, drop, unknown=True)
    assert torch.equal(got[2]["z_target"], on.byte_emb(tokens).detach())  # the target is kept
    outside = _outside_rows(on)
    with _Swapped(off, None, drop, outside):
        ref = off(tokens, ctx, _start(off, B), reset)
    _assert_same_window_and_row_gradients(on, got, off, ref, outside)
    # the row is not a zero row: the window differs from the window with zero rows
    with _reads(off, drop):
        zeroed = off(tokens, ctx, _start(off, B), reset)
    assert not torch.equal(zeroed[0], got[0])
    on.eval()
    off.eval()
    with torch.no_grad():
        _assert_same_window(
            on(tokens, ctx, _start(on, B), reset), off(tokens, ctx, _start(off, B), reset)
        )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_unseen_bytes_as_unknown_reads_the_row_at_exactly_the_unseen_bytes(
    family: str, training: bool
) -> None:
    """Never-trained bytes read the unknown row and the other bytes their own rows, in both
    modes; in training the dropped positions read it too. Nothing else changes, and the row
    gets the gradient of the positions that read it."""
    cfg = _cfg(*FAMILIES[family], DROP, TO_UNKNOWN, UNSEEN_UNKNOWN)
    on, off = _build_unknown(cfg).train(training), _build(_cfg(*FAMILIES[family])).train(training)
    assert on.unseen_bytes_as == "unknown" and on.byte_seen is not None
    on.set_train_step(5)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    unseen = _half_seen(on, tokens)
    reads_row = unseen | on.content_drop_mask(B, T) if training else unseen
    assert bool((reads_row & ~unseen).any()) == training
    with torch.set_grad_enabled(training):
        with _reads(on) as reads:
            got = on(tokens, ctx, _start(on, B), reset)
            _assert_blanked(on, reads, tokens, reads_row, unknown=True)
        outside = _outside_rows(on)
        with _Swapped(off, None, reads_row, outside):
            ref = off(tokens, ctx, _start(off, B), reset)
        if training:
            _assert_same_window_and_row_gradients(on, got, off, ref, outside)
        else:
            _assert_same_window(got, ref)
        # with every byte of the window seen, only the dropped positions read the row
        on.mark_bytes_seen(tokens)
        with _reads(on) as reads:
            got = on(tokens, ctx, _start(on, B), reset)
        if training:
            _assert_blanked(on, reads, tokens, on.content_drop_mask(B, T), unknown=True)
        else:
            _assert_same_window(got, off(tokens, ctx, _start(off, B), reset))


def test_unseen_bytes_as_unknown_in_a_single_position() -> None:
    """:meth:`HSN.step` (evaluation mode) reads the unknown row at a never-trained byte."""
    cfg = _cfg(*FAMILIES["hsn"], DROP, TO_UNKNOWN, UNSEEN_UNKNOWN)
    on, off = _build_unknown(cfg).eval(), _build(_cfg(*FAMILIES["hsn"])).eval()
    tokens, ctx, _ = _window(cfg)
    x, c = tokens[:, 0], ctx[:, 0]
    on.mark_bytes_seen(x[:1])
    unseen = x != x[0]
    assert bool(unseen.any()) and not bool(unseen.all())
    with torch.no_grad():
        got = on.step(x, c, _start(on, x.shape[0]))
        with _Swapped(off, None, unseen, _outside_rows(on)):
            ref = off.step(x, c, _start(off, x.shape[0]))
        plain = off.step(x, c, _start(off, x.shape[0]))
    assert torch.equal(got.logits, ref.logits) and torch.equal(got.state.h, ref.state.h)
    assert torch.equal(got.z_target, plain.z_target)  # the target keeps the byte's row
    assert not torch.equal(got.logits, plain.logits)


def test_zero_unseen_bytes_with_dropout_to_the_unknown_row() -> None:
    """``model.unseen_bytes_as=zero`` with ``train.content_dropout_target=unknown``: a dropped
    position reads the unknown row whatever its byte, an unseen byte that is not dropped reads
    zero, and in evaluation the unseen bytes read zero."""
    cfg = _cfg(*FAMILIES["hsn_t"], DROP, TO_UNKNOWN, UNSEEN_ZERO)
    on, off = _build_unknown(cfg), _build(_cfg(*FAMILIES["hsn_t"]))
    on.set_train_step(2)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    unseen = _half_seen(on, tokens)
    drop = on.content_drop_mask(B, T)
    assert bool((drop & unseen).any()) and bool((unseen & ~drop).any())
    zero, unknown = on._content_masks(tokens)
    assert zero is not None and unknown is not None
    assert torch.equal(unknown, drop) and torch.equal(zero, unseen & ~drop)
    outside = _outside_rows(on)
    got = on(tokens, ctx, _start(on, B), reset)
    with _Swapped(off, unseen & ~drop, drop, outside):
        ref = off(tokens, ctx, _start(off, B), reset)
    _assert_same_window_and_row_gradients(on, got, off, ref, outside)
    on.eval()
    with _reads(on) as reads, torch.no_grad():
        on(tokens, ctx, _start(on, B), reset)
        _assert_blanked(on, reads, tokens, unseen)


@pytest.mark.parametrize("training", [True, False])
def test_zero_unseen_bytes_is_unseen_bytes_as_zero(training: bool) -> None:
    """The older spelling and the new one build the same model: the same state-dict keys,
    the same window bit for bit, and one's checkpoint loads into the other."""
    old_cfg = _cfg(*FAMILIES["hsn_t"], DROP, ZERO)
    new_cfg = _cfg(*FAMILIES["hsn_t"], DROP, UNSEEN_ZERO)
    assert old_cfg.model.unseen_bytes_mode == new_cfg.model.unseen_bytes_mode == "zero"
    old, new = _build(old_cfg).train(training), _build(new_cfg).train(training)
    assert old.unseen_bytes_as == new.unseen_bytes_as == "zero"
    assert "zero_unseen_bytes=True" in new.extra_repr()
    tokens, ctx, reset = _window(old_cfg)
    B = tokens.shape[0]
    _half_seen(old, tokens)
    new.load_state_dict(old.state_dict())  # strict: byte_seen with the marks
    old.set_train_step(4)
    new.set_train_step(4)
    with torch.set_grad_enabled(training):
        got = new(tokens, ctx, _start(new, B), reset)
        ref = old(tokens, ctx, _start(old, B), reset)
        _assert_same_window(got, ref)
        if training:
            _assert_same_gradients(new, got, old, ref)
    # two spellings of one model, two configuration hashes
    assert config_hash(old_cfg) != config_hash(new_cfg)


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, [UNSEEN_UNKNOWN], "only train.content_dropout_target=unknown builds"),
        (RECIPE_A6, ["train.content_dropout=0.1", UNSEEN_UNKNOWN], "builds and trains"),
        (RECIPE_A6, [TO_UNKNOWN], "no position is dropped"),
        (RECIPE_A6, [TO_UNKNOWN, UNSEEN_UNKNOWN], "no position is dropped"),
        (RECIPE_A6, [ZERO, UNSEEN_ZERO], "older spelling"),
        (RECIPE_A6, [ZERO, "train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_UNKNOWN], "older"),
        (RECIPE_A6, ["train.content_dropout_target=blank"], "content_dropout_target"),
        (RECIPE_A6, ["model.unseen_bytes_as=random"], "unseen_bytes_as"),
        (GRU_2M, ["train.content_dropout=0.1", TO_UNKNOWN], "generic trainer"),
        (GRU_2M, [UNSEEN_ZERO], "generic trainer"),
        (READER_LM, ["train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_UNKNOWN], "generic trainer"),
        (READER_LM, [UNSEEN_ZERO], "generic trainer"),
        (NO_MEMORY, [UNSEEN_ZERO], "model.memory=false"),
        (NO_MEMORY, ["train.content_dropout=0.1", TO_UNKNOWN], "model.memory=false"),
    ],
)
def test_unknown_configuration_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_unknown_options_refuse_the_fused_window_and_the_probes() -> None:
    recipe = (*A5, ALL, "model.cand_norm=rms")
    for options in ((DROP, TO_UNKNOWN), (DROP, TO_UNKNOWN, UNSEEN_UNKNOWN), (UNSEEN_ZERO,)):
        with pytest.raises(ValueError, match="hardware.window=fused is untested"):
            _cfg(*recipe, "hardware.window=fused", *options)
        assert _cfg(*recipe, "hardware.window=split", *options).hardware.window == "split"
    probes = ("train.lambda_u=0.1", "train.probe_horizon=4")
    with pytest.raises(ValueError, match="counterfactual probes"):
        _cfg(*A5, *probes, DROP, TO_UNKNOWN, UNSEEN_UNKNOWN)
    assert _cfg(*A5, *probes, UNSEEN_ZERO).model.unseen_bytes_mode == "zero"


def test_model_and_trainer_refuse_unknown_settings_that_are_not_the_configurations(
    tmp_path: Path,
) -> None:
    cfg = hsn_config(4, *A5)
    unroll = cfg.train.unroll
    with pytest.raises(ValueError, match="content_dropout_target must be"):
        HSN(cfg.model, seed=0, unroll=unroll, content_dropout=0.1, content_dropout_target="blank")
    with pytest.raises(ValueError, match="needs content_dropout > 0"):
        HSN(cfg.model, seed=0, unroll=unroll, content_dropout_target="unknown")
    unseen = hsn_config(4, *A5, "train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_UNKNOWN)
    with pytest.raises(ValueError, match="model.unseen_bytes_as=unknown"):
        HSN(unseen.model, seed=0, unroll=unroll, content_dropout=0.1)
    ctx = resolve(cfg.hardware)
    drop = hsn_config(4, *A5, "train.content_dropout=0.1")
    to_unknown = hsn_config(4, *A5, "train.content_dropout=0.1", TO_UNKNOWN)
    with pytest.raises(ValueError, match="train.content_dropout_target"):
        TBPTTTrainer(build_model(to_unknown), drop, ctx, run_dir=tmp_path)
    with pytest.raises(ValueError, match="train.content_dropout_target"):
        TBPTTTrainer(build_model(drop), to_unknown, ctx, run_dir=tmp_path)
    zero_model = build_model(
        hsn_config(4, *A5, "train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_ZERO)
    )
    with pytest.raises(ValueError, match="model.unseen_bytes_as"):
        TBPTTTrainer(zero_model, unseen, ctx, run_dir=tmp_path)


def test_unknown_keys_are_hashed_and_only_the_model_key_is_in_the_memory_key() -> None:
    drop = load_config(RECIPE_A6, [*RD32, "train.content_dropout=0.1"])
    target = load_config(RECIPE_A6, [*RD32, "train.content_dropout=0.1", TO_UNKNOWN])
    both = load_config(RECIPE_A6, [*RD32, "train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_UNKNOWN])
    zero = load_config(RECIPE_A6, [*RD32, "train.content_dropout=0.1", TO_UNKNOWN, UNSEEN_ZERO])
    configs = (drop, target, both, zero)
    assert len({config_hash(c) for c in configs}) == 4
    assert len({_arm_id(c) for c in configs}) == 4
    assert memory_key(target, "pdg", PROTOCOL) == memory_key(drop, "pdg", PROTOCOL)
    assert memory_key(both, "pdg", PROTOCOL) != memory_key(drop, "pdg", PROTOCOL)
    assert memory_key(zero, "pdg", PROTOCOL) not in {
        memory_key(c, "pdg", PROTOCOL) for c in (drop, both)
    }
    dump = json.loads(canonical_json(both))
    assert dump["train"]["content_dropout_target"] == "unknown"
    assert dump["model"]["unseen_bytes_as"] == "unknown"


def test_resume_with_the_unknown_rows(tmp_path: Path) -> None:
    """The rows are in the checkpoint with their optimiser state (decayed like their tables):
    a resumed run takes the steps of the uninterrupted one, bit for bit."""
    steps = 8
    options = (
        "train.content_dropout=0.2",
        TO_UNKNOWN,
        UNSEEN_UNKNOWN,
        "train.ckpt_every=4",
        "train.eval_every=4",
    )
    cfg = hsn_config(steps, *TRAINED_FAMILIES["hsn_t"], *options)
    train_ds, dev_ds = datasets()
    full = _trainer(cfg, tmp_path / "full")
    byte_row, reader_row = full.hsn.unknown_rows()
    assert byte_row is not None and reader_row is not None
    decayed = [p for g in full.optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]]
    for row, table in (
        (byte_row, full.hsn.byte_emb.weight),
        (reader_row, full.hsn.reader.emb.weight),
    ):
        assert any(p is row for p in decayed) == any(p is table for p in decayed)
    start = byte_row.detach().clone()
    expected = full.fit(train_ds, dev_ds, steps)
    assert not torch.equal(byte_row.detach(), start)
    resumed = _trainer(cfg, tmp_path / "resumed")
    assert resumed.resume(tmp_path / "full" / "step_00000004.pt") == 4
    actual = resumed.fit(train_ds, dev_ds, steps)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[4:]]
    assert [e["loss"] for e in actual.evals] == [e["loss"] for e in expected.evals[1:]]
    state, other = full.model.state_dict(), resumed.model.state_dict()
    assert set(UNKNOWN_ROWS) <= set(state)
    assert state.keys() == other.keys() and all(_same(other[k], v) for k, v in state.items())


HOLD_MANIFEST = REPO / "data" / "synthetic_hold" / "manifest.json"
#: the hold-out arm of key_unknown.yaml (hsn-experiments/remote/specs) on the CPU: its base
#: overrides without the card's (profile, compile mode, graph aliasing), at a small batch
HOLD_ARM = (
    "model_name=hsn_t",
    "data.dataset=synthetic_hold",
    f"data.manifest={HOLD_MANIFEST}",
    "data.tasks=[kv, delayed2h]",
    "data.variants=[in_range]",
    "model.k_select=12",
    "model.reader=transformer",
    "model.reader_span=32",
    "model.address_gate=true",
    "model.address_gate_input=summary",
    "model.n_cells=64",
    "train.feedback_detach=[summary, eps, router]",
    "model.scorer_position=wrap",
    "model.address_ngram=2",
    "model.address_history=ngram",
    "train.unit_credit=executed",
    "train.soft_until_step=0",
)


@pytest.mark.skipif(not HOLD_MANIFEST.is_file(), reason="needs data/synthetic_hold")
def test_hold_arm_trains_with_the_unknown_rows(tmp_path: Path) -> None:
    """Three steps of the hold-out arm's own recipe (its schedule, update limits, weight
    average and route guard) on the hold data through the real trainer, with dropout 0.1 to
    the unknown rows and the never-trained bytes reading them: finite losses, the rows move,
    their gradients are in the byte_emb and reader groups, bytes 112 to 143 stay unseen and
    the evaluation runs."""
    steps = 3
    overrides = [
        "profile=cpu",
        *HOLD_ARM,
        "train.batch=2",
        "train.micro_batch=2",
        f"train.steps={steps}",
        "train.warmup_steps=1",  # the recipe's warm-up of 200 ends inside its decay
        f"train.eval_every={steps}",
        "train.ckpt_every=1000",
        "train.stats_every=1",
        "train.content_dropout=0.1",
        TO_UNKNOWN,
        UNSEEN_UNKNOWN,
    ]
    cfg = load_config(RECIPE_A6, overrides)
    assert cfg.train.weight_ema and cfg.train.update_limit is not None
    train_ds, dev_ds = load_datasets(cfg)
    dev_ds = ByteDataset(dev_ds.examples[:16])  # a short evaluation
    trainer = _trainer(cfg, tmp_path / "run")
    rows = trainer.hsn.unknown_rows()
    start = [r.detach().clone() for r in rows if r is not None]
    assert len(start) == 2
    history = trainer.fit(train_ds, dev_ds, steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    for row, before in zip(rows, start, strict=True):
        assert row is not None and bool(torch.isfinite(row).all())
        assert not torch.equal(row.detach(), before)
    for group in ("byte_emb", "reader"):
        assert history.steps[-1].stats[f"grad_norm/{group}"] > 0.0
    seen = trainer.hsn.byte_seen
    assert seen is not None and bool(seen.any())
    assert not bool(seen[112:144].any())  # the held-out key bytes are never in a training window
    assert history.evals and math.isfinite(history.evals[-1]["loss"])
