"""The byte-content options (``docs/key_content.md``): ``train.content_dropout`` and
``model.zero_unseen_bytes``, for the HSN and for HSN-T.

Both act on what the encoder ``E`` and HSN-T's reader read of a byte: at some positions a zero
row in place of the byte's row of the two content tables. The tests hold them to that: the
defaults are the model of f3f271a, the blanked positions are exactly the drawn ones (training
only) and exactly the bytes no training window held, nothing else of the window changes, and
the trainer, the compiled window and the split window run with them.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, RunConfig, canonical_json, config_hash, load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import arm_identity, memory_key
from hsn.model.hsn import HSN
from hsn.runner import build_model, warm_start
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


def _assert_blanked(model: HSN, reads: _Reads, tokens: Tensor, blank: Tensor) -> None:
    """``E`` and the reader read the tables' rows, zero exactly where ``blank`` is True."""
    assert 0 < int(blank.sum()) < blank.numel()  # the case holds both kinds of position
    tables = [(reads.encoder, model.byte_emb(tokens).detach())]
    if model.reader is not None:
        tables.append((reads.reader, model.reader.emb(tokens).detach()))
    for read, rows in tables:
        assert read is not None
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
    assert {"train.content_dropout", "model.zero_unseen_bytes"} <= set(HASH_OMIT_AT_DEFAULT)
    for overrides, (hashed, memory, arm) in F3F271A.items():
        cfg = load_config(RECIPE_A6, list(overrides))
        named = load_config(
            RECIPE_A6, [*overrides, "train.content_dropout=0.0", "model.zero_unseen_bytes=false"]
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
    assert off._content_blank(tokens) is None  # training mode: nothing is drawn or computed
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


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager_with_both_options(family: str) -> None:
    """The options act in the window's prologue, which the compiled window runs eagerly: the
    compiled training window and the compiled evaluation window equal the eager ones."""
    reader = READER if family == "hsn_t" else ()
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        *reader,
        DROP,
        ZERO,
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
        _assert_blanked(model, reads, tokens, blank)
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]
    model.eval()
    with torch.no_grad():
        ref = p28._run(lambda: model(tokens, cx, state, reset))
        with _reads(model) as reads:
            got = p28._run(lambda: window(tokens, cx, state, reset))
            _assert_blanked(model, reads, tokens, unseen)
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


def test_split_window_equals_the_autograd_window_with_both_options() -> None:
    """``hardware.window=split`` (the HSN's): the sweep and the replay start from the same
    prologue, so the split window's values and gradients are the autograd window's.

    The values agree to float64 rounding. The gradients are compared term by term, each
    tensor to 1e-6 of its largest entry: the router's cost path is float32 in both windows
    (``expected_cost``), the scorer's gradient is rounded there, and the two windows add it
    up in different orders. Measured on 15 Nov 2026 with this model, with the options and
    with the same reads zeroed from outside in the model without them: gaps up to 2e-9 of
    the largest entry in one term and 7e-9 in a summed objective, the largest of them in
    the scorer's linear layer. A window that blanked other positions than the other would
    differ by its whole size."""
    cfg = _cfg(*A5, ALL, "model.cand_norm=rms", DROP, ZERO)
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
    logits, ref_state, ref_info = model(tokens, ctx, state, reset)
    with _reads(model) as reads:
        s_logits, s_state, s_info = SplitWindow(model, cfg, None)(tokens, ctx, state, reset)
        _assert_blanked(model, reads, tokens, blank)
    torch.testing.assert_close(s_logits, logits, rtol=1e-10, atol=1e-12)
    for field in ("h", "q"):
        torch.testing.assert_close(
            getattr(s_state, field), getattr(ref_state, field), rtol=1e-10, atol=1e-12
        )
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])
    ref_terms, got_terms = _terms(logits, ref_info, mask), _terms(s_logits, s_info, mask)
    live = 0
    for name, term in ref_terms.items():
        ref = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        got = torch.autograd.grad(got_terms[name], params, retain_graph=True, allow_unused=True)
        for want, have in zip(ref, got, strict=True):
            assert (want is None) == (have is None), name
            if want is not None and have is not None:
                largest = float(want.abs().max())
                assert float((have - want).abs().max()) <= 1e-6 * largest + 1e-12, name
                live += int(largest > 0)
    assert live >= 20  # real gradients, not zeros


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
    "options", [("train.content_dropout=0.1",), (ZERO,), ("train.content_dropout=0.1", ZERO)]
)
@pytest.mark.parametrize("family", list(TRAINED_FAMILIES))
def test_recipe_with_the_options_trains(
    family: str, options: tuple[str, ...], tmp_path: Path
) -> None:
    """Thirty TBPTT steps on the tiny CPU configuration of the trainer tests: finite losses
    that fall, and a final evaluation."""
    steps = 30
    cfg = hsn_config(steps, *TRAINED_FAMILIES[family], *options)
    trainer = _trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    assert history.evals and math.isfinite(history.evals[-1]["loss"])


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
    with pytest.raises(RuntimeError, match="byte_seen"):
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
