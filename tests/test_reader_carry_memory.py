"""Reading past the window with the memory on (``docs/amendments/reader_carry_memory.md``).

``model.reader_carry = C > 0`` on an HSN-T with its memory: the reader also reads the last ``C``
positions of its document before the window, from a cache of detached keys and values, so its
sight does not fall to one byte at a window boundary. The cache is not part of the cells' state:
it rides with the address carry as a :class:`~hsn.model.hsn.WindowCarry`, which
:func:`~hsn.model.hsn.address_history` builds from a window's ``info`` and the forward pass takes
as ``prev_tokens``.

These tests check that the default is untouched (hash, ``info``, the carry's type), the
refusals, that windows read with the carry equal the *sighted reading* of the same weights (the
oracle of the diagnostic that proposed the key, ``~/hsn-handoff/TEXT-GAP-DESIGN-2026-11-17.md``:
the reader's output over the whole stream with a per-block span of ``C + 1``, handed in window
by window, the memory run as it is), that a document start clears the carry and no later byte
is read, that a caller who carries the bytes alone is refused, that the evaluation carries it,
that the compiled window equals the eager one without a new graph across windows, and that the
trainer carries, checkpoints and resumes it bit for bit.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, RunConfig, config_hash, load_config
from hsn.device import resolve, seed_everything
from hsn.eval.evaluate import DEFAULT_METRICS, evaluate_windows
from hsn.model.hsn import (
    READER_CACHE_KEYS,
    WindowCarry,
    address_history,
    clone_carry,
    pack_carry,
    reader_carried,
    split_packed_carry,
)
from hsn.model.hsn_t import HSNT
from hsn.model.state import CellGraphState
from hsn.runner import build_model
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5
from test_hsn_t import CUT_C, READER, _gap, _model, _reader_cfg
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_tbptt_windows import HSNT as HSNT_RECIPE
from test_tbptt_windows import STREAM, _assert_equal, _long_datasets, _trainer
from test_word_address import _compiled_windows, _story, _wa
from test_write_events import CATTN, NGRAM, STE, _ev
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
MEMORY = REPO / "configs" / "experiments" / "hsn_t_memory.yaml"
WORDS = REPO / "configs" / "experiments" / "hsn_t_memory_words.yaml"

C = 5  # the carry of the small models (their window is 12 bytes)
CARRY = f"model.reader_carry={C}"
WORD = (_ev("word"), _wa("exact"))
#: the four forms the key is built for: per-byte writes and word events, C-read off and on
FORMS: dict[str, tuple[str, ...]] = {
    "per_byte": (),
    "per_byte_cread": (CATTN,),
    "word_events": WORD,
    "word_events_cread": (*WORD, CATTN),
}


def _pair(*extra: str) -> tuple[HSNT, HSNT, RunConfig]:
    """The model with the carry and the same weights without it (the key adds no parameter)."""
    cfg = _reader_cfg(CARRY, *extra)
    carried = _model(cfg).eval()
    plain = _model(_reader_cfg(*extra)).eval()
    plain.load_state_dict(carried.state_dict())
    return carried, plain, cfg


def _stream(cfg: RunConfig, windows: int, seed: int = 11) -> tuple[Tensor, Tensor, Tensor]:
    """``windows`` consecutive windows of three text-like rows as one stream (``tokens``,
    ``ctx``, ``reset``): document starts anywhere, also at a window's first column and inside
    the carry's reach of a boundary."""
    T = cfg.train.unroll
    tokens, ctx, reset = _story(3, windows * T, seed)
    reset[:, 0] = True
    if windows > 1:
        reset[1, T] = True  # a document that starts with a window
        reset[2, T - 2] = True  # and one that starts two bytes before a boundary
    return tokens, ctx, reset


def _read(
    model: HSNT,
    stream: tuple[Tensor, Tensor, Tensor],
    T: int,
    *,
    reader_out: Tensor | None = None,
) -> tuple[Tensor, list[Tensor | WindowCarry]]:
    """The stream read window by window as a caller reads it (``address_history`` with the
    window's ``info``, as the trainer, ``evaluate_windows`` and the hand-off scorers do):
    the logits and what was carried out of every window. ``reader_out`` hands the reader's
    output in (``[B, stream, w]``), window by window."""
    tokens, ctx, reset = stream
    assert model.reader is not None
    real = model.reader.forward
    state = model.init_state(tokens.shape[0])
    prev: Tensor | WindowCarry | None = None
    logits, carried = [], []
    with torch.no_grad():
        for lo in range(0, tokens.shape[1], T):
            sl = slice(lo, lo + T)
            if reader_out is not None:
                model.reader.forward = lambda _t, _r, _c=reader_out[:, sl]: _c  # type: ignore[method-assign]
            try:
                out, state, info = model(
                    tokens[:, sl], ctx[:, sl], state, reset[:, sl], prev_tokens=prev
                )
            finally:
                if reader_out is not None:
                    model.reader.forward = real  # type: ignore[method-assign]
            prev = address_history(model.cfg, tokens[:, sl], prev, info)
            logits.append(out)
            carried.append(prev)
    return torch.cat(logits, dim=1), carried


def _sighted(plain: HSNT, stream: tuple[Tensor, Tensor, Tensor], span: int) -> Tensor:
    """The reader's output over the whole stream, every block reading at most ``span``
    positions back (the read-time intervention of the text-gap note)."""
    assert plain.reader is not None and plain.reader.span is None
    plain.reader.span = span
    try:
        with torch.no_grad():
            out: Tensor = plain.reader(stream[0], stream[2])
    finally:
        plain.reader.span = None
    return out


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_is_untouched() -> None:
    assert "model.reader_carry" in HASH_OMIT_AT_DEFAULT
    plain = _reader_cfg()
    assert plain.model.reader_carry == 0 and not reader_carried(plain.model)
    assert config_hash(_reader_cfg("model.reader_carry=0")) == config_hash(plain)
    carried = _reader_cfg(CARRY)
    assert reader_carried(carried.model) and config_hash(carried) != config_hash(plain)
    # the recipes resolve as before and take the key
    for recipe in (MEMORY, WORDS):
        cfg = load_config(recipe)
        assert not reader_carried(cfg.model)
        top0 = ["model.reader_span=null", "model.reader_carry=256"]
        assert reader_carried(load_config(recipe, overrides=top0).model)
        assert reader_carried(
            load_config(recipe, overrides=[*top0, "model.cell_attention_heads=4"]).model
        )
    # without the key the carry is the tensor it was, info holds nothing new, and a
    # WindowCarry is refused
    model = _model(plain).eval()
    tokens, ctx, reset = _stream(plain, 1)
    with torch.no_grad():
        _, _, info = model(tokens, ctx, model.init_state(3), reset)
        assert not set(READER_CACHE_KEYS) & set(info)
        carry = address_history(plain.model, tokens, None, info)
        assert isinstance(carry, Tensor) and torch.equal(carry, tokens[:, -1])
        wrong = WindowCarry(carry, (carry, carry, carry))
        with pytest.raises(ValueError, match="no carried reader"):
            model(tokens, ctx, model.init_state(3), reset, prev_tokens=wrong)
    # HSN-T without its memory keeps the cache as its state: not this carry
    assert not reader_carried(carried.model.model_copy(update={"memory": False}))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ([CARRY, "model.key_address=false"], "handed from window to window with the address"),
        ([CARRY, "model.reader_top_words=4", "model.reader_rotary_pairs=0"], "reader_top_words"),
        (
            [CARRY, "model.address_gate=true", "model.address_gate_input=reader_prev"],
            "reader_prev",
        ),
        ([CARRY, "train.lambda_u=0.1"], "lambda_u"),
        ([CARRY, "model.reader_span=4"], "sliding span"),
        ([CARRY, "model.reader_position=learned"], "learned"),
    ],
)
def test_configuration_refusals(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _reader_cfg(*overrides)


# --------------------------------------------------------------------------------------------------
# The carried reading
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", list(FORMS))
def test_carried_windows_equal_the_sighted_reading(form: str) -> None:
    """Three windows read with the carry equal the same weights' sighted reading: the reader's
    output over the whole stream with a span of ``C + 1`` per block, handed in window by
    window, the memory run as it is. They differ from the blind reading (every window read
    alone) in the later windows, where the reader had lost its sight."""
    carried, plain, cfg = _pair(STE, CUT_C, *NGRAM, *FORMS[form])
    T = cfg.train.unroll
    stream = _stream(cfg, 3)
    got, carries = _read(carried, stream, T)
    want, _ = _read(plain, stream, T, reader_out=_sighted(plain, stream, C + 1))
    assert _gap(got, want) <= 1e-12
    blind, _ = _read(plain, stream, T)
    assert _gap(got[:, T:], blind[:, T:]) > 1e-3
    for carry in carries:  # what a caller carries: the address and the detached cache
        assert isinstance(carry, WindowCarry) and len(carry.reader) == 3
        keys, values, valid = carry.reader
        assert keys.shape == values.shape == (2, 3, 2, C, 4) and valid.shape == (3, C)
        assert not keys.requires_grad and not values.requires_grad and valid.dtype is torch.bool
    if not form.startswith("word"):  # per byte: the whole stream as one window is the same
        assert plain.reader is not None
        plain.reader.span = C + 1
        try:
            with torch.no_grad():
                tokens, ctx, reset = stream
                whole, _, _ = plain(tokens, ctx, plain.init_state(3), reset)
        finally:
            plain.reader.span = None
        assert _gap(got, whole) <= 1e-10


@pytest.mark.parametrize("form", list(FORMS))
def test_a_document_start_clears_the_carry_and_no_later_byte_is_read(form: str) -> None:
    carried, _, cfg = _pair(STE, CUT_C, *NGRAM, *FORMS[form])
    T = cfg.train.unroll
    tokens, ctx, reset = _stream(cfg, 2, seed=23)
    reset[:, 1:] = False
    reset[1, T] = True  # row 1: a document starts with the second window
    reset[2, T + 4] = True  # row 2: one starts inside it; row 0 goes on
    logits, carries = _read(carried, (tokens, ctx, reset), T)
    # the cache after the second window holds the document going on, and nothing before it
    first, second = carries
    assert isinstance(first, WindowCarry) and isinstance(second, WindowCarry)
    assert bool(first.reader[2].all()) and bool(second.reader[2].all())  # T - 4 >= C
    # another first window: the documents that start later read nothing of it
    other = tokens.clone()
    other[:, :T] = _story(3, T, 5)[0]
    changed, _ = _read(carried, (other, ctx, reset), T)
    assert _gap(changed[0, T:], logits[0, T:]) > 1e-6  # row 0 carries its document on
    assert _gap(changed[1, T:], logits[1, T:]) <= 1e-12
    assert _gap(changed[2, T + 4 :], logits[2, T + 4 :]) <= 1e-12
    assert _gap(changed[2, T : T + 4], logits[2, T : T + 4]) > 1e-6
    # a later byte changes nothing before it, in its window or the one before
    t0 = T + 5
    later = tokens.clone()
    later[:, t0::2], later[:, t0 + 1 :: 2] = 98, 32
    changed, _ = _read(carried, (later, ctx, reset), T)
    assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
    assert _gap(changed[:, t0:], logits[:, t0:]) > 1e-6
    # a reset at a window's last column leaves one cached position
    reset[0, T - 1] = True
    _, carries = _read(carried, (tokens, ctx, reset), T)
    assert isinstance(carries[0], WindowCarry)
    assert carries[0].reader[2][0].tolist() == [False] * (C - 1) + [True]


def test_the_carry_must_be_carried() -> None:
    """A caller that carries the bytes alone, or asks for the carry without the window's
    ``info``, is refused; ``None`` is a stream's first window (nothing cached)."""
    carried, plain, cfg = _pair(*NGRAM)
    T = cfg.train.unroll
    tokens, ctx, reset = _stream(cfg, 2)
    state = carried.init_state(3)
    with torch.no_grad():
        logits, state, info = carried(tokens[:, :T], ctx[:, :T], state, reset[:, :T])
        assert set(READER_CACHE_KEYS) <= set(info)
        with pytest.raises(ValueError, match="pass the info"):
            address_history(cfg.model, tokens[:, :T], None)
        carry = address_history(cfg.model, tokens[:, :T], None, info)
        assert isinstance(carry, WindowCarry)
        bytes_alone = address_history(plain.cfg, tokens[:, :T], None, info)
        assert isinstance(bytes_alone, Tensor) and torch.equal(bytes_alone, carry.address)
        with pytest.raises(ValueError, match="WindowCarry of the window before"):
            carried(tokens[:, T:], ctx[:, T:], state, reset[:, T:], prev_tokens=bytes_alone)
        short = WindowCarry(carry.address, tuple(t[..., :-1] for t in carry.reader))
        with pytest.raises(ValueError, match="cache must be"):
            carried(tokens[:, T:], ctx[:, T:], state, reset[:, T:], prev_tokens=short)
        second = (tokens[:, T:], ctx[:, T:], state, reset[:, T:])
        with_carry, _, _ = carried(*second, prev_tokens=carry)
        fresh, _, _ = carried(*second)
        assert _gap(with_carry[0], fresh[0]) > 1e-6
    # copies and the checkpoint form
    clone = clone_carry(carry)
    assert isinstance(clone, WindowCarry) and clone.reader[0] is not carry.reader[0]
    _assert_equal(tuple(clone), tuple(carry), "clone")
    packed = pack_carry(carry)
    assert set(packed) == {"address", "reader"} and len(packed["reader"]) == 3
    address, reader = split_packed_carry(cfg.model, packed, "ckpt")
    assert torch.equal(address, carry.address) and reader is not None
    _assert_equal(reader, list(carry.reader), "packed")
    assert split_packed_carry(cfg.model, None, "ckpt") == (None, None)
    assert split_packed_carry(plain.cfg, packed["address"], "ckpt")[1] is None
    with pytest.raises(ValueError, match="written without the key"):
        split_packed_carry(cfg.model, packed["address"], "ckpt")
    assert clone_carry(None) is None and pack_carry(None) is None
    assert isinstance(clone_carry(bytes_alone), Tensor)
    assert isinstance(pack_carry(bytes_alone), Tensor)


def test_evaluation_carries_the_reader() -> None:
    """``evaluate_windows`` reads consecutive windows with the carry: its loss is the carried
    reading's, not the blind one's."""
    carried, plain, cfg = _pair(*NGRAM)
    T = cfg.train.unroll
    tokens, ctx, reset = _stream(cfg, 3)
    targets = torch.roll(tokens, -1, dims=1)
    mask = torch.ones_like(reset)
    mask[:, -1] = False

    def windows() -> list[dict[str, Tensor]]:
        return [
            {
                "tokens": tokens[:, lo : lo + T],
                "context": ctx[:, lo : lo + T],
                "reset": reset[:, lo : lo + T],
                "targets": targets[:, lo : lo + T],
                "loss_mask": mask[:, lo : lo + T],
                "doc": torch.zeros(3, T, dtype=torch.int64),
            }
            for lo in range(0, tokens.shape[1], T)
        ]

    def loss(logits: Tensor) -> float:
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="none"
        )
        return float(nll[mask.reshape(-1)].double().mean())

    device = torch.device("cpu")
    got = evaluate_windows(carried, windows(), DEFAULT_METRICS, device=device)
    want = loss(_read(carried, (tokens, ctx, reset), T)[0])
    blind = loss(_read(plain, (tokens, ctx, reset), T)[0])
    assert got["loss"] == pytest.approx(want, abs=1e-9)
    assert abs(want - blind) > 1e-3
    assert evaluate_windows(plain, windows(), DEFAULT_METRICS, device=device)["loss"] == (
        pytest.approx(blind, abs=1e-9)
    )


# --------------------------------------------------------------------------------------------------
# The compiled window
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager_and_compiles_once(
    form: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader runs in the eager prologue, so the carry changes no graph: over five
    consecutive windows the compiled window (top-K, straight-through gates, A9's features, the
    address gate) equals the eager one (values, the next cache, the carried address, the
    parameter gradients), compiles one graph and keeps it."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    T = 20
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        STE,
        CUT_C,
        *READER,
        *NGRAM,
        "model.address_gate=true",
        "model.address_gate_input=summary",
        CARRY,
        *FORMS[form],
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=True, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(True)
    window = p28._window(model, p28._ctx())
    tol = p28.TOL[("cpu", "fp32")]
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    graphs = -1
    for i, (tokens, ctx, reset) in enumerate(_compiled_windows(T)):
        ref = p28._run(functools.partial(model, tokens, ctx, state, reset, prev_tokens=carry))
        compiled = functools.partial(window, tokens, ctx, state, reset, prev_tokens=carry)
        if i == 0:
            got = p28._run(compiled)
            graphs = counters["stats"]["unique_graphs"]
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(compiled)
        assert set(READER_CACHE_KEYS) <= set(got[2])
        diff = p28._max_diff(got, ref)  # the cache's valid, the carried address: exact
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], (i, diff)
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"], i
        carry = address_history(cfg.model, tokens, carry, ref[2])
        assert isinstance(carry, WindowCarry)
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs
    assert window.calls == 5


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["per_byte", "word_events_cread"])
def test_recipe_with_the_carry_trains(form: str, tmp_path: Path) -> None:
    """Thirty TBPTT steps of the small recipe with the carry: finite losses that fall, and the
    trainer holds the carry of its last window."""
    steps = 30
    words = ("model.word_contexts=[1,2,3,4,5]",) if form.startswith("word") else ()
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        "train.gate_mode=ste",
        "train.lr=3e-3",
        *READER,
        *NGRAM,
        CARRY,
        *FORMS[form],
        *words,
    )
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    carry = trainer._prev_tokens
    assert isinstance(carry, WindowCarry)
    assert carry.reader[0].shape[1] == cfg.train.batch and carry.reader[0].shape[3] == C


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("form", ["per_byte", "word_events_cread"])
def test_compiled_training_with_the_carry_cuda(form: str, precision: str, tmp_path: Path) -> None:
    """On the card: twelve steps of the small recipe with the carry through the compiled window
    with CUDA Graphs (fp32 and bf16 autocast): finite losses, and the trainer holds the carry
    on the device with the cache in the parameters' dtype. (Whether the full configuration
    compiles one graph and steps at the expected rate is
    ``scripts/word_address_gpu_compile_check.py`` on an arm of ``text_carry_e1.yaml``.)"""
    from cards import CUDA_PROFILE

    p28._cuda_ctx()
    steps = 12
    words = ("model.word_contexts=[1,2,3,4,5]",) if form.startswith("word") else ()
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        "train.gate_mode=ste",
        *READER,
        *NGRAM,
        CARRY,
        *FORMS[form],
        *words,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        f"hardware.autocast_dtype={precision}",
        f"train.precision={precision}",
    )
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps and all(math.isfinite(s.loss) for s in history.steps)
    carry = trainer._prev_tokens
    assert isinstance(carry, WindowCarry)
    keys, values, valid = carry.reader
    assert keys.device.type == "cuda" and valid.device.type == "cuda"
    assert keys.dtype == values.dtype == torch.float32 and bool(valid.any())


@pytest.mark.parametrize("segments", [1, 2])
def test_resume_continues_bit_for_bit(segments: int, tmp_path: Path) -> None:
    """A run resumed from its step-2 checkpoint (other initial weights) continues the
    uninterrupted run bit for bit: records, weights, optimiser moments, the carried cell state
    and the carry (address bytes and the reader's cache), on documents that run over several
    windows; also with a gradient segment of two windows (``train.tbptt_windows=2``: the cache
    stays detached). The checkpoint holds the cache on the CPU, in plain containers."""
    cfg = hsn_config(
        4,
        *HSNT_RECIPE,
        STREAM,
        CARRY,
        f"train.grad_accum={2 * segments}",
        f"train.tbptt_windows={segments}",
        "train.q_resync_every=3",
        "train.ckpt_every=2",
        "train.stats_every=2",
    )
    assert reader_carried(cfg.model)
    train_ds, dev_ds = _long_datasets()
    trainer_a = _trainer(cfg, tmp_path / "a")
    history_a = trainer_a.fit(train_ds, dev_ds, 4)
    checkpoint = tmp_path / "a" / "step_00000002.pt"
    payload = torch.load(checkpoint, weights_only=True)["extra"]["prev_tokens"]
    assert set(payload) == {"address", "reader"}
    keys, values, valid = payload["reader"]
    assert keys.device.type == "cpu" and not keys.requires_grad and not values.requires_grad
    assert keys.shape == (2, cfg.train.batch, 2, C, 4) and bool(valid.any())
    trainer_b = _trainer(cfg, tmp_path / "b", seed=1)
    assert trainer_b.resume(checkpoint) == 2
    assert isinstance(trainer_b._prev_tokens, WindowCarry)
    history_b = trainer_b.fit(train_ds, dev_ds, 4)
    for a, b in zip(history_a.steps[2:], history_b.steps, strict=True):
        assert (a.step, a.loss, a.total, a.parts, a.grad_norm, a.windows) == (
            b.step,
            b.loss,
            b.total,
            b.parts,
            b.grad_norm,
            b.windows,
        )
    _assert_equal(trainer_a.model.state_dict(), trainer_b.model.state_dict(), "weights")
    _assert_equal(trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict(), "optimiser")
    state_a, state_b = trainer_a._carried_state, trainer_b._carried_state
    assert isinstance(state_a, CellGraphState) and isinstance(state_b, CellGraphState)
    for name in ("h", "q", "age", "pos"):
        assert torch.equal(getattr(state_a, name), getattr(state_b, name)), name
    carry_a, carry_b = trainer_a._prev_tokens, trainer_b._prev_tokens
    assert isinstance(carry_a, WindowCarry) and isinstance(carry_b, WindowCarry)
    _assert_equal(tuple(carry_a), tuple(carry_b), "carry")
    # a checkpoint written without the key is refused by name, not read blind
    with pytest.raises(ValueError, match="written without the key"):
        split_packed_carry(cfg.model, payload["address"], str(checkpoint))
