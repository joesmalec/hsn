"""The byte-content options (``docs/key_content.md``) with main's features since f3f271a.

The options were built on f3f271a (15 Nov 2026); main then added the reader's carried keys and
values (``model.reader_carry``), reach dropout (``train.reach_dropout``), the loop over word
events (``model.write_events = word``), the attention read of the cells
(``model.cell_attention_heads``), gradient segments across windows (``train.tbptt_windows``),
the generic trainer's host reads and fused AdamW (``train.host_sync``, ``train.fused_adamw``),
HSN-T without its memory in more recipes, and the Transformer-XL and Gated DeltaNet baselines.
These tests hold the merge to two things:

* at their defaults the options change nothing of main: every configuration of main keeps the
  configuration hash, memory key and arm id that main (c5bfa5a) computes;
* with the options on, every combination either works as the options say (only the content
  reads of ``E`` and of the reader change, at exactly the positions the options name, with the
  carried cache holding what its windows read) or is refused with a clear error.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.embeddings import replace_content
from hsn.model.hsn import CONTENT_STATE_KEYS, HSN, READER_CACHE_KEYS, WindowCarry, address_history
from hsn.model.hsn_t import HSNT, HSNTNoMemory
from hsn.runner import build_model, warm_start
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5, _cfg, _start
from test_key_content import (
    BLANKS,
    CUT_C,
    DROP,
    PROTOCOL,
    TO_UNKNOWN,
    UNKNOWN_ROWS,
    UNSEEN_UNKNOWN,
    ZERO,
    _arm_id,
    _assert_same_window,
    _build,
    _build_unknown,
    _half_seen,
    _outside_rows,
)
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_reader_carry_memory import _stream
from test_word_address import _compiled_windows
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
BASELINES = REPO / "configs" / "baselines"
MEMORY = EXPERIMENTS / "hsn_t_memory.yaml"

#: HSN-T's whole-window reader of the A6 tests' six-cell model (d = 8)
READER = (
    "model_name=hsn_t",
    "model.reader=transformer",
    "model.reader_width=8",
    "model.reader_heads=2",
)
NGRAM = ("model.address_ngram=2", "model.address_history=ngram")
C = 5  # the carry of the small models (their window is 12 bytes)
CARRY = f"model.reader_carry={C}"
WORD = ("model.write_events=word", "model.word_address=exact")
CREAD = "model.cell_attention_heads=2"
#: main's forms of HSN-T with its memory that read a byte's content: per byte, with the carried
#: reader, with reach dropout (whole window and carried), the loop over word events, the
#: attention read of the cells
FORMS: dict[str, tuple[str, ...]] = {
    "carry": (*NGRAM, CARRY),
    "carry_cread": (*NGRAM, CARRY, CREAD),
    "reach": (),
    "carry_reach": (*NGRAM, CARRY),
    "word_events": (*NGRAM, *WORD),
    "word_events_carry_cread": (*NGRAM, *WORD, CARRY, CREAD),
}
REACH_SPAN = 3


# --------------------------------------------------------------------------------------------------
# The defaults: main's identities
# --------------------------------------------------------------------------------------------------

#: (configuration hash, memory key, arm id) of every configuration file under ``configs/`` that
#: loads as a run, as main computes them (c5bfa5a, 20 Nov 2026; ``PROTOCOL`` and variant
#: ``pdg`` as in ``tests/test_key_content.py``).
MAIN_C5BFA5A: dict[str, tuple[str, str, str]] = {
    "configs/_example.yaml": (
        "c0e69378254263768cd31059e1a8b55b22371aa623c504dcab8ff54e3726368e",
        "ab5d52cceaa43dcd",
        "b888ebf3df64c138",
    ),
    "configs/baselines/gated_deltanet_2m.yaml": (
        "619ad30afcb25bede747d70f0320f28de0661e9608aec9185a77b7bc22d6c1b8",
        "dbae876a53472e3e",
        "37022028dac5e3c8",
    ),
    "configs/baselines/gru_2m.yaml": (
        "c8dcbaa48e3b27ecdde3d078359ab95f7482f49935d076bed34f521aeb88cf60",
        "4bf72c523631d8d7",
        "749d07351397f01b",
    ),
    "configs/baselines/mamba_2m.yaml": (
        "9f3bab36efded64a82fcb4299ab35ba5a33b1d89def2b105e391109f882765d6",
        "4ba27af44e8c2693",
        "17f9a1238c19e8e2",
    ),
    "configs/baselines/reader_w64l2.yaml": (
        "9553d883a4bc17c73dc26c04b2e779f1c6906ec13d779095216f83f838a54bbf",
        "12e81665c599d148",
        "d68be918123f9874",
    ),
    "configs/baselines/transformer_2m.yaml": (
        "65066a8915b79a328801b42780ae621f7185f23b553247a70c4f1b4dbd75071f",
        "cb4836305499cc91",
        "3d975c8e13b0e3e9",
    ),
    "configs/baselines/transformer_xl_2m.yaml": (
        "36525fc92a0602824946534e544d330bd197441b3741fb0d2dc9c3e42830b220",
        "1419d72e94680983",
        "20736d1b186e5784",
    ),
    "configs/experiments/g1_ref_a1.yaml": (
        "ff3b6a42c8827327269edeeaf8704781b2f96d8c89205b3ea02fb73044c2f4a8",
        "ec2426311b89e984",
        "3c335991e29ac7f1",
    ),
    "configs/experiments/hsn_recipe_a6.yaml": (
        "d8b36a25971c9694b25251b4a6f20545c617e6d2796b7e4d1ac544acc3e78e28",
        "ed70cf96715a05ef",
        "c42f0b97c13e4c55",
    ),
    "configs/experiments/hsn_recipe_b095.yaml": (
        "59e2910be74e204843c72e15bb79f220527e96276a45f0991843064abdd26cc7",
        "5de7b1bd75a5529a",
        "f0981d224dbb2542",
    ),
    "configs/experiments/hsn_t_memory.yaml": (
        "18c85210eec76722e4fd4c016d9213cd906fe0c11c184d34b04d064907d67f2c",
        "14c19ee8511588e7",
        "e49978fd43c88a9a",
    ),
    "configs/experiments/hsn_t_memory_carry.yaml": (
        "b4cc146fea59ec5c77cf0fbba694f7d8bbc917a8c7061b911a8f77c5cabb6ca4",
        "031619eb8ce49f3c",
        "4c701349350f4cb8",
    ),
    "configs/experiments/hsn_t_memory_words.yaml": (
        "6d520b79442976f238fec5b4ce114a1a71d9b02e649059bdca81390e31ae847c",
        "5045d2995fed5a82",
        "a03415fb9c1f0752",
    ),
    "configs/experiments/hsn_t_memory_words_carry.yaml": (
        "896f6682659d41043d9204f9297c2e989c5d3c8556f42fcfa6262dec2775ac79",
        "7c65cf5d245b9db6",
        "22f725fdbe4e5a09",
    ),
    "configs/experiments/hsn_t_nomem.yaml": (
        "427e8944984cd16906758d1a311b56264140136b4d7618e0c993f0b5e3db6592",
        "c28e8dc1ffe17d21",
        "14b8011fe4ebd308",
    ),
    "configs/experiments/hsn_t_nomem_top0_w1024.yaml": (
        "ddc1f2099be5b1ce2cf406bd4864a3e571ed42bae093aec5fe23b394f7f2f23c",
        "aef9b4defd0af9ad",
        "f4e76f4ab3682fc7",
    ),
    "configs/experiments/hsn_t_nomem_top0_w1024_l3.yaml": (
        "2215d13701407bced99ecbbbb737b8cb13fef778506b7d9749923865c7271227",
        "39e914ddc0d1b12b",
        "ab4bd7f8f7391905",
    ),
    "configs/experiments/hsn_t_nomem_top0_w1024_l3_fast.yaml": (
        "efac564106b082aeafbe67cde906bc7463c0a19b95221d13595898dda6deaa15",
        "cfcff7e5e944fd35",
        "49452fec44604e17",
    ),
    "configs/experiments/p2_3_kv_probes.yaml": (
        "8109a7a8e4b714fe36470155c948fae242eba9c3f9fa1ac7ea64cbaeb74a12be",
        "f935d7d8ade32dc8",
        "052bc608c9396c73",
    ),
    "configs/ref_a1.yaml": (
        "0831924010f9c5ecf6f1794c811f0c61ac96cdb5e3fbcc5a3c99b97a8001aa70",
        "ab5d52cceaa43dcd",
        "b888ebf3df64c138",
    ),
    "configs/tiny_gradcheck.yaml": (
        "1afdc30113f9030179f6110f43401686e57b41878eb0c3cb671b5416c9b8b031",
        "907e472c543a7988",
        "6ea34ec7554e96a9",
    ),
}


def _identity(cfg: RunConfig) -> tuple[str, str, str]:
    return config_hash(cfg), memory_key(cfg, "pdg", PROTOCOL), _arm_id(cfg)


@pytest.mark.parametrize("path", sorted(MAIN_C5BFA5A))
def test_main_configurations_keep_their_identities(path: str) -> None:
    """Every configuration of main, as it is and with the four keys named at their defaults,
    has main's configuration hash, memory key and arm id: the keys at their defaults stay out
    of all three, whatever main added since f3f271a."""
    named = [
        "train.content_dropout=0.0",
        "model.zero_unseen_bytes=false",
        "train.content_dropout_target=zero",
        "model.unseen_bytes_as=own",
    ]
    expected = MAIN_C5BFA5A[path]
    assert _identity(load_config(REPO / path)) == expected
    assert _identity(load_config(REPO / path, named)) == expected


# --------------------------------------------------------------------------------------------------
# Configuration: what refuses, what runs
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        # the baselines and HSN-T without its memory: the generic trainer
        (BASELINES / "transformer_xl_2m.yaml", [DROP], "generic trainer"),
        (BASELINES / "transformer_xl_2m.yaml", [ZERO], "generic trainer"),
        (BASELINES / "gated_deltanet_2m.yaml", [DROP], "generic trainer"),
        (BASELINES / "gated_deltanet_2m.yaml", [ZERO], "generic trainer"),
        (
            BASELINES / "gru_2m.yaml",
            ["hardware.baseline_compile=reduce-overhead", DROP],
            "generic trainer",
        ),
        (EXPERIMENTS / "hsn_t_nomem_top0_w1024.yaml", [DROP], "model.memory=false"),
        (EXPERIMENTS / "hsn_t_nomem_top0_w1024_l3_fast.yaml", [ZERO], "model.memory=false"),
        (
            EXPERIMENTS / "hsn_t_nomem_top0_w1024.yaml",
            [DROP, TO_UNKNOWN, UNSEEN_UNKNOWN],
            "model.memory=false",
        ),
        # the generic trainer's host reads and fused AdamW are not the HSN trainer's
        (MEMORY, [DROP, "train.host_sync=stats", "train.fused_adamw=true"], "host_sync"),
        (MEMORY, [ZERO, "train.fused_adamw=true"], "fused_adamw"),
        # a gradient segment of several windows would read one draw in every window
        (MEMORY, [DROP, "train.grad_accum=2", "train.tbptt_windows=2"], "tbptt_windows=2"),
        (
            MEMORY,
            [DROP, TO_UNKNOWN, UNSEEN_UNKNOWN, "train.grad_accum=2", "train.tbptt_windows=2"],
            "repeats every train.unroll",
        ),
    ],
)
def test_main_combinations_that_refuse(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


@pytest.mark.parametrize(
    ("recipe", "overrides"),
    [
        ("hsn_t_memory_carry.yaml", []),  # the carried reader and the attention read of the cells
        ("hsn_t_memory_carry.yaml", ["train.reach_dropout=0.5", "train.reach_dropout_span=8"]),
        ("hsn_t_memory_words.yaml", []),  # word events
        ("hsn_t_memory_words_carry.yaml", []),
        ("hsn_t_memory.yaml", ["model.reader_span=null", "train.reach_dropout=0.5"]),
        ("hsn_t_memory.yaml", ["hardware.compile_mode=default"]),
        ("hsn_t_memory.yaml", ["model.reader_span=null", "model.reader_mask=causal"]),
        (
            "hsn_t_memory.yaml",
            [
                "hardware.compile_mode=reduce-overhead",
                "hardware.round_products=merged",
                "hardware.message_gradients=state",
            ],
        ),
    ],
)
@pytest.mark.parametrize("blanks", list(BLANKS))
def test_main_combinations_that_run(recipe: str, overrides: list[str], blanks: str) -> None:
    cfg = load_config(EXPERIMENTS / recipe, [*overrides, *BLANKS[blanks]])
    assert cfg.train.content_dropout == 0.3
    assert cfg.model.unseen_bytes_mode == ("zero" if blanks == "zero" else "unknown")


def test_segments_take_the_unseen_bytes_without_the_dropout() -> None:
    """``train.tbptt_windows`` takes the unseen-byte rows (marked before each window, as at one
    window per segment), not the dropout."""
    segments = ["train.grad_accum=2", "train.tbptt_windows=2"]
    assert load_config(MEMORY, [*segments, ZERO]).model.zero_unseen_bytes
    with pytest.raises(ValueError, match="content_dropout_target=unknown"):
        load_config(MEMORY, [*segments, UNSEEN_UNKNOWN])  # its rows are trained by the dropout


# --------------------------------------------------------------------------------------------------
# The window: only the content reads change, also with the carry, reach dropout, word events
# --------------------------------------------------------------------------------------------------


class _Outside:
    """The reference: a model without the options whose two content reads are replaced from
    outside, ``E``'s input (a pre-hook of the encoder) and the reader's embedding lookup (a hook
    on ``reader.emb``, which both :meth:`Reader.forward` and :meth:`Reader.read` call), with a
    zero row where ``zero`` is True and ``rows`` (outside copies of the unknown rows) where
    ``unknown`` is."""

    def __init__(self, model: HSN, rows: tuple[Tensor | None, Tensor | None]) -> None:
        self.zero: Tensor | None = None
        self.unknown: Tensor | None = None
        self.rows = rows
        self._handles = [model.encoder.register_forward_pre_hook(self._encoder)]
        assert model.reader is not None
        self._handles.append(model.reader.emb.register_forward_hook(self._reader))

    def _encoder(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (replace_content(args[0], self.zero, self.unknown, self.rows[0]), *args[1:])

    def _reader(self, module: Any, args: tuple[Any, ...], out: Tensor) -> Tensor:
        return replace_content(out, self.zero, self.unknown, self.rows[1])

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def _pair(form: str, blanks: str) -> tuple[HSN, HSN, RunConfig]:
    """The model of ``form`` with the options ``blanks`` and the same weights without them."""
    words = ("model.word_contexts=[1,2,3]",) if "word" in form else ()
    extra = (*A5, CUT_C, *READER, *FORMS[form], *words)
    cfg = _cfg(*extra, *BLANKS[blanks])
    on = _build_unknown(cfg) if blanks == "unknown" else _build(cfg)
    return on, _build(_cfg(*extra)), cfg


@pytest.mark.parametrize("blanks", list(BLANKS))
@pytest.mark.parametrize("form", list(FORMS))
def test_only_the_content_reads_change_in_main_forms(form: str, blanks: str) -> None:
    """Three consecutive training windows of a stream (documents that start inside a window,
    with a window, two bytes before a boundary) and one evaluation window: with the options
    on, each window is, bit for bit, the window of the model without them whose two content
    reads are replaced from outside at the positions the options name (the bytes not marked
    as seen, and in training the positions drawn for the window's step); every output of
    ``info`` agrees, the reader's next cache included, and so does what is carried into the
    next window. The gradients of the three windows (each from a detached state, as the
    trainer runs them) agree for every shared parameter, and the unknown rows get the
    gradient of their outside copies. With reach dropout both models read with the same
    short rows."""
    on, off, cfg = _pair(form, blanks)
    T = cfg.train.unroll
    stream = _stream(cfg, 4)
    tokens_all, ctx_all, reset_all = stream
    B = int(tokens_all.shape[0])
    _half_seen(on, tokens_all[:, :T])  # the bytes of the first window that count as seen
    assert on.byte_seen is not None
    unknown = blanks == "unknown"
    rows = _outside_rows(on) if unknown else (None, None)
    outside = _Outside(off, rows)
    reach = torch.tensor([True, False, True]) if "reach" in form else None
    state_on, state_off = _start(on, B), _start(off, B)
    prev_on: Tensor | WindowCarry | None = None
    prev_off: Tensor | WindowCarry | None = None
    loss_on: Tensor | None = None
    loss_off: Tensor | None = None
    plain = 0
    try:
        for index, lo in enumerate(range(0, 4 * T, T)):
            training = index < 3
            on.train(training)
            off.train(training)
            sl = slice(lo, lo + T)
            tokens, ctx, reset = tokens_all[:, sl], ctx_all[:, sl], reset_all[:, sl]
            on.set_train_step(3 + index)
            off.set_train_step(3 + index)
            if reach is not None:
                on.set_reach_rows(reach, REACH_SPAN)
                off.set_reach_rows(reach, REACH_SPAN)
            blank = ~on.byte_seen[tokens]
            if training:
                blank = blank | on.content_drop_mask(B, T)
            assert 0 < int(blank.sum()) < blank.numel()
            outside.zero, outside.unknown = (None, blank) if unknown else (blank, None)
            with torch.set_grad_enabled(training):
                got = on(tokens, ctx, state_on, reset, prev_tokens=prev_on)
                ref = off(tokens, ctx, state_off, reset, prev_tokens=prev_off)
            _assert_same_window(got, ref)
            if "carry" in form:
                assert set(READER_CACHE_KEYS) <= set(got[2])
            prev_on = address_history(on.cfg, tokens, prev_on, got[2])
            prev_off = address_history(off.cfg, tokens, prev_off, ref[2])
            state_on, state_off = got[1].detach(), ref[1].detach()
            if training:
                loss_on = p28._loss(got) + (0.0 if loss_on is None else loss_on)
                loss_off = p28._loss(ref) + (0.0 if loss_off is None else loss_off)
            outside.zero = outside.unknown = None
            with torch.no_grad():  # the same window without the options' reads
                plain += int(not torch.equal(off(tokens, ctx, ref[1], reset)[0], got[0]))
            if reach is not None:
                on.set_reach_rows(None)
                off.set_reach_rows(None)
    finally:
        outside.close()
    assert plain >= 3  # the options act on these windows
    assert loss_on is not None and loss_off is not None
    names = [n for n, p in on.named_parameters() if p.requires_grad]
    params = [p for n, p in on.named_parameters() if p.requires_grad]
    grads = dict(zip(names, torch.autograd.grad(loss_on, params, allow_unused=True), strict=True))
    off_names = [n for n, p in off.named_parameters() if p.requires_grad]
    extra = [r for r in rows if r is not None]
    off_grads = torch.autograd.grad(
        loss_off,
        [p for p in off.parameters() if p.requires_grad] + extra,
        allow_unused=True,
    )
    live = 0
    for name, g in zip(off_names, off_grads[: len(off_names)], strict=True):
        assert (g is None) == (grads[name] is None), name
        if g is not None:
            assert torch.equal(grads[name], g), name
            live += int(bool(g.abs().max() > 0))
    assert live >= 20
    assert len(off_grads) - len(off_names) == (len(UNKNOWN_ROWS) if unknown else 0)
    for name, g in zip(UNKNOWN_ROWS, off_grads[len(off_names) :], strict=False):
        assert g is not None and bool(g.any()), name  # the rows are trained in these forms
        assert torch.equal(grads[name], g), name


def test_the_cache_holds_what_its_windows_read() -> None:
    """``model.reader_carry``: a position dropped in its training window is cached with the
    content that window read (its keys and values are not computed again), so the next window
    reads the cached positions of the window before as that window read them. The model with
    the options and the model without them, from the same cache, differ in the window only
    where the options act on it: run from a cache made with the options, the reference reads
    the options' cache, and the outputs match."""
    on, off, cfg = _pair("carry", "zero")
    T = cfg.train.unroll
    tokens_all, ctx_all, reset_all = _stream(cfg, 2)
    B = int(tokens_all.shape[0])
    on.mark_bytes_seen(tokens_all)  # no unseen byte: the dropout alone
    first = slice(0, T)
    on.set_train_step(5)
    got = on(tokens_all[:, first], ctx_all[:, first], _start(on, B), reset_all[:, first])
    carry = address_history(on.cfg, tokens_all[:, first], None, got[2])
    plain = off(tokens_all[:, first], ctx_all[:, first], _start(off, B), reset_all[:, first])
    plain_carry = address_history(off.cfg, tokens_all[:, first], None, plain[2])
    assert isinstance(carry, WindowCarry) and isinstance(plain_carry, WindowCarry)
    assert not torch.equal(carry.reader[0], plain_carry.reader[0])  # the dropped positions
    # the second window in evaluation: nothing is dropped there, so the two models differ in
    # what they read from the cache alone, and the model without the options given the
    # options' cache is the model with them
    on.eval()
    off.eval()
    second = slice(T, 2 * T)
    window = (tokens_all[:, second], ctx_all[:, second], got[1].detach(), reset_all[:, second])
    with torch.no_grad():
        keyed = on(*window, prev_tokens=carry)
        swapped = off(*window, prev_tokens=carry)
        fresh = off(*window, prev_tokens=plain_carry)
    _assert_same_window(keyed, swapped)
    assert not torch.equal(keyed[0], fresh[0])


@pytest.mark.usefixtures("small_blocks")
def test_compiled_event_window_with_the_carry_equals_eager_with_the_unknown_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The options act in the window's prologue, which the compiled window runs eagerly, also
    for the loop over word events with the carried reader and the attention read of the
    cells: over five consecutive training windows (each at its own step, so with its own
    drawn positions) the compiled window equals the eager one (values, the next cache, the
    carried address, every parameter gradient, the unknown rows' among them) and compiles no
    new graph after the first window."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    T = 20
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        CUT_C,
        *READER,
        *NGRAM,
        "model.address_gate=true",
        "model.address_gate_input=summary",
        CARRY,
        *WORD,
        CREAD,
        *BLANKS["unknown"],
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=True, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(True)
    windows = _compiled_windows(T)
    _half_seen(model, windows[0][0])
    window = p28._window(model, p28._ctx())
    tol = p28.TOL[("cpu", "fp32")]
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    graphs = -1
    named = [n for n, p in model.named_parameters() if p.requires_grad]
    for i, (tokens, ctx, reset) in enumerate(windows):
        model.set_train_step(i)
        ref = p28._run(functools.partial(model, tokens, ctx, state, reset, prev_tokens=carry))
        compiled = functools.partial(window, tokens, ctx, state, reset, prev_tokens=carry)
        if i == 0:
            got = p28._run(compiled)
            graphs = counters["stats"]["unique_graphs"]
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(compiled)
        diff = p28._max_diff(got, ref)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], (i, diff)
        grads, grads_ref = p28._grads(model, got), p28._grads(model, ref)
        assert p28._grad_diff(grads, grads_ref) <= tol["grads"], i
        for name in UNKNOWN_ROWS:
            grad = grads[named.index(name)]
            assert grad is not None and bool(grad.any()), (i, name)
        carry = address_history(cfg.model, tokens, carry, ref[2])
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs
    assert window.calls == 5


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------

TRAINED = (
    *A5,
    CUT_C,
    "model.cand_norm=rms",
    "model.k_select=3",
    "train.soft_until_step=0",
    "train.gate_mode=ste",
    "train.lr=3e-3",
    *READER,
    *NGRAM,
)
TRAINED_FORMS: dict[str, tuple[str, ...]] = {
    "carry_reach": (CARRY, "train.reach_dropout=0.5", f"train.reach_dropout_span={REACH_SPAN}"),
    "word_events_carry_cread": (*WORD, "model.word_contexts=[1,2,3,4,5]", CARRY, CREAD),
}


@pytest.mark.parametrize("form", list(TRAINED_FORMS))
def test_main_recipes_train_with_the_unknown_rows(form: str, tmp_path: Path) -> None:
    """Thirty TBPTT steps of the small recipe with the carried reader and reach dropout, and
    with word events, the carried reader and the attention read of the cells, all with
    dropout to the learned unknown rows and the unseen bytes reading them: finite losses that
    fall, both rows trained, the bytes of the training windows marked, and the trainer holds
    the carry of its last window."""
    steps = 30
    cfg = hsn_config(steps, *TRAINED, *TRAINED_FORMS[form], DROP, TO_UNKNOWN, UNSEEN_UNKNOWN)
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    start = [r.detach().clone() for r in trainer.hsn.unknown_rows() if r is not None]
    assert len(start) == 2
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    assert history.evals and math.isfinite(history.evals[-1]["loss"])
    rows = [r for r in trainer.hsn.unknown_rows() if r is not None]
    for row, before in zip(rows, start, strict=True):
        assert bool(torch.isfinite(row).all()) and not torch.equal(row.detach(), before)
    assert trainer.hsn.byte_seen is not None and bool(trainer.hsn.byte_seen.any())
    assert isinstance(trainer._prev_tokens, WindowCarry)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("form", list(TRAINED_FORMS))
def test_compiled_training_with_the_unknown_rows_cuda(
    form: str, precision: str, tmp_path: Path
) -> None:
    """On the card: twelve steps of :func:`test_main_recipes_train_with_the_unknown_rows`'s
    recipes through the compiled window with CUDA Graphs (fp32 and bf16 autocast): finite
    losses, compiled windows, both unknown rows trained and finite, the seen bytes and the
    reader's carry on the device."""
    from cards import CUDA_PROFILE

    p28._cuda_ctx()
    steps = 12
    cfg = hsn_config(
        steps,
        *TRAINED,
        *TRAINED_FORMS[form],
        DROP,
        TO_UNKNOWN,
        UNSEEN_UNKNOWN,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        f"hardware.autocast_dtype={precision}",
        f"train.precision={precision}",
    )
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    start = [r.detach().clone() for r in trainer.hsn.unknown_rows() if r is not None]
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps and all(math.isfinite(s.loss) for s in history.steps)
    assert history.measured["compiled_windows"] > 0.0
    rows = [r for r in trainer.hsn.unknown_rows() if r is not None]
    assert len(rows) == len(start) == 2
    for row, before in zip(rows, start, strict=True):
        assert row.device.type == "cuda" and bool(torch.isfinite(row).all())
        assert not torch.equal(row.detach(), before)
    seen = trainer.hsn.byte_seen
    assert seen is not None and seen.device.type == "cuda" and bool(seen.any())
    carry = trainer._prev_tokens
    assert isinstance(carry, WindowCarry) and carry.reader[0].device.type == "cuda"


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("blanks", list(BLANKS))
def test_compiled_event_window_with_the_carry_equals_eager_cuda(
    blanks: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the card (inductor, CUDA Graphs: warm-up, recording, replay, fp32):
    :func:`test_compiled_event_window_with_the_carry_equals_eager_with_the_unknown_rows` for
    both option sets. Over five consecutive training windows, each at its own step (so with
    its own drawn positions, copied to the card in the eager prologue), the compiled window
    equals the eager one and compiles no new graph after the first window."""
    from torch._dynamo.utils import counters

    ctx = p28._cuda_ctx()
    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    T = 20
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        CUT_C,
        *READER,
        *NGRAM,
        "model.address_gate=true",
        "model.address_gate_input=summary",
        CARRY,
        *WORD,
        CREAD,
        *BLANKS[blanks],
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=True, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model = model.to(ctx.device).train(True)
    windows = [tuple(t.to(ctx.device) for t in w) for w in _compiled_windows(T)]
    _half_seen(model, windows[0][0])
    window = p28._window(model, ctx, backend="inductor")
    tol = p28.TOL[("cuda", "fp32")]
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    graphs = -1
    for i, (tokens, cx, reset) in enumerate(windows):
        model.set_train_step(i)
        ref = p28._run(functools.partial(model, tokens, cx, state, reset, prev_tokens=carry))
        compiled = functools.partial(window, tokens, cx, state, reset, prev_tokens=carry)
        if i == 0:
            got = p28._run(compiled)
            graphs = counters["stats"]["unique_graphs"]
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(compiled)
        diff = p28._max_diff(got, ref)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], (i, diff)
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"], i
        carry = address_history(cfg.model, tokens, carry, ref[2])
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs
    assert window.calls == 5


# --------------------------------------------------------------------------------------------------
# Warm start and the model without its memory
# --------------------------------------------------------------------------------------------------


def _checkpoint(cfg: RunConfig, run_dir: Path) -> Path:
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)
    history = trainer.fit(*datasets(), cfg.train.steps)
    return Path(history.checkpoints[-1])


def test_warm_start_needs_the_source_runs_options(tmp_path: Path) -> None:
    """``train.init_from`` loads strictly: a checkpoint loads into a model with the same
    seen bytes and unknown rows, and every mismatch is refused with the options to set
    (without the rows into a model with them and the reverse; without the seen bytes into a
    model that reads never-trained bytes as unknown, and the reverse)."""
    steps = 2
    base = hsn_config(steps, *TRAINED)
    keyed = hsn_config(steps, *TRAINED, DROP, TO_UNKNOWN, UNSEEN_UNKNOWN)
    rows_only = hsn_config(steps, *TRAINED, DROP, TO_UNKNOWN)
    plain_ckpt = _checkpoint(base, tmp_path / "plain")
    keyed_ckpt = _checkpoint(keyed, tmp_path / "keyed")
    rows_ckpt = _checkpoint(rows_only, tmp_path / "rows")
    # the same options: the tensors are taken over
    warm = build_model(keyed)
    assert isinstance(warm, HSN) and warm_start(warm, keyed_ckpt)["step"] == steps
    source = torch.load(keyed_ckpt, map_location="cpu", weights_only=True)["model"]
    for key in CONTENT_STATE_KEYS:
        assert torch.equal(warm.state_dict()[key], source[key]), key
    cases = [
        (keyed, plain_ckpt, "this model has the learned unknown rows"),
        (keyed, plain_ckpt, "the bytes its run trained on are not known"),
        (base, keyed_ckpt, "the checkpoint holds the learned unknown rows"),
        (base, keyed_ckpt, "model.unseen_bytes_as=own"),
        (keyed, rows_ckpt, "no byte_seen"),
        (rows_only, keyed_ckpt, "read never-trained bytes"),
        (rows_only, plain_ckpt, "this model has the learned unknown rows"),
    ]
    for cfg, checkpoint, message in cases:
        with pytest.raises(ValueError, match=message) as caught:
            warm_start(build_model(cfg), checkpoint)
        assert "train.init_from=" in str(caught.value) and "docs/key_content.md" in str(
            caught.value
        )
    # a mismatch that is not the options' keeps the loader's own error
    with pytest.raises(RuntimeError, match="size mismatch"):
        warm_start(build_model(hsn_config(steps, *TRAINED, "model.reader_width=16")), plain_ckpt)


def test_the_model_without_its_memory_refuses_the_options_tensors() -> None:
    """``HSNTNoMemory.load_memory_on_state`` takes the six modules' tensors of an HSN-T with
    its memory; the options' tensors have no place there, and are refused rather than
    dropped."""
    nomem = load_config(
        EXPERIMENTS / "hsn_t_nomem.yaml",
        ["model.d=8", "model.context_dim=8", "model.reader_width=8", "model.reader_heads=2"],
    )
    off = HSNTNoMemory.from_config(nomem)
    for options in ((ZERO,), (DROP, TO_UNKNOWN), BLANKS["unknown"]):
        cfg = load_config(
            MEMORY,
            ["model.d=8", "model.context_dim=8", "model.reader_width=8", "model.reader_heads=2"]
            + ["model.n_cells=8", "model.k_select=2", "model.hidden=16", "model.scorer_hidden=8"]
            + list(options),
        )
        seed_everything(0)
        with pytest.raises(ValueError, match="byte-content options"):
            off.load_memory_on_state(HSNT.from_config(cfg).state_dict())
