"""The reader's causal mask (``model.reader_mask = causal``, ``docs/reader_causal_flash.md``,
17 Nov 2026).

With the key the reader's attention is ``scaled_dot_product_attention(..., is_causal=True)`` with
no mask (the flash kernel on CUDA) in place of the additive document mask. These tests check that
the default leaves every hash, memory key and output as it was; that the key is refused wherever
the two masks can differ; that the loader's ``data.align_docs`` windows have their resets at
column 0 only, the condition under which the two are the same mask; that the causal reader then
equals the document reader (outputs and gradients, float64 and float32, rotary and learned
positions; HSN-T without and with its memory; the generic and the TBPTT trainer; compiled and
eager); that a reset after column 0 is caught on the device; and, on a card, that the flash
kernel runs, compiled under CUDA graphs as one graph, with no host synchronisation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    ModelConfig,
    canonical_json,
    config_hash,
    load_config,
)
from hsn.data.dataset import ByteDataset, collate_windows, iter_windows
from hsn.data.synthetic import Example
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.hsn_t import HSNT, HSNTNoMemory
from hsn.model.reader import Reader
from hsn.runner import build_model, make_trainer
from hsn.train.loop import StreamLoader, Trainer
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, train_overrides
from test_a6_feedback_detach import A5, _cfg, _start
from test_reader_rotary_table import _reference_forward
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
NOMEM_160x3 = REPO / "configs" / "experiments" / "hsn_t_nomem_top0_w1024_l3.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"

CAUSAL = ("model.reader_mask=causal",)
#: the memory-off recipe at a size the CPU runs in a moment (two heads of four pairs)
TINY = ("model.d=8", "model.context_dim=8", "model.hidden=16", "model.reader_width=16")
TINY = (*TINY, "model.reader_heads=2")
#: the six-cell HSN-T of the A6 tests (its memory on)
HSN_T = (*A5, "model_name=hsn_t", "model.reader=transformer", "model.reader_width=8")
HSN_T = (*HSN_T, "model.reader_heads=2")
#: position settings the causal mask must leave alone
POSITIONS = [
    {},
    {"reader_rotary_pairs": [4, 0]},
    {"reader_rotary_dtype": "parameters"},
    {"reader_rotary_base": 1e6},
    {"reader_position": "learned"},
]


def _gap(a: Tensor, b: Tensor) -> float:
    return float((a - b).detach().abs().max())


def _reader(mask: str, dtype: torch.dtype = torch.float64, **keys: object) -> Reader:
    """A reader of width 16 with two heads and two blocks, away from its start so that every
    weight matters; the same weights for either mask."""
    cfg = ModelConfig(
        reader="transformer",
        reader_width=16,
        reader_heads=2,
        reader_mask=mask,  # type: ignore[arg-type]
        **keys,  # type: ignore[arg-type]
    )
    reader = Reader(cfg, seed=0).to(dtype)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in reader.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(dtype))
    return reader.eval()


def _aligned(B: int, T: int, seed: int) -> tuple[Tensor, Tensor]:
    """Bytes and a reset of ``data.align_docs`` windows: column 0 resets in some rows (a document
    starts) and not in the others (a document goes on from the window before), nothing after."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 256, (B, T), generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = torch.rand(B, generator=g) < 0.5
    reset[0, 0], reset[-1, 0] = True, False
    return tokens, reset


def _same(got: list[float], ref: list[float], tol: float) -> bool:
    """Equal to ``tol``, NaN where the other is NaN (a step of windows without a loss
    position records NaN)."""
    return len(got) == len(ref) and all(
        (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol
        for a, b in zip(got, ref, strict=True)
    )


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite)


def _grads(module: torch.nn.Module) -> dict[str, Tensor]:
    return {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


# --------------------------------------------------------------------------------------------------
# The key: opt-in, hashes, memory keys, refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (NOMEM, ()),
        (NOMEM_160x3, ()),
        (READER_LM, ()),
        (RECIPE_A6, ("model_name=hsn_t", "model.reader=transformer", "model.k_select=12")),
    ],
)
def test_key_is_opt_in_and_the_default_keeps_hashes_and_memory_keys(
    path: Path, overrides: tuple[str, ...]
) -> None:
    base = load_config(path, list(overrides))
    assert base.model.reader_mask == "document"
    assert "reader_mask" not in json.loads(canonical_json(base))["model"]
    assert "model.reader_mask" in HASH_OMIT_AT_DEFAULT
    named = load_config(path, [*overrides, "model.reader_mask=document"])
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert config_hash(named) == config_hash(base)
    assert memory_key(named, "none", protocol) == memory_key(base, "none", protocol)
    causal = load_config(path, [*overrides, *CAUSAL])
    assert causal.model.reader_mask == "causal"
    assert json.loads(canonical_json(causal))["model"]["reader_mask"] == "causal"
    assert config_hash(causal) != config_hash(base)
    assert memory_key(causal, "none", protocol) != memory_key(base, "none", protocol)


def test_the_memory_keys_of_the_arms_before_the_key_are_unchanged() -> None:
    """The values ``tests/test_reader_positions.py`` pins from main before the reader keys."""
    protocol = {"name": "reads_state", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(load_config(RECIPE_A6), "pdg", protocol) == "c33ec88833506533"
    hybrid = ["model_name=hsn_t", "model.reader=transformer", "model.k_select=12"]
    assert memory_key(load_config(RECIPE_A6, hybrid), "pdg", protocol) == "c8bcf4024c9285fe"


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, [*CAUSAL], "set model.reader=transformer or leave it at document"),
        (GRU_2M, [*CAUSAL], "set model.reader=transformer or leave it at document"),
        (READER_LM, [*CAUSAL, "model.reader_span=8"], "leave model.reader_span null"),
        (NOMEM, [*CAUSAL, "model.reader_span=8"], "leave model.reader_span null"),
        (NOMEM, [*CAUSAL, "model.reader_carry=16"], "leave model.reader_carry at 0"),
        (READER_LM, [*CAUSAL, "model.reader_carry=16"], "leave model.reader_carry at 0"),
        (
            NOMEM,
            [
                *CAUSAL,
                "model.reader_carry=16",
                "model.reader_top_words=4",
                "model.reader_rotary_pairs=[8,0]",
            ],
            "leave model.reader_carry at 0",
        ),
        (NOMEM, [*CAUSAL, "data.align_docs=false"], "needs data.align_docs=true"),
        (READER_LM, [*CAUSAL, "data.align_docs=false"], "needs data.align_docs=true"),
        (
            RECIPE_A6,
            [*CAUSAL, "model_name=hsn_t", "model.reader=transformer", "data.align_docs=false"],
            "needs data.align_docs=true",
        ),
        (READER_LM, ["model.reader_mask=block"], "reader_mask"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_the_reader_refuses_a_span_or_a_carry_it_was_handed_past_the_configuration() -> None:
    for keys in ({"reader_span": 4}, {"reader_carry": 4}):
        cfg = ModelConfig(reader="transformer", reader_width=16, reader_heads=2, **keys)  # type: ignore[arg-type]
        cfg = cfg.model_copy(update={"reader_mask": "causal"})  # no validation: a direct build
        with pytest.raises(ValueError, match="only without model.reader_span"):
            Reader(cfg)


def test_accepted_settings_and_repr() -> None:
    for path, extra in (
        (NOMEM, []),
        (NOMEM_160x3, []),
        (READER_LM, ["model.reader_position=learned"]),
        (RECIPE_A6, ["model_name=hsn_t", "model.reader=transformer", "model.k_select=12"]),
    ):
        cfg = load_config(path, [*CAUSAL, *extra])
        assert cfg.model.reader_mask == "causal" and cfg.data.align_docs
        reader = build_model(cfg).reader
        assert reader.causal and "mask=causal" in repr(reader)
    assert "mask" not in repr(build_model(load_config(NOMEM)).reader)


# --------------------------------------------------------------------------------------------------
# The loader: data.align_docs windows reset at column 0 only
# --------------------------------------------------------------------------------------------------


def _docs(lengths: list[int], seed: int) -> ByteDataset:
    rng = np.random.default_rng(seed)
    examples = []
    for i, n in enumerate(lengths):
        tokens = rng.integers(1, 256, n).astype(np.uint8)
        meta = {
            "task": "t",
            "variant": "v",
            "family": "f",
            "split": "train",
            "index": i,
            "doc_id": f"t:v:train:{i}",
            "dep_len": 1,
            "in_range": True,
            "deps": [1] * n,
        }
        examples.append(
            Example(
                tokens=tokens,
                targets=np.roll(tokens, -1),
                loss_mask=np.ones(n, dtype=np.bool_),
                context=np.zeros(n, dtype=np.uint8),
                meta=meta,
            )
        )
    return ByteDataset(examples)


@pytest.mark.parametrize("T", [1, 7, 16])
@pytest.mark.parametrize("batch", [1, 3])
def test_the_loader_with_align_docs_resets_at_column_0_only(T: int, batch: int) -> None:
    """The guarantee the key rests on (``StreamLoader``: with ``align_docs`` a row that ends a
    document pads to the window's end, and the next document starts at the next window's column
    0): documents shorter than, as long as and longer than a window, shuffled over several
    epochs, the finite evaluation loader, a rollback's fresh documents, and the collated windows
    of :func:`collate_windows`. Without ``align_docs`` resets after column 0 do occur."""
    rng = np.random.default_rng(T * 10 + batch)
    lengths = [int(x) for x in rng.integers(1, 3 * T + 2, 40)] + [T, 2 * T, 1]
    data = _docs(lengths, seed=T)
    starts = 0
    for shuffle, epochs, n in ((True, None, 200), (False, 1, 10_000)):
        loader = StreamLoader(
            data, batch=batch, unroll=T, seed=5, shuffle=shuffle, align_docs=True, epochs=epochs
        )
        for i, window in enumerate(loader):
            reset = window["reset"]
            assert not bool(reset[:, 1:].any()), (shuffle, i)
            # a row that starts a document at column 0 holds it from there (no padding before)
            assert bool((window["doc"][:, 0][reset[:, 0]] >= 0).all())
            starts += int(reset[:, 0].sum())
            if i == 50 and epochs is None:
                loader.drop_documents()  # P1.16's rollback: fresh documents in every row
                after = next(loader)["reset"]
                assert bool(after[:, 0].all()) and not bool(after[:, 1:].any())
            if i + 1 >= n:
                break
    assert starts > len(lengths)
    collated = collate_windows(data.examples, T, batch=batch, align_docs=True)
    assert all(not bool(w["reset"][:, 1:].any()) for w in iter_windows(collated, T))
    if T > 1:  # the control: without align_docs documents start inside windows
        loose = StreamLoader(
            data, batch=batch, unroll=T, seed=5, shuffle=False, align_docs=False, epochs=1
        )
        assert any(bool(w["reset"][:, 1:].any()) for w in loose)


# --------------------------------------------------------------------------------------------------
# The reader: the causal mask is the document mask on aligned windows
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("keys", POSITIONS, ids=lambda k: ",".join(k) or "default")
def test_causal_equals_document_on_aligned_windows_float64(keys: dict[str, object]) -> None:
    doc, causal = _reader("document", **keys), _reader("causal", **keys)
    causal.load_state_dict(doc.state_dict())
    for seed in range(3):
        tokens, reset = _aligned(4, 37, seed)
        outs = []
        for reader in (doc, causal):
            reader.zero_grad(set_to_none=True)
            out = reader(tokens, reset)
            out.square().sum().backward()
            outs.append((out.detach(), _grads(reader)))
        (ref, ref_g), (got, got_g) = outs
        assert _gap(got, ref) <= 1e-12
        assert ref_g.keys() == got_g.keys() and all(
            _gap(got_g[n], ref_g[n]) <= 1e-11 for n in ref_g
        )


def test_causal_equals_document_in_float32_and_under_bf16_autocast() -> None:
    doc, causal = _reader("document", torch.float32), _reader("causal", torch.float32)
    causal.load_state_dict(doc.state_dict())
    tokens, reset = _aligned(3, 64, 4)
    assert _gap(causal(tokens, reset), doc(tokens, reset)) <= 1e-5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a, b = causal(tokens, reset).float(), doc(tokens, reset).float()
    assert _gap(a, b) <= 0.05  # bf16 rounding of two attention kernels


def test_the_default_is_the_reader_as_it_was_bit_for_bit() -> None:
    """``reader_mask = document``, named or not, is the forward of before the key
    (``_reference_forward``, the code kept in ``tests/test_reader_rotary_table.py``)."""
    base = ModelConfig(reader="transformer", reader_width=16, reader_heads=2)
    named = base.model_copy(update={"reader_mask": "document"})
    a, b = Reader(base, seed=3).eval(), Reader(named, seed=3).eval()
    assert not a.causal and not b.causal and repr(a) == repr(b)
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 256, (3, 40), generator=g)
    reset = torch.rand(3, 40, generator=g) < 0.2  # resets inside the window, too
    reset[:, 0] = True
    with torch.no_grad():
        assert torch.equal(a(tokens, reset), _reference_forward(a, tokens, reset))
        assert torch.equal(b(tokens, reset), a(tokens, reset))


def test_the_two_masks_differ_where_a_document_starts_inside_the_window() -> None:
    """The control of the equality above: the document mask cuts at a reset after column 0,
    and the causal reader refuses such a window (on the CPU at once; on CUDA a device-side
    assert)."""
    doc, causal = _reader("document"), _reader("causal")
    causal.load_state_dict(doc.state_dict())
    tokens, reset = _aligned(2, 20, 0)
    late = reset.clone()
    late[1, 9] = True
    with torch.no_grad():
        assert _gap(doc(tokens, late), doc(tokens, reset)) > 1e-3
        with pytest.raises(RuntimeError, match="reset after column 0"):
            causal(tokens, late)


def test_the_guard_runs_compiled_and_adds_no_graph_break() -> None:
    from torch._dynamo.utils import counters

    reader = _reader("causal")
    compiled = torch.compile(reader, backend="aot_eager", fullgraph=True)
    tokens, reset = _aligned(2, 16, 1)
    try:
        with torch.no_grad():
            assert _gap(compiled(tokens, reset), reader(tokens, reset)) <= 1e-12
            graphs = counters["stats"]["unique_graphs"]
            late = reset.clone()
            late[0, 3] = True
            with pytest.raises(RuntimeError, match="reset after column 0"):
                compiled(tokens, late)
            assert counters["stats"]["unique_graphs"] == graphs
    finally:
        torch._dynamo.reset()


# --------------------------------------------------------------------------------------------------
# The models and the trainers
# --------------------------------------------------------------------------------------------------


def test_memory_off_model_causal_equals_document_and_compiles_as_one_graph() -> None:
    """HSN-T without its memory in float64: the causal model equals the document model, and
    compiled (``aot_eager``, ``fullgraph``) equals eager, the second window reusing the graph."""
    from torch._dynamo.utils import counters

    seed_everything(0)
    doc = build_model(load_config(NOMEM, list(TINY))).double().train()
    causal = build_model(load_config(NOMEM, [*TINY, *CAUSAL])).double().train()
    assert isinstance(causal, HSNTNoMemory) and causal.reader.causal
    causal.load_state_dict(doc.state_dict())
    compiled = torch.compile(causal, backend="aot_eager", fullgraph=True)
    g = torch.Generator().manual_seed(3)
    try:
        graphs = None
        for _ in range(2):
            tokens, reset = _aligned(3, 24, int(torch.randint(0, 99, (), generator=g)))
            ctx = torch.randint(0, 16, (3, 24), generator=g)
            results = []
            for model, forward in ((doc, doc), (causal, causal), (causal, compiled)):
                model.zero_grad(set_to_none=True)
                logits, state, info = forward(tokens, ctx, None, reset)
                logits.square().mean().backward()
                results.append((logits.detach(), _grads(model)))
                assert state is None and info == {}
            (ref, ref_g), *others = results
            for got, got_g in others:
                assert _gap(got, ref) <= 1e-12
                assert all(_gap(got_g[n], ref_g[n]) <= 1e-12 for n in ref_g)
            if graphs is not None:
                assert counters["stats"]["unique_graphs"] == graphs, "the second window recompiled"
            graphs = counters["stats"]["unique_graphs"]
    finally:
        torch._dynamo.reset()


def test_hsn_t_with_its_memory_causal_equals_document() -> None:
    """HSN-T with its memory reads the reader in the window's prologue: on an aligned window
    (in float64) logits and gradients are the document reader's."""
    cfg_doc, cfg_causal = _cfg(*HSN_T), _cfg(*HSN_T, *CAUSAL)
    seed_everything(0)
    doc = HSNT.from_config(cfg_doc).double().train()
    seed_everything(0)
    causal = HSNT.from_config(cfg_causal).double().train()
    assert causal.reader is not None and causal.reader.causal
    causal.load_state_dict(doc.state_dict())
    B, T = cfg_doc.train.batch, cfg_doc.train.unroll
    tokens, reset = _aligned(B, T, 7)
    tokens = tokens % cfg_doc.model.vocab_size
    ctx = torch.randint(
        0, cfg_doc.model.n_context, (B, T), generator=torch.Generator().manual_seed(8)
    )
    outs = []
    for model in (doc, causal):
        model.zero_grad(set_to_none=True)
        logits, _, _ = model(tokens, ctx, _start(model, B), reset)
        logits.square().mean().backward()
        outs.append((logits.detach(), _grads(model)))
    (ref, ref_g), (got, got_g) = outs
    assert _gap(got, ref) <= 1e-12
    assert ref_g.keys() == got_g.keys() and all(_gap(got_g[n], ref_g[n]) <= 1e-11 for n in ref_g)


def _fit(cfg_overrides: list[str], path: Path, steps: int, tmp: Path) -> list[float]:
    cfg = load_config(path, overrides=cfg_overrides)
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp)
    history = trainer.fit(*datasets(), steps)
    assert len(history.evals) == 1 and math.isfinite(history.evals[0]["loss"])
    return [s.loss for s in history.steps] + [history.evals[0]["loss"]]


@pytest.mark.parametrize("path", [NOMEM, READER_LM])
def test_the_generic_trainer_gives_the_same_losses(path: Path, tmp_path: Path) -> None:
    """Twenty steps in float32 on windows of 16 (the synthetic documents of 16 to 32 bytes give
    rows that start a document and rows that go on): the same losses and evaluation to float32
    rounding."""
    steps = 20
    extra = [*train_overrides(steps), "train.unroll=16", "train.lr=3e-3"]
    extra += list(TINY) if path == NOMEM else ["model.reader_width=16", "model.reader_heads=2"]
    ref = _fit(extra, path, steps, tmp_path / "doc")
    got = _fit([*extra, *CAUSAL], path, steps, tmp_path / "causal")
    assert _same(got, ref, 1e-4), (got, ref)
    assert _finite_mean(ref[-7:-1]) < _finite_mean(ref[:6])


def test_the_tbptt_trainer_gives_the_same_losses_with_the_memory_on(tmp_path: Path) -> None:
    steps = 12
    extra = [*HSN_T[len(A5) :], "train.unroll=16", "train.soft_until_step=0", "train.lr=3e-3"]
    extra += ["model.cand_norm=rms", "model.k_select=3", "train.gate_mode=ste"]
    histories = []
    for mask in ([], list(CAUSAL)):
        cfg = hsn_config(steps, *A5, *extra, *mask)
        seed_everything(0)
        trainer = TBPTTTrainer(
            build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / str(len(mask))
        )
        assert trainer.model.reader is not None
        assert trainer.model.reader.causal == bool(mask)
        histories.append(trainer.fit(*datasets(), steps))
    ref, got = ([s.loss for s in h.steps] + [h.evals[-1]["loss"]] for h in histories)
    assert sum(math.isfinite(x) for x in ref) >= steps // 2
    assert _same(got, ref, 1e-4), (got, ref)


def test_the_generic_trainer_keeps_the_trainer_type(tmp_path: Path) -> None:
    cfg = load_config(NOMEM, [*train_overrides(2), *TINY, *CAUSAL])
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
    assert type(trainer) is Trainer


# --------------------------------------------------------------------------------------------------
# CUDA (skipped without a card)
# --------------------------------------------------------------------------------------------------


def _cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")


@pytest.mark.cuda
def test_flash_takes_the_causal_reader_on_the_card() -> None:
    """Under bf16 autocast with the reader of the 160 x 3 recipe (head width 40): the flash
    kernel alone runs it, forward and backward (SDPA raises when the only backend allowed
    cannot), and the profiler sees flash kernels; the document reader cannot run on flash."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    _cuda_or_skip()
    cfg = load_config(NOMEM_160x3, list(CAUSAL)).model
    assert cfg.reader_width // cfg.reader_heads == 40
    reader = Reader(cfg, seed=0).cuda().train()
    tokens, reset = (t.cuda() for t in _aligned(4, 256, 0))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.autocast("cuda", dtype=torch.bfloat16):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            reader(tokens, reset).float().square().mean().backward()
            torch.cuda.synchronize()
    names = " ".join(e.key for e in prof.key_averages())
    assert "flash_fwd" in names and "flash_bwd" in names, names
    doc = Reader(load_config(NOMEM_160x3).model, seed=0).cuda()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.autocast("cuda", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError):
            doc(tokens, reset)


@pytest.mark.cuda
def test_causal_equals_document_on_the_card_with_no_host_sync() -> None:
    """fp32 and bf16 autocast on the card: the causal reader equals the document reader on
    aligned windows, and its forward and backward synchronise with the host nowhere
    (``torch.cuda.set_sync_debug_mode('error')`` raises on any synchronising call)."""
    _cuda_or_skip()
    cfg = load_config(NOMEM_160x3, list(CAUSAL)).model
    causal = Reader(cfg, seed=0).cuda()
    doc = Reader(load_config(NOMEM_160x3).model, seed=0).cuda()
    tokens, reset = (t.cuda() for t in _aligned(4, 512, 2))
    with torch.no_grad():
        assert _gap(causal(tokens, reset), doc(tokens, reset)) <= 1e-4
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert _gap(causal(tokens, reset).float(), doc(tokens, reset).float()) <= 0.05
    causal(tokens, reset).sum().backward()  # warm: the first call may load kernels
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = causal(tokens, reset)
        out.float().sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


@pytest.mark.cuda
def test_compiled_memory_off_model_runs_flash_as_one_cuda_graph(tmp_path: Path) -> None:
    """The memory-off 160 x 3 model under ``hardware.baseline_compile=reduce-overhead`` and bf16
    autocast: ten training steps of the generic trainer, one graph, no CUDA-graph skip, no
    recompile, flash kernels in the replayed graph, and the losses of the document model to bf16
    rounding."""
    from torch.profiler import ProfilerActivity, profile

    _cuda_or_skip()
    from cards import CUDA_PROFILE

    steps = 10
    extra = [
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=none",
        "hardware.baseline_compile=reduce-overhead",
        "train.unroll=256",
        "train.batch=4",
        "train.micro_batch=4",
        f"train.steps={steps}",
        "train.eval_every=1000",
        "train.ckpt_every=1000",
        "train.stats_every=1000",
        "train.warmup_steps=0",
    ]
    losses = {}
    for mask in ("document", "causal"):
        cfg = load_config(NOMEM_160x3, [*extra, f"model.reader_mask={mask}"])
        seed_everything(0)
        trainer = make_trainer(
            build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / mask
        )
        history = trainer.fit(*datasets(), steps)
        report = trainer.compiled_baseline.report()
        assert report["cudagraph_skips"] == 0 and report["graphs"] == 2, report  # train + eval
        losses[mask] = [s.loss for s in history.steps]
        if mask == "causal":
            compiled = trainer.compiled_baseline
            window = next(trainer._loader(datasets()[0], shuffle=True, epochs=None))
            batch = trainer._to_device(window)
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)  # as the trainer does at every step
            with torch.compiler.set_stance("fail_on_recompile"):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _, _ = compiled(
                            batch["tokens"], batch["context"], None, batch["reset"]
                        )
                    logits.float().sum().backward()
                    torch.cuda.synchronize()
            names = " ".join(e.key for e in prof.key_averages())
            assert "flash_fwd" in names and "flash_bwd" in names, names
    assert _same(losses["causal"], losses["document"], 0.02), losses
