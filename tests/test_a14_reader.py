"""Amendment A14 (``docs/amendments/A14_reader.md``): ``model.reader``, the window reader.

With ``model.reader = transformer`` a small causal Transformer reads the bytes of the window and
its output ``c_t`` takes the place of A8's fast read in ``S`` and in the decoder. These tests
check that the keys are opt-in and leave the default model and its hashes alone, what ``c_t`` may
depend on (the document's bytes up to ``t`` inside the window, the last ``reader_span`` of them
with a span), that the rest of the model is the model without the key (initialisation, and the
forward pass once the new input columns are zero), the closed forms, the refusals, the compiled
window, and that the recipe trains with it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config, split_window_scope_error
from hsn.device import seed_everything
from hsn.experiments.variants import install_variant
from hsn.model.hsn import HSN
from hsn.model.reader import Reader
from hsn.train.memory_plan import expected_param_count, reader_param_count
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, _cfg, _model, _start, _window
from test_a8_fast_cells import COMPILED
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"

W = 8  # the reader's width on the six-cell model of the A6 tests (d = 8)
READER = ("model.reader=transformer", f"model.reader_width={W}", "model.reader_heads=2")
SPAN = 4
SHORT = (*READER, f"model.reader_span={SPAN}")
#: the candidate's cut: the cell's read of the state keeps its gradient
CUT_C = "train.feedback_detach=[summary,eps,router]"


def _reader_cfg(*extra: str) -> RunConfig:
    return _cfg(*A5, *READER, *extra)


def _gap(a: Tensor, b: Tensor) -> float:
    """The largest absolute difference of two tensors."""
    return float((a - b).detach().abs().max())


def _bytes(cfg: RunConfig, B: int, T: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.model.vocab_size, (B, T), generator=g)


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_keys_are_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.reader == "none" and base.model.reader_span is None
    named = load_config(
        RECIPE_A6,
        [
            "model.reader=none",
            "model.reader_width=64",
            "model.reader_layers=2",
            "model.reader_heads=4",
            "model.reader_span=null",
        ],
    )
    assert config_hash(named) == config_hash(base)
    on = load_config(RECIPE_A6, ["model.reader=transformer"])
    short = load_config(RECIPE_A6, ["model.reader=transformer", "model.reader_span=32"])
    assert len({config_hash(base), config_hash(on), config_hash(short)}) == 3
    assert HSN.from_config(base).reader is None
    assert "reader" not in HSN.from_config(base).param_report()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["model.reader=transformer", "model.selected_readout=false"], "selected_readout"),
        (["model.reader_span=8"], "span of model.reader"),
        (["model.reader=transformer", "model.reader_width=30"], "multiple of"),
        (["model.reader=transformer", "model.reader_width=12"], "even width per head"),
        (["model.reader=transformer", "model.reader_span=0"], "reader_span"),
    ],
)
def test_configuration_refusals(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(RECIPE_A6, overrides)


def test_split_and_fused_windows_refuse_the_reader() -> None:
    recipe = ["train.gate_mode=ste", "model.reader=transformer"]
    reason = split_window_scope_error(load_config(RECIPE_A6, recipe))
    assert reason is not None and "model.reader" in reason
    for window in ("split", "fused"):
        with pytest.raises(ValueError, match="reader"):
            load_config(RECIPE_A6, [*recipe, f"hardware.window={window}"])


def test_baselines_and_variants_refuse_the_key() -> None:
    gru = REPO / "configs" / "baselines" / "gru_2m.yaml"
    # the reader needs A5's readout, which a baseline cannot have (routed HSN only)
    with pytest.raises(ValueError, match="selected_readout"):
        load_config(gru, ["model.reader=transformer"])
    with pytest.raises(ValueError, match="selected_readout"):
        load_config(gru, ["model.reader=transformer", "model.selected_readout=true"])
    model = HSN.from_config(_reader_cfg())
    with pytest.raises(ValueError, match="reader"):
        install_variant(model, "gru_cell", {})


# --------------------------------------------------------------------------------------------------
# The reader alone: what c_t may depend on
# --------------------------------------------------------------------------------------------------


def _reader(*extra: str) -> tuple[Reader, RunConfig]:
    cfg = _reader_cfg(*extra)
    reader = Reader(cfg.model, seed=0).double().eval()
    return reader, cfg


def test_reader_output_reads_no_later_byte() -> None:
    reader, cfg = _reader()
    B, T, t0 = 3, 12, 7
    tokens = _bytes(cfg, B, T, 1)
    reset = torch.zeros(B, T, dtype=torch.bool)
    out = reader(tokens, reset)
    assert out.shape == (B, T, W)
    later = tokens.clone()
    later[:, t0:] = (later[:, t0:] + 5) % cfg.model.vocab_size
    changed = reader(later, reset)
    assert torch.equal(changed[:, :t0], out[:, :t0])
    assert float((changed[:, t0:] - out[:, t0:]).detach().abs().amax(dim=-1).min()) > 0.0


def test_reader_output_stays_inside_its_document() -> None:
    """A reset column starts a document: nothing before it is read at or after it, and the row
    without a reset reads its whole window."""
    reader, cfg = _reader()
    B, T, k = 3, 12, 5
    tokens = _bytes(cfg, B, T, 2)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:2, k] = True  # rows 0 and 1 start a document at column k; row 2 does not
    out = reader(tokens, reset)
    before = tokens.clone()
    before[:, :k] = (before[:, :k] + 9) % cfg.model.vocab_size
    changed = reader(before, reset)
    assert torch.equal(changed[:2, k:], out[:2, k:])
    assert _gap(changed[2, k:], out[2, k:]) > 0.0


def test_span_limits_the_reader_to_its_last_bytes() -> None:
    reader, cfg = _reader(f"model.reader_span={SPAN}")
    wide, _ = _reader()
    B, T = 2, 12
    tokens = _bytes(cfg, B, T, 3)
    reset = torch.zeros(B, T, dtype=torch.bool)
    out, out_wide = reader(tokens, reset), wide(tokens, reset)
    t = 9
    # two blocks: a byte reaches c_t over two hops of SPAN - 1, so the last 2 SPAN - 1 bytes
    reach = 2 * (SPAN - 1)
    older = tokens.clone()
    older[:, : t - reach] = (older[:, : t - reach] + 3) % cfg.model.vocab_size
    assert torch.equal(reader(older, reset)[:, t], out[:, t])
    assert _gap(wide(older, reset)[:, t], out_wide[:, t]) > 0.0
    # one hop: the attention of position t itself never reads further than SPAN - 1 back
    mask = reader.attention_mask(reset)[0, 0]
    assert mask[t].nonzero().flatten().tolist() == list(range(t - SPAN + 1, t + 1))
    assert bool(reader.attention_mask(reset)[..., torch.arange(T), torch.arange(T)].all())


def test_reader_reads_distances_not_window_positions() -> None:
    """Rotary positions: the same bytes give the same output wherever they sit in the window,
    once the span no longer reaches the bytes before them."""
    reader, cfg = _reader(f"model.reader_span={SPAN}")
    L, reach = 8, 2 * (SPAN - 1)
    body = _bytes(cfg, 1, L, 4)
    early = torch.cat([_bytes(cfg, 1, 2, 5), body], dim=1)
    late = torch.cat([_bytes(cfg, 1, 9, 6), body], dim=1)
    no = lambda x: torch.zeros_like(x, dtype=torch.bool)  # noqa: E731
    a = reader(early, no(early))[:, 2 + reach :]
    b = reader(late, no(late))[:, 9 + reach :]
    assert a.shape == b.shape and a.shape[1] == L - reach
    assert _gap(a, b) <= 1e-12


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [READER, SHORT])
def test_param_report_matches_the_closed_form(extra: tuple[str, ...]) -> None:
    for cfg in (_cfg(*A5, *extra), load_config(RECIPE_A6, ["model.reader=transformer"])):
        model = HSN.from_config(cfg)
        report, expected = model.param_report(), expected_param_count(cfg.model)
        for row, count in expected.items():
            assert report[row] == count, row
        assert report["reader"] == reader_param_count(cfg.model)
        assert report["total_trainable"] == report["total"] + report["edges"] + report["reader"]
    m = cfg.model
    assert reader_param_count(m) == 116_480
    plain = expected_param_count(load_config(RECIPE_A6).model)["total"]
    assert expected["total"] - plain == 2 * m.reader_width * m.hidden
    assert reader_param_count(load_config(RECIPE_A6).model) == 0


def test_reader_keeps_the_other_parameters_initialisation() -> None:
    """The reader and the new input columns are drawn from their own generators: every
    parameter the two models share starts equal, the widened layers' old columns too, and the
    global random stream is where it was."""
    torch.manual_seed(0)
    plain = HSN.from_config(_cfg(*A5))
    after_plain = torch.rand(4)
    torch.manual_seed(0)
    read = HSN.from_config(_reader_cfg())
    after_read = torch.rand(4)
    assert torch.equal(after_plain, after_read)
    theirs = dict(read.named_parameters())
    for name, p in plain.named_parameters():
        q = theirs.pop(name)
        if name in ("shared.fc1.weight", "decoder.fc1.weight"):
            assert q.shape[1] == p.shape[1] + W
            assert torch.equal(q[:, : p.shape[1]], p), name
            assert float(q[:, p.shape[1] :].detach().abs().max()) > 0.0
        else:
            assert torch.equal(p, q), name
    assert theirs and all(name.startswith("reader.") for name in theirs)


@pytest.mark.parametrize("cut", [(), (CUT_C,)])
def test_model_with_zeroed_reader_columns_is_the_model_without_it(cut: tuple[str, ...]) -> None:
    """The reader enters through the new input columns of ``S`` and ``D`` and nowhere else: with
    those columns zero the window is the window of the model without the key."""
    seed_everything(0)
    plain = HSN.from_config(_cfg(*A5, *cut)).double().train()
    seed_everything(0)
    read = HSN.from_config(_reader_cfg(*cut)).double().train()
    with torch.no_grad():
        for layer in (read.shared.fc1, read.decoder.fc1):
            layer.weight[:, -W:] = 0.0
    cfg = _reader_cfg(*cut)
    tokens, ctx, reset = _window(cfg)
    reset[1, 6] = True
    state = _start(plain, tokens.shape[0])
    ref_logits, ref_state, _ = plain(tokens, ctx, state, reset)
    logits, new_state, _ = read(tokens, ctx, state, reset)
    assert _gap(logits, ref_logits) <= 1e-12
    assert _gap(new_state.h, ref_state.h) <= 1e-12
    assert _gap(new_state.q, ref_state.q) <= 1e-12


@pytest.mark.parametrize("extra", [READER, SHORT])
def test_logits_read_no_later_byte_and_no_earlier_document(extra: tuple[str, ...]) -> None:
    cfg = _cfg(*A5, *extra)
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    k, t0 = 4, 8
    reset[0, k] = True
    state = _start(model, B)
    logits, _, _ = model(tokens, ctx, state, reset)
    later = tokens.clone()
    later[:, t0:] = (later[:, t0:] + 5) % cfg.model.vocab_size
    changed, _, _ = model(later, ctx, state, reset)
    assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
    assert _gap(changed[:, t0:], logits[:, t0:]) > 1e-6
    before = tokens.clone()
    before[0, :k] = (before[0, :k] + 9) % cfg.model.vocab_size
    changed, _, _ = model(before, ctx, state, reset)
    assert _gap(changed[0, k:], logits[0, k:]) <= 1e-12


@pytest.mark.parametrize("cut", [(), (CUT_C,)])
def test_gradients_reach_the_reader_and_the_new_columns(cut: tuple[str, ...]) -> None:
    cfg = _reader_cfg(*cut)
    model = _model(cfg)
    assert model.reader is not None
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    logits.square().mean().backward()
    for name, p in model.reader.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name
    for layer in (model.shared.fc1, model.decoder.fc1):
        grad = layer.weight.grad
        assert grad is not None and float(grad[:, -W:].abs().sum()) > 0.0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


def test_shared_transform_reads_the_reader_only_through_its_columns() -> None:
    """``S`` reads ``[z ; q ; ctx ; c]``: without the decoder's columns the reader still moves the
    logits (through ``u``), and without both it does not."""
    cfg = _reader_cfg()
    model = _model(cfg).eval()
    assert model.reader is not None
    tokens, ctx, reset = _window(cfg)
    state = _start(model, tokens.shape[0])

    def run() -> Tensor:
        with torch.no_grad():
            return model(tokens, ctx, state, reset)[0]

    base = run()
    saved = {n: p.detach().clone() for n, p in model.reader.named_parameters()}
    with torch.no_grad():
        model.decoder.fc1.weight[:, -W:] = 0.0
    through_u = run()
    with torch.no_grad():
        for p in model.reader.parameters():
            p.add_(0.5 * torch.randn_like(p))
    moved = run()
    assert _gap(moved, through_u) > 1e-6
    with torch.no_grad():
        model.shared.fc1.weight[:, -W:] = 0.0
    cut_off = run()
    with torch.no_grad():
        for n, p in model.reader.named_parameters():
            p.copy_(saved[n])
    assert _gap(run(), cut_off) <= 1e-12
    assert _gap(base, through_u) > 1e-6


def test_step_refuses_the_reader() -> None:
    cfg = _reader_cfg()
    model = _model(cfg)
    tokens, ctx, _ = _window(cfg)
    with pytest.raises(ValueError, match="reads the bytes of a window"):
        model.step(tokens[:, 0], ctx[:, 0], _start(model, tokens.shape[0]))


def test_fast_cells_may_stay_beside_the_reader() -> None:
    """``S([z ; q ; ctx ; c ; f])`` and ``D([q ; u ; r ; f ; c])``."""
    fast = ("model.k_select=4", "model.fast_cells=2", "model.fast_readout=true")
    cfg = _cfg(*A5, *fast, *READER, "train.gate_mode=ste")
    model = _model(cfg)
    d = cfg.model.d
    report, expected = model.param_report(), expected_param_count(cfg.model)
    assert all(report[row] == count for row, count in expected.items())
    assert model.shared.fc1.in_features == 2 * d + cfg.model.context_dim + W + d
    assert model.decoder.fc1.in_features == 4 * d + W
    tokens, ctx, reset = _window(cfg)
    state = _start(model, tokens.shape[0])
    logits, _, _ = model(tokens, ctx, state, reset)
    later = tokens.clone()
    later[:, 8:] = (later[:, 8:] + 5) % cfg.model.vocab_size
    changed, _, _ = model(later, ctx, state, reset)
    assert _gap(changed[:, :8], logits[:, :8]) <= 1e-12
    logits.square().mean().backward()
    assert model.reader is not None
    assert all(float(p.grad.abs().sum()) > 0.0 for p in model.reader.parameters())


def test_window_under_bf16_autocast_is_finite() -> None:
    cfg = _reader_cfg("train.gate_mode=ste")
    model = _model(cfg, torch.float32)
    tokens, ctx, reset = _window(cfg)
    state = _start(model, tokens.shape[0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, new_state, _ = model(tokens, ctx, state, reset)
    assert bool(torch.isfinite(logits).all()) and new_state.h.dtype == torch.float32
    logits.float().square().mean().backward()
    assert model.reader is not None
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.reader.parameters())


# --------------------------------------------------------------------------------------------------
# The compiled autograd window and training
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [READER, SHORT])
@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str, extra: tuple[str, ...]) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *extra,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


@pytest.mark.parametrize("extra", [READER, SHORT])
def test_recipe_with_the_reader_trains(extra: tuple[str, ...], tmp_path: Path) -> None:
    """Thirty TBPTT steps of the small A6 recipe with the candidate's cut on the tiny CPU
    configuration of the trainer tests: finite losses that fall."""
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        "train.gate_mode=ste",
        "train.lr=3e-3",
        *extra,
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
