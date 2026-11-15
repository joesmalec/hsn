"""``model.address_history`` (14 Nov 2026): what crosses a window boundary for A9's n-gram
addresses. With ``byte`` (the default and every run before this key) one byte is carried, and at
the first column of a window that continues a document the byte two back reads as BOS. With
``ngram`` the last ``address_ngram`` bytes are carried and the features are those of one long
window. These tests pin both behaviours at a boundary (a value written right after a split, a
key that spans the split, a new document), the carry itself, the model against one long window,
the compiled window, the trainer and its checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config
from hsn.data.synthetic import BOS
from hsn.model.hsn import HSN, address_history
from hsn.model.hsn_t import HSNT
from hsn.model.state import CellGraphState
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
TWO = ("model.address_ngram=2", "train.gate_mode=ste")
NGRAM = (*TWO, "model.address_history=ngram")


def _two_windows(cfg: RunConfig, seed: int = 1) -> tuple[Tensor, Tensor, Tensor]:
    """Tokens, contexts and resets of two consecutive windows of one document per row
    (``[B, 2T]``; the only reset is the document's start at column 0)."""
    g = torch.Generator().manual_seed(seed)
    B, T = cfg.train.batch, 2 * cfg.train.unroll
    tokens = torch.randint(16, cfg.model.vocab_size, (B, T), generator=g)
    ctx = torch.randint(0, cfg.model.n_context, (B, T), generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    return tokens, ctx, reset


def _scalars(
    model: HSN,
    tokens: Tensor,
    ctx: Tensor,
    state: CellGraphState,
    reset: Tensor,
    previous: Tensor | None,
) -> Tensor:
    """The scorer's position-independent features of a window, ``[B, T, N, F]``."""
    pro = model._window_prologue(
        tokens,
        ctx,
        state,
        reset,
        model._window_mode(),
        reset_first=bool(reset[:, 0].any()),
        prev_tokens=previous,
    )
    return pro.scalars.detach()


def _split_features(
    cfg: RunConfig, tokens: Tensor, ctx: Tensor, reset: Tensor, model: HSN
) -> tuple[Tensor, Tensor]:
    """The features of the second half as part of one long window, and as a window of its own
    with what ``model.address_history`` carries from the first half."""
    T = tokens.shape[1] // 2
    start = model.init_state(tokens.shape[0])
    whole = _scalars(model, tokens, ctx, start, reset, None)[:, T:]
    with torch.no_grad():
        _, state, _ = model(tokens[:, :T], ctx[:, :T], start, reset[:, :T])
    carried = address_history(cfg.model, tokens[:, :T])
    second = _scalars(model, tokens[:, T:], ctx[:, T:], state, reset[:, T:], carried)
    return whole, second


# --------------------------------------------------------------------------------------------------
# Configuration and the carry
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6, ["model.address_ngram=2"])
    assert base.model.address_history == "byte"
    named = load_config(RECIPE_A6, ["model.address_ngram=2", "model.address_history=byte"])
    assert config_hash(named) == config_hash(base)
    on = load_config(RECIPE_A6, ["model.address_ngram=2", "model.address_history=ngram"])
    assert config_hash(on) != config_hash(base)
    with pytest.raises(ValueError, match="needs model.address_ngram > 1"):
        load_config(RECIPE_A6, ["model.address_history=ngram"])


def test_the_carry_is_the_last_byte_or_the_last_n_bytes_newest_first() -> None:
    byte, ngram = _cfg(*A5, *TWO).model, _cfg(*A5, *NGRAM).model
    tokens = torch.tensor([[20, 21, 22, 23], [30, 31, 32, 33]])
    assert address_history(byte, tokens).tolist() == [23, 33]
    assert address_history(ngram, tokens).tolist() == [[23, 22], [33, 32]]
    # a window shorter than n is completed from what was carried into it, or BOS
    one = tokens[:, :1]
    assert address_history(ngram, one, torch.tensor([[7, 8], [9, 10]])).tolist() == [
        [20, 7],
        [30, 9],
    ]
    assert address_history(ngram, one, torch.tensor([7, 9])).tolist() == [[20, 7], [30, 9]]
    assert address_history(ngram, one).tolist() == [[20, BOS], [30, BOS]]


def test_window_refuses_a_carry_of_the_wrong_width() -> None:
    cfg = _cfg(*A5, *NGRAM)
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    wrong = torch.zeros(tokens.shape[0], 3, dtype=torch.int64)
    with pytest.raises(ValueError, match="prev_tokens must be"):
        model(tokens, ctx, _start(model, tokens.shape[0]), reset, prev_tokens=wrong)


# --------------------------------------------------------------------------------------------------
# The boundary cases, on the scorer's features
# --------------------------------------------------------------------------------------------------


def test_one_byte_carried_changes_one_feature_at_the_first_column() -> None:
    """The behaviour of every run before the key, pinned: in a window that continues a document
    the features are those of a long window except at column 0, where exactly one of them (the
    address of the two bytes before the current one) is computed with BOS for the older byte."""
    cfg = _cfg(*A5, *TWO)
    model = _model(cfg)
    whole, second = _split_features(cfg, *_two_windows(cfg), model)
    differs = (whole - second).abs() > 1e-12  # [B, T, N, F]
    assert bool(differs.any())
    assert not bool(differs[:, 1:].any())  # only the first column
    columns = differs[:, 0].any(dim=0).any(dim=0).nonzero().flatten().tolist()
    assert len(columns) == 1  # one feature: a key that spans the split (a_cur2) is exact


def test_n_bytes_carried_give_the_features_of_one_long_window() -> None:
    """A value written right after a split under a key that ended the window before, and a key
    that spans the split, have the addresses they have inside a window."""
    cfg = _cfg(*A5, *NGRAM)
    model = _model(cfg)
    whole, second = _split_features(cfg, *_two_windows(cfg), model)
    assert float((whole - second).abs().max()) <= 1e-12


@pytest.mark.parametrize("column", [0, 1, 2])
def test_a_new_document_reads_nothing_of_the_one_before(column: int) -> None:
    """A reset in the first columns of the second window: the carried bytes belong to the
    document before and are not read at or after the reset, whatever they are."""
    cfg = _cfg(*A5, *NGRAM)
    model = _model(cfg)
    tokens, ctx, reset = _two_windows(cfg)
    T = tokens.shape[1] // 2
    reset[:, T + column] = True
    whole, second = _split_features(cfg, tokens, ctx, reset, model)
    assert float((whole - second).abs().max()) <= 1e-12
    other = tokens.clone()
    other[:, : T + column] = (other[:, : T + column] + 7) % 200 + 16
    _, changed = _split_features(cfg, other, ctx, reset, model)
    assert float((changed[:, column:] - second[:, column:]).abs().max()) == 0.0


def test_hsn_t_has_the_same_address_features_across_a_split() -> None:
    """The reader starts again at a window boundary; the address features do not."""
    family = (
        "model_name=hsn_t",
        "model.reader=transformer",
        "model.reader_width=8",
        "model.reader_heads=2",
    )
    cfg = _cfg(*A5, *NGRAM, *family)
    torch.manual_seed(0)
    model = HSNT.from_config(cfg).double()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    whole, second = _split_features(cfg, *_two_windows(cfg), model.train())
    assert float((whole - second).abs().max()) <= 1e-12


# --------------------------------------------------------------------------------------------------
# The model, the compiled window, the trainer
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("extra", "exact"), [(NGRAM, True), (TWO, False)])
def test_two_windows_against_one_long_window(extra: tuple[str, ...], exact: bool) -> None:
    """With the last n bytes carried, two windows of a document compute what one window of
    twice the length computes (float64, to rounding). With one byte carried they do not."""
    cfg = _cfg(*A5, *extra)
    model = _model(cfg).eval()
    tokens, ctx, reset = _two_windows(cfg)
    T = tokens.shape[1] // 2
    start = model.init_state(tokens.shape[0])
    with torch.no_grad():
        whole, whole_state, _ = model(tokens, ctx, start, reset)
        first, state, _ = model(tokens[:, :T], ctx[:, :T], start, reset[:, :T])
        carried = address_history(cfg.model, tokens[:, :T])
        second, split_state, _ = model(
            tokens[:, T:], ctx[:, T:], state, reset[:, T:], prev_tokens=carried
        )
    assert float((first - whole[:, :T]).abs().max()) <= 1e-9
    gap = float((second - whole[:, T:]).abs().max())
    state_gap = float((split_state.h - whole_state.h).abs().max())
    if exact:
        assert gap <= 1e-9 and state_gap <= 1e-9
    else:
        assert gap > 1e-6


@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_takes_the_carried_bytes() -> None:
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *NGRAM,
    )
    model = p28._model(cfg, training=True, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    reset = torch.zeros_like(reset)  # the window continues a document
    carried = torch.randint(16, cfg.model.vocab_size, (tokens.shape[0], 2))
    ref = p28._run(lambda: model(tokens, cx, state, reset, prev_tokens=carried))
    got = p28._run(lambda: window(tokens, cx, state, reset, prev_tokens=carried))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    one = p28._run(lambda: model(tokens, cx, state, reset, prev_tokens=carried[:, 0]))
    assert p28._max_diff(one, ref)["values"] > 0.0  # the older byte is read


@pytest.mark.parametrize(("extra", "shape"), [(NGRAM, (4, 2)), (TWO, (4,))])
def test_trainer_carries_and_checkpoints_the_history(
    extra: tuple[str, ...], shape: tuple[int, ...], tmp_path: Path
) -> None:
    steps = 4
    cfg = hsn_config(steps, *A5, *extra, "train.ckpt_every=2")
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps
    assert trainer._prev_tokens is not None and tuple(trainer._prev_tokens.shape) == shape
    checkpoint = sorted((tmp_path / "run").glob("step_*.pt"))[-1]
    again = hsn_trainer(cfg, tmp_path / "again")
    again.resume(checkpoint)
    assert again._prev_tokens is not None and tuple(again._prev_tokens.shape) == shape
    assert torch.equal(again._prev_tokens.cpu(), trainer._prev_tokens.cpu())


def test_hsn_t_requires_the_carried_n_gram_history() -> None:
    """Adopted on 14 Nov 2026: HSN-T carries the address bytes across windows."""
    family = ["model_name=hsn_t", "model.reader=transformer", "model.k_select=12"]
    with pytest.raises(ValueError, match="needs model.address_history=ngram"):
        load_config(RECIPE_A6, [*family, "model.address_ngram=2"])
    on = load_config(RECIPE_A6, [*family, "model.address_ngram=2", "model.address_history=ngram"])
    assert on.model.address_history == "ngram"
    one_byte = load_config(RECIPE_A6, family)  # one-byte addresses carry one byte
    assert one_byte.model.address_ngram == 1 and one_byte.model.address_history == "byte"
    # the HSN keeps the default of every earlier run
    assert load_config(RECIPE_A6, ["model.address_ngram=2"]).model.address_history == "byte"
