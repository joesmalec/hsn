"""The HSN-T reader alone with a next-byte head (``model_name: reader``, ``hsn.baselines.
reader_lm``): the exact-reader diagnostic of the review amendments of 14 Nov 2026 (04a-3). These
tests check that it is the reader of HSN-T and nothing else (the same module, the same starting
weights for the same seed, the same outputs), what it may read, the configuration rules, and
that the generic trainer trains it."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from hsn.baselines.reader_lm import ReaderLM
from hsn.config import config_hash, load_config
from hsn.device import resolve, seed_everything
from hsn.model.hsn_t import HSNT
from hsn.model.reader import Reader
from hsn.runner import build_model, make_trainer
from hsn.train.loop import Trainer
from hsn.train.memory_plan import reader_param_count
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, train_overrides

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"


def test_the_config_is_the_reader_that_is_run() -> None:
    cfg = load_config(READER_LM)
    assert cfg.model_name == "reader" and cfg.model.reader == "transformer"
    model = build_model(cfg)
    assert type(model) is ReaderLM and isinstance(model.reader, Reader)
    report = model.param_report()
    assert report["reader"] == reader_param_count(cfg.model) == 116_480
    assert report["head"] == 64 * 256 + 256
    assert report["stored"] == report["trainable"] == sum(p.numel() for p in model.parameters())
    short = load_config(READER_LM, ["model.reader_span=32"])
    assert config_hash(short) != config_hash(cfg) and build_model(short).reader.span == 32


@pytest.mark.parametrize("span", ["null", "32"])
@pytest.mark.parametrize("seed", [0, 3])
def test_it_starts_as_the_reader_inside_an_hsn_t_of_the_same_seed(span: str, seed: int) -> None:
    """Copied-weight check before any training: the standalone reader and the reader of an
    HSN-T built from the same reader keys and seed hold the same parameters and give the same
    hidden outputs, bit for bit (one module, one generator, float32 on the CPU)."""
    keys = ["model.reader=transformer", f"model.reader_span={span}", f"seed={seed}"]
    alone = build_model(load_config(READER_LM, [f"model.reader_span={span}", f"seed={seed}"]))
    hybrid = build_model(load_config(RECIPE_A6, ["model_name=hsn_t", "model.k_select=12", *keys]))
    assert isinstance(alone, ReaderLM) and isinstance(hybrid, HSNT) and hybrid.reader is not None
    ours, theirs = dict(alone.reader.named_parameters()), dict(hybrid.reader.named_parameters())
    assert ours.keys() == theirs.keys()
    for name, value in ours.items():
        assert torch.equal(value, theirs[name]), name
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 256, (3, 40), generator=g)
    reset = torch.zeros(3, 40, dtype=torch.bool)
    reset[:, 0] = True
    reset[1, 17] = True
    with torch.no_grad():
        assert torch.equal(alone.reader(tokens, reset), hybrid.reader(tokens, reset))
    # and its head is drawn apart: another seed changes the reader and the head
    other = build_model(load_config(READER_LM, [f"model.reader_span={span}", f"seed={seed + 1}"]))
    assert not torch.equal(other.head.weight, alone.head.weight)


def test_logits_read_what_the_reader_reads_and_not_the_context() -> None:
    cfg = load_config(READER_LM, ["model.reader_span=4"])
    seed_everything(0)
    model = build_model(cfg).double().eval()
    g = torch.Generator().manual_seed(2)
    tokens = torch.randint(0, 256, (2, 16), generator=g)
    context = torch.randint(0, cfg.model.n_context, (2, 16), generator=g)
    reset = torch.zeros(2, 16, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 9] = True
    with torch.no_grad():
        logits, state, info = model(tokens, context, None, reset)
        assert state is None and info == {} and logits.shape == (2, 16, 256)
        # the context symbol is not an input
        assert torch.equal(model(tokens, torch.zeros_like(context), None, reset)[0], logits)
        # no later byte
        later = tokens.clone()
        later[:, 8:] = (later[:, 8:] + 5) % 256
        assert torch.equal(model(later, context, None, reset)[0][:, :8], logits[:, :8])
        # nothing before a document's start, and nothing beyond 1 + 2 (4 - 1) = 7 bytes
        before = tokens.clone()
        before[0, :9] = (before[0, :9] + 9) % 256
        assert torch.equal(model(before, context, None, reset)[0][0, 9:], logits[0, 9:])
        far = tokens.clone()
        far[1, :8] = (far[1, :8] + 3) % 256
        assert torch.equal(model(far, context, None, reset)[0][1, 14:], logits[1, 14:])
    assert model.init_state(2, "cpu") is None
    assert model.reset_state(None, torch.zeros(2, dtype=torch.bool)) is None
    with pytest.raises(ValueError, match="carries no state"):
        model(tokens, context, object(), reset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (READER_LM, ["model.reader=none"], "set model.reader=transformer"),
        (GRU_2M, ["model.reader=transformer"], "HSN-T family"),
        (RECIPE_A6, ["model.reader=transformer"], "HSN-T family"),
        (
            RECIPE_A6,
            ["model_name=hsn_t", "model.reader=transformer", "model.selected_readout=false"],
            "selected_readout",
        ),
        (READER_LM, ["model.reader_width=30"], "multiple of"),
        (READER_LM, ["model.key_address=true"], "routed HSN"),
    ],
)
def test_configuration_rules(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_the_generic_trainer_trains_it(tmp_path: Path) -> None:
    steps = 30
    cfg = load_config(
        READER_LM,
        overrides=[
            *train_overrides(steps),
            "model.reader_width=16",
            "model.reader_heads=2",
            "train.lr=3e-3",
        ],
    )
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert type(trainer) is Trainer
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
