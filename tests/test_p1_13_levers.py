"""P1.13: the recurrent stability levers, each opt-in with a default that is today's behaviour
(``tests/test_p1_13_defaults.py`` holds the default-equality tests).

* ``train.schedule=wsd`` with ``train.decay_fraction``: warm-up, constant, then a 1 - sqrt decay
  to 0 over the last fraction of ``train.steps`` (:mod:`hsn.train.loop`, both trainers).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from hsn.config import RunConfig, load_config, wsd_decay_steps
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer
from hsn.train.loop import SCHEDULES
from p1_13_support import GRU_2M, REF_A1, case_config, datasets

SMALL_HSN = [
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=4",
    "model.in_degree=2",
    "model.n_units=2",
    "model.k_select=2",
    "model.hidden=16",
    "model.scorer_hidden=8",
    "train.batch=4",
    "train.micro_batch=4",
    "train.unroll=32",
]


def _trainer(cfg: RunConfig) -> Any:
    seed_everything(cfg.seed)
    return make_trainer(build_model(cfg), cfg, resolve(cfg.hardware))


def _lrs(trainer: Any, n: int) -> list[list[float]]:
    """The learning rate of every group at scheduler steps 0 .. n - 1 (the lr optimiser step
    ``k`` trains with is entry ``k - 1``)."""
    out = []
    for _ in range(n):
        out.append([float(g["lr"]) for g in trainer.optimizer.param_groups])
        trainer.optimizer.step()  # no gradients: nothing moves; keeps the scheduler's order check
        trainer.scheduler.step()
    return out


# --------------------------------------------------------------------------------------------------
# train.schedule=wsd
# --------------------------------------------------------------------------------------------------


def _wsd(steps: int, warmup: int, fraction: float, *extra: str, base: Any = REF_A1) -> RunConfig:
    small = SMALL_HSN if base == REF_A1 else []
    return load_config(
        base,
        overrides=[
            *small,
            "train.lr=0.003",
            f"train.steps={steps}",
            f"train.warmup_steps={warmup}",
            "train.schedule=wsd",
            f"train.decay_fraction={fraction}",
            *extra,
        ],
    )


def test_p1_13_wsd_schedule_values() -> None:
    """Warm-up end, the stable phase, the decay start, the last step and the end, for the
    3,000-step recipe with a 20 % decay (D = 600, decay start s = 2,400) on both trainers."""
    assert "wsd" in SCHEDULES
    steps, warmup, lr = 3000, 200, 0.003
    for base in (REF_A1, GRU_2M):
        trainer = _trainer(_wsd(steps, warmup, 0.2, base=base))
        lrs = [row[0] for row in _lrs(trainer, steps + 2)]
        D = wsd_decay_steps(0.2, steps)
        s = steps - D
        assert (D, s) == (600, 2400)
        assert lrs[0] == lr / warmup  # optimiser step 1
        assert lrs[warmup - 1] == lr  # the last warm-up step reaches the peak
        assert all(v == lr for v in lrs[warmup - 1 : s + 1])  # stable through the decay start
        assert lrs[s + 1] == lr * (1.0 - math.sqrt(1 / D))  # the first decayed step (k = s + 2)
        assert lrs[steps - 1] == lr * (1.0 - math.sqrt((D - 1) / D))  # the last optimiser step
        assert 0.0 < lrs[steps - 1] < 3e-6
        assert lrs[steps] == 0.0 and lrs[steps + 1] == 0.0  # 0 at train.steps, held after
        decay = lrs[s : steps + 1]
        assert all(a > b for a, b in zip(decay, decay[1:], strict=False))  # strictly decreasing
        # a quarter of the way through the decay the factor is 1 - sqrt(1/4) = 0.5
        assert lrs[s + D // 4] == pytest.approx(0.5 * lr, rel=1e-12)


def test_p1_13_wsd_without_warmup_and_full_decay() -> None:
    """No warm-up: the decay may cover every step (fraction 1): 1 - sqrt(s / steps)."""
    trainer = _trainer(_wsd(16, 0, 1.0))
    lrs = [row[0] for row in _lrs(trainer, 17)]
    assert lrs == [0.003 * (1.0 - math.sqrt(k / 16)) for k in range(17)]


def test_p1_13_wsd_in_a_fit_before_the_decay() -> None:
    """P1.11's protocol trains fewer steps than ``train.steps``: 6 steps of a 20-step WSD run
    with a 25 % decay never reach it, and the logged learning rates are warm-up then peak."""
    cfg = _wsd(20, 2, 0.25, "train.eval_every=6", "train.ckpt_every=20", "profile=cpu")
    train_ds, dev_ds = datasets()
    trainer = _trainer(cfg)
    history = trainer.fit(train_ds, dev_ds, 6)
    assert [r.lr for r in history.steps] == [0.0015, 0.003, 0.003, 0.003, 0.003, 0.003]
    # the run continues into the decay when trained on (a resumed or longer fit)
    history = trainer.fit(train_ds, dev_ds, 20)
    lrs = {r.step: r.lr for r in history.steps}  # steps 7 .. 20; D = 5, decay start s = 15
    assert lrs[16] == 0.003 and lrs[17] == 0.003 * (1 - math.sqrt(1 / 5))  # s + 1, s + 2
    assert lrs[20] == 0.003 * (1 - math.sqrt(4 / 5))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (["train.schedule=wsd"], "needs train.decay_fraction"),
        (["train.schedule=wsd", "train.decay_fraction=0"], r"must lie in \(0, 1\]"),
        (["train.schedule=wsd", "train.decay_fraction=1.5"], r"must lie in \(0, 1\]"),
        (["train.schedule=wsd", "train.decay_fraction=0.0004"], "at least one"),
        (
            ["train.schedule=wsd", "train.decay_fraction=0.95", "train.warmup_steps=200"],
            "inside the warm-up",
        ),
        (["train.schedule=cosine", "train.decay_fraction=0.2"], "read by train.schedule=wsd"),
        (["train.decay_fraction=0.2"], "read by train.schedule=wsd"),  # constant by default
    ],
)
def test_p1_13_wsd_config_refusals(overrides: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):  # load_config wraps the ValidationError
        load_config(REF_A1, overrides=["train.steps=1000", *overrides])


def test_p1_13_wsd_config_accepted() -> None:
    """The edge cases that are valid: the decay starting exactly at the warm-up's end."""
    cfg = load_config(
        REF_A1,
        overrides=[
            "train.steps=1000",
            "train.warmup_steps=200",
            "train.schedule=wsd",
            "train.decay_fraction=0.8",
        ],
    )
    assert cfg.train.decay_fraction == 0.8 and wsd_decay_steps(0.8, 1000) == 800
    assert case_config("hsn_dense").train.decay_fraction is None  # the default
