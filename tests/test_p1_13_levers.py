"""P1.13: the recurrent stability levers, each opt-in with a default that is today's behaviour
(``tests/test_p1_13_defaults.py`` holds the default-equality tests).

* ``train.schedule=wsd`` with ``train.decay_fraction``: warm-up, constant, then a 1 - sqrt decay
  to 0 over the last fraction of ``train.steps`` (:mod:`hsn.train.loop`, both trainers).
* ``train.cell_lr_mult`` / ``train.cell_weight_decay``: the HSN cell's parameters (exactly
  ``HSN.cell``, the P1.11 diagnostics' ``cell`` group) in optimiser groups of their own, for AdamW
  and lazy AdamW; clipping and the P1.12 guard stay global; the baselines refuse the keys.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig, load_config, wsd_decay_steps
from hsn.device import resolve, seed_everything
from hsn.model.hsn import HSN
from hsn.runner import build_model, make_trainer
from hsn.train.lazy_adam import LazyAdamW, cell_parameters
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


# --------------------------------------------------------------------------------------------------
# The cell group: train.cell_lr_mult, train.cell_weight_decay
# --------------------------------------------------------------------------------------------------

CELL_LEVER = ["train.cell_lr_mult=0.25", "train.cell_weight_decay=0.05"]


def _ids(params: Any) -> set[int]:
    return {id(p) for p in params}


def _cell_names(model: HSN) -> set[str]:
    return {f"cell.{name}" for name, p in model.cell.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("case", ["hsn_dense", "hsn_topk"])
def test_p1_13_cell_group_parameter_sets(case: str) -> None:
    """The cell groups hold exactly ``HSN.cell``'s parameters (V, W_out, the gate, the A1 norm,
    the unit bank, the edge logits) and the other groups everything else; group 0 is a non-cell
    group at the global lr; the cell's ndim >= 2 tensors decay at cell_weight_decay, its ndim < 2
    tensors not at all. Lazy AdamW keeps the units in its unit groups (at the cell's rate)."""
    cfg = case_config(case, extra=CELL_LEVER)
    trainer = _trainer(cfg)
    model = trainer.model
    assert isinstance(model, HSN)
    names = _cell_names(model)
    assert {
        "cell.V.weight",
        "cell.W_out.weight",
        "cell.gate.weight",
        "cell.gate.bias",
        "cell.norm.weight",
        "cell.graph.edge_logit",
    } <= names
    assert any(n.startswith("cell.units.") for n in names)
    cell = _ids(model.cell.parameters())
    assert _ids(cell_parameters(model)) == cell
    lr, mult, groups = cfg.train.lr, 0.25, trainer.optimizer.param_groups
    in_cell = [bool(_ids(g["params"]) & cell) for g in groups]
    for g, flag in zip(groups, in_cell, strict=True):
        ids = _ids(g["params"])
        assert ids <= cell if flag else not ids & cell  # no group mixes the two sets
        assert g["initial_lr"] == (lr * mult if flag else lr)
        decays = all(p.ndim >= 2 for p in g["params"])
        assert decays or all(p.ndim < 2 for p in g["params"])
        assert g["weight_decay"] == ((0.05 if flag else 0.01) if decays else 0.0)
    assert not in_cell[0] and in_cell[-1]  # the global groups first
    assert (
        set().union(*(_ids(g["params"]) for g, f in zip(groups, in_cell, strict=True) if f)) == cell
    )
    every = _ids(p for p in model.parameters() if p.requires_grad)
    assert set().union(*(_ids(g["params"]) for g in groups)) == every
    assert sum(len(g["params"]) for g in groups) == len(every)  # each tensor in one group
    if case == "hsn_topk":
        assert isinstance(trainer.optimizer, LazyAdamW)
        units = _ids(model.cell.units.parameters())
        unit_groups = [g for g in groups if "n_units" in g]
        assert set().union(*(_ids(g["params"]) for g in unit_groups)) == units
        assert all(g["initial_lr"] == lr * mult for g in unit_groups)
        assert [("n_units" in g, f) for g, f in zip(groups, in_cell, strict=True)] == [
            (False, False),
            (False, False),
            (False, True),
            (False, True),
            (True, True),
        ]  # shared d / nd, cell d / nd, unit d (the unit bank has no ndim < 2 tensor)
    else:
        assert type(trainer.optimizer) is torch.optim.AdamW and len(groups) == 4


@pytest.mark.parametrize("schedule", ["wsd", "cosine"])
def test_p1_13_cell_group_lr_through_warmup_and_decay(schedule: str) -> None:
    """LambdaLR scales each group's own base rate, so the cell groups stay at mult x the global
    rate through the warm-up, the stable phase and the decay, to the end of the schedule."""
    extra = ["train.decay_fraction=0.25"] if schedule == "wsd" else []
    cfg = load_config(
        REF_A1,
        overrides=[
            *SMALL_HSN,
            *CELL_LEVER,
            "train.lr=0.003",
            "train.steps=20",
            "train.warmup_steps=3",
            f"train.schedule={schedule}",
            *extra,
        ],
    )
    trainer = _trainer(cfg)
    cell = _ids(trainer.model.cell.parameters())
    in_cell = [bool(_ids(g["params"]) & cell) for g in trainer.optimizer.param_groups]
    rows = _lrs(trainer, 21)
    peak = 0.003
    assert rows[2][0] == peak and rows[-1][0] == 0.0  # warm-up end, schedule end
    for row in rows:
        base = row[0]
        for value, flag in zip(row, in_cell, strict=True):
            assert value == pytest.approx(base * 0.25 if flag else base, rel=1e-14, abs=0.0)
    if schedule == "wsd":  # D = 5, decay start 15: the peak through scheduler step 15
        assert all(row[0] == peak for row in rows[2:16]) and rows[16][0] < peak
    else:
        # the warm-up's last step and the cosine's first are both the peak; then it decreases
        assert rows[3][0] == peak
        assert all(a[0] > b[0] for a, b in zip(rows[3:20], rows[4:21], strict=True))


def test_p1_13_cell_group_scales_the_first_step(tmp_path: Path) -> None:
    """One training step from the same initialisation and window: the cell parameters move by
    mult x their default displacement (Adam's first step is lr x (g / (|g| + eps)) + lr x wd x p,
    linear in lr), and every other parameter moves exactly as with the defaults."""
    train_ds, dev_ds = datasets()
    moved: dict[str, dict[str, torch.Tensor]] = {}
    for label, extra in (("default", []), ("lever", ["train.cell_lr_mult=0.25"])):
        cfg = case_config(
            "hsn_dense",
            extra=["train.schedule=constant", "train.warmup_steps=0", "train.steps=1", *extra],
        )
        trainer = _trainer(cfg)
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
        trainer.run_dir = tmp_path / label
        trainer.fit(train_ds, dev_ds, 1)
        moved[label] = {
            n: (p.detach() - before[n]).double() for n, p in trainer.model.named_parameters()
        }
    cell = _cell_names(trainer.model)
    for name, delta in moved["default"].items():
        lever = moved["lever"][name]
        if name in cell:
            if float(delta.norm()) > 0:
                ratio = float(lever.norm() / delta.norm())
                assert ratio == pytest.approx(0.25, rel=1e-3), name
        else:
            assert torch.equal(lever, delta), name


def test_p1_13_cell_group_guard_and_clip_stay_global() -> None:
    """With the cell groups on, the P1.12 guard zeroes every group's gradients on a non-finite
    norm, and the clipping norm the trainer logs is the global one over every parameter."""
    trainer = _trainer(case_config("hsn_dense", extra=CELL_LEVER))
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    for p in params:
        p.grad = torch.ones_like(p)
    want = math.sqrt(sum(p.numel() for p in params))
    assert float(torch.nn.utils.clip_grad_norm_(params, math.inf)) == pytest.approx(want)
    next(iter(trainer.model.cell.parameters())).grad.fill_(math.inf)  # type: ignore[union-attr]
    assert trainer._guard_nonfinite(math.inf, 1) is True
    assert all(p.grad is None for g in trainer.optimizer.param_groups for p in g["params"])
    assert trainer.skipped_nonfinite == 1


@pytest.mark.parametrize(
    ("base", "override"),
    [
        (GRU_2M, "train.cell_lr_mult=0.5"),
        (GRU_2M, "train.cell_weight_decay=0.0"),
        (Path(REF_A1).parent / "baselines" / "transformer_2m.yaml", "train.cell_lr_mult=0.5"),
    ],
)
def test_p1_13_cell_group_refused_for_baselines(base: Path, override: str) -> None:
    """A baseline has no HSN cell: the keys are refused at load, not ignored; and a model without
    a PDGCell refuses them when the optimiser is built."""
    with pytest.raises(ValueError, match="act on the HSN cell"):
        load_config(base, overrides=[override])
    gru = build_model(load_config(GRU_2M, overrides=["baseline.gru.d_model=16"]))
    with pytest.raises(ValueError, match="one PDGCell"):
        cell_parameters(gru)
