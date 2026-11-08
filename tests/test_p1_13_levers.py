"""P1.13: the recurrent stability levers, each opt-in with a default that is today's behaviour
(``tests/test_p1_13_defaults.py`` holds the default-equality tests).

* ``train.schedule=wsd`` with ``train.decay_fraction``: warm-up, constant, then a 1 - sqrt decay
  to 0 over the last fraction of ``train.steps`` (:mod:`hsn.train.loop`, both trainers).
* ``train.cell_lr_mult`` / ``train.cell_weight_decay``: the HSN cell's parameters (exactly
  ``HSN.cell``, the P1.11 diagnostics' ``cell`` group) in optimiser groups of their own, for AdamW
  and lazy AdamW; clipping and the P1.12 guard stay global; the baselines refuse the keys.
* ``model.cand_norm=rms`` (amendment A4): RMSNorm with a learned gain on ``W_out(.)`` before the
  candidate's tanh in every path that computes the candidate (dense, masked, sparse, static sparse,
  the eager window, P2.8's compiled window), its accounting, checkpoints and refusals.
"""

from __future__ import annotations

import gc
import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import test_p1_5_hsn as p15
import test_p2_5_sparse_path as p25
import test_p2_8_compile as p28
import torch
from cards import CUDA_PROFILE
from hsn.config import RunConfig, load_config, load_hardware_profile, wsd_decay_steps
from hsn.device import resolve, seed_everything
from hsn.experiments.variants import install_variant
from hsn.kernels.costs import profile_dense_costs
from hsn.model.hsn import HSN, PARAM_REPORT_ROWS
from hsn.runner import build_model, make_trainer
from hsn.train.checkpoint import load_checkpoint, save_checkpoint
from hsn.train.lazy_adam import LazyAdamW, cell_parameters
from hsn.train.loop import SCHEDULES
from hsn.train.memory_plan import expected_param_count
from p1_13_support import GRU_2M, REF_A1, case_config, datasets
from torch import Tensor, nn

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


# --------------------------------------------------------------------------------------------------
# model.cand_norm (amendment A4)
# --------------------------------------------------------------------------------------------------

RMS = ["model.cand_norm=rms"]
REPO = Path(REF_A1).resolve().parents[1]


class _Capture:
    """Forward hook on ``cell.cand_norm``: every (input, output) pair it sees."""

    def __init__(self, module: nn.Module) -> None:
        self.calls: list[tuple[Tensor, Tensor]] = []
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args: tuple[Tensor, ...], out: Tensor) -> None:
        self.calls.append((args[0].detach(), out.detach()))

    def close(self) -> None:
        self.handle.remove()


def _assert_unit_rms(x: Tensor, out: Tensor, gain: Tensor) -> None:
    """``out = gain ⊙ x / sqrt(mean(x²) + eps)`` (eps 1e-6) over the width: the RMS of
    ``out / gain`` is 1, short of it only by the eps term, ``sqrt(m / (m + eps))``."""
    x, out, g = x.double(), out.double(), gain.double()
    m = x.pow(2).mean(dim=-1, keepdim=True)
    torch.testing.assert_close(out, g * x / (m + 1e-6).sqrt(), rtol=1e-5, atol=1e-6)
    rms = (out / g).pow(2).mean(dim=-1, keepdim=True).sqrt()
    torch.testing.assert_close(rms, (m / (m + 1e-6)).sqrt(), rtol=0, atol=1e-6)
    assert float(m.min()) > 1e-3  # so the RMS is 1 to better than 5e-4


def _random_gain(model: HSN, seed: int = 41) -> Tensor:
    """Set the candidate gain to seeded values in [0.5, 2) (not 1, so a skipped gain shows)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    norm = model.cell.cand_norm
    assert isinstance(norm, nn.RMSNorm) and norm.weight is not None
    with torch.no_grad():
        values = 0.5 + 1.5 * torch.rand(norm.weight.shape, generator=generator)
        norm.weight.copy_(values.to(norm.weight))
    return norm.weight.detach()


def test_p1_13_cand_norm_accounting() -> None:
    """The Appendix C.1 row ``cand_norm`` (amendment A4): d gain parameters with rms, 0 by
    default; the closed form, the measured report, the state-dict key and the script agree."""
    assert PARAM_REPORT_ROWS.index("cand_norm") == PARAM_REPORT_ROWS.index("cell_norm") + 1
    cfg, model = p15._model(REF_A1, RMS)
    report, expected = model.param_report(), expected_param_count(cfg.model)
    assert list(expected) == [*PARAM_REPORT_ROWS, "total"]
    assert all(report[k] == v for k, v in expected.items())
    assert report["cand_norm"] == 64 == model.cell.cand_norm.weight.numel()
    assert report["total"] == 1_820_617 + 64 == 1_820_681
    assert report["total_trainable"] == sum(p.numel() for p in model.parameters()) == 1_820_809
    state = model.state_dict()
    assert tuple(state["cell.cand_norm.weight"].shape) == (64,)
    assert bool((state["cell.cand_norm.weight"] == 1.0).all())  # ones at init
    cfg_t, tiny = p15._model(p15.TINY, RMS)
    assert tiny.param_report()["total"] == 1_099 + 4 == expected_param_count(cfg_t.model)["total"]
    cfg_d, default = p15._model(REF_A1)
    assert (
        default.param_report()["cand_norm"] == 0 == expected_param_count(cfg_d.model)["cand_norm"]
    )
    assert isinstance(default.cell.cand_norm, nn.Identity)
    # every other weight is drawn exactly as without the norm (the gain's init draws nothing)
    plain = default.state_dict()
    assert set(state) - set(plain) == {"cell.cand_norm.weight"}
    assert all(torch.equal(plain[k], state[k]) for k in plain if isinstance(plain[k], Tensor))
    run = subprocess.run(
        [sys.executable, "scripts/count_params.py", "--config", "configs/ref_a1.yaml", *RMS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "| Candidate RMSNorm gain (amendment A4) | 64 | 64 |" in run.stdout
    assert "| **Total** |  | **1,820,681** |" in run.stdout


@pytest.mark.parametrize("mode", ["dense", "masked", "sparse"])
def test_p1_13_cand_norm_unit_rms_per_round(mode: str) -> None:
    """In every round the candidate's pre-activation is RMS-normalised to the gain: the norm
    reads ``W_out(mixed)`` (dense, masked) or ``W_out(F)`` (sparse) and the candidate is its tanh
    (times α in the sparse round), with weights perturbed so the raw pre-activation is far from
    unit RMS."""
    routed = ["model.routing=topk", "train.gate_mode=ste"] if mode != "dense" else []
    cfg, model = p25._model(REF_A1, [*RMS, *routed], perturb=43, scale=0.1)
    gain = _random_gain(model)
    cell = model.cell
    with torch.no_grad():
        cell.W_out.weight.mul_(10.0)  # a saturating pre-activation without the norm
    B = 4
    state = p25._random_state(model, B, seed=44)
    tokens, ctx = p25._ids(cfg, B, 1, seed=45)
    x, c = tokens[:, 0], ctx[:, 0]
    capture = _Capture(cell.cand_norm)
    try:
        with torch.no_grad():
            out = model.step(x, c, state, mode)
            assert len(capture.calls) == 1
            pre, normed = capture.calls[0]
            K = cfg.model.k_select if mode == "sparse" else cfg.model.n_cells
            assert tuple(normed.shape) == (B, K, cfg.model.d)
            raw_rms = float(pre.pow(2).mean().sqrt())
            assert raw_rms > 3.0, raw_rms  # the norm has something to do
            _assert_unit_rms(pre, normed, gain)
            if mode == "sparse":  # the round itself: c = tanh(α ⊙ RMSNorm(W_out F))
                sel = out.selection
                assert sel is not None
                h_new, info = cell.step_sparse(out.u, out.eps, state, sel)
                unit_out, c_t = info["unit_out"], info["c"]
                assert isinstance(unit_out, Tensor) and isinstance(c_t, Tensor)
                want = torch.tanh(sel.alpha.unsqueeze(-1) * cell.cand_norm(cell.W_out(unit_out)))
                torch.testing.assert_close(c_t, want, atol=1e-6, rtol=0)
            else:  # c = tanh(RMSNorm(W_out mixed))
                alpha = torch.softmax(torch.randn(B, cfg.model.n_cells, cfg.model.n_units), -1)
                _, info = cell.step_dense(out.u, out.eps, state, alpha)
                pre_d = cell.W_out(info["mixed"])
                want = torch.tanh(cell.cand_norm(pre_d))
                torch.testing.assert_close(info["c"], want, atol=1e-6, rtol=0)
                _assert_unit_rms(pre_d, cell.cand_norm(pre_d), gain)
    finally:
        capture.close()


def test_p1_13_cand_norm_static_sparse_round() -> None:
    """P2.8's capturable sparse round (``static=True``) normalises the same way."""
    cfg, model = p25._model(
        REF_A1, [*RMS, "model.routing=topk", "train.gate_mode=ste"], perturb=46, scale=0.1
    )
    gain = _random_gain(model)
    B = 4
    state = p25._random_state(model, B, seed=47)
    tokens, ctx = p25._ids(cfg, B, 1, seed=48)
    with torch.no_grad():
        out = model.step(tokens[:, 0], ctx[:, 0], state, "sparse")
        assert out.selection is not None
        _, eager = model.cell.step_sparse(out.u, out.eps, state, out.selection)
        capture = _Capture(model.cell.cand_norm)
        try:
            _, static = model.cell.step_sparse(out.u, out.eps, state, out.selection, static=True)
        finally:
            capture.close()
    _assert_unit_rms(*capture.calls[0], gain)
    torch.testing.assert_close(static["c"], eager["c"], atol=1e-6, rtol=0)


def test_p1_13_cand_norm_window_path() -> None:
    """The P1.10a window (``HSN.forward``, eager) normalises every position's candidate, and the
    window still matches the per-position reference ``HSN.step``."""
    cfg, model = p15._model(REF_A1, [*RMS, *SMALL_HSN[:8]])
    p15._perturb(model, seed=49, scale=0.1)
    gain = _random_gain(model)
    B, T = 3, 6
    tokens, ctx = p15._ids(cfg, B, T, seed=50)
    state = p15._random_state(model, B, seed=51)
    reset = torch.zeros(B, T, dtype=torch.bool)
    capture = _Capture(model.cell.cand_norm)
    try:
        with torch.no_grad():
            logits, final, _ = model(tokens, ctx, state, reset)
    finally:
        capture.close()
    assert len(capture.calls) == T
    for pre, normed in capture.calls:
        _assert_unit_rms(pre, normed, gain)
    with torch.no_grad():
        ref = state
        for t in range(T):
            out = model.step(tokens[:, t], ctx[:, t], ref)
            ref = out.state
    torch.testing.assert_close(final.h, ref.h, atol=1e-5, rtol=0)
    torch.testing.assert_close(logits[:, -1], out.logits, atol=1e-5, rtol=1e-5)
    assert float(final.h.abs().max()) <= 1.0  # the tanh and the convex gate keep h in [-1, 1]


def test_p1_13_cand_norm_matches_pure_python_reference() -> None:
    """The P1.5 pure-Python reference, extended with the candidate norm, matches ``HSN.step``
    in fp64 (tiny configuration, perturbed weights, so the gain is not 1)."""
    for overrides in ([*RMS], [*RMS, "model.edge_norm=sigmoid"]):
        cfg, model = p15._model(p15.TINY, overrides)
        assert model.dtype is torch.float64
        p15._perturb(model, seed=52)
        assert not bool((model.cell.cand_norm.weight == 1.0).all())
        B = 2
        state = p15._random_state(model, B, seed=53)
        tokens, ctx = p15._ids(cfg, B, 2, seed=54)
        out = p15._check_position(cfg, model, state, tokens[:, 0], ctx[:, 0])
        p15._check_position(cfg, model, out.state, tokens[:, 1], ctx[:, 1])


def test_p1_13_cand_norm_sparse_equals_dense_and_masked() -> None:
    """K = N with one-hot α: the sparse round equals the dense one (P2.5's property, with the
    norm); and with K < N the sparse round takes the masked reference's decision and values."""
    cfg, model = p25._model(
        REF_A1, [*RMS, "model.routing=topk", "model.k_select=32"], perturb=29, scale=0.1
    )
    _random_gain(model)
    cell = model.cell
    B, N, M, d = 4, cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    state = p25._random_state(model, B, seed=31)
    tokens, ctx = p25._ids(cfg, B, 1, seed=30)
    x, c = tokens[:, 0], ctx[:, 0]
    with torch.no_grad():
        u, eps, logits = p25._prefix(model, x, c, state)
        generator = torch.Generator(device="cpu").manual_seed(5)
        perm = torch.argsort(torch.rand(B, N, generator=generator), dim=1)
        units = logits.argmax(dim=-1).gather(1, perm)
        valid = torch.ones(B, N, dtype=torch.bool)
        sel = p25._selection(perm, units, valid, logits)
        alpha = p25._one_hot_alpha(perm, units, N, M, state.dtype)
        h_dense, info_d = cell.step_dense(u, eps, state, alpha)
        h_sparse, info_s = cell.step_sparse(u, eps, state, sel)
        index = perm.unsqueeze(-1).expand(B, N, d)
        torch.testing.assert_close(h_sparse, h_dense.gather(1, index), atol=p25.ATOL, rtol=0)
        c_s = info_s["c"]
        assert isinstance(c_s, Tensor)
        torch.testing.assert_close(c_s, info_d["c"].gather(1, index), atol=p25.ATOL, rtol=0)
    cfg_k, routed = p25._model(
        REF_A1, [*RMS, "model.routing=topk", "train.gate_mode=ste"], perturb=11, scale=0.1
    )
    _random_gain(routed)
    tokens, ctx = p25._ids(cfg_k, 4, 1, seed=12)
    start = routed.init_state(4)
    masked = routed.step(tokens[:, 0], ctx[:, 0], start, "masked")
    sparse = routed.step(tokens[:, 0], ctx[:, 0], start, "sparse")
    assert masked.selection is not None and sparse.selection is not None
    assert torch.equal(masked.selection.cell_idx, sparse.selection.cell_idx)
    torch.testing.assert_close(sparse.state.h, masked.state.h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sparse.logits, masked.logits, atol=1e-5, rtol=1e-5)


def test_p1_13_cand_norm_keeps_the_selection_gradient() -> None:
    """The sparse round applies α after the norm, so P2.2's straight-through gradient still
    reaches the scorer through ``Selection.alpha``: with α inside a scale-invariant norm its
    gradient would vanish (shown on the same tensors)."""
    cfg, model = p25._model(
        REF_A1, [*RMS, "model.routing=topk", "train.gate_mode=ste"], perturb=11, scale=0.1
    )
    _random_gain(model)
    tokens, ctx = p25._ids(cfg, 4, 1, seed=12)
    out = model.step(tokens[:, 0], ctx[:, 0], model.init_state(4), "sparse")
    assert out.selection is not None and out.selection.alpha.requires_grad
    out.logits.square().mean().backward()
    grad = model.router.scorer.net.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 1e-8
    # the counterfactual: α inside the norm leaves (almost) nothing for α
    cell = model.cell
    unit_out = torch.randn(4, cfg.model.k_select, cfg.model.d, dtype=torch.float64)
    alpha = torch.ones(4, cfg.model.k_select, dtype=torch.float64, requires_grad=True)
    norm = nn.RMSNorm(cfg.model.d, eps=1e-6).double()
    W = cell.W_out.weight.detach().double()
    inside = torch.tanh(norm((alpha.unsqueeze(-1) * unit_out) @ W.T)).sum()
    (g_inside,) = torch.autograd.grad(inside, alpha)
    after = torch.tanh(alpha.unsqueeze(-1) * norm(unit_out @ W.T)).sum()
    (g_after,) = torch.autograd.grad(after, alpha)
    assert float(g_inside.abs().max()) < 1e-5 < float(g_after.abs().max())


def test_p1_13_cand_norm_old_checkpoints(tmp_path: Path) -> None:
    """A checkpoint of today's model (no ``cell.cand_norm`` key; the layout test pins it to the
    base commit's) loads strictly into a default model and is refused by an rms model: a missing
    gain is not defaulted, because that model was never trained with the norm."""
    _, plain = p15._model(REF_A1)
    path = tmp_path / "step_00000001.pt"
    save_checkpoint(path, plain, None, None, 1, {}, None)
    _, again = p15._model(REF_A1)
    load_checkpoint(path, again)
    assert all(
        torch.equal(a, b) for a, b in zip(plain.parameters(), again.parameters(), strict=True)
    )
    _, rms = p15._model(REF_A1, RMS)
    with pytest.raises(RuntimeError, match=r"Missing key\(s\).*cell\.cand_norm\.weight"):
        load_checkpoint(path, rms)


def test_p1_13_cand_norm_refusals() -> None:
    """The baselines refuse ``model.cand_norm=rms`` (they have no candidate), and so do the P1.11
    variants other than ``pdg`` (the coordinator-agreed hook); ``pdg`` takes it."""
    for base in (GRU_2M, Path(REF_A1).parent / "baselines" / "transformer_2m.yaml"):
        with pytest.raises(ValueError, match="act on the HSN cell"):
            load_config(base, overrides=RMS)
    _, model = p15._model(REF_A1, RMS)
    assert install_variant(model, "pdg", 0) == {"variant": "pdg"}
    for variant in ("cell_offset", "gru_cell", "linear_head"):
        with pytest.raises(ValueError, match="cand_norm=rms is refused"):
            install_variant(model, variant, 0)


def test_p1_13_cand_norm_profile_dense_costs() -> None:
    """The segmented replica of the dense step (P2.1 cost profiles) includes the norm: its drift
    check against ``HSN.step`` passes with rms and a non-unit gain."""
    cfg, model = p15._model(REF_A1, [*RMS, *SMALL_HSN[:8]])
    _random_gain(model)
    ctx = resolve(load_hardware_profile("cpu"))
    report = profile_dense_costs(model, ctx, batch=2, positions=3)
    assert report["cost_profile"]["d"] == cfg.model.d


# --------------------------------------------------------------------------------------------------
# The compiled window (P2.8) with the levers: CPU inductor, and the cuda-marked training steps
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def _compiled() -> Iterator[None]:
    """Blocks of at most 4 positions (P2.8's small blocks: short compiles); afterwards dynamo's
    caches and the CUDA Graph trees are released, as P2.8's own tests do."""
    from hsn.train import compile as compile_mod

    old = compile_mod.MAX_BLOCK
    compile_mod.MAX_BLOCK = 4
    try:
        yield
    finally:
        compile_mod.MAX_BLOCK = old
        torch._dynamo.reset()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees

            reset_cudagraph_trees()
            gc.collect()
            torch.cuda.empty_cache()


def _compiled_vs_eager(ctx: Any, backend: str, device: str) -> None:
    """P2.8's compiled window with the candidate norm equals the eager window (P2.8's measured
    fp32 tolerance), for a training and an evaluation window, and differs from the unit-gain
    model, so the compiled window applies the gain."""
    for training in (True, False):
        _compiled_vs_eager_once(ctx, backend, p28.TOL[(device, "fp32")], training)


def _compiled_vs_eager_once(ctx: Any, backend: str, tol: dict[str, float], training: bool) -> None:
    model = p28._model(p28._cfg(*RMS), training=training).to(ctx.device)
    _random_gain(model)
    window = p28._window(model, ctx, backend=backend)
    tokens, cx, state, reset = p28._inputs(model)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    for _ in range(3):  # warm-up, recording, replay
        got = p28._run(lambda: window(tokens, cx, state, reset))
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        grads = p28._grad_diff(p28._grads(model, got), p28._grads(model, ref))
        assert grads <= tol["grads"], grads
    with torch.no_grad():
        model.cell.cand_norm.weight.fill_(1.0)
        unit = p28._run(lambda: model(tokens, cx, state, reset))
    assert float((unit[0] - got[0]).abs().max()) > 1e-4  # the gain is not skipped


@pytest.mark.usefixtures("_compiled")
def test_p1_13_cand_norm_compiled_window_cpu() -> None:
    """The compiled window (inductor on the CPU) with ``model.cand_norm=rms``."""
    _compiled_vs_eager(resolve(load_hardware_profile("cpu")), p28.CPU_BACKEND, "cpu")


@pytest.mark.cuda
@pytest.mark.usefixtures("_compiled")
def test_p1_13_cand_norm_compiled_window_cuda() -> None:
    """The compiled window with CUDA Graphs and the candidate norm, as on the CPU."""
    _compiled_vs_eager(p28._cuda_ctx(), "inductor", "cuda")


@pytest.mark.cuda
def test_p1_13_cand_norm_autocast_bound_cuda() -> None:
    """Under bf16 autocast (P1.9) the norm runs on, and returns, the state dtype (fp32), and over
    a 256-position window of the reference model with saturating weights the state stays in
    [-1, 1]: the tanh and the convex gate are untouched by the norm."""
    ctx = p28._cuda_ctx()
    cfg, model = p15._model(REF_A1, RMS)
    model = model.to(ctx.device)
    _random_gain(model)
    with torch.no_grad():
        model.cell.W_out.weight.mul_(20.0)
    B, T = 4, 256
    tokens, ctx_ids = p15._ids(cfg, B, T, seed=60)
    state = model.init_state(B)
    reset = torch.zeros(B, T, dtype=torch.bool, device=ctx.device)
    capture = _Capture(model.cell.cand_norm)
    try:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, final, _ = model(tokens.to(ctx.device), ctx_ids.to(ctx.device), state, reset)
    finally:
        capture.close()
    assert all(x.dtype == out.dtype == torch.float32 for x, out in capture.calls)
    assert final.h.dtype == torch.float32 and float(final.h.abs().max()) <= 1.0


LEVERS = {
    "wsd": ["train.schedule=wsd", "train.decay_fraction=0.5"],
    "cell_group": ["train.cell_lr_mult=0.25", "train.cell_weight_decay=0.05"],
    "cand_norm": RMS,
}


@pytest.mark.cuda
@pytest.mark.usefixtures("_compiled")
@pytest.mark.parametrize("lever", list(LEVERS))
def test_p1_13_compiled_training_steps_cuda(lever: str, tmp_path: Path) -> None:
    """A few training steps of the HSN trainer in P2.8's compiled window on CUDA under bf16
    autocast with each lever alone: finite losses, the lever in effect (the WSD learning rates,
    the cell groups' rates, the norm in the model) and no compile after the first steps."""
    cfg = load_config(
        REF_A1,
        overrides=[
            f"profile={CUDA_PROFILE}",
            "hardware.compile_mode=reduce-overhead",
            *SMALL_HSN,
            "train.lr=0.003",
            "train.steps=8",
            "train.warmup_steps=2",
            "train.eval_every=8",
            "train.ckpt_every=8",
            "train.stats_every=1",
            *LEVERS[lever],
        ],
    )
    ctx = resolve(cfg.hardware)
    assert ctx.compile_enabled and ctx.device.type == "cuda"
    train_ds, dev_ds = datasets()
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, ctx, run_dir=tmp_path / "run")
    history = trainer.fit(train_ds, dev_ds, 8)
    assert all(math.isfinite(r.loss) for r in history.steps)
    assert history.measured["compiled_windows"] >= 8.0
    assert 1.0 <= history.measured["compile_steps"] <= 2.0  # compile, then record; no recompile
    lrs = [r.lr for r in history.steps]
    if lever == "wsd":  # warm-up 2, D = 4, decay start 4
        assert lrs == [0.0015, 0.003, 0.003, 0.003, 0.003] + [
            0.003 * (1 - math.sqrt(k / 4)) for k in (1, 2, 3)
        ]
    elif lever == "cell_group":
        cell = _ids(model.cell.parameters())
        for g in trainer.optimizer.param_groups:
            in_cell = bool(_ids(g["params"]) & cell)
            assert g["initial_lr"] == (0.003 * 0.25 if in_cell else 0.003)
    else:
        assert isinstance(model.cell.cand_norm, nn.RMSNorm)
        gain = model.cell.cand_norm.weight
        assert gain is not None and not bool((gain == 1.0).all())  # the gain trained


# --------------------------------------------------------------------------------------------------
# The lever arms' sweep specs
# --------------------------------------------------------------------------------------------------


def test_p1_13_sweep_specs_expand(tmp_path: Path) -> None:
    """Both committed P1.13 specs expand (no database) into the planned arms: five HSN memory
    checks of 600 steps, each with one lever, and six GRU arms on a 3,000-step WSD schedule."""
    from hsn.experiments.spec import expand, load_spec
    from p1_11_support import fake_repo

    repo = fake_repo(tmp_path)
    spec, _ = load_spec(REPO / "configs" / "sweeps" / "p1_13_levers_hsn_h600.yaml")
    hsn_arms = {a.label: a for a in expand(spec, repo=repo)}
    assert sorted(hsn_arms) == sorted(
        f"{n}/s0" for n in ("l1_wsd", "l2_cell_lr", "l3_cand_norm", "r_cand_norm", "r_cell_lr")
    )
    for arm in hsn_arms.values():
        cfg = arm.config
        assert arm.family == "hsn" and arm.steps == 600 and arm.timing
        assert cfg.model.d == 64 and cfg.train.batch == 32
        assert cfg.hardware.compile_mode == "reduce-overhead"
        levers = [
            cfg.train.schedule == "wsd",
            cfg.train.cell_lr_mult != 1.0,
            cfg.model.cand_norm != "none",
        ]
        assert sum(levers) == 1, arm.label  # one lever at a time
    e1 = {k: hsn_arms[f"{k}/s0"].config.train for k in ("l1_wsd", "l2_cell_lr", "l3_cand_norm")}
    assert all(t.lr == 0.003 and t.steps == 3000 and t.warmup_steps == 200 for t in e1.values())
    assert e1["l1_wsd"].decay_fraction == 0.2 and e1["l2_cell_lr"].cell_lr_mult == 0.33
    r = {k: hsn_arms[f"{k}/s0"].config for k in ("r_cand_norm", "r_cell_lr")}
    assert all(c.train.lr == 0.002 and c.train.steps == 9000 for c in r.values())
    assert r["r_cand_norm"].model.cand_norm == "rms" and r["r_cell_lr"].train.cell_lr_mult == 0.5

    spec, _ = load_spec(REPO / "configs" / "sweeps" / "p1_13_levers_gru_wsd.yaml")
    gru_arms = expand(spec, repo=repo)
    assert len(gru_arms) == 6
    for arm in gru_arms:
        train = arm.config.train
        assert arm.family == "gru" and not arm.timing and arm.steps == 3000
        assert (train.schedule, train.decay_fraction, train.steps) == ("wsd", 0.2, 3000)
        assert arm.config.baseline.gru.d_model in (64, 128)
        assert train.lr in (0.002, 0.0025, 0.003)
