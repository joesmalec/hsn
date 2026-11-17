"""The generic trainer's host reads (``train.host_sync``, ``train.fused_adamw``,
``docs/trainer_host_sync.md``, 17 Nov 2026).

At ``host_sync = stats`` a step's loss, accuracy and gradient norm stay on the device and the host
reads them one step later (or at a stats, evaluation or checkpoint step, or when a record is
read); the loss is the masked mean and a non-finite step is skipped on the device by the fused
AdamW. These tests check that the defaults leave hashes, memory keys and the trainer as they
were; the refusals; that the masked loss is the indexed loss (values and gradients); that the
opt-in gives today's losses, accuracies, gradient norms, evaluations and log lines to rounding
(memory-off HSN-T and the GRU); that a non-finite step is skipped exactly as the host guard skips
it (weights, moments, step count) and counted, logged and stopped on as before; that a record
read early (a callback) holds the right values; and, on a card, that the steps between stats steps
synchronise with the host nowhere.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from hsn.config import HASH_OMIT_AT_DEFAULT, canonical_json, config_hash, load_config
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.runner import build_model, make_trainer
from hsn.train.loop import History, StepRecord, Trainer, masked_cross_entropy
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, train_overrides

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
NOMEM_160x3 = REPO / "configs" / "experiments" / "hsn_t_nomem_top0_w1024_l3.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"

FUSED = ("train.fused_adamw=true",)
STATS = ("train.host_sync=stats", "train.fused_adamw=true", "train.divergence_dump=false")
#: the memory-off recipe at a size the CPU trains in a moment
TINY = ("model.d=8", "model.context_dim=8", "model.hidden=16", "model.reader_width=16")
TINY = (*TINY, "model.reader_heads=2")


def _nomem(steps: int, *extra: str) -> Any:
    over = [*train_overrides(steps), *TINY, "train.unroll=16", "train.lr=3e-3", *extra]
    return load_config(NOMEM, over)


def _gru(steps: int, *extra: str) -> Any:
    return gru_config(steps, "train.unroll=16", "train.lr=3e-3", *extra)


def _fit(cfg: Any, tmp: Path, callback: Any = None) -> tuple[Trainer, History]:
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp)
    assert type(trainer) is Trainer
    return trainer, trainer.fit(*datasets(), cfg.train.steps, callback=callback)


def _close(a: float, b: float, tol: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(b))


def _same_history(got: History, ref: History, tol: float) -> None:
    assert len(got.steps) == len(ref.steps)
    for g, r in zip(got.steps, ref.steps, strict=True):
        assert (g.step, g.lr, g.tokens, g.loss_positions) == (
            r.step,
            r.lr,
            r.tokens,
            r.loss_positions,
        )
        for key in ("loss", "accuracy", "grad_norm"):
            assert _close(getattr(g, key), getattr(r, key), tol), (g.step, key, g, r)
    assert [e["step"] for e in got.evals] == [e["step"] for e in ref.evals]
    for g, r in zip(got.evals, ref.evals, strict=True):
        assert _close(g["loss"], r["loss"], tol) and _close(g["accuracy"], r["accuracy"], tol)


# --------------------------------------------------------------------------------------------------
# The keys: opt-in, hashes, memory keys, refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [NOMEM, NOMEM_160x3, GRU_2M, READER_LM])
def test_keys_are_opt_in_and_the_defaults_keep_hashes_and_memory_keys(path: Path) -> None:
    base = load_config(path)
    assert base.train.host_sync == "per_step" and base.train.fused_adamw is False
    dumped = json.loads(canonical_json(base))["train"]
    assert "host_sync" not in dumped and "fused_adamw" not in dumped
    assert {"train.host_sync", "train.fused_adamw"} <= set(HASH_OMIT_AT_DEFAULT)
    named = load_config(path, ["train.host_sync=per_step", "train.fused_adamw=false"])
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert config_hash(named) == config_hash(base)
    assert memory_key(named, "none", protocol) == memory_key(base, "none", protocol)
    keys = {memory_key(base, "none", protocol)}
    hashes = {config_hash(base)}
    for switch in (FUSED, STATS):
        other = load_config(path, list(switch))
        hashes.add(config_hash(other))
        keys.add(memory_key(other, "none", protocol))
    assert len(hashes) == 3 and len(keys) == 3


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, [*FUSED], "trains with the HSN trainer"),
        (RECIPE_A6, [*STATS], "trains with the HSN trainer"),
        (
            RECIPE_A6,
            [*FUSED, "model_name=hsn_t", "model.reader=transformer", "model.k_select=12"],
            "trains with the HSN trainer",
        ),
        (
            NOMEM,
            ["train.host_sync=stats", "train.divergence_dump=false"],
            "needs train.fused_adamw",
        ),
        (NOMEM, ["train.host_sync=stats", *FUSED], "train.divergence_dump=True"),
        (GRU_2M, [*STATS, "train.spike_factor=1000"], "train.spike_factor=1000.0"),
        (GRU_2M, [*STATS, "train.rollback_every=50"], "train.rollback_every=50"),
        (GRU_2M, [*STATS, "train.adaptive_lr.enabled=true"], "train.adaptive_lr.enabled"),
        (
            GRU_2M,
            [
                *STATS,
                "train.gradient_growth.enabled=true",
                'train.gradient_growth.groups={a: ["*"]}',
            ],
            "train.gradient_growth.enabled",
        ),
        (
            NOMEM,
            [*STATS, 'train.update_limit.groups={r: {selectors: ["reader.*"], max_ratio: 0.01}}'],
            "train.update_limit.groups",
        ),
        (GRU_2M, [*STATS, "train.weight_ema=0.9"], "train.weight_ema=0.9"),
        (GRU_2M, [*STATS, "train.answer_weight=2.0"], "train.answer_weight=2.0"),
        (GRU_2M, [*STATS, "train.grad_norm_precision=float64"], "grad_norm_precision"),
        (GRU_2M, [*STATS, "train.precision=fp16"], "train.precision='fp16'"),
        (GRU_2M, ["train.host_sync=fast"], "host_sync"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("(", r"\(")):
        load_config(path, overrides)


def test_accepted_settings() -> None:
    for path in (NOMEM, NOMEM_160x3, GRU_2M, READER_LM):
        assert load_config(path, list(FUSED)).train.fused_adamw
        assert load_config(path, list(STATS)).train.host_sync == "stats"
    # fused alone keeps every per-step feature
    cfg = load_config(GRU_2M, [*FUSED, "train.spike_factor=1000", "train.rollback_every=50"])
    assert cfg.train.divergence_dump and cfg.train.spike_factor == 1000


def test_fused_adamw_builds_the_fused_optimiser(tmp_path: Path) -> None:
    for extra, fused in (((), False), (FUSED, True)):
        cfg = _gru(2, *extra)
        trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
        assert all(bool(g["fused"]) == fused for g in trainer.optimizer.param_groups)
        decays = [g["weight_decay"] for g in trainer.optimizer.param_groups]
        assert decays == [cfg.train.weight_decay, 0.0]


def test_the_fit_refuses_hooks_that_read_a_step_on_the_host(tmp_path: Path) -> None:
    cfg = _gru(2, *STATS)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
    trainer.grad_hook = lambda t: None
    with pytest.raises(RuntimeError, match="grad_hook / update_post_hooks"):
        trainer.fit(*datasets(), 2)
    trainer.grad_hook = None
    trainer.update_post_hooks.append(lambda t: None)
    with pytest.raises(RuntimeError, match="grad_hook / update_post_hooks"):
        trainer.fit(*datasets(), 2)


# --------------------------------------------------------------------------------------------------
# The masked loss
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_the_masked_loss_is_the_indexed_loss(dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    B, T, V = 3, 17, 11
    base = torch.randn(B, T, V, generator=g, dtype=torch.float64).to(dtype)
    targets = torch.randint(0, V, (B, T), generator=g)
    mask = torch.rand(B, T, generator=g) < 0.6
    count = int(mask.sum())
    results = []
    for masked in (False, True):
        logits = base.clone().requires_grad_(True)
        if masked:
            loss, correct = masked_cross_entropy(logits, targets, mask, count)
        else:
            flat, tg = logits[mask].float(), targets[mask]
            loss, correct = F.cross_entropy(flat, tg), (flat.argmax(dim=-1) == tg).sum()
        loss.backward()
        results.append((loss.detach(), int(correct), logits.grad))
    (ref, ref_c, ref_g), (got, got_c, got_g) = results
    tol = {torch.float64: 1e-13, torch.float32: 1e-6, torch.bfloat16: 1e-6}[dtype]
    assert got.dtype == ref.dtype and abs(float(got - ref)) <= tol * float(ref)
    assert got_c == ref_c
    assert float((got_g.double() - ref_g.double()).abs().max()) <= tol
    assert float(got_g[~mask].abs().max()) == 0.0  # nothing reaches a position off the mask


# --------------------------------------------------------------------------------------------------
# Training: the same steps, read later
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["nomem", "gru"])
def test_stats_gives_todays_steps_evaluations_and_log_lines(
    model: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Twenty steps (statistics every 5, an evaluation at 10 and 20, windows of 16 over documents
    of 16 to 32 bytes, so windows without a loss position too): per_step with the fused
    optimiser against stats, the same records, evaluations and log lines to float32 rounding
    (the masked loss sums in another order); and the default (foreach) optimiser to rounding."""
    make = _nomem if model == "nomem" else _gru
    extra = ("train.stats_every=5", "train.eval_every=10", "train.divergence_dump=false")
    logs = {}
    histories = {}
    for name, keys in (("default", ()), ("fused", FUSED), ("stats", STATS)):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="hsn.train"):
            _, histories[name] = _fit(make(20, *extra, *keys), tmp_path / name)
        logs[name] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("step ")]
    _same_history(histories["stats"], histories["fused"], 1e-5)
    _same_history(histories["fused"], histories["default"], 1e-4)
    assert all(math.isnan(s.loss) == (s.loss_positions == 0) for s in histories["stats"].steps)
    assert any(s.loss_positions == 0 for s in histories["stats"].steps)
    assert len(logs["stats"]) == len(logs["fused"]) == 4
    for got, ref in zip(logs["stats"], logs["fused"], strict=True):
        g, r = got.split(), ref.split()
        assert g[:2] == r[:2] and g[6:8] == r[6:8]  # step, lr
        assert all(_close(float(g[i]), float(r[i]), 1e-3) for i in (3, 5, 9))  # loss, acc, norm
    measured = histories["stats"].measured
    assert measured["skipped_nonfinite_steps"] == 0.0 and measured["steps"] == 20.0


def test_a_record_read_early_holds_its_values(tmp_path: Path) -> None:
    """A callback that reads every step's record at once (the harness reads them at its early-stop
    steps): the values are read then, the same as booked; the records pickle, copy and dump as
    plain step records."""
    seen: list[tuple[float, float, float]] = []

    def callback(trainer: Trainer, history: History) -> bool:
        rec = history.steps[-1]
        seen.append((rec.loss, rec.accuracy, rec.grad_norm))
        return False

    cfg = _nomem(8, *STATS)
    _, history = _fit(cfg, tmp_path / "a", callback)
    _, plain = _fit(cfg, tmp_path / "b")
    assert len(seen) == 7  # the callback runs before the last step
    for (loss, acc, norm), rec in zip(seen, plain.steps, strict=False):
        assert _close(loss, rec.loss, 0.0) and _close(acc, rec.accuracy, 0.0)
        assert norm == rec.grad_norm
    record = history.steps[0]
    assert type(pickle.loads(pickle.dumps(record))) is StepRecord
    assert dataclasses.asdict(copy.deepcopy(record)) == dataclasses.asdict(record)
    assert json.loads(history.to_json())["steps"][0]["grad_norm"] == record.grad_norm


# --------------------------------------------------------------------------------------------------
# The non-finite guard on the device
# --------------------------------------------------------------------------------------------------


def test_the_fused_step_with_found_inf_changes_nothing() -> None:
    """The device skip: AdamW(fused=True) with ``found_inf`` = 1 leaves the weights, both moments
    and the step count as they were, bit for bit (the host guard never calls the step)."""
    g = torch.Generator().manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5, 3, generator=g))
    opt = torch.optim.AdamW([p], lr=1e-2, fused=True)
    for _ in range(3):
        p.grad = torch.randn(5, 3, generator=g)
        opt.found_inf = torch.zeros(())  # type: ignore[attr-defined]
        opt.step()
    before = (p.detach().clone(), {k: v.clone() for k, v in opt.state[p].items()})
    p.grad = torch.full((5, 3), math.nan)
    opt.found_inf = torch.ones(())  # type: ignore[attr-defined]
    opt.step()
    assert torch.equal(p.detach(), before[0])
    assert all(torch.equal(opt.state[p][k], v) for k, v in before[1].items())


class _Poison:
    """A forward hook that makes the model's logits NaN at the given calls of the training
    windows (counted from 1)."""

    def __init__(self, calls: set[int]) -> None:
        self.calls, self.n = calls, 0

    def __call__(self, module: Any, args: Any, out: Any) -> Any:
        if not module.training:
            return out
        self.n += 1
        if self.n in self.calls:
            return (out[0] * math.nan, *out[1:])
        return out


@pytest.mark.parametrize("model", ["nomem", "gru"])
def test_a_non_finite_step_is_skipped_counted_and_logged_as_before(
    model: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make = _nomem if model == "nomem" else _gru
    out = {}
    for name, keys in (("fused", (*FUSED, "train.divergence_dump=false")), ("stats", STATS)):
        cfg = make(12, *keys)
        seed_everything(0)
        trainer = make_trainer(
            build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / name
        )
        trainer.model.register_forward_hook(_Poison({4, 9}))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hsn.train"):
            history = trainer.fit(*datasets(), 12)
        warnings = [r.getMessage() for r in caplog.records if "not finite" in r.getMessage()]
        params = [p.detach().clone() for p in trainer.model.parameters()]
        steps = {int(s["step"].item()) for s in trainer.optimizer.state.values()}
        out[name] = (history, trainer.skipped_steps, warnings, params, steps, trainer)
    (h_ref, skips_ref, warn_ref, p_ref, steps_ref, _), (h, skips, warn, p, steps, tr) = (
        out["fused"],
        out["stats"],
    )
    assert skips == skips_ref and [s["step"] for s in skips] == [4, 9]
    assert len(warn) == len(warn_ref) == 2 and warn == warn_ref
    assert steps == steps_ref == {10}  # 12 steps, 2 skipped: the step count went back
    assert tr.skipped_nonfinite == 2 and h.measured["skipped_nonfinite_steps"] == 2.0
    assert all(math.isnan(h.steps[i - 1].grad_norm) for i in (4, 9))
    _same_history(h, h_ref, 1e-5)
    assert all(float((a - b).abs().max()) <= 1e-4 for a, b in zip(p, p_ref, strict=True))


def test_the_stop_rule_stops_the_run_with_todays_message(tmp_path: Path) -> None:
    """Ten consecutive non-finite steps: per_step raises at the tenth, stats when it reads the
    tenth (one step later); the message is the same."""
    messages = {}
    for name, keys in (("fused", (*FUSED, "train.divergence_dump=false")), ("stats", STATS)):
        cfg = _gru(30, *keys)
        seed_everything(0)
        trainer = make_trainer(
            build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / name
        )
        trainer.model.register_forward_hook(_Poison(set(range(3, 30))))
        with pytest.raises(RuntimeError, match="P1.12 stop rule") as err:
            trainer.fit(*datasets(), 30)
        messages[name] = (str(err.value), trainer.step)
    assert messages["stats"][0] == messages["fused"][0]
    assert "step 12: 10 consecutive optimizer steps skipped" in messages["stats"][0]
    # per_step raises inside step 12 (before it counts); stats reads step 12 during step 13
    assert messages["fused"][1] == 11 and messages["stats"][1] == 13


def test_a_non_finite_loss_of_an_applied_step_still_raises(tmp_path: Path) -> None:
    """The check after the guard: a loss that is not finite while the gradient norm is (a NaN
    logit at a position whose gradient does not reach a parameter) raises as before."""
    cfg = _gru(6, *STATS)
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
    book = trainer._book_host_values
    record = StepRecord(
        step=3, loss=math.nan, accuracy=0.0, lr=1e-3, grad_norm=1.0, tokens=4, loss_positions=4,
        seconds=0.0,
    )  # fmt: skip
    with pytest.raises(RuntimeError, match="loss is nan at step 3"):
        book(record, 4)
    book(dataclasses.replace(record, grad_norm=math.nan), 4)  # skipped: no loss check
    assert trainer.skipped_nonfinite == 1


# --------------------------------------------------------------------------------------------------
# CUDA (skipped without a card)
# --------------------------------------------------------------------------------------------------


def _cuda_cfg(steps: int, *extra: str) -> Any:
    from cards import CUDA_PROFILE

    return load_config(
        NOMEM_160x3,
        [
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
            *extra,
        ],
    )


@pytest.mark.cuda
def test_on_the_card_stats_steps_synchronise_nowhere_and_match_per_step(tmp_path: Path) -> None:
    """The memory-off 160 x 3 model under CUDA graphs and bf16: steps 4 to 9 of stats run under
    ``torch.cuda.set_sync_debug_mode('error')`` (any synchronising call raises: a host read, a
    boolean index, a blocking copy); the losses and norms are per_step's (fused) to bf16
    rounding, and the event of the step before is all the host waits for."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    steps = 12

    def strict(trainer: Trainer, history: History) -> bool:
        if trainer.step == 3:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
        if trainer.step == 9:
            torch.cuda.set_sync_debug_mode("default")
        return False

    histories = {}
    for name, keys in (("fused", (*FUSED, "train.divergence_dump=false")), ("stats", STATS)):
        cfg = _cuda_cfg(steps, *keys, "model.reader_mask=causal")
        try:
            _, histories[name] = _fit(cfg, tmp_path / name, strict if name == "stats" else None)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    ref, got = histories["fused"], histories["stats"]
    for g, r in zip(got.steps, ref.steps, strict=True):
        assert _close(g.loss, r.loss, 0.02) and _close(g.grad_norm, r.grad_norm, 0.1), (g, r)
    assert got.measured["skipped_nonfinite_steps"] == 0.0
