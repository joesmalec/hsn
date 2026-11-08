"""P1.15 tests: the divergence dump (``train.divergence_dump``, on by default) and its replay.

At the first step a guard skips (non-finite, or a spike with the guard on) the trainer writes the
state that step started from, in checkpoint format: it equals, tensor for tensor, the checkpoint an
identical run writes after the step before. Replaying it (``hsn.train.divergence.replay``)
regenerates the step's windows and, under deterministic kernels, reproduces the recorded loss and
pre-clip norm bit for bit on the CPU (both trainers), compiled too; on CUDA (cuda marker) two
compiled replays agree bit for bit and the snapshot never synchronises the device. The snapshot
leaves training bitwise unchanged; ``false`` writes nothing; a failing write never stops the run.
"""

from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_config
from hsn.experiments.cli import main as cli_main
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import protocol_params
from hsn.train import compile as compile_mod
from hsn.train import tbptt
from hsn.train.compile import compile_window
from hsn.train.divergence import dump_paths, load_dump, replay
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    disk_overrides,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
)
from p1_15_support import (
    GUARD,
    INF_GRADIENT,
    SPIKE,
    StepProbe,
    datasets,
    gru_config,
    gru_trainer,
    hsn_config,
    hsn_trainer,
    plant,
    plant_by_call,
    plant_gru_on_window,
    plant_on_window,
    window_targets,
    window_tokens,
)

pytestmark = pytest.mark.usefixtures("restore_determinism")
CPU_BACKEND = "inductor" if sys.platform.startswith("linux") else "aot_eager"


@pytest.fixture(autouse=True)
def _release_compiled_state() -> Any:
    yield
    torch._dynamo.reset()


def _equal(a: Any, b: Any, path: str = "") -> None:
    """Nested containers of tensors and plain values, equal bit for bit."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            _equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _equal(x, y, f"{path}/{i}")
    else:
        assert a == b, (path, a, b)


def test_p1_15_dump_is_the_previous_steps_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spike planted at step 6 (guard on) and an infinite gradient at step 7: one dump, at
    step 6, holding exactly what a checkpoint after step 5 of the same run holds (parameters,
    buffers, optimiser, scheduler, generators, loader, carried state, resync and guard counters),
    plus the event, the step's window and its documents; its sidecar JSON names the event."""
    cfg = hsn_config(8, *GUARD)
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, {6: SPIKE, 7: INF_GRADIENT})
    trainer.fit(train_ds, dev_ds, 8)
    (path,) = dump_paths(tmp_path / "run")
    assert path.name == "divergence_00000006.pt"
    summary = trainer.divergence
    assert summary is not None and summary["kind"] == SPIKE and summary["step"] == 6
    assert "error" not in summary and summary["bytes"] == path.stat().st_size
    reference = hsn_trainer(cfg, tmp_path / "ref")
    plant(monkeypatch, reference, {})
    ckpt = reference.fit(train_ds, dev_ds, 5).checkpoints[-1]
    want = torch.load(ckpt, map_location="cpu", weights_only=True)
    got = load_dump(path)
    info = got["extra"].pop("divergence")
    assert got["step"] == 5
    for key in ("model", "optimizer", "scheduler", "rng_state", "loader_state", "extra"):
        _equal(got[key], want[key], key)
    assert info["step"] == 6 and info["grad_norm"] > 1e6 and len(info["windows"]) == 1
    assert info["windows"][0]["tokens"].shape == (4, 32)
    assert len(info["window_docs"][0]) == 4 and info["config"]["train"]["spike_factor"] == 100.0
    assert all(v["finite"] for v in info["gradients"].values() if "finite" in v)
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["kind"] == SPIKE and side["step"] == 6


def test_p1_15_replay_reproduces_the_step_bitwise_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay regenerates the step's window from the dumped loader and, deterministic on the
    CPU, reproduces the recorded loss and pre-clip norm exactly, twice; the raw per-parameter
    gradient norms and the step's skip are reported."""
    cfg = hsn_config(8, *GUARD)
    train_ds, dev_ds = datasets()
    plant_on_window(monkeypatch, window_tokens(cfg, train_ds, 6), SPIKE)
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(train_ds, dev_ds, 8)
    (path,) = dump_paths(tmp_path / "run")
    out = replay(path, train_ds=train_ds, repeats=2)
    assert out["windows_match"] and out["replays_identical"] and not out["compiled"]
    first = out["replays"][0]
    assert first["grad_norm"] == history.steps[5].grad_norm == out["recorded"]["grad_norm"]
    assert first["loss"] == history.steps[5].loss
    assert first["skipped"] == [SPIKE] and first["top_grad_norms"]
    assert first["nonfinite_params"] == []
    assert out["grad_norm_ratio_to_recorded"] == 1.0


def test_p1_15_replay_reproduces_the_step_bitwise_gru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generic trainer (the baselines): the same dump and the same bitwise replay."""
    cfg = gru_config(8, *GUARD)
    train_ds, dev_ds = datasets()
    plant_gru_on_window(monkeypatch, window_targets(cfg, train_ds, 6))
    trainer = gru_trainer(cfg, tmp_path / "run")
    history = trainer.fit(train_ds, dev_ds, 8)
    assert [s["step"] for s in trainer.skipped_steps] == [6]
    (path,) = dump_paths(tmp_path / "run")
    out = replay(path, train_ds=train_ds, repeats=2)
    assert out["windows_match"] and out["replays_identical"]
    assert out["replays"][0]["grad_norm"] == history.steps[5].grad_norm
    assert out["replays"][0]["loss"] == history.steps[5].loss


def test_p1_15_dump_on_by_default_and_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On by default, the dump fires on P1.12's non-finite skip with the spike guard off; a clean
    run writes nothing and trains bit for bit as with the dump off; ``false`` keeps no snapshot."""
    train_ds, dev_ds = datasets()
    assert hsn_config(6).train.divergence_dump is True
    on = hsn_trainer(hsn_config(6), tmp_path / "on")
    off_cfg = hsn_config(6, "train.divergence_dump=false")
    off = hsn_trainer(off_cfg, tmp_path / "off")
    with StepProbe(on) as probe_on:
        a = on.fit(train_ds, dev_ds, 6)
    with StepProbe(off) as probe_off:
        b = off.fit(train_ds, dev_ds, 6)
    assert [(r.loss, r.grad_norm) for r in a.steps] == [(r.loss, r.grad_norm) for r in b.steps]
    for n in range(1, 7):
        assert all(
            torch.equal(x, y) for x, y in zip(probe_on.params[n], probe_off.params[n], strict=True)
        )
    assert dump_paths(tmp_path / "on") == [] and on.divergence is None
    nonfinite = hsn_trainer(hsn_config(6), tmp_path / "nonfinite")
    plant(monkeypatch, nonfinite, {3: INF_GRADIENT})
    nonfinite.fit(train_ds, dev_ds, 6)
    assert nonfinite.divergence is not None and nonfinite.divergence["kind"] == "nonfinite"
    assert [p.name for p in dump_paths(tmp_path / "nonfinite")] == ["divergence_00000003.pt"]
    info = load_dump(dump_paths(tmp_path / "nonfinite")[0])["extra"]["divergence"]
    assert info["grad_norm"] in ("inf", "nan") and info["clip_coef"] is None
    quiet = hsn_trainer(off_cfg, tmp_path / "quiet")
    plant(monkeypatch, quiet, {3: INF_GRADIENT})
    calls: list[int] = []
    monkeypatch.setattr("hsn.train.loop.write_dump", lambda *a, **k: calls.append(1))
    quiet.fit(train_ds, dev_ds, 6)
    assert quiet._step_snapshot is None and calls == [] and dump_paths(tmp_path / "quiet") == []


def test_p1_15_dump_failure_never_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("hsn.train.checkpoint.save_checkpoint", broken)
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(hsn_config(6), tmp_path / "run")
    plant(monkeypatch, trainer, {3: INF_GRADIENT})
    history = trainer.fit(train_ds, dev_ds, 6)
    assert len(history.steps) == 6
    assert trainer.divergence is not None and "disk full" in trainer.divergence["error"]


def test_p1_15_dump_in_the_sweep_record(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reads_state run with a non-finite step at 3: the record names the dump; ``replay
    --dump`` regenerates its window from the run's data on disk (the plant is not active in the
    replay, so the replayed norm is finite) and writes its report."""
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(tiny_manifest, steps=6)])
    params = protocol_params({"name": "reads_state", "diag_every": 2, "probe_batch": 8})
    arm = ArmRun(
        cfg=cfg,
        family="hsn",
        variant="pdg",
        variant_params={},
        params=params,
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={},
    )
    plant_by_call(monkeypatch, {3: INF_GRADIENT})
    record = run_protocol(arm)
    div = record["divergence"]
    assert div["kind"] == "nonfinite" and div["step"] == 3 and Path(div["path"]).is_file()
    info = load_dump(div["path"])["extra"]["divergence"]
    assert info["notes"]["variant"] == "pdg" and info["notes"]["protocol"] == "reads_state"
    plant_by_call(monkeypatch, {})
    out = tmp_path / "replay.json"
    assert cli_main(["replay", "--dump", div["path"], "--repeats", "1", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["windows_match"] and math.isfinite(result["replays"][0]["grad_norm"])
    capsys.readouterr()


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


@pytest.mark.usefixtures("small_blocks")
def test_p1_15_replay_compiled_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compiled run (P2.8's window on the CPU) dumps a planted spike; two compiled replays under
    inductor's deterministic mode agree bit for bit and with the recorded norm; an eager replay
    agrees to P2.8's compiled-versus-eager tolerance."""
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend=CPU_BACKEND)
    )
    cfg = hsn_config(8, *GUARD, "hardware.compile_mode=reduce-overhead")
    train_ds, dev_ds = datasets()
    plant_on_window(monkeypatch, window_tokens(cfg, train_ds, 6), SPIKE)
    trainer = hsn_trainer(cfg, tmp_path / "run")
    assert trainer.compiled_window is not None
    history = trainer.fit(train_ds, dev_ds, 8)
    (path,) = dump_paths(tmp_path / "run")
    assert load_dump(path)["extra"]["divergence"]["compiled"] is True
    compiled = replay(path, train_ds=train_ds, repeats=2)
    assert compiled["compiled"] and compiled["replays_identical"] and compiled["windows_match"]
    assert compiled["replays"][0]["grad_norm"] == history.steps[5].grad_norm
    eager = replay(path, train_ds=train_ds, repeats=1, eager=True)
    got, want = eager["replays"][0]["grad_norm"], history.steps[5].grad_norm
    assert not eager["compiled"] and abs(got - want) <= 1e-4 * abs(want)
    assert "cand_norm_input_rms" not in eager["replays"][0]  # cand_norm is off here


def test_p1_15_replay_reports_the_cand_norm_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``model.cand_norm=rms`` (amendment A4) an eager replay reports the RMS of the
    candidate norm's input over the step's positions (exp 171's near-zero hypothesis)."""
    cfg = hsn_config(8, *GUARD, "model.cand_norm=rms")
    train_ds, dev_ds = datasets()
    plant_on_window(monkeypatch, window_tokens(cfg, train_ds, 6), SPIKE)
    hsn_trainer(cfg, tmp_path / "run").fit(train_ds, dev_ds, 8)
    (path,) = dump_paths(tmp_path / "run")
    out = replay(path, train_ds=train_ds, repeats=1)
    rms = out["replays"][0]["cand_norm_input_rms"]
    assert rms["n"] > 0 and 0.0 <= rms["min"] <= rms["median"]


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
def test_p1_15_dump_and_replay_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On CUDA with CUDA Graphs under bf16 (profile:rtx3060's training path): a spike planted on a
    replayed step is dumped; two compiled replays under deterministic kernels and inductor's
    deterministic mode agree bit for bit; an eager replay agrees to bf16 rounding."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    from cards import CUDA_PROFILE

    cfg = hsn_config(
        8,
        *GUARD,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        "hardware.autocast_dtype=bf16",
        "train.precision=bf16",
    )
    train_ds, dev_ds = datasets()
    plant_on_window(monkeypatch, window_tokens(cfg, train_ds, 6), SPIKE)
    trainer = hsn_trainer(cfg, tmp_path / "run")
    assert trainer.compiled_window is not None and trainer.compiled_window.cuda_graphs
    history = trainer.fit(train_ds, dev_ds, 8)
    (path,) = dump_paths(tmp_path / "run")
    info = load_dump(path)["extra"]["divergence"]
    assert info["compiled"] and info["device"].startswith("cuda")
    compiled = replay(path, train_ds=train_ds, repeats=2)
    assert compiled["windows_match"] and compiled["replays_identical"]
    recorded = history.steps[5].grad_norm
    assert abs(compiled["replays"][0]["grad_norm"] - recorded) <= 1e-2 * recorded
    eager = replay(path, train_ds=train_ds, repeats=1, eager=True)
    assert abs(eager["replays"][0]["grad_norm"] - recorded) <= 5e-2 * recorded


@pytest.mark.cuda
def test_p1_15_snapshot_never_synchronises_cuda(tmp_path: Path) -> None:
    """The per-step snapshot (loader position, generators, copies of the carried state and the
    buffers) runs without a host synchronisation: none is raised under the sync debug mode."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    from cards import CUDA_PROFILE
    from hsn.train.loop import StreamLoader

    cfg = hsn_config(4, f"profile={CUDA_PROFILE}", "hardware.compile_mode=none")
    train_ds, _ = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "run")
    loader = StreamLoader(
        train_ds,
        batch=cfg.train.batch,
        unroll=cfg.train.unroll,
        seed=cfg.seed,
        shuffle=True,
        align_docs=True,
        epochs=None,
    )
    state = trainer.hsn.init_state(cfg.train.batch, trainer.ctx.device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        snapshot = trainer._take_snapshot(loader, state)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert snapshot is not None and snapshot.state.h.device.type == "cuda"
