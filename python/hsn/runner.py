"""P0.8: the ``hsn`` command — run a configuration over seeds, rerun a run, package a gate.

Plan references: chunk P0.8 (interfaces and steps), §4.3 (every run writes
``config.resolved.yaml`` and ``run_manifest.json`` with measured numbers), §0.5 ("measured, not
assumed"), §2.10 (cost arithmetic), Track E (evidence packages).

::

    hsn run     --config configs/<x>.yaml --seeds 0,1,2,3,4 --profile cpu --out runs/
    hsn rerun   runs/<run_id> [--tolerance 0]     # retrains from the resolved config; exact on cpu
    hsn package --gate G0 --runs runs/<a> runs/<b> ... --out evidence/G0.zip

The runner wires config (P0.2) → device (P0.3) → data (P0.4) → model (the registry below: the P0.5
baselines and, since P1.9, the P1.5 HSN) → trainer (:func:`make_trainer`: the P1.7 TBPTT trainer for
the HSN, the P0.5 trainer otherwise) → evaluation → result store (P0.8) → run manifest. One run
directory ``<out>/<run_id>`` holds:

``config.resolved.yaml``, ``run_manifest.json`` (P0.2 writer), ``history.json`` (every step and
evaluation), ``metrics.json`` (the final dev evaluation, measured numbers, hashes, data provenance),
``profile_train.json`` / ``profile_serve.json`` (P0.6 reports of a separate profiling pass),
``step_*.pt`` checkpoints, ``run.log``; a rerun directory adds ``rerun.json``. Results are appended
to ``<out>/results.jsonl`` (:class:`~hsn.eval.results.ResultStore`).

Profiling is a separate pass over a copy of the trained model, not the training run itself: the
P0.6 profiler synchronises the device at every segment boundary (latency-audit conditions), which
would distort the trainer's tokens/s, and its segment names are the HSN ones. For a dense baseline
the whole forward pass is the ``shared_path`` segment (the "update all cells" path of the ablation
table); ``backward`` and ``optimizer`` are what they say; the first window of each pass is
``startup``. The mapping is written into ``metrics.json`` under ``profile_segments``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.baselines.gru import GRUBaseline
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import RunConfig, config_hash, load_config, write_run_manifest
from hsn.data.dataset import ByteDataset
from hsn.device import DeviceContext, resolve, seed_everything, set_vram_cap, synchronize
from hsn.eval.results import (
    FREEZE_PATH,
    HISTORY_FILE,
    METRICS_FILE,
    PROFILE_SERVE_FILE,
    PROFILE_TRAIN_FILE,
    RERUN_FILE,
    RESOLVED_CONFIG_FILE,
    STORE_FILE,
    ResultStore,
    RunRecord,
    evidence_package,
)
from hsn.model.hsn import HSN
from hsn.model.router import SCORED_MODES
from hsn.profiler import (
    NullMeter,
    NvmlEnergyMeter,
    PowerMeter,
    PowermetricsMeter,
    Profiler,
    ProfileReport,
)
from hsn.train.amp import autocast_ctx
from hsn.train.loop import Batch, History, SequenceModel, StreamLoader, Trainer, detach_state
from hsn.train.probes import choose_probe_position, probe_window, window_split
from hsn.train.tbptt import TBPTTTrainer

__all__ = [
    "MODEL_REGISTRY",
    "PROFILE_SEGMENTS",
    "PowerChoice",
    "build_model",
    "load_datasets",
    "main",
    "make_power_meter",
    "make_trainer",
    "model_hash",
    "parse_seeds",
    "profile_run",
    "register_model",
    "rerun",
    "run_one",
    "run_seeds",
]

logger = logging.getLogger("hsn.runner")

ModelFactory = Callable[[RunConfig], nn.Module]
PowerChoice = Literal["none", "profile"]

# --------------------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelFactory] = {
    "gru": GRUBaseline,
    "transformer": TransformerBaseline,
    "hsn": HSN.from_config,  # P1.9: the dense predictive cell (P1.5) trained by TBPTTTrainer (P1.7)
}


def register_model(name: str, factory: ModelFactory, *, replace: bool = False) -> None:
    """Register ``factory(cfg) -> nn.Module`` under ``cfg.model_name == name`` (HSN from P1.5)."""
    if not name or not name.isidentifier():
        raise ValueError(f"model name must be an identifier, got {name!r}")
    if name in MODEL_REGISTRY and not replace:
        raise ValueError(f"model {name!r} is already registered; pass replace=True to override")
    MODEL_REGISTRY[name] = factory


def build_model(cfg: RunConfig) -> nn.Module:
    """The model ``cfg.model_name`` names, freshly constructed (seed before calling)."""
    try:
        factory = MODEL_REGISTRY[cfg.model_name]
    except KeyError:
        raise KeyError(
            f"no model registered as {cfg.model_name!r}; registered: {sorted(MODEL_REGISTRY)}"
        ) from None
    return factory(cfg)


def make_trainer(
    model: nn.Module,
    cfg: RunConfig,
    ctx: DeviceContext,
    *,
    run_dir: str | Path | None = None,
    profiler: Any | None = None,
) -> Trainer:
    """The trainer for ``model`` (P1.9): the P1.7 :class:`TBPTTTrainer` for an :class:`HSN` (its
    window loop, loss assembly and checkpoint contents are the HSN ones, and its ``fit`` reads the
    HSN ``param_report``), the generic P0.5 :class:`Trainer` for everything else."""
    if isinstance(model, HSN):
        return TBPTTTrainer(model, cfg, ctx, profiler, run_dir=run_dir)
    return Trainer(model, cfg, ctx, profiler, run_dir=run_dir)


def _detach(state: Any) -> Any:
    """``state`` without autograd history: the state's own ``detach`` when it has one (the HSN
    :class:`~hsn.model.state.CellGraphState`), else the P0.5 tensor walk."""
    if (
        state is not None
        and not isinstance(state, Tensor)
        and callable(getattr(state, "detach", None))
    ):
        return state.detach()
    return detach_state(state)


# --------------------------------------------------------------------------------------------------
# Data, hashes, meters
# --------------------------------------------------------------------------------------------------


def load_datasets(cfg: RunConfig) -> tuple[ByteDataset, ByteDataset]:
    """The train and dev datasets ``cfg.data`` names (tasks, variants, manifest; hashes checked)."""
    if cfg.data.manifest is None:
        raise ValueError(
            "data.manifest is required: build the data first "
            "(python -m hsn.data.synthetic build --spec ... --out ...)"
        )
    manifest = Path(cfg.data.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"data.manifest {manifest} is not a file (paths are CWD-relative)")
    tasks = list(cfg.data.tasks) or None
    variants = list(cfg.data.variants) or None
    train = ByteDataset.load(manifest, "train", tasks, variants, expected_unroll=cfg.train.unroll)
    dev = ByteDataset.load(manifest, "dev", tasks, variants, expected_unroll=cfg.train.unroll)
    return train, dev


def _data_provenance(cfg: RunConfig, train_ds: ByteDataset, dev_ds: ByteDataset) -> dict[str, Any]:
    manifest = Path(str(cfg.data.manifest))
    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "manifest": str(manifest),
        "dataset": loaded.get("name"),
        "spec_hash": loaded.get("spec_hash"),
        "split_hashes": loaded.get("split_hashes"),
        "tasks": list(cfg.data.tasks),
        "variants": list(cfg.data.variants),
        "n_train_docs": len(train_ds),
        "n_dev_docs": len(dev_ds),
        "align_docs": cfg.data.align_docs,
    }


def model_hash(model: nn.Module) -> str:
    """blake2b-256 over the state dict in key order: name, dtype, shape and the raw bytes of every
    tensor; a non-tensor entry (a module's ``get_extra_state``, which the HSN uses for its
    versions and topology) contributes its canonical JSON instead."""
    digest = hashlib.blake2b(digest_size=32)
    for name, value in sorted(model.state_dict().items()):
        if isinstance(value, Tensor):
            flat = value.detach().to("cpu").contiguous().reshape(-1)
            digest.update(f"{name}:{flat.dtype}:{tuple(value.shape)}\n".encode())
            digest.update(flat.view(torch.uint8).numpy().tobytes())
        else:
            text = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
            digest.update(f"{name}:extra_state:{len(text)}\n".encode())
            digest.update(text.encode())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_power_meter(cfg: RunConfig, choice: PowerChoice) -> PowerMeter:
    """``none`` → :class:`NullMeter`; ``profile`` → the meter ``hardware.power_meter`` names."""
    if choice == "none":
        return NullMeter()
    kind = cfg.hardware.power_meter
    if kind == "none":
        return NullMeter()
    if kind == "powermetrics":
        return PowermetricsMeter()
    if kind == "nvml":
        return NvmlEnergyMeter()
    raise NotImplementedError(
        f"hardware.power_meter={kind!r}: a wall meter is read by hand (plan §3.8); no PowerMeter"
    )


# --------------------------------------------------------------------------------------------------
# Profiling pass
# --------------------------------------------------------------------------------------------------

PROFILE_SEGMENTS: dict[str, str] = {
    "startup": "first window of the pass (warm-up; P0.6 standalone segment)",
    "shared_path": "the complete forward pass of the model (every cell of the dense HSN, or the "
    "dense baseline): training pass with the task cross-entropy, serving pass in eval mode under "
    "no_grad; the HSN prediction term is not part of this profiled loss (P2.13 profiles the HSN "
    "segments separately)",
    "backward": "loss.backward()",
    "optimizer": "gradient clipping and the AdamW step",
    "probe": "P2.3, only when train.lambda_u > 0: one counterfactual utility probe of the window "
    "per profiled step (no-grad rollouts from the window-start state, at the first step past a "
    "scored routing's soft warm-up); a training run pays it on train.probe_fraction of its windows",
}


def _profile_probe(
    model: HSN,
    cfg: RunConfig,
    ctx: DeviceContext,
    train_ds: ByteDataset,
    window: Batch,
    state: Any,
    profiler: Profiler,
    index: int,
) -> None:
    """Time one P2.3 probe of ``window`` from ``state`` (timing only; nothing is kept), at the
    first step whose gate is hard, from a generator seeded by ``(run.seed, index)``."""
    train = cfg.train
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed * 1_000_003 + index)
    t = choose_probe_position(window["loss_mask"], train.probe_horizon, generator)
    if t is None:
        return
    split = window_split(train_ds, window["doc"])
    first = train.soft_until_step if cfg.model.routing in SCORED_MODES else 0
    step = model.train_step
    model.set_train_step(first)
    try:
        with profiler.segment("probe", tokens=_n_tokens(window)), autocast_ctx(ctx):
            probe_window(
                model,
                state,
                {**window, "split": split},
                t,
                n_probe=train.n_probe,
                horizon=train.probe_horizon,
                generator=generator,
            )
    finally:
        model.set_train_step(step)


def _to_device(window: Batch, device: torch.device) -> Batch:
    return {key: value.to(device) for key, value in window.items()}


def _window_loss(logits: Tensor, batch: Batch) -> Tensor:
    """The trainer's loss: cross-entropy over the loss positions, zero when a window has none."""
    loss_mask = batch["loss_mask"]
    if int(loss_mask.sum()) == 0:
        return logits.float().sum() * 0.0
    return F.cross_entropy(logits[loss_mask].float(), batch["targets"][loss_mask])


def _n_tokens(window: Batch) -> int:
    return int((window["doc"] >= 0).sum())


def profile_run(
    model: nn.Module,
    cfg: RunConfig,
    ctx: DeviceContext,
    train_ds: ByteDataset,
    dev_ds: ByteDataset,
    *,
    steps: int = 20,
    power: Callable[[], PowerMeter] = NullMeter,
) -> tuple[ProfileReport, ProfileReport]:
    """P0.6 reports of ``steps`` training steps and ``steps`` serving windows (a copy of ``model``).

    The copy keeps the trained weights of ``model`` untouched. Timing only: the training pass runs
    AdamW at ``train.lr`` without loss scaling, so its losses are not results. With
    ``train.lambda_u > 0`` (an HSN) every profiled training step also times one P2.3 counterfactual
    probe of its window under the ``probe`` segment (:data:`PROFILE_SEGMENTS`).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    device = ctx.device
    count = device.type == "cuda"
    replica = copy.deepcopy(model).to(device=device, dtype=ctx.master_dtype)
    batch_size, unroll = cfg.train.batch, cfg.train.unroll

    train_prof = Profiler(ctx, power=power(), count_launches=count)
    loader = StreamLoader(
        train_ds,
        batch=batch_size,
        unroll=unroll,
        seed=cfg.seed,
        shuffle=True,
        align_docs=cfg.data.align_docs,
        epochs=None,
    )
    params = [p for p in replica.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.train.lr, betas=cfg.train.betas, weight_decay=cfg.train.weight_decay
    )
    max_norm = cfg.train.grad_clip if cfg.train.grad_clip > 0.0 else math.inf
    replica.train()
    api = cast(SequenceModel, replica)
    state: Any = api.init_state(batch_size, device)
    window = _to_device(next(loader), device)
    with train_prof.segment("startup", tokens=_n_tokens(window)):
        with autocast_ctx(ctx):
            logits, state, _ = replica(window["tokens"], window["context"], state, window["reset"])
        _window_loss(logits, window).backward()  # type: ignore[no-untyped-call]
    optimizer.zero_grad(set_to_none=True)
    counterfactual = isinstance(replica, HSN) and cfg.train.lambda_u > 0.0
    for index in range(steps):
        window = _to_device(next(loader), device)
        state = _detach(state)
        tokens = _n_tokens(window)
        if counterfactual:
            assert isinstance(replica, HSN)
            _profile_probe(replica, cfg, ctx, train_ds, window, state, train_prof, index)
        with train_prof.segment("shared_path", tokens=tokens), autocast_ctx(ctx):
            logits, state, _ = replica(window["tokens"], window["context"], state, window["reset"])
            loss = _window_loss(logits, window)
        with train_prof.segment("backward", tokens=tokens):
            loss.backward()  # type: ignore[no-untyped-call]
        with train_prof.segment("optimizer"):
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    serve_prof = Profiler(ctx, power=power(), count_launches=count)
    replica.eval()
    dev_loader: Iterator[Batch] = iter(
        StreamLoader(
            dev_ds,
            batch=batch_size,
            unroll=unroll,
            seed=cfg.seed,
            shuffle=False,
            align_docs=cfg.data.align_docs,
            epochs=1,
        )
    )
    state = api.init_state(batch_size, device)
    with torch.no_grad():
        for index in range(steps + 1):
            try:
                window = _to_device(next(dev_loader), device)
            except StopIteration:
                break
            name = "startup" if index == 0 else "shared_path"
            with serve_prof.segment(name, tokens=_n_tokens(window)), autocast_ctx(ctx):
                _, state, _ = replica(window["tokens"], window["context"], state, window["reset"])
    synchronize(device)
    return train_prof.report(), serve_prof.report()


# --------------------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------------------


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _launches_per_step(report: ProfileReport) -> float:
    """Device launches per training step (sum of the three step segments), -1 when not counted."""
    launches = [
        report.segments[name].launches
        for name in ("shared_path", "backward", "optimizer")
        if name in report.segments
    ]
    if not launches or any(value < 0 for value in launches):
        return -1.0
    n = report.segments["shared_path"].n
    return float(sum(launches)) / n if n else -1.0


def _store_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    common = {
        "config_hash": metrics["config_hash"],
        "model_name": metrics["model_name"],
        "profile": metrics["profile"],
        "seed": metrics["seed"],
        "git_rev": metrics["git_rev"],
        "step": metrics["steps"],
        "is_rerun": metrics["is_rerun"],
    }
    rows: list[dict[str, Any]] = []
    final = metrics["final"]
    for metric in ("loss", "accuracy"):
        rows.append(
            {
                **common,
                "split": "dev",
                "group": "all",
                "metric": metric,
                "value": float(final[metric]),
            }
        )
    for group, entry in final["per_group"].items():
        for metric in ("loss", "accuracy"):
            rows.append(
                {
                    **common,
                    "split": "dev",
                    "group": group,
                    "metric": metric,
                    "value": float(entry[metric]),
                }
            )
    for key, value in metrics["measured"].items():
        rows.append(
            {
                **common,
                "split": "train",
                "group": "all",
                "metric": key,
                "value": None if value is None else float(value),
            }
        )
    rows.append(
        {
            **common,
            "split": "train",
            "group": "all",
            "metric": "model_hash",
            "value": None,
            "text": metrics["model_hash"],
        }
    )
    return rows


def run_one(
    cfg: RunConfig,
    *,
    run_id: str,
    out: Path,
    profile_steps: int = 20,
    power: PowerChoice = "none",
    rerun_of: str | None = None,
) -> Path:
    """Train ``cfg`` once into ``<out>/<run_id>`` and write every run artefact; returns the dir."""
    out = Path(out)
    run_dir = out / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory {run_dir} exists; runs are never overwritten")
    if profile_steps < 1:
        raise ValueError(f"profile_steps must be >= 1, got {profile_steps}")
    started_at = _utc_now()
    started = time.perf_counter()
    run_dir.mkdir(parents=True)
    root_logger = logging.getLogger("hsn")
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        logger.info(
            "run %s: config %s profile %s seed %d", run_id, config_hash(cfg), cfg.profile, cfg.seed
        )
        ctx = resolve(cfg.hardware)
        if cfg.hardware.vram_cap_fraction < 1.0:
            set_vram_cap(ctx.device, cfg.hardware.vram_cap_fraction)
        train_ds, dev_ds = load_datasets(cfg)
        seed_everything(cfg.seed)
        model = build_model(cfg)
        trainer = make_trainer(model, cfg, ctx, run_dir=run_dir)
        history: History = trainer.fit(train_ds, dev_ds, cfg.train.steps)
        train_report, serve_report = profile_run(
            model,
            cfg,
            ctx,
            train_ds,
            dev_ds,
            steps=profile_steps,
            power=lambda: make_power_meter(cfg, power),
        )
        train_report_path = run_dir / PROFILE_TRAIN_FILE
        train_prof_json = train_report.to_dict()
        serve_prof_json = serve_report.to_dict()
        nulled = [0]
        _write_json(run_dir / HISTORY_FILE, _null_nonfinite(json.loads(history.to_json()), nulled))
        _write_json(train_report_path, train_prof_json)
        _write_json(run_dir / PROFILE_SERVE_FILE, serve_prof_json)

        measured = dict(history.measured)
        measured["launches_per_step"] = _launches_per_step(train_report)
        manifest_path = write_run_manifest(run_dir, cfg, measured)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoints = {
            Path(path).name: _sha256_file(Path(path))
            for path in history.checkpoints
            if Path(path).is_file()
        }
        grad_norms = [record.grad_norm for record in history.steps]
        finite_norms = [value for value in grad_norms if math.isfinite(value)]
        finished_at = _utc_now()
        metrics: dict[str, Any] = {
            "run_id": run_id,
            "config_run_id": cfg.run_id,
            "config_hash": manifest["config_hash"],
            "model_name": cfg.model_name,
            "profile": cfg.profile,
            "device": str(ctx.device),
            "autocast": history.autocast,
            "seed": cfg.seed,
            "git_rev": manifest.get("git_rev"),
            "git_dirty": manifest.get("dirty"),
            "is_rerun": rerun_of is not None,
            "rerun_of": rerun_of,
            "steps": int(history.measured["steps"]),
            "param_report": history.param_report,
            "model_hash": model_hash(model),
            "checkpoints": checkpoints,
            "final": history.evals[-1],
            "evals": history.evals,
            "train_loss_final": history.steps[-1].loss,
            "train_accuracy_final": history.steps[-1].accuracy,
            "grad_norms_finite": len(finite_norms) == len(grad_norms),
            "grad_norm_nonfinite_steps": len(grad_norms) - len(finite_norms),
            "grad_norm_max_finite": max(finite_norms) if finite_norms else None,
            "measured": measured,
            "data": _data_provenance(cfg, train_ds, dev_ds),
            "profile_segments": PROFILE_SEGMENTS,
            "profile_steps": profile_steps,
            "power": {"requested": power, "method": train_prof_json["power"]["method"]},
            "torch_threads": torch.get_num_threads(),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_s": time.perf_counter() - started,
        }
        metrics = _null_nonfinite(metrics, nulled)
        metrics["nonfinite_values_nulled"] = nulled[0]
        _write_json(run_dir / METRICS_FILE, metrics)
        store = ResultStore(out / STORE_FILE)
        for row in _store_rows(metrics):
            store.append(run_id, row)
        logger.info(
            "run %s done in %.1f s: dev loss %.4f accuracy %.4f (%s)",
            run_id,
            metrics["elapsed_s"],
            metrics["final"]["loss"],
            metrics["final"]["accuracy"],
            run_dir,
        )
    finally:
        root_logger.removeHandler(handler)
        handler.close()
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _null_nonfinite(value: Any, counter: list[int]) -> Any:
    """``value`` with every non-finite float replaced by ``None`` (counted in ``counter[0]``).

    Strict JSON has no inf or nan. A loss scaler (fp16 profiles) skips the steps whose gradients
    overflowed, and the trainer records those steps with an infinite gradient norm; the run files
    keep them as ``null`` and ``metrics.json`` says how many values were nulled.
    """
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        counter[0] += 1
        return None
    if isinstance(value, dict):
        return {key: _null_nonfinite(item, counter) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_null_nonfinite(item, counter) for item in value]
    return value


# --------------------------------------------------------------------------------------------------
# run / rerun
# --------------------------------------------------------------------------------------------------


def parse_seeds(text: str) -> list[int]:
    """``"0,1,2"`` → ``[0, 1, 2]``; empty, negative or repeated seeds raise."""
    seeds: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"empty seed in {text!r}")
        value = int(item)
        if value < 0:
            raise ValueError(f"seeds must be >= 0, got {value}")
        if value in seeds:
            raise ValueError(f"seed {value} repeats in {text!r}")
        seeds.append(value)
    return seeds


def run_seeds(
    config: str | Path,
    seeds: Sequence[int],
    *,
    out: str | Path = "runs",
    profile: str | None = None,
    overrides: Sequence[str] = (),
    profile_steps: int = 20,
    power: PowerChoice = "none",
) -> list[Path]:
    """``hsn run``: one run directory ``<out>/<run_id>-<profile>-s<seed>`` per seed, in order."""
    if not seeds:
        raise ValueError("at least one seed is required")
    base_overrides = [f"profile={profile}"] if profile else []
    base = load_config(config, overrides=[*base_overrides, *overrides])
    run_dirs: list[Path] = []
    for seed in seeds:
        run_id = f"{base.run_id}-{base.profile}-s{seed}"
        cfg = load_config(
            config,
            overrides=[*base_overrides, *overrides, f"seed={seed}", f"run_id={run_id}"],
        )
        run_dirs.append(
            run_one(cfg, run_id=run_id, out=Path(out), profile_steps=profile_steps, power=power)
        )
    return run_dirs


def _final_diffs(original: Mapping[str, Any], again: Mapping[str, Any]) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for metric in ("loss", "accuracy"):
        diffs[f"all/{metric}"] = abs(float(original[metric]) - float(again[metric]))
    groups_a, groups_b = original["per_group"], again["per_group"]
    if set(groups_a) != set(groups_b):
        raise ValueError(
            f"the rerun evaluated different groups: {sorted(groups_a)} vs {sorted(groups_b)}"
        )
    for group, entry in groups_a.items():
        for metric in ("loss", "accuracy"):
            diffs[f"{group}/{metric}"] = abs(float(entry[metric]) - float(groups_b[group][metric]))
    return diffs


def rerun(
    run_dir: str | Path,
    *,
    tolerance: float = 0.0,
    out: str | Path | None = None,
    profile_steps: int = 20,
    power: PowerChoice = "none",
) -> Path:
    """``hsn rerun``: retrain ``run_dir`` from its resolved config and compare the final metrics.

    With ``tolerance == 0`` every final dev metric (loss and accuracy, overall and per group) must
    agree exactly and the model hashes must be equal (exact on ``profile:cpu``). With a tolerance
    (GPU profiles: cuDNN and MPS kernels are not bit-reproducible, P0.8 pitfalls) the overall dev
    accuracy, the primary metric of the synthetic suites (``metrics_freeze.json``), must agree
    within it (``gated_abs_diff``); the per-group accuracy differences (``max_abs_diff`` is the
    largest, overall or per group) and the loss differences (nats, another unit;
    ``max_abs_diff_loss``) are recorded alongside but not gated: on a steep part of a learning
    curve a single task moves by more than the tolerance between two MPS runs of the same seed
    (G0 measurements). The verdict goes to ``<rerun dir>/rerun.json`` and the result store; the
    rerun directory is returned.
    """
    source = Path(run_dir)
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite number >= 0, got {tolerance}")
    original = RunRecord.load(source)
    if original.is_rerun:
        raise ValueError(f"{source} is itself a rerun; rerun its original instead")
    cfg = load_config(source / RESOLVED_CONFIG_FILE)
    if config_hash(cfg) != original.metrics["config_hash"]:
        raise ValueError(
            f"{source / RESOLVED_CONFIG_FILE} no longer hashes to the recorded configuration "
            f"{original.metrics['config_hash']}"
        )
    new_id = f"{original.run_id}-rerun"
    target = Path(out) if out is not None else source.parent
    new_dir = run_one(
        cfg,
        run_id=new_id,
        out=target,
        profile_steps=profile_steps,
        power=power,
        rerun_of=original.run_id,
    )
    again = json.loads((new_dir / METRICS_FILE).read_text(encoding="utf-8"))
    diffs = _final_diffs(original.final, again["final"])
    gated_abs_diff = diffs["all/accuracy"]
    max_abs_diff = max(value for key, value in diffs.items() if key.endswith("/accuracy"))
    max_abs_diff_loss = max(value for key, value in diffs.items() if key.endswith("/loss"))
    hashes_equal = bool(original.metrics["model_hash"] == again["model_hash"])
    if tolerance == 0.0:
        passed = max(diffs.values()) == 0.0 and hashes_equal
    else:
        passed = gated_abs_diff <= tolerance
    verdict = {
        "original_run_id": original.run_id,
        "original_dir": str(source),
        "rerun_run_id": new_id,
        "rerun_dir": str(new_dir),
        "profile": cfg.profile,
        "config_hash": again["config_hash"],
        "tolerance": tolerance,
        "exact": tolerance == 0.0,
        "metric": "accuracy",
        "gated": "everything" if tolerance == 0.0 else "all/accuracy",
        "gated_abs_diff": gated_abs_diff,
        "diffs": diffs,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_loss": max_abs_diff_loss,
        "model_hash_equal": hashes_equal,
        "original_model_hash": original.metrics["model_hash"],
        "rerun_model_hash": again["model_hash"],
        "original_git_rev": original.metrics.get("git_rev"),
        "rerun_git_rev": again.get("git_rev"),
        "passed": passed,
        "compared_at": _utc_now(),
    }
    _write_json(new_dir / RERUN_FILE, verdict)
    store = ResultStore(target / STORE_FILE)
    row: dict[str, Any] = {
        "config_hash": again["config_hash"],
        "model_name": again["model_name"],
        "profile": cfg.profile,
        "seed": cfg.seed,
        "git_rev": again.get("git_rev"),
        "step": again["steps"],
        "is_rerun": True,
        "split": "dev",
        "group": "all",
        "text": "PASS" if passed else "FAIL",
    }
    store.append(new_id, {**row, "metric": "rerun_abs_diff_all", "value": gated_abs_diff})
    store.append(new_id, {**row, "metric": "rerun_max_abs_diff", "value": max_abs_diff})
    logger.info(
        "rerun of %s: overall |diff| accuracy %.3g (tolerance %g), largest per-task %.3g, "
        "loss %.3g, model hash equal %s -> %s",
        original.run_id,
        gated_abs_diff,
        tolerance,
        max_abs_diff,
        max_abs_diff_loss,
        hashes_equal,
        "PASS" if passed else "FAIL",
    )
    return new_dir


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hsn", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train a configuration over seeds, one run directory each")
    run.add_argument("--config", required=True, help="run configuration YAML")
    run.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2,3,4")
    run.add_argument("--profile", help="hardware profile (configs/hardware/<name>.yaml)")
    run.add_argument("--out", default="runs", help="parent of the run directories")
    run.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override",
    )
    run.add_argument("--profile-steps", type=int, default=20, help="steps in the profiling pass")
    run.add_argument(
        "--power",
        choices=("none", "profile"),
        default="none",
        help="energy meter: none, or the profile's hardware.power_meter",
    )

    again = sub.add_parser("rerun", help="retrain a run from its resolved config and compare")
    again.add_argument("run_dir", help="runs/<run_id>")
    again.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="0 = exact (cpu); otherwise the overall dev accuracy must agree within it",
    )
    again.add_argument("--out", help="parent of the rerun directory (default: next to run_dir)")
    again.add_argument("--profile-steps", type=int, default=20)
    again.add_argument("--power", choices=("none", "profile"), default="none")

    package = sub.add_parser("package", help="assemble a gate evidence zip")
    package.add_argument("--gate", required=True, help="G0 ... G6")
    package.add_argument("--runs", nargs="+", required=True, help="run directories (incl. reruns)")
    package.add_argument("--out", required=True, help="evidence/<gate>.zip")
    package.add_argument("--freeze", default=str(FREEZE_PATH), help="metrics freeze file")
    package.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="extra file to include at the package root (e.g. decision.json=...)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``hsn`` console script; returns the exit code."""
    args = _parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "run":
        run_dirs = run_seeds(
            args.config,
            parse_seeds(args.seeds),
            out=args.out,
            profile=args.profile,
            overrides=args.override,
            profile_steps=args.profile_steps,
            power=args.power,
        )
        for run_dir in run_dirs:
            print(run_dir)
        return 0
    if args.command == "rerun":
        new_dir = rerun(
            args.run_dir,
            tolerance=args.tolerance,
            out=args.out,
            profile_steps=args.profile_steps,
            power=args.power,
        )
        verdict = json.loads((new_dir / RERUN_FILE).read_text(encoding="utf-8"))
        print(
            f"{new_dir}: {'PASS' if verdict['passed'] else 'FAIL'} overall |diff| accuracy "
            f"{verdict['gated_abs_diff']:.3g} (tolerance {verdict['tolerance']:g}), largest "
            f"per-task {verdict['max_abs_diff']:.3g}, loss {verdict['max_abs_diff_loss']:.3g}, "
            f"model hash equal {verdict['model_hash_equal']}"
        )
        return 0 if verdict["passed"] else 1
    if args.command == "package":
        extras: dict[str, Path] = {}
        for item in args.extra:
            name, sep, path = item.partition("=")
            if not sep or not name or not path:
                raise SystemExit(f"--extra expects NAME=PATH, got {item!r}")
            extras[name] = Path(path)
        out = evidence_package(
            args.gate,
            [Path(p) for p in args.runs],
            Path(args.out),
            freeze=args.freeze,
            extra_files=extras,
        )
        print(out)
        return 0
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
