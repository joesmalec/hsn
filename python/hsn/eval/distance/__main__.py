"""``python -m hsn.eval.distance``: the by-distance tables of a run that exists.

    python -m hsn.eval.distance --run-dir RUN [--checkpoint step_00027000.pt]
        [--override model.reader_carry=256 ...] [--set far=data/synthetic_far/manifest.json ...]
        [--no-dev] [--max-docs N] [--device cpu|cuda] [--out FILE.json] [--dump DIR]
    python -m hsn.eval.distance --checkpoint RUN/step_00018000.pt ...

    python -m hsn.eval.distance --from-dump DUMP.npz --manifest data/<name>/manifest.json
        [--tasks ...] [--variants ...] [--window 256] [--out FILE.json]

**A run.** The model is built from ``RUN/config.resolved.yaml`` with the overrides (read-time
keys such as ``model.reader_carry``), the checkpoint's weights are loaded for scoring (the
run's weight average when it holds one; :meth:`hsn.train.loop.Trainer.load_for_evaluation`, so
a configuration hash changed by an override is not an obstacle), and every set takes one
:meth:`~hsn.train.loop.Trainer.evaluate` pass: the run's dev set (unless ``--no-dev``), the sets
of the run's own ``eval.by_distance.sets`` and those of ``--set NAME=MANIFEST[:VARIANTS[:TASKS]]``
(comma lists). ``--device cpu`` (default) reads with the CPU profile in fp32; ``cuda`` keeps the
run's hardware and autocast with every compile mode off. Paths in the configuration are relative
to the working directory, as in a run: start it where ``data/`` is.

**A dump.** ``--from-dump`` builds a text set's tables from per-byte losses that exist
(:meth:`hsn.eval.distance.capture.Captured.save`'s layout, which is the grammar evaluation's):
no model, a few seconds.

The tables go to the terminal; ``--out`` writes the JSON (the ``by_distance`` block of
``docs/distance_eval.md`` with what was scored); ``--dump`` writes every set's per-byte dump.

    python -m hsn.eval.distance --frequent-words configs/data/<name>.frequent_words.json
        --manifest data/<name>/manifest.json [--task T] [--threshold 100000] [--also NAME ...]

**A frequent-word list.** Counts the words of a text dataset's training shard (CPU, about 25 s
for 200 MB) and writes the list the words table needs (:mod:`hsn.eval.distance.text`).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hsn.config import (
    REPO_ROOT,
    ByDistanceConfig,
    DistanceSetConfig,
    EvalConfig,
    RunConfig,
    load_config,
)

__all__ = ["main", "score_run"]


def _parse_set(text: str) -> DistanceSetConfig:
    name, sep, rest = text.partition("=")
    if not sep or not rest:
        raise SystemExit(f"--set {text!r} is not NAME=MANIFEST[:VARIANTS[:TASKS]]")
    manifest, _, more = rest.partition(":")
    variants, _, tasks = more.partition(":")
    return DistanceSetConfig.model_validate(
        {
            "name": name,
            "manifest": manifest,
            "variants": [v for v in variants.split(",") if v],
            "tasks": [t for t in tasks.split(",") if t],
        }
    )


def _hardware_overrides(run_config: Path, device: str) -> list[str]:
    if device == "cpu":
        return [f"hardware={REPO_ROOT / 'configs' / 'hardware' / 'cpu.yaml'}", "profile=cpu"]
    text = run_config.read_text(encoding="utf-8")
    return [
        f"hardware.{key}=none"
        for key in ("compile_mode", "baseline_compile")
        if f"  {key}:" in text
    ]


def score_run(
    run_dir: Path,
    *,
    checkpoint: Path | None = None,
    overrides: Sequence[str] = (),
    sets: Sequence[DistanceSetConfig] = (),
    dev: bool = True,
    max_docs: int | None = None,
    device: str = "cpu",
    eval_batch: int = 128,
    dump_dir: Path | None = None,
) -> dict[str, Any]:
    """Score one checkpoint of ``run_dir`` (module docstring); returns the report."""
    import torch

    from hsn.data.dataset import ByteDataset
    from hsn.device import resolve, seed_everything
    from hsn.eval.distance.score import RunScorer
    from hsn.runner import build_model, make_trainer

    run_dir = Path(run_dir)
    run_config = run_dir / "config.resolved.yaml"
    if checkpoint is None:
        found = sorted(run_dir.glob("step_*.pt"))
        if not found:
            raise FileNotFoundError(f"{run_dir}: no step_*.pt checkpoint")
        checkpoint = found[-1]
    elif not Path(checkpoint).is_file():
        checkpoint = run_dir / checkpoint
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA device is visible")
    applied = [*_hardware_overrides(run_config, device), *overrides]
    cfg: RunConfig = load_config(run_config, applied)
    own = cfg.eval.by_distance
    by = ByDistanceConfig(enabled=True, max_docs=max_docs, sets=[*sets] if sets else [*own.sets])
    cfg = cfg.model_copy(update={"eval": EvalConfig(by_distance=by)})
    started = time.perf_counter()
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    with tempfile.TemporaryDirectory(prefix="hsn_distance_") as scratch:
        trainer = make_trainer(model, cfg, ctx, run_dir=scratch)
        loaded = trainer.load_for_evaluation(checkpoint)
        trainer.eval_batch = eval_batch
        dev_ds = None
        if dev:
            if cfg.data.manifest is None:
                raise ValueError(f"{run_config}: data.manifest is not set")
            dev_ds = ByteDataset.load(
                Path(cfg.data.manifest),
                "dev",
                list(cfg.data.tasks) or None,
                list(cfg.data.variants) or None,
                expected_unroll=cfg.train.unroll,
            )
        scorer = RunScorer(cfg, dev_ds)
        block = scorer.final(trainer, step=int(loaded["step"]), dump_dir=dump_dir)
    extra = loaded.get("extra") or {}
    return {
        "run_dir": str(run_dir),
        "checkpoint": Path(checkpoint).name,
        "checkpoint_config_hash": extra.get("config_hash"),
        "overrides": applied,
        "device": str(ctx.device),
        "torch_threads": torch.get_num_threads(),
        "seconds": round(time.perf_counter() - started, 1),
        "by_distance": block,
    }


def _from_dump(args: argparse.Namespace) -> dict[str, Any]:
    from hsn.data.dataset import ByteDataset
    from hsn.data.splits import SplitManifest
    from hsn.eval.distance.capture import Captured
    from hsn.eval.distance.score import SCHEMA, SetScorer

    if args.manifest is None:
        raise SystemExit("--from-dump needs --manifest (the documents the dump scored)")
    ds = ByteDataset.load(args.manifest, "dev", args.tasks or None, args.variants or None)
    if any(str(e.meta["family"]) != "text" for e in ds.examples):
        raise SystemExit("--from-dump: a dump holds losses and hits; only text sets are tabled")
    dataset = SplitManifest.load(args.manifest).name
    scorer = SetScorer(ds, window=args.window, dataset=dataset)
    started = time.perf_counter()
    captured = Captured.from_dump(ds, args.from_dump)
    groups = scorer.tables(captured)
    scored = captured.scored
    source = {
        "manifest": str(args.manifest),
        "tasks": list(args.tasks),
        "variants": list(args.variants),
        "dataset": dataset,
        "split": "dev",
        "n_docs": len(ds),
        "max_docs": None,
    }
    entry = {
        "source": source,
        "accuracy": float(captured.hit[scored].mean()),
        "loss": float(captured.nll[scored].astype("float64").mean()),
        "n_positions": int(scored.sum()),
        "seconds": round(time.perf_counter() - started, 3),
        "groups": groups,
    }
    block = {"schema": SCHEMA, "window": args.window, "sets": {"dump": entry}}
    return {"dump": str(args.from_dump), "by_distance": block}


def _frequent_words(args: argparse.Namespace) -> int:
    from hsn.eval.distance.text import CONTENT_THRESHOLD, frequent_words_file

    if args.manifest is None:
        raise SystemExit("--frequent-words needs --manifest (the dataset's manifest)")
    payload = frequent_words_file(
        args.manifest,
        task=args.task,
        threshold=CONTENT_THRESHOLD if args.threshold is None else args.threshold,
        also=args.also,
    )
    args.frequent_words.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"{args.frequent_words}: {len(payload['words'])} words at or above "
        f"{payload['threshold']} uses"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hsn.eval.distance",
        description="Score a run's checkpoint, or a per-byte dump, by distance.",
    )
    parser.add_argument("--run-dir", type=Path, help="a run directory (config.resolved.yaml)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="a checkpoint file, or its name in --run-dir (default: the run's newest step_*.pt; "
        "alone: its folder is the run directory)",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--set", action="append", default=[], dest="sets", metavar="NAME=MANIFEST")
    parser.add_argument("--no-dev", action="store_true", help="skip the run's own dev set")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--out", type=Path, help="write the report as JSON")
    parser.add_argument("--dump", type=Path, help="write every set's per-byte dump here")
    parser.add_argument("--from-dump", type=Path, help="table a per-byte dump (no model)")
    parser.add_argument("--manifest", type=Path, help="with --from-dump: the scored dataset")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--window", type=int, default=256, help="with --from-dump")
    parser.add_argument("--frequent-words", type=Path, help="write a frequent-word list here")
    parser.add_argument("--task", default=None, help="with --frequent-words: the text task")
    parser.add_argument("--threshold", type=int, default=None, help="with --frequent-words")
    parser.add_argument("--also", nargs="*", default=[], help="with --frequent-words")
    args = parser.parse_args(argv)
    run = args.run_dir is not None or args.checkpoint is not None
    if sum([run, args.from_dump is not None, args.frequent_words is not None]) != 1:
        parser.error("give one of --run-dir (or --checkpoint), --from-dump and --frequent-words")
    if args.frequent_words is not None:
        return _frequent_words(args)
    if args.from_dump is not None:
        report = _from_dump(args)
    else:
        if args.threads is not None:
            import torch

            torch.set_num_threads(args.threads)
        report = score_run(
            args.run_dir if args.run_dir is not None else args.checkpoint.parent,
            checkpoint=args.checkpoint,
            overrides=args.override,
            sets=[_parse_set(text) for text in args.sets],
            dev=not args.no_dev,
            max_docs=args.max_docs,
            device=args.device,
            eval_batch=args.eval_batch,
            dump_dir=args.dump,
        )
    from hsn.eval.distance.score import render

    print(render(report["by_distance"]))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"report: {args.out}")
    failed = [n for n, e in report["by_distance"]["sets"].items() if "error" in e]
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
