#!/usr/bin/env python
"""G1 diagnosis (user brief of 5 Nov 2026, items 2-4): train-subset vs dev accuracy of every
finished checkpoint, the chance floors of the frozen tasks, and the byte-embedding row norms of the
HSN runs at initialisation and after training.

    python scripts/g1_diagnosis_eval.py --runs <gate run dir> --out evidence/G1/diagnosis

For every run directory with a metrics.json (search trials excluded unless --include-search), the
final checkpoint is evaluated with the trainer's own evaluate() on (a) a fixed held-out subset of
the TRAIN split, the last --n-train documents of each frozen task in manifest order, and (b) the
whole dev split of the frozen tasks. "Held-out" names a fixed evaluation subset: every train
document was available to training (12,000 steps at batch 16 cover the 40,000 training documents
several times). The chance floors are computed from the data: 1/|value alphabet| for delayed (the
brief's definition) and the majority-answer rate over the answer positions (NONE included) for kv,
on dev and on the
train subset. Embedding norms: mean/min/max L2 norm of the byte_emb rows for train keys (64-111),
dev keys (112-127), test keys (128-143), values (144-191) and distractors (192-255), at the seeded
initialisation (seed_everything(cfg.seed) then build_model, as the runner does) and at the final
checkpoint. Everything measured goes to <out>/diagnosis_eval.json; nothing in the repository or the
run directories is modified (evaluation scratch goes under <out>).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint

G1_TASKS = ("kv", "delayed")
GROUPS = {
    "control_0_15": (0, 16),
    "markers_16_31": (16, 32),
    "train_keys_64_111": (64, 112),
    "dev_keys_112_127": (112, 128),
    "test_keys_128_143": (128, 144),
    "values_144_191": (144, 192),
    "distractors_192_255": (192, 256),
}
NONE = 5


def answer_distribution(ds: ByteDataset) -> dict[str, Any]:
    """Majority-answer rate per task over the answer (loss_mask) positions, NONE included."""
    counts: dict[str, Counter[int]] = {}
    for i, ex in enumerate(ds.examples):
        task = str(ds.meta(i).get("task"))
        targets = np.asarray(ex.targets)[np.asarray(ex.loss_mask, dtype=bool)]
        counts.setdefault(task, Counter()).update(int(t) for t in targets.tolist())
    out = {}
    for task, c in counts.items():
        n = sum(c.values())
        sym, k = c.most_common(1)[0]
        out[task] = {
            "n_answer_positions": n,
            "majority_symbol": sym,
            "majority_symbol_is_NONE": sym == NONE,
            "majority_rate": k / n,
            "none_rate": c.get(NONE, 0) / n,
            "distinct_answers": len(c),
        }
    return out


def embedding_table(model: Any) -> torch.Tensor:
    emb = model.byte_emb
    for name in ("weight", "table", "emb"):
        t = getattr(emb, name, None)
        if isinstance(t, torch.nn.Parameter | torch.Tensor):
            return t.detach().float()
        if isinstance(t, torch.nn.Embedding):
            return t.weight.detach().float()
    for mod in emb.modules():
        if isinstance(mod, torch.nn.Embedding):
            return mod.weight.detach().float()
    raise AttributeError("byte_emb holds no embedding table I recognise")


def row_norms(table: torch.Tensor) -> dict[str, Any]:
    norms = table.norm(dim=1).cpu()
    out = {}
    for name, (lo, hi) in GROUPS.items():
        seg = norms[lo:hi]
        out[name] = {
            "mean": float(seg.mean()),
            "min": float(seg.min()),
            "max": float(seg.max()),
            "rows": hi - lo,
        }
    return out


def subset(ds: ByteDataset, n_per_task: int) -> ByteDataset:
    """The last ``n_per_task`` documents of each task, in manifest order."""
    by_task: dict[str, list[int]] = {}
    for i in range(len(ds)):
        by_task.setdefault(str(ds.meta(i).get("task")), []).append(i)
    keep = sorted(j for idx in by_task.values() for j in idx[-n_per_task:])
    return ByteDataset([ds.examples[j] for j in keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--include-search", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated run ids")
    args = ap.parse_args()
    runs_dir = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    result: dict[str, Any] = {
        "runs": {},
        "floors": {},
        "embedding_norms": {},
        "errors": {},
        "held_out_train": {},
    }
    floors_done = False
    # Run directories: those with metrics.json, plus directories that hold a final checkpoint but no
    # metrics.json (a run stopped after training finished, before the runner wrote its files); the
    # latter take their configuration from a finished sibling of the same run prefix, with the seed
    # of the directory name, and are marked as such.
    candidates: list[tuple[Path, dict[str, Any] | None]] = []
    for mpath in sorted(runs_dir.rglob("metrics.json")):
        candidates.append((mpath.parent, json.load(open(mpath))))
    with_metrics = {d for d, _ in candidates}
    for ckpt in sorted(runs_dir.rglob("step_*.pt")):
        d = ckpt.parent
        if d in with_metrics or any(c[0] == d for c in candidates):
            continue
        candidates.append((d, None))
    for run_dir, metrics in candidates:
        rel = run_dir.relative_to(runs_dir).parts
        if any(p.startswith("_") or ".crashed-" in p for p in rel):
            continue
        if not args.include_search and ("search" in rel or "-trial" in run_dir.name):
            continue
        config_source = run_dir / "config.resolved.yaml"
        sibling_note = None
        if metrics is None:
            import re

            m = re.match(r"^(.*)-s(\d+)$", run_dir.name)
            siblings = [
                c
                for c in candidates
                if c[1] is not None
                and m
                and c[0].name.startswith(m.group(1) + "-s")
                and (c[0] / "config.resolved.yaml").is_file()
            ]
            if not m or not siblings or latest_checkpoint(run_dir) is None:
                continue
            config_source = siblings[0][0] / "config.resolved.yaml"
            sib_metrics = siblings[0][1]
            assert sib_metrics is not None
            metrics = {
                "run_id": run_dir.name,
                "model_name": sib_metrics["model_name"],
                "seed": int(m.group(2)),
                "steps": sib_metrics["steps"],
                "final": {"per_group": {}},
                "git_rev": None,
                "is_rerun": False,
            }
            sibling_note = (
                "no metrics.json (stopped after training); config from "
                f"{siblings[0][0].name} with seed={m.group(2)}"
            )
        run_id = str(metrics["run_id"])
        if only and run_id not in only:
            continue
        model_name = str(metrics["model_name"])
        t0 = time.perf_counter()
        try:
            overrides = [f"train.batch={args.eval_batch}", f"train.micro_batch={args.eval_batch}"]
            if sibling_note:
                overrides += [f"seed={metrics['seed']}", f"run_id={run_id}"]
            cfg = load_config(config_source, overrides=overrides)
            if (
                sibling_note
                and latest_checkpoint(run_dir) is not None
                and int(cfg.train.steps)
                != int(str(latest_checkpoint(run_dir)).rsplit("_", 1)[1].split(".")[0])
            ):
                print(f"{run_id}: skipped, last checkpoint is not the final step", flush=True)
                continue
            ctx = resolve(cfg.hardware)
            manifest = Path(str(cfg.data.manifest))
            tasks = list(cfg.data.tasks) or list(G1_TASKS)
            train_all = ByteDataset.load(
                manifest, "train", tasks, ["in_range"], expected_unroll=cfg.train.unroll
            )
            train_sub = subset(train_all, args.n_train)
            dev = ByteDataset.load(
                manifest, "dev", tasks, ["in_range"], expected_unroll=cfg.train.unroll
            )
            if not floors_done:
                dev_dist = answer_distribution(dev)
                train_dist = answer_distribution(train_sub)
                values = 48  # synthetic_v1.yaml: values [144, 192)
                result["floors"] = {
                    "delayed/in_range": {
                        "chance_1_over_value_alphabet": 1.0 / values,
                        "value_alphabet": values,
                        "majority_rate_dev": dev_dist.get("delayed", {}).get("majority_rate"),
                        "answers_dev": dev_dist.get("delayed"),
                    },
                    "kv/in_range": {
                        "majority_rate_dev": dev_dist.get("kv", {}).get("majority_rate"),
                        "majority_symbol_dev": dev_dist.get("kv", {}).get("majority_symbol"),
                        "none_rate_dev": dev_dist.get("kv", {}).get("none_rate"),
                        "answers_dev": dev_dist.get("kv"),
                        "answers_train_subset": train_dist.get("kv"),
                    },
                }
                result["held_out_train"] = {
                    "definition": (
                        f"the last {args.n_train} documents of each task's train split in "
                        "manifest order (a fixed evaluation subset; every train document was "
                        "available to training)"
                    ),
                    "doc_ids_first": [
                        train_sub.meta(0).get("doc_id"),
                        train_sub.meta(len(train_sub) - 1).get("doc_id"),
                    ],
                    "n_docs": len(train_sub),
                }
                floors_done = True
            seed_everything(cfg.seed)
            model = build_model(cfg)
            entry: dict[str, Any] = {
                "model": model_name,
                "seed": metrics["seed"],
                "steps": metrics["steps"],
                "is_rerun": bool(metrics.get("is_rerun")),
                "git_rev": metrics.get("git_rev"),
                "dev_final_from_run": {
                    g: e["accuracy"] for g, e in metrics["final"]["per_group"].items()
                },
                "note": sibling_note,
            }
            if model_name == "hsn":
                init_norms = row_norms(embedding_table(model))
            checkpoint = latest_checkpoint(run_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"{run_dir} has no checkpoint")
            loaded = load_checkpoint(checkpoint, model)
            entry["checkpoint"] = checkpoint.name
            entry["checkpoint_step"] = int(loaded["step"])
            trainer = make_trainer(model, cfg, ctx, run_dir=out / "_scratch" / run_dir.name)
            r_train = trainer.evaluate(train_sub)
            r_dev = trainer.evaluate(dev)
            entry["train_subset"] = {
                g: {
                    "accuracy": float(e["accuracy"]),
                    "loss": float(e["loss"]),
                    "n_positions": int(e["n_positions"]),
                }
                for g, e in r_train["per_group"].items()
            }
            entry["dev"] = {
                g: {
                    "accuracy": float(e["accuracy"]),
                    "loss": float(e["loss"]),
                    "n_positions": int(e["n_positions"]),
                }
                for g, e in r_dev["per_group"].items()
            }
            if model_name == "hsn":
                end_norms = row_norms(embedding_table(model))
                result["embedding_norms"][run_id] = {
                    "init": init_norms,
                    "final": end_norms,
                    "ratio_final_over_init_mean": {
                        k: end_norms[k]["mean"] / init_norms[k]["mean"]
                        if init_norms[k]["mean"]
                        else None
                        for k in GROUPS
                    },
                }
            entry["seconds"] = round(time.perf_counter() - t0, 1)
            result["runs"][run_id] = entry
            train_s = ", ".join(
                f"{g}={v['accuracy']:.3f}" for g, v in entry["train_subset"].items()
            )
            dev_s = ", ".join(f"{g}={v['accuracy']:.3f}" for g, v in entry["dev"].items())
            print(
                f"{run_id}: train {{{train_s}}} dev {{{dev_s}}} ({entry['seconds']}s)", flush=True
            )
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - every run is reported; a failure is a fact
            result["errors"][run_id] = f"{type(exc).__name__}: {exc}"
            print(f"{run_id}: ERROR {type(exc).__name__}: {exc}", flush=True)
    with open(out / "diagnosis_eval.json", "w") as fh:
        json.dump(result, fh, indent=1)
    n_runs, n_err = len(result["runs"]), len(result["errors"])
    print(f"wrote {out / 'diagnosis_eval.json'}: {n_runs} runs, {n_err} errors")


if __name__ == "__main__":
    main()
