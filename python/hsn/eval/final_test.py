"""Score one checkpoint on the untouched final test set, once per gate claim.

``configs/data/synthetic_final.yaml`` holds fresh documents of the hold-out tasks with a key
alphabet that no run has been trained or scored on. A number on it is only worth what the set's
isolation is worth, so this module is the one way to score it and it keeps a ledger:

* every scoring appends one line to ``docs/final_test_ledger.jsonl`` (the claim it was made for,
  the run, the checkpoint's hash, the commit of the scoring code, every number);
* a checkpoint that is already in the ledger is refused, unless ``--again "<reason>"`` says why
  it is scored a second time, and the reason is written down with it.

Never select a recipe, a checkpoint or a seed on these documents: choose on the dev data, then
score the chosen checkpoint here.

What is scored (all on fresh documents; ``kv`` has one-byte keys, ``delayed2h`` two-byte keys):

==========  ==============  =========================================================
task        condition       keys
==========  ==============  =========================================================
kv          training keys   the training key bytes
kv          final keys      the final alphabet (bytes 32 to 47)
delayed2h   seen pairs      pairs of training bytes that training may contain
delayed2h   new pairs       pairs of training bytes that training never contains
delayed2h   final bytes     the final alphabet
==========  ==============  =========================================================

Each condition is reported in total and for the answers **beyond the reach**: the queries
whose fact lies more than ``reach - 1`` positions back, ``reach`` being the complete receptive
field of the model's reader in bytes (``1 + L (n - 1)`` for ``L`` blocks of span ``n``; 0 for a
model without a reader, for which every answer is beyond reach). A reader can answer a query
inside its reach from the window alone, so only the answers beyond it show that a memory was
used. The 95 % interval is a bootstrap over documents (answers of one document are not
independent).

Each document is run as one window from a cleared state, on the CPU, with the checkpoint's
averaged weights where the run kept an average (the weights its own evaluations scored).

    python -m hsn.eval.final_test RUN_DIR --claim "gate K, rung 1, seed 0" --reach 63
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from hsn.config import Split, config_hash, load_config
from hsn.data.synthetic import DatasetSpec, TaskSpec, generate, load_dataset_spec
from hsn.device import seed_everything
from hsn.model.hsn import HSN
from hsn.runner import build_model
from hsn.train.checkpoint import load_checkpoint

__all__ = ["CONDITIONS", "LEDGER", "SPEC", "main", "score_condition"]

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "configs" / "data" / "synthetic_final.yaml"
LEDGER = REPO / "docs" / "final_test_ledger.jsonl"
CPU_PROFILE = REPO / "configs" / "hardware" / "cpu.yaml"

#: task of the run's training data -> condition -> (task of the final set, split)
CONDITIONS: dict[str, dict[str, tuple[str, Split]]] = {
    "kv": {"training keys": ("kv", "train"), "final keys": ("kv", "test")},
    "delayed2h": {
        "seen pairs": ("delayed2h", "train"),
        "new pairs": ("delayed2h_new", "train"),
        "final bytes": ("delayed2h", "test"),
    },
}
_BOOTSTRAP = 1000
I64 = npt.NDArray[np.int64]


def _task(dataset: DatasetSpec, name: str) -> TaskSpec:
    return next(task for task in dataset.tasks if task.name == name)


def score_condition(
    model: torch.nn.Module, task: TaskSpec, split: Split, n_docs: int, reach: int, batch: int = 64
) -> I64:
    """Per document ``[right, answers, right beyond the reach, answers beyond the reach]`` for
    the first ``n_docs`` documents of ``task``'s ``split`` (``[n_docs, 4]``)."""
    docs = [generate(task, split, i) for i in range(n_docs)]
    order = sorted(range(n_docs), key=lambda i: docs[i].length)
    counts = np.zeros((n_docs, 4), dtype=np.int64)
    with torch.no_grad():
        for lo in range(0, n_docs, batch):
            idx = order[lo : lo + batch]
            width, rows = max(docs[i].length for i in idx), len(idx)
            tokens = torch.zeros(rows, width, dtype=torch.int64)
            ctx = torch.zeros(rows, width, dtype=torch.int64)
            for row, i in enumerate(idx):
                tokens[row, : docs[i].length] = torch.from_numpy(docs[i].tokens.astype(np.int64))
                ctx[row] = int(docs[i].context[0])
            reset = torch.zeros(rows, width, dtype=torch.bool)
            reset[:, 0] = True
            state = (
                model.init_state(rows)
                if isinstance(model, HSN)
                else model.init_state(rows, tokens.device)  # type: ignore[operator]
            )
            predicted = model(tokens, ctx, state, reset)[0].argmax(dim=-1).numpy()
            for row, i in enumerate(idx):
                mask = docs[i].loss_mask
                hit = (predicted[row, : len(mask)] == docs[i].targets)[mask]
                far = np.asarray(docs[i].meta["deps"]) > reach - 1
                counts[i] = (hit.sum(), hit.size, hit[far].sum(), far.sum())
    return counts


def _summary(right: I64, answers: I64, rng: np.random.Generator) -> dict[str, Any]:
    """Accuracy, answer count and a 95 % bootstrap interval over documents."""
    total = int(answers.sum())
    if total == 0:
        return {"accuracy": None, "n": 0, "ci95": None}
    draws = rng.integers(0, len(right), size=(_BOOTSTRAP, len(right)))
    resampled = right[draws].sum(axis=1) / np.maximum(answers[draws].sum(axis=1), 1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "accuracy": float(right.sum() / total),
        "n": total,
        "ci95": [float(low), float(high)],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _ledger_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _cell(entry: dict[str, Any]) -> str:
    if entry["accuracy"] is None:
        return "-"
    low, high = entry["ci95"]
    return f"{100 * entry['accuracy']:.1f} % ({100 * low:.1f} to {100 * high:.1f}; n {entry['n']})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hsn.eval.final_test", description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a run folder: config.resolved.yaml, step_*.pt")
    parser.add_argument("--claim", required=True, help="the gate claim this scoring is made for")
    parser.add_argument("--checkpoint", default=None, help="file name; default: the last step")
    parser.add_argument("--reach", type=int, required=True, help="the reader's reach in bytes; 0")
    parser.add_argument("--again", default=None, help="why a checkpoint in the ledger is rescored")
    parser.add_argument("--docs", type=int, default=None, help="documents per condition (all)")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--override", action="append", default=[], help="config override k=v")
    args = parser.parse_args(argv)
    if args.reach < 0:
        parser.error("--reach must be 0 (no reader) or the reader's receptive field in bytes")

    run_dir: Path = args.run_dir
    checkpoints = sorted(run_dir.glob("step_*.pt"))
    if not checkpoints:
        parser.error(f"{run_dir} holds no step_*.pt checkpoint")
    checkpoint = run_dir / args.checkpoint if args.checkpoint else checkpoints[-1]
    if not checkpoint.is_file():
        parser.error(f"{checkpoint} does not exist")
    digest = _sha256(checkpoint)
    earlier = [e for e in _ledger_entries(args.ledger) if e["checkpoint_sha256"] == digest]
    if earlier and not args.again:
        print(
            f"refused: {checkpoint.name} of {run_dir.name} is in the ledger already "
            f'({earlier[-1]["time"]}, claim "{earlier[-1]["claim"]}"). The final set is scored '
            'once per gate claim; pass --again "<reason>" to score it again on the record.',
            file=sys.stderr,
        )
        return 2

    cfg = load_config(
        run_dir / "config.resolved.yaml",
        [f"hardware={CPU_PROFILE}", "profile=cpu", *args.override],
    )
    tasks = [task for task in cfg.data.tasks if task in CONDITIONS]
    if not tasks:
        parser.error(
            f"the run trained on {list(cfg.data.tasks)}; the final set scores {list(CONDITIONS)}"
        )
    dataset = load_dataset_spec(SPEC)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    load_checkpoint(checkpoint, model, averaged=True)
    model.eval()

    started = time.time()
    rng = np.random.default_rng(0)
    results: dict[str, dict[str, Any]] = {}
    for name in tasks:
        results[name] = {}
        for condition, (task_name, split) in CONDITIONS[name].items():
            task = _task(dataset, task_name)
            n_docs = min(args.docs or task.n_examples(split), task.n_examples(split))
            counts = score_condition(model, task, split, n_docs, args.reach)
            results[name][condition] = {
                "docs": n_docs,
                "total": _summary(counts[:, 0], counts[:, 1], rng),
                "beyond_reach": _summary(counts[:, 2], counts[:, 3], rng),
            }
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "claim": args.claim,
        "again": args.again,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": digest,
        "config_hash": config_hash(cfg),
        "model_name": cfg.model_name,
        "overrides": list(args.override),
        "reach": args.reach,
        "scoring_commit": _commit(),
        "dataset_spec_hash": dataset.spec_hash(),
        "seconds": round(time.time() - started, 1),
        "results": results,
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f'{run_dir.name} {checkpoint.name}, reach {args.reach} bytes, claim "{args.claim}"')
    print("| task | condition | total | beyond the reach |")
    print("| --- | --- | --- | --- |")
    for name, conditions in results.items():
        for condition, entry in conditions.items():
            print(
                f"| {name} | {condition} | {_cell(entry['total'])} | "
                f"{_cell(entry['beyond_reach'])} |"
            )
    print(f"ledger: {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
