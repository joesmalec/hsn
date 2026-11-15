"""Score the candidates of one frozen plan on the untouched final test set, once.

``configs/data/synthetic_final.yaml`` holds fresh documents of the hold-out tasks with a key
alphabet that no run has been trained or scored on. A number on it is worth what the set's
isolation is worth, so this module is the one way to score it, and the set can be used for one
campaign only (the review amendments of 14 Nov 2026, "lock the final evaluation campaign"):

1. **Freeze.** Write down, before anything is scored, the claims with their pass conditions,
   the candidates (run folder, checkpoint, the reader's reach) and the analysis.
   ``freeze`` checks every candidate (the checkpoint exists; the run's training stream never
   contained a byte of the final alphabet, in any role), records the checkpoint hashes and
   writes the frozen plan with its own hash. Commit the frozen plan.
2. **Run.** ``run`` scores every candidate of the frozen plan in one go and appends one record
   to ``docs/final_test_ledger.jsonl``. It refuses a plan that is not committed, a checkpoint
   that is not the frozen one, and any plan at all once the set has a campaign on the ledger:
   the set is then spent. A redesigned claim needs a new final set (a new dataset spec with
   another alphabet and other seeds), not a second look at this one.

Never select a recipe, a checkpoint or a seed on these documents: choose on the dev data.

What is scored (fresh documents; ``kv`` has one-byte keys, ``delayed2h`` two-byte keys):

==========  ==============  ===============================================================
task        condition       what the keys are, relative to training
==========  ==============  ===============================================================
kv          training keys   key bytes seen in training (fresh documents)
kv          final keys      key bytes absent from every training input
delayed2h   seen pairs      pairs of training bytes that training may contain
delayed2h   new pairs       unseen combinations of bytes seen in training
delayed2h   final bytes     key bytes absent from every training input
==========  ==============  ===============================================================

Each condition is reported in total and for the answers **beyond the reach**: the queries whose
fact lies more than ``reach - 1`` positions back, ``reach`` being the complete receptive field of
the candidate's reader in bytes (``1 + L (n - 1)`` for ``L`` blocks of span ``n``; 0 for a model
without a reader, for which every answer is beyond reach). "Beyond the reach" says where the
designated source stands; it does not by itself show that the answer came from the memory. The
95 % interval is a bootstrap over documents.

Each document is run as one window from a cleared state, on the CPU, with the checkpoint's
averaged weights where the run kept an average (the weights its own evaluations scored).

    python -m hsn.eval.final_test freeze DRAFT.json --out docs/final_test_plans/NAME.json
    python -m hsn.eval.final_test run docs/final_test_plans/NAME.json
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

from hsn.config import RunConfig, Split, config_hash, load_config
from hsn.data.synthetic import DatasetSpec, TaskSpec, generate, load_dataset_spec
from hsn.device import seed_everything
from hsn.model.hsn import HSN
from hsn.runner import build_model
from hsn.train.checkpoint import load_checkpoint

__all__ = ["CONDITIONS", "LEDGER", "SPEC", "exposure", "main", "score_condition"]

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "configs" / "data" / "synthetic_final.yaml"
LEDGER = REPO / "docs" / "final_test_ledger.jsonl"
CPU_PROFILE = REPO / "configs" / "hardware" / "cpu.yaml"

#: task of the run's training data -> condition -> (task of the final set, split, population)
CONDITIONS: dict[str, dict[str, tuple[str, Split, str]]] = {
    "kv": {
        "training keys": ("kv", "train", "key bytes seen in training"),
        "final keys": ("kv", "test", "key bytes absent from every training input"),
    },
    "delayed2h": {
        "seen pairs": ("delayed2h", "train", "pairs of training bytes that training may contain"),
        "new pairs": ("delayed2h_new", "train", "unseen combinations of bytes seen in training"),
        "final bytes": ("delayed2h", "test", "key bytes absent from every training input"),
    },
}
_BOOTSTRAP = 1000
I64 = npt.NDArray[np.int64]


def _task(dataset: DatasetSpec, name: str) -> TaskSpec:
    return next(task for task in dataset.tasks if task.name == name)


def _final_bytes(dataset: DatasetSpec) -> set[int]:
    """The final alphabet: the test keys of the final set's tasks."""
    final: set[int] = set()
    for task in dataset.tasks:
        lo, hi = task.params["keys"]["test"]
        final.update(range(lo, hi))
    return final


def _training_bytes(task: TaskSpec) -> dict[str, set[int]]:
    """The bytes a task's training documents may hold, by role: every alphabet and marker of
    its parameters, with the training split's keys only (dev and test keys never enter a
    training document)."""
    roles: dict[str, set[int]] = {}
    for name, value in task.params.items():
        if name == "keys" and isinstance(value, dict):
            lo, hi = value["train"]
            roles["keys"] = set(range(lo, hi))
        elif isinstance(value, dict):  # markers: role -> byte
            roles[name] = {int(byte) for byte in value.values() if isinstance(byte, int)}
        elif (
            isinstance(value, list | tuple)
            and len(value) == 2
            and all(isinstance(x, int) for x in value)
        ):
            roles[name] = set(range(value[0], value[1]))
    return roles


def exposure(cfg: RunConfig, final: set[int]) -> list[str]:
    """Why a run may have seen a byte of the final alphabet in training, in any role (empty:
    it has not). A new alphabet that was never scored is not thereby absent from training: text
    holds bytes 32 to 47 everywhere, and other synthetic tasks use them as symbols."""
    problems: list[str] = []
    if cfg.train.init_from is not None:
        problems.append(
            f"the run starts from {cfg.train.init_from}: the training stream of that "
            "checkpoint is not known to this check"
        )
    path = REPO / "configs" / "data" / f"{cfg.data.dataset}.yaml"
    try:
        dataset = load_dataset_spec(path)
    except (OSError, ValueError, KeyError, TypeError):
        return [
            *problems,
            f"the run trained on {cfg.data.dataset!r}, which is not a synthetic task set: its "
            "bytes are not known by role (text holds the final alphabet's bytes everywhere)",
        ]
    for task in dataset.tasks:
        # an empty list in the run's configuration means every task / every variant
        if cfg.data.tasks and task.name not in cfg.data.tasks:
            continue
        if cfg.data.variants and task.variant not in cfg.data.variants:
            continue
        for role, used in _training_bytes(task).items():
            seen = sorted(used & final)
            if seen:
                problems.append(
                    f"{cfg.data.dataset}/{task.key}: training documents hold final bytes "
                    f"{seen[0]} to {seen[-1]} as {role}"
                )
    return problems


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


def _canonical(plan: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in plan.items() if k != "plan_sha256"}, sort_keys=True)


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()


def _spent(ledger: Path, spec_hash: str) -> dict[str, Any] | None:
    """The campaign already on the ledger for this final set, if there is one."""
    for entry in _ledger_entries(ledger):
        if entry.get("dataset_spec_hash") == spec_hash:
            return entry
    return None


def _committed(path: Path) -> bool:
    """Whether ``path`` is a file of this repository, committed and unchanged."""
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=REPO,
            capture_output=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not status.stdout.strip()


def _run_config(run_dir: Path) -> RunConfig:
    return load_config(run_dir / "config.resolved.yaml", [f"hardware={CPU_PROFILE}", "profile=cpu"])


def _freeze(args: argparse.Namespace) -> int:
    draft = json.loads(Path(args.draft).read_text("utf-8"))
    dataset = load_dataset_spec(SPEC)
    spent = _spent(args.ledger, dataset.spec_hash())
    if spent is not None:
        print(
            f"refused: the final set {dataset.name} was spent by the campaign "
            f'"{spent["plan"]}" on {spent["time"]}. A new claim needs a new final set.',
            file=sys.stderr,
        )
        return 2
    problems: list[str] = []
    for key in ("name", "analysis"):
        if not isinstance(draft.get(key), str) or not draft[key].strip():
            problems.append(f"the plan needs a non-empty {key!r}")
    claims = draft.get("claims")
    if not isinstance(claims, list) or not claims:
        problems.append("the plan needs at least one claim")
    else:
        for claim in claims:
            if not isinstance(claim, dict) or not all(claim.get(k) for k in ("id", "text", "pass")):
                problems.append(f"every claim needs id, text and pass (its condition): {claim}")
    candidates = draft.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        problems.append("the plan needs at least one candidate")
        candidates = []
    final = _final_bytes(dataset)
    frozen: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = Path(str(candidate.get("run_dir", "")))
        checkpoint = run_dir / str(candidate.get("checkpoint", ""))
        reach = candidate.get("reach")
        label = f"{run_dir.name}/{checkpoint.name}"
        if not checkpoint.is_file() or not (run_dir / "config.resolved.yaml").is_file():
            problems.append(f"{label}: the checkpoint or config.resolved.yaml does not exist")
            continue
        if isinstance(reach, bool) or not isinstance(reach, int) or reach < 0:
            problems.append(f"{label}: reach must be 0 or the reader's receptive field in bytes")
            continue
        cfg = _run_config(run_dir)
        if not [task for task in cfg.data.tasks if task in CONDITIONS]:
            problems.append(
                f"{label}: trained on {list(cfg.data.tasks)}; the set scores {list(CONDITIONS)}"
            )
        problems += [f"{label}: {problem}" for problem in exposure(cfg, final)]
        frozen.append(
            {
                "run": run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": _sha256(checkpoint),
                "config_hash": config_hash(cfg),
                "model_name": cfg.model_name,
                "reach": reach,
            }
        )
    if problems:
        print("refused, the plan is not frozen:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    plan: dict[str, Any] = {
        "name": draft["name"],
        "claims": claims,
        "analysis": draft["analysis"],
        "candidates": frozen,
        "dataset": dataset.name,
        "dataset_spec_hash": dataset.spec_hash(),
        "final_alphabet": [min(final), max(final) + 1],
        "populations": "every condition in total and beyond the reach (source more than "
        "reach - 1 positions back); 95 % bootstrap over documents, 1,000 resamples, seed 0",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "frozen_by_commit": _commit(),
    }
    plan["plan_sha256"] = _plan_hash(plan)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n", "utf-8")
    print(f"frozen: {out} ({len(frozen)} candidates, {len(claims)} claims). Commit it, then run.")
    return 0


def _score_candidate(candidate: dict[str, Any], dataset: DatasetSpec, n_docs: int | None) -> Any:
    run_dir = Path(candidate["run_dir"])
    checkpoint = run_dir / candidate["checkpoint"]
    cfg = _run_config(run_dir)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    load_checkpoint(checkpoint, model, averaged=True)
    model.eval()
    rng = np.random.default_rng(0)
    results: dict[str, dict[str, Any]] = {}
    for name in [task for task in cfg.data.tasks if task in CONDITIONS]:
        results[name] = {}
        for condition, (task_name, split, population) in CONDITIONS[name].items():
            task = _task(dataset, task_name)
            docs = min(n_docs or task.n_examples(split), task.n_examples(split))
            counts = score_condition(model, task, split, docs, candidate["reach"])
            results[name][condition] = {
                "population": population,
                "docs": docs,
                "total": _summary(counts[:, 0], counts[:, 1], rng),
                "beyond_reach": _summary(counts[:, 2], counts[:, 3], rng),
            }
    return results


def _run(args: argparse.Namespace) -> int:
    path = Path(args.plan).resolve()
    plan = json.loads(path.read_text("utf-8"))
    dataset = load_dataset_spec(SPEC)
    refusals: list[str] = []
    if plan.get("plan_sha256") != _plan_hash(plan):
        refusals.append("the plan was changed after it was frozen (its hash does not match)")
    if plan.get("dataset_spec_hash") != dataset.spec_hash():
        refusals.append("the plan was frozen for another version of the final set")
    spent = _spent(args.ledger, dataset.spec_hash())
    if spent is not None:
        refusals.append(
            f'the final set {dataset.name} was spent by the campaign "{spent["plan"]}" on '
            f"{spent['time']}: one campaign per final set, a new claim needs a new set"
        )
    committed = _committed(path)
    if not committed and not args.allow_uncommitted:
        refusals.append(
            "the frozen plan is not committed in this repository (commit it before scoring, so "
            "that the plan is on record before any number exists)"
        )
    for candidate in plan.get("candidates", []):
        checkpoint = Path(candidate["run_dir"]) / candidate["checkpoint"]
        if not checkpoint.is_file() or _sha256(checkpoint) != candidate["checkpoint_sha256"]:
            refusals.append(f"{candidate['run']}/{candidate['checkpoint']} is not the frozen file")
    if refusals:
        print("refused, nothing was scored:", file=sys.stderr)
        for refusal in refusals:
            print(f"  - {refusal}", file=sys.stderr)
        return 2

    started = time.time()
    results = {
        f"{c['run']}/{c['checkpoint']}": _score_candidate(c, dataset, args.docs)
        for c in plan["candidates"]
    }
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan": plan["name"],
        "plan_sha256": plan["plan_sha256"],
        "plan_committed": committed,
        "claims": plan["claims"],
        "candidates": plan["candidates"],
        "dataset": dataset.name,
        "dataset_spec_hash": dataset.spec_hash(),
        "docs_per_condition": args.docs,
        "scoring_commit": _commit(),
        "seconds": round(time.time() - started, 1),
        "results": results,
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    for candidate in plan["candidates"]:
        key = f"{candidate['run']}/{candidate['checkpoint']}"
        print(f"\n{key}, reach {candidate['reach']} bytes")
        print("| task | condition | total | beyond the reach |")
        print("| --- | --- | --- | --- |")
        for name, conditions in results[key].items():
            for condition, entry in conditions.items():
                print(
                    f"| {name} | {condition} | {_cell(entry['total'])} | "
                    f"{_cell(entry['beyond_reach'])} |"
                )
    print(f"\nledger: {args.ledger}. The final set {dataset.name} is spent.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hsn.eval.final_test", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="check a plan and write its frozen form")
    freeze.add_argument("draft", type=Path, help="JSON: name, claims, candidates, analysis")
    freeze.add_argument("--out", type=Path, required=True, help="the frozen plan to write")
    freeze.add_argument("--ledger", type=Path, default=LEDGER)
    run = sub.add_parser("run", help="score every candidate of a frozen, committed plan")
    run.add_argument("plan", type=Path, help="a frozen plan")
    run.add_argument("--ledger", type=Path, default=LEDGER)
    run.add_argument("--docs", type=int, default=None, help="documents per condition (all)")
    run.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="score a plan that is not committed; the ledger records that it was not",
    )
    args = parser.parse_args(argv)
    return _freeze(args) if args.command == "freeze" else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
