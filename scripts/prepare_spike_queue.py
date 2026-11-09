"""Prepare an ordered diagnostic sweep from existing SQLite experiments, without enqueueing.

Run with the repository environment. This script imports no torch, loads no checkpoint tensors,
and never changes the database. Enqueue the resulting spec only during the explicitly owned turn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(
    db: Path, parent: int, comparisons: list[int], target_doc: int | None, out: Path
) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    arms: list[dict[str, Any]] = []
    previous: str | None = None
    source: dict[int, dict[str, Any]] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for exp_id in dict.fromkeys([parent, *comparisons]):
            row = conn.execute(
                "SELECT e.*, a.variant FROM experiments e JOIN arms a USING(arm_id) "
                "WHERE exp_id = ?",
                (exp_id,),
            ).fetchone()
            if row is None or row["status"] not in ("done", "failed", "stopped_early", "cancelled"):
                raise ValueError(f"experiment {exp_id} is not a finished source")
            if row["variant"] != "pdg" or not row["config_json"]:
                raise ValueError(f"experiment {exp_id} lacks supported configuration provenance")
            directory = Path(row["run_dir"])
            dumps = sorted(directory.glob("divergence_*.pt"))
            if not dumps:
                raise ValueError(f"experiment {exp_id} has no saved spike")
            cfg = json.loads(row["config_json"])
            config_file = out.with_name(f"{out.stem}.parent-{exp_id}.yaml")
            config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
            source[exp_id] = {
                "config": str(config_file.resolve()),
                "seed": cfg["seed"],
                "compile": cfg["hardware"]["compile_mode"],
                "dump": str(dumps[0]),
                "dump_sha256": checksum(dumps[0]),
            }
        jobs = [(parent, stage) for stage in ("reproduce", "isolate", "trace", "update")]
        jobs.extend((e, "compare") for e in comparisons)
        for index, (exp_id, stage) in enumerate(jobs, 1):
            src = source[exp_id]
            name = f"{index:02d}_{stage}_exp{exp_id}"
            modes = ["eager", "compiled"] if stage in ("reproduce", "update") else ["eager"]
            protocol: dict[str, Any] = {
                "parent_exp": exp_id,
                "dump": src["dump"],
                "dump_sha256": src["dump_sha256"],
                "stage": stage,
                "modes": modes,
                "deterministic": stage not in ("reproduce", "update"),
                "timeout_s": 600 if "compiled" in modes else 240,
                "requires": [previous] if previous else [],
            }
            if stage == "isolate":
                protocol["target_doc"] = target_doc
            if stage == "trace":
                spike_step = int(Path(src["dump"]).stem.split("_")[-1])
                retained = sorted(
                    p
                    for p in Path(src["dump"]).parent.glob("step_*.pt")
                    if int(p.stem.split("_")[-1]) < spike_step
                )
                if retained:
                    protocol.update(
                        reference_checkpoint=str(retained[-1]),
                        reference_sha256=checksum(retained[-1]),
                    )
            arms.append(
                {
                    "name": name,
                    "family": "hsn",
                    "config": src["config"],
                    "overrides": [f"hardware.compile_mode={src['compile']}"],
                    "seeds": [src["seed"]],
                    "protocol": protocol,
                }
            )
            previous = f"{name}/s{src['seed']}"
        spec = {
            "name": f"a5_spike_investigation_exp{parent}",
            "description": "Ordered backward diagnostics before training; pause on failure.",
            "base": {"config": source[parent]["config"]},
            "timing": True,
            "protocol": {"name": "spike_diagnostic"},
            "arms": arms,
        }
        out.write_text(yaml.safe_dump(spec, sort_keys=False))
        return spec
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--compare", type=int, action="append", default=[])
    parser.add_argument("--target-doc", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = prepare(args.db, args.parent, args.compare, args.target_doc, args.out)
    print(
        json.dumps(
            {"spec": str(args.out), "jobs": [a["name"] for a in spec["arms"]], "enqueued": False},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
