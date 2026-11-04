#!/usr/bin/env python
"""Assemble a gate evidence package (plan P0.8, Track E); the same as ``hsn package``.

    python scripts/evidence_package.py --gate G0 --runs runs/g0_cpu/* --out evidence/G0.zip

Every run directory must carry the files ``hsn run`` writes and at least one must be a rerun
(``rerun.json``). The zip holds ``<gate>/SUMMARY.md`` listing every required artefact and where it
is inside the zip; see :func:`hsn.eval.results.evidence_package`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hsn.eval.results import FREEZE_PATH, evidence_package


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gate", required=True, help="G0 ... G6")
    parser.add_argument("--runs", nargs="+", required=True, help="run directories (incl. reruns)")
    parser.add_argument("--out", required=True, help="evidence/<gate>.zip (never overwritten)")
    parser.add_argument("--freeze", default=str(FREEZE_PATH), help="metrics freeze file")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="extra file at the package root, e.g. decision.json=evidence/G0/decision.json",
    )
    args = parser.parse_args(argv)
    extras: dict[str, Path] = {}
    for item in args.extra:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--extra expects NAME=PATH, got {item!r}")
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


if __name__ == "__main__":
    sys.exit(main())
