#!/usr/bin/env python
"""Build ``textfacts``: the delayed task with story text in place of its random filler.

Why (amendment A8, ``docs/amendments/A8_fast_cells.md``): the delayed task asks the model to hold
facts across filler it can ignore, TinyStories asks it to follow text with nothing to hold. This
dataset asks for both in one document: facts ``k v`` are stated, real story text follows, and the
facts are queried after it. A document is a ``delayed`` document of synthetic_v1's parameters
(same generator, same alphabets, same per-split key ranges, another seed) whose filler bytes are
replaced, position by position, by a cut of a TinyStories story:

    BOS story-text (k v)^n story-text SEP QRY k_1 v_1 (QRY k v)* EOS

so lengths, dependency distances, the number of facts and queries, and with them the key-blind
ceilings, are those of synthetic_v1's delayed task. Text is plain ASCII (bytes 32..126 and the
newline marker 16), so a value byte (144..191) still occurs only as a value. Key bytes are
letters: unlike in synthetic_v1, the dev keys' bytes do occur in training, as text, never as keys.

Tasks of the manifest (``data.tasks`` selects one; the documents are the same):

* ``tf``   train: the answers and one text position in ``--text-every`` are supervised (the
           model has to follow the text and hold the facts); dev / test: the answers only, so
           dev accuracy is answer accuracy, comparable with the delayed task's.
* ``tfq``  train, dev, test: the answers only (the text is an unsupervised distractor).

``--oor DIR`` also writes an evaluation-only manifest of out-of-range documents (the facts lie
more than a window before their queries; dev and test only, answers supervised).

Text for the train documents comes from the stories of the TinyStories train shard after the
first ``--skip-stories`` (those are left for runs that mix plain stories in), text for dev and
test from the dev and test stories. Everything is a pure function of the arguments.

    python scripts/build_textfacts.py --stories data/tinystories_v1 --out <dir>/textfacts_v1 \\
        --oor <dir>/textfacts_v1_oor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import hsn
import numpy as np
from hsn.data.splits import (
    MANIFEST_NAME,
    ShardRecord,
    SplitManifest,
    _split_hashes,
    read_shard,
    write_shard,
)
from hsn.data.synthetic import (
    SPLITS,
    Example,
    TaskSpec,
    generate,
    load_dataset_spec,
)

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "configs" / "data" / "synthetic_v1.yaml"
NEWLINE = 16  # tinystories_v1's newline marker
FILLER = (192, 256)  # synthetic_v1's delayed filler alphabet


class Stories:
    """The story bodies of a TinyStories shard (``BOS body EOS`` documents), as byte arrays."""

    def __init__(self, path: Path, first: int = 0) -> None:
        arrays, _ = read_shard(path)
        self.tokens = arrays["tokens"]
        offsets = arrays["offsets"]
        self.starts = offsets[first:-1] + 1
        self.ends = offsets[first + 1 :] - 1
        self.lengths = self.ends - self.starts
        if len(self.starts) == 0:
            raise ValueError(f"{path}: no stories after the first {first}")
        ok = np.zeros(256, dtype=np.bool_)
        ok[32:127] = True
        ok[NEWLINE] = True
        self.plain = ok

    def cut(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` consecutive plain bytes from one story, starting after a space when it can."""
        for _ in range(200):
            s = int(rng.integers(len(self.starts)))
            if self.lengths[s] < n + 1:
                continue
            lo = int(self.starts[s] + rng.integers(self.lengths[s] - n))
            # move to the byte after the next space, when the cut still fits in the story
            window = self.tokens[lo : min(lo + 24, int(self.ends[s]) - n)]
            spaces = np.flatnonzero(window == 32)
            if spaces.size:
                lo += int(spaces[0]) + 1
            piece = self.tokens[lo : lo + n]
            if piece.shape[0] == n and bool(self.plain[piece].all()):
                return piece
        raise RuntimeError(f"no plain story cut of {n} bytes found")


def with_text(
    example: Example, stories: Stories, seed: int, text_every: int, *, spec: TaskSpec
) -> Example:
    """``example`` (a delayed document) with its filler runs replaced by story text. ``text_every``
    > 0 also supervises one text position in that many (positions whose target is the next byte
    of the same run), drawn per document; 0 keeps the answers only."""
    tokens = example.tokens.copy()
    filler = (tokens >= FILLER[0]) & (tokens < FILLER[1])
    rng = np.random.default_rng(
        [seed, int(example.meta["index"]), SPLITS.index(example.meta["split"])]
    )
    edges = np.flatnonzero(np.diff(np.concatenate([[False], filler, [False]]).astype(np.int8)))
    for lo, hi in zip(edges[0::2], edges[1::2], strict=True):
        tokens[lo:hi] = stories.cut(rng, int(hi - lo))
    loss_mask = example.loss_mask.copy()
    answers = np.flatnonzero(loss_mask)
    dep_of = dict(zip(answers.tolist(), example.meta["deps"], strict=True))
    if text_every > 0:
        inside = filler[:-1] & filler[1:]  # position t and its target t + 1 lie in one run
        candidates = np.flatnonzero(inside)
        chosen = candidates[rng.random(candidates.shape[0]) < 1.0 / text_every]
        loss_mask[chosen] = True
    deps = tuple(dep_of.get(int(t), 1) for t in np.flatnonzero(loss_mask))
    targets = np.concatenate([tokens[1:], np.zeros(1, dtype=np.uint8)])
    meta: dict[str, Any] = {
        **example.meta,
        "task": spec.name,
        "doc_id": f"{spec.name}:{spec.variant}:{example.meta['split']}:{example.meta['index']}",
        "deps": deps,
    }
    return Example(tokens, targets, loss_mask, example.context.copy(), meta)


def task_spec(
    base: TaskSpec, name: str, seed: int, n: tuple[int, int, int], context_id: int
) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            **base.model_dump(),
            "name": name,
            "seed": seed,
            "n_train": n[0],
            "n_dev": n[1],
            "n_test": n[2],
            "context_id": context_id,
        }
    )


def build(args: argparse.Namespace) -> int:
    v1 = load_dataset_spec(V1)
    stories_dir = Path(args.stories)
    text = {
        "train": Stories(stories_dir / "tinystories.out_of_range.train.npz", args.skip_stories),
        "dev": Stories(stories_dir / "tinystories.out_of_range.dev.npz"),
        "test": Stories(stories_dir / "tinystories.out_of_range.test.npz"),
    }
    sizes = (args.n_train, args.n_dev, args.n_test)
    jobs: list[tuple[Path, str, list[tuple[TaskSpec, dict[str, int]]]]] = []
    in_range = v1.task("delayed", "in_range")
    # the two tasks share a generator seed, so their documents and their story cuts are the same
    jobs.append(
        (
            Path(args.out),
            args.name,
            [
                (
                    task_spec(in_range, "tf", args.seed, sizes, args.context_id),
                    {"train": args.text_every, "dev": 0, "test": 0},
                ),
                (
                    task_spec(in_range, "tfq", args.seed, sizes, args.context_id),
                    {"train": 0, "dev": 0, "test": 0},
                ),
            ],
        )
    )
    if args.oor:
        oor = v1.task("delayed", "out_of_range")
        jobs.append(
            (
                Path(args.oor),
                args.name + "_oor",
                [
                    (
                        task_spec(
                            oor, "tf", args.seed + 1, (0, args.n_oor, args.n_oor), args.context_id
                        ),
                        {"train": 0, "dev": 0, "test": 0},
                    )
                ],
            )
        )
    for out_dir, name, tasks in jobs:
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[ShardRecord] = []
        for spec, every in tasks:
            for split in SPLITS:
                n = spec.n_examples(split)
                if n == 0:
                    continue
                examples = [
                    with_text(
                        generate(spec, split, i), text[split], args.seed, every[split], spec=spec
                    )
                    for i in range(n)
                ]
                record = write_shard(out_dir, spec, split, examples)
                records.append(record)
                print(
                    f"{name}: {record.task}/{record.variant}/{split}: {record.n_docs} documents, "
                    f"{record.n_tokens} tokens, {record.n_loss} supervised, {record.hash[:12]}"
                )
        # the two tasks share a seed on purpose, which a DatasetSpec refuses: hash the recipe here
        recipe = {
            "builder": "build_textfacts/1",
            "name": name,
            "unroll": v1.unroll,
            "tasks": [[spec.model_dump(mode="json"), every] for spec, every in tasks],
            "seed": args.seed,
            "skip_stories": args.skip_stories,
            "stories": [
                stories_dir.joinpath(f"tinystories.out_of_range.{s}.npz").name for s in SPLITS
            ],
        }
        digest = hashlib.blake2b(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode(), digest_size=32
        ).hexdigest()
        manifest = SplitManifest(
            name=name,
            spec_hash=digest,
            unroll=v1.unroll,
            hsn_version=hsn.__version__,
            numpy_version=np.__version__,
            shards=records,
            split_hashes=_split_hashes(records),
            leakage=[],  # not measured: story text repeats phrases, as in tinystories_v1
        )
        manifest.write(out_dir / MANIFEST_NAME)
        print(f"wrote {out_dir / MANIFEST_NAME}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stories", required=True, help="the tinystories_v1 dataset directory")
    p.add_argument("--out", required=True, help="output directory of the in-range dataset")
    p.add_argument(
        "--oor", default=None, help="output directory of the out-of-range evaluation set"
    )
    p.add_argument("--name", default="textfacts_v1")
    p.add_argument("--seed", type=int, default=31001)
    p.add_argument("--n-train", type=int, default=100000)
    p.add_argument("--n-dev", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-oor", type=int, default=1000, help="dev and test documents of --oor")
    p.add_argument("--text-every", type=int, default=8)
    p.add_argument("--skip-stories", type=int, default=32000)
    p.add_argument("--context-id", type=int, default=7)
    return build(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
