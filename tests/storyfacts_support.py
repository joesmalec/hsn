"""Shared helpers of the story-facts and recall-router tests: small invented stories as source
files and a small ``storyfacts`` spec over them (windows of 32 bytes, three short bands; the nearest
starts at 40 bytes because a question is itself 20 to 55 bytes long before its answer).

Not a test module (no ``test_`` prefix).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from hsn.data.splits import SplitManifest
from hsn.data.storyfacts import StoryFactsSpec, build_storyfacts

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "configs" / "data" / "storyfacts_v1.yaml"
GOLDEN = REPO / "configs" / "data" / "storyfacts_v1.manifest.json"
UNROLL = 32
SEPARATOR = "<|endoftext|>"
BANDS = {"near": (40, 99), "mid": (100, 249), "far": (250, 900)}

_NAMES = ("Tom", "Lily", "Sam", "Mia", "Ben")
_THINGS = ("ball", "kite", "boat", "drum", "hat")
_SENTENCES = (
    "{a} went to the park with {b}.",
    "{a} saw a big red {t} near the old tree.",
    '"Look at the {t}," said {a}.',
    '"I like it. Can we play with it?" asked {b}.',
    "They played with the {t} all day!",
    "Mr. Bear waved at {a} from the hill.",
    "{b} was very happy and {a} was happy too.",
    "Then the sun went down and it was time to go home.",
)


def story(seed: int) -> str:
    """A small story of two to four paragraphs (deterministic)."""
    rng = np.random.default_rng([4242, seed])
    a, b = (_NAMES[int(i)] for i in rng.permutation(len(_NAMES))[:2])
    thing = _THINGS[int(rng.integers(len(_THINGS)))]
    paragraphs = []
    for _ in range(int(rng.integers(2, 5))):
        picks = rng.integers(len(_SENTENCES), size=int(rng.integers(2, 5)))
        paragraphs.append(" ".join(_SENTENCES[int(i)].format(a=a, b=b, t=thing) for i in picks))
    return f"Once upon a time, {a} and {b} had story {seed}.\n" + "\n".join(paragraphs)


def write_sources(folder: Path, n_train: int = 260, n_held: int = 200) -> dict[str, Any]:
    """Write the two source files into ``folder``; returns their ``train`` / ``heldout`` spec
    entries (the train file ends inside a story, the held-out file starts inside one)."""
    train = (SEPARATOR.join(story(i) for i in range(n_train)) + SEPARATOR + "cut").encode()
    held = ("cut" + SEPARATOR + SEPARATOR.join(story(10_000 + i) for i in range(n_held))).encode()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train.txt").write_bytes(train)
    (folder / "held.txt").write_bytes(held)
    return {
        "train": {
            "file": "train.txt",
            "sha256": hashlib.sha256(train).hexdigest(),
            "drop_last": True,
        },
        "heldout": {
            "file": "held.txt",
            "sha256": hashlib.sha256(held).hexdigest(),
            "drop_first": True,
        },
    }


def tiny_spec(sources: dict[str, Any], **top: Any) -> StoryFactsSpec:
    """The committed spec's relations, words and names at a small size over ``sources``."""
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    raw.update(name="storyfacts_tiny", unroll=UNROLL, min_story_bytes=100, **sources)
    raw["stories"].update(
        n={"train": 60, "dev": 30, "test": 16},
        max_bytes=1000,
        zone=120,
        bands=[{"name": name, "lo": lo, "hi": hi, "weight": 1} for name, (lo, hi) in BANDS.items()],
    )
    raw["kv"]["n"] = {"train": 30, "dev": 12, "test": 8}
    raw["names"]["n"] = {"train": 60, "dev": 20, "test": 20}
    raw.update(top)
    return StoryFactsSpec.model_validate(raw)


def build_tiny(folder: Path, **top: Any) -> tuple[StoryFactsSpec, Path, SplitManifest]:
    """Sources in ``folder/raw``, the dataset in ``folder/data``: the spec, the manifest's
    path and the manifest."""
    spec = tiny_spec(write_sources(folder / "raw"), **top)
    manifest, _ = build_storyfacts(spec, folder / "raw", folder / "data")
    return spec, folder / "data" / "manifest.json", manifest
