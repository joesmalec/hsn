"""Bits per letter by the distance of the previous mention, on text (``eval.by_distance``).

Two tables of one text group (documents whose every byte is a target, :mod:`hsn.data.text`),
from the per-byte losses of one pass. Both follow the analyses of 15 to 18 Nov 2026 (the grammar
evaluation's "which name" table and the names / words "beyond 256" readings), definition for
definition, so their numbers continue those series.

**Names** (``names``). A name is a capitalised word of three or more letters that the group's
documents use five or more times inside a sentence and whose lower-case form is at most a tenth
as frequent (:func:`name_list`; six words that pass the rule and are not names are left out,
:data:`NOT_NAMES`). Every mention inside a sentence (not at its start) is one case, by where
the same name was last written in the story, start to start:

* ``same_window_lt64``, ``same_window_64_255``: in the window that predicts this mention;
* ``earlier_window_lt64``, ``earlier_window_64_255``, ``ge256``: in an earlier window, and
  ``ge256``'s parts ``ge256_256_383``, ``ge256_384_511``, ``ge256_ge512``;
* ``first_occurrence``: nowhere before.

A row gives ``n`` (mentions), ``letters`` (their letters after the capital), ``bits_rest``
(bits per letter after the capital: *which* name, once a name is known to start) and
``bits_first`` (bits of the capital), each with a 95 % interval over stories (``ci95_*``:
:data:`BOOTSTRAP` resamples of the documents, fixed seed). A mention whose letters are predicted
in two windows is left out (``dropped_window_straddle``).

**Words** (``words``). Letters of content words by the distance back to the story's last use of
the word (case-folded, start to start): ``new``, ``lt64``, ``64_255``, ``256_511``, ``ge512``,
and of the words last used 256 or more bytes back the first letter and the later letters. A row
gives ``n`` (letters), ``share`` (of the group's scored bytes) and ``bits`` (per letter). A
content word is one the *training* shard uses fewer than :data:`CONTENT_THRESHOLD` times; the
words at or above it are listed once per dataset in ``configs/data/<name>.frequent_words.json``
(:func:`frequent_words_file`; ``python -m hsn.eval.distance --frequent-words`` writes it).
Without a list for the dataset the table is skipped and says so.

The static part (:class:`TextIndex`: where the names and words are) depends on the documents
alone and is built once per dataset; a pass then costs a few array sums.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hsn.config import REPO_ROOT

__all__ = [
    "BOOTSTRAP",
    "CONTENT_THRESHOLD",
    "NOT_NAMES",
    "FrequentWords",
    "TextIndex",
    "build_frequent_words",
    "find_frequent_words",
    "frequent_words_file",
    "name_list",
    "text_table",
]

#: Capitalised words that pass :func:`name_list`'s rule on TinyStories and are not names.
NOT_NAMES: frozenset[bytes] = frozenset(
    {b"God", b"Granny", b"Mama", b"Mrs", b"Tuesday", b"Wednesday"}
)
#: Resamples of the documents behind every 95 % interval, and their seed.
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261115
#: A word the training shard uses this often or more is not a content word.
CONTENT_THRESHOLD = 100_000

_NAME_WORD = re.compile(rb"[A-Za-z]+")
_WORD = re.compile(rb"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")
_SENTENCE_END = frozenset(b".!?\"'")
_SPACE = 32
_LN2 = float(np.log(2.0))
_FREQUENT_SUFFIX = ".frequent_words.json"

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.int64]
Bool = npt.NDArray[np.bool_]


# --------------------------------------------------------------------------------------------------
# Frequent words (the complement of the content words)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequentWords:
    """The case-folded words a dataset's training shard uses ``threshold`` times or more."""

    words: frozenset[bytes]
    threshold: int
    source: str  # the file the list came from


def find_frequent_words(dataset: str | None, root: Path | None = None) -> FrequentWords | None:
    """The committed list that names ``dataset`` (``<root>/*.frequent_words.json``, ``root``
    ``configs/data`` by default), or ``None`` when there is none."""
    if not dataset:
        return None
    folder = Path(root) if root is not None else REPO_ROOT / "configs" / "data"
    for path in sorted(folder.glob(f"*{_FREQUENT_SUFFIX}")):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if dataset in loaded.get("datasets", []):
            return FrequentWords(
                frozenset(word.encode("ascii") for word in loaded["words"]),
                int(loaded["threshold"]),
                path.name,
            )
    return None


def build_frequent_words(
    shard: Path, *, threshold: int = CONTENT_THRESHOLD, chunk: int = 4 << 20
) -> dict[str, Any]:
    """Count the case-folded words of a shard's ``tokens`` (streamed from the ``.npz``: the
    200 MB training shard is never held whole) and list those used ``threshold`` times or
    more. Document boundaries need no care: ``BOS`` and ``EOS`` are not word bytes."""
    counts: Counter[bytes] = Counter()
    tail = b""
    with zipfile.ZipFile(shard) as archive, archive.open("tokens.npy") as stream:
        version = np.lib.format.read_magic(stream)
        header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _fortran, dtype = header(stream)
        if dtype != np.dtype("uint8") or len(shape) != 1:
            raise ValueError(f"{shard}: tokens must be a 1-d uint8 array")
        while block := stream.read(chunk):
            block = tail + block
            cut = len(block)
            while cut and (block[cut - 1 : cut].isalnum() or block[cut - 1] == 39):
                cut -= 1  # never cut inside a word: keep its start for the next block
            counts.update(match.group().lower() for match in _WORD.finditer(block[:cut]))
            tail = block[cut:]
    counts.update(match.group().lower() for match in _WORD.finditer(tail))
    words = sorted(word.decode("ascii") for word, n in counts.items() if n >= threshold)
    return {
        "threshold": threshold,
        "n_words": int(sum(counts.values())),
        "n_types": len(counts),
        "words": words,
    }


# --------------------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------------------


def _capitalised(word: bytes) -> bool:
    return word[:1].isupper() and word[1:].islower() and len(word) >= 3


def _sentence_initial(body: bytes, start: int) -> bool:
    """Whether the word at ``start`` opens a sentence: the byte before it (spaces skipped) ends
    a sentence or opens a quotation, is a control or marker symbol (``BOS``, the newline
    marker) or is not ASCII (a typographic quotation mark)."""
    k = start - 1
    while k > 0 and body[k] == _SPACE:
        k -= 1
    return k < 0 or body[k] in _SENTENCE_END or body[k] < _SPACE or body[k] >= 128


def name_list(bodies: Sequence[bytes]) -> list[bytes]:
    """The names of ``bodies`` by rule (module docstring), sorted."""
    inside: Counter[bytes] = Counter()
    capitalised: Counter[bytes] = Counter()
    lower: Counter[bytes] = Counter()
    for body in bodies:
        for match in _NAME_WORD.finditer(body):
            word = match.group()
            if _capitalised(word):
                capitalised[word] += 1
                if not _sentence_initial(body, match.start()):
                    inside[word] += 1
            elif word.islower():
                lower[word] += 1
    return sorted(
        word
        for word, n in inside.items()
        if n >= 5 and lower[word.lower()] <= 0.1 * capitalised[word] and word not in NOT_NAMES
    )


# --------------------------------------------------------------------------------------------------
# The static index of one text group
# --------------------------------------------------------------------------------------------------


class TextIndex:
    """Where the names and the words of a group's documents are (module docstring).

    ``bodies`` are the scored documents (``BOS`` first) and index ``i`` of the group's flat
    arrays is the position that reads byte ``i`` and predicts byte ``i + 1``; ``names`` is the
    name list (of the whole group, also when fewer documents are scored); ``window`` the
    evaluation window; ``frequent`` the dataset's frequent words (``None``: no words table).
    """

    def __init__(
        self,
        bodies: Sequence[bytes],
        names: Sequence[bytes],
        *,
        window: int,
        frequent: FrequentWords | None,
    ) -> None:
        self.n_docs = len(bodies)
        self.window = window
        self.n_names = len(names)
        self.frequent = frequent
        lengths = np.fromiter((len(body) for body in bodies), dtype=np.int64, count=self.n_docs)
        self.offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
        name_set = frozenset(names)
        mentions: list[tuple[int, int, int, int, int, int]] = []
        self.dropped = 0
        words: list[tuple[int, int, int, int]] = []  # flat first index, length, distance, content
        for d, body in enumerate(bodies):
            base = int(self.offsets[d])
            last: dict[bytes, int] = {}
            for match in _NAME_WORD.finditer(body):
                word, s, e = match.group(), match.start(), match.end()
                if not _capitalised(word) or word not in name_set:
                    continue
                previous = last.get(word)
                last[word] = s
                first = s - 1  # the position that predicts the capital
                if first // window != (e - 2) // window:
                    self.dropped += 1
                    continue
                same = previous is not None and previous >= (first // window) * window
                mentions.append(
                    (
                        d,
                        base + first,
                        e - s,
                        -1 if previous is None else s - previous,
                        int(same),
                        int(_sentence_initial(body, s)),
                    )
                )
            if frequent is None:
                continue
            seen: dict[bytes, int] = {}
            for match in _WORD.finditer(body):
                s, e = match.span()
                text = match.group().lower()
                before = seen.get(text)
                seen[text] = s
                words.append(
                    (
                        base + s - 1,
                        e - s,
                        -1 if before is None else s - before,
                        int(text not in frequent.words),
                    )
                )
        table = np.asarray(mentions, dtype=np.int64).reshape(-1, 6)
        self.m_doc, self.m_index, self.m_len, self.m_dist, same_a, initial = table.T
        self.m_same = same_a.astype(np.bool_)
        self.m_inside = initial == 0
        word_table = np.asarray(words, dtype=np.int64).reshape(-1, 4)
        w_first, w_len, w_dist, w_content = word_table.T
        # one entry per letter: its flat index, whether it is its word's first, the distance
        which = np.repeat(np.arange(w_len.shape[0]), w_len)
        within = np.arange(which.shape[0]) - np.repeat(np.cumsum(w_len) - w_len, w_len)
        self.l_index = w_first[which] + within
        self.l_first = within == 0
        self.l_dist = w_dist[which]
        self.l_content = w_content[which].astype(np.bool_)

    # the cases, in table order ------------------------------------------------------------------

    def name_cases(self) -> list[tuple[str, Bool]]:
        d, same, inside = self.m_dist, self.m_same, self.m_inside
        earlier = inside & ~same
        return [
            ("same_window_lt64", inside & same & (d >= 0) & (d < 64)),
            ("same_window_64_255", inside & same & (d >= 64)),
            ("earlier_window_lt64", earlier & (d >= 0) & (d < 64)),
            ("earlier_window_64_255", earlier & (d >= 64) & (d < 256)),
            ("ge256", earlier & (d >= 256)),
            ("ge256_256_383", inside & (d >= 256) & (d < 384)),
            ("ge256_384_511", inside & (d >= 384) & (d < 512)),
            ("ge256_ge512", inside & (d >= 512)),
            ("first_occurrence", inside & (d < 0)),
        ]

    def word_cases(self) -> list[tuple[str, Bool]]:
        d, content = self.l_dist, self.l_content
        return [
            ("new", content & (d < 0)),
            ("lt64", content & (d >= 0) & (d < 64)),
            ("64_255", content & (d >= 64) & (d < 256)),
            ("256_511", content & (d >= 256) & (d < 512)),
            ("ge512", content & (d >= 512)),
            ("ge256_first_letter", content & (d >= 256) & self.l_first),
            ("ge256_later_letters", content & (d >= 256) & ~self.l_first),
        ]


def _intervals(columns: Float, counts: Float) -> tuple[Float, Float]:
    """2.5 % and 97.5 % of ``sum / count`` over resampled documents, per column: ``columns``
    and ``counts`` are ``[n_docs, n_columns]`` per-document sums."""
    n_docs = columns.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    p = np.full(n_docs, 1.0 / n_docs)
    ratios = []
    for lo in range(0, BOOTSTRAP, 250):  # the weights are drawn row by row: chunks change nothing
        weights = rng.multinomial(n_docs, p, size=min(250, BOOTSTRAP - lo)).astype(np.float64)
        ratios.append((weights @ columns) / np.maximum(weights @ counts, 1e-9))
    stacked = np.concatenate(ratios)
    return np.percentile(stacked, 2.5, axis=0), np.percentile(stacked, 97.5, axis=0)


def text_table(index: TextIndex, nll: npt.NDArray[np.float32], hit: Bool) -> dict[str, Any]:
    """The table of one text group: ``nll`` (nats; NaN where a position has no target) and
    ``hit`` (the arg-max was the target) flat over the documents ``index`` was built from."""
    if nll.shape[0] != int(index.offsets[-1]):
        raise ValueError("text_table: the losses do not cover the indexed documents")
    bits = nll.astype(np.float64) / _LN2
    scored = np.isfinite(bits)
    n_scored = int(scored.sum())
    out: dict[str, Any] = {
        "kind": "text",
        "n_docs": index.n_docs,
        "n": n_scored,
        "bits_per_byte": float(bits[scored].mean()),
        "accuracy": float(hit[scored].mean()),
    }
    # names: per-mention sums over the letters after the capital and the capital's own bits
    total = np.concatenate([np.zeros(1), np.cumsum(np.nan_to_num(bits))])
    at, length = index.m_index, index.m_len
    rest = total[at + length] - total[at + 1]
    first = total[at + 1] - total[at]
    cases = index.name_cases()
    sums = np.zeros((index.n_docs, 2 * len(cases)))
    counts = np.zeros_like(sums)
    for j, (_key, mask) in enumerate(cases):
        docs = index.m_doc[mask]
        sums[:, 2 * j] = np.bincount(docs, weights=rest[mask], minlength=index.n_docs)
        counts[:, 2 * j] = np.bincount(
            docs, weights=(length[mask] - 1).astype(np.float64), minlength=index.n_docs
        )
        sums[:, 2 * j + 1] = np.bincount(docs, weights=first[mask], minlength=index.n_docs)
        counts[:, 2 * j + 1] = np.bincount(docs, minlength=index.n_docs)
    low, high = _intervals(sums, counts)
    rows: dict[str, Any] = {}
    for j, (key, mask) in enumerate(cases):
        n = int(mask.sum())
        row: dict[str, Any] = {"n": n, "letters": int(counts[:, 2 * j].sum())}
        for name, column in (("rest", 2 * j), ("first", 2 * j + 1)):
            count = counts[:, column].sum()
            row[f"bits_{name}"] = float(sums[:, column].sum() / count) if count else None
            row[f"ci95_{name}"] = [float(low[column]), float(high[column])] if count else None
        rows[key] = row
    out["names"] = {
        "n_names": index.n_names,
        "n_mentions": int(index.m_doc.shape[0]),
        "dropped_window_straddle": index.dropped,
        "window": index.window,
        "rows": rows,
    }
    if index.frequent is None:
        out["words"] = {"skipped": "no frequent-word list for this dataset (configs/data)"}
        return out
    letter_bits = bits[index.l_index]
    letter_scored = scored[index.l_index]
    word_rows: dict[str, Any] = {}
    for key, case in index.word_cases():
        mask = case & letter_scored
        n = int(mask.sum())
        word_rows[key] = {
            "n": n,
            "share": n / n_scored,
            "bits": float(letter_bits[mask].mean()) if n else None,
        }
    out["words"] = {
        "frequent_words": index.frequent.source,
        "threshold": index.frequent.threshold,
        "rows": word_rows,
    }
    return out


# --------------------------------------------------------------------------------------------------
# Writing a dataset's frequent-word list
# --------------------------------------------------------------------------------------------------


def frequent_words_file(
    manifest_path: Path,
    *,
    task: str | None = None,
    threshold: int = CONTENT_THRESHOLD,
    also: Sequence[str] = (),
) -> dict[str, Any]:
    """The content of ``configs/data/<name>.frequent_words.json`` for the dataset of
    ``manifest_path``: the words of its training shard (``task``'s; the only one by default)
    at or above ``threshold`` uses, with where they came from. ``also`` names other datasets
    with the same training text (``tinystories_w1024`` has ``tinystories_v1``'s stories)."""
    from hsn.data.splits import SplitManifest

    manifest = SplitManifest.load(manifest_path)
    shards = [
        record
        for record in manifest.shards
        if record.split == "train" and (task is None or record.task == task)
    ]
    if len(shards) != 1:
        raise ValueError(f"{manifest_path}: {len(shards)} training shards match; name the task")
    record = shards[0]
    built = build_frequent_words(Path(manifest_path).parent / record.file, threshold=threshold)
    return {
        "format": 1,
        "datasets": [manifest.name, *also],
        "rule": (
            "case-folded words ([A-Za-z0-9]+ with inner apostrophes) the training shard uses "
            "threshold times or more; every other word is a content word (hsn.eval.distance)"
        ),
        "source": {
            "dataset": manifest.name,
            "shard": record.file,
            "hash": record.hash,
            "n_words": built["n_words"],
            "n_types": built["n_types"],
        },
        "threshold": built["threshold"],
        "words": built["words"],
    }
