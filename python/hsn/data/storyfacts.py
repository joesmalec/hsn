"""Stories with stated facts, in the P0.4 shard format (18 Nov 2026): the corpus of the recall
router (the router itself, a per-document classifier, is not on main).

One corpus that holds what a fast model reads well and what only a memory recalls, so that a
model pair trained on it can be compared document by document. Everything is ordinary story
text under one context symbol: no marker byte and no context id tells a document's kind.

**Documents.** Four tasks, one shard each per split (``data.tasks`` selects them):

* ``plain``   one to four consecutive stories of the source (TinyStories), unchanged, joined by
              a blank line.
* ``stated``  the same, with one to three *facts* inserted as sentences ("Tom's dog is called
              Zukan.") and never asked: a fact in the text is not yet a need to recall it.
* ``facts``   the same, and later at least one *question* with its answer ("What is Tom's dog
              called? Zukan."), at a recall distance drawn from the spec's bands (in the reader's
              reach, in the reach of a carried cache, beyond every cache).
* ``kv``      no story: a sheet of two to eight facts with questions between them.

A story document's kind is drawn independently of its stories, so the three kinds have one
length distribution and read alike up to the first fact.

**Facts.** A fact is a *relation* of the spec: sentence templates for the statement and the
question, a value (an invented name, a number or a word of a list) and slot words. Within a
document every relation occurs at most once. In every template the word directly before the
value is the same in the statement and in the question (``… is called Zukan.`` / ``… called?
Zukan.``; ``… holds 7 apples.`` / ``It holds 7.``): the *key word rule*, checked when the spec
loads. It is what lets a word-addressed memory bind a fact (``docs/amendments/write_events.md``:
the value is written under the code of the word before it and read where that word closes
again); a model that attends over the text does not need it.

**Vocabulary.** Value names are invented (syllables), drawn from one pool that is cut into
disjoint train, dev and test parts, so no name asked at evaluation was ever written in
training. A fact's owner is a character of the document's first story where one is found (so a
fact does not announce itself by a new name), else a name of the split's own list. Numbers,
relation words and slot words are shared: they are the language, not the fact.

**Supervision.** As in :mod:`hsn.data.text`, every byte is a target: ``loss_mask`` is True at
every position but the last, and dev loss / ln 2 is bits per byte. ``deps`` is 1 at text
positions. At an *answer position*, a position whose target is a byte of an asked value, it is
the recall distance: the position minus the position of that byte in the statement. So the
answers are the loss positions with a dependency above 1 (``train.answer_weight`` and
``eval.by_distance`` read them that way), and answer accuracy by distance needs no side file.

**Determinism.** A document is a pure function of the spec, the source files (pinned by
SHA-256), its split and its index: ``numpy.random.default_rng([seed, split, kind, index,
attempt])``, integer draws only. The manifest has no timestamps (``--golden``).

    python -I -m hsn.data.storyfacts build --spec configs/data/storyfacts_v1.yaml \\
        --raw <folder with the source files> --out data/storyfacts_v1 \\
        --golden configs/data/storyfacts_v1.manifest.json
    python -I -m hsn.data.storyfacts describe --out data/storyfacts_v1
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import string
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import hsn
from hsn.config import Split
from hsn.data.splits import (
    MANIFEST_NAME,
    SHARD_FORMAT,
    ShardRecord,
    SplitManifest,
    read_shard,
    shard_file_name,
    shard_hash,
)
from hsn.data.synthetic import BOS, EOS, N_MARKERS, N_RESERVED, SPLITS, Variant
from hsn.data.text import SourceSpec

__all__ = [
    "FAMILY",
    "STORY_TASKS",
    "TASKS",
    "Answer",
    "Band",
    "Document",
    "Relation",
    "StoryFactsSpec",
    "answers_of",
    "build_storyfacts",
    "describe",
    "kv_document",
    "load_storyfacts_spec",
    "main",
    "sentence_boundaries",
    "story_document",
]

U8 = npt.NDArray[np.uint8]

#: The kinds of a story document, in shard order, and the fact sheet.
STORY_TASKS: tuple[str, ...] = ("plain", "stated", "facts")
TASKS: tuple[str, ...] = (*STORY_TASKS, "kv")
#: The shard family of each task: ``text`` gets the text tables of ``eval.by_distance``, any
#: other family its answers by distance.
FAMILY: dict[str, str] = {
    "plain": "text",
    "stated": "storyfacts",
    "facts": "storyfacts",
    "kv": "storyfacts",
}
_SPACE = 32
_ATTEMPTS = 64
#: A full stop after one of these does not end a sentence.
_ABBREVIATIONS = re.compile(rb"(?:^|[^A-Za-z])(?:Mr|Mrs|Ms|Dr|St)\.$")
_SENTENCE_END = re.compile(rb"[.!?](?= [A-Z\"])")
#: A capitalised word inside a sentence: a character of the story.
_CHARACTER = re.compile(rb"(?<=[a-z,] )[A-Z][a-z]{2,}(?![A-Za-z])")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------------------


def _fields(template: str) -> list[str]:
    return [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]


def _key_word(template: str) -> str:
    """The word directly before the value of a template, lower case without punctuation (a
    slot keeps its braces: the same slot gives the same word)."""
    head = template.split("{value}")[0]
    if not head.endswith(" "):
        raise ValueError(f"{template!r}: the value must follow a word and one space")
    return head.rstrip().rsplit(" ", 1)[-1].strip(".,;:!?\"'").lower()


class Relation(_Strict):
    """One kind of fact: its sentences and what fills them. ``value`` is ``name`` (an invented
    name of the split's pool), ``number`` (``number_range``) or the name of a word list;
    ``slots`` maps every other field of the templates but ``owner`` to a word list."""

    name: str
    value: str
    slots: dict[str, str] = Field(default_factory=dict)
    statements: list[str] = Field(min_length=1)
    questions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        allowed = {"owner", "value", *self.slots}
        for template in (*self.statements, *self.questions):
            names = _fields(template)
            if names.count("value") != 1:
                raise ValueError(f"{self.name}: {template!r} needs {{value}} exactly once")
            unknown = sorted(set(names) - allowed)
            if unknown:
                raise ValueError(f"{self.name}: {template!r} has no slot for {unknown}")
            if not template.isascii():
                raise ValueError(f"{self.name}: {template!r} is not ASCII")
        keys = {_key_word(t) for t in (*self.statements, *self.questions)}
        if len(keys) != 1 or "" in keys:
            raise ValueError(
                f"{self.name}: the word before the value must be one word, the same in every "
                f"statement and question (the key word rule); got {sorted(keys)}"
            )
        return self

    @property
    def key_word(self) -> str:
        return _key_word(self.statements[0])


class Band(_Strict):
    """A range of recall distances ``[lo, hi]`` (positions) and its integer weight."""

    name: str
    lo: int = Field(ge=2)
    hi: int
    weight: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.hi < self.lo:
            raise ValueError(f"band {self.name}: hi {self.hi} < lo {self.lo}")
        return self


class StoryParams(_Strict):
    """Story documents. ``count_weights[k - 1]`` weighs a document of ``k`` stories; ``kinds``
    weighs plain / stated / facts; ``fact_weights[n - 1]`` weighs ``n`` facts."""

    n: dict[Split, int]
    count_weights: list[int] = Field(min_length=1)
    max_bytes: int = Field(ge=64)
    gap: int = Field(default=2, ge=1)  # newline markers between two stories
    kinds: dict[str, int]
    fact_weights: list[int] = Field(min_length=1)
    ask_percent: int = Field(ge=0, le=100)  # every fact after the one that is always asked
    zone: int = Field(ge=1)  # facts are stated in the first `zone` bytes ...
    late_percent: int = Field(ge=0, le=100)  # ... or, this often, up to the middle
    story_owner_percent: int = Field(ge=0, le=100)
    bands: list[Band] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if sorted(self.kinds) != sorted(STORY_TASKS):
            raise ValueError(f"stories.kinds must weigh exactly {list(STORY_TASKS)}")
        for label, weights in (
            ("count_weights", self.count_weights),
            ("kinds", list(self.kinds.values())),
            ("fact_weights", self.fact_weights),
            ("bands", [band.weight for band in self.bands]),
        ):
            if min(weights) < 0 or sum(weights) <= 0:
                raise ValueError(f"stories.{label}: weights must be >= 0 with a positive sum")
        if len({band.name for band in self.bands}) != len(self.bands):
            raise ValueError("stories.bands: duplicate names")
        return self


class KVParams(_Strict):
    """Fact sheets: ``facts`` is the range of facts per sheet."""

    n: dict[Split, int]
    facts: tuple[int, int]
    ask_percent: int = Field(ge=0, le=100)
    newline_percent: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 1 <= self.facts[0] <= self.facts[1]:
            raise ValueError(f"kv.facts {self.facts} must satisfy 1 <= lo <= hi")
        return self


class NameParams(_Strict):
    """The invented names: every ``syllables``-fold product of onset + vowel, then a final;
    ``n`` of them per split, disjoint, in an order drawn from ``seed``."""

    seed: int = Field(ge=0)
    onsets: list[str] = Field(min_length=1)
    vowels: list[str] = Field(min_length=1)
    finals: list[str] = Field(min_length=1)
    syllables: int = Field(default=2, ge=1, le=3)
    n: dict[Split, int]
    exclude: list[str] = Field(default_factory=list)


class StoryFactsSpec(_Strict):
    """The dataset (module docstring)."""

    name: str
    unroll: int = Field(ge=1)
    variant: Variant = "out_of_range"
    context_id: int = Field(ge=0, le=15)
    seed: int = Field(ge=0)
    separator: str = Field(min_length=1)
    min_story_bytes: int = Field(ge=1)
    newline: int = Field(default=16, ge=N_RESERVED, lt=N_MARKERS)
    tab: int = Field(default=17, ge=N_RESERVED, lt=N_MARKERS)
    stories: StoryParams
    kv: KVParams
    names: NameParams
    owners: dict[Split, list[str]]
    number_range: tuple[int, int]
    words: dict[str, list[str]]
    relations: list[Relation] = Field(min_length=1)
    train: SourceSpec
    heldout: SourceSpec

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.newline == self.tab:
            raise ValueError("newline and tab need different marker symbols")
        for label, counts in (
            ("stories.n", self.stories.n),
            ("kv.n", self.kv.n),
            ("names.n", self.names.n),
            ("owners", self.owners),
        ):
            if sorted(counts) != sorted(SPLITS):
                raise ValueError(f"{label} must name every split {list(SPLITS)}")
        if len({r.name for r in self.relations}) != len(self.relations):
            raise ValueError("relations: duplicate names")
        if len({r.key_word for r in self.relations}) != len(self.relations):
            raise ValueError("relations: two relations share a key word")
        for relation in self.relations:
            lists = [*relation.slots.values()]
            if relation.value not in ("name", "number"):
                lists.append(relation.value)
            for name in lists:
                if not self.words.get(name):
                    raise ValueError(f"relation {relation.name}: no word list {name!r}")
        for name, words in self.words.items():
            if not all(word.isascii() and word for word in words):
                raise ValueError(f"words.{name}: entries must be non-empty ASCII")
        seen: set[str] = set()
        for split in SPLITS:
            owners = {owner.lower() for owner in self.owners[split]}
            if not owners or len(owners) != len(self.owners[split]) or owners & seen:
                raise ValueError("owners: every split needs its own distinct names")
            seen |= owners
        lo, hi = self.number_range
        if not 0 <= lo <= hi:
            raise ValueError(f"number_range {self.number_range} must satisfy 0 <= lo <= hi")
        if self.kv.facts[1] > len(self.relations) or len(self.stories.fact_weights) > len(
            self.relations
        ):
            raise ValueError("a document cannot hold more facts than there are relations")
        self.vocabulary()  # raises when the pool is too small
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def spec_hash(self) -> str:
        return hashlib.blake2b(self.canonical_json().encode(), digest_size=32).hexdigest()

    def vocabulary(self) -> dict[Split, tuple[str, ...]]:
        """The value names of every split: capitalised, disjoint across splits, none equal to
        an owner, a slot word, a template word or an entry of ``names.exclude``."""
        names = self.names
        syllables = ["".join(pair) for pair in itertools.product(names.onsets, names.vowels)]
        stems = ("".join(parts) for parts in itertools.product(syllables, repeat=names.syllables))
        taken = {word.lower() for word in names.exclude}
        taken |= {owner.lower() for owners in self.owners.values() for owner in owners}
        taken |= {word.lower() for words in self.words.values() for word in words}
        for relation in self.relations:
            for template in (*relation.statements, *relation.questions):
                taken |= {word.lower() for word in re.findall(r"[A-Za-z]+", template)}
        pool = sorted({stem + final for stem in stems for final in names.finals} - taken)
        need = sum(names.n[split] for split in SPLITS)
        if len(pool) < need:
            raise ValueError(f"names: {len(pool)} candidates for {need} names")
        order = np.random.default_rng([names.seed]).permutation(len(pool))
        out: dict[Split, tuple[str, ...]] = {}
        cursor = 0
        for split in SPLITS:
            chosen = order[cursor : cursor + names.n[split]]
            out[split] = tuple(pool[int(i)].capitalize() for i in chosen)
            cursor += names.n[split]
        return out


def load_storyfacts_spec(path: str | Path) -> StoryFactsSpec:
    """Load and validate a story-facts dataset spec YAML."""
    spec_path = Path(path)
    try:
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{spec_path}: invalid YAML: {exc}") from exc
    try:
        return StoryFactsSpec.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"{spec_path}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """An asked value: ``position`` is the document offset of its first answer position (the
    position whose target is the value's first byte), ``length`` its bytes, ``distance`` the
    recall distance of each of them; ``band`` is empty in a fact sheet."""

    position: int
    length: int
    distance: int
    relation: str
    band: str


@dataclass(frozen=True)
class Document:
    """``tokens`` is ``BOS, body, EOS``; ``n_facts`` counts the stated facts, asked or not."""

    task: str
    index: int
    tokens: U8
    answers: tuple[Answer, ...]
    n_facts: int

    @property
    def deps(self) -> npt.NDArray[np.int32]:
        """One dependency per loss position (every position but the last): 1, or the recall
        distance at an answer position."""
        deps = np.ones(self.tokens.shape[0] - 1, dtype=np.int32)
        for answer in self.answers:
            deps[answer.position : answer.position + answer.length] = answer.distance
        return deps


@dataclass(frozen=True)
class _Fact:
    relation: str
    statement: bytes
    question: bytes
    value: bytes
    statement_at: int  # offset of the value in the statement
    question_at: int  # offset of the value in the question


def _weighted(rng: np.random.Generator, weights: Sequence[int]) -> int:
    draw = int(rng.integers(sum(weights)))
    for i, weight in enumerate(weights):
        if draw < weight:
            return i
        draw -= weight
    raise AssertionError("unreachable")


def _percent(rng: np.random.Generator, percent: int) -> bool:
    return int(rng.integers(100)) < percent


def _pick(rng: np.random.Generator, items: Sequence[str]) -> str:
    return items[int(rng.integers(len(items)))]


def _draw_facts(
    spec: StoryFactsSpec,
    names: Sequence[str],
    owners: Sequence[str],
    rng: np.random.Generator,
    n: int,
) -> list[_Fact]:
    """``n`` facts of distinct relations; name values are distinct too."""
    order = rng.permutation(len(spec.relations))[:n]
    used: set[str] = set()
    facts: list[_Fact] = []
    for r in order.tolist():
        relation = spec.relations[r]
        if relation.value == "name":
            value = _pick(rng, names)
            while value in used:
                value = _pick(rng, names)
            used.add(value)
        elif relation.value == "number":
            value = str(int(rng.integers(spec.number_range[0], spec.number_range[1] + 1)))
        else:
            value = _pick(rng, spec.words[relation.value])
        # in the slots' sorted order: the draws must not depend on how a spec file orders them
        fills = {
            slot: _pick(rng, spec.words[relation.slots[slot]]) for slot in sorted(relation.slots)
        }
        fills["owner"] = _pick(rng, owners)
        statement = _pick(rng, relation.statements)
        question = _pick(rng, relation.questions)
        facts.append(
            _Fact(
                relation.name,
                statement.format(value=value, **fills).encode(),
                question.format(value=value, **fills).encode(),
                value.encode(),
                len(statement.split("{value}")[0].format(**fills)),
                len(question.split("{value}")[0].format(**fills)),
            )
        )
    return facts


def sentence_boundaries(text: bytes, newline: int = 16) -> list[int]:
    """The offsets of ``text`` where a sentence may be inserted, ascending: 0, the end, the
    end of every paragraph that closes a sentence, and inside a paragraph every offset behind
    ``.``, ``!`` or ``?`` that a space and a capital or an opening quotation mark follow. Not
    inside an open quotation (an odd number of ``"`` before it in the paragraph), not behind
    an abbreviation (``Mr.``), and nowhere inside a paragraph that holds a byte above 127 (its
    quotation marks may be typographic ones, which are not counted)."""
    out = {0, len(text)}
    start = 0
    for paragraph in text.split(bytes([newline])):
        end = start + len(paragraph)
        if paragraph and paragraph.isascii():
            for match in _SENTENCE_END.finditer(paragraph):
                head = paragraph[: match.end()]
                if head.count(b'"') % 2 == 0 and not _ABBREVIATIONS.search(head):
                    out.add(start + match.end())
            if paragraph[-1:] in (b".", b"!", b"?", b'"') and paragraph.count(b'"') % 2 == 0:
                out.add(end)
        start = end + 1
    return sorted(out)


def _insert(text: bytes, places: Sequence[tuple[int, bytes]]) -> tuple[bytes, list[int]]:
    """``text`` with a sentence inserted at each ``(offset, sentence)`` (ascending offsets;
    equal offsets in the order given), and where each sentence starts in the result. At
    offset 0 the sentence is followed by a space, elsewhere it is preceded by one."""
    parts: list[bytes] = []
    starts: list[int] = []
    cursor = length = 0
    for offset, sentence in places:
        parts.append(text[cursor:offset])
        length += offset - cursor
        cursor = offset
        if offset == 0:
            starts.append(length)
            parts.append(sentence + b" ")
        else:
            starts.append(length + 1)
            parts.append(b" " + sentence)
        length += len(sentence) + 1
    parts.append(text[cursor:])
    return b"".join(parts), starts


def _characters(story: bytes) -> list[str]:
    """Up to three characters of a story: the capitalised words that occur inside a sentence
    two or more times, most frequent first."""
    counts = Counter(match.group() for match in _CHARACTER.finditer(story))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], story.find(item[0])))
    return [name.decode() for name, count in ranked[:3] if count >= 2]


def _tokens(body: bytes) -> U8:
    return np.frombuffer(bytes([BOS]) + body + bytes([EOS]), dtype=np.uint8)


def story_document(
    spec: StoryFactsSpec,
    split: Split,
    index: int,
    stories: Sequence[bytes],
    names: Sequence[str] | None = None,
) -> Document:
    """Story document ``index`` of ``split`` over its ``stories`` (encoded bodies): its kind,
    its facts and its questions are drawn from ``(seed, split, index)`` alone. ``names`` is the
    split's value-name pool (``spec.vocabulary()[split]`` when not given)."""
    params = spec.stories
    names = spec.vocabulary()[split] if names is None else names
    body = bytes([spec.newline] * params.gap).join(stories)
    base = [spec.seed, SPLITS.index(split), 0, index]
    kind = STORY_TASKS[
        _weighted(np.random.default_rng(base), [params.kinds[task] for task in STORY_TASKS])
    ]
    if kind == "plain":
        return Document(kind, index, _tokens(body), (), 0)
    bounds = sentence_boundaries(body, spec.newline)
    for attempt in range(_ATTEMPTS):
        rng = np.random.default_rng([*base, attempt + 1])
        owners: Sequence[str] = spec.owners[split]
        if _percent(rng, params.story_owner_percent):
            owners = _characters(stories[0]) or owners
        facts = _draw_facts(spec, names, owners, rng, 1 + _weighted(rng, params.fact_weights))
        zone = (0, params.zone)
        if _percent(rng, params.late_percent) and len(body) // 2 > params.zone:
            zone = (params.zone, len(body) // 2)
        places = [b for b in bounds if zone[0] <= b < zone[1]] or [0]
        at = sorted(places[int(rng.integers(len(places)))] for _ in facts)
        stated, starts = _insert(body, [(p, f.statement) for p, f in zip(at, facts, strict=True)])
        if kind == "stated":
            return Document(kind, index, _tokens(stated), (), len(facts))
        made = _ask(spec, rng, stated, facts, starts)
        if made is not None:
            return Document(kind, index, _tokens(made[0]), made[1], len(facts))
    raise ValueError(f"story document {split}:{index}: no question fits a band")


def _ask(
    spec: StoryFactsSpec,
    rng: np.random.Generator,
    text: bytes,
    facts: Sequence[_Fact],
    starts: Sequence[int],
) -> tuple[bytes, tuple[Answer, ...]] | None:
    """``text`` (the facts stated, their statements at the offsets ``starts``) with the
    questions of the asked facts inserted, and the answers; ``None`` when no question fits a
    band or a distance left its band (another question came between).

    A distance is ``(offset of the answer's value) - (offset of the stated value) - 1`` in the
    body: the answer position minus the fact position."""
    bands = spec.stories.bands
    bounds = sentence_boundaries(text, spec.newline)
    values = [start + fact.statement_at for start, fact in zip(starts, facts, strict=True)]
    forced = int(rng.integers(len(facts)))
    plan: list[tuple[int, int, Band]] = []  # (offset, fact, band)
    for i, fact in enumerate(facts):
        if i != forced and not _percent(rng, spec.stories.ask_percent):
            continue
        # behind the statement; the value's distance with " <question>" inserted at offset b
        behind = [b for b in bounds if b >= starts[i] + len(fact.statement)]
        shift = fact.question_at - values[i]
        fits = [[b for b in behind if band.lo <= b + shift <= band.hi] for band in bands]
        weights = [band.weight if fit else 0 for band, fit in zip(bands, fits, strict=True)]
        if sum(weights) == 0:
            continue
        chosen = _weighted(rng, weights)
        plan.append((fits[chosen][int(rng.integers(len(fits[chosen])))], i, bands[chosen]))
    if not plan:
        return None
    plan.sort(key=lambda item: (item[0], item[1]))
    made, placed = _insert(text, [(offset, facts[i].question) for offset, i, _ in plan])
    answers: list[Answer] = []
    for (_, i, band), start in zip(plan, placed, strict=True):
        fact = facts[i]
        # the stated value moved by every question inserted before it
        stated = values[i] + sum(len(facts[j].question) + 1 for o, j, _ in plan if o <= values[i])
        answer = start + fact.question_at
        distance = answer - stated - 1
        if not band.lo <= distance <= band.hi:
            return None
        n = len(fact.value)
        assert made[stated : stated + n] == fact.value == made[answer : answer + n]
        # body offset v is token v + 1, and the position whose target is token v + 1 is v
        answers.append(Answer(answer, n, distance, fact.relation, band.name))
    return made, tuple(answers)


def kv_document(
    spec: StoryFactsSpec, split: Split, index: int, names: Sequence[str] | None = None
) -> Document:
    """Fact sheet ``index`` of ``split``: facts in a drawn order, each asked fact's question
    somewhere behind its statement, sentences joined by a space or a line break."""
    names = spec.vocabulary()[split] if names is None else names
    rng = np.random.default_rng([spec.seed, SPLITS.index(split), 1, index])
    n = int(rng.integers(spec.kv.facts[0], spec.kv.facts[1] + 1))
    facts = _draw_facts(spec, names, spec.owners[split], rng, n)
    forced = int(rng.integers(n))
    # (time, 0 statement | 1 question, fact): a question comes 0 to n - 1 statements later
    items = [(2 * i, 0, i) for i in range(n)]
    for i in range(n):
        if i == forced or _percent(rng, spec.kv.ask_percent):
            items.append((2 * (i + int(rng.integers(n))) + 1, 1, i))
    items.sort()
    parts: list[bytes] = []
    length = 0
    stated: dict[int, int] = {}
    answers: list[Answer] = []
    for _, is_question, i in items:
        fact = facts[i]
        if parts:
            parts.append(bytes([spec.newline if _percent(rng, spec.kv.newline_percent) else 32]))
            length += 1
        if is_question:
            answer = length + fact.question_at
            answers.append(
                Answer(answer, len(fact.value), answer - stated[i] - 1, fact.relation, "")
            )
            parts.append(fact.question)
        else:
            stated[i] = length + fact.statement_at
            parts.append(fact.statement)
        length += len(parts[-1])
    return Document("kv", index, _tokens(b"".join(parts)), tuple(answers), n)


def answers_of(
    loss_mask: npt.NDArray[np.bool_], deps: npt.NDArray[Any]
) -> list[tuple[int, int, int]]:
    """``(position, length, distance)`` of every answer of one document, read back from its
    shard arrays: a run of consecutive loss positions with one dependency above 1."""
    positions = np.flatnonzero(loss_mask)
    deps = np.asarray(deps, dtype=np.int64)
    marked = np.flatnonzero(deps > 1)
    out: list[tuple[int, int, int]] = []
    for j in marked.tolist():
        position, distance = int(positions[j]), int(deps[j])
        if out and out[-1][0] + out[-1][1] == position and out[-1][2] == distance:
            out[-1] = (out[-1][0], out[-1][1] + 1, distance)
        else:
            out.append((position, 1, distance))
    return out


# --------------------------------------------------------------------------------------------------
# Sources and shards
# --------------------------------------------------------------------------------------------------


def _stories(raw_dir: Path, spec: StoryFactsSpec, source: SourceSpec) -> list[bytes]:
    """The distinct story bodies of a source in file order (the rules of :mod:`hsn.data.text`:
    carriage returns removed, stripped, bytes below 32 mapped), those of ``min_story_bytes`` to
    ``stories.max_bytes`` bytes."""
    path = raw_dir / source.file
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != source.sha256:
        raise ValueError(f"{path}: sha256 {actual} != spec sha256 {source.sha256}")
    pieces = data.split(spec.separator.encode())
    if source.drop_first:
        pieces = pieces[1:]
    if source.drop_last:
        pieces = pieces[:-1]
    table = bytearray(range(256))
    for value in range(N_MARKERS):
        table[value] = _SPACE
    table[ord("\n")] = spec.newline
    table[ord("\t")] = spec.tab
    seen: set[bytes] = set()
    out: list[bytes] = []
    for piece in pieces:
        body = piece.replace(b"\r", b"").strip()
        if not spec.min_story_bytes <= len(body) <= spec.stories.max_bytes:
            continue
        body = body.translate(bytes(table))
        digest = hashlib.blake2b(body, digest_size=16).digest()
        if digest not in seen:
            seen.add(digest)
            out.append(body)
    return out


def _assign(
    spec: StoryFactsSpec, split: Split, pool: Sequence[bytes], first: int
) -> tuple[list[list[bytes]], int]:
    """The stories of every story document of ``split``: consecutive stories of ``pool`` from
    ``first``, and the index behind the last one used. Document ``i`` takes its drawn number
    of stories, fewer where the next one would pass ``stories.max_bytes``. A document's kind
    plays no part."""
    params = spec.stories
    rng = np.random.default_rng([spec.seed, SPLITS.index(split), 2])
    counts = [1 + _weighted(rng, params.count_weights) for _ in range(params.n[split])]
    taken: list[list[bytes]] = []
    cursor = first
    for index, count in enumerate(counts):
        if cursor >= len(pool):
            raise ValueError(f"{split}: the source ran out of stories at document {index}")
        stories = [pool[cursor]]
        size = len(pool[cursor])
        cursor += 1
        while len(stories) < count and cursor < len(pool):
            size += params.gap + len(pool[cursor])
            if size > params.max_bytes:
                break
            stories.append(pool[cursor])
            cursor += 1
        taken.append(stories)
    return taken, cursor


def _story_documents(
    spec: StoryFactsSpec, split: Split, pool: Sequence[bytes], first: int
) -> tuple[list[Document], int]:
    """The story documents of ``split`` (:func:`_assign`) and the index behind their stories."""
    names = spec.vocabulary()[split]
    taken, cursor = _assign(spec, split, pool, first)
    docs = [story_document(spec, split, i, stories, names) for i, stories in enumerate(taken)]
    return docs, cursor


def _write(
    out_dir: Path, spec: StoryFactsSpec, split: Split, task: str, docs: Sequence[Document]
) -> ShardRecord:
    n = len(docs)
    lengths = np.fromiter((doc.tokens.shape[0] for doc in docs), dtype=np.int64, count=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    loss_mask = np.ones(int(offsets[-1]), dtype=np.bool_)
    loss_mask[offsets[1:] - 1] = False  # the last position's target is PAD
    arrays: dict[str, npt.NDArray[Any]] = {
        "tokens": (
            np.concatenate([doc.tokens for doc in docs]) if docs else np.zeros(0, dtype=np.uint8)
        ),
        "loss_mask": loss_mask,
        "deps": (
            np.concatenate([doc.deps for doc in docs]) if docs else np.zeros(0, dtype=np.int32)
        ),
        "offsets": offsets,
        "index": np.asarray([doc.index for doc in docs], dtype=np.int64),
    }
    scalars: dict[str, Any] = {
        "format": SHARD_FORMAT,
        "task": task,
        "variant": spec.variant,
        "split": split,
        "family": FAMILY[task],
        "context_id": spec.context_id,
        "unroll": spec.unroll,
    }
    file = shard_file_name(task, spec.variant, split)
    payload: dict[str, Any] = {
        **arrays,
        **{key: np.asarray(value) for key, value in scalars.items()},
    }
    np.savez(out_dir / file, **payload)
    return ShardRecord(
        task=task,
        variant=spec.variant,
        split=split,
        family=FAMILY[task],
        context_id=spec.context_id,
        file=file,
        n_docs=n,
        n_tokens=int(offsets[-1]),
        n_loss=int(arrays["deps"].shape[0]),
        hash=shard_hash(arrays, scalars),
    )


def build_storyfacts(
    spec: StoryFactsSpec, raw_dir: Path, out_dir: Path
) -> tuple[SplitManifest, dict[str, int]]:
    """Write the twelve shards and ``manifest.json`` of ``spec`` into ``out_dir``; returns the
    manifest and the counts of what was read and used.

    Test documents take the held-out source's stories from its first, dev documents the
    stories behind those (so with TinyStories' validation file the stories of
    ``tinystories_v1``'s dev and test splits, its first 4,000, lie in this dataset's test
    split, which is scored once, not in its dev split); train documents take the train
    source's stories from its first, without any story that a dev or test document uses. The
    n-gram overlap between splits is not measured (story text repeats phrases;
    ``tinystories_v1``'s manifest has the counts of the same sources)."""
    held = _stories(raw_dir, spec, spec.heldout)
    by_split: dict[Split, list[Document]] = {}
    by_split["test"], test_end = _story_documents(spec, "test", held, 0)
    by_split["dev"], dev_end = _story_documents(spec, "dev", held, test_end)
    used = set(held[:dev_end])
    pool = [story for story in _stories(raw_dir, spec, spec.train) if story not in used]
    by_split["train"], train_end = _story_documents(spec, "train", pool, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = spec.vocabulary()
    records: dict[tuple[str, Split], ShardRecord] = {}
    for split in SPLITS:
        docs = by_split[split]
        sheets = [kv_document(spec, split, i, names[split]) for i in range(spec.kv.n[split])]
        for task in TASKS:
            chosen = sheets if task == "kv" else [doc for doc in docs if doc.task == task]
            records[task, split] = _write(out_dir, spec, split, task, chosen)
    ordered = [records[task, split] for task in TASKS for split in SPLITS]
    split_hashes: dict[str, str] = {}
    for split in SPLITS:
        digest = hashlib.blake2b(digest_size=32)
        for record in ordered:
            if record.split == split:
                digest.update(bytes.fromhex(record.hash))
        split_hashes[split] = digest.hexdigest()
    manifest = SplitManifest(
        name=spec.name,
        spec_hash=spec.spec_hash(),
        unroll=spec.unroll,
        hsn_version=hsn.__version__,
        numpy_version=np.__version__,
        shards=ordered,
        split_hashes=split_hashes,
        leakage=[],
    )
    manifest.write(out_dir / MANIFEST_NAME)
    counts = {
        "heldout_stories": len(held),
        "test_stories_used": test_end,
        "dev_stories_used": dev_end - test_end,
        "train_stories": len(pool),
        "train_stories_used": train_end,
    }
    return manifest, counts


def describe(out_dir: Path, edges: Sequence[int] = (64, 256, 1024)) -> dict[str, Any]:
    """Shape statistics of a built dataset, read from its shards: per split and task the
    documents, bytes, median length and answers, and the answers by distance (``edges`` cut
    the distances into ``<e0, e0..e1-1, …, >=e_last``)."""
    manifest = SplitManifest.load(out_dir / MANIFEST_NAME)
    labels = [f"lt{edges[0]}"]
    labels += [f"{lo}_{hi - 1}" for lo, hi in itertools.pairwise(edges)]
    labels.append(f"ge{edges[-1]}")
    out: dict[str, Any] = {"name": manifest.name, "splits": {}}
    for record in manifest.shards:
        arrays, _ = read_shard(out_dir / record.file)
        lengths = np.diff(arrays["offsets"])
        answers = answers_of(arrays["loss_mask"], arrays["deps"])
        distances = np.asarray([distance for _, _, distance in answers], dtype=np.int64)
        buckets = np.bincount(np.digitize(distances, list(edges)), minlength=len(labels))
        doc_of = np.searchsorted(arrays["offsets"], [p for p, _, _ in answers], side="right")
        out["splits"].setdefault(record.split, {})[record.task] = {
            "documents": record.n_docs,
            "bytes": record.n_tokens,
            "median_bytes": int(np.median(lengths)) if record.n_docs else 0,
            "answers": len(answers),
            "answer_bytes": int((arrays["deps"] > 1).sum()),
            "documents_with_answers": int(np.unique(doc_of).shape[0]),
            "answers_by_distance": dict(zip(labels, buckets.tolist(), strict=True)),
        }
    return out


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def _print_description(table: Mapping[str, Any]) -> None:
    for split, tasks in table["splits"].items():
        for task, row in tasks.items():
            by_distance = "  ".join(f"{k} {v}" for k, v in row["answers_by_distance"].items())
            print(
                f"  {split}/{task}: docs={row['documents']} bytes={row['bytes']} "
                f"median={row['median_bytes']} answers={row['answers']} "
                f"(in {row['documents_with_answers']} documents; {by_distance})"
            )


def _cmd_build(args: argparse.Namespace) -> int:
    spec = load_storyfacts_spec(args.spec)
    started = time.perf_counter()
    manifest, counts = build_storyfacts(spec, Path(args.raw), Path(args.out))
    print(f"built {spec.name} in {time.perf_counter() - started:.1f} s")
    for name, value in counts.items():
        print(f"  {name}: {value}")
    for shard in manifest.shards:
        print(
            f"  {shard.task}/{shard.variant}/{shard.split}: "
            f"docs={shard.n_docs} tokens={shard.n_tokens} {shard.hash}"
        )
    status = 0
    if args.write_golden:
        golden = Path(args.write_golden)
        golden.write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote golden manifest {golden}")
    if args.golden:
        if Path(args.golden).read_text(encoding="utf-8") != manifest.to_json():
            print(f"ERROR: manifest differs from the golden file {args.golden}", file=sys.stderr)
            status = 1
        else:
            print(f"manifest matches the golden file {args.golden}")
    return status


def _cmd_describe(args: argparse.Namespace) -> int:
    table = describe(Path(args.out), [int(edge) for edge in args.edges.split(",")])
    print(table["name"])
    _print_description(table)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hsn.data.storyfacts", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the shards and the manifest of the dataset")
    build.add_argument("--spec", required=True)
    build.add_argument("--raw", required=True, help="folder holding the spec's source files")
    build.add_argument("--out", required=True)
    build.add_argument("--golden", default=None, help="manifest the build must reproduce")
    build.add_argument("--write-golden", default=None)
    build.set_defaults(func=_cmd_build)
    shape = sub.add_parser("describe", help="shape statistics of a built dataset")
    shape.add_argument("--out", required=True, help="the dataset directory")
    shape.add_argument("--edges", default="64,256,1024", help="distance edges, comma separated")
    shape.add_argument("--json", default=None, help="also write the table here")
    shape.set_defaults(func=_cmd_describe)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
