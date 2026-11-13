"""Synthetic task generators (P0.4): deterministic byte-level documents for five task families.

Every example is one *document*: a byte stream ``BOS … EOS`` with next-symbol targets. The five
families (plan P0.4 "Context you need") are key–value state (``kv``), delayed relevance
(``delayed``), balanced brackets (``brackets``), multi-digit addition with the least significant
digit first (``arithmetic``) and copy / reverse / sort of short lists (``algorithmic``).

**Symbols.** Bytes ``0..15`` are reserved control symbols (``PAD=0, BOS=1, EOS=2, QRY=3, SEP=4,
NONE=5``; the rest unused). Bytes ``16..31`` hold per-family *marker* symbols (``INS``, ``OVR``,
``DEL``, ``PLUS``, …) and data alphabets start at ``32``. Every alphabet and marker value is a
parameter of the task spec (``configs/data/synthetic_v1.yaml``), validated to be disjoint within a
family; the reserved values never change (chunk "Do not"). Context symbols are the separate id
space ``0..15``: a synthetic task uses ``ctx = context_id`` at every position, one id per family.

**Targets and the loss mask.** ``targets[t] = tokens[t + 1]`` (the next symbol) and ``PAD`` at the
last position, where no next symbol exists. ``loss_mask[t]`` is True only where the family defines
the next symbol as a task answer (a queried value, a bracket type after a ``QRY`` marker, a sum
digit, a list element, the terminating ``SEP``); everywhere else the next symbol is a free choice of
the generator and would only inflate accuracy with trivial tokens (chunk pitfall).

**Dependency length.** For a loss position ``t`` the dependency distance is ``t - s`` where ``s`` is
the position of the earliest input symbol the answer directly depends on (the value written by the
defining operation, the stated fact, the matching open bracket, the input digit ``a_i``, the input
list element). ``meta["deps"]`` holds that distance for every True position of ``loss_mask`` in
order and ``meta["dep_len"]`` is their maximum. Each emitter builds its document around one
*anchor* dependency of exactly ``D`` positions, drawn from ``dep_len_range``, and keeps every other
dependency at most ``D``, so ``dep_len == D`` and ``in_range = D <= unroll``: the dependency fits
one TBPTT window of ``unroll`` positions (in range) or exceeds it (out of range, for length
extrapolation).

**Determinism.** ``generate(spec, split, index)`` is a pure function. Its random stream is
``numpy.random.default_rng`` seeded from ``blake2b(spec.seed, split, index)`` (never Python's salted
``hash``); nothing is shared between examples or splits. Only integer draws are used. The split
enters a generator in exactly one place: the per-split key alphabet of the ``kv`` and ``delayed``
families (disjoint across train / dev / test, so no key seen at evaluation time was ever written
in training). Every other parameter is identical for every split.

**Reference solvers.** Each family also has a *solver* that re-derives the loss positions, targets
and dependency distances by interpreting the token stream (a dictionary, a stack, Python ``int``
arithmetic, ``sorted``), independently of the emitter's construction; :func:`reference_targets`
exposes it and ``tests/test_p0_4_synthetic.py`` compares the two.

CLI: ``python -m hsn.data.synthetic build --spec configs/data/synthetic_v1.yaml --out
data/synthetic_v1`` writes the shards and manifest (:mod:`hsn.data.splits`); ``verify``
regenerates and compares; ``bench`` measures examples per second.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hsn.config import Split

__all__ = [
    "BOS",
    "EOS",
    "FAMILIES",
    "NONE",
    "N_MARKERS",
    "N_RESERVED",
    "PAD",
    "QRY",
    "SEP",
    "SPLITS",
    "AlgorithmicParams",
    "ArithmeticParams",
    "BracketsParams",
    "DatasetSpec",
    "DelayedParams",
    "Example",
    "Family",
    "KVParams",
    "TaskSpec",
    "Variant",
    "check_example",
    "example_seed",
    "generate",
    "load_dataset_spec",
    "main",
    "reference_targets",
]

# --------------------------------------------------------------------------------------------------
# Symbols (plan P0.4: byte values 0-15 are reserved control symbols; never change them)
# --------------------------------------------------------------------------------------------------

PAD = 0
BOS = 1
EOS = 2
QRY = 3
SEP = 4
NONE = 5
N_RESERVED = 16  # bytes 0..15: control symbols
N_MARKERS = 32  # bytes 16..31: per-family marker symbols; data alphabets start here

Family = Literal["kv", "delayed", "brackets", "arithmetic", "algorithmic"]
Variant = Literal["in_range", "out_of_range"]
FAMILIES: tuple[Family, ...] = ("kv", "delayed", "brackets", "arithmetic", "algorithmic")
SPLITS: tuple[Split, ...] = ("train", "dev", "test")

ByteRange = tuple[int, int]  # half-open [lo, hi)
U8 = npt.NDArray[np.uint8]
Bool = npt.NDArray[np.bool_]
I64 = npt.NDArray[np.int64]

_META_KEYS = (
    "task",
    "variant",
    "family",
    "split",
    "index",
    "doc_id",
    "dep_len",
    "in_range",
    "deps",
)
_SIXTEEN = 1 << 16  # integer thresholds replace float comparisons in every random decision


# --------------------------------------------------------------------------------------------------
# Example
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Example:
    """One document. All arrays have length T; the arrays are made read-only on construction.

    ``meta`` carries ``task`` (spec name), ``variant``, ``family``, ``split``, ``index``, ``doc_id``
    (``"<task>:<variant>:<split>:<index>"``), ``dep_len``, ``in_range`` and ``deps`` (one dependency
    distance per True entry of ``loss_mask``, in position order).
    """

    tokens: U8  # uint8 [T]
    targets: U8  # uint8 [T]: next symbol; PAD at the last position
    loss_mask: Bool  # bool [T]: True where the target counts
    context: U8  # uint8 [T]: context symbol id (0..15)
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        fields: tuple[tuple[str, npt.NDArray[Any], type[np.generic]], ...] = (
            ("tokens", self.tokens, np.uint8),
            ("targets", self.targets, np.uint8),
            ("loss_mask", self.loss_mask, np.bool_),
            ("context", self.context, np.uint8),
        )
        for name, array, dtype in fields:
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.ndim != 1:
                raise TypeError(f"Example.{name} must be a 1-d {np.dtype(dtype).name} array")
            if array.shape != self.tokens.shape:
                raise ValueError(
                    f"Example.{name} has length {array.shape[0]}, tokens has {self.tokens.shape[0]}"
                )
            array.flags.writeable = False
        missing = [key for key in _META_KEYS if key not in self.meta]
        if missing:
            raise ValueError(f"Example.meta is missing {missing}")
        if len(self.meta["deps"]) != int(self.loss_mask.sum()):
            raise ValueError("Example.meta['deps'] must have one entry per True loss_mask position")

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


def _next_symbol_targets(tokens: U8) -> U8:
    targets = np.empty_like(tokens)
    targets[:-1] = tokens[1:]
    targets[-1] = PAD
    return targets


# --------------------------------------------------------------------------------------------------
# Per-family parameters (the YAML `params` mapping, validated)
# --------------------------------------------------------------------------------------------------


def _check_range(label: str, byte_range: ByteRange) -> None:
    lo, hi = byte_range
    if not (N_MARKERS <= lo < hi <= 256):
        raise ValueError(
            f"{label}: byte range [{lo}, {hi}) must satisfy {N_MARKERS} <= lo < hi <= 256 "
            f"(bytes below {N_MARKERS} are control and marker symbols)"
        )


def _check_disjoint(
    ranges: Sequence[tuple[str, ByteRange]], markers: Sequence[tuple[str, int]]
) -> None:
    """Every alphabet range and marker byte of one family must be distinct."""
    owner: dict[int, str] = {}
    for name, (lo, hi) in ranges:
        _check_range(name, (lo, hi))
        for byte in range(lo, hi):
            if byte in owner:
                raise ValueError(f"{name} and {owner[byte]} both use byte {byte}")
            owner[byte] = name
    for name, byte in markers:
        if not (N_RESERVED <= byte < N_MARKERS):
            raise ValueError(f"{name}={byte} must be a marker byte in [{N_RESERVED}, {N_MARKERS})")
        if byte in owner:
            raise ValueError(f"{name} and {owner[byte]} both use byte {byte}")
        owner[byte] = name


def _check_keys(keys: dict[Split, ByteRange]) -> list[tuple[str, ByteRange]]:
    missing = [split for split in SPLITS if split not in keys]
    if missing:
        raise ValueError(f"keys must name every split; missing {missing}")
    return [(f"keys[{split}]", keys[split]) for split in SPLITS]


class _Params(BaseModel):
    """Base of every family parameter model: unknown keys raise, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class KVParams(_Params):
    """``kv``: INS / OVR / DEL operations over keys, distractors, queries of the current value."""

    markers: dict[Literal["ins", "ovr", "del"], int] = {"ins": 16, "ovr": 17, "del": 18}
    keys: dict[Split, ByteRange] = {"train": (64, 112), "dev": (112, 128), "test": (128, 144)}
    values: ByteRange = (144, 192)
    distractors: ByteRange = (192, 256)
    n_keys: int = Field(default=8, ge=2)  # keys a document may touch (the anchor key and others)
    op_rate: float = Field(default=0.15, gt=0.0, le=1.0)  # P(a middle position starts an op)
    p_del: float = Field(default=0.25, ge=0.0, le=1.0)  # DEL (else OVR) when the key is present
    anchor: dict[Literal["ins", "ovr", "del"], float] = {"ins": 0.5, "ovr": 0.3, "del": 0.2}
    max_queries: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, mapping in (("markers", self.markers), ("anchor", self.anchor)):
            missing = [kind for kind in ("ins", "ovr", "del") if kind not in mapping]
            if missing:
                raise ValueError(f"{name} must name ins, ovr and del; missing {missing}")
        if any(weight < 0.0 for weight in self.anchor.values()) or sum(self.anchor.values()) <= 0:
            raise ValueError(f"anchor weights must be >= 0 with a positive sum: {self.anchor}")
        _check_disjoint(
            [*_check_keys(self.keys), ("values", self.values), ("distractors", self.distractors)],
            [(f"markers[{kind}]", byte) for kind, byte in self.markers.items()],
        )
        return self


class DelayedParams(_Params):
    """``delayed``: facts ``k v`` stated early, queried after a filler of controlled length."""

    keys: dict[Split, ByteRange] = {"train": (64, 112), "dev": (112, 128), "test": (128, 144)}
    values: ByteRange = (144, 192)
    filler: ByteRange = (192, 256)
    max_facts: int = Field(default=6, ge=1)
    max_queries: int = Field(default=4, ge=1)
    # Bytes per key (12 Nov 2026, the multi-byte address of amendment A9). 1: synthetic_v1's
    # documents, symbol for symbol. L > 1: a fact is ``k_1 .. k_L v`` and a query ``QRY k_1 ..
    # k_L``; a document's keys are cells of a small grid (the fewest symbols per key position
    # that give ``n`` distinct keys), so keys of one document share their bytes and no single
    # byte identifies a key.
    key_len: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_disjoint(
            [*_check_keys(self.keys), ("values", self.values), ("filler", self.filler)], []
        )
        return self

    def grid_side(self, n: int) -> int:
        """The fewest symbols per key position that give ``n`` distinct keys of ``key_len``."""
        side = 1
        while side**self.key_len < n:
            side += 1
        return side


class BracketsParams(_Params):
    """``brackets``: Dyck words over ``n_pairs`` bracket types with bounded depth."""

    opens: ByteRange = (32, 36)
    closes: ByteRange = (36, 40)
    verdict: dict[Literal["valid", "invalid"], int] = {"valid": 24, "invalid": 25}
    max_depth: int = Field(default=8, ge=2)  # the anchor pair takes one level
    p_query: float = Field(default=0.25, ge=0.0, le=1.0)  # P(a close is marked with QRY)
    p_invalid: float = Field(default=0.5, ge=0.0, le=1.0)  # P(one close gets a wrong type)
    block_max: int = Field(default=48, ge=3)  # longest balanced block before the anchor

    @property
    def n_pairs(self) -> int:
        return self.opens[1] - self.opens[0]

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = [kind for kind in ("valid", "invalid") if kind not in self.verdict]
        if missing:
            raise ValueError(f"verdict must name valid and invalid; missing {missing}")
        _check_disjoint(
            [("opens", self.opens), ("closes", self.closes)],
            [(f"verdict[{kind}]", byte) for kind, byte in self.verdict.items()],
        )
        if self.n_pairs < 2 or self.closes[1] - self.closes[0] != self.n_pairs:
            raise ValueError(
                f"opens {self.opens} and closes {self.closes} must have the same size >= 2"
            )
        return self


class ArithmeticParams(_Params):
    """``arithmetic``: ``a PLUS b FILLER EQ s SEP`` items, digits least significant first."""

    digits: ByteRange = (48, 58)  # exactly ten symbols, '0'..'9'
    markers: dict[Literal["plus", "eq"], int] = {"plus": 19, "eq": 20}
    filler: ByteRange = (192, 256)
    digits_max: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = [kind for kind in ("plus", "eq") if kind not in self.markers]
        if missing:
            raise ValueError(f"markers must name plus and eq; missing {missing}")
        _check_disjoint(
            [("digits", self.digits), ("filler", self.filler)],
            [(f"markers[{kind}]", byte) for kind, byte in self.markers.items()],
        )
        if self.digits[1] - self.digits[0] != 10:
            raise ValueError(f"digits {self.digits} must span exactly ten symbols")
        return self


class AlgorithmicParams(_Params):
    """``algorithmic``: ``OP x_1..x_n FILLER SEP y_1..y_n SEP`` with OP in copy / reverse / sort."""

    markers: dict[Literal["copy", "reverse", "sort"], int] = {"copy": 21, "reverse": 22, "sort": 23}
    items: ByteRange = (64, 128)
    filler: ByteRange = (192, 256)
    n_min: int = Field(default=2, ge=1)
    n_max: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = [kind for kind in ("copy", "reverse", "sort") if kind not in self.markers]
        if missing:
            raise ValueError(f"markers must name copy, reverse and sort; missing {missing}")
        _check_disjoint(
            [("items", self.items), ("filler", self.filler)],
            [(f"markers[{kind}]", byte) for kind, byte in self.markers.items()],
        )
        if self.n_min > self.n_max:
            raise ValueError(f"n_min={self.n_min} must be <= n_max={self.n_max}")
        return self


# --------------------------------------------------------------------------------------------------
# Task and dataset specs
# --------------------------------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """One task of a dataset: a family, a variant, split sizes, ranges, a seed and parameters.

    ``name`` is the task name that ``data.tasks`` and ``metrics_freeze.json`` refer to (for example
    ``kv``); ``variant`` distinguishes the in-range and out-of-range dependency ranges of the same
    task, so ``(name, variant)`` identifies a spec. ``unroll`` is the window length that defines
    ``in_range``; :class:`DatasetSpec` propagates its own value. ``params`` is validated into the
    family's parameter model (:attr:`typed_params`). Derive specs with
    ``TaskSpec.model_validate({**spec.model_dump(), ...})``: ``model_copy`` skips validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    variant: Variant
    family: Family
    context_id: int = Field(ge=0, lt=N_RESERVED)
    n_train: int = Field(ge=0)
    n_dev: int = Field(ge=0)
    n_test: int = Field(ge=0)
    length_range: tuple[int, int]
    dep_len_range: tuple[int, int]
    seed: int = Field(ge=0, lt=2**63)
    unroll: int = Field(default=256, ge=1)
    params: dict[str, Any] = {}

    @functools.cached_property
    def typed_params(self) -> _Params:
        """``params`` validated into the family's model (cached per instance)."""
        typed: _Params = _FAMILIES[self.family].params_model.model_validate(self.params)
        return typed

    @property
    def key(self) -> str:
        return f"{self.name}/{self.variant}"

    def n_examples(self, split: Split) -> int:
        _check_split(split)
        return {"train": self.n_train, "dev": self.n_dev, "test": self.n_test}[split]

    @model_validator(mode="after")
    def _check(self) -> Self:
        for label, (lo, hi) in (
            ("length_range", self.length_range),
            ("dep_len_range", self.dep_len_range),
        ):
            if not 1 <= lo <= hi:
                raise ValueError(f"{self.key}: {label}={[lo, hi]} must satisfy 1 <= lo <= hi")
        dep_lo, dep_hi = self.dep_len_range
        if self.variant == "in_range" and dep_hi > self.unroll:
            raise ValueError(
                f"{self.key}: in_range needs dep_len_range[1]={dep_hi} <= unroll={self.unroll}"
            )
        if self.variant == "out_of_range" and dep_lo <= self.unroll:
            raise ValueError(
                f"{self.key}: out_of_range needs dep_len_range[0]={dep_lo} > unroll={self.unroll}"
            )
        generator = _FAMILIES[self.family]
        try:
            typed = self.typed_params
        except ValidationError as exc:
            raise ValueError(f"{self.key}: params: {exc}") from exc
        generator.validate(self, typed)
        return self


class DatasetSpec(BaseModel):
    """A named list of tasks sharing one ``unroll`` (``configs/data/<name>.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    unroll: int = Field(ge=1)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _propagate_unroll(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("tasks"), list) and "unroll" in data:
            tasks = []
            for task in data["tasks"]:
                if isinstance(task, dict) and "unroll" not in task:
                    task = {**task, "unroll": data["unroll"]}
                tasks.append(task)
            data = {**data, "tasks": tasks}
        return data

    @model_validator(mode="after")
    def _check(self) -> Self:
        keys = [task.key for task in self.tasks]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.name}: duplicate task (name, variant): {keys}")
        seeds = [task.seed for task in self.tasks]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{self.name}: every task needs a distinct seed: {seeds}")
        bad = [task.key for task in self.tasks if task.unroll != self.unroll]
        if bad:
            raise ValueError(
                f"{self.name}: tasks {bad} do not use the dataset unroll {self.unroll}"
            )
        return self

    def task(self, name: str, variant: Variant) -> TaskSpec:
        for task in self.tasks:
            if task.name == name and task.variant == variant:
                return task
        raise KeyError(f"{self.name} has no task {name}/{variant}")

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def spec_hash(self) -> str:
        """blake2b (32-byte digest, hex) of the canonical JSON; independent of YAML key order."""
        return hashlib.blake2b(self.canonical_json().encode(), digest_size=32).hexdigest()


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Load and validate a dataset spec YAML (top level: ``name``, ``unroll``, ``tasks``)."""
    spec_path = Path(path)
    try:
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{spec_path}: invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{spec_path}: top level must be a mapping")
    try:
        return DatasetSpec.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"{spec_path}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Random helpers (integer draws only; every decision compares an integer against a threshold)
# --------------------------------------------------------------------------------------------------


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")


def example_seed(seed: int, split: Split, index: int) -> int:
    """``blake2b(spec.seed, split, index)`` as a 256-bit integer (the chunk's seeding rule)."""
    digest = hashlib.blake2b(f"{seed}\0{split}\0{index}".encode(), digest_size=32).digest()
    return int.from_bytes(digest, "little")


def _rng(seed: int, split: Split, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(example_seed(seed, split, index)))


def _threshold(probability: float) -> int:
    return int(round(probability * _SIXTEEN))


def _bits(rng: np.random.Generator, n: int) -> list[int]:
    values: list[int] = rng.integers(0, _SIXTEEN, size=n).tolist()
    return values


def _draw(rng: np.random.Generator, byte_range: ByteRange, n: int) -> U8:
    lo, hi = byte_range
    out: U8 = rng.integers(lo, hi, size=n, dtype=np.uint8)
    return out


def _weighted(rng: np.random.Generator, weights: Sequence[float]) -> int:
    """Index drawn with probability proportional to ``weights`` (integer cumulative thresholds)."""
    total = float(sum(weights))
    scaled = np.cumsum([int(round(w / total * (1 << 30))) for w in weights], dtype=np.int64)
    draw = int(rng.integers(0, int(scaled[-1])))
    return int(np.searchsorted(scaled, draw, side="right"))


def _byte_table(ranges: Sequence[ByteRange]) -> list[bool]:
    table = [False] * 256
    for lo, hi in ranges:
        for byte in range(lo, hi):
            table[byte] = True
    return table


# --------------------------------------------------------------------------------------------------
# Family generators: emitter + reference solver
# --------------------------------------------------------------------------------------------------

Emitted = tuple[
    U8, list[int], list[int]
]  # tokens, loss positions (ascending), dependency distances
Solved = tuple[I64, U8, I64]  # positions, expected targets, dependency distances


class _Generator[P: _Params]:
    family: Family
    params_model: type[P]

    def params(self, spec: TaskSpec) -> P:
        typed = spec.typed_params
        if not isinstance(typed, self.params_model):
            raise TypeError(f"{spec.key}: params are {type(typed).__name__}, not {self.family}")
        return typed

    def validate(self, spec: TaskSpec, params: P) -> None:
        """Family-specific feasibility: ranges wide enough for the anchor construction."""
        dep_lo, dep_hi = spec.dep_len_range
        if dep_lo < self.min_dep(params):
            raise ValueError(
                f"{spec.key}: dep_len_range[0]={dep_lo} must be >= {self.min_dep(params)} "
                f"for family {self.family}"
            )
        needed = self.min_length(params, dep_hi)
        if spec.length_range[1] < needed:
            raise ValueError(
                f"{spec.key}: length_range[1]={spec.length_range[1]} is too short: the longest "
                f"anchor dependency {dep_hi} needs documents of at least {needed} symbols"
            )

    def min_dep(self, params: P) -> int:
        return 1

    def min_length(self, params: P, dep: int) -> int:
        raise NotImplementedError

    def emit(
        self,
        params: P,
        split: Split,
        rng: np.random.Generator,
        length: int,
        dep: int,
    ) -> Emitted:
        raise NotImplementedError

    def solve(self, params: P, tokens: U8) -> Solved:
        raise NotImplementedError


def _assemble(parts: Sequence[Sequence[int] | U8]) -> U8:
    tokens: U8 = np.concatenate([np.asarray(part, dtype=np.uint8) for part in parts])
    return tokens


def _sorted_solution(positions: list[int], targets: list[int], deps: list[int]) -> Solved:
    order = np.argsort(np.asarray(positions, dtype=np.int64), kind="stable")
    return (
        np.asarray(positions, dtype=np.int64)[order],
        np.asarray(targets, dtype=np.uint8)[order],
        np.asarray(deps, dtype=np.int64)[order],
    )


def _frame(tokens: list[int]) -> None:
    if len(tokens) < 2 or tokens[0] != BOS or tokens[-1] != EOS:
        raise ValueError("document must be BOS ... EOS")


# ---- kv ------------------------------------------------------------------------------------------


class _KV(_Generator[KVParams]):
    """``BOS prefix anchor middle SEP (QRY k a)* EOS``.

    The prefix holds distractors and, for an OVR or DEL anchor, a stale ``INS k* v0``; the anchor
    is the *last* operation on the anchor key ``k*``; the middle holds exactly ``D - 3`` symbols of
    operations on other keys (INS when absent, OVR or DEL when present) mixed with distractors;
    the first query asks ``k*`` (answer: the anchor's value, or NONE after DEL) at distance D from
    the anchor's last symbol; further queries ask keys touched in the middle. ``T == L`` exactly.
    """

    family: Family = "kv"
    params_model = KVParams
    _KINDS: tuple[Literal["ins", "ovr", "del"], ...] = ("ins", "ovr", "del")

    def validate(self, spec: TaskSpec, params: KVParams) -> None:
        super().validate(spec, params)
        for split in SPLITS:
            lo, hi = params.keys[split]
            if hi - lo < params.n_keys:
                raise ValueError(
                    f"{spec.key}: keys[{split}] has {hi - lo} symbols but n_keys={params.n_keys}"
                )

    def min_dep(self, params: KVParams) -> int:
        return 3

    def min_length(self, params: KVParams, dep: int) -> int:
        # BOS + stale INS (3) + anchor (3) + middle (dep - 3) + SEP + anchor query (3) + EOS
        return dep + 9

    def _middle(
        self,
        params: KVParams,
        rng: np.random.Generator,
        others: list[int],
        n: int,
    ) -> tuple[U8, dict[int, int], dict[int, int], list[int]]:
        """Distractors with operations sprinkled in; returns tokens, state, last-op ends, order."""
        ins, ovr, dele = params.markers["ins"], params.markers["ovr"], params.markers["del"]
        out = _draw(rng, params.distractors, n)
        decisions = rng.integers(0, _SIXTEEN, size=n)
        starts: list[int] = np.flatnonzero(decisions < _threshold(params.op_rate)).tolist()
        m = len(starts)
        key_idx: list[int] = rng.integers(0, len(others), size=m).tolist()
        values: list[int] = rng.integers(params.values[0], params.values[1], size=m).tolist()
        del_bits = _bits(rng, m)
        del_thr = _threshold(params.p_del)
        value_of: dict[int, int] = {}
        last_end: dict[int, int] = {}
        touched: list[int] = []
        free = 0
        for j, start in enumerate(starts):
            if start < free:
                continue
            key = others[key_idx[j]]
            present = key in value_of
            op_len = 2 if present and del_bits[j] < del_thr else 3
            if start + op_len > n:
                continue
            if key not in last_end:
                touched.append(key)
            out[start + 1] = key
            if op_len == 2:
                out[start] = dele
                del value_of[key]
            else:
                out[start] = ovr if present else ins
                out[start + 2] = values[j]
                value_of[key] = values[j]
            last_end[key] = start + op_len - 1
            free = start + op_len
        return out, value_of, last_end, touched

    def emit(
        self,
        params: KVParams,
        split: Split,
        rng: np.random.Generator,
        length: int,
        dep: int,
    ) -> Emitted:
        key_lo, key_hi = params.keys[split]
        chosen = key_lo + rng.choice(key_hi - key_lo, size=params.n_keys, replace=False)
        active: list[int] = chosen.tolist()
        k_star, others = active[0], active[1:]
        kind = self._KINDS[_weighted(rng, [params.anchor[k] for k in self._KINDS])]
        stale = kind != "ins"
        anchor_len = 2 if kind == "del" else 3
        middle_len = dep - 3
        middle, value_of, last_end, touched = self._middle(params, rng, others, middle_len)
        # Budget after BOS, the stale INS, the anchor, the middle, SEP, the anchor query and EOS.
        avail = length - 3 - (3 if stale else 0) - anchor_len - middle_len - 3
        if avail < 0:
            raise RuntimeError(f"kv: length {length} below min_length for dep {dep}")
        # Extra queries sit 3 symbols further right per slot, so slot q (0-based) may only ask
        # a key whose defining operation ended at least 3q + 2 symbols into the middle; otherwise
        # that query's dependency would exceed the anchor's D. Candidates come in random order.
        order: list[int] = rng.permutation(len(touched)).tolist()
        extra_keys: list[int] = []
        for which in order:
            if len(extra_keys) >= min(params.max_queries - 1, avail // 3):
                break
            key = touched[which]
            if last_end[key] >= 3 * len(extra_keys) + 2:
                extra_keys.append(key)
        extra = len(extra_keys)
        prefix_len = avail - 3 * extra + (3 if stale else 0)
        prefix = _draw(rng, params.distractors, prefix_len)
        value = int(rng.integers(params.values[0], params.values[1]))
        if stale:
            stale_value = int(rng.integers(params.values[0], params.values[1]))
            at = int(rng.integers(0, prefix_len - 2))
            prefix[at : at + 3] = (params.markers["ins"], k_star, stale_value)
        if kind == "del":
            anchor = [params.markers["del"], k_star]
            answer = NONE
        else:
            anchor = [params.markers[kind], k_star, value]
            answer = value
        anchor_end = 1 + prefix_len + anchor_len - 1  # position of the anchor's last symbol
        middle_start = anchor_end + 1
        sep_pos = anchor_end + dep - 2
        queries: list[int] = [QRY, k_star, answer]
        positions = [sep_pos + 2]
        deps = [dep]
        for q, key in enumerate(extra_keys):
            queries += [QRY, key, value_of.get(key, NONE)]
            key_pos = sep_pos + 2 + 3 * (q + 1)
            positions.append(key_pos)
            deps.append(key_pos - (middle_start + last_end[key]))
        tokens = _assemble([[BOS], prefix, anchor, middle, [SEP], queries, [EOS]])
        if tokens.shape[0] != length:
            raise RuntimeError(f"kv: assembled {tokens.shape[0]} symbols, wanted {length}")
        return tokens, positions, deps

    def solve(self, params: KVParams, tokens: U8) -> Solved:
        ins, ovr, dele = params.markers["ins"], params.markers["ovr"], params.markers["del"]
        t: list[int] = tokens.tolist()
        _frame(t)
        store: dict[int, int] = {}
        last: dict[int, int] = {}
        i = 1
        try:
            while t[i] != SEP:
                byte = t[i]
                if byte == ins or byte == ovr:
                    store[t[i + 1]] = t[i + 2]
                    last[t[i + 1]] = i + 2
                    i += 3
                elif byte == dele:
                    store.pop(t[i + 1], None)
                    last[t[i + 1]] = i + 1
                    i += 2
                else:
                    i += 1
            i += 1
            positions: list[int] = []
            targets: list[int] = []
            deps: list[int] = []
            while t[i] != EOS:
                if t[i] != QRY:
                    raise ValueError(f"kv: expected QRY at position {i}, got {t[i]}")
                key = t[i + 1]
                if key not in last:
                    raise ValueError(f"kv: query at {i} asks a key that no operation touched")
                positions.append(i + 1)
                targets.append(store.get(key, NONE))
                deps.append(i + 1 - last[key])
                i += 3
        except IndexError as exc:
            raise ValueError("kv: truncated document") from exc
        if i != len(t) - 1:
            raise ValueError("kv: symbols after EOS")
        return _sorted_solution(positions, targets, deps)


# ---- delayed -------------------------------------------------------------------------------------


class _Delayed(_Generator[DelayedParams]):
    """``BOS prefix-filler (k v)^n filler SEP QRY k_1 v_1 (QRY k v)* EOS``.

    The first query asks the first fact; its key position is D after that fact's value, so the
    filler between facts and queries has exactly ``D - 2n - 1`` symbols. ``T == L`` exactly.

    With ``key_len = K > 1`` a key is ``K`` symbols: a fact has ``K + 1`` symbols, a query ``K +
    2``, the supervised position is the key's last symbol, and the filler has ``D - (K + 1) n -
    1`` symbols. ``key_len = 1`` is the form above, symbol for symbol.
    """

    family: Family = "delayed"
    params_model = DelayedParams

    def validate(self, spec: TaskSpec, params: DelayedParams) -> None:
        super().validate(spec, params)
        need = params.max_facts if params.key_len == 1 else params.grid_side(params.max_facts)
        for split in SPLITS:
            lo, hi = params.keys[split]
            if hi - lo < need:
                raise ValueError(
                    f"{spec.key}: keys[{split}] has {hi - lo} symbols but "
                    f"max_facts={params.max_facts}"
                    + ("" if params.key_len == 1 else f" needs {need} per key position")
                )

    def min_dep(self, params: DelayedParams) -> int:
        return params.key_len + 2  # one fact (key and value) and an empty filler

    def min_length(self, params: DelayedParams, dep: int) -> int:
        # BOS + facts and filler (dep - 1) + SEP + first query (key_len + 2) + EOS
        return dep + params.key_len + 4

    def emit(
        self,
        params: DelayedParams,
        split: Split,
        rng: np.random.Generator,
        length: int,
        dep: int,
    ) -> Emitted:
        key_lo, key_hi = params.keys[split]
        klen = params.key_len
        fact_w, query_w = klen + 1, klen + 2  # symbols per fact and per query
        n_max = min(params.max_facts, (dep - 1) // fact_w)
        n = int(rng.integers(1, n_max + 1))
        keys: list[list[int]]
        if klen == 1:  # synthetic_v1's draw, unchanged
            keys = [[k] for k in (key_lo + rng.choice(key_hi - key_lo, size=n, replace=False))]
        else:  # n distinct cells of a side^klen grid of key symbols (class docstring)
            side = params.grid_side(n)
            axes = [
                (key_lo + rng.choice(key_hi - key_lo, size=side, replace=False)).tolist()
                for _ in range(klen)
            ]
            keys = []
            for cell in rng.choice(side**klen, size=n, replace=False).tolist():
                key = []
                for axis in axes:
                    cell, digit = divmod(cell, side)
                    key.append(axis[digit])
                keys.append(key)
        keys = [[int(k) for k in key] for key in keys]
        values: list[int] = rng.integers(params.values[0], params.values[1], size=n).tolist()
        filler_len = dep - fact_w * n - 1
        avail = length - dep - (klen + 4)
        if avail < 0:
            raise RuntimeError(f"delayed: length {length} below min_length for dep {dep}")
        # Query slot q (0-based) lies query_w * q symbols right of the first query, so it may
        # only ask a fact stated at least query_w * q / fact_w facts after the first one;
        # otherwise its dependency would exceed the anchor's D. Candidates come in random order.
        order: list[int] = (1 + rng.permutation(n - 1)).tolist() if n > 1 else []
        chosen = [0]
        for which in order:
            if len(chosen) > min(params.max_queries - 1, avail // query_w):
                break
            if fact_w * which >= query_w * len(chosen):
                chosen.append(which)
        extra = len(chosen) - 1
        prefix_len = avail - query_w * extra
        facts: list[int] = []
        for key, value in zip(keys, values, strict=True):
            facts += [*key, value]
        fact_start = 1 + prefix_len
        sep_pos = fact_start + fact_w * n + filler_len
        queries: list[int] = []
        positions: list[int] = []
        deps: list[int] = []
        for q, which in enumerate(chosen):
            queries += [QRY, *keys[which], values[which]]
            key_pos = sep_pos + 1 + klen + query_w * q  # the key's last symbol
            positions.append(key_pos)
            deps.append(key_pos - (fact_start + fact_w * which + klen))
        tokens = _assemble(
            [
                [BOS],
                _draw(rng, params.filler, prefix_len),
                facts,
                _draw(rng, params.filler, filler_len),
                [SEP],
                queries,
                [EOS],
            ]
        )
        if tokens.shape[0] != length:
            raise RuntimeError(f"delayed: assembled {tokens.shape[0]} symbols, wanted {length}")
        return tokens, positions, deps

    def solve(self, params: DelayedParams, tokens: U8) -> Solved:
        is_key = _byte_table(list(params.keys.values()))
        is_filler = _byte_table([params.filler])
        t: list[int] = tokens.tolist()
        _frame(t)
        klen = params.key_len
        value_of: dict[tuple[int, ...], int] = {}
        value_pos: dict[tuple[int, ...], int] = {}
        i = 1
        try:
            while is_filler[t[i]]:
                i += 1
            while is_key[t[i]]:
                key = tuple(t[i : i + klen])
                if not all(is_key[k] for k in key):
                    raise ValueError(f"delayed: malformed key at position {i}")
                if key in value_of:
                    shown = key[0] if klen == 1 else list(key)
                    raise ValueError(f"delayed: key {shown} stated twice")
                value_of[key] = t[i + klen]
                value_pos[key] = i + klen
                i += klen + 1
            while is_filler[t[i]]:
                i += 1
            if t[i] != SEP:
                raise ValueError(f"delayed: expected SEP at position {i}, got {t[i]}")
            i += 1
            positions: list[int] = []
            targets: list[int] = []
            deps: list[int] = []
            while t[i] != EOS:
                key = tuple(t[i + 1 : i + 1 + klen])
                if t[i] != QRY or key not in value_of:
                    raise ValueError(f"delayed: malformed query at position {i}")
                positions.append(i + klen)
                targets.append(value_of[key])
                deps.append(i + klen - value_pos[key])
                i += klen + 2
        except IndexError as exc:
            raise ValueError("delayed: truncated document") from exc
        if i != len(t) - 1:
            raise ValueError("delayed: symbols after EOS")
        return _sorted_solution(positions, targets, deps)


# ---- brackets ------------------------------------------------------------------------------------


def _dyck(
    rng: np.random.Generator,
    n: int,
    cap: int,
    params: BracketsParams,
) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    """A balanced bracket segment of exactly ``n`` symbols with depth <= ``cap``.

    A close may be preceded by ``QRY`` (probability ``p_query``), which costs one extra symbol.
    Returns the symbols, the queried closes as ``(QRY position, matching open position)`` and the
    positions of the closes that are not queried (candidates for corruption). Any ``n != 1`` is
    feasible; the walk never enters a state it cannot finish from (depth ``d`` with ``r``
    remaining symbols is finishable iff ``r >= d`` and not ``(d, r) == (0, 1)``).
    """
    if n == 1:
        raise ValueError("a balanced segment cannot have exactly one symbol")
    open_lo, close_lo, n_types = params.opens[0], params.closes[0], params.n_pairs
    query_thr = _threshold(params.p_query)
    open_bits: list[int] = rng.integers(0, 2, size=n).tolist()
    query_bits = _bits(rng, n)
    types: list[int] = rng.integers(0, n_types, size=n).tolist()
    out: list[int] = []
    stack: list[tuple[int, int]] = []
    queries: list[tuple[int, int]] = []
    plain_closes: list[int] = []
    depth = 0
    remaining = n
    step = 0
    while remaining > 0:
        pos = len(out)
        can_open = depth < cap and remaining >= depth + 2
        if depth == 0 or (can_open and open_bits[step]):
            if not can_open:
                raise RuntimeError("dyck: unfinishable state")  # pragma: no cover
            out.append(open_lo + types[step])
            stack.append((pos, types[step]))
            depth += 1
            remaining -= 1
        else:
            plain_ok = not (depth == 1 and remaining == 2)
            query_ok = remaining >= depth + 1 and not (depth == 1 and remaining == 3)
            open_pos, kind = stack.pop()
            if (query_bits[step] < query_thr and query_ok) or not plain_ok:
                out.append(QRY)
                out.append(close_lo + kind)
                queries.append((pos, open_pos))
                remaining -= 2
            else:
                out.append(close_lo + kind)
                plain_closes.append(pos)
                remaining -= 1
            depth -= 1
        step += 1
    return out, queries, plain_closes


class _Brackets(_Generator[BracketsParams]):
    """``BOS blocks* O_a inner QRY C_a SEP verdict EOS``.

    The anchor span ``O_a … C_a`` has exactly ``D`` symbols, so the verdict at ``SEP`` is ``D``
    after the anchor open; blocks before it are balanced segments shorter than the span. With
    probability ``p_invalid`` one close that carries no ``QRY`` inside the span gets a wrong type
    (an unqueried close exists in practice; otherwise the document stays valid). Loss positions:
    every ``QRY`` (target: the close type matching the stack top) and ``SEP`` (target: VALID or
    INVALID). ``T`` is ``L`` or ``L - 1`` (a one-symbol remainder cannot be balanced).
    """

    family: Family = "brackets"
    params_model = BracketsParams

    def min_dep(self, params: BracketsParams) -> int:
        return 5  # O_a + an inner segment of >= 2 symbols + QRY + C_a

    def min_length(self, params: BracketsParams, dep: int) -> int:
        return dep + 4  # BOS + span + SEP + verdict + EOS

    def emit(
        self,
        params: BracketsParams,
        split: Split,
        rng: np.random.Generator,
        length: int,
        dep: int,
    ) -> Emitted:
        open_lo, close_lo, n_types = params.opens[0], params.closes[0], params.n_pairs
        inner, inner_queries, inner_plain = _dyck(rng, dep - 3, params.max_depth - 1, params)
        anchor_type = int(rng.integers(0, n_types))
        span = [open_lo + anchor_type, *inner, QRY, close_lo + anchor_type]
        if inner_plain and int(rng.integers(0, _SIXTEEN)) < _threshold(params.p_invalid):
            at = inner_plain[int(rng.integers(0, len(inner_plain)))]
            shift = 1 + int(rng.integers(0, n_types - 1))
            span[at + 1] = close_lo + (inner[at] - close_lo + shift) % n_types
            verdict = params.verdict["invalid"]
        else:
            verdict = params.verdict["valid"]
        budget = length - 1 - dep - 3
        if budget < 0:
            raise RuntimeError(f"brackets: length {length} below min_length for dep {dep}")
        cap = min(params.block_max, dep)
        blocks: list[int] = []
        positions: list[int] = []
        deps: list[int] = []
        while budget >= 2:
            block_len = int(rng.integers(2, min(cap, budget) + 1))
            if budget - block_len == 1:
                block_len = block_len + 1 if block_len < cap else block_len - 1
            block, queries, _ = _dyck(rng, block_len, params.max_depth, params)
            offset = 1 + len(blocks)
            for query_pos, open_pos in queries:
                positions.append(offset + query_pos)
                deps.append(query_pos - open_pos)
            blocks += block
            budget -= block_len
        span_start = 1 + len(blocks)
        for query_pos, open_pos in inner_queries:
            positions.append(span_start + 1 + query_pos)
            deps.append(query_pos - open_pos)
        positions.append(span_start + dep - 2)  # the anchor's QRY
        deps.append(dep - 2)
        positions.append(span_start + dep)  # SEP: the verdict
        deps.append(dep)
        tokens = _assemble([[BOS], blocks, span, [SEP, verdict, EOS]])
        order = np.argsort(np.asarray(positions, dtype=np.int64), kind="stable").tolist()
        return tokens, [positions[k] for k in order], [deps[k] for k in order]

    def solve(self, params: BracketsParams, tokens: U8) -> Solved:
        open_lo, open_hi = params.opens
        close_lo, close_hi = params.closes
        t: list[int] = tokens.tolist()
        _frame(t)
        stack: list[tuple[int, int]] = []
        positions: list[int] = []
        targets: list[int] = []
        deps: list[int] = []
        mismatch = False
        last_open = -1
        i = 1
        try:
            while t[i] != SEP:
                byte = t[i]
                if open_lo <= byte < open_hi:
                    stack.append((i, byte - open_lo))
                    i += 1
                elif byte == QRY:
                    close = t[i + 1]
                    if not (close_lo <= close < close_hi):
                        raise ValueError(f"brackets: QRY at {i} is not followed by a close")
                    open_pos, kind = stack.pop()
                    if close - close_lo != kind:
                        raise ValueError(f"brackets: queried close at {i + 1} has a wrong type")
                    positions.append(i)
                    targets.append(close_lo + kind)
                    deps.append(i - open_pos)
                    last_open = open_pos
                    i += 2
                elif close_lo <= byte < close_hi:
                    open_pos, kind = stack.pop()
                    if byte - close_lo != kind:
                        mismatch = True
                    last_open = open_pos
                    i += 1
                else:
                    raise ValueError(f"brackets: unexpected byte {byte} at position {i}")
            if stack:
                raise ValueError("brackets: unbalanced word")
            positions.append(i)
            targets.append(params.verdict["invalid"] if mismatch else params.verdict["valid"])
            deps.append(i - last_open)
            if t[i + 1] not in params.verdict.values() or i + 2 != len(t) - 1:
                raise ValueError("brackets: expected `SEP verdict EOS`")
        except IndexError as exc:
            raise ValueError("brackets: truncated document") from exc
        return _sorted_solution(positions, targets, deps)


# ---- arithmetic ----------------------------------------------------------------------------------


def _add_digits(a: list[int], b: list[int]) -> list[int]:
    """Digit-wise addition, least significant digit first, with a vectorised carry loop."""
    n = max(len(a), len(b))
    s = np.zeros(n + 1, dtype=np.int64)
    s[: len(a)] += np.asarray(a, dtype=np.int64)
    s[: len(b)] += np.asarray(b, dtype=np.int64)
    carry = s // 10
    while carry.any():
        s %= 10
        s[1:] += carry[:-1]
        carry = s // 10
    digits: list[int] = s.tolist()
    if digits[-1] == 0:
        digits.pop()
    return digits


def _arith_source(i: int, la: int, lb: int, ls: int) -> int:
    """Item-relative position of the input digit that output ``i`` (``ls`` = SEP) depends on."""
    if i < ls and i < la:
        return i  # a_i
    if i < ls and i < lb:
        return la + 1 + i  # b_i
    return la - 1  # a final carry digit or the terminating SEP: the most significant digit of a


def _digits_to_int(digits: Sequence[int]) -> int:
    return int("".join(str(d) for d in reversed(digits)))


class _Arithmetic(_Generator[ArithmeticParams]):
    """``BOS (a PLUS b FILLER EQ s SEP)+ EOS``, digits least significant first.

    Sum digit ``s_i`` is predicted at the position before it (EQ for ``s_0``); its direct
    dependency is the input digit ``a_i`` (or ``b_i`` when ``i >= len(a)``, or the most significant
    digit of ``a`` for a final carry digit); carries are unit-distance links and are not counted.
    The terminating SEP depends on the most significant digit of ``a`` too, and that is the
    item's longest dependency: ``len(b) + filler + len(s) + 2``. The first item's filler makes it
    exactly ``D``; later items are shorter. ``T <= L``.
    """

    family: Family = "arithmetic"
    params_model = ArithmeticParams

    def min_dep(self, params: ArithmeticParams) -> int:
        return 2 * params.digits_max + 3  # lb + filler (>= 0) + ls (<= digits_max + 1) + 2

    def min_length(self, params: ArithmeticParams, dep: int) -> int:
        # BOS + a (<= digits_max) + PLUS + b, filler, EQ, sum and SEP (dep + 1) + EOS
        return dep + params.digits_max + 3

    def _number(self, rng: np.random.Generator, n: int) -> list[int]:
        digits: list[int] = rng.integers(0, 10, size=n).tolist()
        if n >= 2:
            digits[-1] = int(rng.integers(1, 10))  # the most significant digit is never zero
        return digits

    def _item(
        self,
        params: ArithmeticParams,
        rng: np.random.Generator,
        a: list[int],
        b: list[int],
        s: list[int],
        fill: int,
    ) -> tuple[list[int], list[int], list[int]]:
        d0 = params.digits[0]
        la, lb, ls = len(a), len(b), len(s)
        body = [
            *(d0 + x for x in a),
            params.markers["plus"],
            *(d0 + x for x in b),
            *_draw(rng, params.filler, fill).tolist(),
            params.markers["eq"],
            *(d0 + x for x in s),
            SEP,
        ]
        eq = la + 1 + lb + fill
        positions = [eq + i for i in range(ls + 1)]
        deps = [eq + i - _arith_source(i, la, lb, ls) for i in range(ls + 1)]
        return body, positions, deps

    def emit(
        self,
        params: ArithmeticParams,
        split: Split,
        rng: np.random.Generator,
        length: int,
        dep: int,
    ) -> Emitted:
        d_max = params.digits_max
        la, lb = int(rng.integers(1, d_max + 1)), int(rng.integers(1, d_max + 1))
        a, b = self._number(rng, la), self._number(rng, lb)
        s = _add_digits(a, b)
        body, positions, deps = self._item(params, rng, a, b, s, dep - lb - len(s) - 2)
        remaining = length - 2 - len(body)
        if remaining < 0:
            raise RuntimeError(f"arithmetic: length {length} below min_length for dep {dep}")
        parts: list[list[int]] = [body]
        all_positions = [1 + p for p in positions]
        all_deps = list(deps)
        offset = 1 + len(body)
        while True:
            la, lb = int(rng.integers(1, d_max + 1)), int(rng.integers(1, d_max + 1))
            a, b = self._number(rng, la), self._number(rng, lb)
            s = _add_digits(a, b)
            ls = len(s)
            fill_max = min(dep - lb - ls - 2, remaining - (la + lb + ls + 3))
            if fill_max < 0:
                break
            fill = int(rng.integers(0, fill_max + 1))
            body, positions, deps = self._item(params, rng, a, b, s, fill)
            parts.append(body)
            all_positions += [offset + p for p in positions]
            all_deps += deps
            offset += len(body)
            remaining -= len(body)
        tokens = _assemble([[BOS], *parts, [EOS]])
        return tokens, all_positions, all_deps

    def solve(self, params: ArithmeticParams, tokens: U8) -> Solved:
        d_lo, d_hi = params.digits
        is_filler = _byte_table([params.filler])
        plus, eq = params.markers["plus"], params.markers["eq"]
        t: list[int] = tokens.tolist()
        _frame(t)
        positions: list[int] = []
        targets: list[int] = []
        deps: list[int] = []
        i = 1
        try:
            while t[i] != EOS:
                start = i
                a: list[int] = []
                while d_lo <= t[i] < d_hi:
                    a.append(t[i] - d_lo)
                    i += 1
                if not a or t[i] != plus:
                    raise ValueError(f"arithmetic: expected digits then PLUS at position {start}")
                i += 1
                b: list[int] = []
                while d_lo <= t[i] < d_hi:
                    b.append(t[i] - d_lo)
                    i += 1
                if not b:
                    raise ValueError(f"arithmetic: expected digits after PLUS at position {i}")
                while is_filler[t[i]]:
                    i += 1
                if t[i] != eq:
                    raise ValueError(f"arithmetic: expected EQ at position {i}, got {t[i]}")
                eq_pos = i
                total = _digits_to_int(a) + _digits_to_int(b)
                s = [int(ch) for ch in reversed(str(total))]
                la, lb, ls = len(a), len(b), len(s)
                for k in range(ls + 1):
                    positions.append(eq_pos + k)
                    targets.append(d_lo + s[k] if k < ls else SEP)
                    deps.append(eq_pos + k - (start + _arith_source(k, la, lb, ls)))
                i = eq_pos + 1
                while d_lo <= t[i] < d_hi:
                    i += 1
                if t[i] != SEP:
                    raise ValueError(f"arithmetic: expected SEP at position {i}, got {t[i]}")
                i += 1
        except IndexError as exc:
            raise ValueError("arithmetic: truncated document") from exc
        if i != len(t) - 1:
            raise ValueError("arithmetic: symbols after EOS")
        return _sorted_solution(positions, targets, deps)


# ---- algorithmic ---------------------------------------------------------------------------------


def _algo_source(op: str, i: int, n: int) -> int:
    """Item-relative position of the input element output ``i`` (``n`` = SEP) depends on."""
    if i >= n or op == "sort":
        return 1  # x_0: the terminating SEP needs the whole list; sort needs every element
    if op == "copy":
        return 1 + i  # x_i
    return 1 + (n - 1 - i)  # reverse: x_{n-1-i}


class _Algorithmic(_Generator[AlgorithmicParams]):
    """``BOS (OP x_1..x_n FILLER SEP y_1..y_n SEP)+ EOS`` with OP in copy / reverse / sort.

    Output ``y_i`` is predicted at the position before it; dependencies: copy ``y_i <- x_i``
    (``n + F``), reverse ``y_i <- x_{n-1-i}`` (``F + 1 + 2i``), sort ``y_i <- x_0`` (``n + F + i``,
    every input matters), terminating SEP ``<- x_0`` (``2n + F``, the longest). The first item's
    filler makes that ``D``; later items are shorter. ``T <= L``.
    """

    family: Family = "algorithmic"
    params_model = AlgorithmicParams
    _OPS: tuple[Literal["copy", "reverse", "sort"], ...] = ("copy", "reverse", "sort")

    def min_dep(self, params: AlgorithmicParams) -> int:
        return 2 * params.n_max

    def min_length(self, params: AlgorithmicParams, dep: int) -> int:
        return dep + 5  # BOS + OP + inputs, filler and outputs (dep) + SEP + SEP + EOS

    def _item(
        self,
        params: AlgorithmicParams,
        rng: np.random.Generator,
        op: Literal["copy", "reverse", "sort"],
        n: int,
        fill: int,
    ) -> tuple[list[int], list[int], list[int]]:
        x = _draw(rng, params.items, n)
        y = x if op == "copy" else (x[::-1] if op == "reverse" else np.sort(x))
        body = [
            params.markers[op],
            *x.tolist(),
            *_draw(rng, params.filler, fill).tolist(),
            SEP,
            *y.tolist(),
            SEP,
        ]
        sep = 1 + n + fill
        positions = [sep + i for i in range(n + 1)]
        deps = [sep + i - _algo_source(op, i, n) for i in range(n + 1)]
        return body, positions, deps

    def emit(
        self,
        params: AlgorithmicParams,
        split: Split,
        rng: np.random.Generator,
        length: int,
        dep: int,
    ) -> Emitted:
        op = self._OPS[int(rng.integers(0, 3))]
        n = int(rng.integers(params.n_min, params.n_max + 1))
        body, positions, deps = self._item(params, rng, op, n, dep - 2 * n)
        remaining = length - 2 - len(body)
        if remaining < 0:
            raise RuntimeError(f"algorithmic: length {length} below min_length for dep {dep}")
        parts: list[list[int]] = [body]
        all_positions = [1 + p for p in positions]
        all_deps = list(deps)
        offset = 1 + len(body)
        while True:
            op = self._OPS[int(rng.integers(0, 3))]
            n = int(rng.integers(params.n_min, params.n_max + 1))
            fill_max = min(dep - 2 * n, remaining - (2 * n + 3))
            if fill_max < 0:
                break
            fill = int(rng.integers(0, fill_max + 1))
            body, positions, deps = self._item(params, rng, op, n, fill)
            parts.append(body)
            all_positions += [offset + p for p in positions]
            all_deps += deps
            offset += len(body)
            remaining -= len(body)
        tokens = _assemble([[BOS], *parts, [EOS]])
        return tokens, all_positions, all_deps

    def solve(self, params: AlgorithmicParams, tokens: U8) -> Solved:
        is_item = _byte_table([params.items])
        is_filler = _byte_table([params.filler])
        op_of = {byte: op for op, byte in params.markers.items()}
        t: list[int] = tokens.tolist()
        _frame(t)
        positions: list[int] = []
        targets: list[int] = []
        deps: list[int] = []
        i = 1
        try:
            while t[i] != EOS:
                start = i
                if t[i] not in op_of:
                    raise ValueError(f"algorithmic: expected an OP marker at position {i}")
                op = op_of[t[i]]
                i += 1
                x: list[int] = []
                while is_item[t[i]]:
                    x.append(t[i])
                    i += 1
                if not x:
                    raise ValueError(f"algorithmic: empty input list at position {start}")
                while is_filler[t[i]]:
                    i += 1
                if t[i] != SEP:
                    raise ValueError(f"algorithmic: expected SEP at position {i}, got {t[i]}")
                sep = i
                n = len(x)
                if op == "copy":
                    y = list(x)
                elif op == "reverse":
                    y = list(reversed(x))
                else:
                    y = sorted(x)
                for k in range(n + 1):
                    positions.append(sep + k)
                    targets.append(y[k] if k < n else SEP)
                    deps.append(sep + k - (start + _algo_source(op, k, n)))
                i = sep + 1
                while is_item[t[i]]:
                    i += 1
                if t[i] != SEP:
                    raise ValueError(f"algorithmic: expected SEP at position {i}, got {t[i]}")
                i += 1
        except IndexError as exc:
            raise ValueError("algorithmic: truncated document") from exc
        if i != len(t) - 1:
            raise ValueError("algorithmic: symbols after EOS")
        return _sorted_solution(positions, targets, deps)


_FAMILIES: dict[str, _Generator[Any]] = {
    generator.family: generator
    for generator in (_KV(), _Delayed(), _Brackets(), _Arithmetic(), _Algorithmic())
}


# --------------------------------------------------------------------------------------------------
# generate / reference solver
# --------------------------------------------------------------------------------------------------


def generate(spec: TaskSpec, split: Split, index: int) -> Example:
    """The ``index``-th example of ``split``: a pure function of ``(spec, split, index)``."""
    n = spec.n_examples(split)
    if not 0 <= index < n:
        raise IndexError(f"{spec.key}: index {index} outside [0, {n}) for split {split!r}")
    generator = _FAMILIES[spec.family]
    params = generator.params(spec)
    rng = _rng(spec.seed, split, index)
    dep_lo, dep_hi = spec.dep_len_range
    dep = int(rng.integers(dep_lo, dep_hi + 1))
    length_lo = max(spec.length_range[0], generator.min_length(params, dep))
    length = int(rng.integers(length_lo, spec.length_range[1] + 1))
    tokens, positions, deps = generator.emit(params, split, rng, length, dep)
    loss_mask: Bool = np.zeros(tokens.shape[0], dtype=np.bool_)
    loss_mask[positions] = True
    dep_len = max(deps)
    if dep_len != dep:
        raise RuntimeError(f"{spec.key}: emitted dep_len {dep_len} != anchor {dep}")
    context: U8 = np.full(tokens.shape[0], spec.context_id, dtype=np.uint8)
    meta: dict[str, Any] = {
        "task": spec.name,
        "variant": spec.variant,
        "family": spec.family,
        "split": split,
        "index": index,
        "doc_id": f"{spec.name}:{spec.variant}:{split}:{index}",
        "dep_len": dep_len,
        "in_range": dep_len <= spec.unroll,
        "deps": tuple(deps),
    }
    return Example(tokens, _next_symbol_targets(tokens), loss_mask, context, meta)


def reference_targets(spec: TaskSpec, tokens: U8) -> Solved:
    """Loss positions, expected targets and dependency distances re-derived from ``tokens`` alone.

    Raises ``ValueError`` when the document is not a well-formed ``spec.family`` document.
    """
    generator = _FAMILIES[spec.family]
    return generator.solve(generator.params(spec), np.ascontiguousarray(tokens, dtype=np.uint8))


def check_example(spec: TaskSpec, example: Example) -> None:
    """Raise ``ValueError`` unless ``example`` agrees with the reference solver and conventions."""
    positions, targets, deps = reference_targets(spec, example.tokens)
    mask_positions = np.flatnonzero(example.loss_mask)
    problems: list[str] = []
    if not np.array_equal(mask_positions, positions):
        problems.append("loss_mask positions differ from the reference solver")
    else:
        if not np.array_equal(example.targets[example.loss_mask], targets):
            problems.append("targets at loss positions differ from the reference solver")
        if not np.array_equal(np.asarray(example.meta["deps"], dtype=np.int64), deps):
            problems.append("meta['deps'] differ from the reference solver")
    if not np.array_equal(example.targets, _next_symbol_targets(example.tokens)):
        problems.append("targets are not the next symbol with PAD last")
    if example.meta["dep_len"] != (int(deps.max()) if deps.size else 0):
        problems.append("meta['dep_len'] is not the largest dependency")
    if bool(np.any(example.context != spec.context_id)):
        problems.append("context differs from the task's context_id")
    if problems:
        raise ValueError(f"{example.meta['doc_id']}: " + "; ".join(problems))


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    from hsn.data.splits import build_splits  # local import: splits imports this module

    dataset = load_dataset_spec(args.spec)
    out_dir = Path(args.out)
    started = time.perf_counter()
    manifest = build_splits(dataset, out_dir, n_gram=args.n_gram, leakage=not args.no_leakage)
    elapsed = time.perf_counter() - started
    n_docs = sum(shard.n_docs for shard in manifest.shards)
    print(
        f"built {dataset.name}: {len(manifest.shards)} shards, {n_docs} documents, "
        f"{sum(shard.n_tokens for shard in manifest.shards)} tokens in {elapsed:.1f} s "
        f"({n_docs / elapsed:.0f} documents/s including hashing and leakage checks)"
    )
    for shard in manifest.shards:
        print(f"  {shard.task}/{shard.variant}/{shard.split}: n={shard.n_docs} {shard.hash}")
    if manifest.leakage:
        worst = max(
            (pair.exact_duplicates + pair.overlapping_ngrams for pair in manifest.leakage),
            default=0,
        )
        print(f"leakage: {len(manifest.leakage)} pairs checked, worst count {worst}")
    status = 0
    if not manifest.is_clean:
        print("ERROR: leakage detected (see manifest.json)", file=sys.stderr)
        status = 1
    if args.write_golden:
        golden = Path(args.write_golden)
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote golden manifest {golden}")
    if args.golden:
        golden_text = Path(args.golden).read_text(encoding="utf-8")
        if golden_text != manifest.to_json():
            print(f"ERROR: manifest differs from the golden file {args.golden}", file=sys.stderr)
            status = 1
        else:
            print(f"manifest matches the golden file {args.golden}")
    return status


def _cmd_verify(args: argparse.Namespace) -> int:
    from hsn.data.splits import verify_splits  # local import: splits imports this module

    dataset = load_dataset_spec(args.spec)
    problems = verify_splits(dataset, Path(args.out))
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"verified {dataset.name} in {args.out}: every shard hash and every document agree")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    dataset = load_dataset_spec(args.spec)
    total_n = 0
    total_s = 0.0
    print(f"{'task':<28} {'n':>7} {'examples/s':>11} {'mean T':>7}")
    for task in dataset.tasks:
        n = min(int(args.n), task.n_train)
        if n == 0:
            continue
        started = time.perf_counter()
        symbols = 0
        for index in range(n):
            symbols += generate(task, "train", index).length
        elapsed = time.perf_counter() - started
        total_n += n
        total_s += elapsed
        print(f"{task.key:<28} {n:>7} {n / elapsed:>11.0f} {symbols / n:>7.1f}")
    if total_s > 0:
        print(f"{'overall':<28} {total_n:>7} {total_n / total_s:>11.0f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hsn.data.synthetic")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="generate shards and manifest.json")
    build.add_argument("--spec", required=True, help="dataset spec YAML")
    build.add_argument("--out", required=True, help="output directory")
    build.add_argument("--n-gram", type=int, default=32)
    build.add_argument("--no-leakage", action="store_true", help="skip the leakage checks")
    build.add_argument("--golden", help="compare the manifest with this file; exit 1 on mismatch")
    build.add_argument("--write-golden", help="write the manifest to this file as well")
    build.set_defaults(func=_cmd_build)
    verify = sub.add_parser("verify", help="regenerate and compare with the manifest")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--out", required=True)
    verify.set_defaults(func=_cmd_verify)
    bench = sub.add_parser("bench", help="measure generator throughput")
    bench.add_argument("--spec", required=True)
    bench.add_argument("--n", type=int, default=20_000, help="examples per task (train split)")
    bench.set_defaults(func=_cmd_bench)
    args = parser.parse_args(argv)
    status: int = args.func(args)
    return status


if __name__ == "__main__":
    # `python -m hsn.data.synthetic` loads this file as `__main__`; `hsn.data.splits` imports it
    # again as `hsn.data.synthetic`. Delegate to that copy so there is one set of classes.
    from hsn.data.synthetic import main as _main

    raise SystemExit(_main())
