"""Sweep specs (P1.11): one YAML file per sweep, expanded deterministically into arms with stable
ids. Imports without torch.

::

    name: p1_11_validation
    base: {config: configs/experiments/g1_ref_a1.yaml, overrides: [profile=rtx3060, ...]}
    families:                      # optional: per-family config + overrides, after base
      gru: {config: configs/baselines/gru_2m.yaml, overrides: [train.warmup_steps=200]}
    protocol: {name: reads_state, diag_every: 500, eval_batch: 128, probe_batch: 64}
    timing: false                  # true: the arm runs alone on an idle card and an idle CPU
    constants: {chance: 0.0208333}
    seeds: [0]
    rules: {verdicts: {...}, early_stop: [...]}     # hsn.experiments.rules
    arms:
      - name: bridge
        family: hsn                # checked against the resolved model_name
        variant: pdg               # or {name: cell_offset, offset_std: 0.5}
        overrides: [train.lr=0.001]
        grid: {model.d: [64, 128]} # optional, cartesian product in the declared order
        seeds: [0, 1, 2]
        protocol: {diag_every: 25} # merged over the sweep's protocol (unless it names another)
        timing: true
        constants: {ref_dev: 0.0696}

**Expansion** goes arm entry, grid point (in declared order), seed. The configuration overrides are
applied in the order base, family, arm, grid, ``seed=<s>``, and ``run_id=<slug>-s<seed>-<id12>``
(set by the harness), later winning; :func:`hsn.config.load_config` validates the result, so an
unknown key raises. Config paths are relative to the repository root of the imported ``hsn``.
Labels (``name[k=v,...]/s<seed>``) are unique within a sweep.

**Arm id** (:func:`arm_identity`): the first 16 hex digits of blake2b over the canonical JSON of the
resolved configuration's non-default values (``model_dump(exclude_defaults=True)``, without
``run_id`` and ``notes``: a new configuration key with a default leaves every id unchanged; a
changed default is a code change, tracked like any code change by the recorded commit), the data
(blake2b of the manifest file), the model family, the variant and its parameters, the protocol's
name, version and science parameters (not the execution-only :data:`EXECUTION_ONLY` ones, which
change speed and rounding only), and the early-stop rules that apply to the arm, with the arm's
constants substituted (two arms whose stops differ only by a constant are different arms). The seed
sits in the configuration. Verdict rules are not part of the id: they are pure functions of the
stored record. The *group id* is the same hash without the seed (seeds of one recipe); the *memory
key* hashes what sets the device memory (model and baseline blocks, batch, unroll, hardware,
variant, protocol execution parameters).

**Checks at expansion** (errors over silence): a CUDA arm that does not pin
``hardware.compile_mode`` in its overrides (profile defaults change: ``profile:rtx3060`` compiles by
default once P2.8 is merged), a compile mode on a baseline (P2.8 compiles the HSN window only; a
baseline would train eager under a compiled label), a family that does not match the configuration's
``model_name``, a variant on a baseline, a variant other than ``pdg`` with ``hardware.compile_mode``
set (untested under P2.8's compiled window), ``routing=random`` with the reads-state protocol (its
evaluation draws batch-shaped random numbers, so the batched probe would not match), protocol steps
beyond the schedule, duplicate labels, and every check of :func:`hsn.experiments.rules.check_rules`.

**P1.14 protocol keys** (approved with the P1.14 plan):
- ``max_step_s`` and ``step_time_approval``: the step-time guard. Execution-only, so not in the id.
- ``diag_dense_every`` and ``diag_windows``: the mixed cadence. The windows are derived from the
  arm's rules at expansion; both keys are in the id only when not at their defaults, see
  :data:`IDENTITY_DEFAULTS`.
- reads_state ``version``: 2 adds probe v2. It fills the id's existing protocol-version slot, so
  version 1 is today's payload.

The step-time refusals happen at enqueue (:func:`hsn.experiments.db.enqueue_arms`), not here, so
expansion of every existing spec is unchanged.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hsn.config import REPO_ROOT, RunConfig, config_hash, load_config
from hsn.experiments.rules import (
    Cadence,
    RuleSet,
    check_rules,
    merge_windows,
    parse_rules,
    resolved_early_stop,
    rule_windows,
)
from hsn.experiments.steptime import STEP_TIME_LIMIT_S

__all__ = [
    "EXECUTION_ONLY",
    "FAMILIES",
    "PROTOCOL_PARAMS",
    "PROTOCOL_SOURCES",
    "Arm",
    "SpecError",
    "SweepSpec",
    "arm_identity",
    "canonical",
    "digest",
    "expand",
    "load_spec",
    "slug",
]

FAMILIES = ("hsn", "gru", "transformer")
Family = Literal["hsn", "gru", "transformer"]
# P1.14: the step-time guard's limit and approval decide whether an arm may run, not what it
# computes
EXECUTION_ONLY = frozenset(
    {"eval_batch", "probe_batch", "checkpoint", "sleep_s", "max_step_s", "step_time_approval"}
)
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class SpecError(ValueError):
    """A sweep spec that cannot be expanded."""


def canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact, ASCII, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def digest(obj: Any, n: int = 16) -> str:
    """The first ``n`` hex digits of blake2b over :func:`canonical` of ``obj``."""
    return hashlib.blake2b(canonical(obj).encode(), digest_size=32).hexdigest()[:n]


def slug(text: str) -> str:
    """``text`` mapped to the run-id alphabet ``[A-Za-z0-9._-]`` (config.py's run_id rule)."""
    out = _SLUG_RE.sub("-", text).strip("-._")
    return out or "arm"


# --------------------------------------------------------------------------------------------------
# Protocol parameters (torch-free; the protocols themselves are hsn.experiments.protocols)
# --------------------------------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _TrainParams(_Strict):
    steps: int | None = Field(default=None, ge=1)  # None: train.steps (the schedule horizon)
    eval_batch: int | None = Field(default=None, ge=1)  # None: train.batch
    checkpoint: Literal["final", "none"] = "final"
    diag_window_docs: int = Field(default=4, ge=1)  # documents of the diagnostics window
    # P1.14, the step-time guard (hsn.experiments.steptime): the limit on the median step time,
    # and the approval note ("<who, when>") an arm above the 0.5 s rule (or an eager CUDA HSN arm)
    # needs at enqueue; both execution-only
    max_step_s: float = Field(default=STEP_TIME_LIMIT_S, gt=0.0)
    step_time_approval: str = ""


Window = tuple[int, int]


class _DiagCadence(_TrainParams):
    """P1.14's mixed diagnostic cadence: with ``diag_dense_every`` > 0, diagnostics every
    ``diag_dense_every`` steps inside ``diag_windows`` (the windows the arm's rules read, derived at
    expansion and merged with any the spec gives) and every ``diag_every`` steps elsewhere."""

    diag_dense_every: int = Field(default=0, ge=0)
    diag_windows: tuple[Window, ...] = ()

    @field_validator("diag_windows")
    @classmethod
    def _windows(cls, value: tuple[Window, ...]) -> tuple[Window, ...]:
        for lo, hi in value:
            if lo < 1 or hi < lo:
                raise ValueError(f"a diagnostics window [lo, hi] needs 1 <= lo <= hi, got {lo, hi}")
        return value


class ReadsStateParams(_DiagCadence):
    """The 5(b) training recipe (in the spec's overrides) plus the 5(c) edit probe."""

    name: Literal["reads_state"]
    # P1.14: version 2 adds probe v2, the vocabulary-matched edit (hsn.experiments.probe); the
    # version is the arm id's protocol version, so version 1 arms keep their ids
    version: Literal[1, 2] = 1
    diag_every: int = Field(default=500, ge=0)  # 0: no diagnostics during training
    diag_docs: int = Field(default=32, ge=1)  # documents of trained_state
    n_train: int = Field(default=0, ge=0)  # 0: the whole split
    n_dev: int = Field(default=0, ge=0)
    probe_batch: int = Field(default=64, ge=2)


class TrainEvalParams(_DiagCadence):
    name: Literal["train_eval"]
    diag_every: int = Field(default=0, ge=0)
    diag_docs: int = Field(default=32, ge=1)
    n_train: int = Field(default=0, ge=0)
    n_dev: int = Field(default=0, ge=0)


class OverfitParams(_DiagCadence):
    """The G1 overfit recipe: the first ``n_docs`` training documents as train and dev."""

    name: Literal["overfit"]
    n_docs: int = Field(default=64, ge=1)
    diag_every: int = Field(default=50, ge=0)
    thresholds: tuple[float, ...] = (0.90, 0.99)


class DryRunParams(_TrainParams):
    """No training: a synthetic record (queue tests, end-to-end checks of a spec)."""

    name: Literal["dry_run"]
    sleep_s: float = Field(default=0.0, ge=0.0)
    fail: bool = False
    accuracy: float = 0.5
    record: dict[str, Any] = Field(default_factory=dict)  # merged into the synthetic record


ProtocolParams = Annotated[
    ReadsStateParams | TrainEvalParams | OverfitParams | DryRunParams, Field(discriminator="name")
]
PROTOCOL_PARAMS: dict[str, type[_TrainParams]] = {
    "reads_state": ReadsStateParams,
    "train_eval": TrainEvalParams,
    "overfit": OverfitParams,
    "dry_run": DryRunParams,
}
PROTOCOL_VERSION = {"reads_state": 1, "train_eval": 1, "overfit": 1, "dry_run": 1}
PROTOCOL_SOURCES: dict[str, tuple[str, ...]] = {
    "reads_state": ("trained", "probe", "final"),
    "train_eval": ("trained", "final"),
    "overfit": ("overfit", "final"),
    "dry_run": ("trained", "probe", "probe_v2", "overfit", "final"),
}


def protocol_sources(params: Any) -> tuple[str, ...]:
    """The scalar rule sources a protocol writes (P1.14: reads_state version 2 adds probe_v2)."""
    sources = PROTOCOL_SOURCES[params.name]
    if params.name == "reads_state" and getattr(params, "version", 1) == 2:
        sources = (*sources, "probe_v2")
    return sources


def protocol_params(
    raw: Mapping[str, Any],
) -> ReadsStateParams | TrainEvalParams | OverfitParams | DryRunParams:
    name = raw.get("name")
    if name not in PROTOCOL_PARAMS:
        raise SpecError(f"protocol name must be one of {sorted(PROTOCOL_PARAMS)}, got {name!r}")
    try:
        params = PROTOCOL_PARAMS[str(name)].model_validate(dict(raw))
    except ValidationError as exc:
        raise SpecError(f"protocol {name}: {exc}") from exc
    assert isinstance(params, (ReadsStateParams, TrainEvalParams, OverfitParams, DryRunParams))
    return params


# --------------------------------------------------------------------------------------------------
# Spec schema
# --------------------------------------------------------------------------------------------------


class BaseRecipe(_Strict):
    config: str
    overrides: list[str] = Field(default_factory=list)


class FamilyRecipe(_Strict):
    config: str | None = None
    overrides: list[str] = Field(default_factory=list)


class VariantSpec(_Strict):
    name: str
    offset_std: float | None = None


Scalar = float | int | str | bool


class ArmSpec(_Strict):
    name: str = Field(pattern=_NAME_RE)
    family: Family | None = None
    config: str | None = None
    variant: str | VariantSpec | None = None
    overrides: list[str] = Field(default_factory=list)
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    seeds: list[int] | None = None
    protocol: dict[str, Any] = Field(default_factory=dict)
    timing: bool | None = None
    constants: dict[str, Scalar] = Field(default_factory=dict)

    @field_validator("grid")
    @classmethod
    def _grid(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key, values in value.items():
            if not values:
                raise ValueError(f"grid {key}: no values")
            if key != "variant" and "=" in key:
                raise ValueError(f"grid key {key!r} must be a dotted config key")
        return value


class SweepSpec(_Strict):
    name: str = Field(pattern=_NAME_RE)
    description: str = ""
    base: BaseRecipe
    families: dict[str, FamilyRecipe] = Field(default_factory=dict)
    protocol: dict[str, Any]
    timing: bool = False
    constants: dict[str, Scalar] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [0])
    rules: dict[str, Any] | None = None
    arms: list[ArmSpec] = Field(min_length=1)

    @field_validator("families")
    @classmethod
    def _families(cls, value: dict[str, FamilyRecipe]) -> dict[str, FamilyRecipe]:
        unknown = set(value) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown model families {sorted(unknown)}; known: {FAMILIES}")
        return value


def load_spec(path: str | Path) -> tuple[SweepSpec, str]:
    """Parse a spec file; returns the spec and its text (stored at enqueue)."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: a sweep spec is a mapping")
    try:
        return SweepSpec.model_validate(raw), text
    except ValidationError as exc:
        raise SpecError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------------------------------


@dataclass
class Arm:
    """One expanded arm: everything a worker needs to run it, and its identity."""

    label: str
    name: str
    position: int
    family: str
    variant: str
    variant_params: dict[str, Any]
    seed: int
    config_path: str
    overrides: list[str]
    config: RunConfig
    protocol: ReadsStateParams | TrainEvalParams | OverfitParams | DryRunParams
    timing: bool
    constants: dict[str, Any]
    data_hash: str
    arm_id: str
    group_id: str
    memory_key: str
    run_id: str
    early_stop: list[dict[str, Any]] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return self.protocol.steps or self.config.train.steps


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _label_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return ",".join(f"{k}={_label_value(v)}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:g}"
    return _value_text(value)


def _resolve_path(path: str, repo: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo / p


def data_hash(cfg: RunConfig, repo: Path, *, required: bool) -> str:
    """blake2b of the data manifest (the data's identity: the manifest holds the shard hashes)."""
    manifest = cfg.data.manifest
    if manifest is None:
        if required:
            raise SpecError("data.manifest is not set")
        return "none"
    path = _resolve_path(str(manifest), repo)
    if not path.exists():
        if required:
            raise SpecError(f"data manifest {path} does not exist (build the dataset first)")
        return "none"
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()[:16]


def config_view(cfg: RunConfig) -> dict[str, Any]:
    """The non-default values of the resolved configuration, without ``run_id`` and ``notes``."""
    view = cfg.model_dump(mode="json", exclude_defaults=True)
    view.pop("run_id", None)
    view.pop("notes", None)
    return view


#: P1.14: protocol keys that enter the arm id only when they differ from these values, so that every
#: arm from before them keeps its id
IDENTITY_DEFAULTS: dict[str, Any] = {"diag_dense_every": 0, "diag_windows": []}


def arm_identity(
    cfg: RunConfig,
    *,
    family: str,
    variant: str,
    variant_params: Mapping[str, Any],
    protocol: Mapping[str, Any],
    early_stop: Sequence[Mapping[str, Any]],
    data: str,
) -> tuple[str, str]:
    """``(arm_id, group_id)`` (module docstring). ``protocol`` is the protocol's full parameter
    dump; the execution-only parameters are dropped here."""
    name = str(protocol["name"])
    science = {
        k: v
        for k, v in protocol.items()
        if k not in EXECUTION_ONLY
        and k != "version"
        and not (k in IDENTITY_DEFAULTS and _plain(v) == IDENTITY_DEFAULTS[k])
    }
    version = int(protocol.get("version", PROTOCOL_VERSION[name]))  # P1.14: reads_state v2
    payload: dict[str, Any] = {
        "config": config_view(cfg),
        "data": data,
        "family": family,
        "variant": {"name": variant, **dict(variant_params)},
        "protocol": {**science, "version": version},
        "early_stop": list(early_stop),
    }
    arm_id = digest(payload)
    payload["config"] = {k: v for k, v in payload["config"].items() if k != "seed"}
    return arm_id, digest(payload)


def _plain(value: Any) -> Any:
    """Tuples as lists (a dumped protocol and a stored one compare alike)."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def memory_key(cfg: RunConfig, variant: str, protocol: Mapping[str, Any]) -> str:
    dump = cfg.model_dump(mode="json")
    return digest(
        {
            "model_name": cfg.model_name,
            "model": dump["model"],
            "baseline": dump["baseline"],
            "batch": cfg.train.batch,
            "unroll": cfg.train.unroll,
            "gate_mode": cfg.train.gate_mode,  # forks a compiled graph (P2.8)
            "hardware": dump["hardware"],
            "variant": variant,
            "protocol": {
                k: protocol.get(k) for k in ("name", "eval_batch", "probe_batch", "diag_docs")
            },
        }
    )


def _variant(arm: ArmSpec, family: str) -> tuple[str, dict[str, Any]]:
    from hsn.experiments.variant_names import variant_params  # torch-free

    raw = arm.variant
    if raw is None:
        name, params = ("pdg" if family == "hsn" else "none"), dict[str, Any]()
    elif isinstance(raw, str):
        name, params = raw, {}
    else:
        name = raw.name
        params = {} if raw.offset_std is None else {"offset_std": raw.offset_std}
    if family != "hsn" and name != "none":
        raise SpecError(f"arm {arm.name}: variant {name!r} needs model family hsn, not {family}")
    if family == "hsn" and name == "none":
        raise SpecError(f"arm {arm.name}: an hsn arm takes variant pdg (the model as configured)")
    try:
        return name, variant_params(name, params)
    except (ValueError, NotImplementedError) as exc:
        raise SpecError(f"arm {arm.name}: {exc}") from exc


def expand(spec: SweepSpec, *, repo: Path = REPO_ROOT) -> list[Arm]:
    """Every arm of ``spec`` in a deterministic order (module docstring); raises
    :class:`SpecError`."""
    try:
        rules = parse_rules(spec.rules)
    except ValueError as exc:
        raise SpecError(f"rules: {exc}") from exc
    arms: list[Arm] = []
    labels: set[str] = set()
    for entry in spec.arms:
        grid_keys = list(entry.grid)
        points = list(itertools.product(*(entry.grid[k] for k in grid_keys))) or [()]
        for point in points:
            assignment = dict(zip(grid_keys, point, strict=True))
            for seed in entry.seeds if entry.seeds is not None else spec.seeds:
                arm = _expand_one(spec, entry, assignment, seed, rules, repo, len(arms))
                if arm.label in labels:
                    raise SpecError(f"duplicate arm label {arm.label!r}")
                labels.add(arm.label)
                arms.append(arm)
    return arms


def _grid_label(assignment: Mapping[str, Any]) -> str:
    if not assignment:
        return ""
    short = [k.rsplit(".", 1)[-1] for k in assignment]
    keys = short if len(set(short)) == len(short) else list(assignment)
    return (
        "["
        + ",".join(f"{k}={_label_value(v)}" for k, v in zip(keys, assignment.values(), strict=True))
        + "]"
    )


def _expand_one(
    spec: SweepSpec,
    entry: ArmSpec,
    assignment: dict[str, Any],
    seed: int,
    rules: RuleSet,
    repo: Path,
    position: int,
) -> Arm:
    label = f"{entry.name}{_grid_label(assignment)}/s{seed}"
    config_overrides = dict(assignment)
    if "variant" in config_overrides:
        value = config_overrides.pop("variant")
        variant_spec = VariantSpec.model_validate(value) if isinstance(value, Mapping) else value
        entry = entry.model_copy(update={"variant": variant_spec})
    grid_overrides = [f"{k}={_value_text(v)}" for k, v in config_overrides.items()]
    head = [*spec.base.overrides]
    tail = [*entry.overrides, *grid_overrides, f"seed={seed}"]
    family: str
    try:
        if entry.family is None:
            path = entry.config or spec.base.config
            family = load_config(_resolve_path(path, repo), [*head, *tail]).model_name
        else:
            family = entry.family
        recipe = spec.families.get(family)
        config_path = entry.config or (recipe.config if recipe else None) or spec.base.config
        overrides = [*head, *(recipe.overrides if recipe else []), *tail]
        cfg = load_config(_resolve_path(config_path, repo), overrides)
    except (ValueError, OSError) as exc:
        raise SpecError(f"arm {entry.name}: {exc}") from exc
    if family not in FAMILIES:
        raise SpecError(f"arm {entry.name}: unknown model family {family!r}")
    if cfg.hardware.device == "cuda" and not any(
        o.split("=", 1)[0].strip() == "hardware.compile_mode" for o in overrides
    ):
        raise SpecError(
            f"arm {entry.name}: a CUDA arm must pin hardware.compile_mode (none or "
            "reduce-overhead) in its overrides: profile:rtx3060 compiles by default once P2.8 is "
            "merged, and an unpinned arm would change its id (and what it computes) with it"
        )
    if cfg.model_name != family:
        raise SpecError(
            f"arm {entry.name}: family {family} but the configuration's model_name is "
            f"{cfg.model_name}"
        )
    if family != "hsn" and cfg.hardware.compile_mode != "none":
        # P2.8: the generic trainer ignores the compile mode and trains a baseline eager, so a
        # record would claim a compile that never ran
        raise SpecError(
            f"arm {entry.name}: hardware.compile_mode={cfg.hardware.compile_mode} compiles the HSN "
            f"training window only; a {family} baseline trains eager: pin "
            "hardware.compile_mode=none"
        )
    variant, vparams = _variant(entry, family)
    if cfg.hardware.compile_mode != "none" and variant not in ("pdg", "none"):
        raise SpecError(
            f"arm {entry.name}: variant {variant} with hardware.compile_mode="
            f"{cfg.hardware.compile_mode} is untested (P2.8's compiled window changes the cell's "
            "call); run variants eager: add hardware.compile_mode=none (profile:rtx3060 compiles "
            "by default once P2.8 is merged)"
        )
    # an arm that names another protocol starts from that protocol's defaults, not the sweep's
    other = "name" in entry.protocol and entry.protocol["name"] != spec.protocol.get("name")
    protocol = protocol_params(
        dict(entry.protocol) if other else {**spec.protocol, **entry.protocol}
    )
    if protocol.name == "reads_state" and family == "hsn" and cfg.model.routing == "random":
        raise SpecError(
            f"arm {entry.name}: routing=random with reads_state: its evaluation draws batch-shaped "
            "random numbers, so the batched edit probe would not match the per-document one"
        )
    steps = protocol.steps or cfg.train.steps
    if steps > cfg.train.steps:
        raise SpecError(f"arm {entry.name}: protocol steps {steps} > train.steps {cfg.train.steps}")
    constants = {**spec.constants, **entry.constants}
    diag_every = int(getattr(protocol, "diag_every", 0))
    dense = int(getattr(protocol, "diag_dense_every", 0))
    windows: list[list[int]] = []
    explicit = list(getattr(protocol, "diag_windows", ()))
    if explicit and not dense:
        raise SpecError(f"arm {label}: diag_windows needs diag_dense_every (P1.14's mixed cadence)")
    if dense:
        try:
            derived = rule_windows(rules, family, protocol.name, steps)
        except ValueError as exc:
            raise SpecError(f"arm {label}: {exc}") from exc
        windows = merge_windows([*explicit, *derived])
        if not windows:
            # no rule of this arm reads a diag: window (a GRU arm beside hsn-only rules, say): the
            # single cadence, recorded as such, so that its id is the plain arm's
            dense = 0
        protocol = protocol.model_copy(
            update={
                "diag_dense_every": dense,
                "diag_windows": tuple((lo, hi) for lo, hi in windows),
            }
        )
    cadence = Cadence.of(
        steps,
        cfg.train.eval_every,
        diag_every,
        protocol_sources(protocol),
        protocol.name,
        dense_every=dense,
        windows=windows,
    )
    try:
        check_rules(rules, cadence, constants, family)
    except ValueError as exc:
        raise SpecError(f"arm {label}: {exc}") from exc
    stops = resolved_early_stop(rules, family, protocol.name, constants)
    data = data_hash(cfg, repo, required=protocol.name != "dry_run")
    pdump = protocol.model_dump(mode="json")
    arm_id, group_id = arm_identity(
        cfg,
        family=family,
        variant=variant,
        variant_params=vparams,
        protocol=pdump,
        early_stop=stops,
        data=data,
    )
    run_id = f"{slug(entry.name)}-s{seed}-{arm_id[:12]}"
    cfg = load_config(_resolve_path(config_path, repo), [*overrides, f"run_id={run_id}"])
    return Arm(
        label=label,
        name=entry.name,
        position=position,
        family=family,
        variant=variant,
        variant_params=vparams,
        seed=seed,
        config_path=config_path,
        overrides=overrides,
        config=cfg,
        protocol=protocol,
        timing=entry.timing if entry.timing is not None else spec.timing,
        constants=constants,
        data_hash=data,
        arm_id=arm_id,
        group_id=group_id,
        memory_key=memory_key(cfg, variant, pdump),
        run_id=run_id,
        early_stop=stops,
    )


def config_identity(cfg: RunConfig) -> str:
    """The full configuration hash (provenance; changes with every key, unlike the arm id)."""
    return config_hash(cfg)
