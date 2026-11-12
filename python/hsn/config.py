"""Configuration schema (Appendix A), YAML loading, dotted overrides, hashing and the run manifest.

P0.2. Every key of Appendix A exists here with the same name and default. Configs are the single
source of truth: nothing in code hard-codes a dimension, and nothing *derived* (parameter counts,
measured memory, tokens/s) is stored in a config; those are measured and recorded in the run
manifest (:func:`write_run_manifest`). Thresholds are not config either: they live in
``configs/metrics_freeze.json``.

Defaults where Appendix A lists alternatives without naming one are the first alternative listed
(``edge_norm=softmax``, ``scorer=linear``, ``routing=dense``, ``predictor_target=fixed_embedding``,
``optimizer=adamw``, ``precision=bf16``, ``gate_mode=soft``, ``activation_checkpoint=none``,
``stream.policy=utility_lru``, ``serve.encryption=none``, ``serve.capture_policy=disabled``).
One default follows another key: ``train.optimizer`` left unset resolves to ``lazy_adamw`` when
``model.routing`` is a selection mode (P2.6: per-unit lazy Adam is the default of sparse runs) and
stays ``adamw`` for dense routing, so a dense configuration such as ``configs/ref_a1.yaml`` keeps
its resolved values and hash; an explicit ``train.optimizer`` is always kept.
Schedules, exploration and auxiliary loss weights default to neutral (off) so that nothing is
active unless a config asks for it (global rule 4: one auxiliary term at a time); ``lambda_p``
defaults to 1.0 because ``L_pred`` is part of the base objective (Plan §2.6), not an auxiliary term.

Unknown keys are rejected everywhere (``extra="forbid"``): a misspelt key raises instead of being
ignored. Models are frozen; a config never changes after it has been hashed.

Keys beyond Appendix A, added by later chunks with the same conventions: ``baseline.*`` (P0.5, the
sizes of the GRU and Transformer baselines), ``train.search_trials`` (P0.5, the hyperparameter
search budget every method gets), ``data.align_docs`` (P0.5, documents start at window
boundaries when streams are built), ``train.ema_tau`` / ``train.variance_floor_gamma`` (P1.6,
the EMA rate of the target encoder and the per-dimension standard-deviation floor on the batch
that ``model.predictor_target=ema_encoder`` uses; a fixed-embedding target ignores both) and
``model.cost_profile`` (P2.1, the cost profile the router's cost table is read from; ``null`` is
the uniform placeholder) and ``train.reinforce_baseline_decay`` (P2.2, the EMA rate of the REINFORCE
reward baseline; the P2.2 schedule keys ``gate_mode``, ``soft_until_step``, ``temperature_*``,
``epsilon_*`` and ``p_dense`` are Appendix A.3 keys that P2.2 validates and reads through
:class:`hsn.train.schedules.SelectionSchedule`). The P2.3 keys ``probe_fraction``, ``n_probe``,
``probe_horizon`` and ``ema_decay`` act only when ``train.lambda_u > 0`` (counterfactual probes,
the utility term and the scorer's EMA features; :mod:`hsn.train.probes`), so their defaults leave
every other configuration unchanged; such a configuration must be routed, probe at least some
windows, keep the horizon inside the window and not run soft gates throughout. P1.16's rollback
keys (``train.rollback_*``, :data:`ROLLBACK_KEYS`) are off by default and, at their defaults, left
out of the hash (:data:`HASH_OMIT_AT_DEFAULT`), so every configuration written before them keeps its
``config_hash``.

Override values (``load_config(path, overrides=["model.k_select=4"])``) are parsed as YAML scalars
(``4``, ``0.5``, ``true``, ``null``, ``[0, 1]``, ``{a: 1}``) unless the target field is a string, a
string ``Literal`` or a path, in which case the text is taken verbatim; pydantic then validates and
coerces the value exactly as it does for the YAML file.
"""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Annotated, Any, Literal, Self, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = [
    "AUX_TERMS",
    "HARDWARE_PROFILE_DIR",
    "HASH_OMIT_AT_DEFAULT",
    "MEASURED_KEYS",
    "PROFILES",
    "REPO_ROOT",
    "ROLLBACK_KEYS",
    "BaselineConfig",
    "DataConfig",
    "FastConfig",
    "GRUBaselineConfig",
    "GrowthConfig",
    "HardwareConfig",
    "ModelConfig",
    "RecipeConfig",
    "ReplayConfig",
    "RunConfig",
    "RunManifest",
    "SanConfig",
    "ScorerFeaturesConfig",
    "ServeConfig",
    "ServeLimitsConfig",
    "StreamConfig",
    "TrainConfig",
    "TransformerBaselineConfig",
    "canonical_json",
    "config_hash",
    "dump_resolved",
    "load_config",
    "load_hardware_profile",
    "profile_path",
    "write_run_manifest",
    "wsd_decay_steps",
]

# --------------------------------------------------------------------------------------------------
# Enumerations: Literal strings so YAML stays readable (chunk P0.2 pitfalls).
# --------------------------------------------------------------------------------------------------

EdgeNorm = Literal["softmax", "sigmoid"]
Scorer = Literal["linear", "mlp"]
Routing = Literal["dense", "topk", "greedy_budget", "random", "fixed"]
PredictorTarget = Literal["fixed_embedding", "ema_encoder"]
Tokenizer = Literal["bytes"]
Split = Literal["train", "dev", "test"]
Variant = Literal["in_range", "out_of_range"]  # P0.4 dependency-length variants
Optimizer = Literal["adamw", "lazy_adamw"]
Precision = Literal["bf16", "fp16", "fp32", "fp64"]
GradNormPrecision = Literal["native", "float64"]
#: A6: the state's feedback reads a backward pass can leave out (``train.feedback_detach``).
FeedbackPath = Literal["summary", "eps", "state", "router"]
FEEDBACK_PATHS: tuple[str, ...] = ("summary", "eps", "state", "router")
GateMode = Literal["soft", "ste", "reinforce"]
ActivationCheckpoint = Literal["none", "per_round", "per_window"]
StreamPolicy = Literal["lru", "utility_lru", "pinned"]
Encryption = Literal["none", "aes_gcm"]
CapturePolicy = Literal["disabled", "local_only", "shared_candidates"]
Profile = Literal["rtx3060", "a100", "dgx_spark", "m_series", "cpu", "gpu8gb"]
Device = Literal["cuda", "mps", "cpu"]
AutocastDtype = Literal["bf16", "fp16", "fp32"]
MasterDtype = Literal["fp32", "fp64"]
CompileMode = Literal["none", "reduce-overhead", "max-autotune"]
#: P2.10: how a routed training window runs. ``autograd`` is the window of P1.7 / P2.8 (one
#: autograd graph over the positions); ``split`` is the forward sweep without autograd plus one
#: batched replay (:mod:`hsn.train.split`; needs the full A6 feedback cut); ``fused`` is ``split``
#: with the sweep's cell round as one fused CUDA op (:mod:`hsn.kernels.window`): experimental,
#: correct but slower than ``split`` as measured on 11 Nov 2026 (docs/p2_10_split_window.md).
WindowMode = Literal["autograd", "split", "fused"]
PowerMeter = Literal["nvml", "powermetrics", "wall", "none"]
ModelName = Literal["hsn", "gru", "transformer", "selective_ssm", "moe"]

#: Auxiliary loss terms that ``train.aux_enabled`` may name (Appendix A.3: cost / balance / graph).
#: P2.4 registers its term names here.
AUX_TERMS: tuple[str, ...] = ("cost", "balance", "graph")


#: P1.16: the rollback's keys (``train.*``), ``rollback_every`` first (None: off).
ROLLBACK_KEYS: tuple[str, ...] = (
    "rollback_every",
    "rollback_skip",
    "rollback_lr_mult",
    "rollback_lr_steps",
    "rollback_max",
)


def wsd_decay_steps(decay_fraction: float, steps: int) -> int:
    """P1.13: the length of the warmup-stable-decay schedule's final decay in optimiser steps,
    ``round(decay_fraction * steps)`` (``train.decay_fraction``, ``train.steps``)."""
    return round(decay_fraction * steps)


#: Hardware profile names (Appendix A.5 / E.1). ``gpu8gb`` gets its file in P4.3.
PROFILES: tuple[str, ...] = tuple(get_args(Profile))

#: Keys that every ``measured`` dict handed to :func:`write_run_manifest` must contain (§0.5
#: "measured, not assumed").
MEASURED_KEYS: tuple[str, ...] = (
    "param_count",
    "peak_bytes",
    "reserved_bytes",
    "tokens_per_s",
    "launches_per_step",
)

# python/hsn/config.py -> parents[2] is the repository root. This holds for the editable install
# that ``uv sync`` performs (uv.lock: hsn = { editable = "." }); a wheel install has no configs/.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
HARDWARE_PROFILE_DIR: Path = REPO_ROOT / "configs" / "hardware"

_PREFETCH_TOP_RE = re.compile(r"^\d+K$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NVCC_RELEASE_RE = re.compile(r"release\s+([0-9][0-9.]*)")


class _StrictModel(BaseModel):
    """Base for every config model: unknown keys raise, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


# --------------------------------------------------------------------------------------------------
# Appendix A.1  model.*
# --------------------------------------------------------------------------------------------------


class ScorerFeaturesConfig(_StrictModel):
    """``model.scorer_features``: router-scorer feature ablations."""

    exclude: list[str] = []  # feature names to zero (ablation; e.g. ["eps"])


class ModelConfig(_StrictModel):
    """Appendix A.1 ``model.*``; defaults are the Appendix C reference configuration."""

    vocab_size: int = Field(default=256, ge=1)  # byte vocabulary V
    d: int = Field(default=64, ge=1)  # width
    n_cells: int = Field(default=32, ge=1)  # N
    n_units: int = Field(default=8, ge=1)  # M
    k_select: int = Field(default=4, ge=1)  # K (cells per round); must be <= n_cells
    k_min: int = Field(default=1, ge=1)  # minimum selected cells
    r_max: int = Field(default=1, ge=1, le=3)  # internal rounds cap (1-3)
    in_degree: int = Field(default=4, ge=0)  # D_g; 0 disables propagation (ablation)
    random_edges: int = Field(default=0, ge=0)  # extra random candidate edges
    edge_norm: EdgeNorm = "softmax"  # normalisation of incoming edge weights
    hidden: int = Field(default=512, ge=1)  # H for every two-layer MLP
    n_context: int = Field(default=16, ge=1)  # context symbols
    context_dim: int = Field(default=64, ge=1)  # context embedding width (= d)
    activation: str = "gelu"  # MLP activation (paper unspecified)
    init: str = "trunc_normal_0.02"  # weight init of embeddings, scorer, halting, decoder
    # --- P1.5a (amendment A1, docs/amendments/A1_init_norm.md): the token-to-state path ---
    # init of E, P, S, the unit MLPs, V, W_out and the gate weight: "fan_in" = truncated normal
    # with std 1/sqrt(fan_in) per layer (default), or "trunc_normal_<std>" (the pre-A1 behaviour
    # with trunc_normal_0.02)
    state_init: str = "fan_in"
    # RMSNorm over the 4d cell input v before the unit MLPs and the gate ("none" = the P1.5 cell)
    cell_norm: Literal["rms", "none"] = "rms"
    # the gate's bias at init: sigmoid(-3.0) = 0.047, so cells retain 95 % per position at the start
    gate_bias_init: float = -3.0
    # --- end P1.5a ---
    # --- P1.13: RMSNorm (learned gain of width d) on W_out(.) before the candidate's tanh, c =
    # tanh(cand_norm(W_out . mixed)); "none" = the P1.5 candidate. HSN only (baselines refuse rms).
    cand_norm: Literal["none", "rms"] = "none"
    scorer: Scorer = "linear"  # router scorer
    scorer_hidden: int = Field(default=128, ge=1)  # hidden width when scorer = mlp
    scorer_features: ScorerFeaturesConfig = ScorerFeaturesConfig()
    routing: Routing = "dense"  # selection mode
    # Amendment A5: independently ablatable, routed HSN only; fixed codes have width d.
    key_address: bool = False
    selected_readout: bool = False
    # A5 extension (docs/amendments/A5_key_address.md, "The address columns' start"; opt-in): the
    # scorer's two address columns (a_cur, a_prev) start at this value for every unit instead of
    # the small random init of the other columns, so that from the first step a route follows the
    # codes of the current and the previous byte. They stay ordinary trainable weights. None
    # (default): the random init. Needs key_address and the linear scorer.
    address_init: float | None = Field(default=None, allow_inf_nan=False)
    # Training-only joint dropout of q/u at the decoder; selected cells remain readable.
    # A5 extension: docs/amendments/A5_stability_readout.md. Zero uses the original decoder.
    decoder_shortcut_dropout: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    fixed_route: list[int] = []  # cells for routing = fixed
    budget: float = Field(default=1.0, ge=0.0)  # B_t as a fraction of the dense step cost
    lambda_cost: float = Field(default=0.0, ge=0.0)  # lambda_cost in score = U - lambda * c
    deadline_ms: float | None = None  # wall-clock deadline (serving)
    # --- P2.1: routing cost profile (python/hsn/kernels/costs.py, python/hsn/model/router.py) ---
    # The cost profile (a P0.6 report with a `cost_profile` block, written by
    # hsn.kernels.costs.profile_dense_costs) the router's cost table is read from; a relative
    # path resolves against the repository root. null = the uniform placeholder table (not a
    # measurement; fine for dense runs and tests). Never a parameter count.
    cost_profile: Path | None = None
    # --- end P2.1 ---
    static_k: bool = True  # pad selection to fixed K (needed for graphs / compile)
    age_max: int = Field(default=64, ge=1)  # refresh-probe threshold
    halt_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # inference halting threshold
    predictor_target: PredictorTarget = "fixed_embedding"  # P1.6
    disabled_cells: list[int] = []  # test hook
    # P2.13's "remove shared path" ablation: remove u from v. The cell input is [eps; h_i; m_i]
    # (width 3d, :attr:`cell_input_width`) for the cell-input norm, the unit MLPs and the gate;
    # the scorer's features and the decoder still read u.
    ablate_shared_u: bool = False  # ablation: remove u from v
    state_schema_version: int = Field(default=1, ge=1)  # bumped on any state layout change

    @property
    def cell_input_width(self) -> int:
        """Width of the cell input ``v`` that the cell-input norm, the unit MLPs and the gate
        read: ``4d`` for ``[u; eps; h_i; m_i]``, ``3d`` for ``[eps; h_i; m_i]`` with
        ``ablate_shared_u``. Derived, not a configuration key (it is not dumped or hashed)."""
        return (3 if self.ablate_shared_u else 4) * self.d

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.decoder_shortcut_dropout and not self.selected_readout:
            raise ValueError("model.decoder_shortcut_dropout requires model.selected_readout")
        if self.address_init is not None and not (self.key_address and self.scorer == "linear"):
            raise ValueError(
                "model.address_init requires model.key_address=true and model.scorer=linear"
            )
        if self.routing == "dense" and (self.key_address or self.selected_readout):
            raise ValueError(
                "model.key_address and model.selected_readout require routed HSN; "
                "model.routing=dense makes both A5 features void"
            )
        if self.state_init != "fan_in" and not self.state_init.startswith("trunc_normal_"):
            raise ValueError(
                f"model.state_init={self.state_init!r} must be 'fan_in' or 'trunc_normal_<std>'"
            )
        if self.k_select > self.n_cells:
            raise ValueError(
                f"model.k_select={self.k_select} must be <= model.n_cells={self.n_cells}"
            )
        if self.k_min > self.k_select:
            raise ValueError(f"model.k_min={self.k_min} must be <= model.k_select={self.k_select}")
        if self.in_degree >= self.n_cells:
            raise ValueError(
                f"model.in_degree={self.in_degree} must be < model.n_cells={self.n_cells}"
            )
        if self.deadline_ms is not None and self.deadline_ms <= 0.0:
            raise ValueError(f"model.deadline_ms={self.deadline_ms} must be > 0 or null")
        for name, cells in (
            ("fixed_route", self.fixed_route),
            ("disabled_cells", self.disabled_cells),
        ):
            bad = [i for i in cells if i < 0 or i >= self.n_cells]
            if bad:
                raise ValueError(f"model.{name} has cell ids outside [0, {self.n_cells}): {bad}")
        if self.routing == "fixed" and not self.fixed_route:
            raise ValueError("model.routing=fixed requires a non-empty model.fixed_route")
        # --- P2.1: selection validation (Router.select enforces the same bounds at run time) ---
        if len(set(self.fixed_route)) != len(self.fixed_route):
            raise ValueError(f"model.fixed_route must list distinct cells, got {self.fixed_route}")
        if self.routing == "fixed" and not self.k_min <= len(self.fixed_route) <= self.k_select:
            raise ValueError(
                f"model.routing=fixed selects exactly the {len(self.fixed_route)} cells of "
                f"model.fixed_route, which must lie within [model.k_min, model.k_select] = "
                f"[{self.k_min}, {self.k_select}]"
            )
        if self.routing == "greedy_budget" and self.budget <= 0.0:
            raise ValueError(
                f"model.routing=greedy_budget needs model.budget > 0, got {self.budget}"
            )
        # --- end P2.1 ---
        return self


# --------------------------------------------------------------------------------------------------
# Appendix A.2  data.*
# --------------------------------------------------------------------------------------------------


class DataConfig(_StrictModel):
    """Appendix A.2 ``data.*``."""

    dataset: str = "synthetic_v1"  # dataset name
    manifest: Path | None = None  # split manifest with hashes (P0.4 / P4.1)
    tokenizer: Tokenizer = "bytes"  # only byte-level in this program
    tasks: list[str] = []  # task names or stage ids
    # P0.8: which P0.4 variants of the named tasks a run loads (train and dev alike); empty = every
    # variant the manifest has. The baselines and G1 train on ``[in_range]`` (plan P0.8, P1.9).
    variants: list[Variant] = []
    context_symbols: Path = Path("configs/data/context_symbols.yaml")  # id table
    split: Split = "train"  # guarded: test only in gate scripts
    # P0.5: start every document at a window boundary when streams are built (the tail of its
    # last window is padding). ``in_range`` (P0.4) means "the dependency fits one window", which
    # only holds when documents are aligned; off, a document may straddle a window boundary.
    align_docs: bool = True


# --------------------------------------------------------------------------------------------------
# Appendix A.3  train.*
# --------------------------------------------------------------------------------------------------


class AdaptiveLRConfig(_StrictModel):
    """Training-only feedback controller; disabled preserves existing training and hashes."""

    enabled: bool = False
    interval: int = Field(default=100, ge=1)
    monitor_docs: int = Field(default=32, ge=1)
    patience: int = Field(default=2, ge=1)
    cooldown: int = Field(default=2, ge=0)
    threshold: float = Field(default=0.01, gt=0.0, lt=1.0, allow_inf_nan=False)
    reduce_factor: float = Field(default=0.5, gt=0.0, lt=1.0, allow_inf_nan=False)
    increase_factor: float = Field(default=1.1, gt=1.0, allow_inf_nan=False)
    min_scale: float = Field(default=0.03125, gt=0.0, le=1.0, allow_inf_nan=False)
    max_scale: float = Field(default=1.0, ge=1.0, allow_inf_nan=False)
    grad_growth: float = Field(default=3.0, gt=1.0, allow_inf_nan=False)
    max_update_ratio: float = Field(default=0.01, gt=0.0, allow_inf_nan=False)
    router_max_update_ratio: float = Field(default=0.005, gt=0.0, allow_inf_nan=False)
    readout_max_update_ratio: float = Field(default=0.005, gt=0.0, allow_inf_nan=False)
    trial_windows: int = Field(default=2, ge=1)
    # Independent controller ablations; emergency guard/rollback cuts remain active.
    loss_cuts: bool = True
    gradient_cuts: bool = True
    update_cuts: bool = True
    trial_increases: bool = True


class RouteGuardConfig(_StrictModel):
    """``train.route_guard`` (A6, opt-in; :mod:`hsn.train.route_guard`): when an established
    route (a router unit's share of an input byte's positions) is being taken over by another
    unit, halve the gaining unit's applied router step, at whatever step it happens. Disabled
    preserves existing training and hashes."""

    enabled: bool = False
    # applied optimiser steps between two readings of the per-(input byte, unit) slot counts
    interval: int = Field(default=100, ge=1)
    # the event: the quarter of the input bytes whose established routes lost most have lost
    # this share of them on average
    lost_fraction: float = Field(default=0.1, gt=0.0, lt=1.0, allow_inf_nan=False)
    # a byte with fewer slots than this in an interval is left out of the reading
    min_slots: int = Field(default=100, ge=1)
    # a route (byte, unit) is established once the unit has held this share of the byte's slots
    # at `persistence` consecutive readings
    specialised: float = Field(default=0.25, gt=0.0, le=1.0, allow_inf_nan=False)
    persistence: int = Field(default=3, ge=1)
    # ... and only while training is no longer improving: the interval's mean training loss is
    # above the lowest interval mean so far by this fraction (units sorting the positions out
    # while the loss falls is learning, not a takeover)
    loss_rise: float = Field(default=0.02, ge=0.0, allow_inf_nan=False)
    # no event while the established routes hold less than this share of all slots
    min_usage: float = Field(default=0.001, ge=0.0, lt=1.0, allow_inf_nan=False)
    # the response: the step scale of the gaining unit's scorer rows is multiplied by this,
    # down to min_scale
    reduce_factor: float = Field(default=0.5, gt=0.0, lt=1.0, allow_inf_nan=False)
    min_scale: float = Field(default=0.015625, gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_restore(cls, data: Any) -> Any:
        """``restore`` (also rewind the gaining unit's rows one reading) existed on 10 and 11 Nov
        2026 and was removed after exp 400 (77.9 % at step 5,000 against 90.3 % without it). A
        stored configuration that has it switched off still loads and hashes as it did
        (:func:`_hash_view`); one that has it on cannot be run by this code."""
        if isinstance(data, Mapping) and "restore" in data:
            if data["restore"] not in (False, None):
                raise ValueError(
                    "train.route_guard.restore was removed (exp 400: worse than the guard "
                    "alone); the runs that used it are at commit adeb660"
                )
            data = {k: v for k, v in data.items() if k != "restore"}
        return data


class UpdateLimitGroup(_StrictModel):
    """One joint L2 trust region, selected by model parameter-name globs."""

    selectors: list[str] = Field(min_length=1)
    max_ratio: float = Field(gt=0.0, allow_inf_nan=False)


class UpdateLimitConfig(_StrictModel):
    """Cap proposed optimizer displacement; empty groups disable all work and state."""

    groups: dict[str, UpdateLimitGroup] = {}
    norm_floor: float = Field(default=1e-12, gt=0.0, allow_inf_nan=False)
    # For these groups, cap the radius's reference norm at this multiple of its initial norm.
    # Empty preserves the original current-weight-relative projection and checkpoint format.
    norm_growth_limits: dict[str, Annotated[float, Field(ge=1.0, allow_inf_nan=False)]] = {}

    @model_validator(mode="after")
    def _check_norm_growth_limits(self) -> Self:
        if set(self.norm_growth_limits) - set(self.groups):
            raise ValueError("update_limit.norm_growth_limits must name configured groups")
        return self


class GradientGrowthConfig(_StrictModel):
    """Selective, reversible update damping from sustained pre-clip gradient growth."""

    enabled: bool = False
    groups: dict[str, list[str]] = {}  # disjoint parameter-name globs, chosen by the experiment
    window: int = Field(default=100, ge=1)
    reference_lag: int = Field(default=500, ge=1)
    interval: int = Field(default=100, ge=1)  # observations per accepted finite samples
    growth_threshold: float = Field(default=1.5, gt=1.0, allow_inf_nan=False)
    global_growth_threshold: float = Field(default=1.25, gt=1.0, allow_inf_nan=False)
    min_share: float = Field(default=0.25, ge=0.0, le=1.0, allow_inf_nan=False)
    share_power: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    response_power: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    recovery_threshold: float = Field(default=1.1, ge=1.0, allow_inf_nan=False)
    recovery_rate: float = Field(default=0.1, gt=0.0, le=1.0, allow_inf_nan=False)
    min_scale: float = Field(default=0.25, gt=0.0, le=1.0, allow_inf_nan=False)
    norm_floor: float = Field(default=1e-12, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_growth(self) -> Self:
        if self.reference_lag < self.window:
            raise ValueError("gradient_growth.reference_lag must be >= window")
        if self.recovery_threshold >= self.growth_threshold:
            raise ValueError("gradient_growth.recovery_threshold must be below growth_threshold")
        if self.enabled and (not self.groups or any(not v for v in self.groups.values())):
            raise ValueError("gradient_growth.enabled requires nonempty parameter selectors")
        return self


class TrainConfig(_StrictModel):
    """Appendix A.3 ``train.*`` including loss weights and the one-at-a-time auxiliary rule."""

    # P2.6: unset resolves per model.routing (dense -> adamw, a selection mode -> lazy_adamw) in
    # RunConfig._default_optimizer_by_routing; an explicit value is kept.
    optimizer: Optimizer = "adamw"
    lr: float = Field(default=1e-3, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    schedule: str = "constant"  # constant | cosine | linear | wsd (P1.13)
    # --- P1.13: recurrent stability levers (python/hsn/train/loop.py) ---
    # schedule = wsd: the decay over the last round(decay_fraction * steps) steps (1 - sqrt to 0);
    # required with wsd, refused with any other schedule
    decay_fraction: float | None = None
    # the HSN cell's parameter group (every trainable parameter of HSN.cell): lr = lr * cell_lr_mult
    # and, on its tensors of ndim >= 2, weight decay cell_weight_decay (None = weight_decay)
    cell_lr_mult: float = Field(default=1.0, gt=0.0)
    cell_weight_decay: float | None = Field(default=None, ge=0.0)
    # A5 stability: HSN.router and the whole HSN.decoder get independent learning rates.
    # Defaults preserve the existing optimiser groups and checkpoint layout.
    router_lr_mult: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    readout_lr_mult: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    adaptive_lr: AdaptiveLRConfig = AdaptiveLRConfig()
    update_limit: UpdateLimitConfig = UpdateLimitConfig()
    gradient_growth: GradientGrowthConfig = GradientGrowthConfig()
    route_guard: RouteGuardConfig = RouteGuardConfig()  # A6, opt-in; hash-omitted at its default
    # --- end P1.13 ---
    # --- P1.15: the gradient-spike guard (python/hsn/train/loop.py). None: off (today's training).
    # Skip a step whose global gradient norm (before clipping) exceeds spike_factor x the median of
    # the last spike_window applied finite norms (once min(20, spike_window) are known); counted
    # in P1.12's stop rule. 100 sits at the edge of ordinary variation: on the stored runs it
    # would skip 55 steps in 17 of 108 non-failed runs, learners among them (180x to 572x steps
    # in exps 65, 154, 155, 161); exp 171's jump (7e10x) and the peak of exp 170's burst (5e14x)
    # are caught at any factor up to about 1e10. For experiments 1e3 to 1e4 is the safer choice
    # (only 3 runs skip anything at 1e4). spike_window is read only with spike_factor set (the
    # user's request, 8 Nov).
    spike_factor: float | None = Field(default=None, gt=1.0)
    spike_window: int = Field(default=100, ge=1)
    # Compare the recent spike_window median with the median slow_burst_window applied steps
    # earlier, as well as checking the current norm. Requires spike_factor; None disables it.
    slow_burst_factor: float | None = Field(default=None, gt=1.0, allow_inf_nan=False)
    slow_burst_window: int = Field(default=250, ge=1)
    # write a dump of the pre-step state at the first step a guard skips (non-finite, or a spike
    # with spike_factor set), for a deterministic replay (python/hsn/train/divergence.py)
    divergence_dump: bool = True
    divergence_max_dumps: int = Field(default=1, ge=1, le=32)
    divergence_healthy_every: int = Field(default=0, ge=0)
    divergence_healthy_keep: int = Field(default=2, ge=1, le=8)
    # --- end P1.15 ---
    # --- P1.16: rollback on a guard trigger (python/hsn/train/rollback.py). None: off (today's
    # training). Every rollback_every steps the trainer keeps an in-memory snapshot of the
    # parameters, buffers and optimiser state (a ring of two); a step the guards would skip
    # (non-finite, or a spike with spike_factor set) restores the older snapshot in place, discards
    # rollback_skip steps' worth of windows and starts fresh documents, and trains at lr *
    # rollback_lr_mult for rollback_lr_steps steps. The run stops (failure kind rollback) at its
    # rollback_max + 1-th trigger. A trigger before the first snapshot is P1.15's skip. The other
    # four keys are read only with rollback_every set. Recommended 50 / 10 / 0.5 / 200 / 5. All five
    # are left out of the configuration hash at their defaults (HASH_OMIT_AT_DEFAULT).
    rollback_every: int | None = Field(default=None, ge=1)
    rollback_skip: int = Field(default=0, ge=0)
    rollback_lr_mult: float = Field(default=1.0, gt=0.0, le=1.0)
    rollback_lr_steps: int = Field(default=0, ge=0)
    rollback_max: int = Field(default=5, ge=1)
    # oldest preserves the original recovery; guard_horizon uses the newest snapshot at or
    # before the trigger's reference horizon (cadence for sharp/nonfinite, full slow history).
    rollback_selection: Literal["oldest", "guard_horizon"] = "oldest"
    # --- end P1.16 ---
    unroll: int = Field(default=256, ge=1)  # TBPTT window
    batch: int = Field(default=4, ge=1)  # batching (P4.3)
    micro_batch: int = Field(default=4, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    precision: Precision = "bf16"  # autocast dtype (fp64 only for gradcheck)
    grad_clip: float = Field(default=1.0, ge=0.0)  # global norm clip
    # Opt-in wide norm reduction AND scaling; native preserves historical clipping bitwise.
    # MPS uses CPU float64. True NaN/Inf gradients still reach the existing nonfinite guard.
    grad_norm_precision: GradNormPrecision = "native"
    # A6 (docs/amendments/A6_feedback_gradient.md; opt-in, forward values unchanged): the feedback
    # reads of the recurrent state that are constants in the backward pass. "summary": S reads
    # q[t-1]; "eps": the cell and the scorer read eps = z - P(q[t-1]) (L_pred keeps its gradient
    # into P and q); "state": a cell reads its own h and its neighbours' messages; "router": the
    # scorer's features (u, eps, h). With all four the backward recurrence is the gated carry
    # (1 - b) alone, which cannot amplify a gradient. Empty (default): full BPTT, as before.
    feedback_detach: list[FeedbackPath] = []
    # A6 (same amendment; opt-in): the per-row limit on the gradient a position passes back into
    # the recurrent state. At every position each row's dL/dq is scaled to at most this norm and
    # its dL/dh to at most state_grad_clip / sqrt(n_cells) (the share a summary gradient gives the
    # cells). A row under the limit keeps its gradient unscaled, so a healthy window is plain
    # backpropagation through time; a row whose gradient blows up at a write no longer replaces
    # the batch's. None (default): off. Calibrated at B32 on the delayed task: healthy rows stay
    # below 0.16 in h (2.95 in q) from step 3,000 to 8,500 of exp 346. Tested (exps 354, 355): it
    # did not prevent collapses; not part of the recommended recipe (docs/p1_18_a6_results.md).
    state_grad_clip: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    # A6 (opt-in): the same limit without a scale. At every position a row's dL/dq (and its dL/dh)
    # is at most this multiple of the batch's typical live row there (the 90th percentile of the
    # rows that carry a gradient at that position). On exp 346's checkpoints from step 500 to
    # 8,500 a limit of 3.0 acts on 0 % to 0.9 % of the live row-positions, at most one document
    # per window (the isolated ones at 9x to 21x), at every stage of a run, which one absolute
    # number cannot do (healthy rows grow twentyfold). May be combined with state_grad_clip: the
    # smaller limit applies. None (default): off. Tested (exp 364): it did not remove the drops.
    state_grad_ratio: float | None = Field(default=None, gt=1.0, allow_inf_nan=False)
    # A6 (opt-in; hsn.train.weight_ema): the decay of an exponential moving average of the weights,
    # updated after every applied optimiser step (with a warm-up, so the average follows the
    # weights closely early in a run). Training itself is unchanged, step for step; evaluations
    # during the run, the final evaluation, the trained-state diagnostics and the probes score the
    # averaged weights. A checkpoint's "model" stays the training weights and its
    # extra["weight_ema"] carries the average (load_checkpoint(..., averaged=True) reads it). Any
    # model. None (default): off, every score is of the training weights as before.
    weight_ema: float | None = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)
    steps: int = Field(default=1000, ge=1)  # schedule
    eval_every: int = Field(default=100, ge=1)
    ckpt_every: int = Field(default=500, ge=1)
    stats_every: int = Field(default=50, ge=1)
    q_resync_every: int = Field(default=64, ge=1)  # windows between exact q recomputation
    lambda_p: float = 1.0  # loss weights (§2.6); only L_pred is on by default
    lambda_r: float = 0.0
    lambda_d: float = 0.0
    lambda_e: float = 0.0
    lambda_c: float = 0.0
    lambda_b: float = 0.0
    lambda_g: float = 0.0
    lambda_u: float = 0.0
    lambda_ponder: float = 0.0
    aux_enabled: list[str] = []  # which of cost / balance / graph are on (<= 1 unless justified)
    allow_multiple_aux: str = ""  # written justification enabling > 1
    gate_mode: GateMode = "soft"  # P2.2
    soft_until_step: int = Field(default=0, ge=0)  # P2.2 schedule values (neutral by default)
    temperature_start: float = Field(default=1.0, gt=0.0)
    temperature_end: float = Field(default=1.0, gt=0.0)
    temperature_steps: int = Field(default=0, ge=0)
    # Opt-in, forward values unchanged (docs/a6_takeover_followup.md): which scores the
    # straight-through unit weight of an executed slot differentiates. "softmax" (the default,
    # P2.2): the softmax over the cell's units, so raising the executed unit's weight lowers every
    # other unit's score at that position, and a unit with most of the slots at a kind of position
    # pushes the others down there. "executed": the executed unit's score alone (the other units'
    # scores are constants of that softmax): a unit that did not run gets no gradient from the
    # slot. Not adopted by any recipe: an amendment would have to do that.
    unit_credit: Literal["softmax", "executed"] = "softmax"
    # Opt-in, forward values unchanged (docs/a6_takeover_followup.md): which cells and units the
    # straight-through gradient of a routed training window can reach. "selected" (the default):
    # the sparse round computes the selected cells with their chosen unit only, so a cell or a
    # unit that was not selected gets no gradient: the router is told how useful its choice was,
    # never how useful another one would have been. "all": the training window runs the masked
    # round, the model's own reference for the sparse one (every cell computes its candidate
    # with the selection's unit weights over every unit; only the selected cells are written), so
    # the cell mask's and the unit weights' straight-through gradient covers every cell and
    # unit. Costs the dense step's time. Not adopted by any recipe.
    ste_scope: Literal["selected", "all"] = "selected"
    epsilon_start: float = Field(default=0.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.0, ge=0.0, le=1.0)
    epsilon_steps: int = Field(default=0, ge=0)
    p_dense: float = Field(default=0.0, ge=0.0, le=1.0)
    probe_fraction: float = Field(default=0.05, ge=0.0, le=1.0)  # P2.3
    n_probe: int = Field(default=4, ge=0)
    # None probes to the current window's end, allowing delayed writes to receive utility
    # supervision. Integers preserve the original fixed-horizon experiment protocol.
    probe_horizon: int | None = Field(default=16, ge=1)
    ema_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    replay_fraction: float = Field(default=0.25, ge=0.0, le=1.0)  # rows per window from replay
    distill_temperature: float = Field(default=2.0, gt=0.0)  # P3.2
    obsolete_label_versions: list[int] = []
    ewc_cap: float = Field(default=1.0, gt=0.0)  # P3.3
    ewc_batches: int = Field(default=64, ge=1)
    ewc_groups: dict[str, list[str]] = {}  # group name -> parameter-name prefixes
    pcgrad: bool = False  # P3.6
    pcgrad_eps: float = Field(default=1e-8, gt=0.0)
    interference_window: int = Field(default=32, ge=1)
    shared_lr_scale_on_conflict: float = Field(default=0.5, ge=0.0, le=1.0)
    activation_checkpoint: ActivationCheckpoint = "none"  # P4.3
    deterministic: bool = False  # deterministic kernels and backward
    # P0.5: hyperparameter-search trials every method gets (paper §12: equal search budgets).
    search_trials: int = Field(default=1, ge=1)
    # ---- P1.6: prediction loss with collapse protection (model.predictor_target = ema_encoder) --
    # EMA rate of the target encoder, p_t <- tau * p_t + (1 - tau) * p_online after each optimiser
    # step (1.0 freezes the target, 0.0 copies the online encoder every step).
    ema_tau: float = Field(default=0.99, ge=0.0, le=1.0)
    # Variance floor on the batch of online encodings: mean(relu(gamma - sqrt(var + eps))) per
    # dimension; 0.0 disables the term. Ignored with the fixed-embedding target.
    variance_floor_gamma: float = Field(default=1.0, ge=0.0)
    # ---- end P1.6 -------------------------------------------------------------------------------
    # ---- P2.2: selection gradients (python/hsn/train/schedules.py, hsn.train.tbptt) -------------
    # EMA rate of the REINFORCE reward baseline (gate_mode = reinforce): after every window
    # b <- decay * b + (1 - decay) * R, R = -(task loss of the window). Other gate modes ignore it.
    reinforce_baseline_decay: float = Field(default=0.9, ge=0.0, lt=1.0)
    # ---- end P2.2 -------------------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.gradient_growth.enabled and (
            self.adaptive_lr.enabled
            or self.rollback_every is not None
            or self.spike_factor is not None
        ):
            raise ValueError(
                "gradient_growth requires adaptive_lr, rollback and finite spike skipping off"
            )
        if (
            self.adaptive_lr.enabled
            and self.rollback_every is not None
            and (self.rollback_lr_steps != 0 or self.rollback_lr_mult != 1.0)
        ):
            raise ValueError(
                "adaptive_lr owns persistent recovery cuts: use rollback_lr_steps=0 and "
                "rollback_lr_mult=1 with adaptive_lr.enabled"
            )
        # --- P2.2: a schedule whose end value can never be reached is refused, not ignored ---
        for what in ("temperature", "epsilon"):
            start, end = getattr(self, f"{what}_start"), getattr(self, f"{what}_end")
            if getattr(self, f"{what}_steps") == 0 and start != end:
                raise ValueError(
                    f"train.{what}_steps=0 holds train.{what}_start={start} for the whole run, but "
                    f"train.{what}_end={end} differs and would never be reached; give a step count "
                    "or set the end equal to the start"
                )
        # --- end P2.2 ---
        unknown = [term for term in self.aux_enabled if term not in AUX_TERMS]
        if unknown:
            raise ValueError(f"train.aux_enabled names unknown terms {unknown}; known: {AUX_TERMS}")
        if len(set(self.aux_enabled)) != len(self.aux_enabled):
            raise ValueError(f"train.aux_enabled lists a term twice: {self.aux_enabled}")
        if len(self.aux_enabled) > 1 and not self.allow_multiple_aux.strip():
            raise ValueError(
                f"train.aux_enabled={self.aux_enabled} turns on more than one auxiliary loss; "
                "penalties are added one at a time (global rule 4). Set train.allow_multiple_aux "
                "to a written justification to override."
            )
        if not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"train.betas={self.betas} must lie in [0, 1)")
        # --- P1.13: the WSD decay length is read by schedule = wsd only ---
        if self.schedule == "wsd":
            fraction = self.decay_fraction
            if fraction is None:
                raise ValueError(
                    "train.schedule=wsd needs train.decay_fraction, the length of the final decay "
                    "as a fraction of train.steps (e.g. 0.2)"
                )
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"train.decay_fraction={fraction} must lie in (0, 1]")
            decay = wsd_decay_steps(fraction, self.steps)
            if decay < 1:
                raise ValueError(
                    f"train.decay_fraction={fraction} of train.steps={self.steps} rounds to "
                    f"{decay} decay steps; the decay needs at least one"
                )
            if self.steps - decay < self.warmup_steps:
                raise ValueError(
                    f"train.schedule=wsd: the decay over the last {decay} of {self.steps} steps "
                    f"would start at step {self.steps - decay}, inside the warm-up of "
                    f"train.warmup_steps={self.warmup_steps}; shorten one of them"
                )
        elif self.decay_fraction is not None:
            raise ValueError(
                f"train.decay_fraction={self.decay_fraction} is read by train.schedule=wsd only, "
                f"but train.schedule={self.schedule!r}; drop it or use wsd"
            )
        # --- end P1.13 ---
        # --- P1.15: the spike window is read by the spike guard only ---
        if self.spike_factor is None and self.spike_window != 100:
            raise ValueError(
                f"train.spike_window={self.spike_window} is read by the gradient-spike guard only, "
                "but train.spike_factor is not set; drop it or set the factor"
            )
        # --- end P1.15 ---
        if self.slow_burst_factor is not None and self.spike_factor is None:
            raise ValueError("train.slow_burst_factor requires train.spike_factor")
        if self.slow_burst_factor is None and self.slow_burst_window != 250:
            raise ValueError("train.slow_burst_window requires train.slow_burst_factor")
        # --- P1.16: the rollback's other keys are read only with rollback_every set ---
        if self.rollback_every is None:
            ignored = [
                f"train.{key}={getattr(self, key)}"
                for key in ROLLBACK_KEYS[1:]
                if getattr(self, key) != type(self).model_fields[key].default
            ]
            if self.rollback_selection != "oldest":
                ignored.append(f"train.rollback_selection={self.rollback_selection}")
            if ignored:
                raise ValueError(
                    f"{', '.join(ignored)} are read by the rollback only (P1.16), but "
                    "train.rollback_every is not set; drop them or set the cadence"
                )
        # --- end P1.16 ---
        if len(set(self.feedback_detach)) != len(self.feedback_detach):
            raise ValueError(
                f"train.feedback_detach={self.feedback_detach} names a path twice; known: "
                f"{list(FEEDBACK_PATHS)}"
            )
        if self.weight_ema is not None and self.rollback_every is not None:
            raise ValueError(
                f"train.weight_ema={self.weight_ema} with train.rollback_every="
                f"{self.rollback_every}: the rollback's snapshots do not hold the averaged "
                "weights; use one of them"
            )
        return self


# --------------------------------------------------------------------------------------------------
# Appendix A.4  replay.*, fast.*, growth.*, san.*, stream.*, serve.*
# --------------------------------------------------------------------------------------------------


class ReplayConfig(_StrictModel):
    """Appendix A.4 ``replay.*`` (P3.1)."""

    path: Path | None = None
    capacity: int = Field(default=100_000, ge=0)
    quotas: dict[str, int] = {}  # stratum -> minimum records
    near_dup_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    expiry_days: int | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.expiry_days is not None and self.expiry_days < 1:
            raise ValueError(f"replay.expiry_days={self.expiry_days} must be >= 1 or null")
        return self


class FastConfig(_StrictModel):
    """Appendix A.4 ``fast.*`` (P3.4)."""

    rank: int = Field(default=8, ge=1)
    rho: float = Field(default=1.0, gt=0.0)  # norm bound of the residual
    eta: float = Field(default=0.01, gt=0.0)  # local learning rate
    steps_max: int = Field(default=10, ge=1)
    retention_max_loss_pp: float = Field(default=1.0, ge=0.0)


class GrowthConfig(_StrictModel):
    """Appendix A.4 ``growth.*`` (P3.5 / P4.2)."""

    threshold_per_task: dict[str, float] = {}  # task -> accuracy threshold
    windows: int = Field(default=3, ge=1)
    cap: float = Field(default=0.25, ge=0.0)
    budget_per_stage: dict[str, int] = {}  # stage -> step budget
    quality_tolerance_pp: float = Field(default=1.0, ge=0.0)
    # Set by the growth controller from the *measured* count at stage start; never a stored number.
    starting_params: int | None = None


class SanConfig(_StrictModel):
    """Appendix A.4 ``san.*`` (P4.6)."""

    update_every_windows: int = Field(default=16, ge=1)
    validation_windows: int = Field(default=4, ge=1)
    max_edges: int = Field(default=128, ge=0)  # n_cells * in_degree for the reference configuration
    k_random: int = Field(default=4, ge=0)
    quality_tolerance_pp: float = Field(default=1.0, ge=0.0)


class StreamConfig(_StrictModel):
    """Appendix A.4 ``stream.*`` (P4.4). A hardware profile may override any of these keys."""

    enabled: bool = False
    device_budget_bytes: int = Field(default=0, ge=0)
    host_budget_bytes: int = Field(default=0, ge=0)
    policy: StreamPolicy = "utility_lru"
    hot_windows: int = Field(default=4, ge=1)
    prefetch_top: int | str = "2K"  # an integer or "<n>K" (a multiple of model.k_select)

    @field_validator("prefetch_top", mode="before")
    @classmethod
    def _digits_to_int(cls, value: object) -> object:
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return value

    @field_validator("prefetch_top")
    @classmethod
    def _check_prefetch_top(cls, value: int | str) -> int | str:
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"stream.prefetch_top={value} must be >= 0")
            return value
        if not _PREFETCH_TOP_RE.match(value):
            raise ValueError(
                f"stream.prefetch_top={value!r} must be a non-negative integer or '<n>K' "
                "(a multiple of model.k_select)"
            )
        return value


class ServeLimitsConfig(_StrictModel):
    """``serve.limits.*`` (P5.4 ``Limits``)."""

    max_input_bytes: int = Field(default=65_536, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    max_budget: float = Field(default=1.0, gt=0.0)
    max_deadline_ms: float = Field(default=1000.0, gt=0.0)
    rps_per_tenant: float = Field(default=10.0, gt=0.0)


class ServeConfig(_StrictModel):
    """Appendix A.4 ``serve.*`` (P5.3 / P5.4)."""

    persistence_enabled: bool = False
    state_ttl_s: int = Field(default=3600, ge=1)
    encryption: Encryption = "none"
    limits: ServeLimitsConfig = ServeLimitsConfig()
    capture_policy: CapturePolicy = "disabled"


# --------------------------------------------------------------------------------------------------
# baseline.*  (P0.5; not in Appendix A, which only describes HSN)
# --------------------------------------------------------------------------------------------------


class GRUBaselineConfig(_StrictModel):
    """``baseline.gru.*``: the GRU byte-level language model of P0.5.

    Defaults are the documented ≈2M-parameter setting for the reference data (``model.vocab_size``
    256, ``model.n_context`` 16): 1,977,856 parameters as measured by ``GRUBaseline.param_report``.
    """

    d_model: int = Field(default=384, ge=1)  # embedding and hidden width
    n_layers: int = Field(default=2, ge=1)  # stacked GRU layers
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)  # on embeddings, between layers, on output


class TransformerBaselineConfig(_StrictModel):
    """``baseline.transformer.*``: the causal Transformer of P0.5 (context window ``train.unroll``).

    Defaults are the documented ≈2M-parameter setting for the reference data and ``train.unroll``
    256: 1,930,624 parameters as measured by ``TransformerBaseline.param_report``.
    """

    d_model: int = Field(default=192, ge=1)  # width; must be a multiple of n_heads
    n_layers: int = Field(default=4, ge=1)  # pre-norm blocks
    n_heads: int = Field(default=4, ge=1)  # attention heads
    ffn_hidden: int = Field(default=768, ge=1)  # MLP hidden width (4 × d_model)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)  # embeddings, attention, residual branches

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"baseline.transformer.d_model={self.d_model} must be a multiple of "
                f"n_heads={self.n_heads}"
            )
        return self


class SelectiveSSMBaselineConfig(_StrictModel):
    """``baseline.selective_ssm.*``: the Mamba-style selective state-space model (the paper's
    "selective recurrent memory" control, plan P2.13; :mod:`hsn.baselines.mamba`).

    Defaults are a ≈2M-parameter setting for the reference data: 1,887,744 parameters as measured
    by ``MambaBaseline.param_report``.
    """

    d_model: int = Field(default=256, ge=1)  # residual width
    n_layers: int = Field(default=4, ge=1)  # Mamba blocks
    d_state: int = Field(default=16, ge=1)  # state per channel of the selective scan
    d_conv: int = Field(default=4, ge=1)  # taps of the causal depthwise convolution
    expand: int = Field(default=2, ge=1)  # inner width = expand × d_model
    dt_rank: int | None = Field(default=None, ge=1)  # rank of the step-size path; None = d_model/16


class BaselineConfig(_StrictModel):
    """``baseline.*``: sizes of the P0.5 baselines (``model_name`` selects which one runs)."""

    gru: GRUBaselineConfig = GRUBaselineConfig()
    transformer: TransformerBaselineConfig = TransformerBaselineConfig()
    selective_ssm: SelectiveSSMBaselineConfig = SelectiveSSMBaselineConfig()
    # The HSN's fixed byte codes (amendment A5, ``model.key_address``) as a baseline input: the
    # codes of the current and of the previous byte, frozen, through one learned projection
    # (:mod:`hsn.baselines.address`). Opt-in; a control for how much of the HSN's score on keys it
    # never trained on comes from that fixed code. ``address_dim`` 64 is the recipe's ``model.d``,
    # which with the same seed gives the HSN's own code table.
    address_input: bool = False
    address_dim: int = Field(default=64, ge=1)


# --------------------------------------------------------------------------------------------------
# Appendix A.5  hardware profile file
# --------------------------------------------------------------------------------------------------


class RecipeConfig(_StrictModel):
    """``hardware.recipe``: the training recipe recommended by ``hsn plan`` (P4.3)."""

    batch: int = Field(default=4, ge=1)
    micro_batch: int = Field(default=4, ge=1)
    unroll: int = Field(default=256, ge=1)
    activation_checkpoint: ActivationCheckpoint = "none"


class HardwareConfig(_StrictModel):
    """Appendix A.5: one ``configs/hardware/<profile>.yaml`` per profile (values: Appendix E)."""

    name: Profile
    device: Device
    autocast_dtype: AutocastDtype  # per §3.7
    master_dtype: MasterDtype = "fp32"
    vram_bytes: int = Field(ge=0)  # nominal device memory; 0 = host RAM only (measured separately)
    vram_cap_fraction: float = Field(default=1.0, gt=0.0, le=1.0)  # simulated cap for tests
    unified_memory: bool
    pinned_memory: bool
    compile_mode: CompileMode = "none"
    window: WindowMode = "autograd"  # P2.10: autograd | split | fused (hsn.train.split)
    # P2.10: the split window's forward sweep runs a twin of the model with its autocast weights
    # cast once per window (hsn.train.compile.SweepTwin; the values of eager autocast). False runs
    # the model's own compiled blocks, which cast at every position and whose fused arithmetic
    # rounds as the autograd window's does: about 11 ms per step slower, kept to compare the two.
    sweep_twin: bool = True
    nvcc_arch: list[str] = []  # e.g. [sm_86], [sm_121]
    power_meter: PowerMeter
    recipe: RecipeConfig = RecipeConfig()
    stream: dict[
        str, Any
    ] = {}  # profile-level overrides of stream.* (validated as a partial StreamConfig)

    @field_validator("stream")
    @classmethod
    def _check_stream_overrides(cls, value: dict[str, Any]) -> dict[str, Any]:
        validated = StreamConfig.model_validate(value)  # rejects unknown keys, coerces values
        return {key: getattr(validated, key) for key in value}

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.device == "cuda" and not self.nvcc_arch:
            raise ValueError("hardware.nvcc_arch must list at least one target when device=cuda")
        if self.device != "cuda" and self.nvcc_arch:
            raise ValueError(f"hardware.nvcc_arch={self.nvcc_arch} is only valid for device=cuda")
        return self


# --------------------------------------------------------------------------------------------------
# Appendix A.6  run level
# --------------------------------------------------------------------------------------------------


class RunConfig(_StrictModel):
    """The whole experiment configuration: one YAML file plus the referenced hardware profile."""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    replay: ReplayConfig = ReplayConfig()
    fast: FastConfig = FastConfig()
    growth: GrowthConfig = GrowthConfig()
    san: SanConfig = SanConfig()
    stream: StreamConfig = StreamConfig()
    serve: ServeConfig = ServeConfig()
    baseline: BaselineConfig = BaselineConfig()  # P0.5 baseline sizes (not in Appendix A)
    hardware: HardwareConfig  # resolved from `profile` by load_config, or given inline
    run_id: str = "default"
    seed: int = Field(default=0, ge=0)
    notes: str = ""
    profile: Profile = "cpu"  # name of the hardware profile
    model_name: ModelName = "hsn"

    # --- P2.6: the optimiser default follows the routing ---
    @model_validator(mode="before")
    @classmethod
    def _default_optimizer_by_routing(cls, data: Any) -> Any:
        """``train.optimizer`` left unset resolves to ``lazy_adamw`` when ``model.routing`` is a
        selection mode; dense data is returned untouched (the field default ``adamw`` applies) and
        an explicit value is always kept."""
        if not isinstance(data, dict):
            return data
        train = data.get("train")
        if train is not None and not isinstance(train, dict):
            return data  # a TrainConfig instance, or a type error for the field to report
        if train is not None and "optimizer" in train:
            return data
        model = data.get("model")
        if isinstance(model, dict):
            routing = model.get("routing", "dense")
        else:
            routing = getattr(model, "routing", "dense")
        if routing == "dense":
            return data
        return {**data, "train": {**(train or {}), "optimizer": "lazy_adamw"}}

    # --- end P2.6 ---

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.profile != self.hardware.name:
            raise ValueError(
                f"profile={self.profile!r} does not match hardware.name={self.hardware.name!r}"
            )
        if not _RUN_ID_RE.match(self.run_id):
            raise ValueError(
                f"run_id={self.run_id!r} must match {_RUN_ID_RE.pattern} (it names a directory)"
            )
        # --- P2.10: the split window is only defined inside its scope ---
        if self.hardware.window != "autograd":
            reason = split_window_scope_error(self)
            if reason is not None:
                raise ValueError(
                    f"hardware.window={self.hardware.window!r} is not available for this "
                    f"configuration: {reason}"
                )
        if self.baseline.address_input and self.model_name == "hsn":
            raise ValueError(
                "baseline.address_input is an input of the baselines; the HSN's own address "
                "features are model.key_address"
            )
        # --- P1.13: the cell group and the candidate norm act on the HSN cell only ---
        if self.model_name != "hsn":
            ignored = []
            if self.train.router_lr_mult != 1.0 or self.train.readout_lr_mult != 1.0:
                raise ValueError("train.router_lr_mult and train.readout_lr_mult are HSN-only")
            if self.model.key_address or self.model.selected_readout:
                raise ValueError(
                    "model.key_address and model.selected_readout are HSN-only A5 features"
                )
            if self.train.feedback_detach:
                raise ValueError("train.feedback_detach (A6) is HSN-only")
            if self.train.unit_credit != "softmax":
                raise ValueError("train.unit_credit is HSN-only")
            if self.train.ste_scope != "selected":
                raise ValueError("train.ste_scope is HSN-only")
            if self.train.route_guard.enabled:
                raise ValueError("train.route_guard (A6) is HSN-only")
            if self.train.state_grad_clip is not None or self.train.state_grad_ratio is not None:
                raise ValueError(
                    "train.state_grad_clip and train.state_grad_ratio (A6) are HSN-only"
                )
            if self.model.ablate_shared_u:
                raise ValueError(
                    "model.ablate_shared_u removes u from the HSN cell's input, but "
                    f"model_name={self.model_name!r} has no such cell; leave it at false"
                )
            if self.train.cell_lr_mult != 1.0:
                ignored.append(f"train.cell_lr_mult={self.train.cell_lr_mult}")
            if self.train.cell_weight_decay is not None:
                ignored.append(f"train.cell_weight_decay={self.train.cell_weight_decay}")
            if self.model.cand_norm != "none":
                ignored.append(f"model.cand_norm={self.model.cand_norm}")
            if ignored:
                raise ValueError(
                    f"{', '.join(ignored)} act on the HSN cell (P1.13), but "
                    f"model_name={self.model_name!r} has none; leave them at their defaults"
                )
        # --- end P1.13 ---
        # --- P2.2: routing controls that would be silently ignored are refused ---
        routing, train = self.model.routing, self.train
        if routing in ("dense", "random", "fixed") and train.router_lr_mult != 1.0:
            raise ValueError(
                "train.router_lr_mult requires learned routing (topk or greedy_budget)"
            )
        if routing in ("dense", "random", "fixed") and train.route_guard.enabled:
            raise ValueError("train.route_guard requires learned routing (topk or greedy_budget)")
        explores = train.epsilon_start > 0.0 or train.epsilon_end > 0.0
        if train.gate_mode == "reinforce" and routing != "topk":
            raise ValueError(
                f"train.gate_mode=reinforce samples cells by Gumbel-top-k and needs "
                f"model.routing=topk, got model.routing={routing!r}"
            )
        if routing == "dense" and (explores or train.p_dense > 0.0):
            raise ValueError(
                "model.routing=dense has no selection to explore or fall back from; set "
                "train.epsilon_start / train.epsilon_end / train.p_dense to 0 or choose a sparse "
                "routing"
            )
        if routing in ("random", "fixed") and explores:
            raise ValueError(
                f"model.routing={routing!r} is a control and takes no exploration; set "
                "train.epsilon_start and train.epsilon_end to 0"
            )
        # --- end P2.2 ---
        # --- P2.4: an auxiliary term that could not act is refused ---
        aux = train.aux_enabled
        needs_selection = [name for name in ("cost", "balance") if name in aux]
        if routing == "dense" and needs_selection:
            raise ValueError(
                f"train.aux_enabled={aux}: {needs_selection} penalise a selection (cost: the "
                "executed cost, balance: the routed assignments) but model.routing=dense makes "
                "none; choose a sparse routing or drop the term"
            )
        if "graph" in aux:
            model = self.model
            if model.edge_norm != "sigmoid":
                raise ValueError(
                    f"train.aux_enabled names 'graph' but model.edge_norm={model.edge_norm!r}: "
                    "under softmax Σ|a_ij| is the constant number of non-empty rows and carries "
                    "no gradient; set model.edge_norm=sigmoid"
                )
            if model.in_degree + model.random_edges == 0:
                raise ValueError(
                    "train.aux_enabled names 'graph' but model.in_degree + model.random_edges = 0 "
                    "gives an edgeless graph with nothing to shrink; drop the term"
                )
        # --- end P2.4 ---
        # --- P2.3: counterfactual utility (on when train.lambda_u > 0) that could not act ---
        if train.lambda_u < 0.0:
            raise ValueError(f"train.lambda_u must be >= 0, got {train.lambda_u}")
        if train.lambda_u > 0.0:
            if routing == "dense":
                raise ValueError(
                    "train.lambda_u > 0 regresses the scorer onto counterfactual probes of a "
                    "selection, but model.routing=dense makes none; choose a sparse routing or set "
                    "train.lambda_u to 0"
                )
            if train.probe_fraction == 0.0:
                raise ValueError(
                    "train.lambda_u > 0 but train.probe_fraction = 0: no window would be "
                    "probed and the utility term would silently stay zero"
                )
            if train.probe_horizon is not None and train.probe_horizon > train.unroll:
                raise ValueError(
                    f"train.probe_horizon={train.probe_horizon} exceeds the window "
                    f"train.unroll={train.unroll}; a probe's horizon lies inside its window"
                )
            if routing in ("topk", "greedy_budget") and train.gate_mode == "soft":
                raise ValueError(
                    f"train.lambda_u > 0 with train.gate_mode=soft: model.routing={routing!r} "
                    "would execute soft gates for the whole run, which have no hard "
                    "counterfactual, so no probe would ever run; use ste or reinforce (a soft "
                    "warm-up through train.soft_until_step is fine)"
                )
        # --- end P2.3 ---
        return self


# --------------------------------------------------------------------------------------------------
# Loading, profiles and dotted overrides
# --------------------------------------------------------------------------------------------------


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping (an empty file is an empty mapping)."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def profile_path(name: str) -> Path:
    """Path of the hardware profile file for ``name``; raises for unknown or missing profiles."""
    if name not in PROFILES:
        raise ValueError(f"unknown hardware profile {name!r}; known profiles: {list(PROFILES)}")
    path = HARDWARE_PROFILE_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"hardware profile {name!r} has no file at {path}")
    return path


def load_hardware_profile(name: str) -> HardwareConfig:
    """Load and validate ``configs/hardware/<name>.yaml``."""
    path = profile_path(name)
    try:
        return HardwareConfig.model_validate(_read_yaml_mapping(path))
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _parse_override(item: str) -> tuple[list[str], str]:
    """Split ``"a.b.c=value"`` into ``(["a", "b", "c"], "value")``."""
    key, sep, value = item.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"override {item!r} is not of the form key=value")
    parts = key.split(".")
    if not all(part.isidentifier() for part in parts):
        raise ValueError(f"override key {key!r} is not a dotted path of identifiers")
    return parts, value


def _unwrap_optional(annotation: Any) -> Any:
    """``X | None`` -> ``X``; anything else unchanged."""
    if get_origin(annotation) in (types.UnionType, Union):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _takes_raw_string(annotation: Any) -> bool:
    """Fields whose override text is taken verbatim instead of being parsed as YAML."""
    if annotation is str or annotation is Path:
        return True
    if get_origin(annotation) is Literal:
        return all(isinstance(arg, str) for arg in get_args(annotation))
    return False


def _yaml_scalar(text: str) -> Any:
    """Parse override text as a YAML value; unparseable text stays a string."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _child_mapping(node: dict[str, Any], key: str, dotted: str) -> dict[str, Any]:
    child = node.get(key)
    if child is None:
        child = node[key] = {}
    elif not isinstance(child, dict):
        raise ValueError(f"config key {dotted!r} is not a mapping; cannot set a sub-key of it")
    return child


def _apply_override(raw: dict[str, Any], parts: Sequence[str], value: str) -> None:
    """Set ``raw[parts[0]]...[parts[-1]]`` from command-line text, typed by the schema."""
    schema: type[BaseModel] | None = RunConfig  # schema node for `node`; None inside a dict field
    node = raw
    for depth, part in enumerate(parts):
        dotted = ".".join(parts[: depth + 1])
        last = depth == len(parts) - 1
        if schema is None:  # inside a dict-typed field: remaining parts are dict keys
            if last:
                node[part] = _yaml_scalar(value)
                return
            node = _child_mapping(node, part, dotted)
            continue
        field = schema.model_fields.get(part)
        if field is None:
            raise ValueError(f"unknown config key {dotted!r}")
        annotation = _unwrap_optional(field.annotation)
        if last:
            node[part] = value if _takes_raw_string(annotation) else _yaml_scalar(value)
            return
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            node = _child_mapping(node, part, dotted)
            schema = annotation
        elif get_origin(annotation) is dict:
            node = _child_mapping(node, part, dotted)
            schema = None
        else:
            raise ValueError(f"config key {dotted!r} has no sub-keys")


def _resolve_hardware(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """The raw hardware mapping: inline, a path relative to the config file, or the profile."""
    spec = raw.get("hardware")
    if spec is None:
        profile = raw.get("profile", RunConfig.model_fields["profile"].default)
        if not isinstance(profile, str):
            raise ValueError(f"{config_path}: profile must be a string, got {profile!r}")
        return _read_yaml_mapping(profile_path(profile))
    if isinstance(spec, str):
        path = Path(spec)
        if not path.is_absolute():
            path = config_path.parent / path
        return _read_yaml_mapping(path)
    if isinstance(spec, dict):
        return spec
    raise ValueError(
        f"{config_path}: hardware must be a mapping or a path, got {type(spec).__name__}"
    )


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> RunConfig:
    """Load and validate a run configuration.

    ``overrides`` are dotted ``key=value`` strings such as ``"model.k_select=4"``. Precedence,
    lowest first: schema defaults < hardware profile (``hardware.stream`` for ``stream.*``) < the
    YAML file < overrides. ``profile=<name>`` and ``hardware=<path>`` overrides select the profile
    before it is loaded; ``hardware.*`` overrides apply to the profile before its ``stream`` keys
    are merged.
    """
    config_path = Path(path)
    raw = _read_yaml_mapping(config_path)
    parsed = [_parse_override(item) for item in overrides]

    for parts, value in parsed:
        if parts in (["profile"], ["hardware"]):
            raw[parts[0]] = value
    raw["hardware"] = _resolve_hardware(raw, config_path)
    for parts, value in parsed:
        if parts[0] == "hardware" and len(parts) > 1:
            _apply_override(raw, parts, value)

    profile_stream = raw["hardware"].get("stream") or {}
    run_stream = raw.get("stream") or {}
    if not isinstance(profile_stream, dict) or not isinstance(run_stream, dict):
        raise ValueError(f"{config_path}: stream and hardware.stream must be mappings")
    raw["stream"] = {**profile_stream, **run_stream}

    for parts, value in parsed:
        if parts in (["profile"], ["hardware"]) or parts[0] == "hardware":
            continue
        _apply_override(raw, parts, value)

    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{config_path}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Hashing and the resolved dump
# --------------------------------------------------------------------------------------------------


#: P1.16: keys left out of :func:`canonical_json` (so of :func:`config_hash`) while they hold their
#: schema default, so that adding them changed no existing configuration's hash: checkpoints and
#: divergence dumps written before them still resume and replay (``Trainer.resume`` compares
#: hashes). A configuration that sets one of them to another value hashes it as usual. Dotted paths
#: into the dump; append only (removing a key changes the hash of every configuration that sets it).
HASH_OMIT_AT_DEFAULT: tuple[str, ...] = (
    *tuple(f"train.{key}" for key in ROLLBACK_KEYS),
    "model.key_address",
    "model.selected_readout",
    "train.router_lr_mult",
    "train.readout_lr_mult",
    "train.slow_burst_factor",
    "train.slow_burst_window",
    "train.adaptive_lr",
    "train.adaptive_lr.loss_cuts",
    "train.adaptive_lr.gradient_cuts",
    "train.adaptive_lr.update_cuts",
    "train.adaptive_lr.trial_increases",
    "train.divergence_max_dumps",
    "train.divergence_healthy_every",
    "train.divergence_healthy_keep",
    "train.rollback_selection",
    "train.update_limit",
    "train.update_limit.norm_growth_limits",
    "model.decoder_shortcut_dropout",
    "train.gradient_growth",
    "train.grad_norm_precision",
    "train.feedback_detach",
    "train.state_grad_clip",
    "model.address_init",
    "train.state_grad_ratio",
    "train.weight_ema",
    "train.route_guard",
    "hardware.window",
    "hardware.sweep_twin",
    "train.unit_credit",
    "train.ste_scope",
    "baseline.selective_ssm",
    "baseline.address_input",
    "baseline.address_dim",
)


def split_window_scope_error(cfg: RunConfig) -> str | None:
    """Why ``cfg`` cannot train with the split window (``hardware.window`` = ``split`` or
    ``fused``, P2.10, :mod:`hsn.train.split`), or ``None`` when it can: the window is only
    defined where the heavy backward work is local to its position (the full A6 feedback cut)
    and for the routed hard-gate round it replays. A soft warm-up (``train.soft_until_step``)
    is inside the scope: its steps run every cell with autograd, which is the autograd window's
    work, so they take that window and the split window takes over at the first hard-gate step
    (:meth:`hsn.train.split.SplitWindow.splits_now`)."""
    model, train = cfg.model, cfg.train
    if cfg.model_name != "hsn":
        return f"model_name={cfg.model_name!r} is not the HSN"
    if model.routing != "topk":
        return f"model.routing={model.routing!r}: the split window needs topk"
    if train.gate_mode != "ste":
        return f"train.gate_mode={train.gate_mode!r}: the split window needs ste"
    if train.ste_scope != "selected":
        return f"train.ste_scope={train.ste_scope!r}: the masked round runs every cell"
    missing = [name for name in FEEDBACK_PATHS if name not in train.feedback_detach]
    if missing:
        return (
            f"train.feedback_detach lacks {missing}: without the full feedback cut the heavy "
            "backward work is not local to its position"
        )
    if train.epsilon_start > 0.0 or train.epsilon_end > 0.0:
        return "train.epsilon_*: exploration draws are not replayed"
    if train.lambda_u > 0.0:
        return "train.lambda_u > 0: the counterfactual probes are untested with the split window"
    if train.state_grad_clip is not None or train.state_grad_ratio is not None:
        return "train.state_grad_clip / state_grad_ratio change the state recurrence's backward"
    if model.disabled_cells:
        return f"model.disabled_cells={list(model.disabled_cells)} is a test hook of the full path"
    if model.r_max != 1:
        return f"model.r_max={model.r_max}: one round per position only"
    if train.precision == "fp16":
        return "train.precision=fp16 scales the loss (GradScaler)"
    if train.grad_accum != 1:
        return f"train.grad_accum={train.grad_accum}: one window per step only"
    return None


def _hash_view(cfg: RunConfig) -> dict[str, Any]:
    """The configuration dump :func:`canonical_json` hashes: :meth:`RunConfig.model_dump` without
    the :data:`HASH_OMIT_AT_DEFAULT` keys that hold their default."""
    dump = cfg.model_dump(mode="json")
    for dotted in HASH_OMIT_AT_DEFAULT:
        parts = dotted.split(".")
        node = dump
        schema: type[BaseModel] = RunConfig
        for part in parts[:-1]:
            if part not in node:  # an enclosing default object was already omitted
                break
            node = node[part]
            nested = schema.model_fields[part].annotation
            assert isinstance(nested, type) and issubclass(nested, BaseModel)
            schema = nested
        else:
            key = parts[-1]
            default = schema.model_fields[key].default
            if isinstance(default, BaseModel):
                default = default.model_dump(mode="json")
            if node[key] == default:
                del node[key]
    guard = dump.get("train", {}).get("route_guard")
    if guard is not None:
        # the block was hashed with this key while the option existed (RouteGuardConfig); it
        # stays in the view so that the guarded runs of 10 and 11 Nov 2026 keep their hashes
        # and their kept checkpoints resume
        guard["restore"] = False
    return dump


def canonical_json(cfg: RunConfig) -> str:
    """Canonical JSON of the resolved configuration: sorted keys, compact, ASCII, no NaN (P1.16:
    without the :data:`HASH_OMIT_AT_DEFAULT` keys at their defaults)."""
    return json.dumps(
        _hash_view(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(cfg: RunConfig) -> str:
    """blake2b (32-byte digest, hex) of :func:`canonical_json`; independent of YAML key order."""
    return hashlib.blake2b(canonical_json(cfg).encode(), digest_size=32).hexdigest()


def dump_resolved(cfg: RunConfig, path: Path) -> None:
    """Write the resolved configuration (hardware inline) as YAML; it reloads to an equal config."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        cfg.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )
    header = (
        f"# HSN resolved configuration (python/hsn/config.py). config_hash: {config_hash(cfg)}\n"
    )
    out.write_text(header + body, encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# Run manifest
# --------------------------------------------------------------------------------------------------


@dataclass
class RunManifest:
    """``runs/<run_id>/run_manifest.json``: environment provenance plus *measured* numbers."""

    git_rev: str | None
    dirty: bool
    lock_hash: str | None  # blake2b of uv.lock
    python: str
    torch: str | None
    cuda_toolkit: str | None  # nvcc release string when nvcc is on PATH
    driver: str | None  # NVIDIA driver version when nvidia-smi is on PATH
    profile: str
    measured: dict[
        str, float
    ]  # param_count, peak_bytes, reserved_bytes, tokens_per_s, launches_per_step, ...
    config_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _run_command(command: list[str], cwd: Path | None = None) -> str | None:
    """stdout of ``command`` when the executable exists and exits 0; otherwise None."""
    if shutil.which(command[0]) is None:
        return None
    try:
        proc = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_state(root: Path) -> tuple[str | None, bool]:
    rev = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    if not rev:
        return None, False
    status = _run_command(["git", "status", "--porcelain"], cwd=root)
    return rev, bool(status)


def _lock_hash(root: Path) -> str | None:
    lock = root / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.blake2b(lock.read_bytes(), digest_size=32).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None


def _cuda_toolkit() -> str | None:
    out = _run_command(["nvcc", "--version"])
    if not out:
        return None
    match = _NVCC_RELEASE_RE.search(out)
    return match.group(1) if match else out.splitlines()[-1].strip()


def _nvidia_driver() -> str | None:
    out = _run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return None
    return out.splitlines()[0].strip()


def write_run_manifest(run_dir: Path, cfg: RunConfig, measured: Mapping[str, Any]) -> Path:
    """Write ``run_manifest.json`` and ``config.resolved.yaml`` into ``run_dir``.

    Returns the manifest path.

    ``measured`` must contain :data:`MEASURED_KEYS` with numeric values (bool is rejected); extra
    measured quantities are kept. Nothing is written when validation fails.
    """
    missing = [key for key in MEASURED_KEYS if key not in measured]
    if missing:
        raise ValueError(
            f"measured is missing required keys {missing}; required: {list(MEASURED_KEYS)}"
        )
    checked: dict[str, float] = {}
    for key, value in measured.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"measured[{key!r}] must be a number, got {type(value).__name__}")
        checked[key] = value

    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved(cfg, out_dir / "config.resolved.yaml")
    git_rev, dirty = _git_state(REPO_ROOT)
    manifest = RunManifest(
        git_rev=git_rev,
        dirty=dirty,
        lock_hash=_lock_hash(REPO_ROOT),
        python=platform.python_version(),
        torch=_package_version("torch"),
        cuda_toolkit=_cuda_toolkit(),
        driver=_nvidia_driver(),
        profile=cfg.profile,
        measured=checked,
        config_hash=config_hash(cfg),
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest_path
