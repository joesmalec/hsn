"""Profiled routing costs (P2.1): the cost table the router charges per selected (cell, unit),
read from a P0.6 profiler report and normalised so that the dense all-cells step costs 1.0.

Plan §2.5: ``c_e`` is the *profiled* execution cost of unit ``j`` plus the per-cell overhead
(messages, gate); never a parameter count. The budget ``B_t`` includes the shared path. The scorer
is its own line (global rule 5) and is kept out of the normaliser.

**Cost profile.** A cost profile is a P0.6 report (:meth:`hsn.profiler.ProfileReport.to_dict`:
every top-level field intact, ``tokens`` meaning *positions processed* as P0.6 defines it) plus a
top-level :data:`COST_PROFILE_KEY` block that records what was profiled::

    "cost_profile": {"schema": 1, "producer": "hsn.kernels.costs.profile_dense_costs",
                     "n_cells": 32, "n_units": 8, "d": 64, "hidden": 512,
                     "batch": 4, "positions": 16, "profile": "cpu", "device": "cpu",
                     "dtype": "torch.float32"}

Its five segments (:data:`COST_PROFILE_SEGMENTS`) are flat siblings, one position's work each,
with the same ``tokens`` (``batch × positions``), so no row contains another and "scorer excluded"
is unambiguous. The **cost-model partition** of :func:`profile_dense_costs` is

* ``shared_path``: ``P(q)``, ``E(emb(x))``, ``ε``, ``S([z; q; ctx])`` — once per position;
* ``scorer``: the features, the logits and the dense ``softmax`` over units — once per position;
* ``messages``: everything per cell that is not the unit MLP — the messages ``m``, the cell input
  ``v`` and its RMSNorm (P1.5a), the gate ``b``, the mixing, ``W_out``, ``tanh`` and the gated
  update (two entries per
  position, before and after the unit pass; the chunk's "per-cell overhead (messages, gate)");
* ``selected_cells``: the unit bank only (``UnitBank.forward_all``: every unit on every cell);
* ``decode``: ``D([q_t; u_t])`` — once per position.

A P2.8 *step* profile has a different timing partition and no provenance block, and
:meth:`CostTable.from_profile` refuses it rather than misreading it.

**Derivation** (per position ``ms(seg) = device_ms / tokens``; ``N``, ``M`` from the block)::

    dense         = ms(shared_path) + ms(decode) + ms(messages) + ms(selected_cells)  # no scorer
    shared        = (ms(shared_path) + ms(decode)) / dense
    cell_overhead = ms(messages) / (N · dense)
    unit[j]       = ms(selected_cells) / (N · M · dense)        for every j
    scorer        = ms(scorer) / dense

so that ``shared + N · (cell_overhead + Σ_j unit[j]) = 1.0`` (:meth:`CostTable.dense_cost`). The
per-unit vector is uniform within one profile: the ``M`` units share one shape (``UnitBank``
stacks ``[M, 4d, H]``) and the dense einsum cannot separate them; a profile that times units one
by one only changes :meth:`CostTable.from_profile`. The numbers are dense-path marginal costs at
the profiled batch; at this scale every path is launch-bound (plan A.2), so a selection's
``cost`` is an accounting quantity in dense-step units, not a latency prediction — the profiler
measures latency.

**Provenance.** :attr:`CostTable.hash` is blake2b-256 of a record holding the profile file's own
blake2b, the shapes (``N``, ``M``, ``d``, ``H``), the batch and position count, the profile and
device names and the derived numbers, so a table taken before a unit-shape or profile change
hashes differently (plan P2.1 pitfall: regenerate the table whenever the unit shape or the
profile changes; the run manifest records the hash through ``Router.cost_table_hash``). Without a
profile (``model.cost_profile: null``) :meth:`CostTable.uniform` gives a labelled placeholder —
``shared = 0``, ``cell_overhead = 0``, every unit ``1 / (N · M)`` — which is not a measurement and
says so in its ``source``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from hsn.config import REPO_ROOT, ModelConfig
from hsn.device import DeviceContext
from hsn.profiler import Profiler

if TYPE_CHECKING:  # hsn.model.hsn imports hsn.model.router, which imports this module
    from hsn.model.hsn import HSN

__all__ = [
    "COST_PROFILE_KEY",
    "COST_PROFILE_PRODUCER",
    "COST_PROFILE_SCHEMA",
    "COST_PROFILE_SEGMENTS",
    "UNIFORM_SOURCE",
    "CostTable",
    "cost_table_from_config",
    "load_cost_table",
    "profile_dense_costs",
    "write_cost_profile",
]

#: The segments a cost profile must contain, each with the same ``tokens``.
COST_PROFILE_SEGMENTS: tuple[str, ...] = (
    "shared_path",
    "scorer",
    "messages",
    "selected_cells",
    "decode",
)
#: Top-level key of the provenance block that marks a P0.6 report as a cost profile.
COST_PROFILE_KEY = "cost_profile"
COST_PROFILE_SCHEMA = 1
COST_PROFILE_PRODUCER = "hsn.kernels.costs.profile_dense_costs"
#: ``CostTable.source`` of the placeholder table used when no profile is configured.
UNIFORM_SOURCE = "uniform placeholder (not profiled)"

_BLOCK_INT_KEYS = ("n_cells", "n_units", "d", "hidden", "batch", "positions")
_BLOCK_STR_KEYS = ("producer", "profile", "device", "dtype")
_HEX = frozenset("0123456789abcdef")


def _hash_record(record: dict[str, Any]) -> str:
    """blake2b-256 (hex) of the canonical JSON of ``record``."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _check_fraction(name: str, value: float) -> float:
    if not _is_number(value) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"CostTable.{name} must be a finite number >= 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class CostTable:
    """The profiled costs of one model shape, in units of its dense all-cells step.

    ``unit [M]`` (float64, CPU) is the cost of running unit ``j`` on one cell; ``cell_overhead``
    the per-selected-cell work that is not the unit MLP; ``shared`` the shared path (``E``, ``P``,
    ``S``, ``D``) per position; ``scorer`` the scorer per position, reported beside the others and
    never inside a selection's cost. ``dense_ms_per_position`` is the measured normaliser (``None``
    for the uniform placeholder). ``source`` says where the numbers came from and ``hash`` is the
    provenance digest (module docstring).
    """

    unit: Tensor
    cell_overhead: float
    shared: float
    scorer: float
    n_units: int
    n_cells: int
    dense_ms_per_position: float | None
    source: str
    hash: str

    def __post_init__(self) -> None:
        for name in ("n_units", "n_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"CostTable.{name} must be an int >= 1, got {value!r}")
        if not isinstance(self.unit, Tensor):
            raise TypeError(
                f"CostTable.unit must be a torch.Tensor, got {type(self.unit).__name__}"
            )
        if self.unit.shape != (self.n_units,):
            raise ValueError(
                f"CostTable.unit must have shape [M] = {(self.n_units,)}, got "
                f"{tuple(self.unit.shape)}"
            )
        if self.unit.dtype != torch.float64 or self.unit.device.type != "cpu":
            raise ValueError(
                f"CostTable.unit must be a float64 CPU tensor, got {self.unit.dtype} on "
                f"{self.unit.device}"
            )
        if not bool(torch.isfinite(self.unit).all()) or bool((self.unit < 0).any()):
            raise ValueError("CostTable.unit must be finite and >= 0")
        for name in ("cell_overhead", "shared", "scorer"):
            _check_fraction(name, getattr(self, name))
        if self.dense_ms_per_position is not None and (
            not _is_number(self.dense_ms_per_position)
            or not math.isfinite(self.dense_ms_per_position)
            or self.dense_ms_per_position <= 0.0
        ):
            raise ValueError(
                "CostTable.dense_ms_per_position must be a finite number > 0 or None, got "
                f"{self.dense_ms_per_position!r}"
            )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("CostTable.source must be a non-empty string")
        if not isinstance(self.hash, str) or len(self.hash) != 64 or not set(self.hash) <= _HEX:
            raise ValueError(f"CostTable.hash must be 64 hex characters, got {self.hash!r}")

    # -- accounting -------------------------------------------------------------------------------

    def dense_cost(self) -> float:
        """``shared + N · (cell_overhead + Σ_j unit[j])``: the dense step in table units (1.0 for a
        profiled table, up to rounding)."""
        return self.shared + self.n_cells * (self.cell_overhead + float(self.unit.sum()))

    def max_pair_cost(self) -> float:
        """The dearest (cell, unit) execution: ``cell_overhead + max_j unit[j]``."""
        return self.cell_overhead + float(self.unit.max())

    def to_record(self) -> dict[str, Any]:
        """A JSON-ready copy of the table (for logs and manifests; not an input format)."""
        return {
            "unit": [float(v) for v in self.unit.tolist()],
            "cell_overhead": self.cell_overhead,
            "shared": self.shared,
            "scorer": self.scorer,
            "n_units": self.n_units,
            "n_cells": self.n_cells,
            "dense_ms_per_position": self.dense_ms_per_position,
            "source": self.source,
            "hash": self.hash,
        }

    # -- construction -----------------------------------------------------------------------------

    @classmethod
    def uniform(cls, n_units: int, n_cells: int) -> CostTable:
        """The labelled placeholder: no shared or per-cell cost, every unit ``1 / (N · M)``."""
        for name, value in (("n_units", n_units), ("n_cells", n_cells)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"CostTable.uniform: {name} must be an int >= 1, got {value!r}")
        unit = torch.full((n_units,), 1.0 / (n_units * n_cells), dtype=torch.float64)
        digest = _hash_record({"source": "uniform", "n_cells": n_cells, "n_units": n_units})
        return cls(
            unit=unit,
            cell_overhead=0.0,
            shared=0.0,
            scorer=0.0,
            n_units=n_units,
            n_cells=n_cells,
            dense_ms_per_position=None,
            source=UNIFORM_SOURCE,
            hash=digest,
        )

    @classmethod
    def from_profile(cls, path: Path | str, n_units: int) -> CostTable:
        """Read a cost profile (module docstring) and derive the table for ``n_units`` units.

        Raises ``ValueError`` (naming the file) for anything that is not a complete, consistent
        cost profile: no provenance block, a missing segment, ``tokens == 0``, ``device_ms <= 0``,
        a non-finite value, segments with different ``tokens``, a block whose shapes disagree with
        the report, or ``n_units`` different from the profiled ``M`` (regenerate the profile).
        """
        if isinstance(n_units, bool) or not isinstance(n_units, int) or n_units < 1:
            raise ValueError(f"n_units must be an int >= 1, got {n_units!r}")
        file = Path(path)
        raw = file.read_bytes()
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{file}: invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{file}: a cost profile is a JSON object, got {type(data).__name__}")
        block = _check_block(file, data)
        n_cells, profiled_units = block["n_cells"], block["n_units"]
        if profiled_units != n_units:
            raise ValueError(
                f"{file}: the cost profile was taken with M = {profiled_units} units, this model "
                f"has M = {n_units}; regenerate the cost profile (profile_dense_costs)"
            )
        per = _ms_per_position(file, data, block)
        dense = per["shared_path"] + per["decode"] + per["messages"] + per["selected_cells"]
        if not math.isfinite(dense) or dense <= 0.0:
            raise ValueError(f"{file}: the dense step measures {dense!r} ms per position")
        shared = (per["shared_path"] + per["decode"]) / dense
        cell_overhead = per["messages"] / (n_cells * dense)
        unit_cost = per["selected_cells"] / (n_cells * n_units * dense)
        scorer = per["scorer"] / dense
        unit = torch.full((n_units,), unit_cost, dtype=torch.float64)
        record = {
            "schema": COST_PROFILE_SCHEMA,
            "profile_blake2b": hashlib.blake2b(raw, digest_size=32).hexdigest(),
            **{key: block[key] for key in (*_BLOCK_INT_KEYS, "profile", "device", "dtype")},
            "unit": [unit_cost] * n_units,
            "cell_overhead": cell_overhead,
            "shared": shared,
            "scorer": scorer,
            "dense_ms_per_position": dense,
        }
        table = cls(
            unit=unit,
            cell_overhead=cell_overhead,
            shared=shared,
            scorer=scorer,
            n_units=n_units,
            n_cells=n_cells,
            dense_ms_per_position=dense,
            source=f"profile:{file}",
            hash=_hash_record(record),
        )
        if abs(table.dense_cost() - 1.0) > 1e-9:  # pragma: no cover - the derivation above
            raise RuntimeError(f"{file}: normalisation failed: dense_cost = {table.dense_cost()!r}")
        return table


def _check_block(file: Path, data: dict[str, Any]) -> dict[str, Any]:
    block = data.get(COST_PROFILE_KEY)
    if not isinstance(block, dict):
        raise ValueError(
            f"{file}: not a cost profile: no {COST_PROFILE_KEY!r} block. Produce one with "
            "hsn.kernels.costs.profile_dense_costs; a step profile (P2.8) has a different timing "
            "partition and cannot be read as a cost table"
        )
    if block.get("schema") != COST_PROFILE_SCHEMA:
        raise ValueError(
            f"{file}: {COST_PROFILE_KEY}.schema = {block.get('schema')!r}; this reader understands "
            f"schema {COST_PROFILE_SCHEMA}"
        )
    for key in _BLOCK_INT_KEYS:
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{file}: {COST_PROFILE_KEY}.{key} must be an int >= 1, got {value!r}")
    for key in _BLOCK_STR_KEYS:
        if not isinstance(block.get(key), str) or not block[key]:
            raise ValueError(
                f"{file}: {COST_PROFILE_KEY}.{key} must be a non-empty string, got "
                f"{block.get(key)!r}"
            )
    return block


def _ms_per_position(file: Path, data: dict[str, Any], block: dict[str, Any]) -> dict[str, float]:
    segments = data.get("segments")
    if not isinstance(segments, dict):
        raise ValueError(f"{file}: a P0.6 report has a 'segments' mapping")
    expected_tokens = block["batch"] * block["positions"]
    per: dict[str, float] = {}
    for name in COST_PROFILE_SEGMENTS:
        segment = segments.get(name)
        if not isinstance(segment, dict):
            raise ValueError(
                f"{file}: segment {name!r} is missing; a cost profile has {COST_PROFILE_SEGMENTS}"
            )
        n, tokens, device_ms = segment.get("n"), segment.get("tokens"), segment.get("device_ms")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"{file}: segment {name!r} has n = {n!r}; expected an int >= 1")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError(
                f"{file}: segment {name!r} has tokens = {tokens!r}; a cost profile records the "
                "positions each segment processed"
            )
        if tokens != expected_tokens:
            raise ValueError(
                f"{file}: segment {name!r} has tokens = {tokens}, but the {COST_PROFILE_KEY} block "
                f"says batch × positions = {expected_tokens}; the segments of a cost profile all "
                "process the same positions"
            )
        if (
            isinstance(device_ms, bool)
            or not isinstance(device_ms, int | float)
            or not math.isfinite(device_ms)
            or device_ms <= 0.0
        ):
            raise ValueError(
                f"{file}: segment {name!r} has device_ms = {device_ms!r}; expected a finite "
                "number > 0 (a zero is never a measurement)"
            )
        per[name] = float(device_ms) / tokens
    return per


def load_cost_table(profile_json: Path, n_units: int) -> tuple[Tensor, float, float]:
    """The chunk's interface: ``(per-unit costs [M], cell overhead, shared cost)`` from a cost
    profile (:meth:`CostTable.from_profile`), in units of the dense step."""
    table = CostTable.from_profile(profile_json, n_units)
    return table.unit.clone(), table.cell_overhead, table.shared


def cost_table_from_config(cfg: ModelConfig) -> CostTable:
    """The table ``model.cost_profile`` names (relative paths resolve against the repository root,
    like the other ``configs/...`` paths), or the uniform placeholder when it is ``null``."""
    if not isinstance(cfg, ModelConfig):
        raise TypeError(f"cost_table_from_config needs a ModelConfig, got {type(cfg).__name__}")
    if cfg.cost_profile is None:
        return CostTable.uniform(cfg.n_units, cfg.n_cells)
    path = cfg.cost_profile if cfg.cost_profile.is_absolute() else REPO_ROOT / cfg.cost_profile
    table = CostTable.from_profile(path, cfg.n_units)
    if table.n_cells != cfg.n_cells:
        raise ValueError(
            f"{path}: the cost profile was taken with N = {table.n_cells} cells, model.n_cells = "
            f"{cfg.n_cells}; regenerate the cost profile"
        )
    return table


# --------------------------------------------------------------------------------------------------
# Producing a cost profile
# --------------------------------------------------------------------------------------------------


def profile_dense_costs(
    model: HSN,
    ctx: DeviceContext,
    *,
    batch: int,
    positions: int,
    warmup: int = 1,
    seed: int = 0,
    count_launches: bool = False,
) -> dict[str, Any]:
    """Profile the dense step of ``model`` under the cost-model partition and return the cost
    profile as a JSON-ready dict (:func:`write_cost_profile` writes it).

    Random tokens and context ids are drawn from ``seed`` (CPU generator, moved to the model's
    device); ``warmup`` positions run first under the ``startup`` segment and are not part of the
    measurement; then ``positions`` positions run in the five segments, every segment entry
    processing ``batch`` positions. The replica of the step is checked against ``model.step`` on
    the same inputs at every position (``RuntimeError`` on drift), so the partition cannot
    silently diverge from :mod:`hsn.model.hsn`. Inference conditions: ``torch.no_grad()``, no
    autocast (the profile's dtype policy is applied by the caller through the model's dtype).
    """
    from hsn.model.hsn import HSN as _HSN  # runtime import: this module is imported by the model

    if not isinstance(model, _HSN):
        raise TypeError(f"profile_dense_costs needs an HSN model, got {type(model).__name__}")
    if not isinstance(ctx, DeviceContext):
        raise TypeError(f"ctx must be a DeviceContext, got {type(ctx).__name__}")
    for name, value in (("batch", batch), ("positions", positions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")
    # Parameters report an indexed device (``mps:0``) while a profile names ``mps``: compare the
    # type, and the index only when the context pins one.
    if ctx.device.type != model.device.type or (
        ctx.device.index is not None and ctx.device.index != model.device.index
    ):
        raise ValueError(
            f"ctx names {ctx.device} but the model lives on {model.device}; move the model first"
        )
    cfg = model.cfg
    cell = model.cell
    device, dtype = model.device, model.dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = warmup + positions
    tokens = torch.randint(0, cfg.vocab_size, (batch, total), generator=generator).to(device)
    contexts = torch.randint(0, cfg.n_context, (batch, total), generator=generator).to(device)

    profiler = Profiler(ctx, count_launches=count_launches)
    state = model.init_state(batch)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for t in range(warmup):
                with profiler.segment("startup"):
                    state = model.step(tokens[:, t], contexts[:, t], state, mode="dense").state
            for t in range(warmup, total):
                x_t, ctx_t = tokens[:, t], contexts[:, t]
                with profiler.segment("shared_path", tokens=batch):
                    q_prev = state.q
                    z_hat = model.predictor(q_prev)
                    z = model.encoder(model.byte_emb(x_t))
                    eps = z - z_hat
                    u = model.shared(torch.cat([z, q_prev, model.ctx_emb(ctx_t)], dim=-1))
                with profiler.segment("scorer", tokens=batch):
                    feats = model.scorer.features(
                        u, eps, state, ctx_t, budget_frac=cfg.budget, round_idx=0, ema=None
                    )
                    alpha = torch.softmax(model.scorer(feats), dim=-1)
                with profiler.segment("messages", tokens=batch):
                    B, N, d = state.batch_size, state.n_cells, state.width
                    h_prev = state.h
                    m = cell.graph.messages_dense(h_prev, cell.V)
                    v = torch.cat(
                        [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), h_prev, m],
                        dim=-1,
                    )
                    v = cell.norm(v)  # P1.5a: the units and the gate read the normalised input
                    b = torch.sigmoid(cell.gate(v))
                with profiler.segment("selected_cells", tokens=batch):
                    unit_out = cell.units.forward_all(v)
                with profiler.segment("messages"):  # the per-cell work after the unit pass
                    mixed = (alpha.unsqueeze(-1) * unit_out).sum(dim=2)
                    c = torch.tanh(cell.cand_norm(cell.W_out(mixed)))  # P1.13: Identity if none
                    h_new = (1 - b) * h_prev + b * c
                    if cell.disabled_cells:
                        h_new = torch.where(cell.disabled[None, :, None], h_prev, h_new)
                    new_state = state.with_dense(h_new)
                with profiler.segment("decode", tokens=batch):
                    logits = model.decoder(torch.cat([new_state.q, u], dim=-1))
                out = model.step(x_t, ctx_t, state, mode="dense")  # the reference, unsegmented
                if not torch.allclose(
                    logits.float(), out.logits.float(), rtol=1e-4, atol=1e-5
                ) or not torch.allclose(
                    new_state.h.float(), out.state.h.float(), rtol=1e-4, atol=1e-5
                ):
                    raise RuntimeError(
                        "profile_dense_costs has drifted from HSN.step: the segmented replica of "
                        "the dense step no longer computes the same logits and state"
                    )
                state = out.state
    finally:
        model.train(was_training)

    report = profiler.report().to_dict()
    report[COST_PROFILE_KEY] = {
        "schema": COST_PROFILE_SCHEMA,
        "producer": COST_PROFILE_PRODUCER,
        "n_cells": cfg.n_cells,
        "n_units": cfg.n_units,
        "d": cfg.d,
        "hidden": cfg.hidden,
        "batch": batch,
        "positions": positions,
        "profile": ctx.profile,
        "device": str(device),
        "dtype": str(dtype),
    }
    return report


def write_cost_profile(report: dict[str, Any], path: Path | str) -> Path:
    """Write a cost profile as strict JSON (a NaN or infinity raises); returns the path."""
    if not isinstance(report, dict) or not isinstance(report.get(COST_PROFILE_KEY), dict):
        raise ValueError(f"report is not a cost profile: no {COST_PROFILE_KEY!r} block")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return out
