#!/usr/bin/env python3
"""Print the Appendix C tables (parameters, MACs per token, memory) for a configuration.

P1.1. The numbers come from the closed forms in ``hsn.train.memory_plan``; the layout follows
``HSN_Implementation_Plan.md`` Appendix C.1–C.3 so the output can be compared with it to the digit.
Batch and unroll come from ``train.*`` of the configuration, bytes per value from
``train.precision``. Planner numbers are estimates, never measurements (plan §0.5).

Usage: ``python scripts/count_params.py --config configs/ref_a1.yaml [model.scorer=mlp ...]``
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hsn.config import ModelConfig, RunConfig, load_config
from hsn.train.memory_plan import (
    SCORER_SCALAR_FEATURES,
    TRAIN_STATE_BYTES_PER_PARAM,
    activations_per_token,
    expected_param_count,
    format_bytes,
    macs_per_token,
    memory_plan,
    value_bytes_for,
)

#: The Appendix C.3 "Recurrent state, scaled example" row: the scale_100m shape (N=64, d=256) at
#: batch 16. An illustration printed next to the configured prototype row, not a model dimension.
SCALED_EXAMPLE: dict[str, int] = {"batch": 16, "n_cells": 64, "d": 256}


def _n(value: int) -> str:
    """Thousands separators, as Appendix C prints every count."""
    return f"{value:,}"


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)} %"


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _other_scorer(m: ModelConfig) -> ModelConfig:
    """The same configuration with the other scorer variant (linear <-> mlp)."""
    other = "mlp" if m.scorer == "linear" else "linear"
    return ModelConfig.model_validate({**m.model_dump(mode="json"), "scorer": other})


def _scorer_phrase(m: ModelConfig) -> str:
    """``Linear scorer`` or ``Two-layer scorer, 128 hidden,`` as the C.2 rows read."""
    if m.scorer == "linear":
        return "Linear scorer"
    return f"Two-layer scorer, {_n(m.scorer_hidden)} hidden,"


def _scorer_shape(m: ModelConfig, d_in: int) -> str:
    if m.scorer == "linear":
        return f"{d_in} → {m.n_units}"
    return f"{d_in} → {m.scorer_hidden} → {m.n_units}"


def configuration_line(m: ModelConfig) -> str:
    line = (
        f"Configuration: V={m.vocab_size}, d={m.d}, N={m.n_cells}, M={m.n_units}, "
        f"K={m.k_select}, R={m.r_max}, D_g={m.in_degree}, H={m.hidden}, "
        f"{m.n_context} context symbols of width {m.context_dim}."
    )
    if m.ablate_shared_u:  # the ablation's line only: the default output is Appendix C's
        line += (
            f" Ablation model.ablate_shared_u: the cell input is [ε; h_i; m_i] of width "
            f"{m.cell_input_width} (u removed from v)."
        )
    return line


def parameter_table(m: ModelConfig) -> str:
    """C.1 Parameters."""
    p = expected_param_count(m)
    d, h, v = m.d, m.hidden, m.vocab_size
    v_in = m.cell_input_width  # the cell input: 4d, or 3d with model.ablate_shared_u
    scorer_in = 3 * d + SCORER_SCALAR_FEATURES  # [u; ε; h_i] + eight scalars (Appendix H.2)
    rows = [
        ("Byte embedding", f"{v} × {d}", _n(p["byte_emb"])),
        ("Context embedding", f"{m.n_context} × {m.context_dim}", _n(p["ctx_emb"])),
        ("Encoder E", f"{d} → {h} → {d}", _n(p["encoder"])),
        ("Predictor P", f"{d} → {h} → {d}", _n(p["predictor"])),
        ("Shared transform S", f"{2 * d + m.context_dim} → {h} → {d}", _n(p["shared"])),
        (
            f"Capability units F(1)–F({m.n_units})",
            f"{m.n_units} × ({v_in} → {h} → {d})",
            _n(p["units"]),
        ),
        ("Output projection W_out", f"{d} × {d}", _n(p["w_out"])),
        ("Message projection V", f"{d} × {d}", _n(p["v_msg"])),
        ("Gate B", f"{v_in} → {d}", _n(p["gate"])),
        ("Cell input RMSNorm gain (amendment A1)", f"{v_in}", _n(p["cell_norm"])),
        # P1.13 (amendment A4): the candidate's RMSNorm gain, a row only when model.cand_norm=rms
        *(
            [("Candidate RMSNorm gain (amendment A4)", f"{d}", _n(p["cand_norm"]))]
            if m.cand_norm == "rms"
            else []
        ),
        (f"Router scorer, {m.scorer}", _scorer_shape(m, scorer_in), _n(p["scorer"])),
        ("Halting head", f"{2 * d} → 1", _n(p["halting"])),
        ("Decoder D", f"{2 * d} → {h} → {v}", _n(p["decoder"])),
        ("**Total**", "", f"**{_n(p['total'])}**"),
    ]
    return _table(("Component", "Shape", "Parameters"), rows)


def mac_table(m: ModelConfig) -> str:
    """C.2 Multiply-accumulates per input position (one round; biases counted)."""
    p = expected_param_count(m)
    macs = macs_per_token(m)
    unit = p["units"] // m.n_units
    sparse, dense = macs["sparse_total"], macs["dense_total"]
    other = _other_scorer(m)
    other_macs = macs_per_token(other)
    other_sparse = macs["shared"] + macs["selected"] + other_macs["scorer"]
    rows = [
        ("Shared per-token work E + P + S + D", _n(macs["shared"]), _pct(macs["shared"], sparse)),
        (
            f"One selected cell: unit {_n(unit)} + W_out {_n(p['w_out'])} + "
            f"{m.in_degree} messages through V {_n(m.in_degree * p['v_msg'])} + "
            f"gate B {_n(p['gate'])}",
            _n(macs["per_cell"]),
            _pct(macs["per_cell"], sparse),
        ),
        (f"{m.k_select} selected cells", _n(macs["selected"]), _pct(macs["selected"], sparse)),
        (
            f"{_scorer_phrase(m)} over all {m.n_cells} cells ({m.n_cells} × {_n(p['scorer'])})",
            _n(macs["scorer"]),
            _pct(macs["scorer"], sparse),
        ),
        (f"**Sparse total, K = {m.k_select}**", f"**{_n(sparse)}**", "100 %"),
        (
            f"Dense total, all {m.n_cells} cells ({_n(macs['shared'])} + {m.n_cells} × "
            f"{_n(macs['per_cell'])} + {_n(macs['scorer'])})",
            _n(dense),
            _pct(dense, sparse),
        ),
        (
            f"{_scorer_phrase(other)} over {m.n_cells} cells "
            f"({m.n_cells} × {_n(expected_param_count(other)['scorer'])})",
            _n(other_macs["scorer"]),
            f"would {'lift' if other_sparse > sparse else 'lower'} sparse to "
            f"{_pct(other_sparse, dense)} of dense",
        ),
    ]
    return _table(("Work", "MACs", "Share of sparse total"), rows)


def memory_table(m: ModelConfig, batch: int, unroll: int, value_bytes: int) -> str:
    """C.3 Memory, followed by the two activation-count footnotes."""
    plan = memory_plan(m, batch, unroll, value_bytes)
    acts = activations_per_token(m)
    total = expected_param_count(m)["total"]
    rounds = f"{m.r_max} × " if m.r_max > 1 else ""
    scaled_model = ModelConfig.model_validate(
        {
            **m.model_dump(mode="json"),
            "n_cells": SCALED_EXAMPLE["n_cells"],
            "d": SCALED_EXAMPLE["d"],
            "context_dim": SCALED_EXAMPLE["d"],
        }
    )
    scaled = memory_plan(scaled_model, SCALED_EXAMPLE["batch"], unroll, value_bytes)
    rows = [
        (
            f"Weights and optimiser state at {TRAIN_STATE_BYTES_PER_PARAM} B/param",
            f"{_n(total)} × {TRAIN_STATE_BYTES_PER_PARAM}",
            format_bytes(plan.train_state_bytes),
        ),
        (
            "Recurrent state, prototype",
            f"{batch} × {m.n_cells} × {m.d} × {value_bytes}",
            format_bytes(plan.rec_state_bytes),
        ),
        (
            "Recurrent state, scaled example",
            f"{SCALED_EXAMPLE['batch']} × {SCALED_EXAMPLE['n_cells']} × {SCALED_EXAMPLE['d']} "
            f"× {value_bytes}",
            format_bytes(scaled.rec_state_bytes),
        ),
        (
            f"Saved activations, K = {m.k_select}",
            f"{batch} × {unroll} × ({_n(acts['shared'])} + {rounds}{m.k_select} × "
            f"{_n(acts['per_cell'])}) × {value_bytes}",
            format_bytes(plan.saved_sparse_bytes),
        ),
        (
            f"Saved activations, all {m.n_cells} cells",
            f"{batch} × {unroll} × ({_n(acts['shared'])} + {rounds}{m.n_cells} × "
            f"{_n(acts['per_cell'])}) × {value_bytes}",
            format_bytes(plan.saved_dense_bytes),
        ),
    ]
    footnotes = (
        f"Per-cell saved activations {_n(acts['per_cell'])} = v ({m.cell_input_width}) + unit "
        f"hidden ({_n(m.hidden)}) + c, b, h ({3 * m.d}). Shared activations {_n(acts['shared'])} = "
        f"z, ẑ, ε, u ({4 * m.d}) + 4 hidden layers ({_n(4 * m.hidden)}) + logits "
        f"({m.vocab_size})."
    )
    return _table(("Item", "Formula", "Value"), rows) + "\n\n" + footnotes


def report(cfg: RunConfig, config_path: Path) -> str:
    """The three Appendix C tables for ``cfg`` as one text block."""
    m = cfg.model
    batch, unroll = cfg.train.batch, cfg.train.unroll
    value_bytes = value_bytes_for(cfg.train.precision)
    sections = [
        f"Appendix C tables for {config_path} (closed forms, hsn.train.memory_plan)",
        configuration_line(m),
        "",
        "C.1 Parameters",
        "",
        parameter_table(m),
        "",
        "C.2 Multiply-accumulates per input position (one round; biases counted; 1 MAC = 2 FLOPs)",
        "",
        mac_table(m),
        "",
        f"C.3 Memory (batch {batch}, unroll {unroll}, {cfg.train.precision} values, "
        f"{value_bytes} B/value)",
        "",
        memory_table(m, batch, unroll, value_bytes),
    ]
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path, help="run configuration YAML")
    parser.add_argument(
        "overrides", nargs="*", help="dotted overrides, e.g. model.scorer=mlp train.batch=8"
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=args.overrides)
        print(report(cfg, args.config))
    except (OSError, ValueError) as exc:
        print(f"count_params: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
