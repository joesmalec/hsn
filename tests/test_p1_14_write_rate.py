"""P1.14 tests: the write rate (the share of gates >= 0.5) and the mean gate, reported next to the
gate quantiles in the diagnostics and the trained state, and in ``report``; no rule or preset reads
them.

Gate p50 alone cannot tell "no writes" from "few but strong writes" (blocks B1 and B2: p50 about 0,
p90 0.76 to 0.91, still learning; exp 100: p50 0.0014 with p90 0.17).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.experiments import db as dbm
from hsn.experiments.diagnostics import _write_rate, state_window, window_state_stats
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.report import build_report
from hsn.experiments.rules import PRESETS, parse_rules
from hsn.experiments.spec import protocol_params
from hsn.runner import build_model
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    cpu_ctx,
    db_path,
    disk_overrides,
    docs,
    dry_spec,
    isolated,  # noqa: F401  (fixture)
    restore_determinism,  # noqa: F401  (fixture)
    tiny_hsn_config,
    tiny_manifest,  # noqa: F401  (fixture)
)
from p1_14_support import enqueue_and_claim, run_child


def test_p1_14_write_rate_values() -> None:
    rate = _write_rate(torch.tensor([0.1, 0.5, 0.9, 0.49]))
    assert rate["gate/write_rate"] == 0.5 and rate["gate/mean"] == pytest.approx(0.4975)
    cfg = tiny_hsn_config()
    seed_everything(0)
    model = build_model(cfg)
    window = state_window(cfg, cpu_ctx(), ByteDataset(docs("delayed", "dev", 4)), 4)
    stats = window_state_stats(model, window)
    # an independent computation over the same values: every gate coordinate of the written cells
    # at valid positions
    model.eval()
    with torch.no_grad():
        B = int(window["tokens"].shape[0])
        _, _, info = model(
            window["tokens"], window["context"], model.init_state(B), window["reset"], capture=True
        )
    valid = window["doc"] >= 0
    gates = info["cell_gates"].float()[valid][info["cell_written"][valid]]
    gates = gates[~torch.isnan(gates)]
    assert stats["gate/write_rate"] == float((gates >= 0.5).float().mean())
    assert stats["gate/mean"] == pytest.approx(float(gates.mean()), rel=1e-6)
    assert 0.0 <= stats["gate/write_rate"] <= 1.0
    assert stats["gate/p10"] <= stats["gate/p50"] <= stats["gate/p90"]


@pytest.mark.usefixtures("restore_determinism")
def test_p1_14_write_rate_in_records(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(tiny_manifest, steps=4)])
    arm = ArmRun(
        cfg=cfg,
        family="hsn",
        variant="pdg",
        variant_params={},
        params=protocol_params({"name": "train_eval", "diag_every": 2, "diag_docs": 4}),
        run_dir=tmp_path / "arm",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    for block in (*record["diagnostics"], record["trained_state"]):
        assert {"gate/write_rate", "gate/mean", "gate/p90"} <= set(block)
        assert math.isfinite(block["gate/write_rate"])


def test_p1_14_write_rate_in_no_rule() -> None:
    for name, preset in PRESETS.items():
        params = {"window": 100, "to_step": 1000} if name == "memory_open" else {}
        if name == "memory_writes_run":  # P1.16: needs its last window centre; reads gate/p90
            params = {"to_step": 1000}
        text = json.dumps(preset(params))
        if name == "memory_writes":  # P1.15: the write-rate preset (approved), reads it by design
            assert "diag:gate/write_rate" in text and "gate/mean" not in text
            continue
        assert "write_rate" not in text and "gate/mean" not in text, name


@pytest.mark.usefixtures("isolated")
def test_p1_14_report_shows_write_rate_and_step_time(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    record = {
        "trained_state": {"gate/write_rate": 0.12},
        "step_time": {"run_median_s": 0.2},
        "probe_v2": {"correct_follow_frac": 0.25, "vocab_size": 18, "matched_frac": 1.0},
    }
    for exp_id in enqueue_and_claim(db, dry_spec(tmp_path, n=2, name="wr", record=record)):
        assert run_child(db, exp_id, tmp_path / "runs") == 0
    rows = build_report(db, sweep="wr", per_seed=True).rows
    assert [(r["write_rate"], r["step_s"]) for r in rows] == [(0.12, 0.2), (0.12, 0.2)]
    (group,) = build_report(db, sweep="wr").rows
    assert group["write_rate"] == "0.120 [0.120, 0.120]" and group["step_s"] == "0.200"
    assert [(r["v2_follow"], r["v2_vocab"], r["v2_matched"]) for r in rows] == [(0.25, 18, 1.0)] * 2
    assert rows[0]["v2_unchanged"] is None and group["v2_unchanged"] == ""
    assert group["v2_vocab"] == "18.000 [18.000, 18.000]"
