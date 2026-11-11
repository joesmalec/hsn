from __future__ import annotations

import json

import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.experiments import diagnostics, measurements
from hsn.experiments.db import Database
from hsn.experiments.stability import StabilityObserver, route_sets
from hsn.model.hsn import HSN
from hsn.runner import make_trainer
from p1_11_support import cpu_ctx, docs, tiny_hsn_config


def test_views_distinguish_failed_partial_and_full_drawdown(tmp_path):
    db = Database(tmp_path / "measure.sqlite", allow_tmp=True)
    c = db.conn
    # Minimal relational fixture: preserve the real schema's columns and SQLite version behavior.
    c.execute("PRAGMA foreign_keys=OFF")
    c.execute(
        "INSERT INTO experiments (exp_id,arm_id,label,run_id,status,enqueued_at,config_json) "
        "VALUES (1,'test','failed','failed','failed','now','{}')"
    )
    rec = {
        "evals": [
            {"step": 1, "accuracy": 0.8},
            {"step": 2, "accuracy": 0.7},
            {"step": 3, "accuracy": 0.6},
        ],
        "final": {"eval_step": 3, "dev_accuracy": 0.6, "delayed/in_range/accuracy": 0.6},
    }
    c.execute(
        "INSERT INTO results(exp_id,record_json,verdicts_json) VALUES (1,?,'{}')",
        (json.dumps(rec),),
    )
    measurements.install(c)
    measurements.install(c)
    row = c.execute("SELECT * FROM training_measurements").fetchone()
    assert row["final_dev_accuracy"] is None and row["final_delayed_accuracy"] is None
    assert row["reported_eval_accuracy"] == 0.6
    assert row["max_adjacent_drop_pp"] == pytest.approx(10)
    assert row["max_drawdown_pp"] == pytest.approx(20)
    assert row["swap_follow"] is None and row["skipped_steps"] is None
    assert db.version() == 2
    c.execute(
        "UPDATE results SET record_json=?", (json.dumps({"evals": [{"step": 1, "accuracy": 0.8}]}),)
    )
    assert c.execute("SELECT max_drawdown_pp FROM training_measurements").fetchone()[0] is None


def test_route_sets_ignore_slot_order_and_invalid_padding():
    a = {"cell_idx": torch.tensor([[[2, 1, 99]]]), "valid": torch.tensor([[[True, True, False]]])}
    b = {"cell_idx": torch.tensor([[[1, 2, -99]]]), "valid": a["valid"]}
    assert torch.equal(route_sets(a, 4), route_sets(b, 4))


def test_stability_probe_preserves_model_rng_and_handles_reset():
    cfg = tiny_hsn_config(steps=2)
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"routing": "topk"})})
    model = HSN.from_config(cfg)
    model.train()
    w = diagnostics.state_window(cfg, cpu_ctx(), ByteDataset(docs("delayed", "train", 4)), 2)
    before = {k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    rng = torch.get_rng_state().clone()
    observer = StabilityObserver(1e-3, 2, 2)
    a = observer.observe(model, w, 10)
    b = observer.observe(model, w, 20)
    assert a["stability/route_churn"] is None
    assert b["stability/route_churn"] == 0
    assert b["stability/reference_step"] == 10
    assert b["stability/eligible_rows"] > 0
    assert b["stability/state_gain_max"] >= 0
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items() if k in before)
    assert all(p.grad is None for p in model.parameters())
    w["reset"][:, 2] = True
    reset = observer.observe(model, w, 30)
    assert reset["stability/eligible_rows"] == 0
    assert reset["stability/state_gain_max"] is None


def test_observer_does_not_change_cpu_training(tmp_path):
    train = ByteDataset(docs("delayed", "train", 8))
    dev = ByteDataset(docs("delayed", "dev", 4))
    runs = []
    for enabled in (False, True):
        torch.manual_seed(41)
        cfg = tiny_hsn_config(steps=4)
        model = HSN.from_config(cfg)
        trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=tmp_path / str(enabled))
        diag = diagnostics.Diagnostics(
            model,
            cfg,
            cpu_ctx(),
            dev,
            1,
            n_docs=2,
            stability=enabled,
            sensitivity_position=2,
            sensitivity_horizon=2,
        )
        diag.attach(trainer)
        history = trainer.fit(train, dev, 4)
        diag.detach()
        runs.append(
            ([s.loss for s in history.steps], [p.detach().clone() for p in model.parameters()])
        )
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1], strict=True))


def test_gpu_unsupported_utilization_keeps_memory_reading(monkeypatch):
    from hsn.experiments import machine

    monkeypatch.setattr(machine, "_fake", lambda: False)
    monkeypatch.setattr(machine.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    for field, expected in (("73", 73.0), ("[N/A]", None)):
        monkeypatch.setattr(
            machine,
            "_smi",
            lambda args, timeout, field=field: (
                f"12000, 2048, {field}" if args[0].startswith("--query-gpu=") else ""
            ),
        )
        reading = machine.read_gpu()
        assert reading.ok and reading.used_mib == 2048
        assert reading.utilization_pct == expected


def test_worker_telemetry_bound_and_missing_reading():
    from hsn.experiments.machine import GpuReading
    from hsn.experiments.worker import _Watch

    watch = _Watch()
    for step in range(4):
        watch.sample_telemetry(
            GpuReading(True, True, used_mib=123, utilization_pct=70), {"step": step}, 100, 2
        )
    assert [x["step"] for x in watch.telemetry] == [2, 3]
    assert watch.card(False)["telemetry_dropped"] == 2
    watch.sample_telemetry(GpuReading(True, False), {}, None, 2)
    assert watch.telemetry[-1]["utilization_pct"] is None
    assert watch.telemetry[-1]["process_memory_mib"] is None
