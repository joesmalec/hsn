"""P1.14 tests: probe v2, the vocabulary-matched edit (reads_state version 2).

The vocabulary is the value symbols among v1's first-query answers; each document's new value is
drawn (by a hash of its id) from the vocabulary minus the document's symbols; v2 runs with v1's
batch layout, so its original answers equal v1's; batched equals the per-document loop (exactly in
fp64, wherever the top-2 margin is clear of rounding in fp32). v1's numbers are unchanged by v2,
version 1 arms keep their ids, and ``probe-rerun`` reproduces a stored run's v1 probes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.experiments import db as dbm
from hsn.experiments import probe
from hsn.experiments.probe_check import probe_rerun
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import evaluate_verdicts, parse_rules
from hsn.experiments.spec import SpecError, expand, load_spec, protocol_params
from hsn.runner import build_model
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    cpu_ctx,
    db_path,
    disk_overrides,
    docs,
    fake_repo,
    isolated,  # noqa: F401  (fixture)
    restore_determinism,  # noqa: F401  (fixture)
    tiny_gru_config,
    tiny_hsn_config,
    tiny_manifest,  # noqa: F401  (fixture)
    tiny_transformer_config,
    write_yaml,
)
from p1_14_support import enqueue_and_claim, run_child, tiny_reads_state_spec
from torch import nn

MODELS = {
    "hsn_dense": lambda: tiny_hsn_config(),
    "hsn_topk": lambda: tiny_hsn_config("model.routing=topk"),
    "gru": lambda: tiny_gru_config(),
    "transformer": lambda: tiny_transformer_config(),
}


def _dev() -> ByteDataset:
    return ByteDataset(docs("kv", "dev", 8) + docs("delayed", "dev", 40) + docs("kv", "dev", 8))


def _model(cfg: RunConfig, dtype: torch.dtype) -> nn.Module:
    """P1.11's probe-test model (weights perturbed so that answers differ between documents) with
    its output bias (the one ``[V]`` parameter) raised on the 48 value symbols, so that its answers
    spread over values, as a trained model's do: v2 needs an answer vocabulary."""
    seed_everything(0)
    model = build_model(cfg).to(dtype)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=torch.float64).to(dtype) * 0.3)
        (bias,) = [p for p in model.parameters() if tuple(p.shape) == (cfg.model.vocab_size,)]
        bias[probe.VALUES.start : probe.VALUES.stop] += 3.0
    return model


def test_p1_14_probe_v2_vocabulary_and_draw() -> None:
    v1 = [{"argmax_original": a} for a in (170, 170, 175, 3, 190, 175)]
    vocab, outside = probe.answer_vocabulary(v1)
    assert vocab == [170, 175, 190] and outside == 1  # 3 is not a value symbol
    items = probe.probe_items(_dev())
    assert len(items) > 10
    vocabulary = list(probe.VALUES)[::3]
    for it in items:
        new, n = probe.matched_value(it, vocabulary)
        present = set(it.tokens.tolist())
        assert new is not None and new in vocabulary and new not in present and new != it.old_value
        assert n == len([v for v in vocabulary if v not in present])
        # reproducible, and a function of this document only (the vocabulary order is irrelevant)
        assert probe.matched_value(it, list(reversed(vocabulary))) == (new, n)
    draws = [probe.matched_value(it, vocabulary)[0] for it in items]
    reseeded = [probe.matched_value(it, vocabulary, seed=1)[0] for it in items]
    assert draws != reseeded and len(set(draws)) > 3, "the draw spreads over the candidates"
    # a vocabulary inside the document's own symbols leaves it unmatched
    own = sorted({int(t) for t in items[0].tokens.tolist() if int(t) in probe.VALUES})
    assert probe.matched_value(items[0], own) == (None, 0)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
def test_p1_14_probe_v2_batched_matches_per_document(name: str, dtype: torch.dtype) -> None:
    model = _model(MODELS[name](), dtype)
    ctx, dev = cpu_ctx(), _dev()
    v1 = probe.edit_probe(model, ctx, dev, batch_rows=8)
    out = probe.edit_probe_v2(model, ctx, dev, v1=v1, batch_rows=8, margins=True)
    ref = probe.edit_probe_v2_reference(model, ctx, dev, v1=v1, margins=True)
    assert model.training, "the probe restores the training mode"
    assert len(out) == len(ref) == len(v1) > 10
    # v1's batch layout, row for row: the original answers are v1's exactly
    assert [p["argmax_original"] for p in out] == [p["argmax_original"] for p in v1]
    assert all(p["original_matches_v1"] for p in out)
    keys = [k for k in ref[0] if not k.startswith("margin_") and k != "original_matches_v1"]
    if dtype == torch.float64:
        for a, b in zip(ref, out, strict=True):
            assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    else:
        for a, b in zip(ref, out, strict=True):
            if min(a["margin_original"], a["margin_modified"]) > 1e-4:
                assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    summary = probe.summarize_v2(out, v1=v1)
    vocab, _ = probe.answer_vocabulary(v1)
    assert summary["vocab"] == vocab and summary["vocab_size"] == len(vocab) > 1
    assert summary["matched_n"] == sum(p["matched"] for p in out) > 0
    assert summary["matched_frac"] == summary["matched_n"] / len(out)
    assert summary["original_mismatches"] == 0
    assert sum(summary["all"].values()) == summary["matched_n"]
    assert (
        sum(summary["correct_before_edit"][k] for k in ("follow", "other", "unchanged"))
        == (summary["correct_n"])
    )
    for p in out:
        if p["matched"]:
            assert p["new_value"] in vocab and p["new_value"] != p["old_value"]
        else:
            assert p["new_value"] is None and p["changed"] is None


def test_p1_14_probe_v2_verdict_and_ids(tmp_path: Path) -> None:
    assert probe.verdict_v2(0, 0, 0) == "no matched probes"
    assert probe.verdict_v2(19, 0, 19) == "too few matched probes"
    assert probe.verdict_v2(12, 2, 20) == "reads state"
    assert probe.verdict_v2(3, 10, 20) == "does not read state"
    assert probe.verdict_v2(6, 10, 20) == "partial"
    rules = parse_rules({"verdicts": {"v2": {"preset": "reads_state_v2"}}})

    def label(**fields: Any) -> str:
        record = {"probe_v2": fields, "protocol": "reads_state"}
        return str(evaluate_verdicts(rules, record, {}, "hsn")["v2"]["label"])

    assert label(correct_n=0) == "no matched probes"
    assert label(correct_n=5, correct_follow_frac=1.0) == "too few matched probes"
    assert (
        label(correct_n=40, correct_follow_frac=0.6, correct_unchanged_frac=0.05) == "reads state"
    )
    assert label(correct_n=40, correct_follow_frac=0.1, correct_unchanged_frac=0.5) == (
        "does not read state"
    )
    assert label(correct_n=40, correct_follow_frac=0.3, correct_unchanged_frac=0.5) == "partial"

    def ids(version: int | None, verdicts: dict[str, Any] | None = None) -> str:
        protocol: dict[str, Any] = {"name": "reads_state", "diag_every": 10}
        if version is not None:
            protocol["version"] = version
        raw = {
            "name": "v2",
            "base": {
                "config": "configs/experiments/g1_ref_a1.yaml",
                "overrides": ["train.batch=8", "train.micro_batch=8", "train.steps=40"],
            },
            "protocol": protocol,
            "rules": {"verdicts": verdicts} if verdicts else None,
            "arms": [{"name": "a", "family": "hsn"}],
        }
        path = write_yaml(tmp_path / f"spec_{len(list(tmp_path.glob('spec_*')))}.yaml", raw)
        spec, _ = load_spec(path)
        (arm,) = expand(spec, repo=fake_repo(tmp_path))
        return arm.arm_id

    assert ids(None) == ids(1), "version 1 is the arm id of a spec without the key"
    assert ids(2) != ids(1)
    v2_rule = {"v2": {"preset": "reads_state_v2"}}
    assert ids(2, v2_rule) == ids(2)  # verdicts are not part of the id
    with pytest.raises(SpecError, match="probe_v2 needs reads_state version 2"):
        ids(1, v2_rule)


@pytest.mark.usefixtures("restore_determinism")
def test_p1_14_probe_v2_keeps_v1(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    """The same tiny arm at version 1 and version 2: training, evaluations and v1's probes are
    identical; only version 2 has ``probe_v2`` (and its per-document list)."""
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(tiny_manifest, steps=4)])

    def run(version: int) -> dict[str, Any]:
        params = protocol_params(
            {"name": "reads_state", "version": version, "diag_every": 2, "probe_batch": 8}
        )
        arm = ArmRun(
            cfg=cfg,
            family="hsn",
            variant="pdg",
            variant_params={},
            params=params,
            run_dir=tmp_path / f"v{version}",
            rules=parse_rules(None),
            constants={},
        )
        return run_protocol(arm)

    one, two = run(1), run(2)
    for key in ("train", "evals", "probe", "probes", "trained_state"):
        assert one[key] == two[key], key
    assert "probe_v2" not in one and one["protocol_version"] == 1
    assert two["protocol_version"] == 2 and two["probe_v2"]["n_probed"] == one["probe"]["n_probed"]
    assert len(two["probes_v2"]) == len(two["probes"]) and "probe_v2_s" in two["walls"]


@pytest.mark.usefixtures("isolated", "restore_determinism")
def test_p1_14_probe_rerun_reproduces_a_stored_run(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
) -> None:
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(tmp_path, tiny_manifest, name="rerun", steps=4)
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    out = probe_rerun(db, exp_id)
    assert out["v1_answers_differing"] == 0 and out["v1_summary_equal"]
    assert out["v1_rerun"] == out["v1_stored"]
    assert out["v2"]["n_probed"] == out["v1_rerun"]["n_probed"]
    assert out["v2"]["original_mismatches"] == 0
    stored = json.loads(Path(out["checkpoint"]).parent.joinpath("result.json").read_text())
    assert stored["probe"]["n_probed"] == out["v2"]["n_probed"]
    assert np.isfinite(out["v1_s"]) and out["batch_rows"] == 8
