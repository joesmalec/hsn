"""P1.15 tests: probe v3, the key-addressed edits of the later queries (reads_state version 3).

The documents are parsed and checked (the asked fact is always one the generator's rule makes
eligible); each later query gets a value edit (a vocabulary value absent from the document, on a
copy) and a key swap (a partner not asked before, with another value, on a copy); batched equals the
per-row loop exactly in fp64. Oracle models pin the meaning: a reader that binds keys follows both
edits ("reads state (later queries)", "binds keys"); a key-blind positional reader follows the value
edit but not the swap ("does not bind keys"); a constant answer has too small a vocabulary. Version
3 keeps v1's and v2's numbers, versions 1 and 2 keep their arm ids, ``probe-rerun`` reproduces a
stored v3 run, and ``report`` shows it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import QRY, TaskSpec, generate
from hsn.device import seed_everything
from hsn.experiments import db as dbm
from hsn.experiments import probe
from hsn.experiments.probe_check import probe_rerun
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.report import build_report
from hsn.experiments.rules import evaluate_verdicts, parse_rules
from hsn.experiments.spec import SpecError, expand, load_spec, protocol_params
from hsn.runner import build_model
from p1_11_support import (
    REF_A1,
    SPEC,
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
from torch import Tensor, nn


def _full_dev(n: int = 160) -> ByteDataset:
    """Dev documents of the committed delayed/in_range task at full length (up to 4 queries)."""
    task = SPEC.task("delayed", "in_range")
    spec = TaskSpec.model_validate({**task.model_dump(), "n_dev": max(n, task.n_dev)})
    return ByteDataset([generate(spec, "dev", i) for i in range(n)])


def _tiny_dev() -> ByteDataset:
    return ByteDataset(docs("kv", "dev", 4) + docs("delayed", "dev", 40) + docs("kv", "dev", 4))


def _digest(dev: ByteDataset) -> str:
    h = hashlib.sha256()
    for ex in dev.examples:
        for arr in (ex.tokens, ex.targets, ex.loss_mask, ex.context):
            h.update(np.asarray(arr).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------
# Parsing and items
# --------------------------------------------------------------------------------------------------


def test_p1_15_v3_parse_and_eligibility() -> None:
    """Over full-length dev documents: facts and queries parse, slot 0 asks fact 0, and every later
    query asks a fact the generator's rule (ceiling_check.py's) makes eligible."""
    dev = _full_dev()
    parsed = probe.delayed_docs(dev)
    assert len(parsed) == len(dev)
    later = 0
    single = 0
    for d in parsed:
        assert d.queries[0][1] == 0 and len(d.queries) <= 4
        asked = [0]
        for slot, which, p in d.queries[1:]:
            eligible = probe.eligible_facts(len(d.facts), slot, asked)
            assert which in eligible and int(d.tokens[p - 1]) == QRY
            later += 1
            single += len(eligible) == 1
            asked.append(which)
    assert later > 50 and 0 < single < later
    ex = dev.examples[0]
    bad = np.asarray(ex.loss_mask).copy()
    bad[int(np.flatnonzero(bad)[0]) - 1] = True
    with pytest.raises(ValueError, match="does not ask"):
        probe.parse_delayed(np.asarray(ex.tokens), bad)


def test_p1_15_v3_items_edit_copies() -> None:
    """Each later query's value edit changes only fact k's value, to a vocabulary value absent from
    the document; its key swap trades two keys, the partner not asked before and of another value;
    the dev documents are unchanged."""
    dev = _full_dev(80)
    before = _digest(dev)
    parsed = probe.delayed_docs(dev)
    vocabulary = list(range(144, 192, 2))  # 24 values
    items = probe.v3_items(parsed, vocabulary)
    assert items and _digest(dev) == before
    for it in items:
        d = it.doc
        asked = [w for _, w, _ in d.queries[: it.slot]]
        if it.value_tokens is not None:
            diff = np.flatnonzero(it.value_tokens != d.tokens)
            assert diff.tolist() == [d.facts[it.which][2]]
            assert it.value_new in vocabulary and it.value_new not in set(d.tokens.tolist())
        else:
            assert it.value_candidates == 0
        if it.swap_tokens is not None:
            assert it.swap_with is not None and it.swap_with not in asked
            assert it.swap_with != it.which and it.swap_value != it.old_value
            diff = np.flatnonzero(it.swap_tokens != d.tokens).tolist()
            kpos, jpos = d.facts[it.which][2] - 1, d.facts[it.swap_with][2] - 1
            assert diff == sorted([kpos, jpos])
            assert int(it.swap_tokens[kpos]) == d.facts[it.swap_with][0]
        assert it.eligible >= 1
    again = probe.v3_items(probe.delayed_docs(dev), vocabulary)
    assert [(i.value_new, i.swap_with) for i in again] == [
        (i.value_new, i.swap_with) for i in items
    ]


# --------------------------------------------------------------------------------------------------
# Batched against the per-row loop
# --------------------------------------------------------------------------------------------------

MODELS = {
    "hsn_dense": lambda: tiny_hsn_config(),
    "hsn_topk": lambda: tiny_hsn_config("model.routing=topk"),
    "gru": lambda: tiny_gru_config(),
    "transformer": lambda: tiny_transformer_config(),
}


def _model(cfg: RunConfig) -> nn.Module:
    """P1.14's probe-test model in fp64: perturbed weights, output bias raised on the values."""
    seed_everything(0)
    model = build_model(cfg).to(torch.float64)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=torch.float64) * 0.3)
        (bias,) = [p for p in model.parameters() if tuple(p.shape) == (cfg.model.vocab_size,)]
        bias[probe.VALUES.start : probe.VALUES.stop] += 3.0
    return model


@pytest.mark.usefixtures("restore_determinism")
@pytest.mark.parametrize("name", sorted(MODELS))
def test_p1_15_v3_batched_matches_reference(name: str) -> None:
    """In fp64 the batched probe (odd chunks of 7 rows) equals the per-row B = 1 loop exactly."""
    cfg = MODELS[name]()
    model = _model(cfg)
    ctx = cpu_ctx()
    dev = _tiny_dev()
    batched, originals = probe.edit_probe_v3(model, ctx, dev, batch_rows=7)
    reference, originals_ref = probe.edit_probe_v3(model, ctx, dev, reference=True)
    assert batched == reference and originals == originals_ref
    summary = probe.summarize_v3(batched, originals)
    assert summary["n_later_queries"] == len(batched) > 0
    assert summary["value"]["available_n"] > 0


# --------------------------------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------------------------------


class _Oracle(nn.Module):
    """A SequenceModel that answers every ``QRY k`` position by rule (logits one-hot on the
    answer): ``bind`` reads the value stated after key ``k`` in the facts; ``blind`` reads the value
    of the first fact a key-blind reader may assign to the slot (by position, excluding facts whose
    values were already answered); ``constant`` always answers one value."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.anchor = nn.Parameter(torch.zeros(()))

    def init_state(self, B: int, device: Any = None) -> Tensor:
        return torch.zeros(B)

    def reset_state(self, state: Tensor, mask: Tensor) -> Tensor:
        return state

    def param_report(self) -> dict[str, int]:
        return {"active": 1, "stored": 1, "trainable": 1}

    def _answer(self, row: list[int], p: int) -> int:
        facts = []
        first_qry = next(i for i in range(p + 1) if row[i] == QRY)
        for i in range(first_qry):
            if row[i] in probe.KEYS and row[i + 1] in probe.VALUES:
                if i == 0 or row[i - 1] != QRY:
                    facts.append((row[i], row[i + 1]))
        if self.mode == "constant":
            return 150
        if self.mode == "bind":
            values = {k: v for k, v in facts}
            return values.get(row[p], 144)
        slot = sum(1 for i in range(first_qry, p - 1) if row[i] == QRY)  # the QRYs before this one
        if slot == 0:
            return facts[0][1]
        # which facts the earlier slots asked: a key swap of two facts not asked leaves this alone,
        # and the choice below reads positions only
        earlier = [row[i + 1] for i in range(first_qry, p - 1) if row[i] == QRY]
        asked = [w for w, (k, _) in enumerate(facts) if k in earlier]
        eligible = probe.eligible_facts(len(facts), slot, asked)
        return facts[eligible[0]][1] if eligible else 144

    def forward(
        self, tokens: Tensor, context: Tensor, state: Tensor, reset: Tensor, **_: Any
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        B, T = tokens.shape
        logits = torch.zeros(B, T, 256, dtype=torch.float32)
        rows = tokens.tolist()
        for b in range(B):
            row = rows[b]
            for p in range(1, T):
                if row[p - 1] == QRY and row[p] in probe.KEYS:
                    logits[b, p, self._answer(row, p)] = 10.0
        return logits + self.anchor, state, {}


@pytest.mark.parametrize(
    ("mode", "value_verdict", "swap_verdict"),
    [
        ("bind", "reads state (later queries)", "binds keys"),
        ("blind", "reads state (later queries)", "does not bind keys"),
        ("constant", "vocabulary too small", "too few swap probes"),
    ],
)
def test_p1_15_v3_oracles(mode: str, value_verdict: str, swap_verdict: str) -> None:
    dev = _full_dev()
    probes, originals = probe.edit_probe_v3(_Oracle(mode), cpu_ctx(), dev, batch_rows=32)
    summary = probe.summarize_v3(probes, originals)
    assert summary["value"]["verdict"] == value_verdict, summary["value"]
    assert summary["swap"]["verdict"] == swap_verdict, summary["swap"]
    if mode == "bind":
        assert summary["pass_accuracy"]["later"] == 1.0
        assert summary["value"]["follow_frac"] == 1.0 and summary["swap"]["follow_frac"] == 1.0
        assert summary["swap"]["by_ambiguity"]["single"]["follow_frac"] == 1.0
    if mode == "blind":
        assert summary["pass_accuracy"]["first"] == 1.0
        assert 0 < summary["pass_accuracy"]["later"] < 1.0
        assert summary["swap"]["unchanged_frac"] == 1.0
        # a key-blind reader is right on every later query position alone answers
        single = [p for p in probes if p["eligible"] == 1]
        assert single and all(p["original_correct"] for p in single)
    if mode == "constant":
        assert summary["vocab_size"] == 1 and 0 < summary["correct_later_n"] < 20


def test_p1_15_v3_verdicts() -> None:
    v = probe.verdict_v3_value
    assert v(0, 0, 0, 48) == "no correct later queries"
    assert v(19, 0, 19, 5) == "vocabulary too small"
    assert v(19, 0, 19, 48) == "too few correct later queries"
    assert v(15, 2, 20, 20) == "reads state (later queries)"
    assert v(3, 10, 20, 48) == "does not read state (later queries)"
    assert v(8, 2, 20, 48) == "partial (later queries)"
    assert v(19, 0, 19, 5, min_vocab=4, min_correct=10) == "reads state (later queries)"
    # correct later queries none of which could be edited: the vocabulary decides
    assert v(0, 0, 0, 1, correct_later=4) == "vocabulary too small"
    s = probe.verdict_v3_swap
    assert s(0, 0, 0) == "no swap probes" and s(5, 0, 19) == "too few swap probes"
    assert s(10, 2, 20) == "binds keys" and s(1, 19, 20) == "does not bind keys"
    assert s(6, 10, 20) == "partial"


# --------------------------------------------------------------------------------------------------
# The protocol, ids, rules, rerun and report
# --------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_determinism")
def test_p1_15_v3_keeps_v1_and_v2(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    """The same tiny arm at version 2 and version 3: training, evaluations, v1 and v2 identical;
    only version 3 has ``probe_v3``."""
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

    two, three = run(2), run(3)
    for key in ("train", "evals", "probe", "probes", "probe_v2", "probes_v2", "trained_state"):
        assert two[key] == three[key], key
    assert "probe_v3" not in two and three["protocol_version"] == 3
    block = three["probe_v3"]
    assert block["n_documents"] == len(three["probes_v3_originals"])
    assert block["n_later_queries"] == len(three["probes_v3"]) and "probe_v3_s" in three["walls"]
    assert three["final"]["split/first/n"] == block["n_documents"]


def test_p1_15_v3_arm_ids_and_rules(tmp_path: Path) -> None:
    """Version 3 is a new protocol version (a new id); versions 1 and 2 keep theirs; probe_v3 rules
    read version 3 records and are refused on version 2 arms."""
    repo = fake_repo(tmp_path)

    def spec(version: int, rules: dict[str, Any] | None = None) -> Path:
        return write_yaml(
            tmp_path / f"v{version}.yaml",
            {
                "name": f"v{version}",
                "base": {
                    "config": "configs/experiments/g1_ref_a1.yaml",
                    "overrides": ["train.batch=8", "train.micro_batch=8", "train.steps=40"],
                },
                "protocol": {"name": "reads_state", "version": version},
                "rules": rules,
                "arms": [{"name": "a", "family": "hsn"}],
            },
        )

    ids = {v: expand(load_spec(spec(v))[0], repo=repo)[0].arm_id for v in (1, 2, 3)}
    assert len(set(ids.values())) == 3
    rules = {
        "verdicts": {
            "keys": {
                "cases": [
                    {
                        "label": "binds",
                        "when": {"metric": "probe_v3:swap/follow_frac", "op": ">=", "value": 0.5},
                    }
                ],
                "default": "no",
            }
        }
    }
    assert expand(load_spec(spec(3, rules))[0], repo=repo)
    with pytest.raises(SpecError, match="probe_v3 needs reads_state version 3"):
        expand(load_spec(spec(2, rules))[0], repo=repo)
    record = {"protocol": "reads_state", "probe_v3": {"swap/follow_frac": 0.9}}
    assert evaluate_verdicts(parse_rules(rules), record, {}, "hsn")["keys"]["label"] == "binds"


@pytest.mark.usefixtures("isolated", "restore_determinism")
def test_p1_15_probe_rerun_and_report_v3(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
) -> None:
    """A stored version-3 run: ``probe-rerun`` reproduces its v1 and v2 answers (0 differing) and
    its v3 summary; ``report`` shows the v3 columns."""
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(
        tmp_path, tiny_manifest, name="v3", steps=4, protocol={"version": 3}
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    out = probe_rerun(db, exp_id)
    assert out["v1_answers_differing"] == 0 and out["v1_summary_equal"]
    assert out["v2_stored"] and out["v2_answers_differing"] == 0 and out["v2_summary_equal"]
    stored = json.loads(Path(out["checkpoint"]).parent.joinpath("result.json").read_text())
    for key in ("value", "swap", "vocab_size", "pass_accuracy"):
        assert out["v3"][key] == stored["probe_v3"][key], key
    stored_db = db.row("SELECT record_json FROM results WHERE exp_id = ?", (exp_id,))
    assert stored_db is not None
    compact = json.loads(stored_db["record_json"])
    assert "probes_v3" not in compact and "probes_v3_originals" not in compact
    assert compact["probe_v3"] == stored["probe_v3"]  # the summary stays in the database
    row = build_report(db, per_seed=True).rows[0]
    assert row["v3_vocab"] == stored["probe_v3"]["vocab_size"]
    assert row["swap_follow"] == stored["probe_v3"]["swap/follow_frac"]
