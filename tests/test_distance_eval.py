"""Score by distance (``eval.by_distance``, :mod:`hsn.eval.distance`).

The key is opt-in and out of the configuration hash at its default; the capture places every
loss position of a pass (aligned and unaligned streams, documents that span windows) and
changes no metric; the recall and text tables equal brute-force readings of the same pass; a
run with the key on carries the block, and the CLI scores the run's checkpoint to the same
tables.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hsn.config import REPO_ROOT, canonical_json, config_hash, dump_resolved, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.splits import build_splits
from hsn.data.synthetic import (
    BOS,
    EOS,
    NONE,
    PAD,
    SEP,
    DatasetSpec,
    Example,
    TaskSpec,
    generate,
    load_dataset_spec,
)
from hsn.data.text import SourceSpec, TextSpec, build_text
from hsn.device import resolve, seed_everything
from hsn.eval.distance import (
    DISTANCE_KEYS,
    ByDistance,
    Captured,
    PositionCapture,
    RunScorer,
    SetScorer,
    TextIndex,
    fact_positions,
    find_frequent_words,
    name_list,
    recall_table,
    render,
    text_table,
)
from hsn.eval.distance.__main__ import main as distance_main
from hsn.eval.distance.capture import flat_documents
from hsn.eval.distance.recall import value_range
from hsn.eval.distance.text import BOOTSTRAP, FrequentWords, build_frequent_words
from hsn.eval.evaluate import DEFAULT_METRICS, evaluate_windows
from hsn.experiments import db as dbm
from hsn.experiments.cli import main as experiments_cli
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import arm_identity, config_view, protocol_params
from hsn.runner import build_model, make_trainer, run_one
from hsn.train import compile as compile_mod
from hsn.train import tbptt
from hsn.train.compile import compile_window
from hsn.train.loop import StreamLoader
from p1_11_support import (
    GRU_2M,
    REF_A1,
    SPEC,
    TINY_GRU_OVERRIDES,
    TINY_HSN_OVERRIDES,
    UNROLL,
    cpu_ctx,
    db_path,
    disk_overrides,
    docs,
    enqueue_spec,
    isolated,  # noqa: F401  (fixture)
    restore_determinism,  # noqa: F401  (fixture)
    tiny_gru_config,
    tiny_hsn_config,
    tiny_manifest,  # noqa: F401  (fixture)
    train_overrides,
    write_yaml,
)

pytestmark = pytest.mark.usefixtures("restore_determinism")

REF_HASH = (
    "0831924010f9c5ecf6f1794c811f0c61ac96cdb5e3fbcc5a3c99b97a8001aa70"  # ref_a1, main 80b893d
)
ON = "eval.by_distance.enabled=true"

# --------------------------------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------------------------------


_FAR_SET = "eval.by_distance.sets=[{name: far, manifest: data/far/manifest.json}]"
_ALL_ON = [ON, _FAR_SET, "eval.by_distance.every_eval=true", "eval.by_distance.max_docs=100"]


def test_by_distance_is_off_by_default_and_never_in_the_hash() -> None:
    cfg = load_config(REF_A1)
    by = cfg.eval.by_distance
    assert (by.enabled, by.every_eval, by.max_docs, by.sets) == (False, False, None, [])
    assert config_hash(cfg) == REF_HASH
    # evaluation-only: on, with every key set, it is the same configuration to the hash
    on = load_config(REF_A1, _ALL_ON)
    assert on.eval.by_distance.enabled and on.eval.by_distance.sets[0].name == "far"
    assert on != cfg and canonical_json(on) == canonical_json(cfg)
    assert config_hash(on) == REF_HASH
    assert '"eval"' not in canonical_json(on) and "by_distance" not in canonical_json(on)
    # a training key still changes it, with the report on or off
    assert config_hash(load_config(REF_A1, [*_ALL_ON, "train.lr=0.002"])) != REF_HASH


@pytest.mark.parametrize("protocol", ["train_eval", "reads_state"])
def test_an_arm_keeps_its_id_with_the_report_on(protocol: str) -> None:
    """The key enabled, with sets: the arm id and the group id of the same arm without it."""
    params = protocol_params({"name": protocol}).model_dump(mode="json")

    def identity(*overrides: str) -> tuple[str, str]:
        return arm_identity(
            load_config(REF_A1, [*TINY_HSN_OVERRIDES, *overrides]),
            family="hsn",
            variant="pdg",
            variant_params={},
            protocol=params,
            early_stop=[],
            data="0123456789abcdef",
        )

    plain = identity()
    assert identity(*_ALL_ON) == plain and identity(ON) == plain
    assert identity(*_ALL_ON, "seed=1")[0] != plain[0]  # another seed is another arm
    assert identity(*_ALL_ON, "seed=1")[1] == plain[1]  # of the same group
    assert identity(*_ALL_ON, "train.lr=0.002") != plain
    assert "eval" not in config_view(load_config(REF_A1, _ALL_ON))


@pytest.mark.parametrize(
    "overrides",
    [
        ["eval.by_distance.every_eval=true"],
        ["eval.by_distance.max_docs=10"],
        ["eval.by_distance.sets=[{name: far, manifest: m.json}]"],
        [ON, "eval.by_distance.max_docs=0"],
        [ON, "eval.by_distance.sets=[{name: dev, manifest: m.json}]"],
        [ON, "eval.by_distance.sets=[{name: 'a b', manifest: m.json}]"],
        [ON, "eval.by_distance.sets=[{name: a, manifest: m.json}, {name: a, manifest: n.json}]"],
        [ON, "eval.by_distance.sets=[{name: a, manifest: m.json, split: test}]"],
    ],
)
def test_by_distance_keys_are_validated(overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        load_config(REF_A1, overrides)


def test_enqueueing_an_existing_arm_with_the_report_on_says_what_to_do(
    isolated: Path,  # noqa: F811
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report is not part of an arm id, so an arm that exists is skipped as ever; the
    enqueue says that its attempt holds other ``eval.*`` settings and how to score it."""

    def spec(name: str, *extra: str) -> Path:
        overrides = [*TINY_HSN_OVERRIDES, *train_overrides(steps=4, eval_every=2), *extra]
        raw = {
            "name": name,
            "base": {"config": str(REF_A1), "overrides": overrides},
            "protocol": {"name": "dry_run"},
            "seeds": [0, 1],
            "arms": [{"name": "a", "family": "hsn"}],
        }
        return write_yaml(tmp_path / f"{name}.yaml", raw)

    db = dbm.Database(db_path())
    first = enqueue_spec(db, spec("plain"))
    arm_ids = [a["arm_id"] for a in first.added]
    with db.write() as conn:
        conn.execute(
            "UPDATE experiments SET status = 'done', run_dir = '/runs/plain/a/exp1' "
            "WHERE exp_id = ?",
            (first.added[0]["exp_id"],),
        )
    same = enqueue_spec(db, spec("plain"))
    assert not same.added and all("eval_differs" not in s for s in same.skipped)
    with_report = spec("with_report", ON, _FAR_SET)
    on = enqueue_spec(db, with_report)
    assert not on.added and [s["arm_id"] for s in on.skipped] == arm_ids  # the same arms
    assert [(s["status"], s.get("eval_differs"), s.get("run_dir")) for s in on.skipped] == [
        ("done", True, "/runs/plain/a/exp1"),
        ("queued", True, None),
    ]
    assert experiments_cli(["enqueue", str(with_report)]) == 0
    printed = capsys.readouterr().out
    assert "python -m hsn.eval.distance --run-dir /runs/plain/a/exp1" in printed
    assert "a/s1: exp 2 (queued) keeps the eval.* settings it was queued with" in printed
    assert "0 added, 2 skipped" in printed
    # --force trains the arm again: the new attempt of the same arm holds the report
    forced = enqueue_spec(db, with_report, force=True)
    assert [a["arm_id"] for a in forced.added] == arm_ids
    stored = json.loads(db.experiment(forced.added[0]["exp_id"])["config_json"])
    assert stored["eval"]["by_distance"]["enabled"] is True
    after = enqueue_spec(db, with_report)  # the newest attempts now hold what is asked for
    assert not after.added and all("eval_differs" not in s for s in after.skipped)


# --------------------------------------------------------------------------------------------------
# The capture
# --------------------------------------------------------------------------------------------------


def _long_task(family: str, *, unroll: int = UNROLL, dep: tuple[int, int] = (40, 120)) -> TaskSpec:
    """An out-of-range ``family`` task whose documents span several windows of ``unroll``."""
    base = SPEC.task(family, "out_of_range")  # type: ignore[arg-type]
    return TaskSpec.model_validate(
        {
            **base.model_dump(),
            "n_train": 16,
            "n_dev": 64,
            "n_test": 16,
            "length_range": [dep[0] + 16, dep[1] + 40],
            "dep_len_range": list(dep),
            "unroll": unroll,
        }
    )


def _mixed_documents() -> list[Example]:
    long = [generate(_long_task(f), "dev", i) for f in ("kv", "delayed") for i in range(10)]
    return [*long[:10], *docs("kv", "dev", 6), *long[10:], *docs("delayed", "dev", 6)]


class _Counting(torch.nn.Module):
    """Logits that depend on the byte read and, through the carried state, on its position in
    the document: a loss put at the wrong position or document is a different number."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.table = torch.randn(256, 256, generator=generator)

    def init_state(self, B: int, device: Any = None) -> torch.Tensor:
        return torch.zeros(B, dtype=torch.int64)

    def logits_of(self, tokens: torch.Tensor) -> torch.Tensor:
        """A whole document's logits, read on its own."""
        return self.table[(tokens.to(torch.int64) + 7 * torch.arange(tokens.shape[0])) % 256]

    def forward(
        self, tokens: torch.Tensor, context: torch.Tensor, state: torch.Tensor, reset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        B, T = tokens.shape
        out = torch.empty(B, T, 256)
        state = state.clone()
        for b in range(B):
            position = int(state[b])
            for t in range(T):
                if bool(reset[b, t]):
                    position = 0
                out[b, t] = self.table[(int(tokens[b, t]) + 7 * position) % 256]
                position += 1
            state[b] = position
        return out, state, {}


def _loader(ds: ByteDataset, *, align: bool, batch: int = 4) -> StreamLoader:
    return StreamLoader(
        ds, batch=batch, unroll=UNROLL, seed=0, shuffle=False, align_docs=align, epochs=1
    )


@pytest.mark.parametrize("align", [True, False])
def test_capture_places_every_loss_position_and_changes_no_metric(align: bool) -> None:
    examples = _mixed_documents()
    ds = ByteDataset(examples)
    assert max(e.length for e in examples) > 3 * UNROLL  # documents that span windows
    model = _Counting()
    rows_for = [e.meta["family"] == "delayed" for e in examples]
    capture = PositionCapture(ds, rows_for)
    device = torch.device("cpu")
    watched = evaluate_windows(
        model, _loader(ds, align=align), DEFAULT_METRICS, device=device, observer=capture
    )
    plain = evaluate_windows(model, _loader(ds, align=align), DEFAULT_METRICS, device=device)
    assert watched == plain
    got = capture.result()
    assert np.array_equal(
        got.offsets, np.concatenate([[0], np.cumsum([e.length for e in examples])])
    )
    n_rows = 0
    for d, example in enumerate(examples):
        span = slice(int(got.offsets[d]), int(got.offsets[d + 1]))
        logits = model.logits_of(torch.from_numpy(example.tokens.copy()))
        targets = torch.from_numpy(example.targets.astype(np.int64))
        mask = example.loss_mask
        nll = F.cross_entropy(logits, targets, reduction="none").numpy()
        assert np.isnan(got.nll[span][~mask]).all() and (got.pred[span][~mask] == -1).all()
        np.testing.assert_allclose(got.nll[span][mask], nll[mask], rtol=1e-6, atol=1e-6)
        assert np.array_equal(got.pred[span][mask], logits.argmax(-1).numpy()[mask])
        assert np.array_equal(got.hit[span], (logits.argmax(-1) == targets).numpy() & mask)
        rows = got.row_at[span]
        if not rows_for[d]:
            assert (rows == -1).all()
            continue
        assert (rows[mask] >= 0).all() and (rows[~mask] == -1).all()
        np.testing.assert_allclose(
            got.rows[rows[mask]], torch.log_softmax(logits, -1).numpy()[mask], atol=1e-5
        )
        n_rows += int(mask.sum())
    assert got.rows.shape == (n_rows, 256)
    assert float(np.nanmean(got.nll.astype(np.float64))) == pytest.approx(watched["loss"], abs=1e-6)


def test_capture_refuses_a_partial_or_foreign_pass() -> None:
    ds = ByteDataset(_mixed_documents())
    model = _Counting()
    capture = PositionCapture(ds)
    windows = list(_loader(ds, align=True))
    state = model.init_state(4)
    for window in windows[:-1]:  # the last window is never seen
        logits, state, _ = model(window["tokens"], window["context"], state, window["reset"])
        capture(window, logits)
    with pytest.raises(RuntimeError, match="every one must be visited once"):
        capture.result()
    other = PositionCapture(ByteDataset(list(reversed(ds.examples))))
    with pytest.raises(RuntimeError, match="do not match their documents"):
        for window in windows:
            other(window, torch.zeros(4, UNROLL, 256))


# --------------------------------------------------------------------------------------------------
# Recall tables against a brute-force reading
# --------------------------------------------------------------------------------------------------

_BUCKETS = [
    (0, 63, "lt64"),
    (64, 255, "64_255"),
    (256, 511, "256_511"),
    (512, 768, "512_768"),
    (769, 1023, "769_1023"),
    (1024, 1535, "1024_1535"),
    (1536, 2047, "1536_2047"),
    (2048, 3072, "2048_3072"),
]
_COARSE = [(0, 255), (256, 767), (768, 1535), (1536, 10**9)]
_COARSE_FACT = ["0_255", "256_767", "768_1535", "ge1536"]
_COARSE_DISTANCE = ["lt256", "256_767", "768_1535", "ge1536"]


def _bucket(distance: int) -> str:
    for lo, hi, name in _BUCKETS:
        if lo <= distance <= hi:
            return name
    return "gt3072"


def _coarse(value: int, names: list[str]) -> str:
    return next(name for (lo, hi), name in zip(_COARSE, names, strict=True) if lo <= value <= hi)


def _far_documents(family: str, n: int) -> list[Example]:
    """The far set's shape: distances from 272 to 3,072 in documents of up to 3,200 bytes."""
    task = _long_task(family, unroll=256, dep=(272, 3072))
    task = TaskSpec.model_validate({**task.model_dump(), "length_range": [288, 3200]})
    return [generate(task, "dev", i) for i in range(n)]


def _fake_pass(examples: list[Example], seed: int) -> Captured:
    """A made-up pass: random log-probability rows at every answer, their arg-max as the
    prediction, the target's loss; the right answer made likely at short distances."""
    ds = ByteDataset(examples)
    offsets, tokens = flat_documents(ds)
    rng = np.random.default_rng(seed)
    n = int(offsets[-1])
    nll = np.full(n, np.nan, dtype=np.float32)
    pred = np.full(n, -1, dtype=np.int16)
    row_at = np.full(n, -1, dtype=np.int32)
    rows = []
    for d, example in enumerate(examples):
        for slot, position in enumerate(np.flatnonzero(example.loss_mask).tolist()):
            flat = int(offsets[d]) + position
            target = int(example.targets[position])
            logits = rng.normal(size=256) * 2.0
            logits[NONE] += 3.0
            if int(example.meta["deps"][slot]) < 700 and rng.random() < 0.7:
                logits[target] += 8.0
            row = torch.log_softmax(torch.from_numpy(logits), -1).numpy().astype(np.float32)
            row_at[flat] = len(rows)
            rows.append(row)
            nll[flat], pred[flat] = -row[target], int(row.argmax())
    return Captured(offsets, tokens, nll, pred, row_at, np.stack(rows))


def _acc(records: list[dict[str, Any]], key: str = "ok") -> dict[str, Any]:
    return {
        "n": len(records),
        "accuracy": float(np.mean([r[key] for r in records])) if records else None,
    }


@pytest.mark.parametrize("family", ["kv", "delayed"])
def test_recall_table_equals_a_brute_force_reading(family: str) -> None:
    # one group for the table, which reads a list of documents and not their labels
    examples = [*_far_documents(family, 60), *docs(family, "dev", 12)]
    captured = _fake_pass(examples, seed=5)
    window = 256
    table = recall_table(examples, list(range(len(examples))), captured, window=window)
    records = []
    for d, example in enumerate(examples):
        tokens = example.tokens
        for slot, position in enumerate(np.flatnonzero(example.loss_mask).tolist()):
            flat = int(captured.offsets[d]) + position
            distance = int(example.meta["deps"][slot])
            target, pred = int(tokens[position + 1]), int(captured.pred[flat])
            records.append(
                {
                    "doc": d,
                    "slot": slot,
                    "distance": distance,
                    "fact": position - distance,
                    "back": position // window - (position - distance) // window,
                    "target": target,
                    "pred": pred,
                    "ok": pred == target,
                    "row": captured.rows[captured.row_at[flat]],
                    "nll": float(captured.nll[flat]),
                }
            )
    assert table["kind"] == "recall" and table["family"] == family
    assert (table["n_docs"], table["n"]) == (len(examples), len(records))
    assert table["accuracy"] == pytest.approx(_acc(records)["accuracy"])
    assert table["loss"] == pytest.approx(np.mean([r["nll"] for r in records]))
    by_distance = {
        name: _acc([r for r in records if _bucket(r["distance"]) == name]) for name in DISTANCE_KEYS
    }
    by_distance = {k: v for k, v in by_distance.items() if v["n"]}
    assert table["by_distance"] == by_distance
    assert list(table["by_distance"]) == [k for k in DISTANCE_KEYS if k in by_distance]
    assert {"lt64", "256_511", "512_768", "769_1023", "2048_3072"} <= set(by_distance)
    assert sum(cell["n"] for cell in table["by_distance"].values()) == len(records)
    backs = sorted({r["back"] for r in records})
    assert table["by_windows_back"] == {
        str(b): _acc([r for r in records if r["back"] == b]) for b in backs
    }
    assert max(backs) >= 8
    assert table["by_query"] == {
        "first": _acc([r for r in records if r["slot"] == 0]),
        "later": _acc([r for r in records if r["slot"] > 0]),
    }
    grid: dict[str, Any] = {}
    for r in records:
        row = grid.setdefault(_coarse(r["fact"], _COARSE_FACT), {})
        row.setdefault(_coarse(r["distance"], _COARSE_DISTANCE), []).append(r)
    assert table["by_fact_position"] == {
        fact: {dist: _acc(row[dist]) for dist in _COARSE_DISTANCE if dist in row}
        for fact, row in ((f, grid[f]) for f in _COARSE_FACT if f in grid)
    }
    assert len(table["by_fact_position"]) >= 3
    if family == "kv":
        nones = [r for r in records if r["target"] == NONE]
        values = [r for r in records if r["target"] != NONE]
        assert nones and values
        assert table["by_target"] == {"value": _acc(values), "none": _acc(nones)}
        assert table["predicts_none"] == pytest.approx(
            np.mean([r["pred"] == NONE for r in records])
        )
        assert table["by_distance_value_targets"] == {
            name: _acc([r for r in values if _bucket(r["distance"]) == name])
            for name in DISTANCE_KEYS
            if any(_bucket(r["distance"]) == name for r in values)
        }
        assert "own_values" not in table
        return
    # delayed: the prediction restricted to the values the document states (facts k v before SEP)
    assert "by_target" not in table
    for r in records:
        tokens = examples[r["doc"]].tokens
        end = int(np.flatnonzero(tokens == SEP)[0])
        stated = sorted({int(v) for v in tokens[1:end] if 144 <= v < 192})
        r["right"] = stated[int(np.argmax(r["row"][stated]))] == r["target"]
        r["chance"] = 1.0 / len(stated)
        r["inside"] = r["pred"] in stated
        r["mass"] = float(np.exp(r["row"][stated].astype(np.float64)).sum())
    own = table["own_values"]
    # the alphabet is the range the group's answers span: every value from 144 to 191 here
    assert own["alphabet"] == [144, 191] == list(value_range(examples) or ())
    assert own["accuracy"] == pytest.approx(np.mean([r["right"] for r in records]))
    assert own["chance"] == pytest.approx(np.mean([r["chance"] for r in records]))
    assert own["predicted_share"] == pytest.approx(np.mean([r["inside"] for r in records]))
    assert own["mass"] == pytest.approx(np.mean([r["mass"] for r in records]), rel=1e-5)
    assert own["by_distance"] == pytest.approx(
        {
            name: float(np.mean([r["right"] for r in records if _bucket(r["distance"]) == name]))
            for name in by_distance
        }
    )
    assert own["accuracy"] > table["accuracy"]  # NONE outranks a value outside the restriction


def test_other_families_get_the_distance_tables_only() -> None:
    task = SPEC.task("arithmetic", "in_range")
    examples = [generate(task, "dev", i) for i in range(8)]
    table = recall_table(examples, list(range(8)), _fake_pass(examples, seed=1), window=256)
    assert table["kind"] == "answers" and table["family"] == "arithmetic"
    assert set(table) == {
        "kind",
        "family",
        "n_docs",
        "n",
        "accuracy",
        "loss",
        "by_distance",
        "by_windows_back",
    }


def test_word_key_families_are_recall_groups() -> None:
    """``kvw`` and ``delayedw`` (word keys, the answer predicted at the SPACE after the word)."""
    spec = load_dataset_spec(REPO_ROOT / "configs" / "data" / "synthetic_wordhold.yaml")
    for task, family in (("kvw", "kv_words"), ("delayedw", "delayed_words")):
        examples = [generate(spec.task(task, "in_range"), "dev", i) for i in range(40)]
        captured = _fake_pass(examples, seed=2)
        table = recall_table(examples, list(range(40)), captured, window=256)
        assert (table["kind"], table["family"]) == ("recall", family)
        assert table["by_query"]["first"]["n"] == 40
        assert sum(cell["n"] for cell in table["by_distance"].values()) == table["n"]
        if family == "kv_words":
            assert table["by_target"]["none"]["n"] + table["by_target"]["value"]["n"] == table["n"]
            continue
        low, high = table["own_values"]["alphabet"]
        assert 144 <= low < high <= 191  # the spec's values; key letters and filler lie outside
        # every answer's fact is the value byte right after its key word
        for example in examples:
            answers = np.flatnonzero(example.loss_mask)
            assert np.array_equal(
                example.tokens[fact_positions(example)], example.tokens[answers + 1]
            )
        assert table["own_values"]["accuracy"] >= table["accuracy"]


def test_fact_positions_are_read_from_the_shard_when_it_has_them() -> None:
    example = _far_documents("delayed", 1)[0]
    answers = np.flatnonzero(example.loss_mask)
    derived = answers - np.asarray(example.meta["deps"])
    assert np.array_equal(fact_positions(example), derived)
    # the derived position is the fact's value: the byte the answer repeats
    assert np.array_equal(example.tokens[derived], example.tokens[answers + 1])
    stored = Example(
        example.tokens.copy(),
        example.targets.copy(),
        example.loss_mask.copy(),
        example.context.copy(),
        {**example.meta, "fact_pos": [int(p) + 1 for p in derived]},
    )
    assert np.array_equal(fact_positions(stored), derived + 1)
    bad = Example(
        example.tokens.copy(),
        example.targets.copy(),
        example.loss_mask.copy(),
        example.context.copy(),
        {**example.meta, "fact_pos": [0] * (len(derived) + 1)},
    )
    with pytest.raises(ValueError, match="entries"):
        fact_positions(bad)


# --------------------------------------------------------------------------------------------------
# Text tables
# --------------------------------------------------------------------------------------------------

_TEXT_WINDOW = 64


def _text_example(body: bytes, index: int, task: str = "stories") -> Example:
    tokens = np.frombuffer(bytes([BOS]) + body + bytes([EOS]), dtype=np.uint8).copy()
    targets = np.empty_like(tokens)
    targets[:-1] = tokens[1:]
    targets[-1] = PAD
    mask = np.ones(tokens.shape[0], dtype=np.bool_)
    mask[-1] = False
    meta = {
        "task": task,
        "variant": "out_of_range",
        "family": "text",
        "split": "dev",
        "index": index,
        "doc_id": f"{task}:{index}",
        "dep_len": 1,
        "in_range": False,
        "deps": [1] * int(mask.sum()),
    }
    return Example(tokens, targets, mask, np.full(tokens.shape[0], 6, dtype=np.uint8), meta)


def _stories(n: int = 24) -> list[bytes]:
    """Stories with names that return at chosen distances (newlines are the marker 16)."""
    rng = np.random.default_rng(3)
    people = ["Lily", "Tom", "Ben", "Mama", "Hope"]
    filler = "the cat and the dog ran to the big red tree and they had fun all day long"
    out = []
    for i in range(n):
        a, b = people[i % 3], people[(i + 1) % 3]
        gap = " ".join(rng.permutation(filler.split()).tolist() * int(rng.integers(1, 9)))
        text = (
            f'{a} went out. One day {a} saw {b} and Mama.\x10\x10"{b} is here," said {a}. '
            f"We hope that Hope and {b} can play. Then {gap} and so {a} did not stop. "
            f"After that {gap[: 30 + 7 * i]} it was {b} who ran home with {a}'s hat."
        )
        out.append(text.encode())
    return out


def _brute_names(
    bodies: list[bytes], names: list[bytes], bits: list[np.ndarray[Any, Any]], window: int
) -> tuple[dict[str, dict[str, float]], int]:
    """The grammar evaluation's "which name" reading, mention by mention."""
    cases: dict[str, list[tuple[float, int, float]]] = {}
    dropped = 0
    for body, doc_bits in zip(bodies, bits, strict=True):
        last: dict[bytes, int] = {}
        for match in re.finditer(rb"[A-Za-z]+", body):
            word, s, e = match.group(), match.start(), match.end()
            if word not in names:
                continue
            k = s - 1
            while k > 0 and body[k] == 32:
                k -= 1
            initial = body[k] in b".!?\"'" or body[k] in (1, 16) or body[k] >= 128
            previous = last.get(word)
            last[word] = s
            if (s - 1) // window != (e - 2) // window:
                dropped += 1
                continue
            if initial:
                continue
            distance = -1 if previous is None else s - previous
            same = previous is not None and previous >= ((s - 1) // window) * window
            keys = []
            if distance < 0:
                keys.append("first_occurrence")
            elif same:
                keys.append("same_window_lt64" if distance < 64 else "same_window_64_255")
            elif distance < 64:
                keys.append("earlier_window_lt64")
            elif distance < 256:
                keys.append("earlier_window_64_255")
            else:
                keys.append("ge256")
            if 256 <= distance < 384:
                keys.append("ge256_256_383")
            elif 384 <= distance < 512:
                keys.append("ge256_384_511")
            elif distance >= 512:
                keys.append("ge256_ge512")
            rest = float(doc_bits[s : e - 1].sum())  # letters 2.. are predicted at s .. e - 2
            for key in keys:
                cases.setdefault(key, []).append((rest, e - s - 1, float(doc_bits[s - 1])))
    out = {
        key: {
            "n": len(rows),
            "letters": sum(r[1] for r in rows),
            "bits_rest": sum(r[0] for r in rows) / sum(r[1] for r in rows),
            "bits_first": sum(r[2] for r in rows) / len(rows),
        }
        for key, rows in cases.items()
    }
    return out, dropped


def test_name_list_follows_the_rule() -> None:
    bodies = [bytes([BOS]) + body for body in _stories()]
    names = name_list(bodies)
    assert names == [b"Ben", b"Lily", b"Tom"]
    # "Mama" passes the rule and is not a name; "Hope" is mostly the lower-case word
    assert all(body.count(b"Mama") == 1 and body.count(b"hope") == 1 for body in bodies)
    assert name_list(bodies[:1]) == []  # fewer than five mentions inside a sentence


def test_text_table_equals_a_brute_force_reading() -> None:
    stories = _stories()
    examples = [_text_example(body, i) for i, body in enumerate(stories)]
    bodies = [bytes(e.tokens) for e in examples]
    names = name_list(bodies)
    frequent = FrequentWords(frozenset({b"the", b"and", b"to"}), 3, "test.json")
    index = TextIndex(bodies, names, window=_TEXT_WINDOW, frequent=frequent)
    rng = np.random.default_rng(9)
    n = sum(len(b) for b in bodies)
    nll = rng.gamma(2.0, 0.6, size=n).astype(np.float32)
    hit = rng.random(n) < 0.6
    ends = np.cumsum([len(b) for b in bodies]) - 1
    nll[ends] = np.nan  # a story's last position has no target
    hit[ends] = False
    table = text_table(index, nll, hit)
    bits = nll.astype(np.float64) / np.log(2.0)
    per_doc = np.split(bits, np.cumsum([len(b) for b in bodies])[:-1])
    scored = np.isfinite(bits)
    assert table["kind"] == "text" and (table["n_docs"], table["n"]) == (
        len(bodies),
        n - len(bodies),
    )
    assert table["bits_per_byte"] == pytest.approx(bits[scored].mean())
    assert table["accuracy"] == pytest.approx(hit[scored].mean())

    brute, dropped = _brute_names(bodies, names, per_doc, _TEXT_WINDOW)
    rows = table["names"]["rows"]
    assert list(rows) == [
        "same_window_lt64",
        "same_window_64_255",
        "earlier_window_lt64",
        "earlier_window_64_255",
        "ge256",
        "ge256_256_383",
        "ge256_384_511",
        "ge256_ge512",
        "first_occurrence",
    ]
    assert table["names"]["dropped_window_straddle"] == dropped > 0
    assert (table["names"]["n_names"], table["names"]["window"]) == (3, _TEXT_WINDOW)
    populated = {key for key, row in rows.items() if row["n"]}
    assert populated == set(brute)
    assert {"same_window_lt64", "earlier_window_64_255", "ge256", "ge256_ge512"} <= populated
    assert rows["ge256"]["n"] == sum(rows[k]["n"] for k in rows if k.startswith("ge256_"))
    for key, expected in brute.items():
        row = rows[key]
        assert (row["n"], row["letters"]) == (expected["n"], expected["letters"])
        assert row["bits_rest"] == pytest.approx(expected["bits_rest"])
        assert row["bits_first"] == pytest.approx(expected["bits_first"])
        for which in ("rest", "first"):
            low, high = row[f"ci95_{which}"]
            assert low <= row[f"bits_{which}"] <= high
    empty = next(row for key, row in rows.items() if key not in populated)
    assert empty == {
        "n": 0,
        "letters": 0,
        "bits_rest": None,
        "ci95_rest": None,
        "bits_first": None,
        "ci95_first": None,
    }
    # the intervals are the percentiles of the resampled stories, with the fixed seed
    assert BOOTSTRAP == 2000 and text_table(index, nll, hit) == table

    # words: every letter of every content word, by the word's last use in the story
    cases: dict[str, list[float]] = {}
    for body, doc_bits in zip(bodies, per_doc, strict=True):
        seen: dict[bytes, int] = {}
        for match in re.finditer(rb"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", body):
            s, e = match.span()
            text = match.group().lower()
            before = seen.get(text)
            seen[text] = s
            if text in frequent.words:
                continue
            distance = -1 if before is None else s - before
            for k in range(e - s):
                value = float(doc_bits[s - 1 + k])
                if distance < 0:
                    cases.setdefault("new", []).append(value)
                elif distance < 64:
                    cases.setdefault("lt64", []).append(value)
                elif distance < 256:
                    cases.setdefault("64_255", []).append(value)
                elif distance < 512:
                    cases.setdefault("256_511", []).append(value)
                else:
                    cases.setdefault("ge512", []).append(value)
                if distance >= 256:
                    part = "ge256_first_letter" if k == 0 else "ge256_later_letters"
                    cases.setdefault(part, []).append(value)
    words = table["words"]
    assert (words["frequent_words"], words["threshold"]) == ("test.json", 3)
    assert set(cases) == set(words["rows"])
    for key, values in cases.items():
        row = words["rows"][key]
        assert row["n"] == len(values) and row["share"] == pytest.approx(len(values) / table["n"])
        assert row["bits"] == pytest.approx(np.mean(values))
    without = text_table(TextIndex(bodies, names, window=_TEXT_WINDOW, frequent=None), nll, hit)
    assert "skipped" in without["words"] and without["names"] == table["names"]


def test_frequent_words_are_counted_across_chunks_and_found_by_dataset(tmp_path: Path) -> None:
    text = b"".join(bytes([BOS]) + body + bytes([EOS]) for body in _stories())
    np.savez(tmp_path / "shard.npz", tokens=np.frombuffer(text, dtype=np.uint8))
    counts = Counter(
        m.group().lower() for m in re.finditer(rb"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text)
    )
    expected = sorted(word.decode() for word, n in counts.items() if n >= 40)
    assert expected and "the" in expected and "lily's" not in expected
    for chunk in (7, 64, 1 << 20):  # a block boundary never cuts a word
        built = build_frequent_words(tmp_path / "shard.npz", threshold=40, chunk=chunk)
        assert built["words"] == expected
        assert (built["n_words"], built["n_types"]) == (sum(counts.values()), len(counts))
    # the committed TinyStories list (words the training shard uses 100,000 times or more)
    v1 = find_frequent_words("tinystories_v1")
    assert v1 is not None and v1.threshold == 100_000 and len(v1.words) == 68
    assert {b"the", b"lily", b"tim"} <= v1.words and b"tree" in v1.words and b"have" not in v1.words
    assert find_frequent_words("tinystories_w1024") == v1
    assert find_frequent_words("synthetic_v1") is None and find_frequent_words(None) is None
    listed = json.loads((REPO_ROOT / "configs/data" / v1.source).read_text())
    golden = json.loads((REPO_ROOT / "configs/data/tinystories_v1.manifest.json").read_text())
    train = next(s for s in golden["shards"] if s["split"] == "train")
    assert listed["source"]["hash"] == train["hash"]


# --------------------------------------------------------------------------------------------------
# The trainer's observer, a run, the CLI
# --------------------------------------------------------------------------------------------------


def test_the_observer_adds_its_block_and_changes_no_metric() -> None:
    cfg = tiny_gru_config()
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=None)
    ds = ByteDataset([*docs("kv", "dev", 12), *docs("delayed", "dev", 12)])
    plain = trainer.evaluate(ds)
    trainer.eval_observer = ByDistance(window=UNROLL)
    watched = trainer.evaluate(ds)
    again = trainer.evaluate(ds)  # the dataset's scorer is reused
    groups = watched.pop("by_distance")
    assert watched == plain and again.pop("by_distance") == groups
    assert list(groups) == ["kv/in_range", "delayed/in_range"] == list(plain["per_group"])
    for name, table in groups.items():
        assert table["accuracy"] == pytest.approx(plain["per_group"][name]["accuracy"], abs=1e-12)
        assert table["loss"] == pytest.approx(plain["per_group"][name]["loss"], abs=1e-6)
        assert table["n"] == plain["per_group"][name]["n_positions"]
    assert "own_values" in groups["delayed/in_range"] and "by_target" in groups["kv/in_range"]
    capped = SetScorer(ds, window=UNROLL, max_docs=5)
    assert [len(v) for v in capped.groups.values()] == [5, 5]
    assert capped.ds.examples[5] is ds.examples[12]


def test_a_compiled_window_is_not_recompiled_by_the_observer_or_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compiled HSN window (aot_eager on the CPU, as the P2.8 trainer tests): with the
    observer on every evaluation and the pass at the end, training is the same step for step,
    the same steps compile, and the pass runs on the evaluation variant that exists."""
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 2)
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    train = ByteDataset([*docs("kv", "train", 16), *docs("delayed", "train", 16)])
    dev = ByteDataset([*docs("kv", "dev", 8), *docs("delayed", "dev", 8)])
    histories = {}
    try:
        for name in ("plain", "observed"):
            cfg = tiny_hsn_config(
                "hardware.compile_mode=reduce-overhead", *([ON] * (name != "plain"))
            )
            seed_everything(cfg.seed)
            trainer = make_trainer(
                build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / name
            )
            assert trainer.compiled_window is not None
            scorer = RunScorer(cfg, dev) if name == "observed" else None
            if scorer is not None:
                trainer.eval_observer = scorer.observer()
            histories[name] = trainer.fit(train, dev, 6)
            trainer.eval_observer = None
            if scorer is None:
                continue
            window = trainer.compiled_window
            calls, compiled_s = window.calls, window.compile_seconds
            block = scorer.final(trainer)
            entry = block["sets"]["dev"]
            assert entry["compile_seconds"] == 0.0 and window.compile_seconds == compiled_s
            assert window.calls == calls + entry["n_windows"]  # the pass ran compiled
            assert list(entry["groups"]) == ["kv/in_range", "delayed/in_range"]
            assert entry["accuracy"] == pytest.approx(histories[name].evals[-1]["accuracy"])
    finally:
        torch._dynamo.reset()
    plain, observed = histories["plain"], histories["observed"]
    assert [r.loss for r in observed.steps] == [r.loss for r in plain.steps]
    assert [r.grad_norm for r in observed.steps] == [r.grad_norm for r in plain.steps]
    for key in ("compile_steps", "compiled_windows"):
        assert observed.measured[key] == plain.measured[key] > 0
    assert [e["step"] for e in observed.evals] == [e["step"] for e in plain.evals] == [3, 6]
    for a, b in zip(observed.evals, plain.evals, strict=True):
        tables = a.pop("by_distance")
        assert a == b and list(tables) == ["kv/in_range", "delayed/in_range"]


@pytest.fixture(scope="module")
def far_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second tiny dataset whose distances exceed the window (the role of the far set)."""
    out = tmp_path_factory.mktemp("tiny_far")
    tasks = [
        TaskSpec.model_validate({**_long_task(f).model_dump(), "n_dev": 12})
        for f in ("kv", "delayed")
    ]
    build_splits(DatasetSpec(name="tiny_far", unroll=UNROLL, tasks=tasks), out)
    return out / "manifest.json"


def _arm(family: str, manifest: Path, run_dir: Path, *overrides: str, **protocol: Any) -> ArmRun:
    extra = [*disk_overrides(manifest, steps=4), *overrides]
    if family == "hsn":
        cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *extra])
    else:
        cfg = load_config(GRU_2M, [*TINY_GRU_OVERRIDES, *extra])
    params = protocol_params(
        {"name": "train_eval", "eval_batch": 12, "diagnostics": False, **protocol}
    )
    return ArmRun(
        cfg=cfg,
        family=family,
        variant="pdg" if family == "hsn" else "none",
        variant_params={},
        params=params,
        run_dir=run_dir,
        rules=parse_rules(None),
        constants={},
    )


def _approx(value: Any) -> Any:
    """``value`` with every float compared to rounding (one batch layout or another)."""
    if isinstance(value, dict):
        return {k: _approx(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_approx(v) for v in value]
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-5, abs=1e-6)
    return value


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_a_run_carries_the_block_and_the_cli_reproduces_it(
    family: str,
    tiny_manifest: Path,  # noqa: F811
    far_manifest: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    off = run_protocol(_arm(family, tiny_manifest, tmp_path / "off"))
    assert "by_distance" not in off and "by_distance_s" not in off["walls"]
    assert all("by_distance" not in e for e in off["evals"])

    sets = f"eval.by_distance.sets=[{{name: far, manifest: {far_manifest}}}]"
    arm = _arm(family, tiny_manifest, tmp_path / "on", ON, sets, "eval.by_distance.every_eval=true")
    record = run_protocol(arm)
    # the same training: the key changes no step and no metric
    assert [t["loss"] for t in record["train"]] == [t["loss"] for t in off["train"]]
    assert record["final"] == off["final"]
    block = record["by_distance"]
    assert (block["schema"], block["step"], block["window"]) == (1, 4, UNROLL)
    assert (block["weights"], block["align_docs"]) == ("trained", True)
    assert list(block["sets"]) == ["dev", "far"]
    assert record["walls"]["by_distance_s"] > 0 and "by_distance" in record["phases"]["stages"]
    dev, far = block["sets"]["dev"], block["sets"]["far"]
    assert dev["source"]["dataset"] == "tiny_p1_11" and far["source"]["dataset"] == "tiny_far"
    assert (dev["source"]["n_docs"], far["source"]["n_docs"]) == (32, 24)
    assert dev["compile_seconds"] is None and far["seconds"] > 0  # an eager run compiles nothing
    assert dev["accuracy"] == pytest.approx(record["final"]["dev_accuracy"], abs=1e-12)
    assert dev["loss"] == pytest.approx(record["final"]["dev_loss"], abs=1e-6)
    assert list(dev["groups"]) == ["kv/in_range", "delayed/in_range"]
    assert list(far["groups"]) == ["kv/out_of_range", "delayed/out_of_range"]
    for name, table in dev["groups"].items():
        assert table["accuracy"] == pytest.approx(record["final"][f"{name}/accuracy"], abs=1e-12)
        assert set(table["by_distance"]) == {"lt64"} and set(table["by_windows_back"]) == {"0"}
    assert "64_255" in far["groups"]["delayed/out_of_range"]["by_distance"]
    assert set(far["groups"]["delayed/out_of_range"]["by_windows_back"]) - {"0"}
    # every evaluation carries the dev groups; the last one is the final block's dev set
    assert [e["step"] for e in record["evals"]] == [2, 4]
    assert all(list(e["by_distance"]) == list(dev["groups"]) for e in record["evals"])
    assert record["evals"][-1]["by_distance"] == _approx(dev["groups"])
    json.dumps(record["by_distance"], allow_nan=False)

    # the CLI on the run's checkpoint: the same tables
    dump_resolved(arm.cfg, arm.run_dir / "config.resolved.yaml")
    out = tmp_path / "report.json"
    assert (
        distance_main(["--run-dir", str(arm.run_dir), "--eval-batch", "12", "--out", str(out)]) == 0
    )
    report = json.loads(out.read_text())
    assert report["checkpoint"] == "step_00000004.pt" and report["device"] == "cpu"
    scored = report["by_distance"]
    assert (scored["step"], scored["weights"]) == (4, "trained")
    assert list(scored["sets"]) == ["dev", "far"]
    for name in ("dev", "far"):
        assert scored["sets"][name]["groups"] == _approx(block["sets"][name]["groups"])
        assert scored["sets"][name]["source"] == block["sets"][name]["source"]
    printed = capsys.readouterr().out
    assert (
        "kv/out_of_range: accuracy" in printed and "restricted to the document's values" in printed
    )
    # --set replaces the run's own sets, --no-dev leaves the dev set out, --max-docs caps a group
    code = distance_main(
        [
            "--checkpoint",
            str(arm.run_dir / "step_00000004.pt"),
            "--no-dev",
            "--max-docs",
            "5",
            "--set",
            f"again={far_manifest}:out_of_range:delayed",
            "--out",
            str(out),
        ]
    )
    again = json.loads(out.read_text())["by_distance"]["sets"]
    assert code == 0 and list(again) == ["again"]
    assert (
        again["again"]["source"]["n_docs"]
        == 5
        == again["again"]["groups"]["delayed/out_of_range"]["n_docs"]
    )


def test_hsn_run_writes_the_block_into_metrics_json(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    cfg = load_config(
        GRU_2M,
        [*TINY_GRU_OVERRIDES, *disk_overrides(tiny_manifest), ON, "eval.by_distance.max_docs=6"],
    )
    run_dir = run_one(cfg, run_id="on", out=tmp_path, profile_steps=2)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    block = metrics["by_distance"]
    assert (block["schema"], block["step"], list(block["sets"])) == (1, 4, ["dev"])
    dev = block["sets"]["dev"]
    assert (dev["source"]["n_docs"], dev["source"]["max_docs"]) == (12, 6)
    assert [table["n_docs"] for table in dev["groups"].values()] == [6, 6]
    assert all("by_distance" not in e for e in metrics["evals"])
    plain = load_config(GRU_2M, [*TINY_GRU_OVERRIDES, *disk_overrides(tiny_manifest)])
    off = json.loads(
        (run_one(plain, run_id="off", out=tmp_path, profile_steps=2) / "metrics.json").read_text()
    )
    assert "by_distance" not in off and off["final"] == metrics["final"]


def test_a_set_that_fails_keeps_its_error_and_a_wrong_manifest_fails_at_setup(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = f"eval.by_distance.sets=[{{name: far, manifest: {tmp_path / 'none.json'}}}]"
    arm = _arm("gru", tiny_manifest, tmp_path / "missing", ON, missing)
    with pytest.raises(FileNotFoundError):
        run_protocol(arm)
    assert arm.record["failure"]["stage"] == "setup" and "train" not in arm.record

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no table today")

    monkeypatch.setattr("hsn.eval.distance.score.recall_table", broken)
    every = "eval.by_distance.every_eval=true"
    record = run_protocol(_arm("gru", tiny_manifest, tmp_path / "broken", ON, every))
    entry = record["by_distance"]["sets"]["dev"]
    assert entry["error"] == "RuntimeError: no table today" and "groups" not in entry
    assert entry["source"]["n_docs"] == 32 and record["final"]["steps"] == 4
    # the pass itself ran, and the run's own evaluations went on without their tables
    assert entry["accuracy"] == pytest.approx(record["final"]["dev_accuracy"], abs=1e-12)
    assert [e["by_distance"] for e in record["evals"]] == [{"error": entry["error"]}] * 2
    assert "failed: RuntimeError" in render(record["by_distance"])


def test_text_run_dump_and_tables_from_the_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stories = [body.decode().replace("\x10", "\n") for body in _stories(30)]
    separator = "<|endoftext|>"
    held = ("cut" + separator + separator.join(stories[:16])).encode()
    train = (separator.join(stories[16:]) + separator + "cut").encode()
    (tmp_path / "train.txt").write_bytes(train)
    (tmp_path / "held.txt").write_bytes(held)
    spec = TextSpec(
        name="text_tiny",
        unroll=_TEXT_WINDOW,
        task="stories",
        context_id=6,
        separator=separator,
        min_bytes=8,
        max_train_tokens=10**6,
        n_dev=12,
        n_test=2,
        n_gram=16,
        train=SourceSpec(
            file="train.txt", sha256=hashlib.sha256(train).hexdigest(), drop_last=True
        ),
        heldout=SourceSpec(
            file="held.txt", sha256=hashlib.sha256(held).hexdigest(), drop_first=True
        ),
    )
    build_text(spec, tmp_path, tmp_path / "data")
    manifest = tmp_path / "data" / "manifest.json"
    frequent = FrequentWords(frozenset({b"the", b"and"}), 3, "text_tiny.frequent_words.json")
    monkeypatch.setattr(
        "hsn.eval.distance.score.find_frequent_words",
        lambda dataset: frequent if dataset == "text_tiny" else None,
    )
    cfg = load_config(
        GRU_2M,
        [
            *TINY_GRU_OVERRIDES,
            *disk_overrides(manifest, steps=2, eval_every=2),
            f"train.unroll={_TEXT_WINDOW}",
            "data.dataset=text_tiny",
            "data.tasks=[stories]",
            "data.variants=[out_of_range]",
            ON,
        ],
    )
    arm = ArmRun(
        cfg=cfg,
        family="gru",
        variant="none",
        variant_params={},
        params=protocol_params({"name": "train_eval", "eval_batch": 6, "diagnostics": False}),
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    table = record["by_distance"]["sets"]["dev"]["groups"]["stories/out_of_range"]
    assert table["kind"] == "text" and table["n_docs"] == 12
    assert table["bits_per_byte"] == pytest.approx(
        record["final"]["dev_loss"] / np.log(2), abs=1e-6
    )
    assert table["accuracy"] == pytest.approx(record["final"]["dev_accuracy"], abs=1e-12)
    assert table["names"]["n_names"] == 3 and table["names"]["rows"]["ge256"]["n"] > 0
    assert table["words"]["rows"]["256_511"]["n"] > 0

    dump_resolved(cfg, arm.run_dir / "config.resolved.yaml")
    dumps, out = tmp_path / "dumps", tmp_path / "report.json"
    run = ["--run-dir", str(arm.run_dir), "--eval-batch", "6", "--dump", str(dumps)]
    assert distance_main([*run, "--out", str(out)]) == 0
    scored = json.loads(out.read_text())["by_distance"]["sets"]["dev"]
    assert scored["groups"]["stories/out_of_range"] == _approx(table)
    dump = np.load(dumps / "dev.npz")
    assert set(dump.files) == {"nll", "hit", "offsets"} and dump["nll"].dtype == np.float32
    assert int(np.isfinite(dump["nll"]).sum()) == table["n"]
    # the tables again from the dump alone: no model
    from_dump = [
        "--from-dump",
        str(dumps / "dev.npz"),
        "--manifest",
        str(manifest),
        "--window",
        str(_TEXT_WINDOW),
        "--out",
        str(out),
    ]
    assert distance_main(from_dump) == 0
    tabled = json.loads(out.read_text())["by_distance"]["sets"]["dump"]
    assert tabled["groups"]["stories/out_of_range"] == scored["groups"]["stories/out_of_range"]
    assert tabled["loss"] == pytest.approx(scored["loss"], abs=1e-7)
    assert "names inside a sentence" in capsys.readouterr().out
