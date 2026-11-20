"""The field-dataset converters (hsn.data.fields): document format, splits, determinism, and that
their raw files build through the unchanged text importer."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hsn.data import fields
from hsn.data.dataset import ByteDataset
from hsn.data.fields import (
    MAX_BYTES,
    MIN_BYTES,
    SEPARATOR,
    FieldDataset,
    Remote,
    Source,
    Unit,
    arxiv_field,
    cap,
    chunks,
    clean,
    html_text,
    unwrap,
    word_end_stats,
    write_raw,
)
from hsn.data.text import SourceSpec, TextSpec, build_text
from hsn.model.reader import word_ends

SEP = SEPARATOR.encode()


def _docs(path: Path) -> list[bytes]:
    data = path.read_bytes()
    return data.split(b"\n" + SEP + b"\n") if data else []


def _dataset(units: fields.Units, *, heldout: int = 3000, keep: int = 10_000, group: int = 1):
    return FieldDataset(
        "fields_test", "test", "prose", 10, (), units, "a test document", heldout, keep, group
    )


def _text(i: int, words: int = 30) -> str:
    return f"Document {i}.\n\n" + " ".join(f"w{i}x{j}" for j in range(words)) + "\tend é"


def _units(n: int) -> list[Unit]:
    return [Unit(f"k{i}", f"k{i}", (_text(i),)) for i in range(n)]


def _run(ds: FieldDataset, units: list[Unit], raw: Path) -> tuple[dict, Counter[str]]:
    counts: Counter[str] = Counter()
    return write_raw(ds, iter(units), raw, counts), counts


# --------------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------------


def test_clean_reads_as_the_importer() -> None:
    assert clean("  a\r\nb\tc\x00d\x1fe \x0c") == b"a\nb\tc d e"
    assert clean("x\ud800y") == b"x?y"  # a lone surrogate
    assert clean(b"\n\n bytes \n") == b"bytes"


def test_cap_cuts_at_a_line_end_a_space_or_a_character_start() -> None:
    assert cap(b"short", 16) == b"short"
    assert cap(b"aaaaaaaaaa\nbbbbbbbbbb", 16) == b"aaaaaaaaaa"
    assert cap(b"aaaaaaaaaa bbbbbbbbbb", 16) == b"aaaaaaaaaa"
    assert cap(b"a\nbbbbbbbbbbbbbbbbbbbb", 16) == b"a\nbbbbbbbbbbbbbb"  # the line end keeps < half
    text = ("é" * 20).encode()  # two bytes a character: byte 16 starts one, byte 15 does not
    assert cap(text, 15) == ("é" * 7).encode()
    assert cap(text, 15).decode() and len(cap(text, 15)) <= 15


def test_chunks_pack_paragraphs_and_cut_long_ones() -> None:
    paragraphs = [(f"p{i} " * (i % 5 + 3)).strip().encode() for i in range(30)]
    doc = b"\n\n".join(paragraphs)
    pieces = chunks(doc, 64)
    assert all(len(piece) <= 64 for piece in pieces)
    assert b"\n\n".join(pieces) == doc  # nothing lost, the order kept
    assert len(pieces) > 1
    long = b"x" * 150
    pieces = chunks(b"head\n\n" + long + b"\n\n  \n\ntail", 64)
    assert all(len(piece) <= 64 for piece in pieces)
    assert b"".join(pieces).replace(b"\n\n", b"") == b"head" + long + b"tail"
    assert chunks(b"\n\n \n") == []


def test_unwrap_joins_hard_wrapped_lines() -> None:
    abstract = "  We study a\nthing here.\n  A second\nparagraph.\n\nThird."
    assert unwrap(abstract) == "We study a thing here.\nA second paragraph.\nThird."


def test_html_text_keeps_paragraphs_lists_and_code() -> None:
    source = (
        "<p>Use <code>zip</code> &amp; <b>map</b>:</p>\n<ul>\n<li>one</li>\n<li>two</li>\n</ul>"
        "<pre><code>for a, b in zip(x, y):\n    print(a &lt; b)\n</code></pre><p>Done.<br>Bye</p>"
    )
    assert html_text(source) == (
        "Use `zip` & map:\n\n- one\n- two\n\n"
        "    for a, b in zip(x, y):\n        print(a < b)\n\nDone.\n\nBye"
    )
    assert html_text("<pre>unclosed\ncode") == "    unclosed\n    code"


def test_arxiv_field() -> None:
    assert arxiv_field("physics.bio-ph q-bio.QM") == "bio"
    assert arxiv_field("cond-mat.soft physics.chem-ph") == "chem"
    assert arxiv_field("hep-th gr-qc") == "phys"
    assert arxiv_field("quant-ph cs.LG") == "phys"
    assert arxiv_field("cs.LG quant-ph") is None
    assert arxiv_field("math.RT") is None


# --------------------------------------------------------------------------------------------------
# Splits and files
# --------------------------------------------------------------------------------------------------


def test_write_raw_is_deterministic_and_splits_by_key(tmp_path: Path) -> None:
    ds = _dataset(lambda download, counts, wanted: iter(()))
    units = _units(200)
    first, _ = _run(ds, units, tmp_path / "a")
    second, _ = _run(ds, list(reversed(units)), tmp_path / "b")  # the source order is irrelevant
    assert first == second
    for name in ("train.txt", "heldout.txt"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    held = _docs(tmp_path / "a" / "heldout.txt")
    train = _docs(tmp_path / "a" / "train.txt")
    assert len(held) + len(train) == 200
    assert 30 < len(held) < 90  # 30 % of the keys, by the hash
    assert not set(held) & set(train)
    key = {clean(_text(i)): f"k{i}" for i in range(200)}
    for docs, split in ((held, "heldout"), (train, "train")):
        digests = [fields._digest(ds.name, key[doc]) for doc in docs]
        assert digests == sorted(digests)  # written in the order of the key's digest
        buckets = [fields._bucket(digest) for digest in digests]
        assert all((b < ds.heldout) == (split == "heldout") for b in buckets)
    assert (
        first["train"]["sha256"]
        == hashlib.sha256((tmp_path / "a/train.txt").read_bytes()).hexdigest()
    )
    assert first["train"]["documents"] == len(train)


def test_write_raw_drops_duplicates_short_separator_and_caps(tmp_path: Path) -> None:
    ds = _dataset(lambda download, counts, wanted: iter(()), heldout=0)
    units = _units(5)
    units.append(Unit("dup", "dup", (_text(1),)))  # equal to k1's document
    units.append(Unit("short", "short", ("tiny",)))
    units.append(Unit("sep", "sep", (_text(7) + SEPARATOR,)))
    units.append(Unit("long", "long", (("word " * 4000).strip(),)))
    files, counts = _run(ds, units, tmp_path)
    train = _docs(tmp_path / "train.txt")
    assert len(train) == 6
    assert counts["train_duplicates"] == 1
    assert counts["dropped_short"] == 1
    assert counts["dropped_separator"] == 1
    assert counts["capped"] == 1
    assert all(MIN_BYTES <= len(doc) <= MAX_BYTES for doc in train)
    assert files["heldout"]["documents"] == 0


def test_write_raw_drops_train_documents_equal_to_heldout_ones(tmp_path: Path) -> None:
    ds = _dataset(lambda download, counts, wanted: iter(()))
    units = _units(100)
    held_key = next(
        u.split_key
        for u in units
        if fields._bucket(fields._digest(ds.name, u.split_key)) < ds.heldout
    )
    held_text = next(u.docs[0] for u in units if u.split_key == held_key)
    copy = next(  # a key of the train bucket for a copy of the held-out document
        f"copy{i}"
        for i in range(1000)
        if fields._bucket(fields._digest(ds.name, f"copy{i}")) >= ds.heldout
    )
    units.append(Unit(copy, copy, (held_text,)))
    _, counts = _run(ds, units, tmp_path)
    assert counts["train_equal_to_heldout"] == 1
    assert clean(held_text) in _docs(tmp_path / "heldout.txt")
    assert clean(held_text) not in _docs(tmp_path / "train.txt")


def test_write_raw_groups_lines_and_keeps_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fields, "TRAIN_BYTES", 3000)
    monkeypatch.setattr(fields, "TRAIN_MARGIN", 0)
    ds = _dataset(lambda download, counts, wanted: iter(()), heldout=2000, group=4)
    molecules = [f"CC(=O)N{'C' * (i % 7)}c1ccccc1O{i}" for i in range(400)]
    units = [Unit(m, m, (m,)) for m in molecules + molecules[:10]]  # ten repeated
    files, counts = _run(ds, units, tmp_path)
    train = _docs(tmp_path / "train.txt")
    held = _docs(tmp_path / "heldout.txt")
    assert all(len(doc.split(b"\n")) == 4 for doc in held[:-1] + train[:-1])
    assert sum(len(doc) + 2 for doc in train) <= 3000
    assert files["train"]["positions"] == sum(len(doc) + 2 for doc in train)
    lines = [line for doc in held + train for line in doc.split(b"\n")]
    assert len(lines) == len(set(lines))  # a molecule once, in one split
    assert counts["train_duplicates"] + counts["heldout_duplicates"] == 10


def test_raw_files_build_through_the_text_importer(tmp_path: Path) -> None:
    ds = _dataset(lambda download, counts, wanted: iter(()), heldout=4000)
    files, _ = _run(ds, _units(120), tmp_path / "raw")
    spec = TextSpec.model_validate(
        {
            "name": "fields_test",
            "unroll": 64,
            "task": "fields_test",
            "context_id": 10,
            "separator": SEPARATOR,
            "min_bytes": MIN_BYTES,
            "max_train_tokens": 10**6,
            "n_dev": 20,
            "n_test": 10,
            "n_gram": 16,
            "train": SourceSpec(file="train.txt", sha256=files["train"]["sha256"]),
            "heldout": SourceSpec(file="heldout.txt", sha256=files["heldout"]["sha256"]),
        }
    )
    manifest, counts = build_text(spec, tmp_path / "raw", tmp_path / "out")
    assert counts["train_dropped"] == 0 and counts["heldout_dropped"] == 0  # nothing re-filtered
    assert counts["train_documents"] == files["train"]["documents"]
    assert counts["train_tokens"] == files["train"]["positions"]
    assert all(pair.exact_duplicates == 0 for pair in manifest.leakage)
    dev = ByteDataset.load(
        tmp_path / "out" / "manifest.json", "dev", ["fields_test"], ["out_of_range"]
    )
    assert len(dev) == 20


# --------------------------------------------------------------------------------------------------
# Sources (tiny stand-ins for the downloaded files)
# --------------------------------------------------------------------------------------------------


def _source(source: Source, *names: str) -> Source:
    remotes = tuple(Remote(name, f"https://example.invalid/{name}", 0) for name in names)
    return Source(source.key, source.title, source.repo, source.revision, source.license, remotes)


def _collect(
    units: fields.Units, download: Path, wanted: fields.Wanted = lambda key: True
) -> tuple[list[Unit], Counter[str]]:
    counts: Counter[str] = Counter()
    return list(units(download, counts, wanted)), counts


def test_uniref_units_skip_the_header_and_the_cut_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fields, "UNIREF50", _source(fields.UNIREF50, "a.csv"))
    folder = tmp_path / fields.UNIREF50.key
    folder.mkdir()
    (folder / "a.csv").write_bytes(
        b"id,name,text\n1,UniRef50_A,MKV\n2,UniRef50_B,MAL\n3,UniRef50_C,MS"
    )
    units, counts = _collect(fields._uniref_units, tmp_path)
    assert [(u.split_key, u.docs) for u in units] == [
        ("UniRef50_A", (b"MKV",)),
        ("UniRef50_B", (b"MAL",)),
    ]
    assert counts["sequences"] == 2
    units, _ = _collect(fields._uniref_units, tmp_path, lambda key: key != "UniRef50_A")
    assert [u.split_key for u in units] == ["UniRef50_B"]


def test_stackoverflow_units_group_the_answers_of_a_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fields,
        "STACKOVERFLOW_PYTHON",
        _source(fields.STACKOVERFLOW_PYTHON, "a.parquet", "b.parquet"),
    )
    folder = tmp_path / fields.STACKOVERFLOW_PYTHON.key
    folder.mkdir()
    rows = {
        "question_id": [1, 1, 2],
        "title": ["How?", "How?", "Why?"],
        "question_body": ["<p>Q1</p>", "<p>Q1</p>", "<p>Q2</p>"],
        "answer_id": [10, 11, 12],
        "answer_body": ["<p>low</p>", "<p>high</p>", "<p>A2</p>"],
        "answer_score": [1, 5, 0],
    }
    pq.write_table(pa.table(rows), folder / "a.parquet")
    pq.write_table(pa.table({k: [v[-1]] for k, v in rows.items()}), folder / "b.parquet")  # 2 again
    units, counts = _collect(fields._stackoverflow_units, tmp_path)
    assert units[0].docs == ("How?\n\nQ1\n\nAnswer:\nhigh\n\nAnswer:\nlow",)
    assert units[1].docs == ("Why?\n\nQ2\n\nAnswer:\nA2",)  # the rows of file b continue it
    assert len(units) == 2 and counts["questions"] == 2


def test_codeparrot_units_keep_permissive_files_split_by_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fields, "CODEPARROT", _source(fields.CODEPARROT, "a.json.gz"))
    folder = tmp_path / fields.CODEPARROT.key
    folder.mkdir()
    records = [
        {
            "repo_name": "o/r",
            "path": "a.py",
            "content": "x = 1\n",
            "license": "mit",
            "autogenerated": False,
        },
        {
            "repo_name": "o/r",
            "path": "b.py",
            "content": "y = 2\n",
            "license": "gpl-3.0",
            "autogenerated": False,
        },
        {
            "repo_name": "o/s",
            "path": "c.py",
            "content": "z = 3\n",
            "license": "apache-2.0",
            "autogenerated": True,
        },
    ]
    with gzip.open(folder / "a.json.gz", "wt") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)
    units, counts = _collect(fields._codeparrot_units, tmp_path)
    assert [(u.split_key, u.order_key, u.docs) for u in units] == [
        ("o/r", "o/r/a.py", ("x = 1\n",))
    ]
    assert counts["files"] == 3 and counts["autogenerated"] == 1


def test_arxiv_and_pubmed_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fields, "ARXIV_METADATA", _source(fields.ARXIV_METADATA, "m.parquet"))
    monkeypatch.setattr(fields, "PUBMED_SUBSET", _source(fields.PUBMED_SUBSET, "p.jsonl"))
    (tmp_path / fields.ARXIV_METADATA.key).mkdir()
    table = {
        "id": ["1", "2", "hep-th/9901001"],
        "categories": ["q-bio.NC", "cs.LG", "hep-th"],
        "title": ["A\n  title", "B", "C"],
        "abstract": ["  Bio\nwrapped.", "ml", "  Strings."],
    }
    pq.write_table(pa.table(table), tmp_path / fields.ARXIV_METADATA.key / "m.parquet")
    units, _ = _collect(fields._arxiv_units("phys"), tmp_path)
    assert [(u.split_key, u.docs) for u in units] == [("hep-th/9901001", ("C\n\nStrings.",))]
    units, _ = _collect(fields._arxiv_units("bio"), tmp_path)
    assert units[0].docs == ("A title\n\nBio wrapped.",)
    (tmp_path / fields.PUBMED_SUBSET.key).mkdir()
    lines = [
        {"title": "T  one.", "abstract": "Line  one.\nLine two.", "PMID": 5},
        {"title": "No abstract.", "abstract": "", "PMID": 6},
    ]
    (tmp_path / fields.PUBMED_SUBSET.key / "p.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in lines)
    )
    units, counts = _collect(fields._pubmed_units, tmp_path)
    assert [(u.split_key, u.docs) for u in units] == [("5", ("T one.\n\nLine one.\nLine two.",))]
    assert counts["records_without_abstract"] == 1


def test_chemrxiv_and_pubchem_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fields, "CHEMRXIV", _source(fields.CHEMRXIV, "c.parquet"))
    monkeypatch.setattr(fields, "PUBCHEM_10M", _source(fields.PUBCHEM_10M, "s.parquet"))
    (tmp_path / fields.CHEMRXIV.key).mkdir()
    body = "\n\n".join(f"Paragraph {i}. " + "text " * 200 for i in range(40))
    table = {"doi": ["10.1/a", "10.1/b"], "license": ["CC BY 4.0", "unknown"], "text": [body, "x"]}
    pq.write_table(pa.table(table), tmp_path / fields.CHEMRXIV.key / "c.parquet")
    units, counts = _collect(fields._chemrxiv_units, tmp_path)
    assert len(units) == 1 and counts["papers_not_cc"] == 1
    assert len(units[0].docs) > 1 and all(len(piece) <= MAX_BYTES for piece in units[0].docs)
    (tmp_path / fields.PUBCHEM_10M.key).mkdir()
    pq.write_table(
        pa.table({"smiles": ["CCO", "c1ccccc1"]}), tmp_path / fields.PUBCHEM_10M.key / "s.parquet"
    )
    units, _ = _collect(fields._pubchem_units, tmp_path)
    assert [u.docs for u in units] == [("CCO",), ("c1ccccc1",)]


@pytest.mark.skipif(shutil.which("zstd") is None, reason="needs the zstd command")
def test_latex_units_keep_open_physics_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fields, "ARXIV_METADATA", _source(fields.ARXIV_METADATA, "m.parquet"))
    monkeypatch.setattr(fields, "PROOF_PILE_ARXIV", _source(fields.PROOF_PILE_ARXIV, "a.jsonl.zst"))
    (tmp_path / fields.ARXIV_METADATA.key).mkdir()
    cc = "http://creativecommons.org/licenses/by/4.0/"
    table = {
        "id": ["1", "2", "hep-th/9901001", "4"],
        "categories": ["hep-th", "hep-th", "hep-th", "math.AG"],
        "license": [cc, "http://arxiv.org/licenses/nonexclusive-distrib/1.0/", cc, cc],
    }
    pq.write_table(pa.table(table), tmp_path / fields.ARXIV_METADATA.key / "m.parquet")
    folder = tmp_path / fields.PROOF_PILE_ARXIV.key
    folder.mkdir()
    records = [
        {"text": f"\\section{{S{i}}}\n$x_{i}$", "meta": {"arxiv_id": i}}
        for i in ("1", "2", "hep-th9901001", "4")
    ]
    (folder / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    subprocess.run(
        ["zstd", "-q", str(folder / "a.jsonl"), "-o", str(folder / "a.jsonl.zst")], check=True
    )
    units, counts = _collect(fields._latex_units, tmp_path)
    assert [u.split_key for u in units] == ["1", "hep-th9901001"]
    assert units[0].docs == (b"\\section{S1}\n$x_1$",)
    assert counts["papers"] == 4 and counts["papers_open_physics"] == 2


def test_fetch_checks_size_and_hash(tmp_path: Path) -> None:
    payload = b"field data " * 100
    (tmp_path / "remote.bin").write_bytes(payload)
    url = (tmp_path / "remote.bin").as_uri()
    good = Source(
        "k",
        "t",
        "r",
        "v",
        "l",
        (Remote("f.bin", url, len(payload), hashlib.sha256(payload).hexdigest()),),
    )
    fields.fetch(good, tmp_path / "download", log=lambda line: None)
    assert (tmp_path / "download" / "k" / "f.bin").read_bytes() == payload
    fields.fetch(good, tmp_path / "download", log=lambda line: None)  # present: skipped
    bad = Source("k2", "t", "r", "v", "l", (Remote("f.bin", url, len(payload), "0" * 64),))
    with pytest.raises(ValueError, match="sha256"):
        fields.fetch(bad, tmp_path / "download", log=lambda line: None)
    assert not (tmp_path / "download" / "k2" / "f.bin").exists()


# --------------------------------------------------------------------------------------------------
# Word ends
# --------------------------------------------------------------------------------------------------


def test_word_end_stats_follow_the_reader_rule() -> None:
    rng = np.random.default_rng(0)
    alphabet = np.frombuffer(b"ab Z9.,\x10\xc3\xa9-", dtype=np.uint8)
    docs = [rng.choice(alphabet, size=int(rng.integers(5, 40))) for _ in range(20)]
    tokens = np.concatenate([np.concatenate([[1], doc, [2]]) for doc in docs]).astype(np.uint8)
    stats = word_end_stats(tokens, len(docs))
    reset = torch.from_numpy(tokens == 1)[None]
    ends = word_ends(torch.from_numpy(tokens)[None], reset, torch.zeros(1, dtype=torch.bool))
    assert stats["word_ends"] == int(ends.sum())
    assert stats["events_per_window"] == pytest.approx((int(ends.sum()) + 20) * 256 / tokens.size)
    proteins = np.frombuffer(b"\x01MKVLAAG\x02\x01MSTR\x02", dtype=np.uint8)
    assert word_end_stats(proteins, 2)["word_ends"] == 2  # one word a document, ended by EOS


def test_near_copies_of_heldout_documents_leave_train(tmp_path: Path) -> None:
    ds = _dataset(lambda download, counts, wanted: iter(()))
    units = _units(100)
    held = next(u for u in units if fields._bucket(fields._digest(ds.name, u.split_key)) < 3000)
    copy = next(
        f"fork{i}"
        for i in range(1000)
        if fields._bucket(fields._digest(ds.name, f"fork{i}")) >= ds.heldout
    )
    text = str(held.docs[0])
    units.append(Unit(copy, copy, (text.replace("end", "fin"),)))  # one word changed
    _, counts = _run(ds, units, tmp_path / "plain")
    assert counts["train_near_heldout"] == 0
    assert clean(text.replace("end", "fin")) in _docs(tmp_path / "plain" / "train.txt")
    near = dataclasses.replace(ds, near=0.5)
    _, counts = _run(near, units, tmp_path / "near")
    assert counts["train_near_heldout"] == 1
    train = _docs(tmp_path / "near" / "train.txt")
    assert clean(text.replace("end", "fin")) not in train
    assert len(train) == len(_docs(tmp_path / "plain" / "train.txt")) - 1


def test_gram_hashes_and_near_share() -> None:
    doc = bytes(range(32, 120))
    hashes = fields.gram_hashes(doc, 8)
    assert hashes.shape == (len(doc) - 7,) and len(set(hashes.tolist())) == hashes.size
    assert fields.gram_hashes(b"short", 8).size == 0
    index = np.unique(fields.gram_hashes(doc[:50]))
    share = fields.near_share(doc, index)  # windows of 32 bytes: 19 of 57 inside the first 50
    assert share == pytest.approx(19 / 57)
    assert fields.near_share(b"x" * 10, index) == 0.0
