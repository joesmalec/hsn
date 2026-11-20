"""Field datasets (19-20 Nov 2026): four fields, each as prose and in its native format.

Eight byte-level corpora to compare models within a speed class on different kinds of material
(docs/field_datasets.md). Each is an :mod:`hsn.data.text` dataset: this module fetches pinned
public sources and converts them into the two raw files a text spec names, ``train.txt`` and
``heldout.txt`` (documents joined by :data:`SEPARATOR`); ``python -m hsn.data.text build`` then
writes the shards and the manifest, unchanged.

======================  ============================================  ===========================
dataset                 source (``SOURCES``)                          document
======================  ============================================  ===========================
``fields_bio_prose``    PubMed abstracts (a random subset)            title and abstract
``fields_chem_prose``   ChemRxiv preprints (full text)                a 16 KiB piece of a paper
``fields_phys_prose``   arXiv metadata, physics papers                title and abstract
``fields_code_prose``   Stack Overflow, Python questions              question and its answers
``fields_bio_native``   UniRef50 protein sequences                    one sequence
``fields_chem_native``  PubChem-10M SMILES                            32 molecules, one per line
``fields_phys_native``  arXiv LaTeX bodies (proof-pile-2), physics    a 16 KiB piece of a paper
``fields_code_native``  CodeParrot-clean Python files                 one source file
======================  ============================================  ===========================

**Fetch** (``fetch``). Every remote file is a URL that pins a revision (a Hugging Face commit)
and, where the host publishes one, a SHA-256 that the download must match; a byte-range download
(``first_bytes``) is the head of a pinned file. Files land in ``<root>/download/<source>/``.

**Convert** (``convert``). A source yields *units*: a key that decides the split (the arXiv id, the
PMID, the question, the repository, the UniRef50 cluster, the molecule), a key that decides the
order, and the unit's documents. With ``d = blake2b("<dataset>/<key>")`` and ``b = d mod 10,000``,
a unit is held out when ``b < heldout`` and a train unit when ``heldout <= b < keep`` (``keep``
thins a source far larger than the train budget before its text is held in memory); both sets are
written in the order of the order key's digest. So the held-out documents are a different set of
papers, questions, repositories or clusters, and the same source revision gives the same bytes.
Every document is cleaned as the importer would read it (carriage returns removed, other bytes
below 32 but newline and tab made spaces, stripped), capped at :data:`MAX_BYTES` (cut at a line
end, else a space; a LaTeX or ChemRxiv paper is cut into pieces at paragraph ends instead), and
kept from :data:`MIN_BYTES`. Exact duplicates are dropped (the first in order kept), every train
document equal to a held-out one is dropped, and so is a document holding the separator. Train
documents are written up to :data:`TRAIN_BYTES` plus a margin (the importer's ``max_train_tokens``
makes the cut), the held-out pool whole (the importer takes its first ``n_dev`` and next ``n_test``
documents). ``SOURCE.json`` beside the raw files records the sources, revisions, licences, counts
and the raw files' SHA-256 (the spec's pins).

**Word ends** (``stats``): the bytes per word end of a built dataset by the rule of
:func:`hsn.model.reader.word_ends` (a byte that is not an ASCII letter or digit, right after one
that is; bytes from 128 count as letters), and the events per 256-byte window of
``model.write_events=word`` (word ends plus document starts).

    python -m hsn.data.fields fetch all --root <dir>
    python -m hsn.data.fields convert fields_bio_prose --root <dir>
    python -m hsn.data.text build --spec configs/data/fields_bio_prose.yaml \\
        --raw <dir>/raw/fields_bio_prose --out data/fields_bio_prose \\
        --golden configs/data/fields_bio_prose.manifest.json
    python -m hsn.data.fields stats data/fields_bio_prose/manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "DATASETS",
    "MAX_BYTES",
    "MIN_BYTES",
    "SEPARATOR",
    "SOURCES",
    "TRAIN_BYTES",
    "FieldDataset",
    "Remote",
    "Source",
    "Unit",
    "arxiv_field",
    "cap",
    "chunks",
    "clean",
    "html_text",
    "unwrap",
    "word_end_stats",
    "write_raw",
]

SEPARATOR = "<|endoftext|>"
MAX_BYTES = 16 * 1024  # documents are capped here (a LaTeX paper is cut into pieces of this)
MIN_BYTES = 64  # shorter documents are dropped (the specs' min_bytes)
TRAIN_BYTES = 200_000_000  # the specs' max_train_tokens
TRAIN_MARGIN = 2_000_000  # written past TRAIN_BYTES, so the importer's own cut decides
BUCKETS = 10_000
USER_AGENT = "hsn-data-fields/1.0"
_HF = "https://huggingface.co/datasets"
_SEP = SEPARATOR.encode()

# --------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Remote:
    """One file to download: ``url`` pins a revision; ``sha256`` (when the host gives one) is
    checked; ``first_bytes`` downloads only the head of the file (a byte range)."""

    file: str
    url: str
    size: int
    sha256: str | None = None
    first_bytes: int | None = None


@dataclass(frozen=True)
class Source:
    """A public source: its download folder (``key``), where it lives and under what licence."""

    key: str
    title: str
    repo: str
    revision: str
    license: str
    files: tuple[Remote, ...]

    @property
    def homepage(self) -> str:
        return f"{_HF}/{self.repo}"


def _hf(repo: str, revision: str, path: str, size: int, sha256: str | None = None) -> Remote:
    return Remote(path.rsplit("/", 1)[-1], f"{_HF}/{repo}/resolve/{revision}/{path}", size, sha256)


_ARXIV_META_REPO = "librarian-bots/arxiv-metadata-snapshot"
_ARXIV_META_REV = "c428226b689a0e17db2cd55265133b9eca15d97f"
_ARXIV_META_FILES = (
    (284033689, "2b0cd657765299c06a53f40d18d9a6be707b7bb5ac67cc99afea12c1eb1f6b2d"),
    (280398761, "6a9f1721f2ff83a9554966ea5fb3b6082e06a7b6a0e9a60c65c9b45a49b40f2e"),
    (275725802, "5ceb8990b71f0026e6a99e06ffdaaa6fafb1283a811b91e113638ee17b9d6724"),
    (269293149, "1326619f9081a0ab3e26ce0a47368fe719eca3769e3d0c1fd937c5c5a72a8bbf"),
    (262530975, "c0643e0a72b30a834386abf2442abbea9ce654082d08c2e058baf496c560ffbf"),
    (254418986, "22ccd6bb2b32a3f6bb597b064930b9d0ab87a2e500ae4d30033f4fba6806c68a"),
    (243101946, "35a9c6ef73f51a0b2c57d131297bbfd627df2a2e63ed2052a51fa735413e9d7b"),
    (233806666, "b831a8bd9ca4ff409e12759b0f0529e1304bd8b0be1a97aea1c6ef3ec8c3a694"),
    (222733408, "a3fa8db6eac9fb17dfdfdf65be4fae0c9ed5e65480683a98959e845099180984"),
    (200637171, "f6b76c2cfa337ae252b5aec4e4d1e06a4b9e64fff8890b8aa54db61b60cd192a"),
    (174286181, "e3b1b851f16bb1f579c37f16d2b14f8d2bbd1d1930e7fabf0b4e9639c12fc745"),
)
ARXIV_METADATA = Source(
    key="arxiv-metadata",
    title="arXiv metadata snapshot (the Kaggle snapshot, mirrored weekly)",
    repo=_ARXIV_META_REPO,
    revision=_ARXIV_META_REV,
    license="CC0-1.0 (arXiv metadata, abstracts included)",
    files=tuple(
        _hf(_ARXIV_META_REPO, _ARXIV_META_REV, f"data/train-{i:05d}-of-00011.parquet", n, h)
        for i, (n, h) in enumerate(_ARXIV_META_FILES)
    ),
)

_PUBMED_REPO = "slinusc/PubMedAbstractsSubset"
_PUBMED_REV = "111f2d4a15b4335a4ddf6844264e6caf2d0a614a"
_PUBMED_FILES = (
    (0, 111049374, "a0060730ddb41870a482d378baba8dfef8fdbbc788ef9658e3fdae6f13e9c9b3"),
    (8, 142588703, "a24086e3c952e738e526e841290f601391e331f2d0962d4f940ea7b564fa9dcc"),
    (16, 153961548, "7e2ec7c196c0f3c652fd8c8fa6bf0250a93447c8a9910f7aa1a1b4832385df2c"),
)
PUBMED_SUBSET = Source(
    key="pubmed-subset",
    title="PubMed titles and abstracts, a random subset (2.39M) of the NLM records; 3 of 24 files",
    repo=_PUBMED_REPO,
    revision=_PUBMED_REV,
    license=(
        "CC0-1.0 (card); PubMed records of the NLM (abstract copyright with the publishers, "
        "open to text mining under the NLM terms)"
    ),
    files=tuple(
        _hf(_PUBMED_REPO, _PUBMED_REV, f"pubmed_chunk_{i}.jsonl", n, h) for i, n, h in _PUBMED_FILES
    ),
)

_CHEMRXIV_REPO = "jablonkagroup/chemrxiv_processed"
_CHEMRXIV_REV = "341cc642b3db7de4a6966d73d0d860e5bbfb3240"
CHEMRXIV = Source(
    key="chemrxiv",
    title=(
        "ChemRxiv preprints as text (ChemNLP, Jablonka group; PDFs converted to Markdown with "
        "LaTeX math), the train split"
    ),
    repo=_CHEMRXIV_REPO,
    revision=_CHEMRXIV_REV,
    license="per paper: CC BY 4.0, CC BY-NC 4.0 or CC BY-NC-ND 4.0 (every paper of the set)",
    files=(
        _hf(
            _CHEMRXIV_REPO,
            _CHEMRXIV_REV,
            "data/train-00000-of-00002.parquet",
            172636306,
            "17232c06b5fe1205e874b39936a662c2f3e27173f30b930a45fd4d6c6d551cda",
        ),
        _hf(
            _CHEMRXIV_REPO,
            _CHEMRXIV_REV,
            "data/train-00001-of-00002.parquet",
            172619762,
            "402fec0e29c784a502d8e943950e73d3cdd7a84990fe0b04fe1eb48bdc215bc3",
        ),
    ),
)

_SO_REPO = "koutch/stackoverflow_python"
_SO_REV = "ee6f39553b6fe4424f0a52b61a8de2a893390744"
_SO_FILES = (
    (
        "3d48b7eced91a0b0",
        121435120,
        "014e8293b8aded017138d00c5a578050ce02528132b32b9cdf1eb3e0d1386777",
    ),
    (
        "56eee9940d3dd0b1",
        153843229,
        "75f3ae908220a330898791864e29425dfc34550b1f9325c376d98743b1a69189",
    ),
    (
        "2683a3aec26c96f7",
        175233906,
        "d84be4309a3758e557569f15324d43f0ef6561a758a24f46dd9998427c3280de",
    ),
    (
        "dfcfc12a49338d82",
        185812032,
        "5a09fa4b293076059f7d38834968de9a308e411387d935438fd4b82dcd0d57fa",
    ),
    (
        "69e77510bada753e",
        193223699,
        "2c92c0870c185c7ae12243dd33922aa17a7dfb88eefcfd1fd41ea1f03be4fdf9",
    ),
)
STACKOVERFLOW_PYTHON = Source(
    key="stackoverflow-python",
    title="Stack Overflow Python questions and answers, Aug 2008 - Oct 2016 (Kaggle set)",
    repo=_SO_REPO,
    revision=_SO_REV,
    license="CC BY-SA 3.0 (Stack Overflow user contributions)",
    files=tuple(
        _hf(_SO_REPO, _SO_REV, f"data/train-{i:05d}-of-00005-{tag}.parquet", n, h)
        for i, (tag, n, h) in enumerate(_SO_FILES)
    ),
)

_UNIREF_REPO = "agemagician/uniref50"
_UNIREF_REV = "cc382ac670d6f715e90541f7a60eb3a66381a70c"
# sizes of data/train/train_00.csv ... train_95.csv; the files hold the clusters by sequence
# length, longest first, so the converter reads the head (UNIREF_PERCENT %) of every file
_UNIREF_SIZES = (
    1396605166, 782360563, 647565989, 568300385, 513852921, 473222613, 440557364, 413653695,
    390835886, 371473800, 354545579, 339515330, 326101375, 314098004, 303254139, 293311137,
    284584734, 276420566, 268539212, 261316280, 254427071, 247792744, 241491051, 235413599,
    229675187, 224341998, 219170006, 214108905, 209241773, 204533983, 200159468, 195766728,
    191459541, 187294947, 183184765, 179363714, 175567389, 171970014, 168377413, 164836006,
    161417120, 158101551, 154770882, 151622481, 148536801, 145509412, 142560390, 139684477,
    136807596, 134038815, 131362880, 128738146, 126147723, 123645824, 121212902, 118821860,
    116463998, 114216427, 111999648, 109842826, 107710105, 105630427, 103539961, 101486123,
    99516325, 97482698, 95547318, 93624068, 91702662, 89851368, 87982972, 86146398,
    84369008, 82610321, 80942205, 79270116, 77439397, 75657139, 73890512, 72190221,
    70428680, 68848798, 67309240, 65599609, 64171350, 62559843, 61117699, 59559361,
    58122636, 56612623, 54884812, 52799012, 50590769, 47195756, 42645450, 35719924,
)  # fmt: skip
UNIREF_PERCENT = 2
UNIREF50 = Source(
    key="uniref50",
    title="UniRef50 protein clusters (UniProt), the head of each length-sorted file",
    repo=_UNIREF_REPO,
    revision=_UNIREF_REV,
    license="CC BY 4.0 (UniProt)",
    files=tuple(
        Remote(
            f"train_{i:02d}.head.csv",
            f"{_HF}/{_UNIREF_REPO}/resolve/{_UNIREF_REV}/data/train/train_{i:02d}.csv",
            size,
            None,
            size * UNIREF_PERCENT // 100,
        )
        for i, size in enumerate(_UNIREF_SIZES)
    ),
)

_PUBCHEM_REPO = "sagawa/pubchem-10m-canonicalized"
_PUBCHEM_REV = "f83219601635a0a80fc99c13a9ca37f99ef34f0a"
PUBCHEM_10M = Source(
    key="pubchem-10m",
    title="PubChem-10M (DeepChem / ChemBERTa), canonicalised with RDKit",
    repo=_PUBCHEM_REPO,
    revision=_PUBCHEM_REV,
    license="Apache-2.0 (card); PubChem records are public domain",
    files=(
        _hf(
            _PUBCHEM_REPO,
            _PUBCHEM_REV,
            "data/train-00000-of-00001-e9b227f8c7259c8b.parquet",
            236668162,
            "574264d71e366921b7f538612dac7d392870efaeb5cf39e0945f6894b25c9214",
        ),
    ),
)

_PP2_REPO = "EleutherAI/proof-pile-2"
_PP2_REV = "901a9273a770e9d4138c5ddd91802f9c5c6cdc4b"
_PP2_FILES = (
    (5, 260686633, "d4c3a8eafce124edbd68ada52b78ffc54b988e092cd7c3bfbe0c35f8f17b4a26"),
    (15, 260622653, "6c1f0103b0df00a76af1fede1d4f9d8769ae5e91d52a730872f54ef41523015f"),
    (25, 243131289, "03cb597a3b7ec27ddc5e79f2f13bc707e5a390dad032f49b3232b86f736ba7df"),
    (35, 267147907, "7ec7654ea604872b4b86fb93c2df25e0006e7c1772667b03fc9e90bd2e7d91ec"),
    (45, 258935104, "34fc57f96acfe7d1f1a35371ce4fc9e8bd06daa9240bcaae92d6d460b0589bcf"),
    (55, 273188154, "64eb34e44289426559b0d3905d1cbf28ad6b6da7241fdb825616951b1e33f8d0"),
    (65, 251793786, "657a671a8e64b0bbdb726593b3430c429a892bc8de3ddaeaef310a35a588a431"),
    (75, 246003971, "c7d37397fd4b1035c0b4e8cba536f49c50f23fb7e97c804a29b3186be1026cca"),
    (85, 287811438, "a3a7d50f856ddaa4026df1f9b8519b1cf764e4be385a9f1f2bd5ae82426990fc"),
    (95, 231627345, "72cc24f1b6328a38f861ff3aa6c98f6d1cce259d58b842f4bb5285346d111e57"),
)
PROOF_PILE_ARXIV = Source(
    key="proof-pile-2-arxiv",
    title="arXiv LaTeX bodies of proof-pile-2 (the RedPajama arXiv set), every tenth train file",
    repo=_PP2_REPO,
    revision=_PP2_REV,
    license=(
        "per paper; only papers whose arXiv licence is Creative Commons (BY, BY-SA, BY-NC, "
        "BY-NC-SA, BY-NC-ND) or public domain (CC0) are kept"
    ),
    files=tuple(
        _hf(_PP2_REPO, _PP2_REV, f"arxiv/train/arXiv_{i:03d}.jsonl.zst", n, h)
        for i, n, h in _PP2_FILES
    ),
)

_CP_REPO = "codeparrot/codeparrot-clean"
_CP_REV = "35a59fb025bc0a102f7d96eac09d145b896d487b"
CODEPARROT = Source(
    key="codeparrot-clean",
    title="CodeParrot-clean: Python files from GitHub, exact duplicates removed",
    repo=_CP_REPO,
    revision=_CP_REV,
    license=(
        "per file; only MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense and CC0 "
        "files are kept"
    ),
    files=(
        _hf(
            _CP_REPO,
            _CP_REV,
            "file-000000000001.json.gz",
            246127841,
            "3726a0239b5cb7d0ef3ea36886c533d0becc7404217763015559edb546d53c94",
        ),
    ),
)

SOURCES: dict[str, Source] = {
    source.key: source
    for source in (
        ARXIV_METADATA,
        PUBMED_SUBSET,
        CHEMRXIV,
        STACKOVERFLOW_PYTHON,
        UNIREF50,
        PUBCHEM_10M,
        PROOF_PILE_ARXIV,
        CODEPARROT,
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source: Source, download: Path, *, log: Callable[[str], None] = print) -> None:
    """Download the files of ``source`` into ``download / source.key`` (skipping files already
    there with the right size and hash). Anonymous, with a generic client name."""
    folder = download / source.key
    folder.mkdir(parents=True, exist_ok=True)
    for remote in source.files:
        path = folder / remote.file
        want = remote.first_bytes if remote.first_bytes is not None else remote.size
        if path.exists() and path.stat().st_size == want:
            if remote.sha256 is None or _sha256(path) == remote.sha256:
                continue
        request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
        if remote.first_bytes is not None:
            request.add_header("Range", f"bytes=0-{remote.first_bytes - 1}")
        part = path.with_name(path.name + ".part")
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 22)
        got = part.stat().st_size
        if got != want:
            raise ValueError(f"{remote.url}: {got} bytes, expected {want}")
        if remote.sha256 is not None and _sha256(part) != remote.sha256:
            raise ValueError(f"{remote.url}: sha256 differs from {remote.sha256}")
        part.rename(path)
        log(f"  {source.key}/{remote.file}: {got} bytes in {time.perf_counter() - started:.0f} s")


# --------------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------------

_CONTROL = bytes(b if b in (9, 10) or b >= 32 else 32 for b in range(256))


def clean(text: str | bytes) -> bytes:
    """The bytes of a document as the importer reads them: UTF-8 (a lone surrogate becomes
    ``?``), carriage returns removed, every other byte below 32 but newline and tab a space,
    white space stripped at both ends."""
    data = text if isinstance(text, bytes) else text.encode("utf-8", "replace")
    return data.replace(b"\r", b"").translate(_CONTROL).strip()


def cap(doc: bytes, limit: int = MAX_BYTES) -> bytes:
    """``doc`` cut to at most ``limit`` bytes: at the last line end that keeps half of it, else
    at the last space that does, else at ``limit`` (moved back to a UTF-8 character start)."""
    if len(doc) <= limit:
        return doc
    cut = doc.rfind(b"\n", 0, limit + 1)
    if cut < limit // 2:
        cut = doc.rfind(b" ", 0, limit + 1)
    if cut < limit // 2:
        cut = limit
        while cut > 0 and doc[cut] & 0xC0 == 0x80:
            cut -= 1
    return doc[:cut].rstrip()


def chunks(doc: bytes, limit: int = MAX_BYTES) -> list[bytes]:
    """``doc`` cut into pieces of at most ``limit`` bytes at paragraph ends (blank lines): the
    paragraphs are packed in order, and a paragraph longer than ``limit`` is cut by :func:`cap`
    repeatedly."""
    pieces: list[bytes] = []
    current = b""
    for paragraph in re.split(rb"\n[ \t]*\n", doc):
        paragraph = paragraph.strip(b"\n")
        if not paragraph.strip():
            continue
        if current and len(current) + 2 + len(paragraph) <= limit:
            current += b"\n\n" + paragraph
            continue
        if current:
            pieces.append(current)
        while len(paragraph) > limit:
            head = cap(paragraph, limit)
            pieces.append(head)
            paragraph = paragraph[len(head) :].strip()
        current = paragraph
    if current:
        pieces.append(current)
    return pieces


def unwrap(text: str) -> str:
    """Hard-wrapped text (an arXiv abstract) as paragraphs on one line each: a blank
    line or a line that starts with white space begins a paragraph; runs of white space inside
    a paragraph become one space."""
    paragraphs = re.split(r"\n[ \t]*\n|\n(?=[ \t])", text.strip())
    return "\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


class _HTMLText(HTMLParser):
    """Stack Overflow HTML as plain text (see :func:`html_text`)."""

    _BLOCKS = frozenset(
        ("p", "div", "ul", "ol", "blockquote", "table", "tr", "hr", "dl", "dt", "dd")
        + ("h1", "h2", "h3", "h4", "h5", "h6")
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []  # (kind, text): kind "text", "item" or "code"
        self.current: list[str] = []
        self.kind = "text"
        self.pre = 0
        self.code: list[str] = []

    def _flush(self) -> None:
        text = " ".join("".join(self.current).split())
        if text:
            self.blocks.append((self.kind, text))
        self.current = []
        self.kind = "text"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            if not self.pre:
                self._flush()
                self.code = []
            self.pre += 1
        elif self.pre:
            return
        elif tag in self._BLOCKS or tag == "br":
            self._flush()
        elif tag == "li":
            self._flush()
            self.kind = "item"
            self.current.append("- ")
        elif tag == "code":
            self.current.append("`")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self.pre:
            self.pre -= 1
            if not self.pre:
                self._code()
        elif self.pre:
            return
        elif tag in self._BLOCKS or tag == "li":
            self._flush()
        elif tag == "code":
            self.current.append("`")

    def handle_data(self, data: str) -> None:
        (self.code if self.pre else self.current).append(data)

    def _code(self) -> None:
        lines = "".join(self.code).strip("\n").split("\n")
        code = "\n".join("    " + line if line.strip() else "" for line in lines)
        if code.strip():
            self.blocks.append(("code", code))
        self.code = []

    def text(self, source: str) -> str:
        self.feed(source)
        self.close()
        if self.pre:  # a <pre> left open
            self.pre = 0
            self._code()
        self._flush()
        out: list[str] = []
        previous = ""
        for kind, block in self.blocks:
            if out:
                out.append("\n" if kind == previous == "item" else "\n\n")
            out.append(block)
            previous = kind
        return "".join(out)


def html_text(source: str) -> str:
    """Stack Overflow post HTML as plain text close to the post's Markdown: paragraphs apart by
    a blank line, list items as ``- `` lines, code blocks (``<pre>``) verbatim and indented by
    four spaces, inline code in backticks, every other tag dropped, entities decoded."""
    return _HTMLText().text(source)


_PHYSICS = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-",
    "math-ph",
    "nlin",
    "nucl-",
    "physics.",
    "quant-ph",
)


def arxiv_field(categories: str) -> str | None:
    """The field of an arXiv paper from its categories (the first is the primary): ``bio`` with
    any q-bio category, else ``chem`` with physics.chem-ph, else ``phys`` when the primary is a
    physics archive, else None."""
    names = categories.split()
    if any(name.startswith("q-bio") for name in names):
        return "bio"
    if "physics.chem-ph" in names:
        return "chem"
    if names and names[0].startswith(_PHYSICS):
        return "phys"
    return None


def _open_license(license_url: str | None) -> bool:
    return license_url is not None and "creativecommons.org" in license_url


# --------------------------------------------------------------------------------------------------
# Units of each source
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """What a source yields: ``split_key`` decides the split, ``order_key`` the place, ``docs``
    are written in order (a paper's pieces stay together)."""

    split_key: str
    order_key: str
    docs: tuple[str | bytes, ...]


Wanted = Callable[[str], bool]  # a split key's unit is used (its bucket is below keep)
Units = Callable[[Path, Counter[str], Wanted], Iterator[Unit]]


def _parquet_rows(path: Path, columns: Sequence[str]) -> Iterator[tuple[Any, ...]]:
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=8192, columns=list(columns)):
        yield from zip(*(batch.column(name).to_pylist() for name in columns), strict=True)


def _files(download: Path, source: Source) -> list[Path]:
    return [download / source.key / remote.file for remote in source.files]


def _arxiv_units(field: str) -> Units:
    def units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
        columns = ("id", "categories", "title", "abstract")
        for path in _files(download, ARXIV_METADATA):
            for paper, categories, title, abstract in _parquet_rows(path, columns):
                counts["papers"] += 1
                if arxiv_field(categories) != field:
                    continue
                counts["papers_in_field"] += 1
                if not wanted(paper):
                    continue
                yield Unit(paper, paper, (f"{' '.join(title.split())}\n\n{unwrap(abstract)}",))

    return units


def _chemrxiv_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    columns = ("doi", "license", "text")
    for path in _files(download, CHEMRXIV):
        for doi, license_name, text in _parquet_rows(path, columns):
            counts["papers"] += 1
            if not str(license_name).startswith("CC "):
                counts["papers_not_cc"] += 1
                continue
            counts["license " + str(license_name)] += 1
            key = str(doi)
            if not wanted(key):
                continue
            yield Unit(key, key, tuple(chunks(clean(text))))


def _pubmed_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    for path in _files(download, PUBMED_SUBSET):
        with path.open("rb") as handle:
            for line in handle:
                record = json.loads(line)
                counts["records"] += 1
                key = str(record["PMID"])
                if not record.get("abstract"):
                    counts["records_without_abstract"] += 1
                    continue
                if not wanted(key):
                    continue
                title = " ".join(str(record.get("title") or "").split())
                abstract = "\n".join(
                    " ".join(line.split()) for line in str(record["abstract"]).split("\n")
                )
                yield Unit(key, key, (f"{title}\n\n{abstract}" if title else abstract,))


def _stackoverflow_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    columns = ("question_id", "title", "question_body", "answer_id", "answer_body", "answer_score")
    group: list[tuple[Any, ...]] = []
    seen: set[int] = set()

    def emit(rows: list[tuple[Any, ...]]) -> Unit | None:
        question = int(rows[0][0])
        if question in seen:  # the rows of a question are contiguous in the source
            counts["questions_split"] += 1
            return None
        seen.add(question)
        counts["questions"] += 1
        key = str(question)
        if not wanted(key):
            return None
        answers = sorted({(-(r[5] or 0), int(r[3])): r[4] for r in rows if r[4]}.items())
        parts = [" ".join(str(rows[0][1]).split()), html_text(rows[0][2] or "")]
        parts += [f"Answer:\n{html_text(body)}" for _, body in answers]
        return Unit(key, key, ("\n\n".join(part for part in parts if part),))

    for path in _files(download, STACKOVERFLOW_PYTHON):
        for row in _parquet_rows(path, columns):
            counts["rows"] += 1
            if group and row[0] != group[0][0]:
                unit = emit(group)
                if unit is not None:
                    yield unit
                group = []
            group.append(row)
    if group:
        unit = emit(group)
        if unit is not None:
            yield unit


_PERMISSIVE = frozenset(
    {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}
)


def _codeparrot_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    for path in _files(download, CODEPARROT):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                counts["files"] += 1
                if record["license"] not in _PERMISSIVE:
                    continue
                if record.get("autogenerated"):
                    counts["autogenerated"] += 1
                    continue
                counts["files_permissive"] += 1
                repo = str(record["repo_name"])
                if not wanted(repo):
                    continue
                yield Unit(repo, f"{repo}/{record['path']}", (record["content"],))


def _uniref_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    for path in _files(download, UNIREF50):
        lines = path.read_bytes().split(b"\n")
        if lines[0] != b"id,name,text":
            raise ValueError(f"{path}: not a UniRef50 CSV head")
        for line in lines[1:-1]:  # the header, and the line the byte range cut
            _, name, sequence = line.split(b",", 2)
            counts["sequences"] += 1
            key = name.decode()
            if not wanted(key):
                continue
            yield Unit(key, key, (sequence,))


def _pubchem_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    for path in _files(download, PUBCHEM_10M):
        for (smiles,) in _parquet_rows(path, ("smiles",)):
            counts["molecules"] += 1
            if not wanted(smiles):
                continue
            yield Unit(smiles, smiles, (smiles,))


def _arxiv_open_physics(download: Path) -> dict[str, str]:
    """Normalised arXiv id (no slash) -> licence, for physics papers under an open licence."""
    papers: dict[str, str] = {}
    for path in _files(download, ARXIV_METADATA):
        for paper, categories, license_url in _parquet_rows(path, ("id", "categories", "license")):
            if arxiv_field(categories) == "phys" and _open_license(license_url):
                papers[paper.replace("/", "")] = license_url
    return papers


def _latex_units(download: Path, counts: Counter[str], wanted: Wanted) -> Iterator[Unit]:
    papers = _arxiv_open_physics(download)
    counts["open_physics_papers_in_metadata"] = len(papers)
    for path in _files(download, PROOF_PILE_ARXIV):
        with subprocess.Popen(["zstd", "-dcq", str(path)], stdout=subprocess.PIPE) as process:
            assert process.stdout is not None
            for line in process.stdout:
                record = json.loads(line)
                counts["papers"] += 1
                paper = str(record["meta"]["arxiv_id"]).replace("/", "")
                license_url = papers.get(paper)
                if license_url is None:
                    continue
                counts["papers_open_physics"] += 1
                counts["license " + license_url] += 1
                if not wanted(paper):
                    continue
                text = clean(record["text"])
                yield Unit(paper, paper, tuple(chunks(text)))
        if process.returncode:
            raise RuntimeError(f"zstd failed on {path} ({process.returncode})")


# --------------------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldDataset:
    """One of the eight datasets: its sources, how it splits, and its spec's context id."""

    name: str
    field: str
    form: str  # prose | native
    context_id: int
    sources: tuple[Source, ...]
    units: Units
    document: str
    heldout: int  # held-out units per BUCKETS
    keep: int = BUCKETS  # units used per BUCKETS (heldout included)
    group: int = 1  # documents joined, one per line, into one (SMILES)


DATASETS: dict[str, FieldDataset] = {
    ds.name: ds
    for ds in (
        FieldDataset(
            "fields_bio_prose",
            "biology",
            "prose",
            10,
            (PUBMED_SUBSET,),
            _pubmed_units,
            "title and abstract of a PubMed record",
            heldout=300,
        ),
        FieldDataset(
            "fields_chem_prose",
            "chemistry",
            "prose",
            10,
            (CHEMRXIV,),
            _chemrxiv_units,
            "a piece (<= 16 KiB, cut at paragraph ends) of a ChemRxiv preprint's text",
            heldout=1000,
            keep=5000,
        ),
        FieldDataset(
            "fields_phys_prose",
            "physics",
            "prose",
            10,
            (ARXIV_METADATA,),
            _arxiv_units("phys"),
            "title and abstract of an arXiv paper with a physics primary category",
            heldout=100,
            keep=2000,
        ),
        FieldDataset(
            "fields_code_prose",
            "coding",
            "prose",
            11,
            (STACKOVERFLOW_PYTHON,),
            _stackoverflow_units,
            "a Stack Overflow Python question (title, body) and its answers, best first",
            heldout=150,
            keep=4000,
        ),
        FieldDataset(
            "fields_bio_native",
            "biology",
            "native",
            12,
            (UNIREF50,),
            _uniref_units,
            "one UniRef50 representative protein sequence (amino-acid letters)",
            heldout=150,
        ),
        FieldDataset(
            "fields_chem_native",
            "chemistry",
            "native",
            13,
            (PUBCHEM_10M,),
            _pubchem_units,
            "32 PubChem molecules as canonical SMILES, one per line",
            heldout=200,
            keep=6000,
            group=32,
        ),
        FieldDataset(
            "fields_phys_native",
            "physics",
            "native",
            14,
            (ARXIV_METADATA, PROOF_PILE_ARXIV),
            _latex_units,
            "a piece (<= 16 KiB, cut at paragraph ends) of an openly licensed physics paper's "
            "LaTeX body",
            heldout=2000,
        ),
        FieldDataset(
            "fields_code_native",
            "coding",
            "native",
            15,
            (CODEPARROT,),
            _codeparrot_units,
            "one permissively licensed Python source file",
            heldout=1000,
        ),
    )
}


def _bucket(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "big") % BUCKETS


def _digest(name: str, key: str) -> bytes:
    return hashlib.blake2b(f"{name}/{key}".encode(), digest_size=16).digest()


def _documents(texts: Iterable[str | bytes], counts: Counter[str]) -> list[bytes]:
    docs: list[bytes] = []
    for text in texts:
        doc = clean(text)
        if _SEP in doc:
            counts["dropped_separator"] += 1
            continue
        if len(doc) > MAX_BYTES:
            counts["capped"] += 1
            doc = cap(doc)
        if len(doc) < MIN_BYTES:
            counts["dropped_short"] += 1
            continue
        docs.append(doc)
    return docs


def _grouped(docs: list[bytes], group: int) -> list[bytes]:
    if group == 1:
        return docs
    return [b"\n".join(docs[i : i + group]) for i in range(0, len(docs), group)]


def write_raw(
    ds: FieldDataset, units: Iterable[Unit], raw: Path, counts: Counter[str]
) -> dict[str, Any]:
    """Split, order, clean and write the units of ``ds`` as ``raw/train.txt`` and
    ``raw/heldout.txt`` (see the module docstring); returns the files' sizes and hashes."""
    pools: dict[str, list[tuple[bytes, tuple[str | bytes, ...]]]] = {"heldout": [], "train": []}
    for unit in units:
        bucket = _bucket(_digest(ds.name, unit.split_key))
        if bucket >= ds.keep:
            continue
        split = "heldout" if bucket < ds.heldout else "train"
        pools[split].append((_digest(ds.name, unit.order_key), unit.docs))
    raw.mkdir(parents=True, exist_ok=True)
    held_out: set[bytes] = set()
    files: dict[str, Any] = {}
    for split in ("heldout", "train"):
        pool = sorted(pools.pop(split), key=lambda item: item[0])
        seen: set[bytes] = set()
        docs: list[bytes] = []
        for _, texts in pool:
            for doc in (clean(t) for t in texts) if ds.group > 1 else _documents(texts, counts):
                key = hashlib.blake2b(doc, digest_size=16).digest()
                if key in seen:
                    counts[f"{split}_duplicates"] += 1
                    continue
                if split == "train" and key in held_out:
                    counts["train_equal_to_heldout"] += 1
                    continue
                seen.add(key)
                docs.append(doc)
        del pool
        if ds.group > 1:
            docs = _documents(_grouped(docs, ds.group), counts)
        if split == "heldout":
            held_out = seen
        written = 0
        total = 0
        path = raw / f"{split}.txt"
        with path.open("wb") as out:
            for doc in docs:
                if split == "train" and total + len(doc) + 2 > TRAIN_BYTES + TRAIN_MARGIN:
                    break
                if written:
                    out.write(b"\n" + _SEP + b"\n")
                out.write(doc)
                written += 1
                total += len(doc) + 2
        counts[f"{split}_documents_available"] = len(docs)
        counts[f"{split}_documents_written"] = written
        files[split] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "documents": written,
            "positions": total,
            "sha256": _sha256(path),
        }
        del docs
    return files


def convert(ds: FieldDataset, root: Path, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write ``root/raw/<name>/{train,heldout}.txt`` and ``SOURCE.json`` for ``ds``."""
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    raw = root / "raw" / ds.name

    def wanted(key: str) -> bool:
        return _bucket(_digest(ds.name, key)) < ds.keep

    files = write_raw(ds, ds.units(root / "download", counts, wanted), raw, counts)
    record: dict[str, Any] = {
        "dataset": ds.name,
        "field": ds.field,
        "form": ds.form,
        "document": ds.document,
        "context_id": ds.context_id,
        "separator": SEPARATOR,
        "max_bytes": MAX_BYTES,
        "min_bytes": MIN_BYTES,
        "split": {
            "rule": "blake2b(<dataset>/<split key>) mod 10000: < heldout held out, < keep train",
            "heldout": ds.heldout,
            "keep": ds.keep,
            "group": ds.group,
        },
        "sources": [
            {
                "key": source.key,
                "title": source.title,
                "homepage": source.homepage,
                "revision": source.revision,
                "license": source.license,
                "files": [
                    {
                        "url": remote.url,
                        "sha256": remote.sha256,
                        "bytes": remote.first_bytes or remote.size,
                        "range": None
                        if remote.first_bytes is None
                        else f"0-{remote.first_bytes - 1}",
                    }
                    for remote in source.files
                ],
            }
            for source in ds.sources
        ],
        "files": files,
        "counts": dict(sorted(counts.items())),
    }
    (raw / "SOURCE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"converted {ds.name} in {time.perf_counter() - started:.0f} s")
    for name, value in sorted(counts.items()):
        log(f"  {name}: {value}")
    for split, info in files.items():
        log(f"  {split}: {info['documents']} documents, {info['bytes']} bytes, {info['sha256']}")
    return record


# --------------------------------------------------------------------------------------------------
# Word ends
# --------------------------------------------------------------------------------------------------


def word_end_stats(
    tokens: npt.NDArray[np.uint8], n_docs: int, window: int = 256
) -> dict[str, float]:
    """Word ends of a shard's tokens by the rule of :func:`hsn.model.reader.word_ends`: a byte
    that is not an ASCII letter or digit (bytes from 128 count as letters) right after one that
    is. ``BOS`` and ``EOS`` are not letters, so no word runs across documents. Returns bytes per
    word end and, per ``window`` positions, word ends and ``write_events=word`` events (word
    ends plus document starts)."""
    t = tokens
    letters = (
        ((t >= 48) & (t <= 57)) | ((t >= 65) & (t <= 90)) | ((t >= 97) & (t <= 122)) | (t >= 128)
    )
    ends = int(np.count_nonzero(~letters[1:] & letters[:-1]))
    n = int(t.shape[0])
    return {
        "positions": float(n),
        "word_ends": float(ends),
        "bytes_per_word_end": n / ends if ends else float("inf"),
        "word_ends_per_window": ends * window / n,
        "events_per_window": (ends + n_docs) * window / n,
    }


def _stats(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"name": manifest["name"]}
    for shard in manifest["shards"]:
        with np.load(manifest_path.parent / shard["file"]) as arrays:
            tokens = arrays["tokens"]
            offsets = arrays["offsets"]
        lengths = np.diff(offsets) - 2
        stats = word_end_stats(tokens, int(shard["n_docs"]))
        stats["documents"] = float(shard["n_docs"])
        stats["mean_document_bytes"] = float(lengths.mean())
        stats["median_document_bytes"] = float(np.median(lengths))
        out[shard["split"]] = stats
    return out


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def _names(name: str) -> list[str]:
    if name == "all":
        return list(DATASETS)
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; one of {', '.join(DATASETS)} or all")
    return [name]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hsn.data.fields", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "convert"):
        cmd = sub.add_parser(command)
        cmd.add_argument("dataset", help="a dataset name or all")
        cmd.add_argument("--root", required=True, help="folder holding download/ and raw/")
    stats = sub.add_parser("stats", help="documents, lengths and word ends of built datasets")
    stats.add_argument("manifests", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "stats":
        for manifest in args.manifests:
            print(json.dumps(_stats(Path(manifest)), indent=2))
        return 0
    root = Path(args.root)
    names = _names(args.dataset)
    if args.command == "fetch":
        keys = dict.fromkeys(source.key for name in names for source in DATASETS[name].sources)
        for key in keys:
            fetch(SOURCES[key], root / "download")
        return 0
    for name in names:
        convert(DATASETS[name], root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
