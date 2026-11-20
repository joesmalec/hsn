# Field datasets: four fields, prose and native format

19-20 November 2026 (branch `claude/field-datasets`). Data only: a fetch-and-convert module
(`python/hsn/data/fields.py`), eight text specs and their golden manifests (`configs/data/fields_*`);
the importer (`python -m hsn.data.text`), the loader, the trainer and every model are unchanged.

## What they are

Eight byte-level corpora to compare models within a speed class on different kinds of material:
biology, chemistry, physics and coding, each as **field prose** and in the field's **native
format**. Every dataset has the shape of `tinystories_v1`: about 200 million training positions
(one pass is about 24,400 steps at batch 32 and 256-byte windows, so a 9k-step run reads about a
third and a 27k-step run about 1.1 passes), 2,000 dev and 2,000 test documents from a held-out
set of papers, questions, repositories, clusters or molecules, documents `BOS, bytes..., EOS`,
newline as marker 16, tab as 17, every byte a target (dev loss / ln 2 is bits per byte),
variant `out_of_range` (the state is carried across a document's windows, the gradient is not).

| Dataset | Source (pinned revision) | Licence | Document |
| --- | --- | --- | --- |
| `fields_bio_prose` | PubMed titles and abstracts, a random 2.39M-record subset of the NLM records: [`slinusc/PubMedAbstractsSubset`](https://huggingface.co/datasets/slinusc/PubMedAbstractsSubset) @ `111f2d4a`, files 0, 8 and 16 of 24 (300,000 records) | CC0-1.0 on the card; PubMed records of the NLM (abstract copyright with the publishers; NLM terms allow text mining) | title, blank line, abstract |
| `fields_chem_prose` | ChemRxiv preprints converted to Markdown with LaTeX math (ChemNLP): [`jablonkagroup/chemrxiv_processed`](https://huggingface.co/datasets/jablonkagroup/chemrxiv_processed) @ `341cc642`, the train split (26,045 papers) | per paper, all Creative Commons: CC BY 4.0 (5,146), CC BY-NC 4.0 (2,707), CC BY-NC-ND 4.0 (18,192) | a piece of a paper, at most 16 KiB, cut at paragraph ends |
| `fields_phys_prose` | arXiv metadata snapshot: [`librarian-bots/arxiv-metadata-snapshot`](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) @ `c428226b` (3,224,635 papers; 1,510,619 physics) | CC0-1.0 (arXiv metadata, abstracts included) | title, blank line, abstract (hard wraps joined) |
| `fields_code_prose` | Stack Overflow Python questions and answers, Aug 2008 - Oct 2016 (the Kaggle set): [`koutch/stackoverflow_python`](https://huggingface.co/datasets/koutch/stackoverflow_python) @ `ee6f3955` (539,238 questions, 987,122 answers) | CC BY-SA 3.0 | title, question, then each answer (`Answer:`), best score first; HTML as text |
| `fields_bio_native` | UniRef50 representative sequences: [`agemagician/uniref50`](https://huggingface.co/datasets/agemagician/uniref50) @ `cc382ac6`, the first 2 % of the bytes of each of the 96 train files (1,151,926 sequences) | CC BY 4.0 (UniProt) | one protein sequence (amino-acid letters, no header) |
| `fields_chem_native` | PubChem-10M (DeepChem / ChemBERTa), RDKit-canonical SMILES: [`sagawa/pubchem-10m-canonicalized`](https://huggingface.co/datasets/sagawa/pubchem-10m-canonicalized) @ `f8321960`, the train file (8,999,964 molecules) | Apache-2.0 on the card; PubChem records are public domain | 32 molecules, one SMILES a line |
| `fields_phys_native` | arXiv LaTeX bodies of proof-pile-2 (the RedPajama arXiv set: preamble, comments and bibliography already removed): [`EleutherAI/proof-pile-2`](https://huggingface.co/datasets/EleutherAI/proof-pile-2) @ `901a9273`, train files 005, 015, ..., 095 (156,098 papers), with the arXiv metadata above for the categories and licences | per paper; kept only Creative Commons or public-domain papers (8,136: CC BY 6,359, BY-SA 219, BY-NC-SA 565, BY-NC-ND 633, CC0 / public domain 360) | a piece of a paper, at most 16 KiB, cut at paragraph ends |
| `fields_code_native` | CodeParrot-clean, Python files from GitHub with exact duplicates removed: [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean) @ `35a59fb0`, `file-000000000001.json.gz` (100,000 files, 47,232 repositories) | per file; kept only MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense and CC0 files that are not auto-generated (58,934) | one source file, cut at a line end at 16 KiB |

"Physics" is a paper whose primary arXiv category is in astro-ph, cond-mat, gr-qc, hep-*,
math-ph, nlin, nucl-*, physics.* or quant-ph and that has no q-bio and no physics.chem-ph
category (`arxiv_field`). Full revisions, URLs and SHA-256 of every downloaded file are in
`SOURCES` in `python/hsn/data/fields.py`; each converted dataset's `SOURCE.json` (beside its raw
files) records them again with the conversion counts.

## How they are made

1. **Fetch** (`python -m hsn.data.fields fetch <name|all> --root <dir>`): anonymous downloads
   with a generic client name into `<dir>/download/<source>/`. Every URL pins a Hugging Face
   commit; a file with a published SHA-256 (all but the UniRef50 heads, which are byte ranges of
   pinned files) must match it. 7.4 GB in all.
2. **Convert** (`python -m hsn.data.fields convert <name|all> --root <dir>`) writes
   `<dir>/raw/<name>/train.txt`, `heldout.txt` (documents joined by `<|endoftext|>`) and
   `SOURCE.json`. A source yields units with a split key (PMID, DOI, arXiv id, question id,
   UniRef50 cluster, SMILES, repository); with `b = blake2b("<name>/<key>") mod 10,000` a unit
   is held out when `b < heldout` and used for train when `heldout <= b < keep` (`keep` thins the
   largest sources before their text is held in memory). Both files are in the order of a
   digest of the unit's order key (the split key; for code the repository and path), so the
   source's own order (by date, by length) does not leak into which documents train sees.
   Documents are cleaned as the importer reads them, capped at 16 KiB (at a line end, else a
   space; ChemRxiv and LaTeX papers are cut into pieces at paragraph ends instead), dropped
   below 64 bytes, deduplicated (first in order kept), and a train document equal to a held-out
   one is dropped (for Python also one that is mostly a copy of a held-out file, below). Train is written to 200 MB of positions plus a 2 MB margin; the held-out file
   holds the whole held-out pool (more than the 4,000 documents the importer takes; the rest
   is in no split). Same source revision, same bytes: all eight conversions were rerun into a
   second folder and gave byte-identical files.
3. **Build** with the unchanged importer:

   ```
   python -m hsn.data.text build --spec configs/data/<name>.yaml --raw <dir>/raw/<name> \
       --out data/<name> --golden configs/data/<name>.manifest.json
   ```

   About 80-145 s and 10 GB of memory each (the 32-gram leakage index over 200 MB). The built
   data on this machine is in `/home/joe/hsn-experiments/datasets/fields/<name>`, linked as
   `data/<name>` in this worktree.

Per-source details:

- **PubMed**: the record's title and abstract, white space inside each line collapsed; records
  without an abstract are skipped. 3 % of the PMIDs held out.
- **ChemRxiv**: the full text as the set has it (Markdown headings, LaTeX math in `\( \)`, author
  lines; no reference lists), cut into pieces of at most 16 KiB at blank lines. Held out by
  DOI (10 %; one version per paper in the set); half of the papers are read.
- **arXiv abstracts**: the title with white space collapsed, the abstract with its hard wraps
  joined (a line that starts with white space begins a paragraph). 1 % held out, 20 % read.
- **Stack Overflow**: the rows of a question are grouped; the HTML becomes text close to the
  post's Markdown (paragraphs apart by a blank line, list items as `- `, `<pre>` blocks verbatim
  and indented by four spaces, inline code in backticks, other tags dropped, entities decoded).
  1.5 % of the questions held out, 40 % read.
- **UniRef50**: the mirror's 96 train files hold the clusters sorted by sequence length,
  longest first, about 615,000 a file, so the head of every file (2 % of its bytes, whole lines)
  is a sample spread over all lengths. Sequences shorter than 64 residues are dropped (86,476,
  7.5 %), the 558 longer than 16 KiB cut. 1.5 % of the clusters held out.
- **PubChem**: the split is per molecule (2 % held out, 60 % read); molecules are ordered by
  their digest and grouped 32 to a document, so a document is a random draw of molecules.
  A train molecule equal to a held-out one would be dropped (none are: the set is unique).
- **arXiv LaTeX**: papers of the ten files whose metadata says physics and an open licence (5.2 %
  of the papers read); old-style ids are matched without their slash. 20 % of the papers held
  out (their pieces give 7,175 held-out documents; 4,000 are used).
- **CodeParrot**: one file per document, cut at the last line end before 16 KiB (8,581 files,
  15 %); held out by repository (10 %). The repository split alone left forks and vendored
  copies of held-out files in train (207 of the 2,000 dev files had more than 90 % of their
  32-byte windows in train), so for this set only (`near=0.5`) a train file with more than half
  of its 32-byte windows in the held-out files is dropped as well (5,958 files).

## Sizes and word ends

Train positions include each document's BOS and EOS. Mean (median) document length is over the
train documents, in bytes. Bytes per word end and events per 256-byte window follow
`hsn.model.reader.word_ends` (a byte that is not an ASCII letter or digit right after one that
is; bytes from 128 count as letters; BOS and EOS are separators), counted on the train shard by
`python -m hsn.data.fields stats data/<name>/manifest.json`. Events are what
`model.write_events=word` (with `model.word_address=none`) writes at: word ends plus document
starts. The manifest hash is the SHA-256 of `configs/data/<name>.manifest.json`.

| Dataset | Context id | Train positions | Train documents | Dev / test documents | Mean (median) document bytes | Bytes per word end | Events per window | Manifest SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fields_bio_prose` | 10 | 199,999,919 | 151,980 | 2,000 / 2,000 | 1,314 (1,305) | 6.58 | 39.1 | `5163f9e85993317d9b49d5a88e7386dc890d84b6eea36df70075c61ccc8f9adb` |
| `fields_chem_prose` | 10 | 199,998,754 | 16,268 | 2,000 / 2,000 | 12,292 (15,126) | 6.23 | 41.1 | `d3a87d038cc3e51b64f2bf4650473f067297f6621fc9cd664f99e84881cd26e0` |
| `fields_phys_prose` | 10 | 199,999,516 | 185,650 | 2,000 / 2,000 | 1,075 (1,031) | 6.42 | 40.1 | `f47a960e8d5f93973e32b16de7afcd864339a49b0f4cacfa8508791923d294f6` |
| `fields_code_prose` | 11 | 199,996,459 | 78,837 | 2,000 / 2,000 | 2,535 (1,996) | 6.53 | 39.3 | `cf2aa70eea12b5449a12d03be72648c8585504d3fc227cc09f2118a2ce7726a3` |
| `fields_bio_native` | 12 | 199,996,459 | 645,492 | 2,000 / 2,000 | 308 (212) | 309.84 | 1.7 | `f8f412db3d133cc507a7a05fbb7cd60ee1e5188ec73fb940640fe5c094cc2055` |
| `fields_chem_native` | 13 | 199,999,617 | 137,043 | 2,000 / 2,000 | 1,457 (1,445) | 4.26 | 60.3 | `194f9b10f5cb318254feadd2456d2472cc87805e273c89bf8191326d7186a428` |
| `fields_phys_native` | 14 | 199,994,089 | 14,540 | 2,000 / 2,000 | 13,753 (15,611) | 6.01 | 42.6 | `9d309c7533619ab18cbe29939fedfab873a04376ac26e40581acc7a90f586138` |
| `fields_code_native` | 15 | 199,994,711 | 29,736 | 2,000 / 2,000 | 6,724 (4,623) | 8.13 | 31.5 | `4949bc6e3b94610041e9e6841809d1e53d0bcc4a9b3cb8578f7ab8e5975f835c` |

For comparison, TinyStories (`tinystories_v1`) has 4.97 bytes per word end and 51.8 events a
window on the same count.

What this means for word-event runs:

- **Prose** (all four) has a word end every 6.2-6.6 bytes, 39-41 events a window, the regime
  the word events were built for (a little sparser than TinyStories).
- **LaTeX** (6.0 bytes, 43 events) and **Python** (8.1 bytes, 32 events) are in the same range:
  identifiers, keywords, commands and numbers are words (`\frac{a}{b}` ends three; Python's
  indentation runs are separators). Word events can run them; whether the address of such a
  "word" means much is another question.
- **SMILES** has the most events of all (4.3 bytes per word end, 60 a window): runs of letters
  and ring digits (`CC`, `c1ccccc1`) are cut by every bracket, `=`, `#`, `/`, `@` and line end.
  Word events run, but the "words" are fragments of a molecule string, not words.
- **Proteins** have no word breaks at all: a sequence is one "word" (310 bytes per word end,
  the mean sequence length), ended only by its EOS. Word events would write about once a
  document: 1.7 events a window (0.8 word ends plus 0.8 document starts), most windows with
  one or none. The event loop cannot run proteins in any meaningful way; use the per-byte
  memory (`write_events=byte`), or leave context 12 out of `model.word_contexts` under
  `word_address` (then every byte is its own word and an event, which is the per-byte loop).

## Leakage

From the golden manifests: distinct 32-byte n-grams of the dev and test documents that also
occur in another split (`overlapping_ngrams`), and how many of the 2,000 documents hold at least
one. Exact duplicates are zero everywhere by construction.

| Dataset | train -> dev | train -> test | dev -> test |
| --- | --- | --- | --- |
| `fields_bio_prose` | 86,258 in 1,898 | 88,728 in 1,902 | 6,789 in 934 |
| `fields_chem_prose` | 530,754 in 1,977 | 517,265 in 1,978 | 130,058 in 1,925 |
| `fields_phys_prose` | 87,797 in 1,939 | 87,724 in 1,937 | 4,098 in 759 |
| `fields_code_prose` | 287,074 in 1,928 | 280,896 in 1,925 | 41,697 in 1,471 |
| `fields_bio_native` | 11,564 in 29 | 3,104 in 26 | 0 in 0 |
| `fields_chem_native` | 417,174 in 2,000 | 418,817 in 2,000 | 46,061 in 1,930 |
| `fields_phys_native` | 463,966 in 1,998 | 578,311 in 1,999 | 141,780 in 1,987 |
| `fields_code_native` | 1,050,024 in 1,989 | 1,003,821 in 1,984 | 510,834 in 1,962 |

Natural text shares phrases, so none of these is "clean" in the synthetic sense (TinyStories:
298,688 dev 32-grams in 1,997 dev documents). The large counts have known causes:

- **Python** (the largest): licence headers (the Apache and MIT texts are hundreds of bytes,
  identical in thousands of files), shared imports and boilerplate, and copies of the same file
  in forks under another repository name (the split is by repository, the deduplication exact
  only). Per dev file, by hashed 32-byte windows: without the near-copy filter 32.5 % of all dev
  windows occurred in train, 519 dev files had more than half and 207 more than 90 % of their
  windows there; with it (the build above) 18.3 %, 225 and 6, median share per file 0.155.
- **LaTeX and ChemRxiv**: repeated macro sequences (`\begin{equation}`, `\({}_{2}\)` chemical
  formulas) and set phrases.
- **SMILES**: shared substructures 32 characters long (ring systems with substituents); every
  dev document of 32 molecules holds some. No molecule is in two splits.
- **Proteins**: almost none (UniRef50 clusters are below 50 % identity to each other).

## Selecting a dataset

Each dataset is one task of one variant; select it with four overrides:

| Dataset | Overrides |
| --- | --- |
| `fields_bio_prose` | `data.dataset=fields_bio_prose data.manifest=data/fields_bio_prose/manifest.json data.tasks=[fields_bio_prose] data.variants=[out_of_range]` |
| `fields_chem_prose` | `data.dataset=fields_chem_prose data.manifest=data/fields_chem_prose/manifest.json data.tasks=[fields_chem_prose] data.variants=[out_of_range]` |
| `fields_phys_prose` | `data.dataset=fields_phys_prose data.manifest=data/fields_phys_prose/manifest.json data.tasks=[fields_phys_prose] data.variants=[out_of_range]` |
| `fields_code_prose` | `data.dataset=fields_code_prose data.manifest=data/fields_code_prose/manifest.json data.tasks=[fields_code_prose] data.variants=[out_of_range]` |
| `fields_bio_native` | `data.dataset=fields_bio_native data.manifest=data/fields_bio_native/manifest.json data.tasks=[fields_bio_native] data.variants=[out_of_range]` |
| `fields_chem_native` | `data.dataset=fields_chem_native data.manifest=data/fields_chem_native/manifest.json data.tasks=[fields_chem_native] data.variants=[out_of_range]` |
| `fields_phys_native` | `data.dataset=fields_phys_native data.manifest=data/fields_phys_native/manifest.json data.tasks=[fields_phys_native] data.variants=[out_of_range]` |
| `fields_code_native` | `data.dataset=fields_code_native data.manifest=data/fields_code_native/manifest.json data.tasks=[fields_code_native] data.variants=[out_of_range]` |

Context ids: `model.n_context` is 16 (ids 0-15), 0 is the padding context of a batch and 1-9 are
taken (synthetic 1-5, TinyStories 6, recall and word-key sets 7-8, `synthetic_sumwords` 9 on its
branch), so only 10-15 were free for eight datasets. The three abstract and paper-prose sets
share 10 (English scientific prose; a run reads one dataset, so the id only names the material);
code prose is 11, proteins 12, SMILES 13, LaTeX 14, Python 15. A run with
`model.word_address=exact|bag` treats only `model.word_contexts` as text (default `[6]`), so word
address runs on these sets need e.g. `model.word_contexts=[10]` (or `[11]`, `[14]`, `[15]`).
Giving every dataset its own id would need `model.n_context` above 16 (a model change: the
context table and the router's `ctx / n_context` feature).

## Checks

- Every dataset loads through `ByteDataset.load` and trains: 3 steps of `hsn_t_nomem`
  (`configs/experiments/hsn_t_nomem.yaml`, `--profile cpu`, batch 32, the four overrides) on
  the CPU, then the full dev evaluation. Every run's dev loss is 5.533-5.539 nats (uniform over
  256 bytes is 5.545), as it should be after 3 steps; a run takes 75 s (abstracts, proteins,
  SMILES) to 410 s (LaTeX: 27 MB of dev text) on two threads, most of it the evaluation.
- `tests/test_field_datasets.py` (21 tests): cleaning, capping and cutting at paragraphs, the
  HTML and abstract converters, the arXiv field rule, the hash split and its determinism, the
  duplicate, short, separator, held-out-copy and near-copy rules, SMILES grouping and the train budget,
  every source's unit reader on small stand-in files (LaTeX through `zstd`), the fetch hash
  check, the word-end count against `hsn.model.reader.word_ends`, and a round trip through
  `build_text` and `ByteDataset.load`.
- Conversion is deterministic (all eight rerun, identical SHA-256) and the downloads match their
  pinned hashes (`fetch all` re-verifies and skips them).

## Decisions to check

- **Chemistry prose is full-text pieces, not abstracts.** The abstract sources were too small or
  not chemistry: arXiv physics.chem-ph has 28,502 papers (35 MB of abstracts), ChemRxiv abstracts
  alone about 26,000; the public "chemistry" abstract set found
  (`jablonkagroup/euro_pmc_chemistry_abstracts`, Europe PMC with a topic classifier) labels
  biomedical work as chemistry with full confidence (of 25 sampled titles about five were
  chemistry, the rest biochemistry, pharmacology, clinical or engineering). ChemRxiv full texts
  are chemistry by construction and big enough, but the documents are about 12 KB pieces of papers
  where bio and physics prose are 1-1.3 KB abstracts. Physics LaTeX has the same long pieces.
- **Biology prose is PubMed, not arXiv q-bio** (q-bio: 57,263 papers, 75 MB of abstracts).
- **Licences**: PubMed abstracts carry the publishers' copyright (the card's CC0 covers the
  compilation); Stack Overflow is CC BY-SA 3.0; most ChemRxiv papers are CC BY-NC-ND and some
  arXiv papers BY-NC-*: all allow non-commercial research use, not all allow commercial use or
  derivatives. The arXiv LaTeX set leaves out papers under arXiv's default non-exclusive
  licence on purpose.
- **Shared context id 10** for the three scientific prose sets (see above).
- **Near-copy filter on Python only.** It removes 11 % of the candidate train files (forks,
  vendored modules, licence-only stubs) and changes what "held out" means there from "another
  repository" to "another repository and not mostly a copy". The other sets keep the plain split.
- **Proteins shorter than 64 residues are dropped** (the importer's `min_bytes` of every set).
- **Held-out fractions** were set so that each held-out pool has at least 4,000 documents after
  filtering; for LaTeX that is 20 % of the papers, for ChemRxiv 10 %. Unused held-out documents
  are in no split.
