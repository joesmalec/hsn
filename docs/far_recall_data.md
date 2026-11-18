# Far-recall test data: `synthetic_far` and `synthetic_far_early`

18 November 2026 (branch `claude/far-recall-data`). Synthetic data only: two evaluation sets, two
functions that give the position of an answer's fact, tests. The generators and the shard format
are unchanged, so every existing dataset builds to the same files; no model, trainer, evaluation
or configuration key changes.

## What they are

`synthetic_v1`'s `out_of_range` variant asks a fact at most 768 bytes back. That is inside
Transformer-XL's reach through its four blocks (1,024) and inside the reach of the HSN-T reader
carry, so a score on it cannot tell a memory from a carried cache. The far sets ask
`synthetic_v1`'s `kv` and `delayed` tasks from 272 to about 3,072 bytes back; from 1,024 bytes on,
no cache of either model reaches the fact. The analysis of 18 Nov 2026 built the first one by
hand (`~/hsn-handoff/OOR-RECALL-2026-11-18.md`); on it an HSN-T memory trained in range scored
98.9 % on delayed (98.6 % at 2,048 to 3,072 bytes back).

| Dataset | Tasks (context id) | Dev documents | Answers | Bytes | Distance of the first query |
| --- | --- | --- | --- | --- | --- |
| `synthetic_far` | `kv` (1), `delayed` (2) | 1,000 + 2,000 | 7,887 + 4,324 | 7.28 M | 272 to 3,072 |
| `synthetic_far_early` | `kv_back01` to `kv_back11` (1), `delayed_back01` to `delayed_back11` (2) | 11 x 200 + 11 x 200 | 15,068 + 4,514 | 7.62 M | 272 to 3,056 |

- Both are `synthetic_v1`'s tasks parameter for parameter: the dev key alphabet (bytes 112 to
  127), values, filler, markers and context ids. A model trained on `synthetic_v1` reads them
  as it reads `synthetic_v1`'s dev split. The seeds are new.
- Dev split only (`n_train = n_test = 0`): evaluation only, nothing to train on. The empty train
  and test shards are written and listed in the manifest.
- `synthetic_far` draws each document's length (up to 3,200 bytes, 13 windows) and its distance
  apart. Every delayed answer is 269 bytes or more back, 3,193 of them 1,024 or more. Most kv
  answers ask the middle from nearby; 1,093 are 256 bytes or more back (745 of them 1,024 or
  more), 1,000 of them the first query of a document.
- `synthetic_far_early` holds the fact early and moves the distance alone. Task
  `<family>_back<k>` is `synthetic_v1`'s `in_range` task with `256 k` added to both of its
  ranges, `k` = 1 to 11: an in-range document with `256 k` more bytes between the fact and the
  first query. A document spans `k + 1` windows exactly. Every delayed fact lies in the first
  window (the asked ones: median byte 44, at most 241) and every delayed answer in the last,
  `k` windows later; for kv that holds for the first query of each document.

The full description is in the header of each spec, `configs/data/synthetic_far.yaml` and
`configs/data/synthetic_far_early.yaml`. The generators are `_KV` and `_Delayed` in
`python/hsn/data/synthetic.py`, unchanged; the tests are `tests/test_far_recall_data.py`.

## The fact position, and why a far score needs it

In `synthetic_far` a long document asked a short way back holds its fact late in the stream, so
distance and fact position are mixed. They fail apart: on delayed, a 9k-step checkpoint
recalled facts written in the first 256 bytes at 57 to 84 % and facts written after byte 768 at
22 % or less, at every distance. Score a far set on the grid fact position x distance.

**No shard stores the fact position, and every shard has it.** For the `j`-th scored answer of
a document, at offset `t_j` in the document,

```
fact_pos[j] = t_j - deps[j]
```

`deps` is the dependency distance that every shard already carries, one entry per True
`loss_mask` position, and it is defined as the distance back to the input symbol the answer
depends on. Offsets count from the document's first symbol (`BOS` is 0).

| Family | The fact position is |
| --- | --- |
| `delayed` | the value byte of the queried fact `k v` |
| `kv`, answer a value | the value byte of the last `INS k v` or `OVR k v` on the queried key |
| `kv`, answer `NONE` | the key byte of the last `DEL k` on the queried key |

Two functions return it, both aligned with `deps`:

- `hsn.data.synthetic.fact_positions(example)`: int64, one entry per answer of a document
  (an `Example`, generated or loaded through `ByteDataset`).
- `hsn.data.splits.shard_fact_positions(arrays)`: int32, one entry per answer of a shard, from
  its `loss_mask`, `offsets` and `deps` (the arrays of `read_shard`, or an opened `.npz` file).

In a collated batch the `dep` field holds the distance at loss positions, so the fact position
of a loss position is its offset in its document minus `dep`. Because it is derived, shards
built before this change have it (the copy of the far set that the analysis scored included),
and no shard file, shard hash or manifest of any dataset changed. For text shards `deps` is 1
everywhere and the fact position says nothing.

Answers per cell, 256 bytes or more back (rows: fact position; columns: distance):

| Fact position | `synthetic_far` delayed: 256-767 / 768-1,535 / 1,536-3,072 | `synthetic_far` kv | `synthetic_far_early` delayed | `synthetic_far_early` kv |
| --- | --- | --- | --- | --- |
| 0-255 | 88 / 136 / 937 | 21 / 37 / 222 | 830 / 1,272 / 2,412 | 420 / 600 / 1,200 |
| 256-767 | 170 / 361 / 971 | 37 / 57 / 227 | 0 | 32 / 0 / 0 |
| 768-1,535 | 212 / 457 / 431 | 61 / 108 / 101 | 0 | 61 / 0 / 0 |
| 1,536 and later | 281 / 266 / 14 | 155 / 67 / 0 | 0 | 96 / 0 / 0 |

- The first row is the one that isolates distance: in-range training shows a fact in the first
  256 bytes of a document and nowhere else. In `synthetic_far` it is thin at the short
  distances; `synthetic_far_early` fills it. The later rows, which show whether writes hold late
  in a long stream, are `synthetic_far`'s.
- The cell "fact after byte 1,536, asked 1,536 or more back" is nearly empty by construction:
  the documents end at 3,200 bytes.
- A task of `synthetic_far_early` is one cell on its own: its plain dev accuracy is "fact early,
  asked `k` windows later" (for kv, over the first query of each document, or over the answers
  256 bytes or more back).
- The first query of a document always asks its first fact (delayed) or its anchor key (kv), as
  in `synthetic_v1`. Report first and later queries apart, and read kv's far answers as update
  and interference as well as reach: hundreds of operations on other keys lie in between.

## Building

CPU, about 5 seconds each, from the repository root:

```
PYTHONPATH=python python -m hsn.data.synthetic build --spec configs/data/synthetic_far.yaml \
    --out data/synthetic_far --no-leakage --golden configs/data/synthetic_far.manifest.json
PYTHONPATH=python python -m hsn.data.synthetic build --spec configs/data/synthetic_far_early.yaml \
    --out data/synthetic_far_early --no-leakage \
    --golden configs/data/synthetic_far_early.manifest.json
```

`--golden` fails the build if the manifest differs from the committed one. `--no-leakage`
because the build's leakage checks compare train splits with dev and test splits, and these sets
have no train split. The manifests were written with numpy 2.5.4; split hashes:

| Dataset | dev | train (empty) | test (empty) |
| --- | --- | --- | --- |
| `synthetic_far` | `d64a28c21a70…` | `a1a81da21c26…` | `9623efb214aa…` |
| `synthetic_far_early` | `6f9c7a04649b…` | `78c780f2ece0…` | `6ea95c747fa7…` |

Built on main (`f7e39be`, 18 Nov 2026), `synthetic_far` is the set the analysis built and scored
(`~/hsn-experiments/oor-recall/data/synthetic_far/`), byte for byte: the same manifest text and
the same six shard files, and `verify` passes on that copy with the committed spec. That copy
was built with `--no-leakage`; with the checks on, the shard and split hashes are the same and
the manifest lists ten comparisons, all zero. `synthetic_far_early` is new and no model has
been scored on it.

The sets were compared with `synthetic_v1` once, outside the build (18 Nov 2026): the dev
documents of both share no document and no 32-gram with the training documents of any of
`synthetic_v1`'s ten tasks, with its kv and delayed dev documents, or with each other.

## Loading and scoring them

The sets have no train split, so a training run cannot name one in `data.manifest`: the runner
loads the train and the dev split from one manifest. A scorer loads the dev documents beside the
run's own data and scores them with the run's model:

```
ByteDataset.load("data/synthetic_far/manifest.json", "dev", tasks=["delayed"],
                 variants=["out_of_range"], expected_unroll=256)
ByteDataset.load("data/synthetic_far_early/manifest.json", "dev", expected_unroll=256)
```

`tasks=None` loads every task of a manifest, in manifest order (kv first). Carry the state and
any cache across a document's windows as the run's own evaluation does (`data.align_docs`:
every document starts a window).

- This change adds the data and the fact position only. Scoring by distance inside the
  evaluation is a separate change (branch `claude/distance-eval`, in progress on 18 Nov 2026).
  The analysis used its own script, `~/hsn-experiments/oor-recall/eval_dist.py`.
- Cost of `synthetic_far`, from the analysis: about 3 minutes per HSN-T checkpoint on the
  RTX 3060, about 20 minutes on 4 CPU threads, under a minute for Transformer-XL.
  `synthetic_far_early` has about as many bytes; its cost has not been measured.
- Gate before reading the far cells: a run that has not learnt the in-range binding says
  nothing about distance.
