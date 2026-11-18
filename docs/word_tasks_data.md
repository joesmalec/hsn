# Word-key recall data: `synthetic_wordkeys` and `synthetic_wordhold`

16 November 2026 (branch `claude/word-keys`); put on main on 17 November 2026 so that every model
family can run the word-key tasks. Synthetic data only: no model, trainer or configuration key
changes.

## What they are

Two synthetic recall datasets whose keys are words of 3 to 8 lower-case ASCII letters instead of
single bytes. On the byte-key tasks (`synthetic_hold` and the rest) every context is non-text, so
HSN-T's word events (`model.write_events=word`, `docs/amendments/write_events.md`) reduce to the
per-byte loop there; these tasks put the keys under a context the word address can read as text.

| Dataset | Tasks (family, context id) | Documents per task (train / dev / test) |
| --- | --- | --- |
| `synthetic_wordkeys` | `kvw` (`kv_words`, 7) | 20,000 / 1,000 / 1,000 |
| `synthetic_wordhold` | `kvw` (`kv_words`, 7), `delayedw` (`delayed_words`, 8) | 20,000 / 1,000 / 1,000 |

- `kv_words` is `kv` with word keys: `INS w v`, `OVR w v`, `DEL w SPACE`, queries `QRY w SPACE a`
  with the answer predicted at the `SPACE`.
- `delayed_words` is `delayed` with word keys: facts `w v`, then filler, then queries
  `QRY w SPACE v`.
- Both draw from one fixed vocabulary per split (`vocab_seed`): 2,000 training, 500 dev and
  500 test words, disjoint, in groups of four that share a beginning or an ending, so dev and test
  ask words never seen in training. `kvw` is the same spec in both datasets, so its shards are the
  same files byte for byte.
- In-range only; every document fits one 256-byte window.

The full description (the layouts, why the value byte follows the word directly, how a fact binds
under the word address, measured shape statistics) is in the header of each spec,
`configs/data/synthetic_wordkeys.yaml` and `configs/data/synthetic_wordhold.yaml`. The generators
are `_KVWords` and `_DelayedWords` in `python/hsn/data/synthetic.py`; the tests are
`tests/test_word_keys.py` and `tests/test_word_hold.py`.

## Building

CPU, a few seconds each, from the repository root:

```
PYTHONPATH=python python -m hsn.data.synthetic build --spec configs/data/synthetic_wordkeys.yaml \
    --out data/synthetic_wordkeys --golden configs/data/synthetic_wordkeys.manifest.json
PYTHONPATH=python python -m hsn.data.synthetic build --spec configs/data/synthetic_wordhold.yaml \
    --out data/synthetic_wordhold --golden configs/data/synthetic_wordhold.manifest.json
```

`--golden` fails the build if the manifest differs from the committed one. The manifests were
written with numpy 2.5.4; split hashes:

| Dataset | train | dev | test |
| --- | --- | --- | --- |
| `synthetic_wordkeys` | `9248f50f59ab…` | `b1325644def9…` | `d28a6c0f20cf…` |
| `synthetic_wordhold` | `ee94f1333ae3…` | `8c8bf39b9f71…` | `d9aaa9ec9351…` |

Built on main (17 Nov 2026), both are byte-identical to the copies the 16-17 Nov runs trained on
(built on `claude/word-keys` `ae2dac4` and `aa2e94c`), and the experiment specs of those runs
expand to the same arm ids on either tree.

## Selecting them

Any model family reads them with data overrides alone:

```
data.dataset=synthetic_wordkeys data.manifest=data/synthetic_wordkeys/manifest.json \
    data.tasks=[kvw] data.variants=[in_range]
data.dataset=synthetic_wordhold data.manifest=data/synthetic_wordhold/manifest.json \
    "data.tasks=[kvw, delayedw]" data.variants=[in_range]
```

HSN-T with word events also needs the task contexts read as text:
`model.write_events=word model.word_address=exact|bag "model.word_contexts=[0, 6, 7, 8]"`
(0: the padding behind each document under `data.align_docs`; 6: TinyStories, inert here).
Not for `write_events=word` with `word_address=none`: that rule counts bytes >= 128 as letters, so a
value would join its key word.

**Context 7 is shared** with `synthetic_sum` (`recall_sum`, 14 Nov 2026), whose keys and digits
are word bytes (ASCII digits and letters). The two datasets are not mixed in any spec, but a run on
`synthetic_sum` that lists 7 in `model.word_contexts` reads its documents as text. Context 8 is used
by `delayedw` only. The ids were kept because the context id is stored in every shard: changing it
would change the data and the runs' manifests.
