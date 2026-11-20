# Word-key data: `synthetic_wordkeys`, `synthetic_wordhold` and `synthetic_sumwords`

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

## `synthetic_sumwords`: recall, then compute, with word keys

19 November 2026 (branch `claude/sum-words`). `recall_sum` (`synthetic_sum`, the mixed documents:
digits stated as facts, then sums whose operands are named by key) with the keys replaced by
words, so that the word-event memory can run it: on `synthetic_sum` every byte is an event (its
keys are single bytes under a context no run reads as text). Family `recall_sum_words`, task
`recall_sumw`, **context 9** (used by no other dataset). Data only, as above.

```
BOS filler (w SPACE d SPACE)^n filler SEP (QRY x PLUS y EQ s SEP)+ EOS
[BOS]@,,=;}...;,pfkod 0 pfkn 9 dcbdltkf 5 @_+[\\=)"...)>)*[SEP][QRY]pfkn[PLUS]dcbdltkf[EQ]41[SEP]
[QRY]pfkn[PLUS]pfkod[EQ]9[SEP][EOS]
```

- Key words are `synthetic_wordkeys`' vocabulary (the same fields and `vocab_seed`): 2,000
  training, 500 dev and 500 test words, disjoint. An operand is a key word (recalled) or one
  written digit; the item kinds (two, one, none recalled), their weights, `min_gap` 64 (every
  recalled digit at least 64 bytes before its item's `EQ`), the anchor (the last item recalls the
  first fact at distance D), the answer positions and the document counts are `recall_sum`'s.
- The digit is a word byte, so a fact needs a `SPACE` between key and digit and one after the
  digit. The filler is printable ASCII punctuation (31 bytes, no letter, digit, apostrophe or
  space, nothing >= 128), so no filler byte joins, ends or is a word under any word rule.
- Events (context 9 read as text): BOS, per fact the `SPACE` after the word (key `w`) and the
  `SPACE` after the digit (key `d`, with `a_prev` = W(w): the write of the fact), per item `PLUS`
  and `EQ` (they close the operands; a key operand is read under W(w) in `a_cur`) and the `SEP`
  after the sum: `1 + 2n + 3m` per document. Unlike `kv_words`, whose value byte closes the key
  (the key in `a_cur` at the write), a digit cannot close a word: the fact is bound across two
  events, A5's way (the key in `a_prev` at the write, in `a_cur` at the read).
- The in-range `dep_len_range` starts at 91 (`recall_sum`: 80), the smallest D at which a second
  fact and a last item of two 8-letter words always fit; a document needs D + 13 bytes.

Shape (built on 19 Nov 2026, numpy 2.5.4), against `synthetic_sum`:

| | symbols | facts | items | kinds none / one / two | mean dep_len |
| --- | --- | --- | --- | --- | --- |
| `recall_sumw` train, in range | 217.2 | 3.82 | 2.22 | 14.9 / 29.1 / 56.0 % | 166 |
| `recall_sum` train, in range | 210.4 | 4.00 | 2.45 | 14.9 / 28.5 / 56.7 % | 160 |
| `recall_sumw` dev, out of range | 788.9 | 4.00 | 2.50 | 15.8 / 28.1 / 56.1 % | 516 |

Events per 256-byte window (the trainer's loader, batch 32, `data.align_docs`, the exact address;
E is the event loop's slots per window, the batch's largest row count rounded up to 16):

| data, `model.word_contexts` | events per row-window | bytes per row-window | E |
| --- | --- | --- | --- |
| `recall_sumw` in range, `[0, 6, 9]` | 15.3 (max 25) | 217 | 32 |
| `recall_sumw` out of range (dev), `[0, 6, 9]` | 4.6 (max 13) | 220 | 16 |
| `recall_sumw` in range, `[0, 6]` (9 not listed) | 217 (every byte) | 217 | 256 |
| TinyStories v1 (dev), `[0, 6]` | 44.3 (max 61) | 221 | 64 |

Build (CPU, a few seconds) and select:

```
PYTHONPATH=python python -m hsn.data.synthetic build --spec configs/data/synthetic_sumwords.yaml \
    --out data/synthetic_sumwords --golden configs/data/synthetic_sumwords.manifest.json

data.dataset=synthetic_sumwords data.manifest=data/synthetic_sumwords/manifest.json \
    data.tasks=[recall_sumw] data.variants=[in_range]      # or [out_of_range] for evaluation
"model.word_contexts=[0, 6, 9]"                            # with hsn_t_memory_words(_carry).yaml
```

Split hashes: train `012ab9473d1f…`, dev `d2abafd19516…`, test `e7bae1bb43e9…` (spec
`ec0caa9cafc8…`). Built copy: `/home/joe/hsn-experiments/datasets/synthetic_sumwords`. Score by kind with
`hsn.data.synthetic.recall_sum_kinds(spec, tokens)` (0, 1 or 2 recalled operands per loss
position, aligned with `meta["deps"]`), as for `synthetic_sum`. Tests: `tests/test_sum_words.py`.
