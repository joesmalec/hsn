# Byte-content options: content dropout, zero rows and a learned unknown row for never-trained bytes (experimental)

Status: **four opt-in keys for the HSN and HSN-T, built on 15 Nov 2026 (branch
`claude/key-content`, from main f3f271a; merged with main c5bfa5a on 20 Nov 2026, section
"Where they can be set"); not adopted. The runs of `key_content.yaml` are
quoted (not filed) in the section on the learned unknown row; none of `key_unknown.yaml`
exists.**
- `train.content_dropout = p` (default 0.0): training only. In every training window a random
  share `p` of the positions reads a zero row in place of its byte's content.
- `model.zero_unseen_bytes` (default false): a byte that no training window has held as an
  input reads a zero row, in training, at evaluation and at inference.
- **`model.zero_unseen_bytes` changes what the model reads at a byte it was never trained on.**
  That is a change to the model's input, not to a recipe, so the user may want to decide it as
  an amendment. It has no number here and is not adopted by any configuration.
- Both are left out of the configuration hash at their defaults. Compared with the code of
  f3f271a on the CPU: the configuration hash, memory key and arm id of the A6 recipe, of the
  memory core and of the 32-byte reader are the same values; 147 arms of six existing sweep
  specs expand to the same arm id, group id, memory key, run id and configuration hash; and
  eight trainer steps of a dense HSN, a routed HSN and an HSN-T give the same state-dict keys,
  initial and trained weights, losses, gradient norms, evaluations, logits, gradients and
  global random state, bit for bit.
- They change no cell, graph, unit, gate, router, address code, summary or loss equation, no
  metric, threshold or task.
- Two more opt-in keys of the same day (section "The learned unknown row" below):
  `train.content_dropout_target = zero | unknown` (what a dropped position reads) and
  `model.unseen_bytes_as = own | zero | unknown` (what a never-trained byte reads; `zero` is
  `model.zero_unseen_bytes = true`). With `unknown`, a learned row per content table takes the
  place of the zero row. Left out of the configuration hash at their defaults; compared with
  the code before them on the CPU, all 1,971 arms of the 46 specs in
  `hsn-experiments/remote/specs/` that expand on this branch keep their arm id, group id,
  memory key, run id and configuration hash.

## Cause

The hold-out task `synthetic_hold` trains on key bytes 64 to 111 and holds bytes 112 to 143
back as unseen keys. Of the fifteen 18,000-step runs of 14 and 15 Nov 2026, nine fail the
reliability rule: eight keep far recall on the training keys at 97 to 99 % and are 3.6 to 24
points lower on the unseen keys, and in the ninth the training keys fell too (93.9 %). The
evaluation-only investigation of 15 Nov (hand-off file `unseen_keys/RESULTS.md`; commit
76c2aec) measured where the loss comes in:

| Measured | Reading |
| --- | --- |
| Bytes 112 to 143 occur 0 times in the training shards, in any role | Their rows of the two content tables (the byte embedding that feeds `E`, and the reader's embedding) never get a gradient |
| Those rows at 18,000 steps are the step-0 draw times one number (cosine 1.000 in all 15 runs; 0.64 to 0.68, the weight decay) | They are random vectors, not content |
| With trained keys' content looked up for the unseen key bytes, and the bytes' own address codes left as they are, all seven failing reader runs are within about 1 point of their training keys | The fixed address codes bind a new key; the loss comes in through the content |
| With the never-trained rows set to zero, three of the nine failing runs recover completely (75.4 / 73.1 to 97.5 / 98.1 in the worst); with one training key's content for every key, three more, and the two failing controls in part | The trained weights read the row as if it said something, or expect "this is a key" there |
| In one run the training keys themselves lose 6.9 points when they all share one content | That run tells keys apart by content |

Inferred there, not measured: for a training key the content of the key byte and its address
are always both present and always agree, so nothing in the loss stops the network from also
using the content, and nothing says what to do with a row that was never trained.

The two keys act on that. Content dropout removes the content at random positions in
training, so the network has to do its work from the address there. Zero rows for
never-trained bytes make an unseen byte the blank that the dropout trained on, instead of a
random vector.

## What they do

Both keys act at one place: what the encoder `E` and, in HSN-T, the reader read of a byte.

```
z_t = E(emb(x_t))                 becomes   z_t = E(0)        at a blank position
c   = R(emb_R(x_1 .. x_T))        becomes   emb_R(x_t) = 0    at a blank position
```

One mask of blank positions serves both tables, so the byte's content is gone there for the
whole model (`HSN._content_blank`). A position is blank when

- `train.content_dropout = p`: the model is in training mode and the position was drawn. Every
  position of the window is drawn independently with probability `p`, whatever byte it holds;
- `model.zero_unseen_bytes`: its byte has not been an input of any training window, in every
  mode.

With both keys on, the union. With both at their defaults no mask is computed and the lookups
are the code they were.

**Unchanged at a blank position:** the scorer's fixed byte codes and address features (A5,
A9: the current byte, the previous byte, the n-grams, the bytes carried across windows), which
are computed from the byte ids and never from the content tables; the context symbol; the
reader's attention mask and positions; the cells, the units, the graph, the convex tanh gate,
the router, the address gate, the summary `q`, the readout; the target of the prediction loss
(`z*_t` is the byte's own row: it is a target and not an input, so `P` is trained on the same
targets as before); the decoder's output layer.

### `train.content_dropout`

The positions come from a generator made for the window from `(run seed, optimiser step)`
(`HSN.content_drop_mask`), on the CPU. So they are the same on every backend, they are drawn
again in a resumed run, in a replayed divergence dump and in the extra passes of a statistics
step (the resume and the statistics step are tested; no dump was replayed with the key), and
the global random stream is never read, at any `p`. The columns of a window are
drawn one after the other, so a window that the eager trainer cuts at its trailing padding
(P1.10d) blanks the positions that the uncut, compiled window blanks on those columns. The
windows of one optimiser step (`train.grad_accum > 1`) draw the same positions; with
`train.tbptt_windows > 1` (main, 16 Nov 2026), where they share one backward pass, the key is
refused.

Evaluation and inference run without it: the evaluation-mode model with the key is the model
without it, bit for bit.

A single training position (`HSN.step` in training mode) has no window to draw for and is
refused; in evaluation mode `step` is unchanged.

Since any position can be drawn, a value byte is blank as often as a key byte: at `p = 0.1`
one fact in ten is stored without its value's content. Those answers count in the training
loss; the scores to read are the evaluations, which run without dropout.

### `model.zero_unseen_bytes`

The model keeps a bool vector of length `vocab_size`, `byte_seen` (a buffer; in the state
dict only with the key on). The window trainer marks the input bytes of every training window
before the window runs (`HSN.mark_bytes_seen`, called in `TBPTTTrainer.fit`); nothing else
marks: not a forward pass in either mode, not an evaluation, not a diagnostic.

**Why "seen as an input of a training window" and not the gradients themselves.** The
question the key asks is whether a row has ever been trained. A row of either table can only
get a gradient at a position that holds its byte, so a byte that was never an input has
certainly never been trained: for such a byte the rule is exact. It is read from the batch
alone, with one index write per window, the same on every backend, and it needs nothing from
the data set (no list of key bytes, no task). The other way round it is generous: a byte that
has only stood where no loss reaches it (padding behind a document's last answer), or only in
steps that a guard skipped, counts as seen although its rows were never moved. Reading the
two tables' gradients after every backward pass would close that. It was not built: it ties
the rule to the trainer's step (the guards, a skipped step), where the input rule needs the
batch alone. On `synthetic_hold` the difference does not arise for the key bytes: bytes 112
to 143 are in no training window at all.

**Training is the same with and without the key.** A window's bytes are marked before it runs,
so no lookup of a training window is ever blank. On the CPU a run with the key takes the
steps of the run without it, bit for bit (losses, gradient norms, every weight): eight steps
of the small test models, and six steps of the hold-out arm's own configuration on the hold
data (its schedule, update limits, weight average and route guard, with an evaluation after
every second step). What differs is what is scored: the evaluations during the run and after
it, where an unseen byte reads zero. On the card two runs of one configuration differ
anyway, so there an arm with the key alone differs from the plain arm by run-to-run
variation as well.

One trained model therefore answers both questions. Scored as saved it has the key on;
`model.mark_bytes_seen(torch.arange(model.cfg.vocab_size))` after loading turns every byte to
"seen" and scores the same weights with the key off.

**On `synthetic_hold`** (measured on the CPU, 15 Nov 2026, the 32-byte reader's
configuration with both keys on): the first three training windows of batch 32 and the 40,000 training
documents together hold 169 different bytes (0 to 5, 16 to 18, 64 to 111 and 144 to 255), and
the three windows had marked all of them. Bytes 112 to 143 and 32 to 63 stay unseen. The
only bytes of the dev split that no training document holds are 112 to 127, its keys.

**Checkpoints.** `byte_seen` is saved with the model, and a resume restores it. The rollback's
snapshots and a divergence dump hold the tensors of the state dict, so by the code they hold
it too; neither was run with the key. Loading is strict as before, so a checkpoint written
with the key does not load into a model without it, and the reverse; a warm start
(`train.init_from`) needs a source run with the key, because the bytes that run trained on
are not otherwise known. Since the merge with main (20 Nov 2026) a warm start whose source
and model differ in `byte_seen` or the unknown rows is refused with a `ValueError` that names
the options to set (`hsn.runner.warm_start`; other mismatches keep the loader's own error).
The weight average (`train.weight_ema`) holds parameters only; the averaged model reads the
run's `byte_seen`.

## The learned unknown row (`train.content_dropout_target`, `model.unseen_bytes_as`)

**Why.** Measured on 15 Nov 2026 with HSN-T on the held-out keys, 18,000 steps
(`key_content.yaml`): content dropout to a zero row with the unseen bytes read as zero
repairs kv recall on the unseen keys on all three seeds, but on seed 3 it breaks delayed2h
for never-seen bytes (76.7 / 84.3 / 91.8 % at 12,000 / 15,000 / 18,000 steps, against 98 to
99 % on the seen pairs). With every key byte looking up one shared trained key's row instead,
delayed2h on never-seen bytes is 98.1 / 98.3 / 99.6 % and kv stays fine. Inferred, not
measured: the network has to learn that a missing content means "a byte whose content is not
known", not "blank". The usual way is one shared learned "unknown" row, which these keys build.

**What they do.** Two keys, both opt-in:

- `train.content_dropout_target`: `zero` (default, the dropout above, bit for bit) or
  `unknown`. With `unknown` the positions that `train.content_dropout` draws read a learned
  row in place of the byte's row: `byte_emb.unknown` (`[1, d]`) for `E`, and in HSN-T
  `reader.emb_unknown` (`[1, w]`) for the reader. The two rows are new parameters that exist
  only with this value. Each is drawn as its table's rows are (the byte row with
  `model.state_init` like `byte_emb`, the reader row from a normal of std 0.02 like the
  reader's table), from a CPU generator seeded with `(run seed, tag)` alone, after every
  other parameter: every other parameter starts as it does without the key, and the global
  random stream is not read. They are shaped as one row of their table, so a `fan_in` init
  reads the table's fan-in and the weight decay treats them as their tables. They get the
  gradient of exactly the positions that read them. Needs `train.content_dropout > 0`.
- `model.unseen_bytes_as`: what a byte reads that no training window has held as an input
  (the bytes of `model.zero_unseen_bytes`, marked the same way, the same `byte_seen` buffer),
  in training, at evaluation and at inference. `own` (default): its own rows, bit for bit.
  `zero`: a zero row, which is `model.zero_unseen_bytes = true` (the older spelling is kept;
  setting both is refused, so each configuration has one spelling: the two spellings give
  different configuration hashes for the same model, and one's checkpoint loads into the
  other). `unknown`: the learned rows of `train.content_dropout_target = unknown`, which it
  needs (without it the rows would not exist, or would never be trained: refused).

At a position the model reads, in this order: the byte's own row, or for a never-trained
byte what `model.unseen_bytes_as` names; then, in a training window, a drawn position reads
what `train.content_dropout_target` names, whatever it held before. So with
`unseen_bytes_as = zero` and the target `unknown` (allowed, not in any spec) a dropped
position reads the unknown row even at an unseen byte, and an unseen byte that is not
dropped reads zero. In a training window the trainer marks the window's bytes before it runs,
so with `unseen_bytes_as = unknown` the training lookups read the row only where dropped;
the unseen bytes read it at evaluation. Everything listed above as unchanged at a blank
position is unchanged here too (address features, prediction target, cells, router, gate).

In the code: `HSN._content_masks` returns the zero positions and the unknown positions
apart (never overlapping); `hsn.model.embeddings.replace_content` puts a zero row and the
unknown row in place in the lookup; `HSN.unknown_rows()` returns the two rows;
`ByteEmbedding.add_unknown_row` and `Reader.add_unknown_row` register them;
`Reader.forward(..., unknown=None)`. The rows are parameters of their tables' modules, so they
count in the `byte_emb` and `reader` rows of `param_report` (`d + w` more parameters: 128 at
the hold arm's sizes) and in the `grad_norm/byte_emb` and `grad_norm/reader` groups; the
weight average holds them; the update limits of the recipes (`cell.*`, `router.*`) do not
reach them. A checkpoint with the rows does not load into a model without them, nor the
reverse. `train.content_dropout_target` is a `train.*` key and not in the memory key;
`model.unseen_bytes_as` at a non-default value is (as `model.zero_unseen_bytes` is), so an arm
with it has no stored step-time or memory estimate.

Refused where the zero-row keys are refused: the baselines, the reader alone,
`model.memory = false`, `hardware.window = fused`, and (through `train.content_dropout`)
`train.lambda_u > 0`.

Cost: one `torch.where` per table and its backward in the eager prologue, beside the
`masked_fill` of the zero rows; not measured on the card.

Spec `key_unknown.yaml` (`hsn-experiments/remote/specs/`): `key_content.yaml`'s base,
protocol and 18,000 steps; `_cdunk10` (dropout 0.1 to the unknown row, unseen bytes read it)
on seeds 0, 2 and 3, and `_cdunk05` (the same at 0.05) on seed 0. `key_content.yaml`'s
`_main` and `_cd10` runs are the references.

## Where they can be set

| | |
| --- | --- |
| `model_name: hsn` (the memory core alone) and `hsn_t` | both keys |
| eager window, compiled window (`hardware.compile_mode`) | both keys; the mask is applied in the window's prologue, which the compiled window runs eagerly, before its compiled blocks. Tested on the CPU (inductor); the card's replay of recorded graphs was not run |
| `hardware.window = split` (the HSN's) | both keys; the sweep and the replay start from the same prologue. Tested on the CPU (the uncompiled sweep, by the model and by its sweep twin); the card's compiled sweep was not run |
| `hardware.window = fused` | refused: untested, the fused sweep runs on the card only |
| `model.memory = false`, `model_name: reader`, the baselines | refused: the generic trainer trains them; it draws no positions and marks no bytes |
| `train.content_dropout` with `train.lambda_u > 0` | refused: untested; the counterfactual probes replay parts of a window |
| `model.reader_carry` (HSN-T, with `model.reader_top_words`) | every key. `Reader.read` takes the masks for the window's own positions; the cache holds the keys and values of earlier positions as they were computed when those positions were the window (a position dropped in its training window stays cached without its content; it is never embedded again). In training every window's bytes are marked before it runs, so a cached training position read its own row unless dropped; at evaluation nothing is marked or dropped, and an unseen byte is cached as it was read. The attention mask, the cache's validity and the word ends are computed from the bytes |
| `train.reach_dropout` (whole-window and carried reader) | every key: the short rows narrow the reader's mask, the options replace its embedding rows; both are handed to the same call |
| `model.write_events = word`, `model.word_address`, `model.cell_attention_heads` | every key: the event loop, its epilogue and the attention read of the cells read `z` and `c` from the prologue; the events and the word address are computed from the bytes |
| `hardware.compile_mode` (`default`, CUDA graphs), `hardware.round_products`, `hardware.message_gradients` | every key: the compiled event loop also runs after the eager prologue. Tested on the CPU (inductor, word events with the carry and C-read); not run on the card |
| `train.content_dropout` with `train.tbptt_windows > 1` | refused: the draw is one per optimiser step, so every window of a gradient segment would drop the same columns and the segment's one backward pass would read a mask that repeats every `train.unroll` positions. The seen-byte keys are allowed (each window's bytes are marked before it runs) |
| `train.host_sync`, `train.fused_adamw`, `hardware.baseline_compile`, the Transformer-XL and Gated DeltaNet baselines, `model.memory = false` recipes | refused: the generic trainer's keys and models (refused by their own rules or by the rule above) |

## Where it is in the code

- `python/hsn/config.py`: `TrainConfig.content_dropout`, `ModelConfig.zero_unseen_bytes`, the
  refusals in `RunConfig`, both keys in `HASH_OMIT_AT_DEFAULT`.
- `python/hsn/model/hsn.py`: `HSN.content_drop_mask`, `HSN.mark_bytes_seen`,
  `HSN._content_blank`; the mask is applied to `E`'s input in `_window_prologue` and in
  `_shared_path` (`step`), and handed to the reader in `_window_prologue`.
- `python/hsn/model/reader.py`: `Reader.forward(..., blank=None, unknown=None)` and
  `Reader.read(..., blank=None, unknown=None)` look the window's rows up through
  `Reader._embed`, which puts the zero and unknown rows in place (before the learned position
  table is added).
- `python/hsn/runner.py`: `warm_start` names the options when a checkpoint's `byte_seen` or
  unknown rows do not match the model's (`hsn.model.hsn.CONTENT_STATE_KEYS`);
  `HSNTNoMemory.load_memory_on_state` refuses those tensors.
- `python/hsn/train/tbptt.py`: the trainer marks each training window's bytes, and checks
  that the model was built from the configuration it trains.
- `python/hsn/experiments/spec.py`: `model.zero_unseen_bytes` at its default stays out of the
  memory key.

## Cost

Not measured on the card. Per training window the options add, in the eager prologue and
outside the compiled blocks:

| | Extra work per window | Expected at batch 32, window 256 |
| --- | --- | --- |
| `train.content_dropout` | the draw on the host (61 µs, measured on this machine's CPU), one copy of a `[B, T]` bool to the card, two `masked_fill` and their two backward kernels: five launches | about 0.15 ms |
| `model.zero_unseen_bytes` | one index write (the marking), one lookup of the seen bytes and its negation, two `masked_fill` and their two backward kernels, one more small buffer in the step snapshot: eight launches; four more per evaluation window | about 0.15 ms |
| both | the two, and one `or` (the four `masked_fill` kernels are shared): ten launches | about 0.3 ms |

The estimate counts 20 µs per eager launch (P2.8's spike: a step of 0.229 s with 11.7k host
launches, which bounds one launch from above). Against the 168.7 ms per step of main's HSN-T
with the 32-byte reader alone on the RTX 3060 (15 Nov 2026) that is 0.1 to 0.2 %, below the
spread of that timing (p95 0.9 ms above the median). A timing run on the card has to confirm
it.

## Tests (`tests/test_key_content.py`, `tests/test_key_content_main.py`)

`tests/test_key_content_main.py` (the merge with main, 20 Nov 2026): every configuration file of
main keeps the configuration hash, memory key and arm id main computes (c5bfa5a), also with the
four keys named at their defaults; the refusals and the combinations that run; three training
windows and one evaluation window of a stream with the carried reader, reach dropout, word
events and C-read are, bit for bit and in every gradient, the windows of the model without the
options with its two content reads replaced from outside (the next cache and the carried
address included); the cache holds what its window read; the compiled event loop with the
carry equals the eager one over five windows without a recompile; thirty trainer steps with
the carry and reach dropout, and with word events, the carry and C-read, train the unknown
rows; the warm-start and memory-off refusals. Two tests need the card and were not run (the
card was busy): twelve compiled trainer steps with CUDA Graphs (fp32, bf16) of those two
recipes, and the compiled event window with the carry against the eager one with CUDA Graphs.

The defaults keep the configuration hash, memory key and arm id that f3f271a computes, build
no buffer and no state-dict key, and leave the initialisation alone; the refusals. Content
dropout: the positions are a function of seed, step and shape and do not read the global
stream; in a training window `E` and the reader read zero rows at exactly the drawn positions
and the tables' rows everywhere else, the prediction target keeps the byte's row, and the
window equals, bit for bit and in every gradient, the window of the model without the key
whose two reads are zeroed from outside; in evaluation nothing changes; a single training
position is refused. Zero rows: exactly the bytes that were not marked, in both modes; a
window marks nothing; with every byte of the window seen the key changes nothing; `step`;
both keys blank the union; the buffer is saved only with the key on and the two kinds of
checkpoint do not load into each other. The compiled window equals the eager one and the
split window the autograd one with both keys on (values to float64 rounding; gradients term
by term to 1e-6 of a tensor's largest entry, because the router's cost path is float32 in
both windows); the split window's sweep twin shares the seen bytes and sweeps the model's
tape. The trainer: the compiled run's windows, the extra passes of its statistics steps
among them, blank the positions the eager run's cut windows blank, and its losses,
evaluation and per-term gradient norms follow the eager run's; thirty steps of each key and
of both, for the memory core and for HSN-T; the trainer marks the bytes of its training
windows and no others (the evaluations between the steps mark none), the checkpoint holds
them and a warm start takes them over; the key alone takes the same training steps as the
model without it; a resumed run draws the same positions and keeps the seen bytes.

The learned unknown row: the named defaults (`zero`, `own`) keep the configuration hash,
memory key and arm id; the rows exist only with the target `unknown`, are their generators'
draws in their tables' init, leave every other start value and the global stream alone, and
count in their tables' report rows; in a training window the dropped positions read exactly
the row, and the window equals, bit for bit and in every gradient, the window of the model
without the keys with the row put in from outside at those positions, the row's gradient
being that of its outside copy; never-trained bytes read the row in both modes (and with the
dropout the union) and `step` reads it; with `unseen_bytes_as = zero` and the target `unknown`
the dropped positions read the row and the other unseen bytes zero; `zero_unseen_bytes` and
`unseen_bytes_as = zero` give the same window; the refusals; the keys are hashed and only the
model key is in the memory key; bf16 autocast, the compiled window and the split window with
the rows (their gradients compared); thirty trainer steps for the memory core and HSN-T move
the rows; a resumed run takes the uninterrupted run's steps, the rows decayed like their
tables; three steps of the hold-out arm's own recipe on the hold data (skipped without
`data/synthetic_hold`), where bytes 112 to 143 stay unseen.

## What it has to show

Spec `key_content.yaml` (not in this repository: `hsn-experiments/remote/specs/`), on
`synthetic_hold` with the 32-byte reader on main's form, 18,000 steps, seeds 0, 2 and 3
(three of the four seeds that failed with this shape on 76c2aec; seed 1 passed), six runs:

| Arm | Keys |
| --- | --- |
| baseline | none (the runs of 14 and 15 Nov were made on 76c2aec with `model_name: hsn`; the comparison needs a baseline on this commit) |
| dropout | `train.content_dropout=0.1` |

No arm sets `model.zero_unseen_bytes`: it does not change training (above), so its effect is
read at evaluation. Every run is scored twice, as saved and with zero rows for bytes 112 to
143, which gives four cells: baseline, zero rows alone, dropout alone, dropout with zero
rows. The zero rows of the investigation's tool (`uk_standing.py`: the positions that hold
bytes 112 to 143 read a zero row in both tables) are the key at evaluation on the documents
the rule is scored on. Checked on the CPU on 15 Nov 2026: none of the 3,600 scored documents
holds a never-trained byte outside 112 to 143, and on a batch of each of the dev-key, the
test-key and the never-seen-byte documents the logits of this branch's HSN-T with the key
equal, bit for bit, the tool's and those of the model with the rows set to zero.

Pass, read for each score of the dropout arm: far recall on the unseen keys within 3 points
of the training keys at 12,000, 15,000 and 18,000 steps on every seed, with the training
keys and delayed2h not worse than the baseline's.

The investigation's replacement test says the zero rows alone repair the runs in which the
random row is read as noise (three of nine) and not the others; the baseline scored with
zero rows is that control.

## Results

None yet.
