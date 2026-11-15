# Byte-content options: content dropout and zero rows for never-trained bytes (experimental)

Status: **two opt-in keys for the HSN and HSN-T, built on 15 Nov 2026 (branch
`claude/key-content`, from main f3f271a); not adopted; no run exists.**
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
windows of one optimiser step (`train.grad_accum > 1`) draw the same positions.

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
steps of the run without it, bit for bit (losses, gradient norms, every weight). What differs
is what is scored: the evaluations during the run and after it, where an unseen byte reads
zero. On the card two runs of one configuration differ anyway, so there an arm with the key
alone differs from the plain arm by run-to-run variation as well.

One trained model therefore answers both questions. Scored as saved it has the key on;
`model.mark_bytes_seen(torch.arange(model.cfg.vocab_size))` after loading turns every byte to
"seen" and scores the same weights with the key off.

**On `synthetic_hold`** (measured on the CPU, 15 Nov 2026, the treatment arm's
configuration): the first three training windows of batch 32 and the 40,000 training
documents together hold 169 different bytes (0 to 5, 16 to 18, 64 to 111 and 144 to 255), and
the three windows had marked all of them. Bytes 112 to 143 and 32 to 63 stay unseen. The
only bytes of the dev split that no training document holds are 112 to 127, its keys.

**Checkpoints.** `byte_seen` is saved with the model, and a resume restores it. The rollback's
snapshots and a divergence dump hold the tensors of the state dict, so by the code they hold
it too; neither was run with the key. Loading is strict as before, so a checkpoint written
with the key does not load into a model without it, and the reverse; a warm start
(`train.init_from`) needs a source run with the key, because the bytes that run trained on
are not otherwise known. The weight average (`train.weight_ema`) holds parameters only; the
averaged model reads the run's `byte_seen`.

## Where they can be set

| | |
| --- | --- |
| `model_name: hsn` (the memory core alone) and `hsn_t` | both keys |
| eager window, compiled window (`hardware.compile_mode`) | both keys; the mask is applied in the window's prologue, which the compiled window runs eagerly, before its compiled blocks. Tested on the CPU (inductor); the card's replay of recorded graphs was not run |
| `hardware.window = split` (the HSN's) | both keys; the sweep and the replay start from the same prologue. Tested on the CPU (the uncompiled sweep, by the model and by its sweep twin); the card's compiled sweep was not run |
| `hardware.window = fused` | refused: untested, the fused sweep runs on the card only |
| `model.memory = false`, `model_name: reader`, the baselines | refused: the generic trainer trains them; it draws no positions and marks no bytes |
| `train.content_dropout` with `train.lambda_u > 0` | refused: untested; the counterfactual probes replay parts of a window |

## Where it is in the code

- `python/hsn/config.py`: `TrainConfig.content_dropout`, `ModelConfig.zero_unseen_bytes`, the
  refusals in `RunConfig`, both keys in `HASH_OMIT_AT_DEFAULT`.
- `python/hsn/model/hsn.py`: `HSN.content_drop_mask`, `HSN.mark_bytes_seen`,
  `HSN._content_blank`; the mask is applied to `E`'s input in `_window_prologue` and in
  `_shared_path` (`step`), and handed to the reader in `_window_prologue`.
- `python/hsn/model/reader.py`: `Reader.forward(..., blank=None)` zeroes the embedding's rows
  at the blank positions.
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

## Tests (`tests/test_key_content.py`)

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

## What it has to show

Spec `key_content.yaml` (not in this repository: `hsn-experiments/remote/specs/`), on
`synthetic_hold` with the 32-byte reader on main's form, 18,000 steps, seeds 0, 2 and 3
(three of the four seeds that failed with this shape on 76c2aec; seed 1 passed):

| Arm | Keys |
| --- | --- |
| baseline | none (the runs of 14 and 15 Nov were made on 76c2aec with `model_name: hsn`; the comparison needs a baseline on this commit) |
| dropout | `train.content_dropout=0.1` and `model.zero_unseen_bytes=true` |
| zero rows alone | `model.zero_unseen_bytes=true` |

Pass: far recall on the unseen keys within 3 points of the training keys at 12,000, 15,000
and 18,000 steps on every seed, with the training keys and delayed2h not worse than the
baseline's.

The investigation's replacement test says the zero rows alone repair the runs in which the
random row is read as noise (three of nine) and not the others; that arm is the control.

## Results

None yet.
