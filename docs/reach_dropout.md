# Reach dropout of the HSN-T reader: `train.reach_dropout`

17 November 2026. Branch `claude/reach-dropout`, from main `71c183c`. Written and tested on the
CPU (torch 2.15.0+cu132, the card hidden); nothing has run on the card yet. A training option of
HSN-T with its memory, **not an equation change**: the model, its parameters, the cells, graph,
units, ε, the convex tanh gate, the router, `q` and the reader are unchanged. Only the reader's
attention mask changes, per row, in training windows.

18 November 2026, branch `claude/reach-dropout-carry`, from main `aa652fd`: the key is accepted
beside the carried reader (`model.reader_carry > 0`), which it refused until then. See "With the
carried reader" below. Written and tested on the CPU; nothing has run on the card.

## Why

The design note `~/hsn-handoff/NAMES-DESIGN-2026-11-17.md` (summary, section 4 H1, section 5 E1).
With the whole-window (top0) reader nothing in training asks the memory to carry names: the
reader solves every name inside the window (0.03 to 0.06 bits per letter), the state is detached
at every window, and the name swap test shows that no top0 model uses which name it saw in an
earlier window (0.00 to 0.02 bits per letter, against 3.20 for the 32-byte reader with C-read
and 4.46 for Mamba). The 32-byte reader leaves names more than 63 bytes back to the memory inside
the window, and its memory then learns to store, read and use them, across windows too.

Experiment E1 gives the memory that in-window demand while the reader keeps the whole window:
in training a random part of the batch rows reads with a short per-block span `S`. With `S = 16`
over four blocks the stacked reach is 1 + 4 × 15 = 61 bytes, under the 64-byte edge of the names
test, so in those rows a name 64 or more bytes back must come from the memory. Evaluation always
reads the whole window.

## What the keys do

| Key | Default | Meaning |
| --- | --- | --- |
| `train.reach_dropout` | `0.0` (off) | probability `p` that a row of a training window reads with the short span, in `[0, 1]` |
| `train.reach_dropout_span` | `16` | the per-block span `S` of those rows (`≥ 1`), the mask `model.reader_span = S` gives; read only when `p > 0` |

In every training window the trainer (`TBPTTTrainer._reach_rows`) draws one bool per row,
`rand(B) < p`, from a CPU generator seeded by `(run seed, window counter)` (as the P2.3 probes'
generator): the global random stream is untouched, a resumed run replays the rows (the window
counter is in the checkpoint), and every backend draws the same rows. It hands them to the model
(`HSN.set_reach_rows`) for that window's forward pass, and for a statistics step's per-term
gradient passes of the same window, and clears them after the window. In the prologue
(`HSN._window_prologue`) the reader then reads with `Reader.forward(tokens, reset, short)`:

    mask[b, t, s] = (s ≤ t) & same document & (t − s < S  or  row b not short)

A short row's mask is exactly the mask of `model.reader_span = S`; every other row's is the
whole-window mask, bit for bit. At `p = 0` no row is drawn and the reader is called as before.

A model in evaluation mode ignores the rows (`model.training` is read), and only the trainer
sets them: evaluation in the trainer, the names test and every other script read with the
configured reader.

`p = 1` trains exactly as `model.reader_span = S` (tested, bit for bit) but evaluates the whole
window; `model.reader_span = S` alone is the config-only end point that also evaluates short.

### With the carried reader (`model.reader_carry = C > 0`)

Why (`~/hsn-handoff/NAMES-BEYOND-256-2026-11-18.md`, sections 1.3 and 2): the memory is trained
on distances between the reader's reach and the gradient segment. With the carry the reader
sees `C` bytes at every position of every window (`docs/amendments/reader_carry_memory.md`), so
with `C = 256` and 256-byte windows no loss inside a window needs the memory, and the blind
window start, which asked for it without the carry, is gone. Names last mentioned 256 or more
bytes back (27k steps, bits per letter): 0.58 for word events trained with the carry, 0.59 for
the memory-off model with the same carried reader, 0.22 for the per-byte memory with reach
dropout and no carry, 0.19 and 0.13 for Mamba (lr 1e-3 and 2e-3). Reach dropout beside the
carry asks the memory again while evaluation keeps the carried sight.

What a short row reads. The carried reader's mask (`Reader.carry_mask`) is a sliding span over
the cached positions and the window's, `0 ≤ t − s ≤ C`. A short row narrows it over both:

    mask[b, t, s] = carried mask[b, t, s] & (t − s < S  or  row b not short at t)

That is exactly the mask of `model.reader_carry = S − 1` (the carry is itself a span of `C + 1`
positions), so the short span must be shorter than the carry's: `S ≤ C`. With `S = 16` over four
blocks the reach is 61 bytes everywhere, across windows too; a short row has no blind start
either. `Reader.read(tokens, reset, cache, short)` takes the rows; without them it is the code
it was.

**The cache, and why a row's draw is held.** The cache holds every block's keys and values at
the last `C` positions. From the second block on they are functions of what the blocks below
read, so of the span the position was read with. If a row drew anew in every window, as it
does without the carry, two things would happen at a change:

* short after whole: the short row reads, inside its span, cached entries that were computed
  with the whole sight. They hold what lay up to `C` bytes behind them in every block below,
  so for the first `(L − 1)(S − 1)` positions of the window (45 of 256 with `S = 16` and four
  blocks) the row could see past its reach, to the names the dropout is there to hide, and it
  would not be the `reader_carry = S − 1` reader;
* whole after short: the whole-sight row reads cached entries computed under the short span, a
  reader that evaluation never runs (a training-only mismatch).

Chosen: **a row's draw is held until its next document start.** A document is short or whole
from its first byte to its last, in every window it runs over. The mask reads only positions
of the same document, so every cached entry a position reads was computed under the span the
position itself reads with. A short document is then read, cached positions included, exactly
as the model with `model.reader_carry = S − 1` reads it (tested to rounding, outputs and
cache), and a whole document exactly as evaluation reads it. No third reader is ever trained,
there is no extra reader pass, and nothing new lives on the device.

The other two options of the design memo, and why not:

* A second reader pass that fills the cache with the whole sight in the short rows. It removes
  the second case above and keeps the first: short rows would still read whole-sight entries at
  the start of every window. It also costs one more reader pass per window with a short row.
  Keeping both kinds of entries instead needs a second cache (42 MB at batch 32 for the
  160-wide four-block reader with `C = 256`) beside that pass.
* Showing that the mismatch is harmless needs runs on the card; it was not shown.

How it is done (`TBPTTTrainer._reach_rows`). The trainer draws one bool per row in every window
from the same generator as without the carry. A document that starts in the window takes that
window's draw of its row; the document going on keeps the draw it started with. The trainer
hands the model one flag per position, bool `[B, T]`: with `data.align_docs` (the recipes) a
document starts at a window's first column and the flags are constant in a row; without it a
window may hold the end of one document and the start of the next, each with its own flag. The
draw of each row's last document is kept on the host for the next window (`_reach_held`, a bool
per row) and written into every checkpoint (`extra["reach_held"]`), so a resumed run continues
bit for bit; a checkpoint without it is refused. Without the carry nothing is held, and the
rows are the per-window rows of 17 November, bit for bit.

What this changes in the draw: `p` is the share of documents that are read short, not of
windows. A long document is short, or whole, in all its windows; the expected share of short
positions is still `p`.

At `p = 0` no row is drawn, nothing is held or saved, and the carried reader, the trainer and
the checkpoint are those of main `aa652fd`, bit for bit (below, "How it was tested").

### The compiled window (CUDA graphs)

For HSN-T with its memory the reader runs once per window in the eager prologue, outside the
compiled blocks (`hardware.compile_mode`, `docs/hsn_t.md`). The rows reach the device as a bool
`[B]` tensor (pinned and copied without waiting, `non_blocking`, on CUDA), the mask is built on
the device from tensors of fixed shape, and nothing reads a value back to the host. The compiled
blocks see `c_t` as before: same shape, dtype and strides, other values. So no new graph,
variant or recompile comes from the rows (tested on the CPU with `aot_eager`: no new graph in
later steps while the rows change, `fail_on_recompile` on a second window with other rows).
With the carried reader the same holds: the flags are a bool `[B, T]` tensor built on the host
from the CPU window's `reset` (no device read), the reader still runs in the eager prologue, and
the cache keeps its shapes, dtype and layout (tested on the CPU: five consecutive compiled
windows with other rows on one graph under `fail_on_recompile`, and the compiled trainer).

### Scope

Refused at configuration load (`reach_dropout_scope_error`) when `p > 0`:

| Configuration | Why |
| --- | --- |
| `model_name` other than `hsn_t` | no memory behind a reader |
| `model.memory=false` | `memory_off_refusals`: nothing to carry what the short rows miss |
| `model.reader_span` set | the key shortens a whole-window reader; a span already limits every row |
| `model.reader_mask=causal` | SDPA's causal mask has no mask tensor for the short rows' span to narrow (`docs/reader_causal_flash.md`); `Reader.forward` refuses short rows there too |
| `model.reader_carry = C > 0` with `train.reach_dropout_span > C` | the carry is itself a span of `C + 1` positions: a short row would read as every other row (with `S ≤ C` the pair is in scope since 18 Nov 2026) |
| `train.reach_dropout_span ≥ train.unroll` | a short row would read the whole window |
| `train.lambda_u > 0` (P2.3 probes) | untested: the probes run before the window's rows are drawn |

In scope: per-byte memory and word events (`model.write_events=word`), with or without C-read
(`model.cell_attention_heads`), eager or compiled, `train.grad_accum` and `train.tbptt_windows`
(each window draws its own rows). HSN-T's split and fused windows do not exist. The carried
reader (`model.reader_carry > 0` on the memory, `docs/amendments/reader_carry_memory.md`) is in
scope since 18 Nov 2026, with and without `data.align_docs` ("With the carried reader" above).

The refusals of the carried reader itself stay as they were, because the change does not make
them sound: `model.reader_span` beside the carry (the carry is the span; the short reader in
every row and in evaluation is `model.reader_carry = S − 1` alone), `model.reader_mask=causal`
beside the carry or beside reach dropout (no mask tensor to narrow), `model.reader_top_words`
with the memory (`Reader.read` also refuses short rows on a reader with word slots: they reach
beyond any span), `model.address_gate_input=reader_prev`, `train.lambda_u > 0`.

Both keys are left out of the configuration hash at their defaults (`HASH_OMIT_AT_DEFAULT`), so
every existing configuration keeps its hash and every arm its id. The sweep's memory key (stored
step time and memory) holds no `train` key besides the batch, window, gate and windows per step,
so an arm with the key has the memory key of the arm without it: the step time is the reader's.
Expected cost: none measurable (a `[B, T, T]` boolean `and` per window).

## The E1 arms

Seed 0, 9,000 steps, TinyStories data block as in the earlier text specs, on
`configs/experiments/hsn_t_memory.yaml` (per byte) or `configs/experiments/hsn_t_memory_words.yaml`
(word events) with the top0 reader and C-read:

    TOP0 = model.reader_width=160 model.reader_layers=4 model.reader_rotary_pairs=[20,20,20,0]
           model.reader_span=null model.cell_attention_heads=4

| Arm | Recipe | Overrides |
| --- | --- | --- |
| (a) control | per byte | `TOP0` |
| (b) reach dropout | per byte | `TOP0` `train.reach_dropout=0.5` `train.reach_dropout_span=16` |
| (c) span 16 everywhere | per byte | `TOP0` with `model.reader_span=16` in place of `model.reader_span=null` |
| (b) word events | words | `TOP0` `train.reach_dropout=0.5` `train.reach_dropout_span=16` |
| (c) word events | words | `TOP0` with `model.reader_span=16` |

## How it was tested

`tests/test_reach_dropout.py` (CPU):

* the keys are opt-in: at their defaults the hash is the hash without them, set they change it;
  the memory key does not change; refused outside the scope above;
* the reader: with short rows the mask of a short row is the span reader's and every other
  row's the whole-window reader's, outputs alike, bit for bit; no short row gives the reader
  without the key bit for bit;
* the model (per byte and word events, with and without C-read, float64): a short row's logits
  and final state equal the `model.reader_span = S` model's on that row and every other row the
  whole-window model's, to rounding (1e-12; C-read's batched products round by the other rows,
  2.4e-15 measured, against a gap of 2 to 4 between the readers); in evaluation mode the rows
  change nothing, bit for bit;
* the trainer at `p = 1` equals the `model.reader_span = S` configuration bit for bit (losses,
  per-term gradient norms of the statistics steps, raw gradients, weights), while its evaluation
  is the whole-window configuration's evaluation of the same weights;
* the rows: a rerun with the same seed repeats them (and the losses), they change from window to
  window, another seed draws others, the global random stream is untouched, and at `p = 0` no
  row reaches the model;
* the compiled window (`aot_eager`, per byte and word events, with C-read) equals the eager one
  with short rows, and a second window with other rows runs without a recompile; the compiled
  trainer (`hardware.compile_mode=reduce-overhead`, three statistics steps) equals the eager
  one, and steps 2 and 3 compile no new graph.

`tests/test_reach_dropout_carry.py` (CPU; the carried reader):

* the pair loads on both memory recipes; the keys stay out of the hash at `p = 0`, change it
  when set, and leave the model and the memory key alone; the refusals above;
* the reader without short rows, or with rows of which none is short, gives the same outputs,
  parameter gradients and next cache, bit for bit;
* a short row, over four windows with documents that start with a window, inside one and two
  bytes before a boundary, has the mask, the outputs and the cache of the reader with
  `model.reader_carry = S − 1` (float64, 1e-12), and every other row is the carried reader's
  bit for bit; the same per document with flags per position; a row that changes its span
  inside a document equals neither reader (the mismatch the held draw avoids);
* the model (per byte and word events, with and without C-read, float64): a short row's logits,
  final state and cache over three windows are those of the `model.reader_carry = S − 1` model
  and every other row's the carried model's, to rounding; the same per document; in evaluation
  mode the rows change nothing, bit for bit (logits, state, next cache);
* the trainer at `p = 0` beside the carry hands the model nothing, holds and saves nothing, and
  equals the run without the keys bit for bit (records, raw gradients, weights, the carry);
* the trainer at `p = 1` equals the `model.reader_carry = S − 1` configuration to fp32 rounding
  (totals 1e-5, raw gradients), while its evaluation is the carried configuration's evaluation
  of the same weights and not the short one's;
* the held draw, with and without `data.align_docs`, on documents of two to four windows: the
  flags are the window's draw at a document start and the held draw elsewhere, they change
  between documents, a rerun repeats them; a run resumed from a checkpoint continues bit for
  bit (also with `train.tbptt_windows=2`), and a checkpoint without the held draws is refused;
* the compiled window (inductor on the CPU, per byte and word events, with C-read) equals the
  eager one over five consecutive windows with other short rows in each (rows, flags per
  position, none) on one graph under `fail_on_recompile`; the compiled trainer
  (`hardware.compile_mode=reduce-overhead`, `aot_eager`, four statistics steps on long
  documents) equals the eager one, and the steps after the first compile no new graph.

Beside the tests, `p = 0` with the carry was compared with main `aa652fd` itself (the same
script under both trees, 18 Nov 2026, CPU): the reader's outputs, gradients and cache, and four
trainer steps of three forms (per byte eager, word events with C-read eager, per byte with
C-read compiled with `aot_eager`): records, raw gradients, weights, the carry, the checkpoint's
keys and the evaluation loss, 1,729 values and tensors, all identical. Reach dropout without the
carry was compared the same way (1,096, identical). Arm (b) below at its real size ran six eager
steps on the CPU (18.5 s a step): losses fall, 13 to 17 of 32 rows short, the draws held over
every window boundary.

**To run on the card for the carried reader** (not run; the card is shared). The six `cuda`
tests of `tests/test_reach_dropout_carry.py`:

    pytest -m cuda tests/test_reach_dropout_carry.py

(`test_compiled_window_with_short_rows_on_the_carry_cuda`, two forms: five compiled windows
with other rows on one graph; `test_compiled_training_with_reach_dropout_on_the_carry_cuda`,
per byte and word events in fp32 and bf16: twelve trainer steps with CUDA graphs.) Then the
compile check of the trainer on arm (b) of the memo's experiment 2, the control arm of
`names256_shortcarry.yaml` with the two keys, alone on the card:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python scripts/word_address_gpu_compile_check.py \
        --spec ~/hsn-experiments/remote/specs/names256_shortcarry.yaml \
        --arm n256_top0_cread_c256 --max-recompiles 1 \
        --override train.reach_dropout=0.5 --override train.reach_dropout_span=16

and the same without the two overrides for the step rate of the control. Expected: the graphs
and recompiles of the control, and its step rate (the flags cost a few boolean operations over
`[B, T, C + T]` per window, in the eager prologue).

The arms of the memo's experiment 2 (per byte, `TOP0` as below, on `hsn_t_memory.yaml`):

| Arm | Overrides |
| --- | --- |
| a, control | `TOP0` `model.reader_carry=256` |
| b | `TOP0` `model.reader_carry=256` `train.reach_dropout=0.5` `train.reach_dropout_span=16` |
| c, stand-in | `TOP0` `model.reader_carry=15` |

**To run on the card before the arms** (not run): the GPU compile check of the trainer,
`scripts/word_address_gpu_compile_check.py`, on arm (b) and its word-event twin of the E1 spec
(or on an existing top0 + C-read arm with `--override train.reach_dropout=0.5 --override
train.reach_dropout_span=16`), alone on the card: no recompile after the first step
(`--max-recompiles 1`), and the step rate against the same arm without the key.
