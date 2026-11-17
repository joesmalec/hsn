# Reach dropout of the HSN-T reader: `train.reach_dropout`

17 November 2026. Branch `claude/reach-dropout`, from main `71c183c`. Written and tested on the
CPU (torch 2.15.0+cu132, the card hidden); nothing has run on the card yet. A training option of
HSN-T with its memory, **not an equation change**: the model, its parameters, the cells, graph,
units, ε, the convex tanh gate, the router, `q` and the reader are unchanged. Only the reader's
attention mask changes, per row, in training windows.

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

### The compiled window (CUDA graphs)

For HSN-T with its memory the reader runs once per window in the eager prologue, outside the
compiled blocks (`hardware.compile_mode`, `docs/hsn_t.md`). The rows reach the device as a bool
`[B]` tensor (pinned and copied without waiting, `non_blocking`, on CUDA), the mask is built on
the device from tensors of fixed shape, and nothing reads a value back to the host. The compiled
blocks see `c_t` as before: same shape, dtype and strides, other values. So no new graph,
variant or recompile comes from the rows (tested on the CPU with `aot_eager`: no new graph in
later steps while the rows change, `fail_on_recompile` on a second window with other rows).

### Scope

Refused at configuration load (`reach_dropout_scope_error`) when `p > 0`:

| Configuration | Why |
| --- | --- |
| `model_name` other than `hsn_t` | no memory behind a reader |
| `model.memory=false` | `memory_off_refusals`: nothing to carry what the short rows miss |
| `model.reader_span` set | the key shortens a whole-window reader; a span already limits every row |
| `train.reach_dropout_span ≥ train.unroll` | a short row would read the whole window |
| `train.lambda_u > 0` (P2.3 probes) | untested: the probes run before the window's rows are drawn |

In scope: per-byte memory and word events (`model.write_events=word`), with or without C-read
(`model.cell_attention_heads`), eager or compiled, `train.grad_accum` and `train.tbptt_windows`
(each window draws its own rows). HSN-T's split and fused windows do not exist, and
`model.reader_carry` is refused with the memory anyway.

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

**To run on the card before the arms** (not run): the GPU compile check of the trainer,
`scripts/word_address_gpu_compile_check.py`, on arm (b) and its word-event twin of the E1 spec
(or on an existing top0 + C-read arm with `--override train.reach_dropout=0.5 --override
train.reach_dropout_span=16`), alone on the card: no recompile after the first step
(`--max-recompiles 1`), and the step rate against the same arm without the key.
