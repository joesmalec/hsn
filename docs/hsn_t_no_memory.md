# HSN-T without its memory (`model.memory: false`)

Status: **an opt-in switch of HSN-T, implemented on 14 Nov 2026; first text and arithmetic
runs on 14 and 15 Nov ("Text: the gap to the baselines" below); not adopted. The configuration
to use with the memory off is a user decision of 15 Nov ("Recommended memory-off
configuration" below).** Asked for by the
user on 14 Nov 2026: "I would like the memory option opt in and like
to see if we can beat gru (our original goal) on a compiled kernel", then "I'm looking to beat
gru on speed without memory, I know it can't be done with memory."

- `model.memory` (bool, default `true`), HSN-T only (`model_name: hsn_t`); every other
  `model_name` refuses `false`. Left out of the configuration hash at its default, so no
  existing configuration, hash or arm id changed.
- With `true` nothing changes: the class, the trainer, the parameters and every value are
  HSN-T's (`docs/hsn_t.md`).
- With `false` the model is **not an HSN and not the HSN-T that was run**: it has no cell, no
  state and no recall beyond the window. It is the path of HSN-T that does not pass through
  the memory, built so that its speed can be compared with the GRU baseline.

## What it computes

For a window `x_{1..T}` with context symbols `ctx_{1..T}`, all positions at once:

```
z      = E(emb(x))                    [B, T, d]
c      = R(x, reset)                  [B, T, w]     the reader of HSN-T, unchanged
u      = S([z ; 0 ; ctx ; c])         0 where HSN-T reads q_{t−1}
logits = D([0 ; u ; 0 ; c])           0 where HSN-T reads q_t and r_t
```

- **Built:** the byte and context embeddings, `E`, the reader, `S`, `D`. They are HSN-T's
  modules under HSN-T's names (`byte_emb`, `ctx_emb`, `encoder`, `reader`, `shared`,
  `decoder`), at HSN-T's widths.
- **Not built:** the cells, the router and its scorer, the unit bank, the graph, the
  predictor `P`, the halting head, the summary `q`, the selected-cell readout `r`. None counts
  as a parameter and none runs.
- **No loop over positions and no state.** Nothing is carried from one window to the next. A
  reset column starts a document for the reader, which reads nothing before it. `logits_t`
  depends on the document's bytes up to `t` inside the window (the last `1 + L (n − 1)` of them
  with `model.reader_span = n`) and on `ctx_t`.
- **The zero columns.** `S` keeps its `2d + context_dim + w` inputs and `D` its `3d + w`; the
  `d` columns of `q` in each and the `d` columns of `r` in `D` read zeros. This is what makes
  the model exactly HSN-T with every read of the state removed, and what lets HSN-T's weights
  load (below). It costs arithmetic that produces nothing. Counted as multiply-accumulates per
  position (a count, not a measurement):

  | Sizes | of `S` | of `D` | of the whole model (window 256) |
  | --- | --- | --- | --- |
  | Default (`w` 64, 2 blocks) | 20 % | 25 % | 15 % |
  | Reader widened to the GRU's size (below) | 13 to 15 % | 19 to 21 % | about 5 % |

  The request for this build put it at "a few percent"; that holds for the widened sizes and
  not for the default ones. Leaving the columns out of the arithmetic (the parameters keeping
  their shape) or out of the model would be a later, separate option; it is not built.

## Loading a trained HSN-T

`HSNTNoMemory.load_memory_on_state(state_dict)` takes the state dict of an HSN-T with its
memory (or a checkpoint's `model`) and loads the tensors of the six modules; the memory's
tensors (`predictor.*`, `cell.*`, `router.*`, `halting.*`) are left out, and a state dict that
lacks one of the six is refused. The logits are then the logits of that HSN-T's own window
with `S` reading zeros for `q_{t−1}` and `D` reading zeros for `q_t` and `r_t` (tested in
float64 to 1e-12, with the HSN-T running its cells as usual). `train.init_from` is not
extended: it still needs a checkpoint with exactly the model's keys.

A fresh model of seed `s` starts with HSN-T's draws for that seed where they do not depend on
the memory: the embeddings, `E`, the reader, and the input columns of `r` and `c`. The other
weights of `S` and `D` are other draws (HSN-T draws the predictor, the cell and the router
before them).

## Training

The model is stateless, so the generic trainer trains it
(`hsn.train.loop.Trainer`, as the GRU, Transformer and reader-alone models): `make_trainer`
gives the HSN's window trainer to an `HSN` instance and the generic one to everything else,
and `HSNTNoMemory` is a class of its own, not an `HSN`. The objective is the next-byte
cross-entropy alone.

Recipe: `configs/experiments/hsn_t_nomem.yaml`. Its training settings are the Transformer
baseline's (`configs/baselines/transformer_2m.yaml`: AdamW, 1e-3, cosine, warm-up 100), so
that the two differ in the model only.

In an experiment spec the family is **`hsn_t_nomem`**, a family of its own
(`hsn.experiments.spec.family_of`: `model_name: hsn_t` with `model.memory: false`). The
harness treats it as it treats a baseline: variant `none`, no `hardware.compile_mode`, the
verdict rules scoped to `hsn` or `hsn_t` do not apply, and reports keep it apart from the
HSN-T arms. An arm that names `family: hsn_t` with the memory off is refused, and the
reverse. The family of every existing arm is unchanged.

## What is refused, and what is not read

Refused at load (the training settings in one message that names each of them):

| Setting | Why |
| --- | --- |
| `model.memory: false` under any `model_name` but `hsn_t`, or without `model.reader` | the switch is HSN-T's |
| `model.selected_readout: false` (and so `model.routing: dense`, which A5 refuses) | HSN-T's rule; it keeps the `r` columns of `D` |
| `model.fast_cells`, `model.fast_readout` (A8) | cells of the memory and a read of them |
| `model.decoder_shortcut_dropout > 0` | it makes `D` read the selected cells |
| `train.lambda_p > 0` (the schema default is 1: a memory-off configuration sets 0) | `L_pred` reads the summary through `P` |
| `train.lambda_r`, `lambda_d`, `lambda_e`, `lambda_c`, `lambda_b`, `lambda_g`, `lambda_u`, `lambda_ponder`, `lambda_w` `> 0`; `train.aux_enabled` | the objective has no such term |
| `train.route_guard.enabled` | the router's units |
| `train.feedback_detach`, `train.state_grad_clip`, `train.state_grad_ratio` (A6) | cuts and limits of the state's gradient |
| `train.gate_mode` other than the default, `train.soft_until_step`, `train.unit_credit`, `train.ste_scope` | the routed selection |
| `train.temperature_*`, `train.epsilon_*`, `train.p_dense` | the selection's schedules |
| `train.cell_lr_mult`, `train.cell_weight_decay`, `train.router_lr_mult` | parameter groups of the memory |
| `train.address_gate_hold_step` | the address gate's weights |
| `train.optimizer: lazy_adamw` | the unit bank's groups (left unset, the optimiser resolves to `adamw`) |
| `train.update_limit` and `train.gradient_growth` groups with a selector on `cell.*`, `router.*`, `predictor.*` or `halting.*` | parameters of the memory (any other selector that matches nothing is refused by the limiter itself, as before) |
| `hardware.window: split` or `fused` | the HSN's windows (HSN-T's rule) |

Not read, and accepted:

- The `model.*` keys that describe the memory (`n_cells`, `n_units`, `k_select`, which
  routed mode `routing` names, `key_address`, `address_*`, `write_gate`, `cell_norm`,
  `cand_norm`, `predictor_target` and the like). A memory-off configuration may carry the
  values of the HSN-T it is the memory-less form of; two memory-off configurations that
  differ only in such a key compute the same model under two hashes and two arm ids.
- `train.q_resync_every`, `train.probe_fraction`, `train.n_probe`, `train.probe_horizon`,
  `train.ema_tau`, `train.variance_floor_gamma`, `train.reinforce_baseline_decay`: secondary
  values of settings that are refused above, or bookkeeping of the summary.

Available as for any model of the generic trainer: the spike guard and the rollback,
`train.weight_ema`, `train.adaptive_lr`, `train.grad_accum`, `train.answer_weight`,
`train.readout_lr_mult` (the decoder's group), and update-limit or gradient-growth groups on
the model's own parameter names. The weight average, the spike guard, `readout_lr_mult` and
an update limit on `reader.*` run in a test. The rollback, `adaptive_lr`, a gradient-growth
group, `grad_accum` and `answer_weight` were each run for twelve steps on the CPU during the
build and are not in a test.

`hardware.compile_mode` compiles the HSN's training window only. Under the generic trainer it
is ignored with a warning, and a spec refuses it for this family. The compile path for the
generic trainer's models is a separate change (`hardware.baseline_compile`, branch
`claude/baseline-compile`); on the integration branch `claude/nomem-compile` the two are
merged and the key compiles this model's window as one graph (`docs/baseline_compile.md`).
What this build guarantees is that the
forward compiles as one graph: no data-dependent branch, shape or host read
(`torch.compile(model, backend="aot_eager", fullgraph=True)`, forward and backward equal to
eager, the second window of the same shape reusing the graph; tested). Outside the tests the
`inductor` backend compiled it the same way on the CPU.

## Size

| | Parameters |
| --- | --- |
| Byte and context embeddings | 17,408 |
| `E` (64 → 512 → 64) | 66,112 |
| Reader (width 64, 2 blocks, 4 heads) | 116,480 |
| `S` (256 → 512 → 64) | 164,416 |
| `D` (256 → 512 → 256) | 262,912 |
| **HSN-T without memory, default sizes** | **627,328** |
| HSN-T with its memory, as run | 2,037,801 |
| GRU baseline (`gru_2m`) | 1,977,856 |
| Transformer baseline (`transformer_2m`) | 1,930,624 |

`HSNTNoMemory.param_report()` gives one row per module and
`hsn.train.memory_plan.memory_off_param_count` is the closed form. With the other sizes at
their defaults, a reader of width `w` and `L` blocks gives

```
445,312 + 1,282 w + L (12 w² + 13 w)
```

**To compare quality with the GRU at a matched parameter count** the reader would be widened
(`model.reader_width`, `model.reader_layers`, `model.reader_heads`; the width must be a
multiple of the heads with an even width per head). Three sizes near the GRU's 1.98 M, none
chosen and none run:

| `reader_width` | `reader_layers` | `reader_heads` | Parameters |
| --- | --- | --- | --- |
| 224 | 2 | 4 | 1,942,528 |
| 184 | 3 | 4 | 1,907,192 |
| 160 | 4 | 4 | 1,887,552 |

The speed question and the quality question use different sizes: the default model is a
third of the GRU's size, and a widened one does about three times the default's
multiply-accumulates per position.

## Text: the gap to the baselines (15 Nov 2026)

TinyStories (`tinystories_v1`), 9,000 steps, batch 32, learning rate 1e-3 unless noted, bits
per byte on the 2,000 dev stories (lower is better), uncompiled unless noted:

| Model | Parameters | Seed 0 | Seed 1 |
| --- | --- | --- | --- |
| Transformer baseline (192 × 4) | 1,930,624 | 0.919 | 0.925 |
| GRU baseline (learning rate 2e-3, its best) | 1,977,856 | 0.952 | 0.954 |
| Reader alone, 192 × 4 | 1,878,400 | 0.967 | |
| Without memory, reader 160 × 4 | 1,887,552 | 0.995 (compiled 0.983) | |
| Without memory, reader 184 × 3 | 1,907,192 | 0.996 | |
| Without memory, reader 224 × 2 (the shape chosen for speed) | 1,942,528 | 1.015 (compiled 1.021) | 1.027 |
| The same at learning rate 5e-4 / 2e-3 | | 1.048 / 1.078 | |

A compiled and an eager run of one configuration differ by up to 0.012 bits and two seeds by
0.012, so differences under about 0.015 between single runs are not read.

**Where the gap is** (the step-9,000 checkpoints scored on the CPU in float32; the totals
reproduce the runs). Documents are aligned to windows, so the position in the window is the
number of bytes the model can see:

| Window positions | Share of the bytes | Transformer | Reader 192 × 4 | No memory 160 × 4 | No memory 224 × 2 |
| --- | --- | --- | --- | --- | --- |
| 0 to 7 | 3.6 % | 1.477 | 1.495 | 1.507 | 1.510 |
| 8 to 31 | 10.7 % | 0.932 | 0.931 | 0.940 | 0.949 |
| 32 to 63 | 13.8 % | 0.907 | 0.922 | 0.939 | 0.960 |
| 64 to 127 | 26.0 % | 0.913 | 0.954 | 0.982 | 1.005 |
| 128 to 255 | 45.9 % | 0.880 | 0.954 | 0.993 | 1.014 |
| All | 100 % | 0.919 | 0.967 | 0.995 | 1.015 |

Three readings, each with what it rests on:

1. **The reader does not use the bytes far back, and the baseline does** (0.047 of the 0.096
   bits between the chosen shape and the baseline). `docs/hsn_t.md`, "The reader's positions",
   has the measurements: the same blocks with the baseline's position table in place of the
   rotation are the baseline; the reader's loss stops falling after about 64 bytes of context
   and no block of it attends far. The memory-off models inherit it: their loss rises from
   positions 32 to 63 to the end of the window where the baseline's falls. Three opt-in keys of
   the reader test the cause (`model.reader_rotary_pairs`, also with one count for each block
   so that the last block is not rotated, `model.reader_rotary_dtype`, `model.reader_position`).
2. **At one parameter count the model has a smaller reader than the reader alone** (0.028
   bits from the reader alone at 192 × 4 to the memory-off model at 160 × 4; one run each).
   `E`, `S` and `D` hold 657,280 of the chosen shape's 1,942,528 parameters and the reader
   1,267,840; the reader alone at 192 × 4 has 1,828,992 in the reader. The trained model does
   use them: with `u` held at zero in `D`'s input the chosen shape's loss on 300 dev stories
   rises from 1.016 to 1.420, with `z` at zero in `S`'s input to 1.188, with `c` at zero in
   `S`'s input to 1.234. Whether the same parameters would do more inside the reader is not
   measured; the arms of `text_gap.yaml` run the reader alone at 224 × 2 and a shape with a
   smaller `model.hidden` and a wider reader (an existing key: `S` and `D` keep their inputs
   and their place, their hidden layer narrows).
3. **Two blocks against four** (0.020 to 0.038 bits between 160 × 4 and 224 × 2 over the
   eager, second-seed and compiled runs; about 0.02 from position 32 on and less before it).
   This is the speed choice and is not changed here.

**Not built, by the architecture rule.** Nothing that removes or bypasses `S` or `D`, changes
`u = S([z ; q ; ctx ; c])` or `D([q ; u ; r ; c])`, or leaves the zero columns out. If the arms
show that the cost of point 2 is `S` and `D` themselves and not their size, that is a finding
to report and an amendment to ask for, not a switch.

Shapes near the same size and arithmetic, for point 2 (counts, not measurements; the
multiply-accumulates count the reader's attention at a window of 256):

| `model.hidden` | Reader | Parameters | In the reader | Multiply-accumulates per position |
| --- | --- | --- | --- | --- |
| 512 (default) | 224 × 2 | 1,942,528 | 1,267,840 | 2.09 M |
| 256 | 248 × 2 | 1,905,056 | 1,546,528 | 2.07 M |
| 128 | 264 × 2 | 1,939,936 | 1,747,680 | 2.12 M |

## Reading past the window (opt-in keys, 15 Nov 2026)

Measured before the keys (9k TinyStories steps, bits per byte): the memory-off model w160 × 4
with its top block unrotated (`model.reader_rotary_pairs: [20, 20, 20, 0]`) scores 0.940; its
top block's heads all look 32 or more bytes back. Trained at a window of 256 and read in
windows of 1024 it breaks (3.95 bits at positions 256 to 511, 5.26 at 512 to 1023): it is
asked for distances it never saw. And every window starts cold, while a story (median about
720 bytes) spans about three windows of 256. The two keys below let the reader read on from
the windows before without ever reading a distance it was not trained on. What they do to the
score: "Recommended memory-off configuration" below.

**`model.reader_carry: C`** (default 0: no cache, the model as before, bit for bit). Every
reader block keeps the keys and values of the last `C` positions of the windows before, in the
dtype of the parameters, detached: no gradient crosses a window, as for the GRU's carried
state. Position `t` of a window reads the positions `s` of its own document with
`0 ≤ t − s ≤ C`, cached or in the window: a sliding span. With `C = 256` every byte sees up to
256 bytes behind it at every window length and every document length, and no distance beyond
256 is ever read (the window of 256 alone reads up to 255).

- **Cache.** `(keys, values, valid)`: `keys` / `values [L, B, H, C, w / H]` (block, row, head,
  slot, channel), `valid [B, C]` (bool). Fixed shapes, no data-dependent branch, so the
  window compiles as one graph with the cache going in and out
  (`hardware.baseline_compile`: under CUDA graphs the cache returned is copied out of graph
  memory). Memory: `2 L B C w` values; w160 × 4 at batch 32 and `C = 256` is 10.5 M floats,
  42 MB in float32.
- **Rotation.** The keys are cached un-rotated. Slot `j` holds the position `j − C` before the
  window; it is rotated by that position (the rotary table's rows `C … 1` with the sine's sign
  turned: cosine is even, sine odd) and the window's keys and queries by `0 … T − 1` as
  without the cache, so a query and a key are rotated by their distance as before. An
  unrotated block (pairs 0) rotates nothing.
- **Documents.** `reset` acts as before: a slot is valid while its document goes on (a reset
  anywhere in a row's window invalidates the row's older slots), a window position reads a
  cached slot only when no reset lies at or before it, and a window position reads window
  positions of its own document only. A document's first position reads itself alone.
- **State.** `HSNTNoMemory` and `ReaderLM` (the reader alone, `model_name: reader`) return the
  cache from `init_state` and carry it through `forward`, so the generic trainer carries it
  from window to window in training and in the dev evaluation, detaches it at every window and
  writes it into checkpoints, as for the GRU. `scripts/reader_reach.py` carries it for the
  loss by window position.
- **Refused:** without a reader; HSN-T with its memory (the cells carry the state across
  windows and the reader's cache has no place in it; the HSN's memory code is not touched);
  `model.reader_position: learned` (a learned position is the position in the window, which a
  carried key would share with a key of the current window); `model.reader_span` (the carry is
  a sliding span itself).
- **Cost.** Each block's attention reads `C + T` keys in place of `T`; the MLPs, projections and
  everything outside the reader are unchanged. Measured roughly on the CPU (2 threads, a loaded
  host, w160 × 4 top0, batch 8 × window 256, the cache full): forward 77.8 → 110.6 ms (× 1.42),
  forward and backward 226 → 297 ms (× 1.31). Not timed on a card.

**`model.reader_top_words: N`** (default 0; needs `model.reader_carry > 0` and the top block
unrotated, the last entry of `model.reader_rotary_pairs` 0). The top block also reads the keys
and values at the last `N` word ends of its document before the window, carried and detached as
the cache is. A word end is a byte that is not a letter or digit (a space, punctuation, the
newline and tab markers 16 and 17, the end symbol) directly after one; ASCII letters and
digits and every byte of 128 and above count as letters. At a word end the model has read the
whole word, so the entry is causal. With `N = 256` the top block reaches about 256 words (about
1,300 bytes) back at a fixed cost.

- **Slots.** `(word_keys, word_values [B, H, N, w / H], word_valid [B, N], word_pos [B, N],
  last_letter [B])`: the most recent first, `word_pos` the slot's position from the window's
  start (int64), `last_letter` whether the window's last byte was a letter (the next window's
  first byte may end its word). After each window the slots are the `N` most recent word ends
  of the document going on (a fixed-size `topk` over the old slots and the window's word ends).
- **Not read twice.** A word slot within the span (`t − s ≤ C`) is masked: the span reads that
  position already, with the same key (the top block rotates nothing, so its keys carry no
  position). When the window is longer than `C + 1`, the top block also reads the window's own
  word ends beyond the span, and of these and the slots together the last `N`; at a window of
  256 with `C = 256` there are none.
- **Documents.** As for the cache: a slot of an earlier document is never read, and a
  document's first byte ends no word.
- **Cost.** The top block's attention reads `N` more keys. CPU, as above, with `N = 256`:
  forward × 1.45, forward and backward × 1.36 against no carry.

## Recommended memory-off configuration (user decision, 15 Nov 2026)

**Superseded on 16 Nov 2026** (user: "HSN-T top0, 1024 windows, no carry is the right
balance"): the shape below on 1024-byte windows without the carry is the recipe
`configs/experiments/hsn_t_nomem_top0_w1024.yaml` (data `configs/data/tinystories_w1024.yaml`):
0.7412 / 0.7375 dev bits per byte at 27,000 steps, 26.1 ms a step eager, alone on the RTX 3060
(Transformer at 1024: 0.7366 / 0.7304, 23.2 ms); the recipe's header has the comparison.

**17 Nov 2026** (user: "I agree that 160X3 is the best middle ground"): that recipe with 3 reader
blocks (pairs `[20, 20, 0]`) is `configs/experiments/hsn_t_nomem_top0_w1024_l3.yaml`: 0.7798 /
0.7697 dev bits at 17.6 ms a step against 22.0 ms for 4 blocks, both with CUDA graphs alone on
the RTX 3060; its header has the comparison.
With PR #90's speed settings on (`model.reader_mask: causal`, `train.host_sync: stats`,
`train.fused_adamw: true`, `train.divergence_dump: false`; the user asked for them on 17 Nov) it
is `configs/experiments/hsn_t_nomem_top0_w1024_l3_fast.yaml`: 10.93 ms a step against 17.62, 0.7769
dev bits at 27k (seed 0); its header has what `stats` gives up.

TinyStories (`tinystories_v1`), 9,000 steps, dev bits per byte (lower is better). The figures
are the coordinating session's measurements, quoted as given; they were not re-measured for
this document.

**Shape: reader 160 × 4 blocks, top block unrotated ("top0").** Trained at windows of 256
without the carry:

```
configs/experiments/hsn_t_nomem.yaml
  model.reader_width=160  model.reader_layers=4  model.reader_rotary_pairs=[20,20,20,0]
```

| Model | Parameters | Bits per byte, read cold (seeds) | Compiled step, alone on an RTX 3060 |
| --- | --- | --- | --- |
| **160 × 4, pairs `[20, 20, 20, 0]` (top0)** | 1,887,552 | **0.940 / 0.944** | 15.4 ms |
| 160 × 4, rotation in every block | 1,887,552 | 0.995 | |
| Fast alternative: 192 × 3, pairs `[24, 24, 0]` | 2,026,048 | 0.956 (one seed) | 13.65 ms |
| GRU baseline | 1,977,856 | 0.952 / 0.954 | 15.75 to 16.05 ms |
| Transformer baseline | 1,930,624 | 0.919 / 0.925 | |

**Reading: always with the carry**, `model.reader_carry=256`: a sliding span of 256 bytes
across windows, so no attention distance beyond training is read ("Reading past the window"
above). The carry adds no parameters, so a model trained without it is read with the key set.

- 160 × 4 top0, trained cold, read with the carry: 0.8753 / 0.8777 on all 2,000 dev stories.
- 192 × 3 top0, the same way: 0.900 on 300 dev stories.
- Training with the carry gives 0.8748 (160 × 4 top0): no gain, at × 1.19 the step cost. So
  train at windows of 256 without it and read with it.
- The one exception is the reader alone (`model_name: reader`): 0.8429 trained with the carry,
  against 0.8597 trained without.

**Do not use:**

- Reading past the trained window without the carry or a span: it breaks (3 to 5 bits per byte
  past position 256). Position interpolation, NTK base scaling and a log-length temperature do
  not fix it.
- Top0 or the carry on the 2-block 224 × 2 shape: it does not benefit from either.

**`model.reader_top_words`:** no gain on TinyStories (0.8774), whose stories have a median of
about 720 bytes. Untested on long text.

**Scope:** HSN-T without its memory (`model.memory: false`) only. HSN-T with its memory keeps
its short reader, and refuses the carry by design. The HSN has no reader.

## How to time it

The coordinating session times it alone on the card with its own script. The configuration
of the comparison with the figures given with the request (alone on the RTX 3060, batch 32,
window 256: HSN-T 169 ms per step, of which the reader about 6.5 ms; GRU 16.7 ms and
Transformer 16.2 ms, both uncompiled):

```
configs/experiments/hsn_t_nomem.yaml
  profile=rtx3060  hardware.compile_mode=none  train.batch=32  train.micro_batch=32
```

and for a GRU-sized model the same with, for example, `model.reader_width=224`. The compiled
figure adds `hardware.baseline_compile=reduce-overhead` (branch `claude/nomem-compile`;
`docs/baseline_compile.md`, "HSN-T without its memory").

## Where it is in the code

- `python/hsn/config.py`: the key, its rules (`memory_off_refusals`), the hash registry.
- `python/hsn/model/hsn_t.py`: `HSNTNoMemory`; `HSN` and `HSNT` refuse a memory-off
  configuration.
- `python/hsn/runner.py`: `model_name: hsn_t` builds `HSNT` or, with the memory off,
  `HSNTNoMemory`.
- `python/hsn/train/memory_plan.py`: `memory_off_param_count`.
- `python/hsn/experiments/spec.py`: the family `hsn_t_nomem` and `family_of`; the memory
  key of the experiment database leaves the key out at its default, so the arms from before it
  keep their stored step-time and memory estimates.
- `configs/experiments/hsn_t_nomem.yaml`: the recipe.

## Tests (`tests/test_hsn_t_no_memory.py`)

The key is opt-in and leaves the hashes and the experiment database's memory keys alone;
with its memory the model is built and trained as before; stateless, with no memory module;
logits read no later byte, nothing before a reset, and the context at its own position; a
trained HSN-T's weights load and give its logits with the memory's reads at zero; the
starting weights; the closed form of the parameter count and the three widened sizes; the
refusals; the experiment family; thirty generic-trainer steps lower the loss; the generic
trainer's own controllers run; the forward and backward compile as one graph and match eager.

`tests/test_reader_carry.py` (`model.reader_carry`): the default keeps hashes, memory keys,
arm identities, parameters and outputs; a stream read window by window with the cache equals
the stream read in one pass with a sliding span of `C` (a direct computation and the reader
with `reader_span = C + 1`), at window lengths below, at and above `C`; a document reads
nothing of the one before it; the cache is detached; the refusals; the generic trainer trains,
evaluates and checkpoints with the cache; the carrying window compiles as one graph.
`tests/test_reader_top_words.py` (`model.reader_top_words`): word ends on hand-made bytes
(UTF-8, the newline and tab markers, across a window boundary, after a reset); the default and
0 change nothing; windows with word slots equal a direct computation whose word reads are
chosen by loops (window lengths below and above `C + 1`, with `N` binding and not); word slots
inside the span are masked; no word of an earlier document is read; the slots are detached and
hold the last `N` word ends; refusals; trainer and compiled window as above.

## Limits

- On text, read cold at windows of 256, the recommended shape (160 × 4 top0, 0.940 / 0.944
  bits) is ahead of the GRU baseline (0.952 / 0.954) and behind the Transformer baseline
  (0.919 / 0.925); the shapes of "Text: the gap to the baselines" are behind both. Step times
  other than those quoted in the recommendation are the coordinating session's and are not
  recorded here.
- `hsn run` labels its results by `model_name`, so a memory-off run made with it appears as
  `hsn_t` in the result store; the configuration hash and `model.memory` tell the two apart.
  The experiment harness uses the family.
