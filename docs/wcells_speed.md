# The HSN write of the keyed word cells: where a step goes, and what was sped up

25 November 2026. Branch `claude/wcells-speed`, from main `846ecd3`. Measured on the local RTX 3060
(`profile:rtx3060`), alone on the card (the card protocol's `BUSY_speed`), torch 2.15.0+cu132, the
kernels built from main's sources.

The configuration is the keyed word cells' HSN write as it was trained on 23 November
(`docs/amendments/word_cells.md`, "The 27k-window run"):
`configs/experiments/hsn_t_memory_words_carry_bal_qk.yaml` with `model.word_cells=hash
model.word_cells_value=word`, `hardware.compile_mode=default train.grad_accum=3
train.tbptt_windows=3`, the fused event block (`hardware.event_kernel=fused`) and the CUDA arm's
lines, B 32, T 256, bf16. A step is three windows and one backward pass over them.

**Result.** Three changes that are on by default and bit-identical take the step from 159.3 to
150.8 ms on code native and from 162.9 to 154.2 ms on bio prose (5.3 %). A fourth behind an
opt-in key, `hardware.reader_compile=default` (the reader compiled; the same function, rounded
differently), takes it to **134.6 and 138.4 ms (15 % in all)**. No accuracy run has been made
with the key. Details under "What was changed".

**Phase 1 in short: bit for bit there is little to gain.** Four fifths of the step is the
card's own work on the model's arithmetic.

- The step is **159.9 ms on code native and 162.8 ms on bio prose** at the start of training
  (the 27k run's own median on code native is 162.1 ms), 53 to 54 ms a window.
- **The card is busy for 126 of those 160 ms (79 %).** That time is the arithmetic of the
  model: the reader's attention and linear layers (52 ms, forward and backward), the fused
  event block's kernels (40 ms), and about 5,000 small kernels of everything else (35 ms).
  None of it is waste that a change of how the work is issued removes.
- **The card is idle for 34 ms a step (21 %)**, while the host issues work kernel by kernel.
  A third of that is the keyed write (its forward and its backward pass); the rest is spread
  over a dozen places of 1 to 3 ms each.
- With `compile_mode=default` and the fused block **no compiled graph runs in a training
  window at all**: the prologue (reader), the epilogue (decoder, keyed cells) and the objective
  are eager autograd, the event loop is the kernels' own CUDA graphs. That is why the table arm
  ran equally fast with and without CUDA graphs on the compiled window: the compiled window is
  only a statistics step's term passes and the evaluation.
- The HSN write's cost over the table is **7.9 ms a window** (53.3 against 45.4): 4.6 ms
  for `value=word` (the write's forward pass, with and without the cross-window gradient) and
  3.3 ms more than the table for `tbptt_windows=3` (3.7 ms against the table's 0.4: the write's
  backward pass). Both are host time: the card works 2.4 ms a window on the write, forward and
  backward together.

Phase 2 (below, "What was changed") therefore went after the idle time first, and then after
the one part of the card's work that a compile can shorten, the reader's small kernels.

## How it was measured

`~/hsn-experiments/wcells-speed/prof.py` runs the repository's own trainer (`TBPTTTrainer.fit`:
loader, window, objective, lazy AdamW, update limit, weight average, route guard, a statistics
step every 100) from scratch in one process, in phases. Statistics steps and the steps in which a
compiled window compiled or the kernels captured a CUDA graph are left out of every number.

| Phase | Steps | What it gives |
|---|---|---|
| warm-up | 40 | nothing |
| plain | 200 | the step as it trains: `StepRecord.seconds`, nothing patched (wrappers that do nothing in this phase cost 0.3 ms: 159.9 ms with them, 159.6 ms without) |
| waits | 60 | a wait for the card at every section's boundary, so that a section's time holds its device work. The sections add up to more than a plain step, in which host and card overlap |
| profiler | 12 to 16 | `torch.profiler` (CPU and CUDA): every kernel's time on the card with the section that launched it; a backward kernel goes to the forward section that made its autograd node (by the node's sequence number); the card's busy time is the union of the kernels' intervals; an idle gap goes to what the host was doing in it |

The profiler's own cost is on the host only (a profiled step takes 228 ms): its kernel times and
counts are used, its host times are not, and the idle gaps are read as shares.

## The grid: where a window's time goes

Code native, milliseconds per window (a step is three), from the phase with waits. `word` and
`bytes` are `model.word_cells_value`; 3 and 1 are `train.tbptt_windows` (all with
`train.grad_accum=3`).

| Part | word, 3 | word, 1 | bytes, 3 | bytes, 1 | word, 3, bio prose |
|---|---:|---:|---:|---:|---:|
| Loader, windows to the card, the loop's own work | 2.40 | 2.41 | 2.17 | 2.09 | 2.42 |
| Reader, forward | 6.80 | 6.77 | 6.77 | 6.76 | 6.76 |
| Prologue beside it: embeddings, `E`, word address, scorer scalars, event slots | 4.83 | 4.79 | 4.66 | 4.65 | 4.76 |
| Keyed cells by rule: keys, read plan, write plan | 2.68 | 2.66 | 2.60 | 2.58 | 2.64 |
| Fused block: merged constants, weights, loads | 1.12 | 1.10 | 1.08 | 1.05 | 1.09 |
| Fused block: forward sweep | 6.54 | 6.63 | 6.51 | 6.62 | 6.95 |
| Fused block: `S` replay, the node, `P`, records | 1.24 | 1.21 | 1.10 | 1.07 | 1.21 |
| Epilogue: latest `q` / `r`, `S` between events, decoder, window statistics | 1.98 | 1.96 | 2.06 | 1.95 | 1.96 |
| Keyed cells: the read (`U`, the feed) | 0.76 | 0.73 | 0.40 | 0.38 | 0.74 |
| **Keyed cells: the write** | **4.97** | **4.81** | 1.21 | 1.19 | **4.94** |
| Objective (task, prediction, balance) | 1.18 | 1.17 | 1.17 | 1.16 | 1.17 |
| Backward: the fused block's reverse sweep | 3.61 | 3.67 | 3.59 | 3.68 | 3.84 |
| Backward: the fused block's parameters | 2.13 | 2.10 | 2.06 | 2.11 | 2.27 |
| **Backward: everything else (autograd)** | **19.75** | 16.45 | 16.30 | 16.46 | **19.65** |
| Statistics, route guard | 1.03 | 1.05 | 0.97 | 1.01 | 1.01 |
| Gradient norm and clip, update limit, optimiser, weight average | 1.41 | 1.35 | 1.35 | 1.30 | 1.39 |
| Sum, a wait for the card at every boundary | 62.4 | 58.9 | 54.0 | 54.1 | 62.8 |
| **The step as it trains (median), per window** | **53.3** | **49.6** | **45.4** | **45.0** | **54.3** |
| The step as it trains (median; p10 to p90), ms | 159.9 (155.6 to 163.3) | 148.9 (146.4 to 152.0) | 136.2 (134.1 to 138.3) | 135.0 (132.9 to 137.6) | 162.8 (159.8 to 166.0) |

- **`value=word` costs 4.6 ms a window** without the cross-window gradient (word, 1 against
  bytes, 1: 49.6 against 45.0): the write (4.8 against 1.2 ms), its feed and read (0.35 ms).
- **`tbptt_windows=3` costs 3.7 ms a window with the HSN write and 0.4 ms with the table.** The
  three sets of buffers of the fused block cost nothing per step. What costs is the write's
  backward pass: "everything else" grows from 16.5 to 19.75 ms, and the card works 0.9 ms of
  those 3.3 (below).
- The reader, the fused block and the rest are the same in all four arms.
- **Capture steps are not in these numbers.** In the first 240 steps 63 of 200 steps captured a
  CUDA graph with three windows a segment (23 with one): the mean with them is 178.6 ms
  against 159.7 ms without. They thin out: the 27k run on code native had 468 captures in
  9,000 steps, about 67 s of its 1,561 s (4 %).

## The card: busy 79 %, and with what

From the profiler, steps without a capture (14 each), milliseconds per step of three windows.

| Device time, ms per step (three windows) | word, 3 | bytes, 3 | word, 3, bio prose |
|---|---:|---:|---:|
| Reader, forward | 18.3 | 19.1 | 18.8 |
| Reader, backward | 33.5 | 33.3 | 33.5 |
| Fused block, forward sweep (its CUDA graph) | 22.2 | 21.9 | 24.8 |
| Fused block, reverse sweep and parameters' backward | 15.0 | 15.4 | 15.8 |
| Fused block, the rest (constants, weights, `S`, `P`, records; forward and backward) | 2.6 | 2.5 | 2.9 |
| Keyed cells: write, forward | 4.4 | 1.5 | 4.7 |
| Keyed cells: write, backward | 2.8 | 0.0 | 2.6 |
| Keyed cells: by rule (keys, plans) | 3.1 | 3.2 | 3.4 |
| Keyed cells: read and feed, forward and backward | 3.0 | 1.3 | 3.0 |
| Prologue beside the reader, forward and backward | 8.2 | 8.6 | 8.8 |
| Epilogue and decoder, forward and backward | 7.7 | 7.7 | 8.0 |
| Objective, forward and backward | 3.0 | 2.8 | 2.7 |
| Everything else (optimiser, norms, statistics, loop) | 2.9 | 2.8 | 2.8 |
| **The card busy** | **126.4** | **119.6** | **131.3** |
| The step as it trains (median) | 159.9 | 136.2 | 162.8 |
| **The card idle** | **33.5** (21 %) | **16.5** (12 %) | **31.6** (19 %) |
| Kernels per step | 9,306 | 7,862 | 9,466 |

- **The reader is 41 % of the card's time** (51.8 of 126.4 ms): four blocks over 256 positions
  that read 512 keys (the window's and the 256 carried). Its largest kernels: the attention's
  backward 10.1 ms (0.84 ms a block and window; the masked reader runs PyTorch's
  memory-efficient attention kernel, not the flash kernel), the linear layers' backward 9.7 ms
  and forward 3.9 ms, the attention's forward 3.3 ms, the layer norms 4.9 ms. About 18 ms is
  some 1,100 small kernels (rotations, casts, concatenations of the carried keys, residual
  sums, GELU) of 8 to 45 µs each.
- **The fused block is 31 %** (39.8 ms): its two sweeps and the parameters' backward are the
  kernels of `docs/wev_fused_event.md`, at that document's times (129 µs and 67 µs a slot).
- **The keyed write is 7.2 ms of device time a step** (2.4 ms a window, forward and backward)
  against the 24 ms a step that it costs: it is about 350 small kernels forward and 180
  backward a window, issued one by one, with eleven host reads.
- The other 28 ms are spread over embeddings, the scorer's scalars for every byte and cell
  (3 ms: `[B, T, N, 11]`, of which the event loop reads the event slots), the decoder, the
  objective and the optimiser.

## The idle time: 34 ms a step, and where

The card is idle when the host has not issued the next kernel yet. The profiler gives the gaps
with what the host was doing in each (shares; the profiler slows the host, so its gap times are
not the plain step's).

| Where the host was: share of the card's idle time (and of its ms a step) | word, 3 | bytes, 3 | word, 3, bio prose |
|---|---:|---:|---:|
| The keyed write, forward | 21 % (7.1) | 0 % (0.1) | 21 % (6.5) |
| The keyed write, backward | 13 % (4.2) | 0 % (0.0) | 11 % (3.4) |
| The keyed cells by rule (keys, plans), read and feed | 8 % (2.7) | 13 % (2.2) | 9 % (2.8) |
| The word address, scalars, embeddings, event slots | 13 % (4.3) | 20 % (3.3) | 13 % (4.2) |
| The reader, forward and backward | 3 % (0.9) | 4 % (0.7) | 3 % (0.9) |
| The fused block: constants, weights, the sweeps' loads, the node | 14 % (4.7) | 22 % (3.6) | 14 % (4.5) |
| The epilogue and the decoder | 1 % (0.2) | 1 % (0.2) | 1 % (0.2) |
| The objective, forward and backward | 7 % (2.4) | 12 % (1.9) | 7 % (2.3) |
| Gradient norm, update limit, optimiser, weight average | 5 % (1.7) | 8 % (1.3) | 5 % (1.7) |
| Statistics, route guard | 3 % (1.0) | 5 % (0.8) | 3 % (1.0) |
| Loader, windows to the card, the loop, the backward pass outside a node | 12 % (4.1) | 15 % (2.4) | 13 % (4.0) |

The milliseconds are the plain step's idle time (33.5, 16.5 and 31.6 ms) split by the profiler's
shares. They overstate the sections that are many small kernels behind the reader's forward
pass (the rule, the word address): in a plain step the card still has the reader's work there.

- **Host reads** (an operation whose result's size or value the host needs waits for the card
  to finish everything issued before it): 33 a step in the keyed write (11 a window: the
  requests' count, three masks in each of its three turns, the writes' count, the unit pass),
  27 in the objective (masks of the loss positions and of the live slots), 9 in the
  objective's backward pass, 3 for the event slots, 3 for the fused block's live slots, 1 for
  the gradient norm.
- A host read is not a loss by itself: while the host waits, the card works. It is a loss
  where host work that does not depend on it is issued behind it, with the card idle. The
  keyed write is the clearest case. The host reaches its first host read 2 to 3 ms after it
  started the event loop's forward sweep (7 ms on the card), waits for the sweep, and then
  issues the whole write with the card idle between its small kernels.

## What this says for phase 2

| Lever | What it could give | Taken |
|---|---|---|
| The idle 34 ms: the keyed write without host reads in its turns, its requests ahead | part of its 7 ms | yes (change 1) |
| The idle 34 ms: fewer kernels and autograd nodes in the write's unit pass | part of the write's forward and backward | yes (change 2) |
| The idle 34 ms: the fused block's constants before the window's host reads | about 1 ms a window | yes (change 3) |
| The card's 126 ms: the reader's 1,100 small kernels | a compiled reader fuses them; it rounds differently | yes, behind a key (change 4) |
| The card's 126 ms: the reader's attention and linear layers, the fused block's kernels | nothing without another kernel or other shapes | no ("What is left") |
| The rest of the idle time | 1 ms here and there | no ("What is left") |

## What was changed

Four changes, each measured alone on the card before and after, the trees alternated in one hold
(`prof.py`, 200 plain steps after 40, the median of the steps without a capture; two runs each,
both given). The HSN write, three windows a step.

| | Code native, ms a step | Bio prose, ms a step | On by default |
|---|---:|---:|---|
| main `846ecd3` | 159.5, 159.2 | 163.0, 162.8 | |
| 1. The write: requests ahead, turns without a host read (`1faaaf2`) | 155.7, 155.8 | | yes, bit-identical |
| 2. The unit pass: one split, operands cast once (`3ba15d4`) | 153.6, 153.6 | | yes, bit-identical |
| 3. The block's constants before the window's host reads (`d083a33`) | 151.0, 151.1, 150.8 | 154.5 | yes, bit-identical |
| The head with the key at its default (`d32a805`; a later hold) | 150.7, 150.4 | 153.7, 154.6 | |
| 4. `hardware.reader_compile=default` on the head | **134.3, 134.9** | **138.5, 138.3** | no: opt-in, rounds differently |

- **Default-on: 159.3 → 150.8 ms on code native (8.5 ms, 5.3 %) and 162.9 → 154.2 ms on bio
  prose (8.7 ms, 5.3 %)**: 53.1 → 50.3 ms a window. The three changes give 3.6, 2.2 and 2.6 ms.
- **With the opt-in key: 159.3 → 134.6 ms on code native (24.7 ms, 15.5 %) and 162.9 → 138.4 ms
  on bio prose (24.5 ms, 15.0 %)**: 44.9 and 46.1 ms a window, below the table's 45.4 ms a
  window on code native as it was measured on main (the table arm would gain from the key
  too; not measured).
- Main in the later hold: 158.9 ms on code native (one run), as in the first.
- p10 to p90 of a run is within 4 ms of its median; two runs of a tree agree within 0.4 ms
  on code native.
- Steps that capture a CUDA graph are not in these medians, before or after (63 of the 200
  steps of each run; the mean with them is 178.2 → 170.0 ms).
- Peak allocated memory is unchanged (3,001 → 3,016 MiB over these 240 steps).

### Where the step goes now

Code native, the profile of phase 1 again on the head (`prof.py`; 8 profiled steps without a
capture each, 14 for main). Milliseconds per step of three windows; per window with the waits,
the card's own time in brackets.

| | main `846ecd3` | The head, key at its default | The head, `reader_compile=default` |
|---|---:|---:|---:|
| The step as it trains (median) | 159.9 | 151.0 | 134.2 |
| The card busy | 126.4 | 126.0 | 108.1 |
| The card idle | 33.5 | 25.0 | 26.1 |
| Kernels per step | 9,306 | 9,003 | 8,320 |
| Host reads in the keyed write, per step | 33 | 3 (and 3 with the event slots) | 3 (and 3) |
| The keyed write's kernels per step, forward / backward | 1,050 / 539 | 708 / 439 | 705 / 438 |
| The reader's device time, forward and backward | 51.8 | 51.4 | 34.9 |
| The reader's kernels per step | 1,530 | 1,500 | 861 |
| Reader forward, per window | 6.80 (6.08) | 6.76 (6.19) | 4.20 (3.66) |
| Keyed write, per window | 4.97 (1.47) | 3.80 (1.07) | 3.71 (1.05) |
| Backward outside the fused block, per window | 19.75 (15.09) | 18.95 (14.70) | 15.74 (11.62) |

- The three default-on changes leave the card's work as it was (126.4 → 126.0 ms) and take
  8.5 ms of its idle time (33.5 → 25.0 ms).
- The key takes 17 ms of the card's work, all of it in the reader (51.4 → 34.9 ms: its
  attention and linear layers stay, 640 of its 1,500 kernels are fused away).
- What is left of the idle time is where it was, in smaller shares: the write (its cell
  update behind its one host read, and its backward pass) a quarter, the by-rule integer work
  and the objective most of the rest.

### 1. The keyed write: its requests ahead, its turns without a host read

`WordCells.write_learned` (`model.word_cells_value = word | cell`).

- **Before.** The write found its requests (one per key of a row) by a mask, whose row count is a
  host read: the first thing the write did was to wait for the event loop's forward sweep. Then
  each of its three turns took three more masks (`target[asks]`, `slot[asks]`,
  `taken[target[won]]`), nine host reads, each a round trip with the card idle.
- **Now.** The requests are by rule (the write plan), so they are taken before the event loop
  runs (`WordCells.requests`, called by `HSN._event_slots`, beside the event slots' own host
  read). In the turns a request that does not ask, or does not win, is sent to a spare entry
  behind the rows' cells in place of being left out by a mask: `scatter_reduce` and
  `index_fill` over all requests, no host read. The write issues its scores and its three turns
  while the sweep still runs and reads the card once, for which requests write (the cell
  update runs on those rows alone, as before, so every product has the shape it had).
- **Values.** Integers by the same rule; every floating operation, its operands, its shape and
  the order of the autograd nodes are the write's of main.
- **Measured.** 159.3 → 155.7 ms (3.6 ms).

### 2. The grouped unit pass: one split, the operands cast once

`UnitBank.forward_selected`, which the write runs on its rows.

- **Before.** A slice of the sorted rows per present unit, whose backward fills a `[P, 4d]`
  tensor of zeros that the next is added to; and under autocast every `addmm` cast its three
  operands: five casts per present unit forward and five cast nodes backward.
- **Now.** One `split` (its backward is one concatenation), and under CUDA autocast the bank's
  four parameters and the sorted input are cast once before they are taken apart.
- **Values.** The same products on the same operands. The gradients are the same bit for bit:
  a cast's backward commutes with the zeros that `unbind`'s and `split`'s backward fill in,
  every row is in one block, and `index_select`'s backward (which follows, unchanged) adds the
  rows to zeros either way.
- **Measured.** 155.7 → 153.6 ms (2.2 ms). With change 1 the write's kernels fall from 350 to
  236 a window forward and from 180 to 146 backward.

### 3. The fused block's constants before the window's host reads

`CompiledWindow._run_events`, `event_fused.prepare_block`, `FusedEvents.prepare`.

- **Before.** A fused window took its event slots first. Their count is a host read: the host
  waits until the card has finished the prologue (the reader's forward pass, 6 ms). Only then
  did it issue the block's merged constants, the weights the kernels read and the cell's
  constants, about 70 small kernels a window, with the card idle.
- **Now.** They depend on the parameters alone and are issued before the event slots, while the
  card still runs the reader. The write's requests (change 1) are taken with the event slots,
  behind them.
- **Values.** The same operations. The constants' autograd nodes are made earlier in the
  window; they share no leaf with the event slots' gathers, so no gradient is summed in another
  order (tested: the trainer with the window in both orders, bit for bit).
- **Measured.** 153.6 → 151.0 ms (2.6 ms).

### 4. `hardware.reader_compile=default` (opt-in)

`hsn.train.reader_compile`, `hsn.config.reader_compile_scope_error`.

- **What it does.** The carried reader's read of a training window (`Reader.read`: the same
  Python) runs as one `torch.compile` graph in inductor's default mode, its backward compiled
  with it. No CUDA graphs, so `train.grad_accum` and `train.tbptt_windows` are what they are
  without the key: a segment's three windows are three calls of one compiled function. The
  compiled window hands the model's prologue the read to call (`_window_prologue(read=...)`);
  nothing is installed on the model.
- **Which windows.** Training windows with a gradient, the statistics steps' term passes among
  them. Evaluation and every read without a gradient keep the eager reader, so a run's
  evaluations are computed as without the key.
- **Why a key.** Inductor fuses the reader's elementwise chains (rotations, casts, residual
  sums, GELU, norms) into kernels of its own, which round differently from the eager kernels:
  equal to rounding, not bit for bit. The first step's loss is 5.554753 with the key and
  5.554749 without. At `none` (the default) nothing is built, the prologue calls the reader as
  before, and the key is out of the configuration hash, the arm view and the memory key.
- **Scope.** HSN-T with its memory, `model.reader_carry > 0`, a compiled window
  (`hardware.compile_mode` not `none`), `hardware.window=autograd`. Refused with the reason:
  `train.reach_dropout`, the byte-content options (`train.content_dropout`,
  `model.zero_unseen_bytes`, `model.unseen_bytes_as`), MPS, a model without the carried
  reader.
- **What a run records.** `measured["reader_compiled_windows"]` and
  `measured["reader_compile_seconds"]`; a step in which the reader compiled is a compile step
  (not in `train_seconds`). The compile takes about 27 s at the first step.
- **Measured.** 150.5 → 134.6 ms on code native (15.9 ms, 10.6 %) and 154.1 → 138.4 ms on bio
  prose (15.7 ms, 10.2 %); two runs each, alternated with the key off in one hold. Peak
  allocated memory 3,018 → 3,002 MiB.
- **No accuracy run has been made with it.** It is the same function rounded differently, as
  the compiled window is against the eager one; whether a 27k run reaches the same scores is
  for a run to say.

## Bit for bit

Two runs of one configuration are not bit-identical on the card as it is normally run
(`docs/p2_10b_fused_speed.md`, `docs/wev_fused_event.md`): here two runs of main differ in
the first step's gradient norm and in the second step's loss. With deterministic algorithms
(`train.deterministic=true`) they are, and there the comparison was made, as in P2.10b
(`~/hsn-experiments/wcells-speed/bits.py`): the HSN write on code native, B 32, three windows a
step, from scratch; every step's loss, total, gradient norm, accuracy, loss parts and statistics
as hex floats, and SHA-256 of the final weights, the optimiser's state, the weight average, the
carried state and the carry (the keyed cells, the reader's cache, the address).

| Compared with main `846ecd3` (40 steps, code native) | Result |
|---|---|
| main again | identical in everything |
| change 1 (`1faaaf2`) | identical in everything |
| changes 1 and 2 (`3ba15d4`) | identical in everything |
| changes 1 to 3 (`d083a33`) | identical in everything |
| the head with `hardware.reader_compile` at its default (`d32a805`) | identical in everything |
| the head, 110 steps (a statistics step with its term passes among them) | identical in everything (main and the head, 110 steps each) |
| the head, bio prose, 40 steps | identical in everything (main and the head) |

Beside the runs, per change (`tests/test_wcells_speed.py`):

- the write against the write of `846ecd3` (kept in the test file as it was): the cells, keys,
  write positions, records and every gradient (the parameters', the carried cells', `u`'s and
  `eps`'s) over six carried windows, compared as bit patterns: float64 on the CPU for the five
  learned arms with and without the gate and the router's cut; on the card in fp32 and under
  bf16 autocast for three arms;
- the unit pass against the pass of `846ecd3`: outputs and gradients as bit patterns, CPU
  (float64, fp32) and card (fp32, bf16 autocast);
- the trainer on the fused block's reference (float64) with the window in the new order and in
  the order of before: losses, raw gradients, parameters and statistics with a tolerance of
  zero.

## Tests

| What | Result |
|---|---|
| The whole suite on the CPU at the head `a3d8fa7` (card hidden, `-n 10`, thread caps, nice) | 3,205 passed, 573 skipped (CUDA, MPS and kernel-build tests), 118 subtests passed |
| `mypy --strict python/hsn`; `ruff check`; `ruff format --check` on the changed files | clean (125 source files; 13 files) |
| `tests/test_wcells_speed.py` on the CPU | 27 passed (the write, the unit pass and the window's order against main's, bit for bit; the requests) |
| `tests/test_reader_compile.py` on the CPU | 10 passed (the key, its refusals, the compiled read against the eager one, the trainer with and without it) |
| On the card at the head `a3d8fa7`: every test with `cuda` in its name of `test_wev_fused_event`, `test_wev_fused_trainer`, `test_word_cells`, `test_tbptt_windows`, `test_p1_2_modules`, `test_p2_5_sparse_path`, `test_reader_compile`, `test_wcells_speed` | 44 passed |
| On the card at `d32a805`: `-m cuda` of `test_reader_compile` and `test_wcells_speed` | 9 passed |
| On the card at `d083a33` (changes 1 to 3): `-m cuda` of `test_wcells_speed`, `test_word_cells` (the recipe's steps on both event loops, every arm) and `test_wev_fused_trainer` | 21 passed |

The suite at `d083a33`: 3,195 passed, 572 skipped. At `d32a805` one test failed, the one that
holds the list of keys left out of the hash at their defaults in order (the new key stood in
the middle; `a3d8fa7` puts it at the end, as every key before it).

## What was tried and did not help, or was not taken

| What | Why not |
|---|---|
| Making `reduce-overhead` work under `tbptt_windows=3` | Nothing to gain: with the fused block no compiled graph runs in a training window (phase 1). |
| Rotating the carried keys without autograd (the reader rotates 512 keys of which 256 are constants) | Worked out, not built: the backward saves half of ten kernels' elements per block, the forward gains seven kernels per block. Net about zero. |
| Caching the block's weights (`event_weights`) per step: the three windows of a step read the same weights | Superseded by change 3, which issues them where the card is busy anyway. |
| The reader's cache in bf16 (it is cast to fp32 when kept and back when read: 16 casts and two copies of 21 MB a window, exact both ways) | About 1.3 ms a step of device time, bit-identical in the reader's output; it changes the dtype of a checkpoint's cache and of a compiled baseline's input. Not for a speed branch to decide. |
| The scorer's scalars at the event slots only (they are built for every byte and cell, 3 ms a step, and read at the event slots) | The address features are matrix products whose shapes would change: not bit-identical. |
| Shared row indices for the objective's masks (27 host reads a step) | About 1 ms a step at most (the card is idle there with or without them); not built. |

## What is left

In the order of what it could give. Estimates, not measurements, except where said.

1. **The host reads after the event loop.** The write's cell update (after its one host read),
   the unit pass's histogram and the objective's masks are issued with the card idle: about
   4 ms a window. Fixed shapes (padded requests, every unit on every pair) would let all of
   it be issued while the sweep runs, and then the next window's reader could be read ahead
   behind it. Padded shapes change the matrix products' shapes, so rounding: an opt-in, and a
   rewrite of the write and of the objective's masking. Perhaps 8 to 10 ms a step.
2. **The reader's attention.** The masked reader runs PyTorch's memory-efficient attention
   kernel (0.27 ms forward, 0.84 ms backward a block and window: 13.4 ms a step). Attention
   as two matrix products and a softmax inside the compiled reader may be faster in the
   backward pass; unmeasured, rounds differently.
3. **The capture steps.** 468 captures in a 9,000-step run on code native, about 67 s (4 % of
   the run, outside the step medians). Each capture runs its sweep twice eagerly before it
   records (`capture_forward(warm_up=2)`, `ReverseSweep.capture`). After a set's first capture
   the warm-ups compute nothing new (the same kernels and product shapes); dropping them is
   bit-identical and saves about half of a capture. Not touched: one unexplained
   `cudaErrorStreamCaptureInvalidated` is on record for these captures
   (`docs/amendments/word_cells.md`), and the warm-ups are the cautious side of it.
4. **The epilogue compiled** (decoder, `S` between events, `P`, the keyed read): 8 ms of device
   time a step, mostly four linear layers; a compiled graph would save little of it, and its
   shapes follow the event slots.
5. The by-rule integer work (word address, keyed cells' plans: 360 small kernels a window)
   could be about 100 kernels less; it runs behind the reader's forward pass, where the card is
   the limit, so each kernel less is about 5 µs: 1 to 2 ms a step.

## Files

- `python/hsn/model/word_cells.py`: `WriteRequests`, `WordCells.requests`, the turns of
  `write_learned`. `python/hsn/model/hsn.py`: `_Events.requests`, `_event_slots`,
  `_event_epilogue`, `_window_prologue(read=...)`.
- `python/hsn/model/units.py`: `UnitBank.forward_selected`.
- `python/hsn/train/event_fused.py`: `prepare_block`, `FusedEvents.prepare`, `block(prepared=)`.
  `python/hsn/train/compile.py`: `CompiledWindow._run_events`, `CompiledWindow.reader`,
  `compile_window(reader=)`.
- `python/hsn/train/reader_compile.py` (new): `CompiledReader`, `reader_of`.
  `python/hsn/config.py`: `hardware.reader_compile`, `reader_compile_scope_error`, the hash.
  `python/hsn/experiments/spec.py`: the memory key. `python/hsn/train/tbptt.py`,
  `python/hsn/runner.py`: the trainer's and the profile pass's compiled reader, `measured`.
- `tests/test_wcells_speed.py`, `tests/test_reader_compile.py`.
- Outside the repository, `~/hsn-experiments/wcells-speed/`: `prof.py` (step times, waits,
  profiler), `bits.py` (the bit records and their comparison), `tables.py`, `timing.py`,
  `card.sh` / `env.sh` / `snap.sh` / `job*.sh` (the card protocol, the measured trees as
  archives of their commits under `trees/`), the records under `prof/` and `bits/`, the logs
  under `logs/`.
