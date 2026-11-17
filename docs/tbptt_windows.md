# Gradient segments across windows: `train.tbptt_windows`

16 November 2026. Branch `claude/tbptt-windows`, from main `af21ce3`, merged with main `5c54832`
on 17 November 2026. Written and tested on the CPU (torch 2.15.0+cu132, the card hidden); the
first experiment's card results (local RTX 3060) are under [Results](#results). The design
follows the source audit
`~/hsn-handoff/cross_window/TRAINER_AUDIT.md` ("Proposed option", points 1 to 6). A trainer option:
no model equation changes (cells, graph, units, ε, the convex tanh gate, the router and `q` are
untouched).

## Why

The TBPTT trainer detaches the carried state at every window (P1.7), so no loss in window `t + 1`
ever trains what window `t` wrote into the cells; `train.grad_accum > 1` does not change that,
since every window still starts from `state.detach()` and runs its own backward pass. On
TinyStories, HSN-T with its memory and a whole-window reader learned to carry mostly the context
at the start of a window and few names: carried memory is worth +0.15 bits per byte in later
windows but only +0.18 bits per letter on names 64 to 255 bytes back in an earlier window (+0.45
and +0.76 with the 32-byte reader). Letting the next window's loss reach the writes is the
direct test of that explanation. Raising `train.unroll` is not the same experiment: it also widens
what the reader attends to.

## What the key does

`train.tbptt_windows = K`, a positive integer, default 1.

| | `K = 1` (default) | `K > 1` |
| --- | --- | --- |
| Windows per optimiser step | `train.grad_accum` | `train.grad_accum`, a multiple of `K` |
| State at a window's start | detached | detached at a segment's first window; attached to the window before otherwise |
| Backward passes per step | one per window | one per segment of `K` consecutive windows |
| Reader / model calls | one per window | one per window (no concatenated `K·T` call) |

Inside a segment the trainer carries the returned `h` and `q` with their autograd history;
`age`, `pos` and the address bytes are carried as always (integers, no gradient). The segment's
loss is the sum of its windows' objectives, each weighted exactly as without the key (the task
term by the step's supervised positions, `n_windows · count / step_loss_positions`; every other
term by its mean over the step's windows; a window without a supervised position contributes a
zero task term), divided by `train.grad_accum` and summed. After the segment's backward pass the
state leaves the graph (`state.detach()`), and the next segment starts from it.

Everything per step runs once per step, after all its segments: unscaling, the per-group norms,
clipping, the guards, the optimiser (LazyAdamW's active units are the union over the step's
windows), the update caps, the weight average, `update_target`, the schedules. No parameter
changes while a segment's graph is alive. Checkpoints, evaluations, rollbacks and stops stay at
step boundaries; the checkpoint holds the detached state and the exact loader and address
counters, so a resumed run continues bit for bit (tested).

Details that follow from the above:

* **Document resets** are the model's per-row `masked_fill` (`CellGraphState.reset`): a document
  that starts in one row cuts that row's gradient into the earlier window and no other row's.
  Rows are not reset because another row finished its story.
* **The `q` resync** (`train.q_resync_every`) is computed from the attached `h` inside a segment,
  with autograd (it never ran under `no_grad`); the trainer's `_window_start` now takes
  `detach=False` inside a segment and keeps its counter work.
* **The feedback cuts** (`train.feedback_detach`, `summary, eps, router` in the memory recipe)
  stay what they are. The gradient crosses the window boundary through the paths that are
  differentiable inside a window: with cut C, the cell's read of its state, the selected-cell
  readout `r_t` and the decoder's `q_t`. Routing decisions are not made differentiable; the
  straight-through estimator acts as within a window.
* **Statistics** (the step record, the route guard's counts, the P2.3 features) read every
  window from detached values after its segment's backward pass. The per-term gradient norms of
  a statistics step remain those of the step's first window (eager: `autograd.grad` on its terms
  before the segment's backward pass, the graph retained; compiled: the extra compiled passes
  from the same start state), as without the key.
* **Memory**: a segment holds the activations of `K` windows until its backward pass, about
  `P + K·A` (parameters and optimiser `P`, one window's saved activations `A`) instead of
  `P + A`. The boundary state is small (B32, 64 cells, width 64: `h` 0.5 MiB in fp32).

The key is left out of the configuration hash at its default (`HASH_OMIT_AT_DEFAULT`), so every
existing configuration keeps its hash and every arm its id. The sweep's memory key gains
`grad_accum` and `tbptt_windows` when they are not 1 (they set the step time and the activations
held), which leaves every earlier arm's memory key unchanged.

## Scope

Refused at configuration load (`tbptt_windows_scope_error`), each with the reason:

| Configuration | Why |
| --- | --- |
| `train.grad_accum` not a multiple of `K` | a segment never spans two optimiser steps |
| model on the generic trainer (baselines, `hsn_t` with `model.memory=false`) | their state is detached at every window; the segments are the HSN trainer's |
| `hardware.window=split` / `fused` | the split window's tape and replay end at the window: no gradient into its start state |
| `hardware.compile_mode=reduce-overhead` / `max-autotune` | CUDA-graph replay overwrites what a live segment's backward pass still reads |
| `train.lambda_u > 0` (P2.3 probes) | the scorer's EMA features are updated in place after every window while the segment's graph holds them |

In scope: the HSN trainer's autograd window, eager (`hardware.compile_mode=none`) or compiled
without CUDA graphs (`hardware.compile_mode=default`, below).

Main's loop over events (`model.write_events=word`, with or without `model.word_address`; merged
17 November 2026) is in scope too: the event loop reads the window's start `h` and `q`, so the
gradient crosses the window boundary there as in the loop over bytes (`r` before a row's first
event is zeros, not carried), and the word address's carry is integers, handed on window by
window as before. The reader's carried keys (`model.reader_carry`, `model.reader_top_words`) stay
detached at every window: inside a segment the gradient crosses windows through the memory's
state only.

### `hardware.compile_mode=default`

A new value of the HSN's compile mode: `torch.compile` with inductor's default mode, no CUDA
graphs. The compiled window's blocks are AOTAutograd graphs whose outputs keep their autograd
edges to their inputs, so a window started from an attached state passes its gradient back into
the window before. The compiled window's old refusal of `train.grad_accum > 1` now applies to the
CUDA-graph modes only (on every device, so that a configuration means one thing everywhere);
`default` accumulates. The per-term gradient norms of a compiled statistics step now carry the
task weight under accumulation, as the eager ones always did.

Validated on the CPU: two chained compiled windows (inductor) with a loss on the second window
only equal the eager chain to the P2.8 tolerances (values 8e-6, gradients 1.2e-4, including the
gradient into the first window's start state), and the trainer over two optimiser steps
(`aot_eager`) equals eager at `K = 1` and `K = 2`, with no recompile in the second step. Measured
gap of the raw gradients, compiled against eager at `K = 2`: 2.3e-5 and 4.5e-8 (steps 1 and 2),
against 0.38 and 0.038 between `K = 1` and `K = 2`. **The CUDA step time of `default` is
unmeasured**: without CUDA graphs the window is launched block by block from the host, so it will
be slower than `reduce-overhead`, by an amount nobody has measured. The step-time rule (0.5 s alone
on the card) applies; an eager run (`compile_mode=none`, 0.7 to 1.7 s per window) needs the
user's approval.

## How it was tested

`tests/test_tbptt_windows.py` (CPU):

* the key is opt-in (the hash at `K = 1` is the hash without the key) and refused outside its
  scope; the compiled scope accumulates without CUDA graphs only;
* long documents (48 to 96 bytes in windows of 32) and supervision in the step's last window only:
  an earlier window's final state receives a gradient exactly when it shares the supervised
  window's segment (`K = 1, 2, 3`), and the cell's, reader's and router's gradients change between
  `K = 1` and `K = 2`;
* a document starting in row 0 of the second window (column 0, or column 5 with no supervision
  before it) leaves row 0 of the first window's final state with exactly zero gradient while a
  continuing row keeps one;
* an explicit reference (each window its own model call on the attached state, the same masked
  and weighted losses, one backward pass per segment, then clipping and the optimiser) equals the
  trainer bit for bit over two optimiser steps: raw gradients, the gradient reaching every
  window's final state, the parameters after each step. Cases: `K = 1` (the per-window trainer),
  `K = 2` and `K = 3`, two segments per step, a `q` resync inside a segment, short and long
  documents, aligned documents with padding and windows without a supervised position, and the
  dense HSN with the trailing-padding cut;
* `K = 2` with two segments per step resumes from a checkpoint bit for bit (records, weights,
  optimiser moments, carried state, address bytes);
* the compiled window without CUDA graphs against eager (above);
* after the merge with main: the loop over events (`model.write_events=word`, word address
  `none` and `exact`) passes the second window's loss into the first window's final `h` and `q`
  at `K = 2` and not at `K = 1`, and its compiled form (`aot_eager`) equals eager at `K = 2`.

`K = 1` against main: the same five tiny CPU trainings (HSN-T in the memory recipe's form with
`grad_accum` 1, 2 and 3 with and without aligned documents, the dense HSN with the trailing cut,
and the compiled window under `aot_eager`) on main `af21ce3` and on this branch give bit-identical
step records, raw gradients, parameters, carried states and checkpoint contents. The existing P1.7
tests (every window starts detached) pass unchanged.

## The first experiment

`/home/joe/hsn-experiments/remote/specs/xwin_top0.yaml` (seed 0) and `xwin_top0_s1.yaml` (seed 1,
the same arms): HSN-T with its memory and
the top0 reader (width 160, four blocks, the top block unrotated, the reader over the whole
window), TinyStories in windows of 256, the memory recipe, C-read off, 9,000 steps, seed 0,
`hardware.compile_mode=default`. The fair pair of the audit: both arms take two windows per step
(`train.grad_accum=2`, batch 32), so they see the same tokens and make the same updates; arm A
detaches at every window (`tbptt_windows=1`), arm B joins the two (`tbptt_windows=2`).

| Arm | Run id |
| --- | --- |
| A, detached | `ts_memT_top0_ga2_detached-s0-11bd4dada918`, `ts_memT_top0_ga2_detached-s1-fe578bcc29a1` |
| B, one segment of two windows | `ts_memT_top0_ga2_xwin2-s0-28f4e144d44e`, `ts_memT_top0_ga2_xwin2-s1-2a6e11df1376` |

Each step is twice the tokens of `mem_top0` (one window per step): compare A with B, not either
with `mem_top0` at equal steps.

**CPU timing only** (not a card number; the machine was shared, load average about 13 on 32
cores): the spec's configuration with `profile=cpu` (fp32, no autocast), eager, B32, T256, four
threads, four steps of each arm on TinyStories. Steps 2 and 3 (two windows each): arm A 17.1 and
18.8 s, arm B 17.3 and 17.3 s; step 1 (warm-up) 40.5 and 44.2 s; step 4 (statistics step and
checkpoint) 31.6 and 27.5 s. The process's resident memory sampled at the end: 7.06 GB (A),
10.32 GB (B), about 3.3 GB more for holding the first window's activations through the second
(fp32 on the CPU; the card runs bf16 autocast). So on the CPU the joined segment costs memory, not
arithmetic, as the audit expected. The card's numbers are under Results.

## Results

17 November 2026. The pair above, seeds 0 and 1: TinyStories, 9,000 steps, HSN-T with its memory
and the top0 reader, `train.grad_accum=2`, `hardware.compile_mode=default`, each run alone on the
local RTX 3060. Dev is bits per byte; names is the names test, bits per letter of a name whose
earlier mention lies in an earlier window, 64 to 255 bytes back.

| Arm | Dev (s0 / s1) | Names (s0 / s1) | Step time | Peak memory |
| --- | --- | --- | --- | --- |
| A, `tbptt_windows=1` (cut at every window) | 0.8119 / 0.8118 | 0.75 / 0.73 | 752 ms | 2.96 GB |
| B, `tbptt_windows=2` (gradient across 2 windows) | 0.7977 / 0.8048 | 0.71 / 0.72 | 766 ms | 5.65 GB |

Step time is the s1 run's median (`step_time.run_median_s`); peak memory is
`measured.peak_bytes`, the same in both seeds.

* `K = 2` is better on dev by 0.014 (s0) and 0.007 (s1) bits per byte. Its names gain (0.04 and
  0.01 bits per letter) is within the 95 % intervals.
* Both arms run without CUDA graphs (`compile_mode=default`), hence about 0.75 s per step: over
  the 0.5 s limit, under an exception the user approved for this pair (cap 4.0 s). Joining two
  windows costs about 2 % in step time and about 1.9 times the peak memory.
* Decision (the user, 17 November 2026): `train.tbptt_windows` stays an opt-in research option
  and goes into no recipe.

## Open

* Whether the trained model uses the longer credit for names: the names score moved by less than
  its 95 % intervals in both seeds.
* `K > 2`, and segments longer than one optimiser step (more windows per step, or a carried graph
  across steps), are not tried; the latter is out of scope by construction.
* The split and fused windows and CUDA-graph replay would each need their own segment support
  (terminal-state cotangents and an incoming-state gradient for the tape; segment-level graph
  lifetimes for replay). Not attempted.
