# The per-byte memory step on the RTX 3060: where it goes, and what was removed

18 November 2026. Branch `claude/perbyte-speed`, from main `f7e39be`. Measured on the RTX 3060
(`profile:rtx3060`), alone on the card in a plain CUDA context, torch 2.15.0+cu132, trainer
kernels built. The configuration is the merged recipe
`configs/experiments/hsn_t_memory_carry.yaml` on TinyStories: HSN-T with its per-byte memory,
the top0 reader (160 wide, 4 blocks), `model.reader_carry=256`, B 32, T 256, bf16,
`hardware.compile_mode=reduce-overhead`, `hardware.graph_alias_index=true`.

## What this is

The job: find where the per-byte step spends its time and remove what can be removed without
changing results. No equation changes. A change is on by default only if it computes the same
bits operation by operation; what is equal to rounding only is an opt-in key.

| | Median step, ms | Against main |
| --- | ---: | ---: |
| main (`f7e39be`) | 185.5 | |
| the branch at its defaults | 178.3 | −7.2 (−3.9 %) |
| + `hardware.message_gradients=state` (new, opt-in; equal to rounding) | 171.1 | −14.3 (−7.7 %) |
| + `hardware.round_products=merged` as well (the opt-in of 17 Nov) | 152.6 | −32.9 (−17.7 %) |

Medians of the steady steps of runs of 200 steps in one card slot on a quiet host (slot 2,
"Timings"). The three branch rows were measured at `3911d08`, one change before the final
commit: it still had the router comparisons that the profile then showed to cost 0.5 ms
("What did not pay"). The final commit (`bf96b0d`) measured 180.2 ms in slot 3 on a busier
host, where main measured 189.3 ms. The step moves by 4 to 6 ms with the host's load, so
the device times of the profiles are the firmer figure: 179.7 → 173.1 ms busy at `3911d08`.

What was found: **the step is the card's own work in the 256 positions.** Of a step the card
is busy for about 180 ms; 149 ms of that are the 16 forward and 16 backward replays of the
compiled block, 17.6 ms the reader (eager), and about 13 ms everything else. Inside a
position the unit bank is half (it runs all eight units on every selected pair and reads
one), and the rest is about a hundred small kernels. There is no large waste: what could be
removed without changing an operation's result is 4 %; with the two opt-in keys the step is
18 % shorter.

## Where the step goes

`torch.profiler` (CPU and CUDA activities) over 4 steps of the training loop on main
(`prof/base_prof`). Device times are sums of kernel durations, which do not depend on the
host; the wall time of a profiled step does and is not a step time.

| Part of a step | Device time, ms | Kernels |
| --- | ---: | ---: |
| Loop over bytes, backward: 16 graph replays of 16 positions | 95.7 | 17,312 |
| Loop over bytes, forward: 16 graph replays | 53.6 | 14,768 |
| Backward after the loop: the reader, embeddings, prologue (eager) | 11.6 | 357 |
| Reader with its carry, forward (eager) | 6.05 | 192 |
| Block inputs: the graphs' input copies | 3.56 | 85 |
| Backward between the replays: slice gradients, parameter gradient sums (eager) | 3.19 | 606 |
| Optimiser step, clip, update caps, weight average | 1.89 | 278 |
| Prologue without the reader | 1.33 | 53 |
| Epilogue (decoder) and the backward before the loop | 2.02 | 109 |
| The rest (losses, statistics, snapshot, host-to-device) | 1.2 | |
| **Device busy** | **179.7** | 33,002 |

- The card is idle for 5 to 11 ms of a step (95 to 97 % utilisation by NVML during the
  timing runs): the host between two steps (the optimiser's passes, the loader, the
  snapshot, the eager prologue and reader). The step is sensitive to the host: main
  measured 191.4 ms in slot 1 (other jobs on the CPUs, p10 to p90 188 to 198 ms) and
  185.5 ms in slot 2 (184.4 to 186.9).
- **The loop is 149.3 ms, 83 % of the device time**: 209 µs per position forward, 374 µs
  backward. A replay is one launch (3.35 ms forward, 5.98 ms backward), so it is not bound by
  launches. It is the kernels' own time.
- **The reader is 17.6 ms** forward and backward (4 attention kernels with the carry's mask:
  1.1 ms forward, 3.4 ms backward; the rest products and eager elementwise passes).

Inside a position (µs per position, 256 positions per step):

| Kernel | Forward | Backward |
| --- | ---: | ---: |
| Unit bank, first layer, all 8 units: `[384, 256] × [256, 4096]` | 41.4 | 59.7 + 37.5 |
| Unit bank, activation on `[384, 4096]` (backward with the bias sums) | 19.3 | 50.3 |
| Unit bank, second layer, all 8 units: `[8, 384, 512] × [8, 512, 64]` | 18.0 | 21.0 + 19.5 |
| Unit bank, the chosen unit's gather (backward: zeros and a bf16 scatter-add) | under 2 | 9.2 |
| Unit bank, summing the weight gradients of a block's 16 positions | | 9.0 |
| **Unit bank** | **about 80** | **about 206** |
| 10 small products forward, 17 backward (2 to 8 µs each) | 38 | 68 |
| Messages: the gradient of the edge weights' gather (`index_put`, 13 kernels) | | 26.1 |
| Messages: the gradient of `V h`'s gather (a bf16 scatter-add) | | 13.0 |
| The state's gradient through the gated update and the reset | | 9.2 |
| The scorer's features assembled and cast (`[32, 64, 204]`) | 7.9 | |
| The other kernels (selection, two sorts, gathers, norms, update, statistics) | about 80 | about 50 |

- **The unit bank is about 286 µs of the 583 µs: 73 ms per step**, 38 % of the step. The
  static pass runs every unit on every selected pair and gathers one output in eight; its
  first-layer products are single tensor-core products.
- **39 µs per position, 10 ms per step, are two gradient sums of the messages** that the
  compiler cannot fuse on this card: a bf16 scatter-add needs an atomic add that cards below
  compute capability 9.0 do not have, so inductor hands these to ATen. The edge weights' is
  an `index_put` with `accumulate`, which sorts its indices and checks them: 13 kernels for
  a gradient of 256 numbers.
- A forward position is 58 kernels, a backward position 68.

## The changes

### On by default

Each is the same computation operation by operation (tested bit for bit: "Equality"). On
the card a compiled block whose graph changed is another compile, and its results differ in
the last bits, as between any two compiles ("On the card").

**1. Dense block inputs** (`hsn.train.compile._window_blocks`). −4.2 ms of device time.

- **Before.** A block of 16 positions read the slice `x[:, cols]` of each per-position input
  of the window (`[B, T, …]`: the encoded bytes, the context with the reader's output, the
  scorer's scalars, the resets). A CUDA graph copies a strided input by the byte extent it
  spans, and a slice of 16 columns spans nearly all of it: every block copied the whole
  window, 517 MB a step (3.1 ms). In the backward pass each slice made a window-sized zero
  tensor, wrote its gradient into it and added it to the others (1.6 ms).
- **Now.** The window is copied once with the blocks as the leading dimension
  (`reshape → transpose → contiguous`), and `unbind` hands each block a dense piece: 32 MB a
  step (0.7 ms, and 0.2 ms for the one copy), and one backward node that lays the 16
  gradients side by side.
- **Values.** The same. The gradient is the same bit for bit: every element lies in exactly
  one block, so the old sum was that block's gradient plus exact zeros, which is the
  gradient itself except that `-0.0` became `+0.0`. A gradient hook adds `0.0` to do the
  same. One block (a window of at most 16 positions) is the tensor itself, as before.
- This is what `_block_columns` did for the loop over events on 17 Nov
  (`docs/word_event_speed.md`), without a copy per block.

**2. The chosen unit's gather through `pick_unit`** (`UnitBank.forward_static`). −2.3 ms.
The gradient of the gather in the all-units pass was a zero tensor and a bf16 scatter-add
outside the fused kernels, 9.2 µs per position. `pick_unit(…, as_gather=True)` is the same
gather with a masked copy as its backward; with `as_gather` a `-0.0` of the incoming
gradient is written as the `+0.0` that adding it to a zero gives, so the gradient is the
gather's own bit for bit. (The pick of `round_products=merged` is unchanged: it does not set
`as_gather`.)

**3. No per-position views in the compiled window's prologue.** −0.7 ms of host time.
`_window_prologue` made a view of the encoded bytes, the context and the scalars for every
position (`unbind`: 768 at T 256) for the eager loop over positions. The compiled window
reads blocks and never read them, and the card waits while the host makes them. The compiled
window now passes `steps=False`; the eager loops are unchanged.

### Opt-in: `hardware.message_gradients=state`

Default `autocast`: as it was. Left out of the configuration hash and of the experiment
database's memory key at its default. Refused without the memory's cells, with
`model.routing=dense` and with `hardware.window=split|fused`. It applies to the sparse
round's messages wherever they run: eager or compiled, the loop over bytes and the loop over
events, training and evaluation. −7.2 ms (178.3 → 171.1) on the per-byte recipe.

- **What it changes.** `CellGraph.messages_selected` gathers a row of edge weights per
  selected cell and a row of `V h` per source of a selected cell. Under bf16 autocast both
  gathers read bf16 tensors, so autograd adds up the gradients of rows gathered more than
  once in bf16. With the key the gathers read the edge weights before their cast and `V h`
  through the state's dtype (fp32), and cast what they gathered. The two gradients (of the
  edge weights, and of `V h` into `V` and the state) are then summed in fp32.
- **The forward pass.** A cast and a gather commute, so in the eager round the messages are
  the same bit for bit (tested under CPU autocast; the card test of it has not passed yet,
  "On the card"). A compiled block with the key is another compile, and like any two
  compiles the two agree to rounding, not to the bit (first-step loss 5.5417647 with the
  key, 5.5417786 without). One place where they can differ: the key's cast of the gathered
  edge weights sits inside a fused kernel, and a fused kernel may hold a bf16 value in fp32
  until it stores it.
- **Why it is faster.** fp32 sums fuse into the compiled kernels (inductor's own scatter);
  the bf16 ones are the 39 µs per position above.
- **Why it is opt-in.** It is not the same computation under autocast: the sums round once,
  in fp32, instead of at every addition in bf16. The code keeps the bf16 form on purpose in
  one place (`hsn.kernels.ops.csr_messages`: "notably bf16 edge-weight gradients"), so the
  default stays. The fp32 sum is the more exact one, and it is what
  `CellGraph._padded_weights_as` already does across positions ("sums them in the weights'
  own dtype").
- Without autocast (fp32 and fp64 runs, the CPU) the dtypes agree and the key changes
  nothing: the same computation, tested bit for bit.

### `hardware.round_products=merged` on the per-byte recipe

The opt-in of 17 Nov (`docs/word_event_speed.md`) had been timed on word events only. On the
per-byte recipe: 170.4 ms against 191.4 ms on main in slot 1 (−21.0 ms, −11 %), and with
`message_gradients=state` 152.6 ms against 185.5 ms in slot 2. It is the larger of the two
keys because it takes the activation and the second layer of seven of the eight units out
of every position (the first layer stays).

## Timings

Runs through the real `TBPTTTrainer` (evaluation and checkpoints stubbed), one process per
run, alone on the card. The number of a run is the median step time of its steady steps:
without the compile steps and the first 20, and without statistics steps.

Slot 2 (13:36 to 13:53, 200 steps a run, `train.stats_every` beyond the run, 177 steps in
the median):

| Configuration | Tree | Median, ms | p10 to p90 | Load at start |
| --- | --- | ---: | --- | ---: |
| main | `f7e39be` | 185.47 | 184.41 to 186.85 | 7.0 |
| branch, defaults | `3911d08` | 178.29 | 177.89 to 178.85 | 3.2 |
| branch, `message_gradients=state` | `3911d08` | 171.13 | 170.81 to 171.74 | 1.3 |
| branch, `message_gradients=state`, `round_products=merged` | `3911d08` | 152.61 | 152.30 to 153.36 | 2.1 |

Slot 1 (12:29 to 12:45, 340 steps a run, 314 in the median, other jobs on the CPUs):

| Configuration | Tree | Median, ms | p10 to p90 | Load at start |
| --- | --- | ---: | --- | ---: |
| main | `f7e39be` | 191.42 | 187.94 to 197.52 | 7.1 |
| main, `round_products=merged` | `f7e39be` | 170.40 | 168.07 to 176.61 | 6.6 |

Slot 3 (14:03 to 14:21; other jobs on the CPUs; the final commit):

| Configuration | Tree | Median, ms | p10 to p90 | Steps in the median | Load at start, at end |
| --- | --- | ---: | --- | ---: | --- |
| branch, defaults | `bf96b0d` | 180.17 | 179.24 to 182.84 | 177 | 12.4, 5.9 |
| main | `f7e39be` | 189.34 | 188.31 to 196.54 | 37 | 24.3, 11.9 |

- The two slot 3 runs are not a pair: main ran behind the card tests' compiles, under twice
  the load. Slot 2 is the comparison on a quiet host; slot 3 says that the final commit
  runs and is in the same place. A paired timing of the final commit on a quiet host, and
  the two keys on it, are not measured (`slot4.sh` beside the tools would do it).
- Every run: 1 dynamo graph, 0 recompiles. Peak allocated 2.87 to 2.91 GiB (1.92 to 1.94
  with the merged products).
- Device time by the two profiles (`prof/base_prof` on main, `prof/new_prof` on `3911d08`):
  179.7 → 173.1 ms busy. Block input copies 3.56 → 0.73 ms (and 0.2 ms for the window's
  one copy); the eager backward between the replays 3.19 → 1.64 ms; the backward replays
  95.7 → 92.8 ms (the pick); the forward replays 53.6 → 54.1 ms (the router comparisons
  that were taken back, "What did not pay").

## Equality

- **Fixed seeds, the real trainer, CPU** (`equiv.py`, the recipe at B 4 in fp32, 6
  optimiser steps with statistics steps): main against the branch at its defaults, eager
  (`compile_mode=none`) and compiled (`aot_eager`). The SHA-256 of every raw gradient of
  every step (83 tensors, before clipping), the losses and gradient norms, the 85 parameters
  after the last step, the carried state and the reader's cache: identical. The same with
  `hardware.message_gradients=state` on the compiled window (fp32: the key changes nothing).
- **The blocks** (`tests/test_perbyte_speed.py`): values, layout and the gradient against
  the slices bit for bit, a `-0.0` in a block's gradient included; the compiled window on
  the blocks against the compiled window on the slices over five consecutive windows of
  HSN-T with the carried reader (per byte, and with the attention read), on `aot_eager` and
  on inductor: every output, the final state and every parameter gradient bit for bit; six
  trainer steps: the same records, raw gradients and weights.
- **The pick**: `pick_unit(as_gather=True)` against `gather` in fp32, bf16 and fp64 (value
  and gradient bit for bit, with `-0.0` in the incoming gradient); the static unit pass
  against the all-units pass with a gather (every gradient bit for bit).
- **`message_gradients=state`**: without autocast the messages and the three gradients bit
  for bit (fp32 and fp64), and the compiled window with the key against the one without it
  bit for bit; under bf16 autocast (eager) the messages bit for bit, the gradients within
  bf16 rounding of the default's, and the edge weights' gradient against the exact sum of
  the same addends: within 1e-5 (fp32 rounding) with the key and no further from it than
  the default.

### On the card

A compiled block is not reproducible to the bit across compiles: inside a fused kernel bf16
values are held in fp32 and rounded where the kernel stores them, so where the compiler
draws the kernels' boundaries decides where a value is rounded. A change of the block's
graph, however small, is another compile. What the timing runs recorded (bf16, the recipe's
sizes, CUDA graphs; the first 40 losses and gradient norms of every run):

| Runs compared | First-step loss | Largest relative loss difference, steps 1 to 8 | Steps 1 to 40 | Gradient norm, relative: step 1; largest of steps 1 to 40 |
| --- | --- | ---: | ---: | --- |
| main against main (one compile; slots 2 and 3) | equal (5.541769981) | 4.1e-5 | 2.1e-4 | 5.9e-6; 1.0e-2 |
| main against the final branch (`bf96b0d`), slot 2's main | 5.541769981 / 5.541777134 | 3.2e-5 | 3.4e-4 | 6.8e-5; 2.3e-2 |
| main against the final branch, slot 3's main | 5.541769981 / 5.541777134 | 1.6e-5 | 2.3e-4 | 7.4e-5; 1.3e-2 |
| main against the branch at `3911d08` | 5.541769981 / 5.541778564 | 4.4e-5 | 1.2e-3 | 9.7e-5; 9.3e-2 |
| `3911d08` against itself with `message_gradients=state` | 5.541778564 / 5.541764736 | 2.7e-5 | 1.4e-3 | 6.9e-5; 1.0e-1 |
| `3911d08` against itself with both keys | 5.541778564 / 5.541805267 | 4.1e-5 | 1.3e-3 | 6.1e-4; 9.0e-2 |

- Two runs of main from one compile start with the same loss, but only 8 of their 83
  first-step gradient tensors are the same bits (the bf16 scatter-adds of the default take
  their additions in no fixed order), and they are 4e-5 apart by step 8 and 2e-4 by step
  40.
- The final branch differs from main in the first step by 1.3e-6 of the loss and 7e-5 of
  the gradient norm (another compile), and then stays as close to main as main stays to
  itself (2.3e-4 and 3.4e-4 over 40 steps against 2.1e-4).
- The runs at `3911d08` (with the router comparisons, and with the keys) drift further
  from main by step 40 (1.2e-3 to 1.4e-3 of the loss). All are between 2.913 and 2.916
  after 40 steps (main 2.9135 and 2.9138, the final branch 2.9143).
- **Card tests** (slot 3, `pytest -n 0 -m cuda` on the final commit, 24 of the 32 selected
  tests ran before the slot's time limit): 23 passed, 1 failed, 8 did not run.
  - Passed: all 12 of `test_p2_8_compile.py` (the compiled window with CUDA Graphs against
    the eager one in fp32 with exact selections, dense in bf16, the launch counts, the dense
    symmetry, a statistics step), the blocks and the pick against the slices and the gather
    in fp32 and bf16, twelve compiled training steps under bf16 with
    `message_gradients=state` alone and beside the merged products, the four compiled
    training runs with the carried reader, three of the four merged-round comparisons.
  - Failed: `test_state_gradients_on_the_card_keep_the_eager_messages`, a test of this
    branch. The run was cut before pytest printed the failure. The test asserted that the
    messages are bf16; CUDA autocast takes their sum in fp32
    (`hsn.kernels.native.csr_messages`), so that assertion was wrong for the card. It now
    compares the dtypes of the two forms. **Not rerun on the card.**
  - Did not run: `test_merged_products_equal_eager_cuda[events_eval]`,
    `test_merged_pass_under_bf16_on_the_card`, the four card tests of
    `test_p2_5_sparse_path.py`, and the device tests of `test_p2_1_router.py` and
    `test_p2_2_selection_grads.py`.

## Tests

- **CPU suite** (the whole suite, GPU hidden, at the final code): 2,612 passed, 519 skipped
  (CUDA, MPS and kernel-build tests), 118 subtests passed.
- **Card** (slot 3): 23 of the 24 CUDA tests that ran passed; one test of this branch failed
  on an assertion that was wrong for the card and is corrected but not rerun; 8 selected
  tests did not run ("On the card").
- `ruff check`, `ruff format --check` and `mypy --strict python/hsn`: clean.
- Not run: the rest of the CUDA suite (the split and fused windows, the four-op kernels).
  The code they run is touched in one place, `UnitBank.forward_static`, whose value is the
  gather's as before; what changed is its gradient, which their forward sweeps (no
  autograd) do not take.

## What did not pay, and what was not built

- **Comparisons in the router** (commit `79b6bc1`, taken back in `168362a`).
  `Router._selected` counts the valid slots per cell with an int64 `scatter_add` and
  `F.one_hot` in `Router._finish` checks its input's range; a compiled window runs the first
  as an ATen kernel and the second as a kernel of its own at every position. Comparisons
  against `arange` give the same booleans and the same `0.0 / 1.0` and took one kernel a
  position out of the forward replays (14,768 → 14,512), but the comparison fused into a
  costlier reduction: the forward replays went from 53.6 to 54.1 ms. The router is main's.
- **Replacing the other int64 scatter of a position** (`unit_full` in `Router._finish`) by a
  comparison: the split window's replay calls `_finish` on all `B·T` rows at once, where the
  comparison is a `[B·T, K, N]` intermediate. Not built.
- **Per-position statistics once per block** (the state's largest value, the probe rates):
  about 1 ms a step; the mean of a float over the batch may round in another order when it
  is taken for 16 positions in one kernel. Not built.

Proposals, with what the profile and the component timings (`timing/micro1.json`) say they
are worth. None is built.

| Proposal | Expected | What it needs |
| --- | ---: | --- |
| Compile the reader (with its carry) into the window's graph tree | −4.7 ms | Measured as a component: 17.7 ms eager, 13.0 ms compiled, forward and backward. Its output differs from the eager reader's at bf16 rounding (0.03 on values up to 4.6), so it is an opt-in key. The cache has to leave graph memory each window, and evaluation and the statistics passes need their variants |
| The unit bank's first-layer weight gradient once per block | −7 to −8 ms | One product on the 16 positions' stacked inputs measures 32.9 µs per position against 62.5 µs for 16 products. It needs a hand-written backward for the block (the split window's replay, for the weights only), and it rounds a block's sum once in bf16 where today each position's product is rounded and the sum is fp32 |
| The chosen unit only in the first layer | at most −20 ms | 7 of 8 units' products are never read (36 ms a step, forward and backward). Grouped products are no faster on this card (a perfectly balanced split of the pairs measures 23.6 µs against 45.4 µs for all units, and real splits are not balanced; `docs/word_event_speed.md` measured the same). It needs a grouped tensor-core kernel: the fused memory kernel |
| A fused memory kernel for HSN-T (the split and fused windows of P2.10 with the state gradient, a reader and events) | 1.5 to 1.8 times on the loop | What P2.10c measured on the HSN at this shape (131 against 202 ms). Later, by the user's priorities |
| The next window's loader read, snapshot and host-to-device copy before the gradient-norm read | −2 to −3 ms | They run while the card is idle between two steps. The snapshot and the checkpoint read the loader's state, so their order has to be kept exactly |
| `torch._inductor.config.aggressive_fusion`, `max_autotune_gemm` | unknown | Not measured: each needs a cold compile on the card. They change how kernels are fused and which product kernels run: equal to rounding at best |

## Where it is in the code

- `python/hsn/train/compile.py`: `_window_blocks`, `_plus_zero`, the loop of
  `CompiledWindow._run`; `apply_message_gradients`.
- `python/hsn/model/units.py`: `pick_unit(as_gather=…)`, `UnitBank.forward_static`.
- `python/hsn/model/hsn.py`: `_window_prologue(steps=…)`.
- `python/hsn/model/graph.py`: `CellGraph.state_gradients`, `messages_selected`.
- `python/hsn/config.py`: `hardware.message_gradients`, `message_gradients_scope_error`,
  `HASH_OMIT_AT_DEFAULT`. `python/hsn/experiments/spec.py`: the memory key leaves the key
  out at `autocast`. `python/hsn/train/tbptt.py`, `python/hsn/runner.py`: the key is applied
  to the model the trainer and the profile pass run.
- `tests/test_perbyte_speed.py`.
- Measurement tools, traces and results (not in the repository):
  `~/hsn-experiments/perbyte-speed/` (`tools/prof_pb.py`, `tools/prof_pb2.py`,
  `tools/analyze_pb.py`, `tools/equiv.py`, `tools/compare_equiv.py`, `tools/micro1.py`,
  `tools/op_census.py`, `tools/slot2_report.py`, `slot1.sh` to `slot4.sh`,
  `prof/`, `timing/`, `equiv/`, `logs/`).
