# A fused word-event block: phase 1 (design, reference, kernels, measurement)

21 November 2026. Branch `claude/wev-fused-event`, from main `d3626df`. Measured on the RTX 3060
(`profile:rtx3060`), alone on the card in a plain CUDA context, torch 2.15.0+cu132. The
configuration is the field recipe, `configs/experiments/hsn_t_memory_words_carry.yaml` (HSN-T,
word events, exact word address, top0 reader 160 × 4 with carry 256) with its CUDA arm line
(`hardware.compile_mode=reduce-overhead`, `graph_alias_index`, `message_gradients=state`,
`round_products=merged`), bio prose, B 32, T 256, bf16.

**Nothing here reaches a training run.** There is no configuration key and no call site in the
trainer: the block is a function (`hsn.train.event_fused.forward_events`) that tests and the
measurement scripts call. Phase 2 (the key `hardware.event_kernel=fused`, the trainer, runs)
waits for a decision.

## What this is

The word-event memory runs `HSN._position` at every event slot with autograd: about 108
kernels per slot, forward and backward, in the compiled window. The plan
(`~/hsn-handoff/SPEED-WEV-2026-11-20.md`) asked whether one fused round, a few kernels in
place of those, removes at least half of the 27.8 ms that the scorer, the messages and the
selected cells take of a 56.1 ms step, and set the bar: **go on only if the step projects at
least 14 ms faster.**

**Result: not yet.** One training window's forward and backward pass, the compiled autograd
window (the current path) against the same window with the fused block, at the run's
step-27,000 weights on real bio prose windows:

| Event slots in the window | Compiled window | Fused block | Saved | Share of bio prose windows |
| --- | ---: | ---: | ---: | ---: |
| 48 | 50.9 ms | 41.0 ms | **9.9 ms** | 51 % |
| 64 | 61.0 ms | 47.0 ms | **14.0 ms** | 48 % |
| 80 | | | 18.1 ms (extrapolated) | 1 % |
| **The data's mix** | | | **11.9 ms** | |

- The rest of a step (optimiser, clip, loader) is the same in both, so the step projects from
  56.1 ms to **about 44 ms: 11.9 ms faster, 43 % of the 27.8 ms. The bar was 14 ms.** It is
  met on windows of 64 slots and missed on windows of 48.
- The gain is in the backward pass: 108 kernels per slot become about 15, and one pass per
  block for the parameters' gradients. The forward pass gains little: its two large products
  are the compiled round's own, and a small kernel between two products costs 8 to 30 µs
  however little it computes.
- What is built is correct ("Equality"):
  - float64: the block equals the autograd event loop to rounding, every parameter gradient
    and the start state's included.
  - fp32 on the card: the same decisions as the autograd loop at the recipe's shape and at
    the trained weights; gradients within 3e-7 over all parameters, the worst tensor 1.1e-6.
  - bf16: the block is as far from the eager loop and from the compiled window as those two
    are from each other.
- What would move it, with what each rests on, is in "What phase 2 would take": the listed
  trims add up to an estimated 4 to 6 ms more. None of them is built or measured.

## The design

The layout is the split window's (P2.10, P2.10c): a forward sweep without autograd, one
autograd node, a reverse sweep for the state. What is new is that nothing of the cell or of
the scorer is replayed with autograd: the forward sweep writes a tape, and the parameters'
gradients are taken from the tape by hand, for all slots at once.

| Part | What runs | Where |
| --- | --- | --- |
| Prologue, epilogue | the model's own (`_window_prologue`, `_event_slots`, `_event_epilogue`): embeddings, the reader, the word address, the decoder | autograd, unchanged |
| **Forward sweep** | per slot: reset, front (`P`, `S` on the carried summary), scorer, top-K decision with refresh probes, messages, cell input and norm, unit bank, candidate, gate, masked update, read-out | no autograd: three kernels and four matrix products per slot, a tape |
| `S` | replayed once for all slots on the taped summary (a constant under the `summary` cut) | autograd, one batched pass |
| **Node** (`EventRecurrence`) | returns the taped summaries before and after every slot, the read-outs, the logits and the final state | one autograd node |
| `P`, decoder, balance term | read the node's outputs | autograd, unchanged |
| **Backward, the state** | P2.10c's reverse sweep (`ReverseSweep`: three kernels and two transposed products per slot), unchanged except that its middle step also keeps `dhid` | kernels |
| **Backward, the parameters** (`local_backward`) | from the total `dL/dc`, `dL/db` of all slots: the unit bank's weight gradients as one product per unit and layer over the whole block, the norms, `V`, the edge weights, the scorer and its straight-through path, `du`, `dε` | no autograd: matrix products and one kernel |

### What the node passes

- **In (with gradient):** `u` (the replayed `S`), the cell's error read `z − ẑ` (the prediction
  detached: the `eps` cut), the start state `h`, `q`, and eleven parameter tensors: the unit
  bank's merged layouts `w1 [4d, M·H + d]`, `b1`, `w2 [H, M·d]`, `b2`
  (`UnitBank.merged_layout`: the gate rides the first, `W_out` is folded into the second;
  autograd takes them on to the masters), the padded edge weights, the two norms' gains, `V`,
  the scorer's weight and bias and the address gate's weight.
- **Out:** `q` before every slot (after its reset: what `P` reads) and after it, the selected
  read-out, the logits (the balance term's input), the state after the block.
- **The start state is not assumed detached.** The reverse sweep ends with the cotangent of
  the state the block started from and the node returns it, so `train.tbptt_windows > 1`
  carries gradient across windows through the event loop (tested with two windows).

### The backward pass by hand

With `x = [u; ε; h_i; m_i]`, `v = g_n ⊙ x · r`, `hid = v W1_j + b1_j`, `y = act(hid) W2'_j +
b2'_j` (`W_out` folded), `c = tanh(α · g_c ⊙ y · ρ)`, `b = σ(v G + b_g)`, `h' = (1 − b) h + b c`:

1. **State** (per slot, in reverse; `hsn.train.state_gradient`): `A`, `R` the cotangents of `h`
   and `q`; `g = A[cell] + R/N`, `dc = b g`, `db = (c − h) g`, the cell's read of its own state
   and of its sources' through the transposed layers. Unchanged code, already tested.
2. **Parameters** (once per block, `P = B·E·K` rows):
   - `dz = dc (1 − c²)`; `dα = Σ dz ⊙ pre`; the candidate norm's backward gives `dy`;
     `dg = db · b (1 − b)`.
   - The rows are sorted by their unit (one read of `M` counts from the device). Per unit
     `j`: `∂W2'_j = actᵀ dy`, `∂W1_j = vᵀ dhid`, `dv = dhid W1_jᵀ`; the gate's columns from
     `vᵀ dg`. `dhid` is the reverse sweep's own (it computes it at every slot and now keeps
     it), `act` and `v` are the forward sweep's operands.
   - The cell norm's backward gives `du`, `dε` (summed over a row's slots), `dm` and the
     gain's gradient (one kernel, `hsn_ev_pairs`, a block per (row, slot)).
   - Messages: `∂V = dmᵀ (Σ_e a_e h_src)`, `∂a_e = (Vᵀ dm) · h_src`.
   - The straight-through path into the scores of the written slots, in fp32 and in
     autograd's own order of operations (the router computes its scores in fp32 whatever the
     logits' dtype): the cell mask from `α` and from the read-out's weight, through the
     sigmoid, to the cell's largest score (shared evenly where scores are equal, as `amax`'s
     backward shares it); the unit weight from `α`, through the softmax, to the executed
     unit's score alone (`train.unit_credit=executed`). Then the scorer's weight, bias and
     the address gate's weight from the logits' cotangent (the balance term's is added).

### The forward sweep's kernels (`kernels/cuda/hsn_event.cu`)

One block per batch row; everything a row computes between two matrix products depends on
that row alone.

| Launch | What |
| --- | --- |
| `hsn_ev_finish` | finishes slot `t − 1` (candidate norm, `tanh`, gated update, summary, ages, read-out, `max |h|`), then the reset of slot `t` and the front's hidden layer on the carried summary |
| two products | `[u; ẑ] = act_f W2'ᵀ + b2'` (`[32, 1024] × [1024, 128]`), and the state's: `h [Vᵀ \| W_hᵀ]` (`[2048, 64] × [64, 72]`), `V h` of every cell and the state's term of every logit |
| `hsn_ev_decide` | the rest of the scorer's logits (a thread per logit), the decision (`hsn_win_select`'s steps: stable top-K, budget prefix, threshold, refresh probes), the straight-through values, the messages, the cell input and its norm |
| product | `first = v W1 + b1`: every unit's first layer and the gate, `[384, 256] × [256, 4160]` |
| `hsn_ev_pick` | the chosen unit's hidden row, its activation, the gate |
| product | `out = act W2' + b2'`: `[384, 512] × [512, 512]` |

The part of `[z; ctx]` in the front's first layer does not depend on the state: it is one
product per block for all slots (`[B·E, 288] × [288, 1024]`).

The tape is the buffers themselves (the reverse sweep's own record buffers among them): no
copy between the forward sweep and the backward pass.

**Rounding under bf16.** Where the compiled round under autocast holds a bf16 value (what a
matrix product reads and returns) the kernels round to bf16 at the same place; the state, the
norms, the gate's sigmoid and the candidate stay fp32, as there. The weights are cast per
window, as autocast casts them at every use. Where it is not the same sum:

- The front's first layer rounds once more: its part of `[z; ctx]` and its part of the
  summary are two sums.
- The state's product runs in fp32 on operands rounded to bf16 (so that the state's term of
  a logit stays a partial sum of the scorer's one product, rounded once with the rest): its
  `V h` is rounded where the messages read it. The same products as a bf16 product, summed
  in another order.
- In the backward pass the weights' gradients are one product per unit over the block where
  autograd adds a product per slot, and the scorer's largest product runs in bf16 once.

**What keeps a kernel short** (measured here, `card_phase.py`). The first version was one
kernel between the products with the scorer, the messages and the front as plain loops over
rows of the weights: **435 µs** at the recipe's shape. None of it was arithmetic:

| Cause | What it cost | What replaced it |
| --- | --- | --- |
| Shared-memory banks: a row stride of `d = 64` words puts a warp's 32 lanes on one bank at every step | the messages alone 207 µs | a warp's lanes read consecutive words or one word (weights staged transposed) |
| Uncoalesced reads of global memory: each lane its own row | the front's two layers 150 µs | weights laid out with the lane index last |
| A dot product in a kernel is a chain of steps on one warp, about 0.1 µs a step whatever the layout | 122 µs with the two above fixed (front's second layer 18, messages 31, scorer 7) | no kernel multiplies a matrix a product can: the front's second layer and the state's product are cuBLAS; sums a kernel keeps are split over 16 lanes and folded by shuffles |

After that: `hsn_ev_finish` 24 µs, `hsn_ev_decide` 30 µs, `hsn_ev_pick` 8 µs.

## What the scope is (`event_fused.scope_error`)

The word-event recipe's round, each refused by name otherwise: `model.write_events` on,
`model.routing=topk`, the straight-through gate (a training step: evaluation keeps the model's
loop), `train.unit_credit=executed`, `train.feedback_detach=[summary, eps, router]` (the state
gradient), both norms `rms`, the linear scorer without excluded features, A5/A9 addresses,
A11, A12's gate on `u` or on the summary (A13), `r_max = 1`, the selected read-out. Not built:
exploration (`train.epsilon_*`), dense-fallback rows (`train.p_dense`), the per-row gradient
limit, fast cells (A8), the write gate (A10), the attention read of the cells, `cost` in
`train.aux_enabled` (the expected cost leaves the block without a gradient).

The kernels add: `n_units` a power of two, `N·M ≤ 512`, an in-degree of at most 8, fp32 state.

## Equality

**float64, CPU** (`tests/test_wev_fused_event.py`; the six-cell model with the recipe's
memory keys: word events with the exact address, the state gradient, executed credit, both
norms, A9, A11, the summary-gated A12, refresh probes; a reset inside the window, a row that
starts a document, padding rows; the objective reads the logits, the prediction, the balance
term and the final state):

| Comparison | Result |
| --- | --- |
| The window through the fused block against `HSN.forward`'s event loop, the recipe's keys | loss identical; logits within 2e-15; every parameter gradient and the start state's within 1e-15 of its largest entry; integer and boolean records exact |
| The same with each key varied (14 cases: the gate on `u`, no gate, one address byte, no probes, every cell selected, one cell, sigmoid edges, `silu` / `relu` / `tanh`, a temperature of 1.7, the byte address, five units) | values 1e-11, gradients 1e-9 (the tests' bounds) |
| The form the card runs: the buffered reverse sweep that keeps `dhid` (5 cases) | the same bounds |
| Two windows with the state carried attached and one backward pass (`train.tbptt_windows > 1`; 3 cases) | gradients 1e-9; with the carried state detached the gradient differs (the check can fail) |
| Equal scores in a cell (all of the scorer's weights at zero) | the cell mask's gradient is shared among the equal units as `amax`'s backward shares it; the same bounds |
| A unit no slot ran | exact zero gradient |

The straight-through path is fp32 in the model whatever the logits' dtype (`Router.scores`).
The block computes it in fp32 too, in autograd's order of operations and from the router's
own expressions on the router's own shapes: a vectorised `exp` rounds an element by its
place in the tensor, and computing the sigmoid on the 12 selected cells instead of all 64
moved two of 26 entries by one fp32 unit.

**fp32, on the card** (`card1.py`, `eq_r4.json`; B 32, T 256, a real bio prose window of 64
slots, 24,576 slot entries):

| | Random weights (perturbed by 0.05) | The run's step-27,000 weights, the stream's own state |
| --- | --- | --- |
| Decisions, kernels against the autograd loop | identical | identical |
| Decisions slot by slot from the kernels' own states | 1,222 of 1,222 live rows | 1,231 of 1,231 |
| The tape, kernels against the PyTorch reference | within 2.2e-6 of each entry's largest value | within 5.1e-6 |
| Parameter gradients, kernels against the autograd loop (relative L2, all) | 2.1e-7 | 2.5e-7 |
| Worst tensor (the start state's gradient is among those compared) | edge logits 6.4e-7 | edge logits 1.1e-6 |

One caveat, found on a smaller shape: two cells whose scores agree to fp32 rounding (1.596004
twice) are chosen by the last bit of a sum, and the kernels sum in another order than the
model. One of 77 live rows went the other way there. The card tests take inputs whose every
decision has a margin of 1e-3 (checked).

**bf16, on the card.** Under bf16 the logits are quantised and top-12 of 64 has near-ties in
most rows, so two implementations that round differently choose a different last cell in
part of the rows, and a window's trajectories then part. That is true of the two forms the
repository has: the eager loop and the compiled window. Measured at the trained weights
over ten consecutive windows of the stream (`card3.py --trained`, `abt_r7.json`; relative
L2 of the gradients, means, the range in brackets):

| Pair | Slot entries that differ | All parameters | `W1` | `W2` | Scorer | Decoder | Logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager loop against compiled window (the two existing forms) | 43 % | 1.2 % (0.8 to 2.1) | 11.5 % | 8.3 % | 3.0 % | 0.72 % | 1.2 % |
| Eager loop against fused block | 43 % | 1.5 % (0.7 to 3.2) | 11.6 % | 8.3 % | 3.0 % | 0.77 % | 1.2 % |
| Compiled window against fused block | 46 % | 1.4 % (0.5 to 2.9) | 11.8 % | 8.5 % | 3.0 % | 0.75 % | 1.3 % |

- **The fused block is as far from each existing form as they are from each other**, tensor
  by tensor. Over all parameters its mean is 1.4 to 1.5 % against 1.2 %, inside the spread
  of ten windows; ten windows do not say whether that difference is real.
- One round from the same state, kernels against the model's eager round: the whole
  selection (12 cells and their units) is the same in 87 % of the live rows at the trained
  weights (53 % at random weights, where the scores lie closer), the logits within one or
  two bf16 steps, the summary within 0.002.
- A loss of 20.97 in all three forms (20.9689, 20.9710, 20.9654).
- **This is one window, not a run.** Whether training on the block reaches the same scores
  is a question for phase 2's runs: the trajectories differ from the first step, as two
  compiles of the current path do.

## Measurement

All on the RTX 3060 alone (the card protocol's `BUSY_fused`; four holds between 15:25 and
16:53, 28 minutes in all; host load 0.4 to 1.3). The main clone's venv, the kernels built in
this worktree (`scripts/build_window_kernels.py --arch 86`).

### The window against the compiled window (`card3.py --trained`)

One training window's forward and backward pass (no optimiser), same model object, same
inputs, bf16 autocast: the compiled window as the trainer builds it
(`compile_window(..., merged=True)`, CUDA graphs, `graph_alias_index`,
`message_gradients=state`) against `event_fused.forward_events` with the kernels and their
CUDA graphs. The run's step-27,000 weights
(`fields_27k_night3/f27_bio_prose_wevbal_top0-s0`, the recipe with the balance term), real
windows of the bio prose training stream with the state, the address carry and the reader's
cache that the stream carries into them. Medians of 80 passes each, alternated in tens; two
runs.

| Slots | Compiled, wall | Fused, wall | Saved | Compiled, device | Fused, device | Kernels |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 48 | 50.93, 50.95 | 41.01, 41.08 | 9.93, 9.86 | 47.18 | 38.70, 38.80 | 6,207 → 2,133 |
| 64 | 61.10, 60.91 | 47.15, 46.84 | 13.95, 14.07 | 55.37 | 43.43, 42.93 | 7,936 → 2,366 |

- p10 to p90 of a run: within 0.8 ms of its median.
- At random weights (perturbed by 0.05; `ab_r4.json`, before the last change to the local
  backward): 9.5 and 13.5 ms.
- The compiled window's 16 more slots cost 8.2 ms of device time (511 µs a slot, 108
  kernels); the fused block's cost 4.4 ms (about 280 µs a slot, 14.6 kernels; 4.7 and 4.1 in
  the two runs).
- **Slots per window on bio prose** (`slots2.py`: the first 400 training windows by the
  model's own word rule; the slots are a multiple of 16 at or above the batch's largest event
  count): 48 in 203 windows, 64 in 195, 80 in 2. A row has 35.6 events on average and the
  batch's largest count is 49.2, against 56.0 slots: 27 % of a 48-slot window's (row, slot)
  pairs and 43 % of a 64-slot window's are padding rows, computed and discarded in both
  forms.
- **Projection.** (203 × 9.9 + 195 × 14.0 + 2 × 18.1) / 400 = **11.9 ms** a step, from 56.1 to
  about 44.2 ms. The L1 profile's 27.8 ms (scorer, messages, selected cells) is this data's
  mix; 11.9 ms is 43 % of it.
- Peak allocated memory of the fused block alone: 0.57 GB at 64 slots (the model, the tape
  and the sweep's buffers; `parts_r5_64.json`).

### The block's parts (`card2.py`, 64 slots; 48 slots in brackets)

| Part | Time | Per slot |
| --- | ---: | ---: |
| Forward sweep, as a CUDA graph | 8.27 ms (6.23) | 129 µs |
| Forward sweep, plain launches | 9.02 ms, of which 5.8 ms is the host issuing them | |
| Reverse sweep with its loads (P2.10c's, a CUDA graph) | 4.78 ms (3.76) | 67 µs and 0.5 ms a block |
| Local backward | 6.0 ms (4.9) measured, and about 5.4 (4.5) after the last change, which removed a 0.57 ms kernel (0.42); 4.5 ms of it is the host | |
| Block inputs and weights into the buffers | 0.38 ms | |

Inside a forward slot (µs): the first product `[384, 256] × [256, 4160]` 42.1; `hsn_ev_decide`
29.8; `hsn_ev_finish` 24.2; the second product `[384, 512] × [512, 512]` 12.5; `hsn_ev_pick`
8.0; the state's product 7.7 and its split-K reduction 2.0; the front's second layer 4.6.
Inside a reverse slot: the two transposed products 23.5 and 12.3 (and 3.5), the three kernels
14.1, 7.6 and 7.5.

Inside the local backward (245 kernels, 5.7 ms of device time at 64 slots): the matrix
products 1.7 ms (0.57 of it a batched contraction of 2,048 small matrices for the address
gate, since replaced by a product and a sum); `hsn_ev_pairs` 0.49 ms; three gathers of the
block's rows in unit order 0.39 ms; the rest is about 200 small tensor operations. The host
takes 4.5 ms to issue them, nearly as long as the card takes to run them.

- The products with their bias added by the product (`addmm`) are faster than without
  (`mm`, the kernels adding it): 8.27 against 9.08 ms. The block uses `addmm`, which is also
  the compiled round's rounding.

### What it says

| | Compiled window | Fused block | |
| --- | ---: | ---: | --- |
| Forward, per slot | about 180 µs | 129 µs | the two large products are the same 55 µs in both |
| Backward, per slot | about 330 µs | 67 µs, and 5.4 ms a block of 64 slots | |
| Per 16 slots, device | 8.2 ms | 4.4 ms | |
| Fixed per window | | about 2 ms more | the replay of `S`, the weights' layouts and casts, the buffers' loads |

(The compiled window's split between forward and backward is `docs/word_event_speed.md`'s,
scaled to today's 511 µs a slot; the fused block's is measured here.)

## What phase 2 would take

### To train with it (the integration)

| What | Where | Size |
| --- | --- | --- |
| The key `hardware.event_kernel = autograd \| fused` (off by default, out of the hash at its default, a scope function with the refusals above) | `python/hsn/config.py`, `python/hsn/experiments/spec.py` | about 120 lines |
| The call site: `CompiledWindow._run_events` runs `fused_event_block` on a training step with the straight-through gate; evaluation, the soft warm-up and the statistics steps' term passes keep the compiled window | `python/hsn/train/compile.py`, `python/hsn/train/tbptt.py` | about 150 lines |
| One set of buffers per window of a segment (`train.tbptt_windows > 1`: a window's tape must outlive the next window's forward pass) | `python/hsn/kernels/event.py` | about 60 lines |
| Tests: the trainer in float64 on the CPU against the autograd window over some steps; the card tests of the key; a compile check (graphs per number of slots) | `tests/` | about 500 lines |
| Timing and accuracy runs: the recipe on bio prose at 9,000 steps, fused against the current path, two seeds each; then one 27,000-step run | the card | about 2 hours |

About 800 lines beyond this branch.

### To reach the bar (kernel work; estimates, each resting on a number measured here)

| Trim | Estimate at the data's mix | Rests on |
| --- | ---: | --- |
| The local backward's remaining tensor operations as kernels (the scorer's path, the messages, the rows written in unit order by the sweep instead of gathered) | 1.5 to 2 ms | 5.5 ms now, of which the products are 1.5 and `hsn_ev_pairs` 0.5; the host issues 245 kernels |
| Only the slots that hold an event: the exact number of slots instead of a multiple of 16 (49.2 against 56.0 on average), and a slot's live rows only (rows sorted by their event count) | 1.2 to 1.8 ms | 27 to 43 % of the (row, slot) pairs are padding; the kernels' time does not fall with fewer rows, the products' does; needs launches from the extension instead of one CUDA graph per number of slots |
| The first product on the columns that differ per slot: `u` and `ε` are one row per batch row, so their half of `[384, 256] × [256, 4160]` is `[32, 128] × [128, 4160]` | 0.7 ms | 42 µs a slot; rounds as two sums (opt-in on its own) |
| The fixed part: `S` taken from the sweep's tape instead of replayed, the layouts cached | 0.5 to 1 ms | about 2 ms more than the compiled window per window |
| The two small kernels: the front's first layer as a product, the scorer's staging | 0.5 to 0.8 ms | `hsn_ev_finish` 24 µs of which 15 the front; `hsn_ev_decide` 30 µs of which 8 staging |

Together an estimated 4.4 to 6.3 ms, which would put the step 16 to 18 ms faster. None of
these is built, and my estimates for this work have been high twice (the per-byte fused
window, and this block's forward pass, which I put at 70 to 100 µs a slot and measured at
129).

### Risks

1. **bf16 trajectories.** The block rounds differently from the compiled window, as any
   other compile does; one window says it is no further away than the compiled window is
   from the eager loop. Only runs say whether scores hold.
2. **The reverse sweep's kernel changed** (`hsn_win_back_mid` writes `dhid` when asked). Off
   by default; the HSN's split and fused windows' card tests pass with it (123 tests).
3. **The tape is the buffers.** One block's tape is valid until the next forward sweep of
   the same shape; the trainer's statistics steps (one extra pass per loss term) and
   `tbptt_windows` need their own handling.
4. **Scope.** Exploration, dense-fallback rows, the cost term and the per-row gradient limit
   are refused, not built; a recipe that turns one on trains on the compiled window.
5. **A second hand-written derivative of the cell** beside the reverse sweep's. It is tested
   against autograd in float64 for every key in the scope, and a key added to the round
   later has to be added in both.

## Tests

| What | Result |
| --- | --- |
| `tests/test_wev_fused_event.py` on the CPU (GPU hidden): the float64 equalities, the scope, the rounding | 38 passed, 2 skipped (the card tests) |
| The same file with `tests/test_p2_10c_state_sweep.py` and `tests/test_p2_10c_state_gradient.py` on the card (the reverse sweep whose middle kernel changed) | 132 passed |
| The window kernels' other card tests (`test_p2_10c_fast_slots`, `test_p2_10b_folded_sweep`, `test_p2_10b_trainer_kernels`, `test_p2_10_split_window`) | 123 passed |
| The whole suite on the CPU (GPU hidden, `-n 8`, thread caps) | 2,791 passed, 529 skipped (CUDA, MPS and kernel-build tests), 118 subtests passed |
| `ruff check`, `ruff format --check`, `mypy --strict` on the changed modules | clean |

The card tests of this file take inputs whose every decision has a margin of 1e-3 (checked):
fp32 against the autograd loop and the PyTorch reference at the compiled window's CUDA
tolerance (values 1.5e-5, gradients 2e-4, identical decisions), and bf16 under autocast
(where the decisions agree: logits within 2 %, gradients within 5 %).

## Files

- `python/hsn/train/event_fused.py`: the reference (`round_reference`, `forward_reference`,
  `sweep_reference`, `sweep_buffered`, `local_backward`), the node (`EventRecurrence`), the block
  (`fused_event_block`) and the window around it (`forward_events`), the scope.
- `kernels/cuda/hsn_event.cu` (in the window fatbin, `scripts/build_window_kernels.py`):
  `hsn_ev_finish`, `hsn_ev_decide`, `hsn_ev_pick`, `hsn_ev_pairs`.
- `python/hsn/kernels/event.py`: `EventKernels` (the buffers and launches of one block shape),
  `EventKernelSet` (one per number of slots).
- `kernels/cuda/hsn_window.cu`, `python/hsn/kernels/window.py`,
  `python/hsn/train/state_sweep.py`: `hsn_win_back_mid` optionally keeps `dhid` of every
  position (`ReverseSweep.d_hid`; off by default, the sweep's results unchanged).
- `tests/test_wev_fused_event.py`.
- Outside the repository, `~/hsn-experiments/wev-fused/card/`: the card scripts
  (`card1.py` equality, `card2.py` the parts' timing, `card3.py` the window against the
  compiled window and the three forms' distances, `card_phase.py` a kernel's phases,
  `slots2.py` the slots of the field data, `hold_open.sh` / `run_all.sh` the card protocol)
  and their results (`eq_r4.json`, `parts_r5_64.json` / `_48.json`, `abt_r6.json`,
  `abt_r7.json`, `ab_r4.json`, `phase_r4.json`, `slots2.json`, the tests' logs).
