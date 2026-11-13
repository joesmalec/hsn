# P2.10c: a fast training window for the state gradient

13 November 2026. Branch `claude/cutc-fused-window`, from `09db900`. Phase 1 (CPU only) is this
note, the float64 reference and its tests. Phase 2 (the kernels) was approved the same day and
is logged at the end ("Phase 2 log"). Up to that log nothing here was run on a GPU, no kernel
was written, and every step time that is not marked *measured* is an estimate.

## What this is

"The state gradient" is `train.feedback_detach=[summary, eps, router]`: amendment A6's cut
without `state`, so the cell's own read of the state (its `h_i` input and its messages) carries
gradient. It trains only on the autograd window today, because the split and fused windows of
P2.10 are defined for the full cut. The goal is a fast window for it with the same gradients.

Phase 1 delivers:

1. the design (this note): what the split window assumes, what the state gradient adds to the
   backward, how the reverse sweep is built, what it should cost, the risks and the tests;
2. the reference: the split window's computation extended to the state gradient in plain
   PyTorch (`python/hsn/train/state_gradient.py`, `SplitWindow(..., state_gradient=True)`),
   equal to the autograd window's parameter gradients in float64 on the CPU, without and with
   amendment A8's fast cells. It is the oracle for the kernels of phase 2.

Nothing is switched on: no configuration key selects the new path, `hardware.window` is checked
against the same scope as before, and the existing split and autograd windows are bit-identical
to `09db900` (section "How equal it is").

**The short version of the design.** The forward sweep and the batched replay stay as they
are. Only the state recurrence node changes: its reverse sweep gains, per position, the
backward of the selected cells into the state they read. That product is two transposed
matrix products of the unit bank per position, so the reverse sweep costs about half of the
forward sweep's arithmetic. The parameter gradients still come from autograd over the replay.

**What phase 1 found.**

- The reference equals autograd: 1e-9 on every parameter gradient on the small model and at
  the recipe's widths, with and without fast cells; 4e-15 to 4e-14 over full real windows at
  five trained checkpoints, when both follow one trajectory.
- Some trained state-gradient models amplify rounding by 1e9 or more across a window, in the
  forward pass. On the candidate base the model's own eager and capturable autograd rounds
  are 5e-7 and 2.4e-6 apart in float64, and fp32 autograd is order one or more away from
  float64. Equality therefore has to be tested on one trajectory, and the fp32 bar of P2.10
  (two paths, end to end, to rounding) cannot be met on such a model by any implementation.
- Expected step time: 48 to 51 ms at 4 slots and 32 cells (autograd about 100), 76 to 105 ms
  at the candidate base (autograd about 165), 110 to 147 ms with the fast group at 20 slots
  (autograd about 210). The first rests on measured phases; the other two are extrapolations
  whose upper halves are the likelier (about 95 and 130 ms).

## What the split window assumes

The split window (`docs/p2_10_split_window.md`) replaces one autograd graph over 256 positions
by a forward sweep without autograd, one batched replay with autograd and a linear recurrence
over the state. Four places rely on a feedback read being a constant in the backward:

| Where | What it treats as a constant | The cut it relies on | With the state gradient |
| --- | --- | --- | --- |
| `SplitWindow._replay`: `S([z; q_pre; ctx])` on the taped `q` | the summary `S` reads | `summary` | still cut |
| `SplitWindow._replay`: `eps_s = z − P(q_pre)` under `no_grad` | the prediction in the cell's and the scorer's error | `eps` | still cut |
| `_scorer_logits(u.detach(), eps.detach(), h_pre, …)` | the scorer's features | `router` | still cut |
| `PDGCell.replay_sparse`: `h_const = h_read.detach()` | the state the messages, the units and the gate read | `state` | **no longer cut in the model** |

Because of the fourth row, the candidate `c` and the gate `b` of a position depend only on
parameters and on values local to the position, and `StateRecurrence.backward` needs nothing
but the elementwise carry: `hsn_win_recurrence` is one kernel launch per window. That is what
lets the heavy backward (the unit bank, the gate, `W_out`, the messages) run for all `B·T`
rows at once.

Without the `state` cut the model's `step_sparse` reads the state itself. The cotangent of the
state at position `t` then depends on the cell's backward at every later position, and that
part of the backward has to run position by position.

## What the state gradient adds to the backward

Per position and selected slot `k` (cell `i`, unit `j`, slot weight `α`; `x = [u; ε; h_i; m_i]`
of width `n = 4d`):

```
m   = Σ_e a_e · V h[src_e]                           the messages of cell i's sources
v   = g_n ⊙ x · r,   r = 1 / sqrt(mean(x²) + eps)    the cell norm
hid = v W1_j + b1_j;   F = act(hid) W2_j + b2_j      the executed unit
b   = σ(G v + b_g)                                   the gate
y   = W_out F;   c = tanh(α · g_c ⊙ y · ρ),  ρ = 1 / sqrt(mean(y²) + eps)
h_i' = (1 − b) ⊙ h_i + b ⊙ c;   q' = q + (h_i' − h_i) / N
```

The carry `(1 − b) ⊙ h_i` and the summary are P2.10's recurrence. New are the Jacobians of
`(c, b)` with respect to the two state columns of `x`: `h_i` (the cell itself) and `m_i`
(through `a_e V`, every source of the cell's messages). In the direction the backward needs
them, with `dc`, `db` the cotangents of `c` and `b`:

```
dz   = dc ⊙ (1 − c²)
dy   = ρ (g_c ⊙ α dz) − ρ³ y · mean(y ⊙ g_c ⊙ α dz)       candidate norm
dF   = W_outᵀ dy
dhid = (W2_j dF) ⊙ act'(hid)                               second layer, transposed
dg   = db ⊙ b ⊙ (1 − b)
dv_s = (W1_j)_s dhid + (Gᵀ)_s dg                           first layer and gate, state columns
σ    = ⟨hid − b1_j, dhid⟩ + ⟨gate_pre − b_g, dg⟩
ds   = r (g_n)_s ⊙ dv_s − (r² / n) · [h_i; m_i] · σ        cell norm, state columns
dh   = ds[:d];   dm = ds[d:]
```

`dh` goes to the cell's own pre-round state and `a_e · Vᵀ dm` to each source `e`.

One thing keeps this cheap. The cell norm's backward needs the inner product of `v` with the
whole `dv`, all `4d` columns. But `v W1_j = hid − b1_j` and `G v = gate_pre − b_g`, so that
product is `σ`, made of values the forward pass already has. The sweep therefore never forms
the `u` and `ε` columns of `dv`: the transposed first layer runs on `2d` of its `4d` input
columns. (The `u` and `ε` columns are local to the position and stay with the replay.)

The sweep (`A [B, N, d]`, `R [B, d]`: cotangents of the state after position `t`; `D_t`,
`P_t`, `S_t` those of `q` after `t`, `q` before `t` and the selected updated states; `G_t`
that of the fast cells' states after `t`, section "Fast cells"):

```
A = 0; R = 0
for t = T−1 .. 0:
    R += D_t;  A[i] += G_{t,i} (i < F);  A[cell_k] += S_{t,k}            (valid slots)
    g_k = A[cell_k] + R / N
    dc_k = b_k ⊙ g_k;  db_k = (c_k − h_old_k) ⊙ g_k                      as in P2.10
    (dh_k, dm_k) = the cell's backward above                             new
    A[cell_k] = (1 − b_k) ⊙ A[cell_k] − b_k ⊙ R / N + dh_k
    A[src_{k,e}] += a_{cell_k,e} · Vᵀ dm_k                               new
    R += P_t
    A, R = 0 on the rows that reset at t
```

A selected cell that is also a source of another selected cell gets both contributions: the
carry replaces its row first, then the sources add.

## The design

**Forward sweep: unchanged.** The tape already holds the pre-round state of every position
(`tape_h`), the decisions and the written values. Nothing more has to be recorded in the
sweep, and the fused sweep's kernels and tape do not change for the routed shape.

**Replay: the same operations, plus a record.** The replay still reads the taped state as a
constant. Its graph is then the *local* half of the gradient: the partial derivatives of every
position's `(c, b)` with respect to the parameters, `u`, `ε` and the selection's weights, at a
fixed state. It also hands the sweep the values it computed for each slot
(`PDGCell.replay_sparse(record=…)`): `W_out`'s output `y`, `α`, the executed unit's
first-layer pre-activation `hid`, the gate's pre-activation, the messages `m` and the cell
norm's `r`. They are the replay's own values at the *recorded* unit: nothing is decided or
recomputed a second time.

**Recurrence node: `StateGradientRecurrence`.** Forward returns the taped `q` and selected
states as before. Backward runs the sweep above and returns the *total* `dL/dc` and `dL/db` of
every position. Autograd then carries them through the replay's graph, as it does today.

Why this is the whole gradient: the cell at position `t` depends on a parameter directly (at
a fixed state) and through the state it reads. The first is the replay's graph; the second is
inside the total `dc_t`, `db_t`, because the sweep's `A` already holds everything later
positions passed back through their own cells. So the parameter gradients stay autograd's,
and the only hand-written derivative is the path through the state.

**What the sweep reads per slot** (window totals at B32, T256):

| Value | Shape per slot | From | 4 slots | 12 slots | 20 slots |
| --- | --- | --- | ---: | ---: | ---: |
| `c`, `b` | `d` each | the replay (as today) | | | |
| `h_old` | `d` | the tape (as today) | | | |
| `y`, `gate_pre`, `m` | `d` each, state dtype | the replay's record | 8 MB each | 25 MB each | 42 MB each |
| `hid` | `H`, the unit bank's dtype | the replay's record | 34 MB (bf16) | 101 MB | 168 MB |
| `α`, `r`, the unit | scalars | record, tape | | | |

Autograd already keeps `hid` for the activation's backward, so the record is a reordering of
values that exist, not new work.

### Alternatives considered

- **Recording these values in the forward sweep instead.** More small kernels and memory
  traffic in the part of the step that is bound by kernel count, for values the replay computes
  anyway. And under bf16 the sweep's values differ from the replay's by rounding: the sweep's
  product should be the derivative of the same computed `(c, b)` that autograd differentiates.
- **Writing the parameter gradients by hand too** (one node for the whole cell, no autograd
  over the cell). The sweep would then need all `4d` columns of the transposed first layer,
  1.66 times its arithmetic, in the all-units layout, where the batched replay does the same
  products grouped by unit at an eighth of the arithmetic. It is also several times more
  hand-written derivative. Not proposed; the sweep's `trace` leaves it possible.
- **Per-slot Jacobians computed in a batch, then a light sweep.** A `2d × 2d` Jacobian per slot
  needs `2d` backward passes of the unit bank: 128 times the arithmetic.
- **One kernel per position for the whole cell backward** (the split window's author expected
  that only this shape would pay). P2.10 measured custom selected-unit products at 73 µs
  against 35 µs for cuBLAS on tensor cores at 4 slots, and what paid there was to keep the
  products on tensor cores and fuse what lies between them. The proposal below follows that:
  two cuBLAS products and two or three glue kernels per position. Which is faster is a
  measurement for phase 2, and the reference serves either.

## Fast cells (amendment A8)

The candidate shape now includes `model.fast_cells=8`, `fast_units=fixed`, `fast_readout=true`
with `k_select=20` at 64 cells. Checked against the code, not assumed:

- **Reserved slots.** Cells `0 .. F−1` take the first `F` slots at every position
  (`Router.select(reserved=F)`), with a constant mask and, with `fast_units=fixed`, a constant
  one-hot unit weight. The replay's `Router.finish_recorded` now takes `reserved` and
  `reserved_units` and builds the same gradient path from the recorded slots. The recorded
  threshold is already the one taken among the slow cells.
- **A fast cell is an ordinary cell in the sweep.** It reads its own state and its messages
  like any slot, so the cell's backward above applies unchanged. Because it is written at
  every position, its cotangent passes through that backward at every position of the window:
  this is the long chain the state gradient exists for.
- **The fast read into `S`** (`f_{t−1}`, the fast cells' mean before the position) is a
  `summary` read (`HSN._shared_input`, `_summary_read`). Under `[summary, eps, router]` it is
  detached in the model, so it is a constant in the backward: the replay reads it from the
  taped state and no gradient goes from `u` into the fast cells. This would change only if
  `summary` were dropped from the cut, which is outside this scope.
- **The fast read into the decoder** (`f_t`, the fast cells' mean *after* the position,
  `HSN._selected_readout`) does carry gradient. It is a fourth output of the node: the fast
  cells' states after every position, whose cotangent `G_t` enters the sweep at the top of
  position `t`. The value is the slot's written state, or the cell's carried state where the
  budget cut the slot (a fast slot can be invalid under a small `model.budget`).
- **The selected-cell read-out** averages the routed slots only (`_readout_of_slots(fast=F)`).

With `fast_units=fixed` each fast slot's unit is known before the window. For the kernels this
matters: eight fast cells at eight fixed units are a static grouping (a batched product per
unit at an eighth of the all-units arithmetic), in the forward sweep as well as the reverse.

## The reference

`python/hsn/train/state_gradient.py`:

- `cell_state_vjp`: the cell's backward into the state, as written above, for one position's
  slots.
- `state_gradient_backward`: the sweep. With a cell that reads nothing it is P2.10's
  `recurrence_backward` value for value (tested). `trace` returns `A` and `R` at every position
  and the cell's `dh`, `Vᵀ dm`: the intermediate values a kernel can be bisected against.
  `initial` starts it from a given cotangent of the state after the last position, so one
  position can be checked alone.
- `StateGradientRecurrence`, `cell_constants`: the autograd node and the weights it reads.

`SplitWindow(model, cfg, None, state_gradient=True)` wires it in. Scope
(`split_window_scope_error(cfg, state_gradient=True)`): the split window's, with `summary`,
`eps`, `router` cut and `state` not; A8's fast cells and A9's `model.address_ngram` are inside
it; the eager sweep only; a window that drew dense-fallback rows (`train.p_dense`) runs the
autograd window from the same draw, as under `hardware.window=fused`.

Small, default-off hooks in existing code: `PDGCell.replay_sparse(record=)`,
`UnitBank.forward_grouped(record=)`, `Router.finish_recorded(reserved=, reserved_units=)`,
`_readout_of_slots(fast=)`, `split_window_scope_error(state_gradient=)`.

## How equal it is

All on the CPU. "Entries failing" counts parameter-gradient entries outside
`|got − want| ≤ 1e-9 · |want| + 1e-12`.

| Comparison | Result |
| --- | --- |
| The cell's backward against autograd of `PDGCell.step_sparse` (one position; both norms on and off, four activations, both edge norms, random edges, padded slots, the `u` ablation, fast cells) | equal at rtol 1e-11 |
| The sweep's `A` and `R` against autograd's gradient of the state each position reads, at every position of a window (routed, and fast cells with fixed units) | equal at rtol 1e-9 |
| A single step of the sweep at every position, started from autograd's cotangent of the state after it (the same two models) | equal at rtol 1e-9 |
| Window against `HSN.forward`, float64, small model, 21 configurations (norms, activation, edge norm, cost term, executed credit, `u` ablation, A9, refresh probes, padded slots, no read-out, no loss positions, after a soft warm-up, fast cells with routed and fixed units, the fast read, a budget that cuts a fast slot) | every parameter gradient equal at rtol 1e-9, the same set of absent gradients, identical decisions |
| The same at the recipe's widths (d 64, H 512, 8 units), 12 slots and 64 cells, and 8 fast + 12 routed slots, short window | no entry failing; largest difference 2e-15 to 1e-14 of the tensor's largest entry |
| The existing split and autograd windows against `09db900` (23 cases: float64 and fp32, dense-fallback rows, executed credit, the `u` ablation, the recipe's widths) | logits, final state, every parameter gradient and the configuration hashes bit for bit |

### At trained checkpoints, on real windows

Two finished state-gradient runs, their step-9,000 training weights, two real training windows
each at B32 and T256 with the trainer's loss (cross-entropy at the loss positions plus the
prediction term). "Same round" is the model's own autograd window running the capturable
round the sweep runs (`step_sparse(static=True)`, the round of the compiled window); "eager
round" is `HSN.forward` as called by default. The two rounds are the same function with the
unit bank's products in another order.

| | Candidate base: 12 slots, 64 cells, kv and delayed (`g1_k12nw_cutC_cells64`, seed 1) | Fast group: 8 fast cells at fixed units, fast read, 20 slots, 32 cells, text facts (`tfq_f8ru20_cutC`, seed 0) |
| --- | --- | --- |
| Window against autograd, same round, float64, all parameters (relative L2) | 9.7e-15, 8.8e-15 | 9.3e-15, 1.05e-14 |
| The window's logits against that pass | identical (0.0) | identical (0.0) |
| One step of the sweep from autograd's cotangent, worst of 256 positions (of the largest entry) | 1.0e-15, 7.5e-16 | 4.4e-16, 6.3e-16 |
| Window against autograd, eager round, float64 | 4.9e-7, 2.4e-6 | 1.9e-14, 7e-14 (no entry of 1,919,000 outside 1e-9) |
| **Autograd against autograd, eager round against same round, float64** | **4.9e-7, 2.4e-6** (logits apart by 5e-15 at position 0, 1e-8 at 192, 7e-8 and 2.4e-6 at 255) | 1.7e-14, 7.1e-14 |
| Autograd fp32 against autograd float64 (eager round) | 110, 0.57; 98.9 % and 99.3 % of the selected cells the same | 3.4e-5, 4.8e-4; identical decisions |
| Window fp32 against autograd fp32 | 1.14, 0.75; 99.98 % and 99.97 % of the cells the same | 8.5e-7, 6.3e-7; identical decisions |
| The cell's backward left out of the sweep | 100 % off | 24 %, 26 % off |
| State cotangent, per batch row: largest row against the median row | 160 against 1.8e-4 (12 of 32 rows above ten times the median); 26 against 8e-4 (5 rows) | 2.4e-3 against 9.5e-5 |

Three more state-gradient checkpoints, one real window each (B32, T256, float64):

| Run (step 9,000) | Window against autograd, same round | One step, worst position | Autograd against autograd, eager against same round |
| --- | ---: | ---: | --- |
| `a8_cutC`, seed 0: 8 slots, 32 cells, delayed | 3.7e-15 | 4.8e-16 | 1.7e-15 |
| `g1_k12nw_cutC_cells64_ng2`, seed 0: the candidate base with A9's `address_ngram=2` | 5.2e-15 | 7.5e-16 | 1.9e-9 |
| `g1_k12_cutC`, seed 1: 12 slots, 32 cells, kv and delayed, soft warm-up | 3.6e-14 | 1.2e-15 | 2.5: the two rounds take different decisions in 0.16 % of the slots, and their logits end up to 15.8 apart |

What this says:

- **The sweep is exact at trained checkpoints too.** On one trajectory it agrees with autograd
  to between 4e-15 and 4e-14 over a full window on all five checkpoints, and a single step
  agrees to 1e-15 at every position.
- **The candidate-base checkpoint amplifies rounding by 1e9 to 1e10 across a window, in the
  forward pass.** Its own two autograd rounds, both float64, end a window with logits 7e-8
  and 2.4e-6 apart and gradients 4.9e-7 and 2.4e-6 apart. The window's gap to the eager round
  is that gap, to the digit, because the window follows the other round. It varies by run:
  the kv and delayed runs at 12 slots show it (1.9e-9 to 2.4e-6, and on one of them a
  decision flips in float64 and the two rounds part completely); the delayed run at 8 slots
  and the fast-group run on text facts do not.
- **In fp32 the gradient of such a window is not determined.** fp32 autograd is 110 times and
  0.57 times the float64 gradient's norm away from it on the two windows, and takes other
  routing decisions in about 1 % of the slots. Any two fp32 implementations will differ by
  order one there, the eager and the compiled autograd window included. In the first window
  12 of the 32 batch rows have a state cotangent more than ten times the median row's, and
  the largest is a million times it: this is the growth that A6's `state` cut removes, and
  the gradient clip is what bounds its effect on a step.
- So equality has to be tested **on one trajectory**: against the same round, per step, or on
  a model whose own two rounds agree. End-to-end agreement of two paths in fp32 "to rounding
  with identical routing decisions", the bar of P2.10, is not reachable on this checkpoint by
  any implementation, and that is a property of the trained model, not of a window.

## Phase 2: the kernels

**The reverse sweep.** The mirror of the fused forward sweep: compiled blocks of positions
under CUDA graphs, in tape form (a block reads the replay's record and the cotangents of
position `t` and writes `dc`, `db` of that position in place; `A` and `R` live in two buffers
the kernels mutate). Per position:

| Step | What | As |
| --- | --- | --- |
| head | gather `A[cell]`, `g`, `dc`, `db`, `dz`, candidate norm backward, `W_outᵀ` (a `d × d` product per slot), write `dF` into the slot's unit block of the second layer's operand | glue kernel |
| product 1 | `[B·K, M·d] × W2ᵀ → [B·K, H]` | cuBLAS, the all-units layout of the forward sweep |
| middle | `act'(hid)`, `σ`, write `dhid` into the slot's unit block of the first layer's operand, `dg` into its gate columns | glue kernel |
| product 2 | `[B·K, M·H + d] × (W1 with the gate)ᵀ_s → [B·K, 2d]` | cuBLAS |
| tail | cell norm backward, `Vᵀ dm` (`d × d` per slot), the carry, the scatter into the cell and its sources, `R`, the reset | glue kernel, which can also be the head of position `t − 1` |

So two matrix products and two or three glue kernels per position: 4 to 5 kernels against the
forward sweep's 15. The all-units operands are zero outside one unit block per slot; a kernel
writes the block and clears the one the previous position used, instead of writing 4,160
values per slot.

Matrix products per position, in multiply-accumulates (B32, d 64, H 512, 8 units):

| Shape | Forward sweep | Reverse sweep | Ratio | Reverse, grouped by unit |
| --- | ---: | ---: | ---: | ---: |
| 4 slots, 32 cells (the A6 recipe) | 196 M | 103 M | 0.52 | 15 M |
| 12 slots, 64 cells (candidate base) | 552 M | 308 M | 0.56 | 44 M |
| 20 slots (8 fast), 64 cells | 894 M | 514 M | 0.57 | 73 M |

The last column is what the reverse sweep would cost if each slot's products ran against its
own unit only. The all-units layout costs eight times that; it is what keeps the shapes static
and the products on tensor cores, in both sweeps.

**The forward sweep for the candidate shapes.** Independent of the reverse sweep, and needed
before `hardware.window=fused` can run them:

- A8: `hsn_win_select` has to know the reserved slots (the first `F` slots fixed, the unit
  `i mod M` or the best one, the threshold among the slow cells, probes that skip them); the
  merged `S`/`P` product's input and the folded next-row write of `hsn_win_cell_update` need
  the fast read `f`; the sweep twin's layouts for the wider `S`.
- A9: two more scorer columns in the merged scorer/`V` layout.
- A11 (`model.scorer_position=wrap`, not on main) is computed in the window prologue and needs
  nothing in the kernels.

**The record.** The replay's `hid` is produced per unit block in sorted order; the reference
reorders it. A kernel can read it through the permutation instead of copying it.

**The configuration surface** (a decision for the user, nothing is implemented):
`hardware.window=split|fused` would accept `[summary, eps, router]` and select the sweep from
the configuration. No new key; configurations that load today keep their hashes.

**What it takes** (estimates):

| Piece | Work | Card time |
| --- | --- | --- |
| Reverse sweep: three glue kernels, their tape forms, the compiled block, the record read through the permutation, the PyTorch fallback wired as `set_fast_recurrence` is today | 1 to 1.5 days | equality tests in short sessions (5 to 15 minutes each, about ten of them) |
| Forward sweep for A8 and A9 (selection kernel, merged layouts, the fast read in the folded row) | 0.5 to 1 day | the fused sweep's tape-equality tests, a few minutes each |
| Timing: three windows at three shapes, with the phase breakdown | | 30 to 45 minutes alone on the card, of which the two configuration-only timings named under "Expected step time" are 5 minutes and can come first |
| Accuracy runs on the candidate base | | 9,000 steps at about 95 ms is 15 minutes a run; six seeds per arm can share the card |

## Expected step time

*Measured* (alone on the local RTX 3060, the A6 recipe, 4 slots, 32 cells, bf16;
`docs/p2_10b_fused_speed.md`): fused step 40.5 ms: prologue 1.4, forward sweep 19.2, replay
forward 6.6, backward 8.7, the rest about 4.6. Autograd window 94.8 ms.

*Measured, from finished runs* (`step_time.run_median_s` in their `result.json`; the rented
cards are about 10 % slower than the local one on the same configuration, and several arms
may have shared a card): the autograd window with the state gradient takes 137 to 140 ms at 8
slots and 32 cells locally (the "about 134 ms" of the brief), and on the rented cards 148 to
154 ms at 8 slots, 179 to 185 ms at 12 slots and 64 cells, 222 to 226 ms at 20 slots with 8
fast cells (32 cells). The fused window with the full cut: 80.6 to 83.4 ms at 12 slots and
93.7 to 96.3 ms at 16 slots on the rented cards (32 cells); locally 83 to 91 ms at 8 slots
and 64 cells.

*Estimated* (the reverse sweep from the census above and P2.10's measured costs per product
and per kernel; the other phases at the larger shapes scaled from the 4-slot measurements and
cross-checked against the finished runs):

| Phase | 4 slots, 32 cells | 12 slots, 64 cells | 20 slots (8 fast), 64 cells |
| --- | ---: | ---: | ---: |
| Prologue | 1.4 (measured) | 1.6 | 1.7 |
| Forward sweep | 19.2 (measured) | 28 to 37 | 40 to 50 |
| Replay, forward (with the record) | 6.6 to 7.5 | 12 to 18 | 18 to 27 |
| **Reverse sweep (new)** | **8 to 10** | **15 to 19** | **23 to 28** |
| Backward through the replay | 8.3 | 14 to 24 | 22 to 35 |
| Optimiser and the rest | 4.6 | 5 | 5 |
| **Step (sum of the rows)** | **48 to 51** | **76 to 105** | **110 to 147** |
| Autograd window, the same shape, local | about 100 (not measured with this cut) | about 165 (from 179 to 185 on the rented cards) | about 210 (from 222 to 226 at 32 cells, plus the 64 cells) |
| Gain | about 2 times | 1.6 to 2.2 times | 1.4 to 1.9 times |

How sure: the 4-slot column rests on measured phases and one new phase, and I would be
surprised by more than 55 ms. The two candidate columns are extrapolations: neither window
has been timed alone at those shapes, and the range is mostly the unmeasured fused step with
the full cut. The finished runs point to the upper half of both ranges (the rented cards'
fused runs at 12 and 16 slots, taken as alone on their card and scaled to the local one, plus
the 64 cells and the reverse sweep: about 95 ms and about 130 ms, a gain of about 1.7 and
1.6 times). Two configuration-only timings on the card settle that part: the fused window
with the full cut at 12 slots and 64 cells, and at 20 routed slots and 64 cells (the kernels'
shapes are the same with or without the fast group).

The earlier estimate of 60 to 70 ms was for 4 slots and 32 cells and assumed a reverse sweep
that costs what the forward sweep costs. At that shape this design should be under it. At the
candidate shapes the step is no longer bound by kernel count: the all-units products and the
batched replay are arithmetic, and they grow with the slots.

**What would move the larger shapes further** (not part of this design, in order of what the
census says they are worth): the fast slots' products per unit instead of over all units
(static with `fast_units=fixed`: 40 % of the slots at an eighth of the arithmetic, in both
sweeps); a grouped product for the routed slots; the replay reusing the sweep's first-layer
values instead of recomputing them.

## Risks

1. **bf16 and the recorded unit** (the replay-credit fault of 11 Nov, `9e0a61a`). Nothing in
   the sweep chooses a unit: `units` come from the tape and the record is the replay's pass at
   that unit. fp32 and float64 tests cannot show a violation, so phase 2 needs a test that
   forces a near-tie under bf16 and checks which unit's weights the sweep reads.
2. **Rounding along the window, and the test bar.** The recurrence is no longer contractive,
   and on the trained candidate base the forward pass itself amplifies rounding by 1e9 or
   more ("At trained checkpoints"). Two correct fp32 paths cannot be held equal end to end
   there, so the rule "fp32 on the card to rounding with identical routing decisions" has to
   be applied on one trajectory (the test plan below), and a failure of an end-to-end
   comparison on a trained model does not by itself say which side is wrong. Under bf16 the
   cell's backward is rounded at every product and its error passes through later positions;
   how far that moves the gradient against the autograd window's bf16 backward is unknown
   until measured, and on such windows both are noise of the same kind.
3. **`σ` under bf16.** The norm's inner product is taken from `hid − b1` with `hid` in bf16.
   In float64 it is exact; in bf16 its error should be of the size of the products' own, but
   that is an expectation, not a measurement. If it is not, the alternative is to form all
   `4d` columns (1.66 times the sweep's arithmetic).
4. **Arithmetic at 12 to 20 slots.** The gain there is about 1.4 to 2.2 times, not more,
   unless the all-units products go.
5. **Compiled blocks and CUDA graphs.** A new block shape with mutating custom ops; today's
   stall in `torch/_inductor/cudagraph_trees.py` (`maybe_deref`, torch 2.15.0+cu132) is the
   known hazard for a new graph shape. The fallback is a launch loop in the window extension
   without inductor.
6. **Statistics steps.** Per-term gradient norms come from retained backwards of the one
   graph; each now runs the reverse sweep again (once per term, every `stats_every` steps).
7. **Dense-fallback rows** have no sweep; such a window runs the autograd window. The recipe
   has `p_dense = 0`.
8. **Accuracy.** Under bf16 the window follows another trajectory than the autograd window, as
   the split window does. Equal gradients in fp32 do not replace accuracy runs.
9. **The forward sweep's fast-cell kernels** are new decision code in `hsn_win_select`; they
   need the tape-equality tests the fused sweep has.

## Test plan

Phase 1 (done, CPU, `tests/test_p2_10c_state_gradient.py`, 51 tests): the first table of "How
equal it is", the scope and refusals, that the record leaves the replay's values alone, and
that the comparison fails when the cell's backward is left out. The trained-checkpoint rows
are measurements outside the suite (`trained.py`, `onestep.py`).

Phase 2 (on the card, when it is free). Every comparison is on one trajectory: one tape, one
replay, and only the sweep's implementation differs.

**This replaces the rule "fp32 on the card to rounding, end to end, with identical routing
decisions"** that P2.10 was held to (decided 13 Nov with phase 2's approval). The reason is
the measurement under "At trained checkpoints": on the trained candidate base the model's own
two autograd rounds are 5e-7 and 2.4e-6 apart in float64 and fp32 autograd is order one from
float64, so two correct fp32 paths cannot agree end to end there, whatever they are. The
steps below test the same thing where it can be tested: per step, on one tape, end to end on
models whose own rounds agree, and by accuracy runs.

1. **Per step.** Each kernel step against one step of `state_gradient_backward`, both started
   from the reference's cotangent after the position (`initial`): `dc`, `db`, `A`, `R`, and the
   `trace`'s `dh`, `Vᵀ dm`. fp32; small and recipe shapes; resets at block boundaries; padded
   slots; fast cells; a trained checkpoint's real windows. This is the sharp test: nothing
   accumulates, so the tolerance is fp32 rounding.
2. **Per window, on one tape.** The kernel sweep against the reference sweep over whole
   windows, on models whose own two autograd rounds agree (the initial model 300 steps in, as
   P2.10 did, and the fast-group checkpoint), with identical routing decisions by
   construction. On the candidate-base checkpoint the same comparison is reported with the
   autograd window's own fp32 gap beside it, not held to a fixed tolerance.
3. **End to end** against the compiled autograd window where that is meaningful: the initial
   model and the first hundreds of steps, fp32, identical decisions, the tolerance of P2.8.
4. **bf16.** On one tape, the replay and the sweep under bf16 against the same in fp32; the
   window's bf16 gradient norm per parameter tensor against the autograd window's; the forced
   near-tie of risk 1.
5. A deterministic short training run through the trainer, statistics steps included.
6. **Timing** alone on the card: autograd, fused with the full cut, fused with the state
   gradient, at the three shapes; `scripts/bench_window.py phases` for the breakdown.
7. **Accuracy runs** on the candidate base, seeds against the autograd window's. Because the
   gradient on the sensitive windows is not determined in bf16 by any window, these runs, not
   a gradient comparison, are what shows that training is unchanged.

## Phase 2 log

### 13 Nov: approval, the shape, decisions

The user approved phase 2 ("yes on started on this shape, with the cheaper arithmetic for the
fast cells first"). The shape to build for and to time: 64 cells, 8 fast + 12 routed slots
(`model.n_cells=64`, `k_select=20`, `fast_cells=8`, `fast_units=fixed`, `fast_readout=true`,
`selected_readout=true`), the state gradient, executed credit, no soft warm-up. Its four
training runs finished on the autograd window at 229 to 233 ms per step on the rented cards
(arithmetic 99.0 %, TinyStories 0.970 bits per byte, delayed 97.3 %, the pair task 93.4 %,
one seed each). The routed-only shape stays supported; `K`, `N` and the number of fast cells
stay parameters; `model.address_ngram=2` and `model.scorer_position=wrap` may be added.

Decided with it (by the coordinating session, reported to the user):

- **Configuration.** `hardware.window=split|fused` will accept `[summary, eps, router]` and
  fast cells once the path exists and is tested. No new key; existing hashes unchanged.
- **Test bar.** The plan of "Test plan" above, which replaces the end-to-end fp32 rule.
- **Dense-fallback rows** keep going to the autograd window.

Order: the cheaper arithmetic for the fast slots first; then the forward sweep for this shape,
the reverse sweep and the record; then timing alone on the card and accuracy runs.

### Step 1: the fast slots' unit products (CPU, done)

With `fast_units=fixed` slot `i < F` runs unit `i mod M` at every position, so its unit is
known before the window. The static rounds multiplied every pair with every unit's weights
(the all-units layout keeps the shapes static when a pair's unit is the router's choice). For
the fast slots one unit's weights are enough: their weights are stacked per slot once per
window and a slot's pairs take one batched product per layer.

What changed:

- `UnitBank.slot_layout` (the per-slot stacks, with the gate's columns for the merged first
  layer), `forward_static_slots` (the capturable round: values and gradients) and
  `forward_static_chosen_slots` (the fused sweep's pass).
- `WindowConsts.fast_w1/b1/w2/b2`: when a window's constants carry them,
  `PDGCell.step_sparse(static=True)`, `sparse_products` and `sparse_products_raw` send the
  first `F` slots through the per-slot products and the others through the all-units products
  as before.
- Only the split window's sweep sets them: `run_tape` for the eager sweep and
  `SweepTwin` (buffers `sweep_fast_*`, refreshed per window) for the compiled one, both from
  `fixed_fast_slots(model)`. The autograd window's constants never carry them, so its round is
  the one it ran; a model without fast cells, or with fast cells at routed units, takes the
  existing code (tested: the new passes are never entered).
- The reverse sweep's products in the layouts a compiled sweep will multiply with
  (`SweepLayouts`, `sweep_layouts`, `cell_state_vjp(layouts=)`): the all-units operands for
  the routed slots, one batched product per layer for the fast slots. This is the arithmetic
  the kernels of the reverse sweep run; the per-unit reference stays the oracle.

How equal (CPU; `tests/test_p2_10c_fast_slots.py`, 31 tests, and
`scratch/fastslots.py` at the trained fast-group checkpoint, B32, T256, 20 slots of which 8
fast):

| Comparison | Result |
| --- | --- |
| `forward_static_slots` against `forward_static` and the grouped pass: values, and gradients into the inputs and every weight of the bank (1, 3 and 5 fixed slots of 3 units) | rtol 1e-12 |
| `forward_static_chosen_slots` against `forward_static_chosen`, with and without the gate riding | rtol 1e-12 |
| The cell's capturable round, and the fused sweep's products (`sparse_products`, `sparse_products_raw`), with the slot layouts against without | rtol 1e-12 |
| The sweep's tape with the per-slot products against the all-units products, small model (also refresh probes, and a budget that cuts fast slots) | every decision identical, values at rtol 1e-12 |
| The same at the trained checkpoint, two real windows | every decision identical; state values within 4.3e-14 and 2.3e-13 in float64; the fp32 tape bit-identical |
| The sweep twin's slot buffers against `slot_layout`, before and after a weight change; a twin block's tape against the model's | identical |
| The cell's backward and a whole reverse sweep in the static layouts against the per-unit reference, small model | rtol 1e-12 and 1e-11 |
| The same at the trained checkpoint (`dL/dc`, `dL/db`, the state cotangent) | 2e-16 to 7e-16 (relative L2) |
| Window against autograd on the sweep's round at the trained checkpoint, after the change | 9.2e-15; logits identical; one step 5.5e-16 at worst |
| The existing split and autograd windows against `09db900` (the 23 cases of phase 1) | bit for bit |

What it should save, per position, in multiply-accumulates of the matrix products (B32):

| Shape | Forward sweep | Reverse sweep |
| --- | --- | --- |
| 20 slots, 8 fast, 64 cells | 894 M to 601 M (33 % less): first layer with the gate 682 to 447, second layer 168 to 109 | 514 M to 338 M (34 % less): first layer's state rows 341 to 223, second layer 168 to 109 |
| 16 slots, 4 fast | 724 M to 577 M (20 % less) | 411 M to 323 M (21 % less) |
| 12 slots, no fast cells | unchanged | unchanged |

In time, at P2.10's measured rates for these products (7.8e12 and 5.2e12 multiply-accumulates
per second), that is about 41 µs per position less in the forward sweep and 26 µs in the
reverse, against two more matrix-product calls per position in each (the fast slots' batched
products) and the copies that join the two groups' results. What a batched product over 8
small matrices costs on this card is not measured; if it costs what a small matmul does
(2 to 6 µs), the step at this shape is about 10 to 14 ms shorter than the table under
"Expected step time" (110 to 147 ms, likely 130): 96 to 137 ms, likely about 118. On the CPU
the float64 reference sweep went from 11.2 s to 7.9 s per window and the reverse sweep in the
static layouts from 7.3 s to 5.6 s, which shows the arithmetic is gone, not what the card
will do.

## Files

- `python/hsn/train/state_gradient.py`: the cell's backward into the state, the sweep, the
  node; the reverse sweep's static layouts.
- `python/hsn/model/units.py`, `cell.py`, `python/hsn/train/compile.py`: the fixed slots' unit
  products in the static rounds and the sweep twin (phase 2, step 1);
  `tests/test_p2_10c_fast_slots.py`.
- `python/hsn/train/split.py`: `SplitWindow(state_gradient=)`, the fast read and the reserved
  slots in the replay.
- `python/hsn/model/cell.py`, `units.py`, `router.py`, `python/hsn/config.py`: the default-off
  hooks listed under "The reference".
- `tests/test_p2_10c_state_gradient.py`.
- Measurements (outside the repository, `/home/joe/hsn-experiments/cutc-speed/scratch/`):
  `trained.py` (trained checkpoints, real windows, float64 and fp32), `onestep.py` (the two
  autograd rounds against each other, the sweep against autograd along a window and one step
  at a time), `bitwise.py` (the existing windows against the base commit), `recipe_shape.py`,
  and their outputs under `out/`.
