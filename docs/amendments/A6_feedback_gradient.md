# Amendment A6 — the feedback gradient (`train.feedback_detach`, `train.state_grad_ratio`, `train.state_grad_clip`)

Status: opt-in, default off (10 Nov 2026). No forward value changes; two training-recipe choices
about the gradient that flows back through the recurrent state. Results:
`docs/p1_18_a6_results.md`.

Two mechanisms, independent of each other:

- **`train.feedback_detach`** removes named feedback reads from the backward pass altogether.
  With all four the backward recurrence cannot amplify at all, at the price of a biased gradient.
- **`train.state_grad_ratio`** / **`train.state_grad_clip`** keep every gradient path and limit,
  per document and per position, how large the gradient passed back into the state may be
  (relative to the batch's typical document, or as an absolute number). A healthy window is plain
  backpropagation through time.

## The state-gradient clip

At the entry of every position `t`, the gradient that position `t` (and everything after it) passes
back into the state it started from is limited per batch row:

    ‖dL/dq[t−1]‖ ≤ c            ‖dL/dh[t−1]‖ ≤ c / sqrt(N)        (c = train.state_grad_clip)

A row under the limit is passed on as it is (no scaling); a row over it keeps its direction and is
scaled to the limit; a row with a non-finite entry is dropped. `sqrt(N)` is the ratio between a
summary gradient and the cell gradient it induces (`q` is the mean of the `N` cells), so one number
limits both. The function (`hsn.model.gradclip.clip_row_grad`) is the identity in the forward
pass, runs inside the compiled window, and is absent without gradients (evaluation, serving).

Why this and not the trainer's global clip alone: a burst is one document whose state gradient
jumps 7× to 22× at a single write or read, several times in a row (below). The global clip then
rescales the whole batch by that norm, so the other 31 documents' gradients vanish and the step
follows one document's switching direction at full AdamW size. The row limit acts where the jump
happens and leaves the other rows alone. No step is skipped, no document is dropped, nothing is
rolled back.

Calibration (B32, delayed task, exp 346's checkpoints at steps 3,000 and 8,500, 12 windows): the
largest per-row gradient over a window is at most 0.16 in `h` and 2.95 in `q` (medians 0.03 and
0.4). Burst windows of the same run have rows at 0.2 to 32 in `h` and up to 30 in `q`. `c = 4`
(`h`: 0.71) is above every healthy row measured; `c = 2` trims the largest one or two `q` rows of
a window. The limit is an absolute number, like `train.grad_clip`: per-row gradients scale with
1 / (loss positions in the batch), so another batch size or task needs its own look.

**The weakness of an absolute number, and the relative limit.** Healthy rows grow about twentyfold
over a run (the largest `dL/dq` row of a window in exp 346: 0.13 at step 500, 0.35 at 1,000, 1.2
at 2,000, 2.9 from 3,000 on), so `c = 4` is 12 to 30 times a typical row during the first 1,000
steps: it cannot act there, and exp 354 (`c = 4`, no update caps) fell to chance by step 1,000.
`train.state_grad_ratio = k` is the same limit without a scale: at every position a row's `dL/dq`
(and its `dL/dh`) is at most `k` times the batch's *typical live row* there, the 90th percentile
of the rows that carry a gradient at that position (a document that ended earlier carries none and
is not counted; with fewer than ten live rows the typical row is their largest, so nothing is
limited). Measured on exp 346's checkpoints (total loss, 4 windows each, per position,
`/home/joe/hsn-experiments/a5-fix/probe/probe_ratio.py`): the largest row of a window is usually
1.3 to 3.5 times that percentile, and `k = 3` would act on 0 % (step 500), 0.9 % (1,000), 0.2 %
(3,000) and 0.7 % (8,500) of the live row-positions in `q`, at most one document per window: the
isolated documents at 9 and 21 times the percentile that these otherwise healthy checkpoints
already contain. In the burst checkpoint (6,000) the same document-level outliers reach 24 times
in `q` and 11 times in `h`. So the relative limit acts on the same kind of row at every stage of
a run, which no absolute number does. Up to three rows of a B32 batch can blow up without moving
the percentile. Both keys may be set; the smaller limit applies.

## The feedback cut

At every position the recurrent state `(h, q)` is read in these places:

| Read | Path | With A6 |
| --- | --- | --- |
| the carry | `h[t] = (1 − b) ⊙ h[t−1] + b ⊙ c`, `q[t] = q[t−1] + Σ Δh / N` | kept (never cut) |
| the decoder and the selected-cell readout | `D([q[t]; u[t]; readout[t]])` | kept (never cut) |
| the prediction loss | `L_pred(P(q[t−1]), z*[t])` | kept (never cut) |
| `summary` | `u[t] = S([z[t]; q[t−1]; ctx[t]])` | `S` reads `q[t−1].detach()` |
| `eps` | the cell and the scorer read `ε[t] = z[t] − P(q[t−1])` | they read `z[t] − P(q[t−1]).detach()` |
| `state` | a cell's units, gate and messages read `h[t−1]` | they read `h[t−1].detach()` |
| `router` | the scorer's features `[u; ε; h]` | detached; a selection gradient trains the scorer's weights only |

`train.feedback_detach` is a list of the last four names. Each one makes that read a constant in
the backward pass. The values of every tensor are the same as without it (bit for bit on the CPU,
`test_forward_values_are_unchanged`), so the model, its state, the checkpoints' weights, serving and
evaluation are unchanged. Only the gradient is.

With all four, the backward recurrence of the state is

    ∂h[t]/∂h[t−1] = diag(1 − b)   on the cells position t wrote, the identity elsewhere
    ∂q[t]/∂q[t−1] = I,   ∂q[t]/∂h[t−1] = −b / N on the written cells

which has no singular value above 1 in `h` and is linear in `q`. A gradient cannot grow on its way
back through a window (`test_full_cut_leaves_the_carry_as_the_only_recurrence`,
`test_full_cut_bounds_the_state_gradient`). Every module still learns: the task loss reaches the
cell, `S`, the encoder and the embeddings at every write through the carry, the router through the
selection estimator, and the predictor through `L_pred`, which also still shapes `q`.

## Why

The A5 runs lose accuracy mid-run in bursts of exploded gradients (exps 325, 340, 342, 345, 346;
`docs/p1_17_training_findings_2026_11_10.md`). On the stored checkpoints of exps 345 and 346
(`/home/joe/hsn-experiments/a5-fix/probe/`):

- The straight-through selection paths do not cause it: with the mask, the unit weights or every
  scorer feature detached, the gradient norm of an exploding window is unchanged (159 → 158).
- The state gradient does not grow smoothly. It jumps by 7× to 22× at single positions and is flat
  in between. The candidate is saturated on 95–99 % of its coordinates from step 1,500 on and
  the gate is closed or fully open, so the cell is a bank of hard switches: almost no gradient
  anywhere, and a very large one where a switch is in transition.
- Those jumps are in the feedback reads. Cutting `q → P` and `q → S` takes exp 346's worst window
  from 159 to 6.9; cutting the cell's read of `h` and the messages takes it to 2.5 (an ordinary
  window is 2 to 3). Exp 345 before its collapse: `q → S` alone, 29 → 4.7.
- The memory this task needs does not use those paths. A fact is written from `u[t]`, held by the
  carry, and read by the decoder; the key address (A5) picks the cells. The feedback reads are what
  a key-blind strategy needs ("which pair comes next").

Clipping keeps the size of such a gradient at 1 but not its direction: with AdamW every step of a
burst is a full-size step along one document's switching direction. A6 removes those directions
from the gradient instead of skipping or rolling back the steps.

## What it is not

- Not a change to §2.2–§2.6: predict-before-encode, ε as a cell input, the units, the graph, the
  convex tanh update, the router, `q` and `D([q; u])` all run as before.
- Not a guard: no step is skipped, no document dropped, nothing rolled back.
- Not truncation in time: the carry is followed over the whole window.

## Cost and risks

- The gradient is biased: it ignores how an earlier state changes a later write. A strategy that
  needs that credit (counting, "do not overwrite") trains only through the direct reads.
- `eps` leaves the predictor trained by `L_pred` alone; with `train.lambda_p = 0` and `eps` cut it
  would receive no gradient.
- Forward sensitivity is untouched: a hard-switching cell stays sensitive to its state.

## Configuration

    train.state_grad_ratio: null                            # default: off
    train.state_grad_ratio: 3.0                             # a row <= 3x the batch's typical live row
    train.state_grad_clip: null                             # default: off
    train.state_grad_clip: 4.0                              # the absolute per-row limit (q; h: / sqrt(N))
    train.feedback_detach: []                               # default: full backpropagation
    train.feedback_detach: [summary, eps, state, router]    # the carry alone

The keys are left out of the configuration hash at their defaults (existing checkpoints resume
and existing arm ids are unchanged); a set value hashes, and the trainer refuses a model built
with other values. HSN only.
