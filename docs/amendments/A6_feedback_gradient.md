# Amendment A6 — the feedback gradient (`train.feedback_detach`)

Status: opt-in, default off (10 Nov 2026). No forward value changes: a training-recipe choice about
which paths the backward pass follows through the recurrent state. Runs and evidence:
`docs/p1_18_a6_results.md`.

**Tested on the A5 recipe, delayed task, 9,000 steps, three seeds** (exps 360, 361, 349), with exp
346's anchored update caps: final dev 81.2 / 79.0 / 84.0 %, later queries 65.8 / 66.6 / 70.7 %,
worst drop during a run 2.2 / 6.8 / 4.5 points, no step skipped. The same recipe with full
backpropagation (exps 343, 326, 346): 73.8 / 78.2 / 77.5 %, worst drop 12.0 / 4.8 / 36.9 points.

Two further keys of the same family, `train.state_grad_ratio` and `train.state_grad_clip`, are
described at the end. They were tested and **do not remove the drops**; they are not part of the
recommended recipe.

## What it is

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
(`/home/joe/hsn-experiments/a5-fix/probe/`; table in the results document):

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

The global clip keeps the size of such a gradient at 1 but not its direction: with AdamW every
step of a burst is a full-size step along one document's switching direction. A6 removes those
directions from the gradient instead of skipping or rolling back the steps.

## What it is not

- Not a change to §2.2–§2.6: predict-before-encode, ε as a cell input, the units, the graph, the
  convex tanh update, the router, `q` and `D([q; u])` all run as before.
- Not a guard: no step is skipped, no document dropped, nothing rolled back.
- Not truncation in time: the carry is followed over the whole window.

## Cost and risks

- The gradient is biased: it ignores how an earlier state changes a later write. A strategy that
  needs that credit (counting, "do not overwrite") trains only through the direct reads.
- It learns more slowly: 0.6 dev at step 4,000 to 4,500, against 2,500 with full backpropagation
  (exp 346) and 2,500 to 3,000 for the GRU's speed bar.
- `eps` leaves the predictor trained by `L_pred` alone; with `train.lambda_p = 0` and `eps` cut it
  would receive no gradient.
- Forward sensitivity is untouched: a hard-switching cell stays sensitive to its state.
- The update caps are still needed for the result above. Without them the cut is as smooth (exp
  348, worst drop 4.0 points) but the router grows to norm 10 and the run ends at the key-blind
  level (78.2 %, later 59.9 %).
- Tested on one task and one model size. Cutting only `[summary, eps]` was as smooth on one seed
  (exp 368) and ended lower (76.3 %).

## Configuration

    train.feedback_detach: []                               # default: full backpropagation
    train.feedback_detach: [summary, eps, state, router]    # the carry alone (the tested recipe)

The key is left out of the configuration hash at its default (existing checkpoints resume and
existing arm ids are unchanged); a non-empty list hashes, and the trainer refuses a model built
with other paths. HSN only.

## Tested and not sufficient: the per-document limit

`train.state_grad_clip = c` and `train.state_grad_ratio = k` keep every gradient path and limit,
at the entry of every position, the gradient that the position passes back into the state, per
batch row (`hsn.model.gradclip.clip_row_grad`; identity in the forward pass, runs in the compiled
window, absent without gradients):

    absolute:  ‖dL/dq[t−1]‖ ≤ c,   ‖dL/dh[t−1]‖ ≤ c / sqrt(N)
    relative:  each of them ≤ k × the batch's typical live row at that position

A row under the limit is passed on unscaled, a row over it keeps its direction, a row with a
non-finite entry is dropped. The typical live row is the 90th percentile of the rows that carry a
gradient at the position (with fewer than ten such rows, their largest). Healthy rows grow about
twentyfold over a run (largest `dL/dq` row of a window in exp 346: 0.13 at step 500, 2.9 from
3,000 on), which is why an absolute number is loose early or tight late; `k = 3` acts on 0 % to
0.9 % of the live row-positions of exp 346's checkpoints at every stage (`probe/probe_ratio.py`).

Results (seed 2): the absolute limit `c = 4` without update caps collapsed at step 7,000 with the
router at norm 23 (exp 355), and with `model.address_init` it was at chance by step 1,000 (exp
354); the relative limit `k = 3` with the caps fell to chance at steps 5,000 and 6,500 and ended
at 68.2 % (exp 364). A limited row still carries its exploded *direction* at up to `k` times a
typical row's size through every earlier position of the window, and the router's drift is not a
state-gradient effect at all. The keys stay because those runs are in the database; they are off
by default and not recommended.

## Related: `train.weight_ema` (which weights are scored)

Not an equation change and not a change to training: an exponential moving average of the weights
(`hsn.train.weight_ema`), updated after every applied optimiser step,

    avg ← d_n · avg + (1 − d_n) · w,    d_n = min(decay, (1 + n) / (10 + n))

(the warm-up `d_n` makes the average follow the weights closely early in a run; it reaches `decay`
after 890 steps at 0.99 and 4,490 at 0.998). A run with it takes the same optimiser steps, bit for
bit on the CPU, as a run without it (`test_training_is_unchanged_and_the_evaluations_score_the_average`).
What changes is which weights are *scored*:

- `Trainer.evaluate` runs the averaged weights (swapped in place, so the compiled window sees
  them, and swapped back bit for bit); `evaluate(..., averaged=False)` scores the training weights;
- when a run ends the harness installs the average in the model, so the trained-state statistics
  and the probes describe the model the evaluations scored; the record carries `weight_ema` and
  `training_weights_final` (the training weights' own last evaluation, for comparison);
- a checkpoint's `model` stays the training weights and its `extra["weight_ema"]` carries the
  average, so a resume continues both exactly; `load_checkpoint(path, model, averaged=True)` is
  the evaluation-only reader (probe-check, probe-rerun, split-backfill and the re-score of a failed
  run use it). Loading a checkpoint without it gives the training weights, which score lower.

Why: with the cut the recipe no longer collapses, but its dev accuracy still moves 1.8 to 3.0
points from one evaluation to the next (0.4 for the GRU). Replaying exp 371 from step 15,000 with an
evaluation every 100 steps (`a5-fix/probe/ema_probe.py`), the training weights scored 70.0 to
83.2 % (mean 79.6 %, largest fall 10.8 points in 100 steps) and an average of them 83.5 to 86.4 %
(mean 85.3 % at 0.99; 85.2 % at 0.998). The routed cell is a bank of hard switches, so a few
optimiser steps move many routing and gate decisions at once; the average is the centre those
steps move around. Results on the controlled seeds: `docs/p1_18_a6_results.md`, section 3b.

Any model (the baselines accept it, and the comparison runs the GRU with it too). Buffers are not
averaged. Left out of the configuration hash at its default (`null`); refused together with
`train.rollback_every`, whose snapshots do not hold the average.

## Related: `model.address_init` (A5 extension, diagnostic)

The start value of the scorer's two address columns (`a_cur`, `a_prev`) for every unit; they stay
trainable. With 1.0 a route follows the byte codes from the first step. It speeds up the first
1,000 steps and does not last: a trainable router moves off those routes (exps 356, 363), and with
the router frozen on them (exp 366, `train.router_lr_mult = 0.001`) the key swap is followed in
30.3 % of cases, the most of any run, while the first query falls to 59.3 %. Opt-in, linear scorer
only, not part of the recommended recipe.
