# Amendment A6 — the feedback gradient (`train.feedback_detach`)

Status: opt-in, default off (10 Nov 2026). No forward value changes: a training-recipe choice about
which paths the backward pass follows through the recurrent state. Runs and evidence:
`docs/p1_18_a6_results.md`.

**Tested on the A5 recipe, delayed task, 9,000 steps, three seeds** (exps 360, 361, 349), with exp
346's anchored update caps: final dev 81.2 / 79.0 / 84.0 %, later queries 65.8 / 66.6 / 70.7 %,
worst drop during a run 2.2 / 6.8 / 4.5 points, no step skipped. The same recipe with full
backpropagation (exps 343, 326, 346): 73.8 / 78.2 / 77.5 %, worst drop 12.0 / 4.8 / 36.9 points.

**With the candidate norm of A4 on top** (`model.cand_norm=rms`, which diverged under full
backpropagation and trains with the cut), nine runs on six seeds end at 79.1 to 86.5 % (mean
83.9 %; exps 375 to 377, 418 to 420, 433 to 435), eight of them above the GRU's best run, and
seven of them with a mid-run sag of ten points or more. With `train.route_guard` (below) nine
runs on the same seeds end at 78.5 to 95.6 % (mean 87.2 %; exps 421 to 426, 430 to 432), one of
them with such a sag. The tested recipe is therefore the cut, the anchored caps, A4,
`train.weight_ema=0.99` and the route guard at its defaults; each part is opt-in.

**Adopted as the standard recipe for new HSN specs on 11 Nov 2026 (the user's decision):**
`configs/experiments/hsn_recipe_a6.yaml`, which resolves to the configuration of exps 421 to 426
(`docs/sweeps.md`, "Recipes"). The code's defaults are unchanged: a configuration that does not
name these keys trains as before.

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

## Related: `train.route_guard` (acting on a route takeover when it happens)

Not an equation change: a training-time controller on the router's step (`hsn.train.route_guard`,
opt-in, routed HSN only). Evidence and runs: `docs/p1_18_a6_results.md`, sections 3d and 3e.

**The event it watches for.** With the cut and the candidate norm of A4 the model learns to write
a fact's value into the cells its key selects (A5's binding) by step 2,000, and then loses it:
the router unit that reads the query keys keeps growing its weight on the current byte's address
for the whole run, whatever the loss or the learning rate, while the units that serve the
value-write positions level off; when its scores overtake theirs, top-K hands it those positions
and they cannot be won back. Fixed settings (smaller caps, a lower rate) only move the moment;
a loss-triggered rate cut does not reach the router, whose proposed step is about ten times its
update cap.

**What it does.** From every training window it counts the selection slots per (input byte,
unit); no extra forward pass, no knowledge of the task. Every `interval` steps:

    reference[b, u]  the largest share unit u has had of byte b's slots since the last event
    established      routes (b, u) that held at least `specialised` of b's slots at `persistence`
                     consecutive readings
    lost             the mean, over the quarter of the bytes that lost most, of the share of
                     their established routes that is gone

If `lost ≥ lost_fraction` **and** the interval's mean training loss is above the lowest interval
mean so far by `loss_rise` (units sorting positions out while the loss falls is learning), the
applied step of the unit that gained most on those bytes is multiplied by `reduce_factor` on its
rows of the scorer's output layer: `row ← old + scale · (new − old)`, after the optimiser step and
the caps. Every other unit and every other module keeps its full step. No step is skipped, no
document dropped, nothing rolled back; a run in which no event fires trains bit for bit as
without the guard (`test_a_quiet_guard_changes_no_training_step`), the state is checkpointed
(`test_training_with_events_resumes_exactly`) and every event is in the record (`route_guard`).

    train.route_guard:
      enabled: false        # default; hash-omitted
      interval: 100         # steps between readings
      lost_fraction: 0.1    # the event
      specialised: 0.25     # a route is a unit holding this share of a byte's slots ...
      persistence: 3        # ... at this many consecutive readings
      loss_rise: 0.02       # ... and training no longer improving by this margin
      min_slots: 100        # a byte with fewer slots in an interval is left out
      min_usage: 0.001      # no event while established routes hold less of all slots
      reduce_factor: 0.5    # the gaining unit's step scale, per event
      min_scale: 0.015625   # ... down to this

(A `restore` key, which also rewound the gaining unit's rows by one reading, existed on 10 and 11
Nov and was removed: exp 400 ended at 77.9 % against 90.3 % for the guard alone. A stored
configuration with `restore: false` still loads and keeps its hash.)

**What it watches, precisely.** The units that serve the supervised positions are read from the
loss mask (units with at least `specialised` of the slots at loss positions); in every run
inspected the unit that takes the value routes is one of them. A reading is an event only when
such a unit holds a larger share of some input bytes than it ever has while other units'
established routes on those bytes shrink. Routes that move between other units (the filler's do,
constantly) are not events: an earlier version that counted them slowed the wrong units.

**Results** (9,000 steps, six seeds, against the same configuration without the guard): 86.2 /
93.3 / 88.9 / 95.6 / 84.9 / 85.3 % against 85.9 / 86.5 / 86.5 / 84.7 / 82.1 / 84.6 %; worst
drawdown 5.8 / 1.8 / 8.7 / 1.5 / 5.2 / 6.0 points against 10.9 / 21.3 / 20.0 / 5.1 / 13.7 / 14.7;
three of the six pass the key-binding verdict against one. A second run of seeds 0 to 2: 85.1 /
78.5 / 87.3 % against 79.1 / 86.4 / 79.4 % (the guarded seed 1 sagged 13.7 points that time).
Over the nine runs each way: 87.2 % against 83.9 %, and a fall of ten points or more in one run
against seven. Over 27,000 steps (one seed) it removes the early sag and ends at the same score.

**Limits.** It needs a scorer with one output per unit (the linear scorer, or the two-layer
scorer's second layer). A unit whose step has been halved to the floor no longer learns new
routes; an event that comes while a route is still forming slows that run. It slows the unit
that takes a route and does nothing about the units that lose it un-learning it, so a slower
erosion remains (5 to 9 points on four of six seeds; visible over 27,000 steps). Tested on one
task, where the supervised positions are a query's key. Two earlier versions did not work (exps
396 to 398, 402 to 404) and are described in the results.

**Open: the remaining erosion.** Left for now by the user's decision of 11 Nov; what to try is
in `docs/p1_18_a6_results.md`, section 5b. In short: the router's proposed step is about ten times
its cap, so it moves at a fixed pace whatever its gradient, and the units that lose a route also
un-learn it. The proposal is to measure which gradient term does the un-learning, to try the
existing straight-through temperature and exploration controls, and then an amendment that makes
the router's applied step follow the size of its gradient (or bounds each unit's score row), so
that a learned route stops moving. None of it has been run.

## Related: `model.address_init` (A5 extension, diagnostic)

The start value of the scorer's two address columns (`a_cur`, `a_prev`) for every unit; they stay
trainable. With 1.0 a route follows the byte codes from the first step. It speeds up the first
1,000 steps and does not last: a trainable router moves off those routes (exps 356, 363), and with
the router frozen on them (exp 366, `train.router_lr_mult = 0.001`) the key swap is followed in
30.3 % of cases, the most of any run, while the first query falls to 59.3 %. Opt-in, linear scorer
only, not part of the recommended recipe.
