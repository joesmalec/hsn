# A6 results — the A5 accuracy drops, the GPU pauses and the GRU bar (10 Nov 2026)

Branch `claude/a5-fix` (from `codex/a5-norm` at `38123a1`). Every score comes from the canonical
SQLite database `/home/joe/hsn-experiments/experiments.sqlite`; the replay evidence, probe scripts
and logs are in `/home/joe/hsn-experiments/a5-fix/`. Runs are 9,000 steps of the A5 recipe (top-K
routing, key addresses, selected-cell readout, lr 2.5e-3, β2 0.95, WSD, float64 gradient norms) on
the delayed task, B32, compiled, unguarded: no spike guard, no rollback, every document kept.

**What came out.** The mid-run accuracy drops are exploded gradients through the *feedback reads*
of the recurrent state, taken as full-size steps along one document's direction. Cutting those
reads out of the backward pass (`train.feedback_detach`, opt-in, forward unchanged) removes the
drops on all three controlled seeds, and with exp 346's anchored update caps the same recipe
finishes above the GRU:

| | Final dev | Later queries | Worst drop during the run | Skipped steps |
| --- | ---: | ---: | ---: | ---: |
| Before: anchored caps, full backpropagation (exps 343 / 326 / 346, seeds 0 / 1 / 2) | 73.8 / 78.2 / 77.5 % | 57.8 / 60.2 / 62.5 % | 12.0 / 4.8 / 36.9 pp | 0 |
| Before: current-norm caps (exps 342 / 325 / 345) | failed / 81.4 / 2.2 % | – / 65.3 / 2.2 % | 57.9 / 34.6 / 66.8 pp | 10 / 0 / 0 |
| **After: feedback cut + anchored caps (exps 360 / 361 / 349)** | **81.2 / 79.0 / 84.0 %** | **65.8 / 66.6 / 70.7 %** | **2.2 / 6.8 / 4.5 pp** | 0 |
| GRU, delayed task (gru_2m exp 154; delayed-only exp 347; 2e-3 exps 204, 206) | 79.3; 77.8; 78.6, 77.6 % | –; 58.3; 59.7, 57.6 % | 0.5 to 1.3 pp | 0 |

Mean of the three seeds: 81.4 % final and 67.7 % on later queries, against gru_2m's 79.3 % best run
(78.8 % five-run mean) and 58 to 61 % on later queries. Two of three seeds are above the 79.3 % bar;
seed 1 ends 0.3 points under it (peak 79.8 % at step 8,500). All three are above the 62.3 %
key-blind level on later queries, where the dev keys were never seen in training; none passes the
frozen binding verdict (swap follow 15 to 25 %, two "partial"). Learning is slower than under full
backpropagation: 0.6 dev is reached at step 4,000 to 4,500, against 2,500 to 3,000 for the GRU's
speed bar.

Over 27,000 steps the recipe reaches 90.0 % (later queries 82.1 %, exp 371), against 78.7 % for
the GRU and 70.2 % for the Transformer at 27k.

The GPU pauses had two causes. The long ones (4 to 5 s every 250 steps, and 79 s at the end of a
run) were the diagnostics and the probes running the eager routed forward; both now replay
compiled graphs (per 9k run: diagnostics 466 s → 12 s, probes 76 s → 10 s, exps 349 and 360). The
short ones that remained (0.4 to 0.5 s every 10 to 16 s) were full passes of Python's garbage
collector over the compiled graphs' objects, removed by freezing those objects (0.32 s → 0.002 s
per pass). Diagnostics can be switched off per sweep or per enqueue (`protocol.diagnostics`).

![Dev accuracy before and after](training_review/p1_18_a6_curves_2026_11_10.png)

(The GRU line is exp 347, the delayed-only run, evaluated every 500 steps: chance until step
2,000, 76.2 % at 2,500, then a plateau. Every stored delayed-only GRU run has that shape: exps 154,
204, 206. The smoother GRU curves of exps 334 to 336 are the five-task overall score, evaluated
every 3,000 steps.)

## 1. What the drops are

`docs/p1_17_training_findings_2026_11_10.md` left the cause open ("capture the onset ... target the
recurrent computation and routing"). I replayed training windows on the stored checkpoints of exps
345 (current-norm caps, before its collapse) and 346 (anchored caps, inside its 36.9-point drop),
weights frozen, and repeated the backward pass of the exploding windows with one path cut at a
time (`a5-fix/probe/probe_paths.py`, `p345_5000.json`, `p346_6000.json`; forward values identical
across variants).

| Cut in the backward pass | 346 @ 6,000, worst window | 345 @ 5,000, worst window |
| --- | ---: | ---: |
| nothing (the training gradient) | 159.2 | 29.0 |
| the selection's mask (straight-through) | 158.2 | 29.2 |
| the selection's unit weights | 159.6 | 29.0 |
| every scorer feature (`u`, `ε`, `h`) | 158.2 | 29.2 |
| the gate | 105.6 | 28.1 |
| `q → P` (the predictor's read of the summary) | 65.7 | 28.4 |
| `q → S` (the shared transform's read) | 37.5 | 4.7 |
| both reads of `q` | 6.9 | 4.7 |
| the cell's read of `u` and `ε` | 13.6 | 22.6 |
| the cell's read of its own `h` and the messages | 2.5 | 29.5 |
| an ordinary window, nothing cut | 2 to 3 | 1.8 |

What this shows:

- **Not the router's estimator.** Detaching the straight-through mask, the unit weights or all
  scorer features leaves an exploding window's gradient unchanged. The paths that matter are the
  *feedback reads* of the recurrent state: the summary `q` read by `P` (through `ε`) and by `S`
  (through `u`), and a cell's read of its own state and its neighbours' messages.
- **Jumps, not growth.** The state gradient is flat along the window and jumps 7× to 22× at single
  positions (346, window 67: 0.59 for every position after 14, 13 at position 13, 27 before
  position 9). Several jumps in one document multiply.
- **A cell of hard switches.** From step 1,500 on, 95 to 99 % of the candidate's coordinates are
  saturated (`|c| > 0.99`) and the gate is either closed or fully open (1 to 2 % open, mean 0.015).
  Such a cell passes almost no gradient anywhere and a very large one where a coordinate is in
  transition. The forward dynamics are just as sharp: a 1e-3 perturbation of the state at position
  32 has grown 120 to 1,000 times 96 positions later at step 6,000 and 220 to 830 times at the
  healthy step 4,500 (10 to 330 times at steps 1,500 and 8,500; `prof346_*.json`).
- **One document takes the step.** The exploding document's gradient is 10 to 10⁹ times the
  others'. The global clip rescales the batch by that norm, so the other 31 documents vanish from
  the step and AdamW moves at full size along one document's switching direction. A burst is some
  hundreds of such steps (346: a tenth of the steps above a norm of 350 in steps 5,751 to 6,000
  and above 4,300 in 6,001 to 6,250, against a run median of 2).
- **A second mechanism, in the router.** Without the update caps the router's weights grow
  without bound and its routes churn (345: norm 0.7 → 16 before the collapse; exp 355 and 356
  below: 15 and 19 by step 4,500). Top-K always writes K cells; most positions are filler, where a
  write can only hurt, so the selected cells receive a steady "should not have been selected"
  gradient and the router keeps moving away from whatever feature currently picks them. The caps
  hold this back (they cut the router's step about tenfold and the cell's about twofold: 346's
  proposed update ratios are 0.027 and 0.0075 against caps of 0.003).
- **A4 is not the answer here.** With the candidate norm the candidate is not saturated, and the
  same feedback loops compound smoothly instead of in jumps: exp 350 (A4, full backpropagation, no
  caps) reached gradients of 10³⁶ to 10³⁸ and non-finite values at step 1,479.

## 2. What was changed

All opt-in, left out of the configuration hash at the defaults, forward values unchanged; the
equations of §2.2–§2.6 run as before (`docs/amendments/A6_feedback_gradient.md`).

- **`train.feedback_detach`** — the named feedback reads are constants in the backward pass. With
  `[summary, eps, state, router]` the state's backward recurrence is the gated carry `(1 − b) ⊙ h`
  alone, which cannot amplify (tested against the exact product of the gates). The memory path of
  this task (write from `u`, hold by the carry, read by the decoder) keeps its full gradient.
- **`train.state_grad_ratio`** / **`train.state_grad_clip`** — every path kept; at each position a
  document's gradient into the state is limited to a multiple of the batch's typical document
  there (or to an absolute number). A healthy window is plain backpropagation through time.
- **`model.address_init`** — the start value of the scorer's two address columns (A5 extension).

Framework: the diagnostics' state window and the end-of-run probes replay compiled graphs instead
of the eager forward; records carry `phases`; `scripts/gpu_phase_report.py` attributes the card's
time series (section 4).

## 3. Runs

Seed 2 is the seed of exp 345 (collapse) and exp 346 (36.9-point drop). "Worst drawdown" is the
largest fall from an earlier peak over the saved evaluations (every 500 steps). Arms shared the
card three at a time, so step times are not timings. `final` of a failed or cancelled arm is not a
score; its last evaluation is given.

**The recommended recipe** — `train.feedback_detach=[summary,eps,state,router]` with exp 346's
caps (`cell.*` and `router.*` at 0.003, the router's radius anchored to its initial norm):

| Exp | Recipe | Final dev | Later | First | Worst drawdown | Skips | Swap follow / unchanged | Binding | Router norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 360 | **cut + caps**, seed 0 | 81.2% | 65.8% | 98.5% | 2.2 pp | 0 | 15.5% / 64.8% | does not bind keys | 1.84 |
| 361 | **cut + caps**, seed 1 | 79.0% | 66.6% | 92.9% | 6.8 pp | 0 | 24.3% / 44.5% | partial | 2.19 |
| 349 | **cut + caps**, seed 2 | 84.0% | 70.7% | 98.9% | 4.5 pp | 0 | 24.6% / 53.1% | partial | 2.01 |

**What each part does** (seed 2):

| Exp | Recipe (seed 2) | Final dev | Later | First | Worst drawdown | Skips | Swap follow / unchanged | Binding | Router norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 346 | caps only (before) | 77.5% | 62.5% | 94.3% | 36.9 pp | 0 | 9.1% / 73.6% | does not bind keys | 2.67 |
| 348 | cut only | 78.2% | 59.9% | 98.9% | 4.0 pp | 0 | 11.0% / 68.0% | does not bind keys | – |
| 349 | **cut + caps** | 84.0% | 70.7% | 98.9% | 4.5 pp | 0 | 24.6% / 53.1% | partial | 2.01 |
| 368 | `[summary, eps]` cut + caps | 76.3% | 56.5% | 98.7% | 4.0 pp | 0 | 3.5% / 90.0% | does not bind keys | 4.25 |
| 364 | `state_grad_ratio=3` + caps | 68.2% | 47.2% | 91.9% | 71.5 pp | 0 | 6.2% / 80.5% | does not bind keys | 5.79 |
| 355 | `state_grad_clip=4`, no caps (collapsed at 7,000; cancelled at 8,220) | – | – | – | 52.7 pp | 0 | – / – | – | – |
| 354 | `state_grad_clip=4` + `address_init=1`, no caps (at chance from 1,000; cancelled) | – | – | – | 19.1 pp | 0 | – / – | – | – |
| 356 | cut + `address_init=1`, no caps (drifting down; cancelled at 5,177, last evaluation 30.2 %) | – | – | – | 14.5 pp | 0 | – / – | – | – |
| 363 | cut + caps + `address_init=1` | 77.4% | 59.9% | 97.1% | 14.3 pp | 0 | 10.9% / 74.5% | does not bind keys | 23.08 |
| 366 | cut + cell cap + `address_init=1`, router frozen (`router_lr_mult=0.001`) | 63.1% | 66.5% | 59.3% | 13.5 pp | 0 | 30.3% / 49.3% | partial | 4.06 |
| 350 | A4, full backpropagation, no caps (non-finite at 1,479; stopped at 1,487) | – | – | – | 0.0 pp | 10 | – / – | – | – |

The router norm is the last diagnostic sample's; runs without an update limiter do not record it
(348 ends at 10.2, measured from its checkpoint).

- *The cut alone* (348) is as smooth as with the caps but stays at the key-blind level: its router
  grows to a norm of 10 (0.7 at the start; 2.0 with the caps) and later queries do not improve.
- *The caps alone* (346, and 343 / 326 on the other seeds) do not remove the drops.
- *Cutting only the summary's readers* (`[summary, eps]`, 368) is as smooth as the full cut on this
  seed but ends at the key-blind level.
- *A limit on the state gradient instead of the cut* does not remove them either. An absolute
  per-document limit (355) collapsed at step 7,000 with the router at norm 23; the scale-free
  limit with the caps (364) learned faster than any other arm here (0.67 at step 2,500, 0.73 at
  4,000) and then fell to chance twice (steps 5,000 and 6,500). Both keys stay in the code, opt-in, because these runs are in the
  database; neither is a fix.
- *Starting the address columns at 1* (`model.address_init`) gives a fast start (0.52 at step
  1,000 in 363) that does not last: the router moves away from the address routes (in 356 the
  `a_prev` column goes from 0.9 to 0.0 and the other columns' norm from 2.9 to 18.8 by step
  4,500). With the router all but frozen on those routes (366) the
  key swap is followed more often than in any other run (30.3 %, unchanged 49.3 %) and later
  queries are at 66.5 %, but the first query falls to 59.3 % and the run ends at 63.1 %. So the
  memory does follow the key when the routes are consistent; a write route and its query's read
  route share only about a third of their cells under this prior (0.34 simulated, 0.125 by
  chance), which is not enough.
- *A4 with full backpropagation* (350) is non-finite by step 1,479.

**The full run** (exp 371: the recipe on seed 2 for 27,000 steps, alone on the card):

| | Final dev | Later | First | Worst drawdown | Skips | Swap follow / unchanged | Binding |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Routed HSN, cut + caps, 27k (exp 371) | 90.0% | 82.1% | 98.9% | 5.4 pp | 0 | 44.0% / 29.3% | partial |
| GRU 2M, 27k, five-task mix (exp 336), delayed task | 78.7% | 59.9% | – | – | 0 | – | – |
| Transformer 2M, 27k, five-task mix (exp 339), delayed task | 70.2% | 50.7% | – | – | 0 | – | – |
| Transformer 2M, 27k, delayed only (exp 262) | 49.9% | 31.9% | – | – | 0 | – | – |

It reaches 0.6 dev at step 3,500, holds 74 to 85 % from step 8,500 to the start of the decay
(21,600) and is still rising at the end (86.3 % at 23,000, 88.7 % at 24,500, 90.0 % at 27,000).
The gain over the 9k runs is on later queries (70.7 % → 82.1 %) and the key swap is followed
almost twice as often (24.6 % → 44.0 %), still under the frozen 50 % threshold. One seed.

## 3b. The wobble, and scoring averaged weights

With the cut the recipe no longer collapses, but its dev accuracy still moves from one evaluation
to the next (`a5-fix/wobble.py`, evaluations after step 3,000):

| Run | Typical change between evaluations | Largest fall |
| --- | ---: | ---: |
| before, caps only (exp 346) | 6.5 pp | 31.9 pp |
| cut + caps (exps 360 / 361 / 349) | 1.8 / 1.8 / 3.0 pp | 2.2 / 1.2 / 4.0 pp |
| cut + caps, 27k (exp 371) | 1.9 pp | 5.4 pp |
| GRU (exp 347) | 0.4 pp | 0.7 pp |

It is noise of the last few optimiser steps, not of the solution. Replaying exp 371 from step
15,000 with an evaluation every 100 steps (`a5-fix/probe/ema_probe.py`, 14 evaluations):

| Weights scored | Mean dev | Range | Typical change | Largest fall |
| --- | ---: | ---: | ---: | ---: |
| the training weights | 79.6% | 70.0 to 83.2% | 3.2 pp | 10.8 pp |
| their moving average, decay 0.99 | 85.3% | 83.5 to 86.4% | 0.7 pp | 1.7 pp |
| their moving average, decay 0.998 | 85.2% | 81.0 to 87.1% | 0.7 pp | 0.9 pp |

`train.weight_ema` (opt-in; the amendment's "which weights are scored") makes the average what a
run's evaluations and probes score; training is unchanged step for step. At the end of a run,
after the learning-rate decay, the two coincide (exps 375 to 377: 85.9 / 86.5 / 86.5 % averaged,
85.9 / 86.7 / 86.4 % for the training weights), so the average changes the curve, not the final
score. The GRU has not been run with it yet (`p1_18_a6_ema_gru`, queued behind the work below).

One change at a time on seed 2 (`p1_18_a6_wobble`, against exp 349's 84.0 %): a peak rate of
1.5e-3 has no fall at all but learns more slowly and ends at 80.7 % (exp 373); a cosine schedule
ends at 65.3 % (exp 374); the candidate norm is the next section.

## 3c. With the candidate norm (A4)

Under full backpropagation the candidate norm of amendment A4 (`model.cand_norm=rms`) diverged
(exp 350). With the feedback cut it trains without a burst (largest gradient norm of exp 372:
4.1), learns faster than any other arm, and ends higher:

| Exp | Recipe, 9,000 steps | Final dev | Later | First | 0.6 dev at | Swap follow / unchanged | Binding |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 372 | cut + caps + A4, seed 2, training weights | 88.9% | 80.5% | 98.4% | 2,000 | 61.1% / 19.9% | partial |
| 375 | cut + caps + A4, seed 0, averaged weights | 85.9% | 76.7% | 96.3% | 1,500 | 42.0% / 34.0% | partial |
| 376 | cut + caps + A4, seed 1, averaged weights | 86.5% | 76.3% | 98.1% | 1,500 | 52.9% / 28.0% | partial |
| 377 | cut + caps + A4, seed 2, averaged weights | 86.5% | 75.8% | 98.6% | 1,500 | 36.8% / 40.0% | partial |
| 349 / 360 / 361 | cut + caps, without A4 | 84.0 / 81.2 / 79.0% | 70.7 / 65.8 / 66.6% | – | 4,000 to 4,500 | 15 to 25% | – |

Three seeds: 86.3 % mean (85.9 to 86.5), 76.3 % on later queries, against 81.4 % and 67.7 %
without A4 and the GRU's 79.3 % best run. No step skipped.

**But every run sags mid-run.** After reaching 72 to 81 % by step 2,000, each falls 10 to 21
points between steps 2,000 and 4,000 and then recovers (worst drawdown 14.0 / 10.9 / 21.3 /
20.0 points for exps 372, 375, 376, 377). Seeds 1 and 2 drop back under 0.6 dev during it, so the
speed bar is passed and then lost. Gradient norms stay ordinary throughout; this is not the burst
of section 1.

## 3d. What the sag is: the router takes the value routes away

`route-check` and two read-only probes on the kept checkpoints (`a5-fix/routecheck/`,
`a5-fix/probe/route_flip.py`, `address_columns.py`):

| Run @ step | Query route ∩ its fact's value-write route | ∩ other facts' |
| --- | ---: | ---: |
| 375 @ 2,000 → 3,500 | 0.47 → 0.16 | 0.13 → 0.12 |
| 376 @ 2,000 → 4,000 → 9,000 | 0.51 → 0.17 → 0.14 | 0.10 → 0.12 → 0.13 |
| 377 @ 2,000 → 3,000 | 0.44 → 0.19 | 0.12 → 0.11 |
| 372 @ 9,000 | 0.16 | 0.13 |
| 386 @ 5,000 (binds keys, below) | 0.62 | 0.10 |

At step 2,000 a value is written into the cells its key selects: the key-address mechanism of A5
binds. By steps 3,000 to 4,000 it no longer does, and it does not come back; the runs recover to
86 % by a weaker strategy.

How it is lost (exp 376): the positions where a fact's value is written are served by two
dedicated router units (their weight is on the *previous* byte's address, the key); every other
position by one common unit. That unit then takes the value positions over:

| Step | Value-position slots on the dedicated units | On the common unit | Binding overlap (train documents) |
| ---: | ---: | ---: | ---: |
| 1,500 | 63% | 38% | 0.56 |
| 2,000 | 44% | 55% | 0.46 |
| 2,500 | 23% | 76% | 0.31 |
| 3,000 | 14% | 85% | 0.22 |
| 4,000 | 5% | 95% | 0.17 |

In every run inspected (376, 389, 396, 398) the unit that takes them is the one that reads the
query keys. Its weight on the *current* byte's address grows at a steady pace for the whole run
(0.11 → 0.55 in exp 376, 0.13 → 0.62 in exp 389) while the value units' weights level off (0.15
to 0.29); when its scores overtake theirs, top-K hands it the positions, the value units stop
receiving a gradient there and cannot win them back. For a given class of positions the switch
can be abrupt: in exp 386 the training loss doubled within ten steps (2,801 to 2,810) at ordinary
gradient norms. The takeover is not tied to a step count.

**Fixed settings delay it and do not prevent it** (seed 1; all scored on averaged weights):

| Exp | Change against exp 376 | Steps | Peak dev (step) | Final dev | Later | Swap follow / unchanged | Binding |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 376 | – | 9,000 | 81.2% (2,000) | 86.5% | 76.3% | 52.9% / 28.0% | partial |
| 386 | cell and router caps halved (0.0015) | 5,000 | 89.5% (5,000) | 89.5% | 87.8% | 91.1% / 1.7% | **binds keys** |
| 384 | peak rate 1.5e-3 | 5,000 | 87.3% (4,500) | 82.4% | 79.3% | 84.5% / 6.7% | **binds keys** |
| 385 | caps on encoder, shared transform, predictor too | 5,000 | 73.9% (5,000) | 73.9% | 55.9% | 43.5% / 37.9% | partial |
| 388 | cell and router caps halved, full length | 9,000 | 93.1% (3,000) | 84.6% | 75.0% | 70.7% / 15.0% | partial |
| 387 | peak rate 1.5e-3, full length | 9,000 | 93.7% (4,000) | 73.1% | 64.1% | 67.9% / 14.9% | partial |
| 389 | `train.adaptive_lr` (loss-triggered rate cuts) | 5,000 | 97.4% (3,000) | 89.0% | 89.0% | 88.5% / 2.5% | **binds keys** |

While the value routes hold, the model binds held-out keys: exps 384, 386 and 389 are the first
runs to pass the frozen key-swap verdict, and later-query accuracy reaches 85 to 96 % (key-blind
level 62.3 %). They are stopped at step 5,000. Over the full length the same two settings lose
the route later (exps 387, 388) and end at or below the reference. The existing loss-triggered
controller (exp 389) cut every rate 4 to 32 times after its training monitor's loss jumped at
step 2,700, and the takeover went on at the same pace: the router's proposed step is about ten
times its update cap, so a rate cut does not change the step it actually takes. With the router
held still from step 1,500 of exp 376 (`probe/freeze_probe.py`) the run stays at 79 to 83 %
through step 2,750 and is at 74.9 % at 3,250, where the original is at 63.5 % by 3,000: most of
the damage is the router's own movement, not all of it.

## 4. The GPU pauses

`scripts/gpu_phase_report.py` on exp 346's card time series (alone on the card, one sample per
second; phases estimated from the reported step, since the run predates `phases`): 97 of the 961
samples during the fit are below 50 % utilisation (10.1 %), 78 of them within 30 steps of a
diagnostic or evaluation step; the GRU run 347 has none. The pauses are the 250-step diagnostics:
each sample ran the eager routed forward on four documents (0.85 s, twice per sample with an
update limiter) and a batched GPU SVD of 713 32×64 matrices for the participation ratio (0.47 s;
0.05 s on the CPU), 3.7 s per sample with the stability observer, 133 s of the run. The three
end-of-run probes ran the same eager forward for 79 s at about 25 % utilisation.

Changes (all values equal to the eager ones to rounding; an eager trainer is unchanged):

- the diagnostics' state window replays a compiled no-gradient variant of the training window
  that also returns the per-position states and gates (`CompiledWindow(..., capture=True)`),
  compiled once before training starts;
- the participation ratio is computed from the Gram matrix (`(Σσ²)² / Σσ⁴ = ‖H‖_F⁴ / ‖HHᵀ‖_F²`), no
  decomposition;
- the probes pad each batch to the evaluation window's shape and replay that graph
  (`PaddedWindow`), no new compile.

| Same recipe, same load (three arms on the card) | Diagnostics | Probes (v1 + v2 + v3) | Evaluations |
| --- | ---: | ---: | ---: |
| exp 349, eager diagnostics and probes | 465.7 s | 76.4 s | 62.7 s |
| exp 360, compiled diagnostics and probes | 12.4 s (0.36 s per sample) | 10.4 s | 34.7 s |

Alone on the card (exp 371, 27,000 steps): 108 diagnostic samples took 8.5 s in total (0.08 s
each; 3.7 s each in exp 346) and the three probes 5.0 s (78.9 s).

**What was left, and the second fix.** A 5 Hz trace of exp 371 still showed half-second dips to
0 % every 10 to 16 s (2.8 % of the time below 50 %), tied to no training event. An instrumented
solo run (`a5-fix/probe/dip_probe.py`, 1,400 steps: every step's wall time, every collector pass,
the card at 20 Hz) attributes every one of them to a **full pass of Python's garbage collector**:
0.28 to 0.35 s each over about 980,000 tracked objects (the compiled graphs, their guards, the
modules), 12 of them in 1,340 steps, each an idle stretch of 0.35 to 0.5 s on the card. Plain
steps are otherwise steady (median 99.1 ms, 99th percentile 100.3 ms). With the long-lived objects
moved to the collector's permanent generation after an evaluation (`gc.freeze()`; protocol
`gc_freeze`, on by default for compiled runs, undone when the fit ends) a full pass takes 1 to
2 ms and the idle stretches are gone:

| Steps 541 to 1,400, alone on the card | Full collector passes | Idle stretches on the card | Steps over 150 ms (statistics steps aside) |
| --- | ---: | ---: | ---: |
| before (`dip_base.json`) | 7 × 0.32 s | 6 (0.35 to 0.51 s) | 9 |
| frozen after step 520 (`dip_freeze.json`) | 50 × ≤ 0.002 s | 0 | 2 |

What still interrupts the card after both fixes: an evaluation every 500 steps (0.8 s, mostly
busy), a statistics step every 100 steps (0.28 s, busy: two extra passes for the per-term gradient
norms) and the steady 88 % between them (the per-position launches of a small recurrent model).
P2.10's fused kernels would shorten the step; they do not touch any of the pauses above.

**The diagnostics switch.** `protocol: {diagnostics: false}` (or `enqueue --diagnostics off`)
takes no diagnostic sample during training at all: no hooks, no state window, nothing compiled for
them. It is execution-only: the training, the arm id, the evaluations, the kept checkpoints, the
end-of-run state statistics and the probes are the same, and a rule that reads a `diag:` value is
refused at expansion rather than left without data. Default on.

Not changed: the stability observer (`diag_stability`, opt-in) still perturbs the state inside an
eager forward; the one-off `trained_state` window at the end is eager (3 to 6 s); the first
compile of a run is unchanged.

## 5. What is not established

- **Binding.** No run passes the frozen key-swap verdict (follow ≥ 50 %, unchanged ≤ 10 %). The
  later-query gain over the key-blind level is 3 to 8 points. `route-check` on exps 346, 348 and
  349 (`a5-fix/routecheck/`) reads the same on all three: on dev keys a query's route overlaps the
  route at its fact's *key* position by 0.89 to 0.90 and the route at the fact's *value* position
  by 0.19 to 0.32, against 0.14 to 0.25 for the other facts' value positions (train keys read the
  same). The routes follow the current byte's address and not the previous byte's, so a value is
  not written into its key's cells. That is the router's learning (P1.17's other half), not
  stability.
- **Why the caps help the cut.** 349 against 348 is one seed. The caps cut the router's step
  about tenfold and the cell's about twofold; which of the two matters was not separated.
- **Speed.** The cut learns more slowly than full backpropagation (0.6 dev at 4,000 to 4,500
  steps against 2,500). Cutting only the summary's two readers (368) is as smooth
  (4.0-point worst drawdown) and a little faster in the middle of the run, but ends lower (76.3 %,
  later 56.5 %). No arm here is both fast and smooth.
- **Seeds and repeats.** Three seeds, one run each; execution is not bit-reproducible on this
  card (exps 299 / 300), and no arm was repeated.
- **The forward dynamics are unchanged.** The cell is still a bank of hard switches and still
  sensitive to its state; the cut stops the optimiser from following those directions, it does not
  make the cell smooth.
- **Other tasks and the dense HSN** were not run with the cut. The comparison with the
  Transformer and the GRU above is on the delayed task only, where the HSN arms trained on that
  task alone and the 27k baselines on the five-task mix (the delayed-only Transformer is lower).
- **Cost.** The routed HSN step is 0.099 s against 0.014 to 0.016 s for the GRU and the
  Transformer, with 2.1 GB against 0.7 GB of GPU memory.
- **The wobble.** Between evaluations the recipe's dev accuracy moves 1.8 to 3.0 points (largest
  falls 1.2 to 5.4), against 0.4 for the GRU and 0.7 for the dense HSN; see section 3b.

## 6. Reproduce

```bash
cd /home/joe/hsn-a5-fix                      # branch claude/a5-fix
python -m hsn.experiments enqueue configs/sweeps/p1_18_a6_seeds.yaml --mem-estimate-mib 3000
python -m hsn.experiments enqueue configs/sweeps/p1_18_a6_screen.yaml --only "carry_anchored/s2"
setsid nohup python -m hsn.experiments worker --max-concurrent 3 --require-clean \
    --exit-when-empty --telemetry-max-samples 20000 > worker.log 2>&1 &
python scripts/gpu_phase_report.py --exp 346 --exp 360
```

Checks on the branch tip: 1,111 CPU tests pass (144 device skips); the CUDA default-training
golden, the compiled-window and the A5 CUDA tests pass (18); ruff, ruff-format and mypy clean.
Every new key is left out of the configuration hash at its default.
