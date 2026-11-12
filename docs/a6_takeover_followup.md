# The route takeover of the A6 recipe: measurement and remedies

11 November 2026, branch `p2.10-cuda-kernels`. This is the follow-up that section 5b of
`docs/p1_18_a6_results.md` describes ("The remaining erosion: what to try next"), in its order:
the measurement first, then the white paper's own mechanisms as configuration-only arms. Section
5b is not edited here. Runs: `/home/joe/hsn-experiments/p2.10-split/experiments.sqlite`
(`remedy.py` and `table.py` beside it print the tables), the standard recipe
`configs/experiments/hsn_recipe_a6.yaml` (guard on), 9,000 steps, diagnostics off, several arms
on the card under NVIDIA MPS.

## 1. The recipe run again: what the falls are

Same-day controls of the recipe on the autograd window (`configs/sweeps/p2_10_autograd_seeds.yaml`;
the arms of exps 421 to 426 with diagnostics off):

| Seed | Exp | Dev at step 2,000 | 4,000 | 7,000 | Final dev | Later queries | Worst drawdown | Guard events |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7 | 73.2 % | 84.3 % | 83.4 % | 88.0 % | 79.8 % | 10.4 pp | 35 |
| 1 | 8 | 87.6 % | 91.5 % | 89.1 % | 91.4 % | 88.7 % | 3.5 pp | 58 |
| 2 | 9 | 91.1 % | 85.7 % | 90.4 % | 77.1 % | 65.3 % | 14.4 pp | 54 |
| 3 | 13 | 81.8 % | 92.2 % | 90.2 % | 90.6 % | 89.4 % | 3.7 pp | 48 |
| 4 | 14 | 64.1 % | 67.8 % | 73.8 % | 74.4 % | 59.4 % | 2.4 pp | 4 |
| 5 | 15 | 57.9 % | 54.0 % | 64.6 % | 62.6 % | 40.4 % | 11.2 pp | 5 |
| mean | | | | | 80.7 % | 70.5 % | | |

Last night's nine runs of the same configuration (exps 421 to 426, 430 to 432; commit c5a82ea,
diagnostics on, three arms on the card) averaged 87.2 % and 80.8 %; their lowest at step 4,000 was
82.3 %. Today's six are lower on three seeds: a late fall (seed 2) and two runs that never reach
their routes (seeds 4 and 5). The difference of the means is not significant at these counts (run
to run, one configuration differs by up to 15 points), and whether it is the code is checked two
ways in section 4: last night's commit run again today, and the two commits trained side by side
in float64.

Two kinds of fall, one event. `route_flip.py` (the read-only probe of section 3d) on the kept
checkpoints:

| Run | Step | Value-position slots on the dedicated unit(s) | On the unit that reads the query keys | Binding overlap |
| --- | ---: | ---: | ---: | ---: |
| seed 2 (exp 9), late fall 90.4 % → 77.1 % | 6,500 | 52 % | 48 % | 0.52 |
| | 7,000 | 38 % | 62 % | 0.44 |
| | 7,500 | 36 % | 64 % | 0.37 |
| | 8,000 | 15 % | 85 % | 0.23 |
| | 8,500 | 8 % | 91 % | 0.19 |
| | 9,000 | 7 % | 93 % | 0.18 |
| seed 4 (exp 14), 63 % at step 4,000 | 1,500 | 37 % | 63 % | 0.45 |
| | 2,000 | 14 % | 82 % | 0.36 |
| | 2,500 | 4 % | 92 % | 0.34 |
| | 3,000 | 0 % | 95 % | 0.32 |

In both the unit that serves the query keys takes the positions where a fact's value is written,
and the key no longer selects the cells its value went into. In seed 2 that unit's router rows
had been at the guard's floor (1/64 of their step) since step 2,700; the guard fired 54 times.
A run that is "stuck" near 60 % is the same event before the routes were ever established.

## 2. Which gradient does it (section 5b, step 1)

`evidence/P2.10/sag/sag_grad.py`, read-only: from a kept checkpoint's weights, real training
windows run through the split window's replay (the model's own gradients on one graph, weights
frozen), and the gradient of each loss term with respect to the scorer's logits is taken whole
and by path (through the straight-through cell mask; through the straight-through unit weights),
summed over the cells and averaged per kind of position. Below, the *push* on a unit's logit
(minus the gradient, ×1e-6 per position; positive: gradient descent raises the unit's score
there), task loss, four windows of sixteen rows per checkpoint (about 220 value positions and 130
query-key positions each, so single entries are noisy).

Seed 4 (exp 14), the taker is unit 0, the dedicated value units are 1 and 2:

| Step | Value-position slots, unit 0 / 1 / 2 | Push at the value positions: unit 0 | unit 1 | unit 2 | of which unit path: 0 / 1 / 2 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1,000 | 53 % / 31 % / 15 % | +3 | +26 | −314 | +16 / +27 / −119 |
| 1,500 | 63 % / 27 % / 10 % | +243 | +87 | +150 | +86 / +8 / +44 |
| 2,000 | 82 % / 9 % / 6 % | +715 | −50 | −19 | +342 / −81 / −61 |
| 3,000 | 94 % / 0 % / 0 % | +305 | −44 | −36 | +148 / −44 / −36 |

Seed 2 (exp 9), the taker is unit 7 (the reader), the dedicated value unit is 4:

| Step | Value-position slots, unit 4 / 7 | Push at the value positions: unit 4 | unit 7 | Push at the query-key positions: unit 7 | unit 4 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3,000 | 61 % / 33 % | +195 | +18 | +413 | −4 |
| 6,500 | 56 % / 44 % | +24 | +237 | −970 | −348 |
| 7,500 | 35 % / 65 % | +178 | −25 | +292 | −103 |
| 8,500 | 8 % / 91 % | −101 | +721 | −496 | +26 |

What the tables say:

- **The straight-through gradient credits the unit that ran.** At a kind of position, the unit
  that holds most of the slots is pushed up there and the others are pushed down through the unit
  path (the softmax over units: raising the executed unit's weight lowers the rest), the
  dedicated units among them: −81 and −61 per value position at step 2,000 of seed 4, while they
  still held 15 % of the slots. A unit that has lost the majority of a kind of position is
  un-learned there by the term that rewards the unit that took it. This is the "units that lose a
  route un-learn it" of section 5b, point 3, and the term is the unit path of the task loss.
- **The largest gradient in the router is the cell-mask path at the query-key positions**, on the
  reader (hundreds to thousands per position), and its sign changes from checkpoint to
  checkpoint. It is not a steady push; it is the reason the reader's rows move most.
- The prediction loss's gradient on the router is three orders of magnitude smaller than the task
  loss's at these positions.

This is the selection bias the white paper names (§6: "Exploration and dense fallback samples
reduce the bias from observing only selected routes"): only the executed (cell, unit) pairs are
observed, so the estimator cannot tell the router that the unit it stopped selecting would have
done better.

**The same windows under other estimators** (`sag_grad.py --override ...`,
`evidence/P2.10/sag/variants.txt`): the push at the value positions on the taker and on the
dedicated unit that is losing them, at the same weights, for the recipe's estimator, a sharper
constant temperature, and `train.unit_credit=executed` (section 3):

| Checkpoint (taker / dedicated unit, their slot shares) | Recipe (τ = 1) | τ = 0.25 | τ = 0.1 | Executed credit |
| --- | --- | --- | --- | --- |
| Seed 4, step 1,500 (unit 0 / unit 1: 67 % / 24 %) | +419 / −66 | +1,725 / −657 | +2,050 / −882 | +435 / −1 |
| Seed 4, step 2,000 (86 % / 7 %) | +376 / −122 | +1,423 / −649 | +1,137 / −563 | +367 / −45 |
| Seed 2, step 6,500 (unit 7 / unit 4: 45 % / 54 %) | +260 / −5 | +665 / −51 | +1,343 / +48 | +280 / +37 |
| Seed 2, step 8,500 (91 % / 8 %) | +721 / −101 | +2,729 / −577 | +3,696 / −426 | +714 / −46 |

A sharper temperature does not remove the push against the losing unit where the two are close:
relative to the taker's push it grows (seed 4, step 1,500: −0.16 of it at τ = 1, −0.38 at 0.25,
−0.43 at 0.1). With the executed unit alone credited, the losing unit is no longer pushed down by
the taker's success (what remains is its own slots' gradient).

## 3. Remedies

All arms: the recipe with one change, 9,000 steps, diagnostics off, the split training window
(`hardware.window=split`, whose gradients are the model's: `docs/p2_10_split_window.md`) unless
named "native"; `configs/sweeps/a6_5b_remedies.yaml`. One run of one seed says little here: the
same configuration gave seed 4 final accuracies of 58.9 %, 74.4 % and 94.3 % on this day. An arm
is judged by its worst runs over many seeds.

**A sharper straight-through temperature makes it worse** (seeds 4 and 5, exps 61 to 66):

| Arm | Seed 4: final / later / worst drawdown | Seed 5: final / later / worst drawdown |
| --- | --- | --- |
| Recipe (τ = 1) | 94.3 % / 90.2 % / 1.2 pp | 81.9 % / 67.7 % / 5.5 pp |
| τ = 0.25 | 69.6 % / 51.7 % / 25.0 pp | 84.7 % / 73.2 % / 10.9 pp |
| τ = 0.1 | 74.9 % / 58.1 % / 35.3 pp | 80.0 % / 71.9 % / 10.3 pp |

Routes form sooner (seed 4 at 95.1 % by step 2,000 with τ = 0.1) and are then lost harder: all
four runs fall ten points or more. This is what the estimator comparison of section 2 predicts:
at a close decision a sharper softmax pushes the losing unit down more, not less.

**The executed unit alone credited** (`train.unit_credit=executed`; opt-in code on this branch,
commit 29ec392; not a configuration-only change and in no recipe: adopting it would be an
amendment). The straight-through unit weight of a slot differentiates the executed unit's score
only, so a unit that did not run gets no gradient from the slot (section 2, last column of the
estimator table). Seeds 0, 4, 5:

| Seed | Final dev | Later queries | Worst drawdown | The recipe on the split window, same seed |
| ---: | ---: | ---: | ---: | --- |
| 0 | 89.8 % | 82.0 % | 6.6 pp | 88.6 % / 80.9 % / 3.1 pp |
| 4 | 91.4 % | 85.4 % | 2.8 pp | 94.3 % / 90.2 % / 1.2 pp |
| 5 | 77.5 % | 57.9 % | 0.8 pp | 81.9 % / 67.7 % / 5.5 pp |

Its falls after a route has formed are small, as the gradient comparison predicts. It does not
help a run that starts in the wrong place: seed 5 never bound its keys (later queries 57.9 %).
Three runs do not separate it from the recipe on the split window (table at the end).

**Single runs of other arms on the two hard seeds** (final dev / later queries / worst drawdown;
one run each, so a hint at most):

| Arm | Seed 4 | Seed 5 |
| --- | --- | --- |
| The recipe, split window | 94.3 % / 90.2 % / 1.2 pp | 81.9 % / 67.7 % / 5.5 pp |
| The recipe, autograd window | 74.4 % / 59.4 % / 2.4 pp | 62.6 % / 40.4 % / 11.2 pp |
| `split_k8`: eight cells selected instead of four (`model.k_select=8`; the paper's "reconsider sparsity") | 79.1 % / 78.9 % / 11.6 pp | 95.0 % / 93.6 % / 1.5 pp |
| `split_routerlr`: the router's rate at a tenth, so that its proposal sits at its cap and the schedule's decay reaches it (`train.router_lr_mult=0.1`) | 54.6 % / 49.7 % / 17.9 pp | 86.3 % / 77.7 % / 3.7 pp |
| `native_softwarm`: the paper's soft-gate warm-up, 1,000 steps (`train.soft_until_step=1000`; autograd window) | 89.5 % / 86.7 % / 2.1 pp, binds keys | 89.6 % / 86.1 % / 1.6 pp, binds keys |

The soft-gate warm-up is the paper's prescription (section 6: "Warm up with soft gates, then use
hard selection") and the recipe has none: hard straight-through selection starts at step 0. For
the first 1,000 steps every cell executes under the soft mask, so every cell and unit is observed
at every position, which is the missing comparison of section 2 for as long as the warm-up lasts.
At the switch the model scores 4.9 % on dev (it has not learned the task under soft gates); 2,000
steps later it is at 81.5 %, and it holds 87 to 90 % from step 4,000 to the end, the smallest
fall of any autograd-window run of the day. The dense steps take about 7 GB and the dense step's
time, and the experiment guard judges an arm's step time on its first 70 steps, so such an arm has
to start with the card to itself. Four more seeds, and what they changed: section 3b.

**Not run.** At 14:36 the user moved the work back to step time, and the queue was cancelled
with these arms in it (`a6_5b_remedies.yaml` has them; `python -m hsn.experiments ... requeue`
brings the cancelled experiments back):

- the paper's balance loss (`split_balance`: `train.lambda_b=0.02` with
  `train.aux_enabled=[balance]`; the first two launches of this arm failed on the missing
  `aux_enabled` and on a worker started without the card, so it has no run at all). It is the
  mechanism that names the taker: in every run inspected the unit that takes a route over is the
  most used one;
- the paper's dense fallback (`split_dense05`, `split_dense10`). Note what it can and cannot do
  here: a dense-fallback row reads the summary instead of the selected cells, so it trains the
  key-blind path, not the binding;
- the router rate with a long decay (`split_decay`), and executed credit with either rate arm;
- the 12-seed sets that would turn the hints above into counts: the split baseline on seeds 3 and
  6 to 11, the autograd baseline on seeds 6 to 11, executed credit on seeds 1 to 3 and 6 to 11;
- the paper's utility regression (`native_utility`, `train.lambda_u`). It cannot be run usefully
  under the 0.5 s rule today: a probe is nine eager rollouts of a window (the policy's selection,
  four added candidates, four removed slots), several seconds each, on 5 % of the windows by
  default; a compiled probe path would have to come first. It is the paper's own answer to this
  problem (the score trained to predict a measured loss difference cannot outgrow real
  differences) and the one mechanism of section 5b's table that compares routes directly;
- `train.ste_scope=all` (opt-in code, commit 5229fea, tests `tests/test_ste_scope.py`): the routed
  training window on the masked round, the model's own reference for the sparse round: the same
  decision and written values, every cell and unit computed, so that a cell or unit that was not
  selected still receives its first-order credit. This is the direct counter to the selection
  bias of section 2 (a route that lost its slots is told when it would have helped), at the dense
  step's time and about 7 GB. `native_fullste` was queued for seed 4 and cancelled before launch;
- its affordable form for the split window, a zero-weight slot for the best unselected cells
  (`evidence/P2.10/sag/shadow_cells_wip.patch`, not in the tree): the state-recurrence part
  agrees exactly with the masked round's term on every position without a readout; the readout's
  own term for an unselected cell is still missing. Unfinished.

## 3b. The soft warm-up on six seeds, and what its plateau is

After the first two warm-up runs the user asked for more of them whenever the card was idle.
Seeds 0 to 5, `native_softwarm` (the recipe with `train.soft_until_step=1000`, autograd window,
9,000 steps, diagnostics off):

| Seed | Final dev | First queries | Later queries | Worst drawdown | Key-binding probe (follows a swapped key) | Autograd baseline today |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 80.6 % | 95.9 % | 67.1 % | 3.2 pp | partial (28 %) | 88.0 % |
| 1 | 82.9 % | 87.4 % | 79.0 % | 4.1 pp | binds keys (82 %) | 91.4 % |
| 2 | 75.8 % | 95.2 % | 58.6 % | 5.5 pp | does not bind keys (17 %) | 77.1 % |
| 3 | 71.9 % | 84.3 % | 60.9 % | 6.5 pp | partial (53 %) | 90.6 % |
| 4 | 89.5 % | 92.7 % | 86.7 % | 2.1 pp | binds keys (89 %) | 74.4 % |
| 5 | 89.6 % | 93.5 % | 86.1 % | 1.6 pp | binds keys (91 %) | 62.6 % |
| Mean | 81.7 % | | 73.0 % | | | 80.7 % |

No warm-up run fell by 10 pp or more (three of the six baseline runs did), and the largest
drawdown is 6.5 pp. The mean is the baseline's. What replaces the falls is a plateau: three runs
settle between 72 and 81 % and stay there. The user's reading (11 Nov, 17:15): the warm-up is the
more stable arm, and the plateau has to be fixed either way.

**The plateau is the takeover, slow.** Every run answers a document's first query (84 to 96 %);
the plateau runs answer the later ones at the key-blind level (58.6 % and 60.9 % against 62.3 %).
`route_flip.py` on the kept checkpoints of two plateau runs and two runs that bind keys
(`evidence/P2.10/sag/softwarm/`):

| Run | Step | Value-position slots on the unit that reads the query keys | Largest other unit there | Binding overlap |
| --- | ---: | ---: | ---: | ---: |
| Seed 2, plateau | 1,500 | 53 % | 46 % | 0.38 |
| | 3,000 | 75 % | 25 % | 0.38 |
| | 5,000 | 99 % | 1 % | 0.28 |
| | 9,000 | 81 % | 19 % | 0.22 |
| Seed 3, plateau | 1,500 | 54 % | 16 % | 0.44 |
| | 3,000 | 51 % | 27 % | 0.52 |
| | 5,000 | 72 % | 21 % | 0.40 |
| | 9,000 | 81 % | 16 % | 0.31 |
| Seed 4, binds keys | 1,500 | 56 % | 42 % | 0.35 |
| | 3,000 | 45 % | 55 % | 0.56 |
| | 5,000 | 44 % | 55 % | 0.58 |
| | 9,000 | 44 % | 52 % | 0.56 |
| Seed 5, binds keys | 1,500 | 46 % | 42 % | 0.27 |
| | 3,000 | 40 % | 32 % | 0.28 |
| | 5,000 | 29 % | 39 % | 0.40 |
| | 9,000 | 35 % | 51 % | 0.51 |

At step 1,000 (the end of the warm-up) the slots are spread over two to four units and dev is at
2 to 13 %: the model has not learned the task under soft gates. Within 500 steps of the switch one
unit holds 87 to 99 % of the key, filler and query positions in three of the four runs (in seed 5
two units share them until about step 3,000, when one has 95 % of the query keys). From there the
runs differ in one thing: in the runs that bind keys another unit keeps about half of the value
positions to the end; in the plateau runs the unit that reads the query keys takes them, a few
points per 500 steps, and the key stops selecting the cells its value went into. The route guard
reports it throughout (seed 3: "19 to 58 % of the other units' established routes lost to unit 0"
at every check from step 4,800 on) with that unit's router step already at its 1/64 floor.

**The loss hands the positions over.** `sag_grad.py` with the balance term added
(`evidence/P2.10/sag/softwarm/grad-*.txt`; six training windows of 32 documents per checkpoint,
about 1,100 value positions; the push on a unit's logit per value position, times 1e6, positive
raises its score):

| Run | Step | Task loss: reader unit | Task loss: the other unit | Balance loss at weight 1: reader / other |
| --- | ---: | ---: | ---: | --- |
| Seed 2, plateau | 1,500 | +88 | −78 | −147 / +34 |
| | 2,000 | +111 | +8 | −156 / +39 |
| | 2,500 | +124 | +70 | −120 / +38 |
| | 3,000 | +121 | −48 | −174 / +49 |
| Seed 4, binds keys | 1,500 | +10 | −18 | −103 / +7 |
| | 2,000 | +44 | +60 | −112 / +17 |
| | 3,000 | +48 | +56 | −142 / +39 |
| | 5,000 | +30 | +82 | −154 / +54 |

In the plateau run the task loss favours the reader at the value positions at every checkpoint, by
54 to 169, half through the cell mask and half through the unit weights; in the run that binds
keys it favours the other unit from step 2,000 on. The warm-up evens out the start; it does not
change which way the estimator pulls once one unit is ahead. The balance loss (the paper's
mechanism against a unit's share, section 7) pushes the other way by 155 to 225 per unit of
weight at those positions, and by the same amount at every other position, where the reader's
margin is wide.

**What binding is, in the scorer's weights.** A5's key addressing gives every cell two features:
the similarity of its fixed address to the current byte's code (`a_cur`) and to the previous
byte's (`a_prev`). The scorer is linear with one row per unit, so a unit ranks cells by one weight
on each, at every position. A value is written at the position whose previous byte is the key;
it is read at the position whose current byte is the key. The key therefore selects the cells
its value went into only if the unit that runs at the value positions ranks cells by `a_prev`
and the unit that runs at the query keys ranks them by `a_cur`, with the same sign (both
positive in five of the ten control runs, both negative in the others). One unit doing both
jobs ranks the value position's cells by the value byte's own address as well, and the write
lands elsewhere. The ten control runs (six on the autograd window, four on the fused one), by
the share of the value-position slots the query-key reader holds at step 9,000:

| Run | Later queries | Reader's share of the value slots | Binding overlap | Reader's weight on `a_cur` | The other unit's weight on `a_prev` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 5, autograd | 86.1 % | 35 % | 0.51 | +0.19 | +0.39 |
| Seed 2, fused | 83.6 % | 38 % | 0.60 | +0.21 | +0.16 |
| Seed 4, autograd | 86.7 % | 44 % | 0.56 | +0.27 | +0.20 |
| Seed 1, autograd | 79.0 % | 59 % | 0.44 | −0.27 | −0.19 and −0.14 (two units) |
| Seed 1, fused | 68.6 % | 74 % | 0.29 | −0.22 | −0.12, with 16 % of the slots |
| Seed 2, autograd | 58.6 % | 81 % | 0.22 | +0.21 | (the reader's own: +0.08) |
| Seed 3, autograd | 60.9 % | 81 % | 0.31 | −0.20 | −0.16 |
| Seed 3, fused | 57.5 % | 92 % | 0.30 | −0.27 | (the reader's own: −0.15) |
| Seed 0, fused | 71.2 % | 96 % | 0.19 | +0.25 | +0.15, with 3 % of the slots |
| Seed 0, autograd | 67.1 % | 99 % | 0.24 | −0.33 | (none) |

Later-query accuracy follows the reader's share of the value slots: under 45 % in the three runs
that bind keys, 59 % in the one that binds them partly, 74 % or more in the six that do not.
The seed does not decide it (seed 2 plateaued on the autograd window and bound keys on the fused
one, seed 3 the other way round in size): three of ten control runs bind keys, and an arm has to
be compared with that on a dozen runs. The route guard cuts the reader's router step to its
1/64 floor between steps 3,100 and 4,700 in nine of the ten runs (in the tenth it stopped at a
quarter), after the reader's address weights have reached their final size; the slots it goes
on taking after that come from the other units' rows and from the cells' states, which the guard
does not hold.

**Runs on the fused window.** `train.soft_until_step` is inside the split window's scope since
this afternoon (`docs/p2_10_split_window.md`, "Scope"): the warm-up steps run the autograd
window, the fused window takes over at step 1,000, and a run takes 12 minutes instead of 22.
`fused_softwarm` (the control) and the arms with one more mechanism, on the two plateau seeds
first:

| Arm (all with the 1,000-step warm-up) | Seed 2: final dev / later queries | Seed 3 | Runs with later queries at 80 % or more |
| --- | --- | --- | ---: |
| `fused_softwarm`: the control | 89.0 % / 83.6 % | 68.7 % / 57.5 % | 1 of 2 (2 of 6 on seeds 0 to 5; 2 of 6 on the autograd window) |
| `fused_softwarm_k8`: eight cells selected (`model.k_select=8`, the paper's "reconsider sparsity") | 89.9 % / 83.7 % | 86.2 % / 80.6 % | 2 of 2 |
| `fused_softwarm_balance`: the paper's balance loss, `train.lambda_b=0.02` | 89.2 % / 82.0 % (after a 15.6 pp dip) | 87.5 % / 80.9 % | 2 of 2 |
| `fused_softwarm_k8_balance`: both | 88.1 % / 79.0 % | 98.6 % / 97.9 % | 1 of 2 |
| `fused_softwarm_balance10`: `lambda_b=0.1` | 68.4 % / 62.3 % | 92.2 % / 89.0 % | 1 of 2 |
| `fused_softwarm_balance50`: `lambda_b=0.5` | 70.7 % / 62.3 % | 90.9 % / 85.3 % | 1 of 2 |
| `fused_softwarm_executed`: `train.unit_credit=executed` (opt-in candidate, not adopted) | 80.7 % / 67.1 % | 91.5 % / 89.7 % | 1 of 2 |
| `fused_softwarm_temp2`: straight-through temperature 2 | 69.3 % / 56.6 % | 84.9 % / 82.4 % | 1 of 2 |
| `fused_softwarm_mlp`: the plan's two-layer scorer | 84.6 % / 71.8 % | 71.0 % / 52.1 % | 0 of 2 |
| `fused_softwarm2000`: a warm-up of 2,000 steps | 77.3 % / 61.4 % | 78.5 % / 61.5 % | 0 of 2 |

Two runs separate the arms that do nothing from the rest and no more. What the routes of these
runs add:

- *Eight slots.* The reader still takes most of the value slots (78 to 98 % on seed 2, 61 to 78 %
  on seed 3), and the key's cells are written anyway: with eight slots the cells addressed by the
  key are selected whoever takes the others. Neither run has a dip (worst drawdown 2.5 pp).
- *Balance 0.02.* The reader stays at 5 to 30 % of the value slots and the filler spreads over
  all eight units. On seed 2 another unit took the value positions between steps 3,000 and 5,000
  (dev 78.1 % to 67.6 %) and the run recovered to 89 % on a different arrangement.
- *Stronger balance* (0.1, 0.5) gives one very good and one poor run each: not better.
- *Executed credit* keeps what has formed (overlap 0.77 on seed 3) and does not stop the reader
  where it is ahead (97 % of the value slots on seed 2), as in the morning's runs.
- *The two-layer scorer* does not learn to switch between the two rankings inside one unit in
  9,000 steps; *a softer temperature* and *a longer warm-up* leave the takeover as it is.

**Twelve seeds** of the control, eight slots, balance 0.02 and the two together were running
when this was written (11 Nov, 22:20 MST; `fused_softwarm*` in the database, `python3 night.py`
beside it). Counts so far, runs with later queries at 80 % or more: control 2 of 6, eight slots
2 of 3, balance 3 of 3, both 1 of 3. The fused window without the warm-up (`fused_baseline`,
six seeds): 87.6 % mean, 3 of 6 at 80 % or more, 2 of 6 fell 10 pp or more.

## What the day's runs say

| Arm | Runs | Mean final dev | Range | Spread (s.d.) | Mean later queries | Fell 10 pp or more | Worst drawdown | Pass the binding verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Autograd window, last night (c5a82ea, diagnostics on) | 9 | 87.2 % | 78.5 % to 95.6 % | 4.7 pp | 80.8 % | 1 of 9 | 13.7 pp | 3 of 9 |
| Autograd window, today (this branch, diagnostics off) | 6 | 80.7 % | 62.6 % to 91.4 % | 10.4 pp | 70.5 % | 3 of 6 | 14.4 pp | 2 of 6 |
| Split window, first replay (credited a unit that had not run) | 8 | 82.9 % | 58.9 % to 96.0 % | 10.3 pp | 71.4 % | 4 of 8 | 18.8 pp | 2 of 8 |
| Split window, fixed replay (9e0a61a on) | 5 | 88.0 % | 81.9 % to 94.3 % | 4.6 pp | 81.1 % | 0 of 5 | 8.8 pp | 3 of 5 |
| Executed credit (split) | 3 | 86.2 % | 77.5 % to 91.4 % | 6.2 pp | 75.1 % | 0 of 3 | 6.6 pp | 0 of 3 |
| Eight cells selected (split) | 2 | 87.1 % | 79.1 % to 95.0 % | 7.9 pp | 86.2 % | 1 of 2 | 11.6 pp | 1 of 2 |
| Router rate at a tenth (split) | 2 | 70.5 % | 54.6 % to 86.3 % | 15.9 pp | 63.7 % | 1 of 2 | 17.9 pp | 0 of 2 |
| Temperature 0.25 (split) | 2 | 77.1 % | 69.6 % to 84.7 % | 7.5 pp | 62.4 % | 2 of 2 | 25.0 pp | 0 of 2 |
| Temperature 0.1 (split) | 2 | 77.5 % | 74.9 % to 80.0 % | 2.5 pp | 65.0 % | 2 of 2 | 35.3 pp | 0 of 2 |
| Soft-gate warm-up (autograd), section 3b | 6 | 81.7 % | 71.9 % to 89.6 % | 6.5 pp | 73.0 % | 0 of 6 | 6.5 pp | 3 of 6 |
| Soft-gate warm-up, fused window after it, section 3b | 6 | 82.0 % | 68.7 % to 89.0 % | 6.8 pp | 73.2 % | 1 of 6 | 10.2 pp | 1 of 6 |
| Fused window, no warm-up | 6 | 87.6 % | 79.7 % to 94.8 % | 5.7 pp | 79.5 % | 2 of 6 | 16.9 pp | 2 of 6 |

- The falls and the "stuck" runs are one event, the route takeover, and the estimator's selection
  bias is what drives it (section 2). Nothing here removes it.
- The soft warm-up turns the falls into a slow version of the same event (section 3b): no run
  fell by 10 pp, three of six stayed below 81 % without binding their keys, and the mean is the
  baseline's.
- A sharper temperature is ruled out. Executed credit keeps formed routes but does not form them.
  A larger selection and a slower router did not help on the hard seeds in single runs.
- Run to run, one configuration and one seed gave 58.9 %, 74.4 % and 94.3 % (seed 4). A remedy
  has to be judged on a dozen seeds by its worst runs; none of today's arms has that many.
- The split window with the replay fix of this morning is, so far, no worse than the autograd
  window (table); the first replay, which credited a unit that had not run, was.

## 4. Is it this branch? No

Today's controls ran the branch `p2.10-cuda-kernels` with diagnostics off, last night's the commit
c5a82ea with diagnostics on, so the lower controls could have been the code. Two checks:

- **The two commits train identically.** `evidence/P2.10/sag/equiv.py` trains the recipe (guard,
  update caps, `LazyAdamW`, the float64 clip, the weight average) for 300 steps on the CPU, one
  BLAS thread, from each checkout: every step's loss and every final weight are bit-identical
  (`equiv_c5a82ea_vs_9e0a61a.txt`). The route guard differs between the commits only by the
  removed `restore` option and `Router.select` not at all.
- **Last night's commit, run today** (same database, its own worker and environment, diagnostics
  off): seed 4 at 93.7 % by step 2,000 and 92.7 % at 3,000; seed 5 at 47.0 % and 65.8 %, a slow
  start like today's (stopped there to free the card).

On the card the two commits are not bit-identical from step 1 to 6 on (the compiled graphs round
differently), and two runs of one commit part after 40 to 100 steps. The spread between runs of
one configuration is the recipe's, not the branch's.

## Reproduce

```
python /home/joe/hsn-experiments/a5-fix/probe/route_flip.py RUN_DIR STEP [STEP ...] --n-docs 150
python evidence/P2.10/sag/sag_grad.py CKPT [CKPT ...] --windows 4 --batch 16     # CPU, fp32
python -m hsn.experiments --db DB enqueue configs/sweeps/a6_5b_remedies.yaml --only 'split_dense10/s*'
```
