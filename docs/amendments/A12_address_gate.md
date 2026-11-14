# Amendment A12: address weights that depend on the position's content (experimental)

Status: **implemented as an opt-in experiment on 13 Nov 2026; not adopted.** The deciding
27k arm holds (97.7 %, where the base ended at 93.9 and 92.2 %). **Its weakness, found on
14 Nov: with the gate a run can fit the bytes of the training keys** and lose keys of bytes it
has never seen (training keys 98 %, unseen keys from 99 % down to 39 %, by the run and not by
the configuration); keys made of seen bytes, new combinations included, stay at 98 to 99 %.
The 9k run below the kill rule's bar was such a run (sections "Results" and "The hold-out
runs").
- Asked for by the user on 13 Nov 2026 ("lets make it an opt in amendement and test"), after the
  27k runs of the pair task lost what they had learned by 10,000 steps.
- `model.address_gate` (default false; needs `model.key_address` and the linear scorer). Left
  out of the configuration hash at its default: existing configurations, hashes and arm ids are
  unchanged, and a model without the key is the model as before, bit for bit.
- It amends plan §2.2 step 4 (the scorer) by one term. It changes no metric, threshold or task.

## Cause

**What was measured.** Pair task (key-value + delayed), 64 cells, state gradient, 27,000 steps:
the runs peak at 98 to 99 % and end at 93.9 % (12 slots), 90.0 % (8 slots) and 95.6 % (no route
guard). The training loss rises with the dev loss and the loss is the same on keys seen in
training and on new keys, so it is not overfitting. At the positions where a fact is written,
the selected cells move from the cells the key addresses to the cells the value addresses
(`/home/joe/hsn-handoff/unit_shares.py`, in-range kv dev documents):

| Model | Written under the key's address | Under the value's address | Read under the key's address |
| --- | --- | --- | --- |
| 12 slots, 9k | 76.0 % | 26.6 % | 86.2 % |
| 12 slots, 27k | 46.6 % | 59.6 % | 80.7 % |
| 8 slots, 9k | 65.0 % | 38.4 % | 86.7 % |
| 8 slots, 27k | 36.8 % | 69.3 % | 87.6 % |
| 12 slots, 27k, no route guard | 40.6 % | 66.4 % | 73.5 % |

The reads stay with the key. Fewer copies of a fact sit where its query looks, and values land
in cells that hold other keys' facts: wrong answers that are another key's value go from 32 %
of the errors at 9k to 63 to 67 % at 27k.

**Why the scorer drifts there (a reading, not measured).** The scorer is linear in its
features. A unit's weight on `a_cur` (the address of the current byte) is one number, the same
at every position. The unit that reads a queried key needs it large, and A6's route-guard text
records that this weight "keeps growing ... for a whole run". At a value's position the current
byte is the value, so the same weight pulls slots to the value's cells. The only thing a linear
scorer can set against that is an additive term from `u`, which does not change which cells an
address feature prefers. The route guard slows the growth and does not remove the conflict;
with a lower floor it fails another way (training keys 98.8 %, new keys 80 to 83 %).

## The change (opt-in; needs `model.key_address`)

One new matrix `G` (`d → M · A`, no bias, `A` the number of address features: 2 with A5, 4 with
A9's two-byte address). The logit of cell `i` and unit `j` at position `t` gains

```
logit[i, j] += Σ_a (G u_t)[j, a] · address_a[i]
```

so the weight unit `j` puts on address feature `a` is `W[j, a] + (G u_t)[j, a]`: its own
constant weight plus a part that depends on the content `u_t` of the position.

- **What it makes expressible.** A unit can follow the current byte's address where the byte is
  a queried key and not where it is a value, or follow the previous byte's address only at a
  value. A5's text names this as what is learned ("which feature a unit follows at which class
  of byte"); with constant weights it could only be approximated through which unit wins.
- **The addresses are unchanged.** `a_cur` and `a_prev` are still inner products of A5's fixed
  byte codes with the fixed cell addresses. Nothing about any key is learned; a key never seen
  in training has the same kind of address as a seen one.
- **Start.** `G = 0`: the model computes exactly what it computes without the key, in training
  and in evaluation (`test_zero_gate_is_the_model_without_it`). It is built without consuming a
  random draw, so with the same seed every other parameter has the same initial value.
- **`u` as the scorer reads it.** Under `train.feedback_detach` with `router`, the scorer's `u`
  is a constant in the backward, so `G` is trained by the selection's straight-through
  gradient like the scorer's other weights and no new gradient path into the shared transform
  appears.

**What stays.** Cells, graph, units, ε, the convex tanh gate, the selection rule, A5's and A9's
address features and `q`. No attention over the sequence, no new state.

**Parameters:** `d · M · A` (1,024 for the recipe with A5; 2,048 with the two-byte address).
**Cost per position:** one `d → M·A` map and `N · M · A` multiply-adds, scorer work.

**Windows.** Eager and compiled autograd windows, and the split and fused windows (P2.10,
P2.10c) under the full cut and with the state gradient (`hardware.window=split|fused` takes the
key since 14 Nov).
- *The replay* adds the gate's term to the scorer's logits (`hsn.train.split._scorer_logits`),
  with `u` the constant the router cut makes it, so `G` gets its gradient from the replay's
  graph like the scorer's other weights. The term is summed feature by feature.
- *Step time* (fused window, the approved shape, bf16, batch 32, alone on the RTX 3060,
  150 steps after 40 of warm-up, median):

  | | With the key | Without it | Difference |
  | --- | ---: | ---: | ---: |
  | The term as a plain sum, 14 Nov 08:27 MST, round 1 (host load 4.3 to 5.6) | 143.1 ms | 132.2 ms | 10.9 ms |
  | The same, round 2 | 142.6 ms | 133.7 ms | 8.9 ms |
  | The term as one einsum over the window's rows, 14 Nov 01:37 MST | 140.3 ms | 131.0 ms | 9.3 ms |

  The key costs about 10 ms a step in either form. An earlier version of this paragraph said
  the einsum was that cost; the plain sum was written to remove it and did not. Where the
  10 ms sit (the replay's term and its backward, or the sweeps' merged product with its
  `A · d` more columns at every position) is not measured.
- *The eager sweep* runs the model's own scorer. *The compiled sweeps* (the fused and the
  folded block) take their scorer's logits from one merged product (the scorer's linear with
  `V` as `d` more outputs); the gate enters it as `A · d` more input columns,
  `u_k · address_a[i]` (`hsn.train.compile.gated_features`), against `G` as more rows of the
  merged weight (`SweepTwin`), so the selection kernels and their inputs are as they were, and
  without the key the features and the weight are the ones without it.
- *CPU, float64* (`tests/test_a12_address_gate.py`): the split window gives the autograd
  window's decisions, logits, final state and parameter gradients, the gate's among them,
  under the full cut, with the state gradient and with A8's fast cells, for A5's two address
  features and A9's four; with the gate's weights zeroed its decisions are others. The merged
  product equals the model's scorer with the gate, and `V`.
- *Card, RTX 3060, 14 Nov, a gate that is not zero* (`tests/kernels/test_p2_10c_window.py`,
  `test_p2_10c_fused_window.py`): the fused and folded sweeps record the split sweep's tape
  (three shapes, A9's four features among them: every decision equal wherever the scores have
  a margin, 100 % of the positions); the fused window through its kernels against its own
  PyTorch sweep, 5e-8 and 9e-8 in fp32, 9e-4 and 1.5e-3 under bf16; the fused window against
  the compiled autograd window in fp32, identical cells and gradients 6e-8 (12 routed slots)
  and 2.5e-7 (the approved shape) apart. Those runs used the einsum form of the replay's
  term. With the plain sum (14 Nov 08:27 MST) the six gate cases of
  `test_p2_10c_fused_window.py` passed on the card: the window through its kernels against its
  own PyTorch sweep in fp32 and under bf16, and against the compiled autograd window in fp32,
  each at 12 routed slots and at the approved shape.

## The smallest test, and its kill rule

Pair task, 64 cells, 12 slots, state gradient, no warm-up, seed 0:
1. **27,000 steps with the key** against the same run without it (98.0 % at step 10,500,
   93.9 % at the end; a second seed is running). The deciding arm.
2. **9,000 steps with the key**: it must not cost where the base is good (98.0 / 98.6 / 98.3 %).
3. **Delayed recall, 9,000 steps** (99.6 %): the same check.
4. **The combined shape** (8 fast + 12 routed slots), pair task, 9,000 steps (93.4 % without).

Read from each 27k checkpoint with `unit_shares.py`: the share of the write slots under the
key's address.

**It works** if arm 1 ends within 1.5 points of its own peak (the base gave up 4.1, the
no-guard run 2.3) with the write slots under the key's address clearly above the base's
46.6 %, and arms 2 and 3 stay within their seeds' spread. **Kill rule:** if arm 1 gives up as
much as the base, or arm 2 falls below 97 %, the amendment is dropped: either the scorer does
not use the gate to keep the writes with the key, or the cause is not the constant address
weight.

## Results (13 Nov 2026; pair task, 64 cells, state gradient, one seed unless two are given)

All arms ran with `model.scorer_position=wrap` (A11) on the autograd window, on rented 3060s.
Dev accuracy in per cent; "past the window" is the out-of-range dev set (`eval_oor.py`).

| Arm | Run | Final | Peak (step) | Past the window |
| --- | --- | --- | --- | --- |
| 1. 12 slots, 27k, gate | `g1_k12nw_ag_cutC_cells64_27k` | 97.7 | 98.3 (12,000) | 94.1 |
| 1, no route guard | `g1_k12nw_ag_noguard_cutC_cells64_27k` | 97.1 | 98.3 (10,500) | 93.1 |
| base, no gate, 27k | `g1_k12nw_cutC_cells64_poswrap_27k` | 93.9 | 98.0 (10,500) | 82.8 |
| base, no gate, 27k, seed 1 | `g1_k12nw_cutC_cells64_27k` | 92.2 | 99.1 (10,500) | 59.0 (position as trained) |
| 2. 12 slots, 9k, gate | `g1_k12nw_ag_cutC_cells64` | 86.2 | 93.9 (7,500) | 75.6 |
| 3. delayed recall, 9k, gate | `k12nw_ag_cutC_cells64` | 99.5 | 99.7 (8,000) | 99.7 |
| 4. 8 fast + 12 routed, 9k, gate | `g1_f8ru20nw_ag_cutC_cells64` s0, s1 | 99.1, 97.2 | at the end | 96.6, 90.5 |
| 4 without the gate | `g1_f8ru20nw_cutC_cells64` | 93.4 | | |

Where the facts are written (`unit_shares.py`, the measure of the table under "Cause"):

| Model at 27k | Written under the key's address | Under the value's address | Read under the key's address |
| --- | --- | --- | --- |
| 12 slots, no gate | 46.6 % | 59.6 % | 80.7 % |
| 12 slots, gate | 64.1 % | 42.6 % | 90.6 % |
| 12 slots, gate, no route guard | 62.3 % | 36.1 % | 85.4 % |

Against the test:
- **Arm 1 holds.** It ends 0.6 points under its own peak (1.2 without the route guard; the base
  gave up 4.1 and 6.9), and the writes stay with the key more than in the base (64.1 % against
  46.6 %; the base had 76.0 % at 9k, so part of the move still happens). One seed.
- **Arm 2 is below the kill rule's bar** (86.2 %, after 93.9 % at step 7,500). The 27k arm has
  the same seed and settings up to the schedule and stood at 97.8 % at step 9,000 on another
  card. By the rule's letter it is met. What the run was (measured on 14 Nov,
  `key_split_eval.py`): key-value answers 98.0 % on training keys against 81.5 / 87.9 % on
  dev / test keys. It did not lose what it had learned; it had fitted the training keys' bytes
  (next section), without fast cells.
- **Arm 3** is inside the seeds' spread of the base (99.6, 98.0, 98.5 %).
- **Arm 4**: the gate lifts the combined shape from 93.4 % to 99.1 and 97.2 %.

What the gate does not fix, and one cost:
- **Fitting to the training keys.** Fresh documents, key-value answers on training keys against
  dev / test keys (`key_split_eval.py`): arm 1 at 27k 99.0 against 97.0 / 97.0; the base at 9k
  97.7 against 97.2 / 97.4; the combined shape at 9k without the gate 96.1 against 90.9 / 91.4,
  with it 99.4 against 98.7 / 98.4; the combined shape with the gate and
  `train.state_grad_ratio=3.0` at 27k (`g1_f8ru20nw_ag_sg3_cutC_cells64_27k`, final 94.4 %)
  99.5 against 93.7 / 93.3. The claim above that "nothing about any key is learned" holds for
  the addresses, not for `G`: `G` reads `u_t`, which carries the byte, so it can weight an
  address differently for bytes it has seen. The two-point gap of arm 1 is the size of that
  effect on routed slots; the larger gap of the combined shape at 27k is also there without the
  gate at 9k.
- The windows: the split and fused windows still refuse the key, so every run with it is on the
  autograd window (229 to 238 ms per step for the combined shape).

## The hold-out runs of 14 Nov 2026: the fitting to the training keys sits in `G`

Two runs on `synthetic_hold` (key-value with one-byte keys; delayed recall with two-byte keys,
a quarter of all key pairs in no split): 8 fast + 12 routed slots, 64 cells, two-byte address,
`wrap`, state gradient, 18,000 steps, autograd window. Checkpoints scored on fresh documents,
600 per condition (`/home/joe/hsn-handoff/hold_eval.py`; about 1,400 answers each, so one
standard error is 0.3 points near 99 % and 0.8 near 91 %):

| Run, step | delayed: seen pairs | new pairs of seen bytes | never-seen bytes | kv: train keys | dev keys | test keys |
| --- | --- | --- | --- | --- | --- | --- |
| gate, 9,000 | 99.2 | 98.8 | 96.8 | 98.5 | 98.1 | 98.5 |
| gate, 18,000 | 99.1 | 99.0 | 96.0 | 98.3 | 84.2 | 87.3 |
| no gate, 9,000 | 90.8 | 92.7 | 92.0 | 91.9 | 91.9 | 91.5 |
| no gate, 18,000 | 93.3 | 93.8 | 94.1 | 95.5 | 96.1 | 95.5 |

- **New combinations of seen key bytes score like seen pairs**, with and without the gate. The
  model does not learn the trained pairs by heart.
- **With the gate, key-value recall on unseen keys loses 11 to 14 points between 9,000 and
  18,000 steps while the training keys stay.** The run's dev curve (dev keys): 98.1 at 10,500,
  96.7 at 15,000, then 91.7 and 87.6: the loss is in the last fifth of the schedule, where the
  rate decays (from step 14,400). Without the gate there is no gap and no fall, at a lower level.
- So the long runs have two different late events. Without the gate the writes drift to the
  value's address and training keys and new keys fall alike ("Cause"). The gate stops that and
  can fit the training keys.

**Where the fitting sits.** The 18,000-step model of the gate run with single groups of
parameters set to their 9,000-step values (key-value answers, 300 fresh documents each;
`/home/joe/hsn-handoff/hold_overlap.py` for the measure):

| The 18,000-step model with | Training keys | Unseen keys |
| --- | --- | --- |
| its own weights | 98.3 | 85.8 |
| `G` of step 9,000 | 98.6 | 94.7 |
| the scorer's weights and `G` of step 9,000 | 98.9 | 92.0 |
| `G` half-way between the two steps | 99.3 | 92.5 |
| the byte embedding of step 9,000 | 94.0 | 77.6 |
| the decoder of step 9,000 | 97.2 | 84.5 |
| `G` scaled to its norm at 9,000 | 98.3 | 87.7 |
| `G` cut to rank 4 (rank 2, 8, 16) | 98.0 (96.3, 97.6, 98.3) | 91.3 (87.2, 87.9, 86.9) |
| `G` set to zero | 35.6 | 46.3 |

- About nine of the 12.5 points come back with the 2,048 weights of `G` alone. It is the
  direction `G` moved in, not its size, and it is not a matter of rank (`G` has about 13 of 32
  directions in use at both steps).
- At the queried key the model selects fewer of the cells the fact was written to: for unseen
  keys 71.7 % of the read cells at 9,000 steps and 58.6 % at 18,000 (training keys 75.7 and
  68.3 %).
- This is the limit of "nothing about any key is learned" (section "The change"): the addresses
  are fixed, but `G` reads `u_t`, which carries the byte, and can weight an address feature
  differently for the bytes it was trained on.

**It is the path of a run, not its configuration.** Three runs of the gate run's arm that are
one configuration and one seed for their first 9,000 steps (exp 580; exp 584, whose rate stays
constant after step 14,400; exp 585, whose `G` is held after step 9,000), at step 9,000:

| Run | kv: training keys | dev keys | delayed: seen pairs | new pairs | never-seen bytes |
| --- | --- | --- | --- | --- | --- |
| exp 580 | 98.5 | 98.1 | 99.2 | 98.8 | 96.8 |
| exp 584 | 97.9 | 95.8 | 99.2 | 99.2 | 83.1 |
| exp 585 | 97.9 | 48.0 | 98.9 | 98.6 | 59.0 |

Under bf16 a rounding at the third digit reroutes every row within its first positions
(`docs/p2_10c_state_gradient_window.md` on the speed branch), so equal runs part at once. They
end in solutions that are equal on the bytes they trained on and far apart on bytes they never
saw. Exp 585 fits the training bytes from the start (step 6,000: 97.1 against 67.5 %); exp 580
after step 15,000; exp 584 keeps a gap of two to four points to the end (18,000 steps: 97.9
against 95.8 / 97.1; never-seen bytes 98.7). A second seed of the plain arm (exp 586) ends at
99.6 against 97.4 / 97.7 with never-seen bytes at 98.5 and no late fall. Of the four runs of
this shape two keep unseen keys within two to three points of the training keys, one loses
them late and one never has them.

**Every run split by keys so far** (key-value answers, fresh documents, training keys against
dev keys; pair task of synthetic_v1 unless "hold"):

| Model | Runs: training keys / unseen keys |
| --- | --- |
| routed only, no gate | 97.7 / 97.2 (9k); 92.2 / 92.6 and 94.7 / 94.8 (27k) |
| routed only, gate | 98.0 / 81.5 (9k, arm 2); 99.0 / 97.0 and 98.8 / 95.6 (27k); 99.2 / 99.1 (hold, 18k) |
| fast cells, no gate | 96.1 / 90.9 (9k); 95.5 / 96.1 (hold, 18k) |
| fast cells, gate | 99.4 / 98.7 and 98.5 / 96.3 (9k); 99.5 / 93.7 (27k); hold, 18k: 98.3 / 84.2 (exp 580), 97.9 / 95.8 (584), 98.6 / 32.8 (585), 99.6 / 97.4 (586, seed 1) |

- Without the gate and without fast cells no run has a gap. The gate opens one in some runs
  with and without fast cells; the largest are with both.
- **What does not move:** keys made of bytes the model has trained on. New pairs of seen bytes
  score like seen pairs in every checkpoint of every hold-out run (98.6 against 98.9 % in exp
  585 at 9,000 steps, where never-seen bytes are at 59 %).

**Two diagnostics of the hold-out runs, and what they showed.**
- `train.address_gate_hold_step` (default null: off, bit for bit; hashed when set; needs
  `model.address_gate`): `G` is trained for that many optimiser steps and keeps its value from
  then on. Its first run (exp 585, held from 9,000) says nothing about holding: that run's `G`
  was fitted to the training bytes before the hold.
- The rate held constant to the end (exp 584): no late fall, on one path.
Neither is a fix. What `G` may read (it reads `u_t`, which carries the byte, and with A8's
`fast_readout` the fast cells' read) would be an amendment of its own.

## Tests

`tests/test_a12_address_gate.py`: opt-in and hashing; scope (key address, linear scorer; the
split and fused windows refuse); only the gate is added, for A5 and for A9's layout, and every
other parameter starts as before; the logits against a direct computation and a zero gate as
the linear scorer; a unit following an address at one content and not at another; a zero gate
is the model without the key (evaluation and straight-through); the window against stepping
and the masked round against the sparse one; the task loss reaches the gate; the compiled
window against the eager one; thirty training steps; `train.address_gate_hold_step` (opt-in and
hashing, needs the gate, the gate unchanged from its step bit for bit while the scorer's own
weights go on, the run before the step equal to the run without the key, a hold at step 0).
