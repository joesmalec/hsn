# Amendment A12: address weights that depend on the position's content (experimental)

Status: **implemented as an opt-in experiment on 13 Nov 2026; not adopted; no result yet.**
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

## Tests

`tests/test_a12_address_gate.py`: opt-in and hashing; scope (key address, linear scorer; the
split and fused windows refuse); only the gate is added, for A5 and for A9's layout, and every
other parameter starts as before; the logits against a direct computation and a zero gate as
the linear scorer; a unit following an address at one content and not at another; a zero gate
is the model without the key (evaluation and straight-through); the window against stepping
and the masked round against the sparse one; the task loss reaches the gate; the compiled
window against the eager one; thirty training steps.
