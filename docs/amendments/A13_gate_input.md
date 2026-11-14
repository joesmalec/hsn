# Amendment A13: what the address gate reads (experimental)

Status: **implemented as an opt-in experiment on 14 Nov 2026; not adopted; no result yet.**
- Asked for by the user on 14 Nov 2026 ("Let's add that amendment"), after the hold-out runs
  showed that A12's gate can fit the bytes of the training keys.
- `model.address_gate_input`: `u` (default) | `summary` | `u_no_fast`. Needs
  `model.address_gate`; `u_no_fast` needs `model.fast_readout`. Left out of the configuration
  hash at its default: existing configurations, hashes and arm ids are unchanged, and with the
  key at `u` the model is A12's, bit for bit.
- It amends A12 (the one term A12 adds to plan §2.2 step 4) in what `G` is applied to. It
  changes no metric, threshold or task.

## Cause

**What was measured** (A12's document, sections "Results" and "The hold-out runs"). With the
gate a run can score 98 % on keys it trained on and anything from 99 % to 33 % on keys of
bytes it has never seen:

| Model, 18,000 steps on `synthetic_hold` | kv: training keys | dev / test keys | delayed: new pairs of seen bytes | never-seen bytes |
| --- | --- | --- | --- | --- |
| 8 fast + 12 routed, gate, seed 0 | 98.3 | 84.2 / 87.3 | 99.0 | 96.0 |
| the same, rate constant | 97.9 | 95.8 / 97.1 | 99.6 | 98.7 |
| the same, `G` held from 9,000 | 98.6 | 32.8 / 40.6 | 98.4 | 46.6 |
| the same, seed 1 | 99.6 | 97.4 / 97.7 | 98.4 | 98.5 |
| 12 routed, gate, no fast cells | 99.2 | 99.1 / 98.9 | 99.8 | 99.7 |
| 8 fast + 12 routed, no gate | 95.5 | 96.1 / 95.5 | 93.8 | 94.1 |

- The first three runs are one configuration and one seed for 9,000 steps and stand at 98.1,
  95.8 and 48.0 % on dev keys at that step: it is the run's path, not its configuration.
- It happens without fast cells too (the pair task, 9,000 steps: 98.0 against 81.5 / 87.9),
  and never without the gate and without fast cells (three runs split).
- In the first run the 18,000-step model with its 9,000-step `G` scores 94.7 % on unseen keys
  against 85.8 % with its own: most of the fitting is in `G`.
- Keys made of bytes the model trained on, new combinations included, are at 98 to 99 % in
  every checkpoint.
- The baselines that solve the pair task at all solve it for the training keys only (Mamba
  78.5 % against 28.3 %, the Transformer with the byte codes as an input 100.0 against
  54.9 %); the HSN without the gate is at 97.7 against 97.2 %.

**Why the gate can do it (a reading, not measured).** `G` is applied to `u_t`, the output of
the shared transform for the position. `u_t` is computed from the encoded current byte, the
carried summary and the context, and with A8's `fast_readout` from the fast cells' read as
well. So the weight a unit puts on an address feature can depend on which byte the position
holds. For a byte the model has never seen in training `u_t` lies where `G` was never trained.
What the gate has to tell apart is the kind of position (a queried key against a value), which
the byte before it settles, not the byte itself.

## The change (opt-in; needs `model.address_gate`)

A12's term is unchanged in form,

```
logit[i, j] += Σ_a (G r_t)[j, a] · address_a[i]
```

and `r_t`, which is `u_t` in A12, becomes one of:

- **`summary`: `r_t = q_{t−1}`**, the carried summary as `S` reads it (A6's `summary` read).
  It holds what the cells selected at the last position gave out and not the current byte.
  At a queried key it has seen the query symbol and not the key; at a value it has seen the
  key's position.
- **`u_no_fast`: `r_t = S([z_t ; q_{t−1} ; ctx_t ; 0])`**, the shared transform's output with
  the fast group's read left out of its input (the same weights of `S`, a second evaluation).
  It still holds the current byte and no longer what the fast cells hold.

In both, `G` keeps its shape (`d → M · A`), its zero start and its parameter count, and the
addresses are A5's and A9's fixed codes as before. Under `train.feedback_detach` with `router`
`r_t` is a constant in the backward, like the scorer's `u`: `G` is trained by the selection's
straight-through gradient and no gradient path into `S` or the cells appears.

**What stays.** Cells, graph, units, ε, the convex tanh gate, the selection rule, the address
features, `q`, the fast cells. No attention over the sequence, no new state, no new parameter.

**Cost per position:** none for `summary`; one more evaluation of `S` for `u_no_fast`.

**Windows.** Eager and compiled autograd windows. The split and fused windows refuse the key.

## The smallest test, and its kill rule

On `synthetic_hold`, the candidate shape (8 fast + 12 routed slots, 64 cells, two-byte address,
`wrap`, state gradient, 18,000 steps, checkpoints every 3,000), scored on fresh documents with
`hold_eval.py` at every checkpoint:

1. `summary`, three seeds;
2. `u_no_fast`, three seeds;
against the four runs of the table above with the gate reading `u`.

Same-seed runs cannot be compared (bf16 rounding reroutes every row), so the arms are read by
their seeds and by the split into training keys and unseen keys, not by dev curves.

**It works** if on every seed, at 18,000 steps, the dev and test keys are within 3 points of the
training keys and the never-seen bytes within 3 points of the seen pairs, with the training
keys at 97 % or more (the gate still does A12's work). Then: the pair task of synthetic_v1 at
27,000 steps (A12's deciding arm must still hold), and TinyStories and arithmetic on the full
candidate, which must stay within their seeds' spread (0.959 bits per byte, 99.9 %).

**Kill rule.** A variant is dropped if two of its three seeds lose more than 3 points on unseen
keys, or if its training keys end below 95 %. If both variants are dropped, the fitting does not
come from what `G` reads, and the next candidate is a limit on `G` itself.

## Tests

`tests/test_a13_gate_input.py`: opt-in and hashing, scope (needs the gate; `u_no_fast` needs
the fast read; the split and fused windows refuse); the parameters are A12's, value for value;
the logits against a direct computation, the gate independent of `u`, features without the
gate's input refused; a zero gate is the model without it (evaluation and straight-through, both
variants); the window against stepping, and another model than A12 with the same weights;
`summary` reads the carried summary only and is a constant in the backward with the router's
cut; `u_no_fast` is `S` without the fast read and does not move with the fast cells' state; the
task loss reaches the gate under the state gradient's cut; the compiled window against the
eager one; thirty training steps with each variant.
