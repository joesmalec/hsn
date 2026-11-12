# Amendment A7 (proposal): the executed unit's credit (`train.unit_credit`)

Status: **proposed, not adopted** (12 Nov 2026). The key exists as opt-in code (default
`softmax`; `tests/test_unit_credit.py`); no recipe sets it and the code's defaults are unchanged.
No forward value changes: it is a choice about which scores the router's straight-through
gradient reaches. This document is the case for and against adopting it together with two
existing keys. The user asked for it on 12 Nov after the twelve-seed runs of the night before;
the decision is the user's. Runs and measurements: `docs/a6_takeover_followup.md`, sections 3b
and 3c.

## What it is

A selected cell runs one unit, the one with the largest score. In training with the
straight-through gate (`gate_mode=ste`) the unit weights of a cell are

    w = onehot(c) + (p − stopgrad(p)),    p = softmax(s / τ)

with `s` the cell's scores over the M units and `c` the unit that ran. The values are the one-hot
vector; the gradient is the softmax's.

| | Gradient of the chosen unit's weight `w_c` | A unit that did not run |
| --- | --- | --- |
| `train.unit_credit=softmax` (default) | `∂w_c/∂s_c = p_c(1 − p_c)/τ`, `∂w_c/∂s_k = −p_c p_k/τ` | its score on this cell is lowered whenever the chosen unit's is raised |
| `train.unit_credit=executed` | `∂w_c/∂s_c = p_c(1 − p_c)/τ`; the other scores enter the softmax as constants | gets no gradient from the slot |

- **Unchanged:** every forward value (the decision, the written state, the logits, the losses),
  the cell-mask path of the same estimator, the soft warm-up and dense-fallback rows (every unit
  runs there, and they keep the softmax), evaluation, serving, checkpoints.
- **Where it runs:** `Router._finish` (`python/hsn/model/router.py`), in the autograd
  window and in the replay of the split and fused windows.

## What is proposed

Not the key alone. The arm that was tested is the adopted A6 recipe
(`configs/experiments/hsn_recipe_a6.yaml`) with three more keys:

    train.soft_until_step: 1000          # the white paper's soft-gate warm-up (§6)
    train.lambda_b: 0.02                 # the white paper's balance loss (§6, §7)
    train.aux_enabled: [balance]
    train.unit_credit: executed          # this amendment

Only the last is not a mechanism of the white paper. If adopted it would be a new recipe file
(a recipe is never edited once a spec uses it); none has been created.

## Why: what was measured

The delayed task is answered by key when the unit that writes a value ranks cells by the
previous byte's address (`a_prev`) with the same sign as the unit that reads a query key ranks
them by the current byte's (`a_cur`). In the 24 runs whose checkpoints were kept (section 3c):

1. Binding is carried by two to six writer units with small `a_prev` weights of the reader's sign.
2. A run falls when one unit whose `a_prev` weight has the opposite sign takes the value
   positions: all seven runs that fell end with such a unit on 37 to 74 % of the value slots.
3. The default credit manufactures such units. A writer's credit lowers the other units' scores
   on the key's cells, so a unit that never runs at a value position acquires an opposed weight:
   in 12 runs of 12 by step 2,000 with the default credit, in 1 run of 12 with executed credit.
   On one set of weights the push on four idle units' `a_prev` weight is −2.5 each with the
   default credit and 0.00 with executed credit.
4. The balance loss keeps the reader off the value positions (the plateau's cause) and moves
   usage between units; with the default credit it thereby hands value slots to opposed units.

So the two parts are complementary by mechanism, not only by count: the balance loss prevents
the first failure and executed credit removes the source of the second.

## Evidence: the runs

Fused window, 9,000 steps, diagnostics off, 11 to 12 Nov. "Binds keys": later queries at 80 % or
more. "Fell": 10 pp or more below the run's own best.

| Arm | Runs | Mean final dev | Binds keys | Passes the binding verdict | Fell | Drawdown under 5 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm-up only | 12 | 81.3 % | 4 of 12 | 2 of 12 | 2 of 12 | 9 of 12 |
| Warm-up + balance loss | 12 | 85.4 % | 9 of 12 | 6 of 12 | 5 of 12 | 5 of 12 |
| **Warm-up + balance loss + executed credit** | 12 | 87.1 % | 9 of 12 | 8 of 12 | 2 of 12 | 9 of 12 |
| Warm-up + executed credit | 6 | 84.6 % | 1 of 6 | 1 of 6 | 1 of 6 | 5 of 6 |
| Balance loss + executed credit, no warm-up | 6 | 79.9 % | 1 of 6 | 1 of 6 | 2 of 6 | 2 of 6 |
| The adopted recipe (no warm-up, nothing added) | 6 | 87.6 % | 3 of 6 | 2 of 6 | 2 of 6 | 2 of 6 |

- Against the warm-up alone the proposed arm's final dev (rank test, p = 0.04) and its binding
  verdicts (8 of 12 against 2 of 12, p = 0.04) differ; against the balance loss alone its falls
  (2 against 5 of 12, p = 0.37) do not, on these counts. None is corrected for the eleven arms
  compared.
- **It does not raise the mean over the adopted recipe** (87.1 % against 87.6 % on six fused runs
  and 87.2 % on nine autograd runs of 10 Nov). What it changes is how often a run binds its keys
  and how often it stays within 5 pp of its best.

## What it does not do

- It does not meet the rule of `docs/p1_18_a6_results.md`, section 5b (no fall of 10 pp, worst
  drawdown under 5 pp, most runs pass the verdict): two runs of twelve fell, one by 26.4 pp, the
  largest fall of the night.
- It does not work alone or without the warm-up (table).
- The remaining failure is measured on one run (section 3c): a unit that enters the value
  positions is pushed off the key's cells by the task loss and then wins the position's slots
  with other cells; a cell that lost its slot gets no credit.

## What it is not

- Not a change to the cells, the graph, the units, ε, the tanh-convex gate, the router's scores
  or decision, or `q`. Not a guard, a skipped step or a rollback.
- Not unbiased: a unit that never runs at a kind of position now learns nothing about it from
  the unit path. The warm-up (every unit runs) and the balance loss (usage moves) are what give
  such a unit its chance, which is why the three keys are proposed together.

## Cost and risks

- Step time and memory: none beyond the warm-up's (its 1,000 steps run the autograd window at
  0.216 s; a run takes 12 minutes, against about 7 for 9,000 steps of the recipe at 45 ms).
- Tested on one task (delayed, in range), one model size, 9,000 steps, the fused window. Not run:
  27,000 steps, the five-task mix, a second task, the autograd window with all three keys.
- The comparison with the adopted recipe rests on six fused runs of that recipe.

## Experiments that would decide it

Each needs the card; none has been run.

1. Eight slots with executed credit and the warm-up (`model.k_select=8`), with and without the
   balance loss, twelve seeds: the measured remaining failure is a shortage of slots at the value
   positions, and eight slots had no fall in seven runs with the default credit. 13 to 18
   minutes a run.
2. The proposed arm and the adopted recipe side by side, twelve more seeds each, same day and
   window: the recipe's count is the weak side of the comparison.
3. `train.ste_scope=all` with the proposed arm (a cell that lost its slot gets its first-order
   credit), six seeds at the dense step's time (about 32 minutes a run, 7 GB).
4. Three runs at 27,000 steps, and one other task.

## Configuration

    train.unit_credit: softmax     # default; left out of the configuration hash
    train.unit_credit: executed    # hashes; HSN only

`configs/sweeps/a6_5b_remedies.yaml` has the arms (`fused_softwarm_balance_executed`,
`fused_softwarm_executed`, `fused_balance_executed`).

## The decision asked

- **Adopt** as a new standard recipe now; or
- **decide after experiments 1 and 2** (the recommendation: the arm is the best measured, its
  mechanism is understood, and it neither meets the rule nor beats the adopted recipe's mean); or
- **leave it** as an opt-in key.
