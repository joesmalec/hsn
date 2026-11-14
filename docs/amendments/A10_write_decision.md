# Amendment A10: the router's write decision, with a cost (dropped)

Status: **built as an opt-in experiment on 13 Nov 2026, tested the same day and dropped by its
own kill rule (the user's decision, 13 Nov 2026). Its code is not on main.** This text is the
record of what was tried and what it showed. The implementation is the local commit `80f06d7`
on the branch `claude/arch-amendments` (keys `model.write_gate`, `model.write_init`,
`train.lambda_w`; tests `tests/test_a10_write_decision.py`).

## What it was

The scorer got a second linear output of the same features, one write logit per (cell, unit).
A selected routed cell was executed and read as before, and written only where the write logit
of its chosen unit was positive:

```
w_i  = 1 if ω_ij > 0 else 0
h_i' = (1 − w_i·b_i) ⊙ h_i + w_i·b_i ⊙ c_i
```

Straight-through in training, the fast cells of A8 always written, the output starting open so
that at step 0 the model was the model without the key. `train.lambda_w` added a cost: the share
of the routed slots that may write.

## Why it was tried

Selecting a cell executes it, reads it and writes it through a gate that does not see the
address match. On the pair task the key-value score fell as slots were added (8 slots 93.9 /
91.7 %, 12 slots 86.4 / 78.4 %, 16 slots 63.5 / 67.0 %) while the writes per position rose
(0.55, about 1, about 2.7). The reading was that too many cells are written per fact. A second
target was story text, where memory never forms once any text is supervised.

## Results (13 Nov 2026; state gradient, 9,000 steps, one seed per arm)

Pair task, 12 slots, 32 cells (control `g1_k12_cutC`: 89.7 / 78.4 % in range, 79.6 / 72.9 % out
of range):

| Arm | Cost | In range | Key-value part | Out of range | Slots that may write |
| --- | --- | --- | --- | --- | --- |
| `g1_k12_wg_cutC` | 0 | 81.3 | 75.9 | 68.2 | 48 % |
| `g1_k12_wg02_cutC` | 0.02 | 76.9 | 71.0 | 54.6 | 46 % |
| `g1_k12_wg10_cutC` | 0.1 | 89.7 | 86.6 | 35.8 | 24 % |
| `g1_k12_wg50_cutC` | 0.5 | 65.1 | 59.1 | 51.1 | 12 % |

Pair task, 12 slots, 64 cells, position wrapped (A11), cost 0.1:
`g1_k12nw_wg10_cutC_cells64_poswrap` 78.5 % (key-value 70.8 %), against 98.0 / 98.6 / 98.3 %
(key-value 97.3 to 98.1 %) for the same setting without the write decision.

## Outcome

- **The mechanism did what it was built to do**: the write rate fell by half or more.
- **It did not help.** No arm is above the control at 32 cells, the arm at 64 cells loses 20
  points in range, and recall past the window is worse in every arm. Both conditions of the
  kill rule were met ("not above the control at any cost"; "must not cost where the base is
  already good").
- **What limited the pair task instead**, found the same day: the number of cells (64 cells
  give 98 % at 8 and at 12 slots, with no parameter added) and, past the training window, the
  scorer's position feature (A11). What remains past the window is interference from later
  writes at counts no training document has.
- **Story text was never tested.** The arms that combine the write decision with A8's fast
  cells stalled on the GPU at the switch from the soft warm-up to the hard gate (below).

## A defect that was found and not fixed

With `model.write_gate` and fast cells (`fast_cells=8`, `fast_units=fixed`, `fast_readout`,
20 slots) the compiled training window never completed its first hard-gate step on the rented
cards: the process stayed at one busy core with the card idle for 26 minutes, and again in a
diagnostic run with a 20-step warm-up. The one frame a fault-handler dump gave was in
`torch/_inductor/cudagraph_trees.py` (`maybe_deref` calling `data_ptr`, the storage
weak-reference bookkeeping of the CUDA-graph tree), torch 2.15.0+cu132, compile mode
`reduce-overhead`. It did not reproduce on the CPU (no CUDA graphs there): the compiled window
equalled the eager one in all three modes and the trainer passed the switch at full model size.
Evaluations during the soft phase ran, so the forward of the sparse round was not the problem.
The arms without fast cells passed the same switch in a few minutes.

## If it is taken up again

- The write decision removes writes the model needs for binding: at 64 cells the key-value part
  fell from 97 to 71 %. A version that cannot close a write the address selects (for example a
  decision on the routed cells' gate at text positions only) would be a different amendment.
- The stall has to be understood first, or the arms run without CUDA graphs.
