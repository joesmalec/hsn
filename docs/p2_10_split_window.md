# P2.10: the split training window

11 November 2026. Branch `p2.10-cuda-kernels`. Measured on the RTX 3060 (`profile:rtx3060`), the
card to itself, torch 2.15.0+cu132, the standard recipe `configs/experiments/hsn_recipe_a6.yaml`
(routed top-K, B32, T256, bf16) with `hardware.compile_mode=reduce-overhead`.

## What this is

`hardware.window=split` runs a training window of the routed HSN as

1. a **forward sweep without autograd** that records a tape,
2. one **batched replay with autograd** of the work that is local to a position, and
3. a **linear backward recurrence** over the state, as one autograd node,

instead of one autograd graph over the 256 positions. The gradients are the model's own. The
model's equations, the routing code and the trainer's `backward()` call are unchanged; the key is
off by default and leaves every existing configuration hash where it was.

| | Step time, median | Against the autograd window |
| --- | ---: | ---: |
| Autograd window (`hardware.window=autograd`, the default) | 100.7 ms | 1.00× |
| Split window, first version (PyTorch replay, recurrence kernel) | 75.8 ms | 1.33× |
| + the sweep twin | 64.6 ms | 1.56× |
| + the replay without the assembled features and per-unit slices | 59.8 ms | 1.68× |
| + the trainer's per-step work through multi-tensor ops | 56.3 ms | 1.79× |
| Fused sweep (`hardware.window=fused`): two glue kernels that write the tape in place | 50.4 ms | 2.00× |
| + the chosen unit's second layer only, and matrix products that share an input merged | 46.0 ms | 2.19× |
| + the replay's reorderings with a gather-based backward | 45.4 ms | 2.22× |
| + dense inputs for the sweep's blocks (the split window too: 54.3 ms, 1.85×) | 44.6 ms | 2.26× |

(`scripts/bench_window.py time`, 60 warm-up and 300 timed steps per process, statistics, evaluation
and checkpoints off; reports under `evidence/P2.10/speed/`; the split window measured 55.5 to
56.3 ms across the day's sessions.) The stretch target of this work was a factor of six to seven
(the step time of `gru_2m`, about 17 ms). It was not reached; the section "What did not work" says
why and what would be needed.

**Final check, one session** (11 Nov, 23:34 MST, nothing else on the card, the three windows one
after the other; `evidence/P2.10/final/`): autograd 97.8 ms, split 54.3 ms (1.80×), fused 44.4 ms
(2.20×). The autograd window was 3 ms faster than in the afternoon's session that the table above
divides by, so against it the day's ratios are about 3 % high; the split and fused step times
did not move.

## Why the per-op kernels could not win

A compiled autograd window issues about 31,700 GPU kernels per step (`profiles/rtx3060_L1.json`):
about 124 per position, 2.6 µs each, with the card 92 % busy. The step is bound by the *number* of
kernels, not by arithmetic, so replacing one small op by one small custom op (the four P2.9 ops of
this chunk's first half) cannot change it: with the CUDA backend the step measured 93.6 ms against
93.0 ms native. Training also never selected that backend (no configuration key did).

## Why the window can be split

Amendment A6's feedback cut (`train.feedback_detach: [summary, eps, state, router]`, part of the
standard recipe) makes the feedback reads of the state constants in the backward pass. What is
left that crosses positions is

- the gated carry of the cells a position writes, `h' = (1 − b) ⊙ h + b ⊙ c`,
- the linear summary update, `q' = q + Σ (h' − h) / N`,
- the reads of `q` by the predictor and the decoder and of the selected cells by the readout.

Everything expensive (messages, cell norm, gate, the unit bank, `W_out`, the scorer) reads
constants and is local to its position. So its backward does not have to run position by position.

With `A [B, N, d]` and `R [B, d]` the cotangents of the state after position `t`, and `D_t`,
`P_t`, `S_t` those of `q` after `t`, `q` before `t` and the selected updated states:

```
A = 0; R = 0
for t = T−1 .. 0:
    R += D_t;  A[cell_k] += S_{t,k}                     (valid slots)
    g_k = A[cell_k] + R / N
    dĉ_k = b_k ⊙ g_k;  db̂_k = (c_k − h_old_k) ⊙ g_k     (exact zeros on padded slots)
    A[cell_k] = (1 − b_k) ⊙ A[cell_k] − b_k ⊙ R / N
    R += P_t
    A, R = 0 on the rows that reset at t
```

A row that runs the dense round for the window (`train.p_dense`) writes every cell and sets
`q' = mean(h')`, which does not read `q`: with `g = A + R / N` its sweep is `dĉ = b ⊙ g`,
`db̂ = (c − h) ⊙ g`, `A = (1 − b) ⊙ g`, and `R` restarts at `P_t`.

## The parts

**Forward sweep** (`hsn.train.compile._tape_block`, `run_tape`, `CompiledWindow.tape`). The
positions in order under `torch.no_grad()`, as compiled blocks with CUDA Graphs. Each position
calls the model's own `HSN._position`, so the scorer, `Router.select` and every routing control
are the model's code; the sweep only asks for the hard decision (`gate=None`: the same slots).
It records the pre-round state of every position after its reset, the decision (cells, units,
valid slots, the straight-through threshold, which `Selection` now carries), `q` after the
position and the updated states of the selected slots.

**The sweep twin** (`hsn.train.compile.SweepTwin`). Under autograd a compiled block casts every
autocast weight to bf16 at every position (it has to: a window's gradients are summed in the fp32
masters), and it receives the window's `W1` layout as a call argument, which a CUDA graph copies
into its input buffer at every block. The sweep needs neither. It runs a structural copy of the
model whose cast weights are refreshed once per window and whose other tensors are the model's
own. This took 11 of the sweep's 43 ms. `hardware.sweep_twin=false` (default true, omitted from
the hash at the default) runs the sweep on the model's own compiled blocks instead: about 11 ms
per step slower, with inductor's fused casts in place of the twin's strictly rounded weights.

**The fused sweep** (`hardware.window=fused`; `hsn.train.compile._fused_tape_block`). The same
tape from the same matrix products, with less between them. A census of the split sweep's CUDA
graph (`evidence/P2.10/speed/s9/base-census.txt`) counted 52.6 kernels per position and 126 µs of
device time, of which the ten matrix products were 56 µs: the rest was some forty small kernels
(top-K, scatter, gather, reset, statistics, tape stacks) at 1 to 2 µs each. The fused sweep has
23.7 kernels and 81 µs per position:

- *Two glue kernels* (`kernels/cuda/hsn_window.cu`). `hsn_win_select` is `Router.select`'s hard
  decision (scores, the stable top-K, the budget prefix, the threshold, the refresh probes, with
  the router's tie rules); `hsn_win_cell_update` is the end of the round (candidate norm, `tanh`,
  the gate's sigmoid, the gated update, `q`, the ages). Each has a tape form
  (`hsn.kernels.window.select_tape`, `update_tape`) that reads record `t` of the block's tape and
  writes record `t + 1` in place, with the next position's reset applied. A block therefore makes
  no per-position state tensors and stacks nothing. Inductor keeps a chain of mutating custom ops
  on a buffer made inside the compiled function in place (checked: no clone per call).
- *The chosen unit's second layer only* (`UnitBank.forward_static_chosen`). The static pass runs
  every unit on every pair and gathers one. The first layer has to (the pairs' units differ, and
  one matmul over all of them keeps the shapes static), but the chosen unit's hidden row can be
  gathered before the activation, so the activation and the second layer run on `[P, H]` instead
  of `[M, P, H]`: one matmul against `W2` laid out `[H, M·d]`.
- *Merged matrix products* (`SweepTwin(fused=True)`, `sweep_merge_error`). A small bf16 matmul
  costs 2 to 6 µs whatever its size, so maps that read the same input share one, with the same
  operands (a map that reads part of the input has zeros elsewhere): `S` with `P` (one two-layer
  map of `[z; q; ctx]`, the second layer block-diagonal), the scorer with `V` (which reads the
  state columns of the scorer's features), the unit bank's first layer with the gate. Ten
  matmuls per position become six.

What a small kernel costs on this card decided the design of the two kernels (measured by
letting them return early, `evidence/P2.10/speed/s9/`): an empty kernel 0.8 µs; a first stage
with global reads and one `__syncthreads` about 1.2 µs more; serial code on one thread about
3.4 ns per instruction, so a 32-step loop is half a microsecond; and a thread that walks global
memory in a dependent loop waits for every read. The first versions of the two kernels did the
last two and took 5.6 and 10.8 µs, as much as the kernels they replaced. Now each value is read
once by its own thread into shared memory, reductions go through warp shuffles, and they take
4.3 and 6.5 µs in the graph.

*Dense block inputs* (`run_tape`). A CUDA-graph replay copies a strided input by the byte extent
it spans (`torch._inductor.output_code.copy_strided_storage_`, taken whenever the placeholder
is not dense), and a column slice `z[:, t0:t0+16]` of a window spans nearly the whole window:
the sixteen blocks of a step copied 195 MB of `z`, context and scorer scalars between them. The
sweep now lays each of them out by block once per window; this is worth 0.8 ms to the split
window as well. (The autograd window passes the same kind of slices and was left as it is.)

**Replay** (`SplitWindow._replay`, `PDGCell.replay_sparse`, `Router.finish_recorded`). One pass
over all `B·T` rows with the existing modules: `S`, the scorer, the gradient path of the recorded
decision, the messages, norm, gate, the selected unit of every pair, `W_out`, the candidate; then
`P`, the selected readout and the window epilogue (decoder, `info`) of `HSN`. Nothing is decided
again. Three things keep it cheap:

- the linear scorer's product is taken per feature block instead of over the assembled
  `[B·T, N, 3d+8]` features (200 MB with `u` and `eps` repeated for every cell);
- `V` is applied to the gathered message sources only, so nothing is scattered back over all
  cells in the backward;
- `UnitBank.forward_grouped` cuts the sorted pairs with one `split`: a slice per unit costs a full
  `[P, 4d]` gradient tensor per present unit in the backward (eight 33 MB tensors per window);
- its two reorderings of the pairs (sorted by unit, and back) are `permute_rows`: the backward of
  `index_select` adds every row of the gradient into a tensor of zeros, and for a permutation a
  gather in the inverse order gives the same gradient bit for bit.

One thing the replay must not take from its own recomputed scores is *which unit a selected cell's
score belongs to*. `Router.select` reads a cell's score as the largest of its unit scores, and
that is the unit it runs: one tensor decides and is differentiated. The replay's logits are
recomputed, and under bf16 they can round a near-tie between two units the other way than the
sweep's did; the largest recomputed score is then a unit that did not run, and the cell mask's
straight-through gradient went to it. `Router.finish_recorded` therefore reads the score at the
recorded unit (`executed_scores`). On the A6 recipe the two readings differ on 0.01 to 1.3 % of
the slots, which sounds negligible and is not: they are the contested slots. At step 6,000 of
seed 0, while a route was being taken over, 57 % of one unit's scorer-row gradient depended on the
reading; in a run that never formed its routes, 9 to 17 % of five units' rows at step 4,000
(`evidence/P2.10/speed/s8/executed-unit-credit.log`). In fp32 and fp64 the readings are the same
unit, which is why the equality tests did not see it. The accuracy runs of this document are
listed by replay for this reason.

**State recurrence** (`hsn.train.split.StateRecurrence`). Forward returns the taped `q` and
selected states; backward is the sweep above, as `hsn_win_recurrence` (one launch per window, one
block per batch row, 0.4 ms) and `hsn_win_recurrence_dense` for dense-fallback rows, with PyTorch
references for other devices and dtypes. Because it is a node of the one graph, per-term gradient
norms on statistics steps come from retained backwards of the same graph: a split run never
compiles or runs the autograd window.

**Trainer** (`tests/test_p2_10_trainer_speed.py` compares each with its per-tensor original, bit
for bit, on the CPU and on CUDA):

- `LazyAdamW` steps a group's ordinary tensors through torch's multi-tensor ops (3.2 → 1.7 ms);
- the float64 clip makes its float64 copies once and returns early when nothing is clipped
  (1.9 → 1.1 ms);
- the update cap's elementwise steps go through the multi-tensor ops.

## Where a step's time goes now

Synchronised medians (`scripts/bench_window.py phases`, and the trainer's parts timed around it):

| Part | Split (14:53), ms | Fused (17:14), ms |
| --- | ---: | ---: |
| Window prologue (embeddings, `E`, scorer scalars) | 1.5 | 1.5 |
| Forward sweep (52.6 and 23.7 kernels per position) | 31.3 | 20.6 |
| Replay, forward | 6.7 | 6.6 |
| Backward (replay and recurrence) | 9.0 | 8.7 |
| Optimiser step, clip, update cap, weight average, the rest | about 7 | about 7 |

Inside the fused sweep's 81 µs per position: the six matrix products 45 µs (the all-units first
layer with the gate 17.5, the scorer with `V` 9.8, the second layer 6.5, `S`/`P` 4.9 and 4.6,
`W_out` 2.0), the two glue kernels 10.8, and thirteen small kernels (feature assembly, casts,
gathers, norms, activations) 25.

## Several arms on one card

The split window leaves the card idle between its kernels' launches (the sweep is bound by the
host issuing 52 small kernels per position), so concurrent arms gain from NVIDIA MPS, which lets
their kernels overlap instead of taking turns:

| Arms on the card | Step time per arm | Throughput against one arm alone |
| --- | ---: | ---: |
| 1 | 56.3 ms | 1.00× |
| 2, MPS | 85.6 ms | 1.32× |
| 3, MPS | 112.6 ms | 1.50× |
| 6, MPS | 191.7 ms | 1.76× |
| 3, no MPS (time-sliced) | 162.9 ms | 1.04× |

Six split arms under MPS finish a sweep at 3.1 times the rate of the autograd window run one arm
at a time (100.7 ms × 6 / 191.7 ms). Under MPS a process also holds less card memory (the context
is shared): 1.3 GB while training and 1.7 GB at the end-of-run probes for a split arm (1.65 GB
alone), 2.0 GB for an autograd arm (2.6 GB alone). The daemon needs no root with a private pipe
directory, and only processes started with that variable use it:

```
export CUDA_MPS_PIPE_DIRECTORY=<dir>/mps-pipe CUDA_MPS_LOG_DIRECTORY=<dir>/mps-log
nvidia-cuda-mps-control -d                    # start; the worker and its arms inherit the variables
echo quit | nvidia-cuda-mps-control           # stop (with the same CUDA_MPS_PIPE_DIRECTORY)
```

The experiment worker's memory rule reads a new arm's reservation from the peak of finished arms
of the same shape, measured alone; under MPS pass `--mem-estimate-mib` at `enqueue` (1750 for a
split arm, 2200 for an autograd arm on this recipe) or the worker runs fewer arms than fit.
Timing arms still run alone.

## Scope

`hsn.train.split.scope_error` names what a configuration lacks. The split window needs: routed
`topk` with the `ste` gate, all four feedback paths cut, no exploration
(`train.epsilon_*`), no counterfactual probes (`train.lambda_u`), no state-gradient clip, no
disabled cells, one round, bf16 or fp32. Dense-fallback rows (`train.p_dense`), the balance loss
(`train.lambda_b`), the cost term and a temperature schedule are covered: the selection's
gradient path is ordinary PyTorch in the replay, and `info` carries what the autograd window's
does (`unit_logits` with its graph, `expected_cost`, `unit_idx`, `valid`, `cell_idx`, the rates).
A window whose dense-fallback replay would not fit memory (more than about 4 GB) runs the
autograd window from the same draw, as does every window with dense-fallback rows under
`hardware.window=fused` (the fused round has none). Evaluation, probes and diagnostics run the
windows they ran before.

A soft warm-up (`train.soft_until_step`) is inside the scope. Its steps run every cell with
autograd, which is the autograd window's work, so they take that window (the same losses as a run
without the key, bit for bit: checked on the card), and the split or fused window takes over at
the first hard-gate step (`SplitWindow.splits_now`). On the A6 recipe a warm-up step takes
0.216 s and the steps after it 0.046 s; the run holds the warm-up's memory to its end (5.6 GB
allocated, 7.2 GB reserved), so such arms run one at a time.

`hardware.window=fused` also needs CUDA, a compiled window and the built window kernels, and a
model its kernels take (`hsn.kernels.window.select_unsupported`, `cell_update_unsupported`).

## How equal it is

| Comparison | Result |
| --- | --- |
| CPU float64, split against `HSN.forward` (interior resets, carried start state, padded slots, refresh probes, cost and balance terms, a window without loss positions, dense-fallback rows) | every parameter gradient equal at rtol 1e-9, the same set of absent gradients |
| CUDA fp32, compiled, split against the autograd window, 300 training steps in | identical routing decisions; gradients equal to 1.2e-7 relative |
| CUDA bf16, the same, each against the fp32 autograd reference | autograd bf16: 2.7 % relative, cosine 0.99964. Split bf16: 2.8 %, cosine 0.99963 |
| CUDA bf16, one tape, the replay and backward under bf16 against the same replay in fp32 | 0.33 % relative, cosine 0.999996 (the split backward's own bf16 error, with decisions and state values held equal) |
| Sweep twin against the model's own sweep | fp32: bit-identical tape. bf16: equal to rounding (below) |
| CUDA fp32 at a trained checkpoint (step 4,000 of a run), 40 real training windows through the trainer with its full loss, weights frozen | identical routing decisions; every parameter group equal to 1.2e-6 per step |
| The same with a statistics step every other step, and with real evaluations (averaged weights swapped in and out) between the steps | unchanged (below 1e-7) |
| CUDA bf16 at that checkpoint, per step | autograd bf16 against fp32: 96.5 % of the slots run the same unit, gradients 23.7 % apart. Split bf16 against fp32: 96.5 %, 23.1 %. Split bf16 against autograd bf16: 99.1 %, 7.9 % |
| The same, gradient norm of every parameter tensor, split bf16 over autograd bf16 | 0.992 to 1.003 (the eight-element scorer bias 1.03 ± 0.13) |
| The same, routing statistics | unit shares equal to four decimals, mean selected cell index 16.90 in all three bf16 paths (fp32: 17.01): no tie-break skew from the twin |

Under bf16 the split window is not bit-for-bit the autograd window, and a run is not
bit-reproducible against earlier runs:

- bf16 rounding flips top-K picks either way (on a window of random bytes only about 40 % of the
  selected cells agree between *any* two of fp32, autograd bf16 and split bf16), so the two paths
  follow different trajectories of the same model;
- inductor fuses the per-position casts with neighbouring arithmetic, while the twin holds
  strictly rounded weights (the values eager autocast computes);
- a batched matmul rounds its weight gradient to bf16 once per window, where the per-position
  gradients were summed in fp32 (the encoder's and the decoder's already were).

The 2.7 to 2.8 % above are therefore differences between trajectories, which the autograd window
has against fp32 as well; on one trajectory the split backward is within 0.33 % of fp32. The
tolerance to hold the split window to is the fp32 one, plus the accuracy of runs.

At a trained checkpoint on real documents the trajectories part further (a bf16 path against
fp32: 24 % per step), and the two bf16 paths stay closer to each other than either is to fp32.
The measurement is `evidence/P2.10/speed/s8/trained_grads.py`: every arm starts from one
checkpoint's weights, the optimiser step is replaced by a recorder, so all arms see the same
weights and documents for 40 steps.

## Accuracy

The A6 recipe, 9,000 steps, B32, diagnostics off unless said; every run in one database
(`/home/joe/hsn-experiments/p2.10-split/experiments.sqlite` on the development machine):

| Arm | Runs | Mean final dev | Range | Spread (s.d.) | Mean later queries | Fell 10 pp or more | Worst drawdown | Pass the binding verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Autograd window, the night before (c5a82ea, diagnostics on) | 9 | 87.2 % | 78.5 % to 95.6 % | 4.7 pp | 80.8 % | 1 of 9 | 13.7 pp | 3 of 9 |
| Autograd window, 11 Nov (this branch) | 6 | 80.7 % | 62.6 % to 91.4 % | 10.4 pp | 70.5 % | 3 of 6 | 14.4 pp | 2 of 6 |
| Split window, first replay (credited a unit that had not run) | 8 | 82.9 % | 58.9 % to 96.0 % | 10.3 pp | 71.4 % | 4 of 8 | 18.8 pp | 2 of 8 |
| Split window, the replay credits the unit that ran (9e0a61a on) | 5 | 88.0 % | 81.9 % to 94.3 % | 4.6 pp | 81.1 % | 0 of 5 | 8.8 pp | 3 of 5 |
| The same without the sweep twin | 1 | 92.2 % | | | 86.9 % | 0 of 1 | 1.0 pp | 0 of 1 |
| Fused window (the afternoon's sweep, 3008c52) | 6 | 87.6 % | 79.7 % to 94.8 % | 5.7 pp | 79.5 % | 2 of 6 | 16.9 pp | 2 of 6 |

Final dev by seed: autograd on 11 Nov 88.0, 91.4, 77.1, 90.6, 74.4, 62.6 (seeds 0 to 5); split
with the fixed replay 88.6, 83.6, 91.4, 94.3, 81.9 (seeds 0, 1, 2, 4, 5); fused 79.7, 85.4, 88.9,
94.4, 94.8, 82.4 (seeds 0 to 5).

- The split window with the replay fix and the fused window are no worse than the autograd
  window on these counts: 88.0 % on five runs and 87.6 % on six, against 80.7 % on six the same
  day and 87.2 % on nine the night before. The first replay, which credited a unit that had not
  run, was worse.
- One configuration differs by 5 to 10 pp from run to run (one seed gave 58.9 %, 74.4 % and
  94.3 % on the day), so five or six runs do not resolve a difference of a few points between
  windows. What they do show is that no window is far from the others.
- The falls of 10 pp or more (three of the six autograd runs of the day, two of the six fused
  ones) are the route takeover, a property of the recipe and not of a window:
  `docs/a6_takeover_followup.md`.
- With a soft warm-up (`train.soft_until_step=1000`), six seeds each: 81.7 % on the autograd
  window, 82.0 % on the fused window after the warm-up.

## What did not work

**A fused cell round** (the morning's `hardware.window=fused`; `kernels/cuda/hsn_window.cu`: cell
front, routes, hidden layer of the selected unit, output layer, cell back;
`hsn.kernels.window.cell_round`). It
equals the native round to 2e-5 per position and 1e-4 over 256 positions
(`tests/kernels/test_p2_10_window.py`), and it is slower: after four designs about 160 µs per
position against the native ops' 83 µs (now about 50 µs with the twin), 58 ms against 43 ms for the
sweep. What the measurements showed, in the order they corrected my assumptions:

- 32-way shared-memory bank conflicts (a tile laid out with a stride of 32 words) cost one loop
  18 µs; transposed tiles removed that;
- reads of global memory are served at about the DRAM bandwidth even for data that fits the cache:
  a block that re-reads a weight tile every other block also needs pays for it again;
- Nsight Compute (`evidence/P2.10/speed/s3a-ncu-cell-round.csv`): 0.2 to 0.6 eligible warps per
  scheduler and 16 to 29 % of the warps active. The kernels wait on memory; the arithmetic is
  nowhere near the limit.

The native path's all-units product (eight times the arithmetic, on tensor cores) costs about
32 µs per position; the selected-unit kernels here need 73 µs (41 for the hidden layer, 29 for the
output layer, 3 for the routes; measured again in the afternoon). The op stays with its tests; it
is no longer what `hardware.window=fused` runs. The lesson went into the fused sweep above: keep
the matrix products on tensor cores and fuse what lies between them.

**Taking the scorer's product per feature block in the sweep** (one small matmul per row for
`[u; eps]`, one per cell for `[h; age; scalars]` with `V`, no assembled features): 84 µs per
position against 81, because it needs four more kernels than the assembled product. Reverted.

**Six to seven times.** At 44.6 ms the step is 21 ms of sweep, 15 ms of replay and backward and
about 7 ms of trainer work. In the sweep the six matrix products are 11.5 ms, of which the
all-units first layer is 4.5 ms: only a tensor-core kernel that runs each pair's own unit would
remove it. Reaching 15 ms needs that, a compiled replay (its dynamic per-unit shapes stand in
the way) and a fused optimiser. In a run with a soft warm-up the warm-up's own steps (0.216 s
each, every cell and unit with autograd) are a third of the run and are bound by arithmetic.

What the two baselines of the comparison do instead, measured the same night on the delayed task
at B32 and T256 through the repository's trainer (one `torch.profiler` pass each; four plain
steps):

| Model | Parameters | Step time | GPU kernels per step | Device time per step |
| --- | ---: | ---: | ---: | ---: |
| `gru_2m` (eager, bf16; cuDNN runs the recurrence) | 1.98 M | 15.9 ms | 2,318 | 13.2 ms |
| `transformer_2m` (eager, bf16; no loop over positions) | 1.93 M | 13.9 ms | 404 | 12.5 ms |
| HSN, `hardware.window=fused` | 1.85 M | 44.7 ms | 7,133 | 40.9 ms |

All three keep the card busy for most of the step (83, 90 and 91 %). The GRU's kernels are as
small as the HSN's (5.7 µs on average in both) and it runs a third as many: about four per layer
and position, forward and backward together, where the fused sweep alone is 23.7 per position.
The Transformer runs 404 large ones (31 µs on average). The gap that is left is the number of
kernels, not their size. Neither baseline reaches 85 % on this task (`gru_2m` 76.7 to 79.3 %;
`transformer_2m` 32.5 and 35.9 % in two runs on the delayed task alone, at 54,000 and 27,000
steps), so this is a comparison of step time only.

## Using it

```
python scripts/build_window_kernels.py            # once per checkout: the window kernels
hardware.window=split                             # an override, like hardware.compile_mode
hardware.window=fused                             # the split window with the fused sweep (CUDA)
python scripts/bench_window.py time   --window fused|split|autograd
python scripts/bench_window.py phases --window fused|split
python scripts/bench_window.py grads              # split against autograd, fp32 and bf16
```

Without the built kernels (or off CUDA) the recurrence runs its PyTorch reference: correct, and
slower than the autograd window. Example spec: `configs/sweeps/p2_10_split_seeds.yaml`.

## Files

- `python/hsn/train/split.py`: the window, the recurrence node and its references, the scope.
- `python/hsn/train/compile.py`: `_tape_block`, `_fused_tape_block`, `run_tape`,
  `CompiledWindow.tape`, `SweepTwin`, `sweep_merge_error`.
- `python/hsn/model/router.py`: `Selection.threshold`, `Router.finish_recorded`.
- `python/hsn/model/cell.py`: `PDGCell.replay_sparse`, `PDGCell.sparse_products`;
  `python/hsn/model/units.py`: `UnitBank.forward_grouped`, `forward_static_chosen`,
  `permute_rows`.
- `python/hsn/kernels/window.py`, `python/hsn/kernels/window_ext.cpp`,
  `kernels/cuda/hsn_window.cu`, `kernels/include/hsn_window_launch.h`,
  `scripts/build_window_kernels.py`: the window kernels and their launcher, independent of the
  four P2.9 ops (their fatbin, extension, ABI and tests are untouched; no Rust launcher).
- `python/hsn/train/lazy_adam.py`, `grad_norm.py`, `update_limit.py`, `tbptt.py`: the trainer.
- Tests: `tests/test_p2_10_split_window.py`, `tests/kernels/test_p2_10_window.py`,
  `tests/test_p2_10_trainer_speed.py`.
