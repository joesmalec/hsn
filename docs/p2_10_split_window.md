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

(`scripts/bench_window.py time`, 60 warm-up and 300 timed steps per process, statistics, evaluation
and checkpoints off; reports under `evidence/P2.10/speed/`.) The stretch target of this work was a
factor of six to seven (the step time of `gru_2m`, about 17 ms). It was not reached; the section
"What did not work" says why and what would be needed.

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
own. This took 11 of the sweep's 43 ms.

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
  `[P, 4d]` gradient tensor per present unit in the backward (eight 33 MB tensors per window).

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

| Part | ms |
| --- | ---: |
| Window prologue (embeddings, `E`, scorer scalars) | 1.5 |
| Forward sweep (52 kernels per position) | 32.0 |
| Replay, forward | 6.7 |
| Backward (replay and recurrence) | 9.1 |
| Optimiser step, clip, update cap, weight average, the rest | about 7 |

## Scope

`hsn.train.split.scope_error` names what a configuration lacks. The split window needs: routed
`topk` with the `ste` gate from step 0, all four feedback paths cut, no exploration
(`train.epsilon_*`), no counterfactual probes (`train.lambda_u`), no state-gradient clip, no
disabled cells, one round, bf16 or fp32. Dense-fallback rows (`train.p_dense`), the balance loss
(`train.lambda_b`), the cost term and a temperature schedule are covered: the selection's
gradient path is ordinary PyTorch in the replay, and `info` carries what the autograd window's
does (`unit_logits` with its graph, `expected_cost`, `unit_idx`, `valid`, `cell_idx`, the rates).
A window whose dense-fallback replay would not fit memory (more than about 4 GB) runs the
autograd window from the same draw. Evaluation, probes and diagnostics run the windows they ran
before.

## How equal it is

| Comparison | Result |
| --- | --- |
| CPU float64, split against `HSN.forward` (interior resets, carried start state, padded slots, refresh probes, cost and balance terms, a window without loss positions, dense-fallback rows) | every parameter gradient equal at rtol 1e-9, the same set of absent gradients |
| CUDA fp32, compiled, split against the autograd window, 300 training steps in | identical routing decisions; gradients equal to 1.2e-7 relative |
| CUDA bf16, the same, each against the fp32 autograd reference | autograd bf16: 2.7 % relative, cosine 0.99964. Split bf16: 2.8 %, cosine 0.99963 |
| CUDA bf16, one tape, the replay and backward under bf16 against the same replay in fp32 | 0.33 % relative, cosine 0.999996 (the split backward's own bf16 error, with decisions and state values held equal) |
| Sweep twin against the model's own sweep | fp32: bit-identical tape. bf16: equal to rounding (below) |

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

## Accuracy

ACCURACY_TABLE

## What did not work

**A fused cell round** (`hardware.window=fused`, `kernels/cuda/hsn_window.cu`: cell front, routes,
hidden layer of the selected unit, output layer, cell back; `hsn.kernels.window.cell_round`). It
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
32 µs per position; the selected-unit kernels here need 72 µs. The fused option stays as an opt-in
experiment with its tests; it is not a speed path.

**Six to seven times.** Even a sweep at a tenth of its present cost would leave about 25 ms of
replay, backward and trainer work. Reaching 15 ms needs all three smaller by a factor of three:
a fused sweep that is actually fast, a compiled replay (its dynamic per-unit shapes stand in the
way), and a fused optimiser.

## Using it

```
python scripts/build_window_kernels.py            # once per checkout: the recurrence kernels
hardware.window=split                             # an override, like hardware.compile_mode
python scripts/bench_window.py time   --window split|autograd
python scripts/bench_window.py phases --window split
python scripts/bench_window.py grads              # split against autograd, fp32 and bf16
```

Without the built kernels (or off CUDA) the recurrence runs its PyTorch reference: correct, and
slower than the autograd window. Example spec: `configs/sweeps/p2_10_split_seeds.yaml`.

## Files

- `python/hsn/train/split.py`: the window, the recurrence node and its references, the scope.
- `python/hsn/train/compile.py`: `_tape_block`, `run_tape`, `CompiledWindow.tape`, `SweepTwin`.
- `python/hsn/model/router.py`: `Selection.threshold`, `Router.finish_recorded`.
- `python/hsn/model/cell.py`: `PDGCell.replay_sparse`; `python/hsn/model/units.py`:
  `UnitBank.forward_grouped`.
- `python/hsn/kernels/window.py`, `python/hsn/kernels/window_ext.cpp`,
  `kernels/cuda/hsn_window.cu`, `kernels/include/hsn_window_launch.h`,
  `scripts/build_window_kernels.py`: the window kernels and their launcher, independent of the
  four P2.9 ops (their fatbin, extension, ABI and tests are untouched; no Rust launcher).
- `python/hsn/train/lazy_adam.py`, `grad_norm.py`, `update_limit.py`, `tbptt.py`: the trainer.
- Tests: `tests/test_p2_10_split_window.py`, `tests/kernels/test_p2_10_window.py`,
  `tests/test_p2_10_trainer_speed.py`.
