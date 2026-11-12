# P2.10b: the fused window's step, a leaner trainer and a folded sweep

12 November 2026. Branch `claude/p2.11-fused-speed`, from `981a56d` (in the repository this work
is P2.10b, the follow-up of P2.10; the plan's P2.11 is the Metal kernels). Measured on the
RTX 3060 (`profile:rtx3060`), the card to itself, torch 2.15.0+cu132, the standard recipe
`configs/experiments/hsn_recipe_a6.yaml` (routed top-K, B32, T256, bf16) with
`hardware.compile_mode=reduce-overhead`, with the commands of `docs/p2_10_split_window.md`.

## What this is

Two changes to a training step of `hardware.window=fused`. Neither changes a value the model
computes; no configuration key was added and no hash moves.

1. **A leaner trainer.** The work after `backward()` (optimiser step, clip, update caps, weight
   average) runs as a handful of kernel launches over lists of tensors where it issued about
   two hundred small kernels more, and what a step records about its windows is read from the
   card once, after the backward pass is under way, where it was read a dozen times before it.
2. **Fewer kernels per position in the forward sweep.** Seven of the thirteen small compiled
   kernels around the sweep's two glue kernels only moved or cast what those kernels or the
   next matrix product read. The glue kernels now read and write those operands themselves.

| | Step time, median of 300 | Against the base |
| --- | ---: | ---: |
| Base (`981a56d`) | 44.7 ms | |
| + leaner trainer (and the twin's refresh, below) | 41.7 ms | −2.9 ms |
| + folded sweep: this branch | 40.5 ms | −4.2 ms (−9.3 %) |

(`scripts/bench_window.py time --window fused --warm 60 --steps 300`, one process per number, two
of each, alternating, in one session: base 44.60 and 44.73, trainer part only 41.79 and 41.70,
this branch 40.46 and 40.54. The base measured 44.4 ms in the morning and 44.8 ms in the first
session of this work.) The trainer part reaches every window: the split window goes from 55.0
to 52.2 ms and the autograd window from 97.8 (the morning's) to 94.8 ms.

The targets this work was given were 3 ms for the trainer and 4 ms for the sweep. The trainer
part gives 2.9 ms (1.9 ms of it the kernels: with `HSN_TRAINER_KERNELS=0` the step is 42.4 ms).
The sweep part gives 1.3 ms, not 4: the small kernels that could go cost 0.7 µs each in a CUDA
graph, not the 1.9 µs their share of the profile suggested, and the two that are real
arithmetic cost as much inside a glue kernel as outside it ("What did not pay").

## The trainer after `backward()`

**Trainer kernels** (`kernels/cuda/hsn_trainer.cu`, `python/hsn/kernels/trainer.py`; they live
in the window kernels' fatbin and use its launcher, so `scripts/build_window_kernels.py` builds
them). Each walks a list of tensors as one index space, one element per thread, in one launch:

| Kernel | Replaces |
| --- | --- |
| `hsn_tr_adamw` | `LazyAdamW._dense_steps`: eight multi-tensor ops per group of ordinary tensors |
| `hsn_tr_adamw_units` | `LazyAdamW._unit_step`: about twenty ops and several host copies per stacked unit tensor |
| `hsn_tr_scale64` | the float64 clip's scaling: two kernels per gradient, when the clip binds |
| `hsn_tr_copy`, `hsn_tr_limit_squares`, `hsn_tr_limit_project` | the update caps' capture and elementwise passes: eleven per limited tensor |
| `hsn_tr_lerp` | the weight average: one lerp per parameter |

They are exact, not close. Every PyTorch operation rounds once, so every step of a kernel is one
explicitly rounded operation (`__fmul_rn`, `__fadd_rn`, ...: the compiler may not contract
them), and a fused multiply-add stands exactly where PyTorch's kernel has one: `lerp` (`self +
w (end - self)`, contracted; the other branch for `w >= 0.5`), `addcmul` and `addcdiv` (an
explicit `fma` in `ATen/native/cuda/DeviceAddCmulCdiv.cuh`). One thing was not what the code
suggests: a tensor list divided by a scalar is multiplied by the reciprocal, taken in double and
then rounded (`_foreach_div_` against a product with `f32(1.0 / s)`: equal on every one of 200
scalars for which the two candidates differ).

What stays as it was are the reductions: the float64 gradient norm and the caps' three norms are
one `torch` sum per tensor, because the order in which that kernel adds is what defines the
value. They are 100 of the 295 kernels the trainer still issues per step.

The PyTorch paths remain, as the reference and for everything the kernels do not take (the CPU,
other dtypes, a checkout without the build, `HSN_TRAINER_KERNELS=0`): every wrapper returns
`False`, having touched nothing, and the caller runs what it ran before.

**Statistics after the backward pass** (`TBPTTTrainer._window_statistics`). Between the
forward and the backward pass the trainer read the active units, the terms, the total, four
model statistics, three rates, the accuracy count and the parameter sum: about a dozen waits for
the device, with the card idle until the backward pass was issued. They are now issued as two small
device tensors once the backward pass is on its way, and read after the gradient norm, which
waits for the backward pass anyway. The values are the same expressions on the same tensors;
the accuracy count takes the argmax of every position and masks afterwards, which is the count
of the masked lookups without their two waits. With a gradient hook or a profiler the read
stays where it was (a hook may read the snapshot's loss; a profiled step keeps its segments).

**Smaller things.** The route guard asked the device twice per step whether any unit is
scaled; it now asks when its scale tensor changed.

Per step, synchronised (`step_parts.py`, 300 steps; the forward and the backward are unchanged
code):

| Part | Base | Now |
| --- | ---: | ---: |
| Optimiser step | 1.70 ms | 0.51 ms |
| Update caps, after the step | 1.42 ms | 0.80 ms |
| Float64 clip | 0.78 ms | 0.71 ms |
| Weight average | 0.12 ms | 0.12 ms |
| Active units, parameter sum, guard counts, the reads | 0.59 ms and about 0.5 ms of reads | 0.85 ms, after the backward |
| Kernels the trainer issues per step | 499 | 295 |

## The folded sweep

Per position the sweep of P2.10 runs six matrix products, its two glue kernels and thirteen
small compiled kernels. Seven of the thirteen are bookkeeping around the glue kernels:

- three copies and a cast assemble `[z; q; ctx]` for the first product. The block now holds
  that input for all its positions (`x [n, B, 192]` in the products' dtype) and the
  end-of-round kernel, which computes `q`, writes the next position's row;
- two casts bring `W_out`'s output and the gate's pre-activation to the state dtype for the
  end-of-round kernel. It now reads the products as they are (the gate as a strided view of the
  first layer's wider output);
- one cast brings the logits to fp32 for the selection kernel. It now reads them in the
  scorer's product.

`hsn.train.compile._folded_tape_block` is `_fused_tape_block` with these folds
(`hsn.kernels.window.update_fold`, `select_fold`; `hsn_win_cell_update` and `hsn_win_select`
gained the operand descriptions, their plain and tape forms are unchanged). It runs when the
twin's matrix products merge (`sweep_merge_error`), which is when `hardware.window=fused` ran
the merged sweep before; `CompiledWindow.tape(..., fold=0)` is the sweep of P2.10, kept for
comparisons. A block of 16 positions is 239 kernel calls where it was 358 (a step's sweep
and prologue: 4,223 kernels where they were 6,047), and the sweep takes 19.2 ms where it took
20.5 (`bench_window.py phases`, synchronised; 20.7 at the base, before the twin's refresh
below).

One thing to know about buffers a compiled block makes for a kernel: the compiler lays a tensor
out as it likes. Given position-major sources (`z_b.transpose(0, 1)`) it laid the input buffer
out batch-major and then copied it into the kernel's layout and back around every kernel call:
two more kernels per position, silently. A buffer made with `new_empty` and first filled
through a row of the usual layout keeps the layout the kernel indexes; the kernel therefore
writes the whole next row (`z` and the context too), and the block assembles only its first.

**The sweep twin's refresh** (`SweepTwin`). With the merged layouts the sweep reads two cast
weights of its own (the unit bank's second bias and `W_out`); the other sixteen, `W1` and `W2`
among them, were cast once per window for nobody. They now hold NaN (a read cannot pass for a
value) and are left alone, and the twin no longer walks the model's module tree by name every
window. This is host time before the sweep starts: the sweep's launch goes from 3.6 to 3.4 ms.

## What did not pay

**The selected cells' messages in the selection kernel** (the kernel has the cells; the
compiled kernel that gathers `V(h)` from their sources would go). Slower by 0.85 ms per step
(41.2 ms against 40.3): with the messages arriving in a buffer a kernel wrote, the compiler no
longer fused the round's input assembly and norm into one kernel but into four, so a block was
292 calls instead of 244.

**The messages and the round's input norm in the selection kernel** (both compiled kernels
would go: 13 calls per position instead of 15). Bit for bit the compiled kernels' values, which
took reading their PTX: the norm's sum of 256 squares is `fma(e0, e0, e1 e1) + fma(e2, e2, e3
e3)` per thread of four elements, a butterfly of shuffles over a warp and the sum of two warps;
a message is `fma(v0, w0, v1 w1)` and then one `fma` per further source. Two designs: the reads
after the decision in two more stages (40.55, 40.26, 40.29 ms), and everything the decision
selects from preloaded into shared memory with the logits, one warp per slot, no further stage
(40.07, 40.24, 40.24 ms), against 40.42, 40.14, 40.41, 40.10, 40.28 ms without. No gain in
either; removed. What the two kernels cost outside the glue kernel (about 2 µs together per
position) is what their work costs inside it.

The lessons, next to those of P2.10: a small kernel in a CUDA graph costs 0.7 to 0.8 µs (seven
gone: 5 µs per position), the empty-kernel cost and no more; a kernel with real reads costs 1
to 1.5 µs and a stage of reads in a glue kernel costs the same; and changing what a compiled
graph feeds a kernel changes how the compiler fuses the kernels around it, so count the calls
in the generated code (`build/inductor`) before timing.

## How equal it is

| Comparison | Result |
| --- | --- |
| Each trainer kernel against its PyTorch path on the card (40 steps of the optimiser with gradient scales from 1e-9 to 30, masks, tensors without a gradient, zeros of both signs, subnormals; the caps' untouched and projected branches and growth ceilings; 120 steps of the average through its warm-up) | every weight, moment, step count, statistic and average bit for bit |
| The whole sequence (clip, capture, step, caps, average) on the recipe's model and groups, 60 steps of random gradients and unit masks, kernels on against off | bit for bit |
| The trainer on the CPU, this branch against `981a56d`: 8 steps of the recipe and of eight variants (gradient accumulation, soft warm-up, statistics every other step, dense routing, spike guard with a binding clip, native clip, REINFORCE, probes) | every step record, the weights, the optimiser state, the average and the caps' statistics identical |
| The window statistics against the reads they replace (routed, soft gate, dense-fallback rows, dense routing, ties in the logits) | identical |
| Folded sweep against the sweep of P2.10 at the recipe's shapes: bf16 and fp32, batch 6, 8 and 32, blocks of 16 and of 4 positions, three carried windows with resets inside blocks and refresh probes, each fold alone and both, again after the weights moved | the tape (decisions, thresholds, pre-round states, written values, summaries, ages) bit for bit |
| The same on a small model (`d = 8`) in fp32 | decisions, ages and positions identical; values equal to one unit in the last place (6e-8) |
| The kernel's fp32 to bf16 rounding against `torch`'s, every upper half of an fp32 with eight lower halves on, next to and far from the ties | identical, NaN patterns included |

Two things could not be shown and are said here plainly.

*A whole training run on the card is not bit-reproducible, with or without this work.* Two runs
of `981a56d` (fused, bf16, same seed) differ in the gradient norm of step 1 by 2e-12 relative: a
float64 norm of fp32 gradients, so some gradient elements differ in their last place (probably
the replay's backward, which accumulates with atomics; not isolated). Over 60 steps two base
runs drift apart by up to 9e-11 in loss and gradient norm. So "the same run, bit for bit" has no
meaning on the card as it is normally run. With deterministic algorithms
(`train.deterministic=true`) it has, and there the comparison was made: two runs of `981a56d`
and one of this branch (fused, bf16, B32, 40 steps) give the same step records, weights,
optimiser state, weight average and cap statistics, bit for bit.

*At another width the two sweeps agree to the last place, not in it.* On a small model
(`d = 8`, fp32) the folded and the unfolded sweep take the same decisions and differ by one
unit in the last place in some values (6e-8); which values changed from one process to the
next (one fold was identical to the unfolded sweep in one process and not in another). The
folded kernels' inputs and outputs are the same there, so the difference is in a compiled
kernel that both sweeps contain. My reading, from the generated code and not isolated: the
round's norm, whose reduction the compiler lays out by timing candidates when a row has fewer
than 256 values, for every compiled block anew. At the recipe's width that reduction has one
layout, and every comparison at that width was identical.

## Where a step's time goes now

Unsynchronised, as in training (wall time inside each call of a step, mean of 300 steps; total
40.5 ms; `tools/chain.py` in the measurement folder named under "Files"):

| Part | Base | Now |
| --- | ---: | ---: |
| Host waits for the sweep (inside the replay) | 22.8 ms | 21.7 ms |
| Backward, issued | 6.0 ms | 5.6 ms |
| Gradient norm: waits for the backward | 3.4 ms | 3.1 ms |
| Sweep launch (twin refresh, sixteen blocks) | 3.6 ms | 3.4 ms |
| Optimiser step | 1.68 ms | 0.42 ms |
| Update caps | 1.40 ms | 0.82 ms |
| Window prologue | 1.40 ms | 1.38 ms |
| Loader | 0.92 ms | 0.84 ms |
| Divergence snapshot | 0.54 ms | 0.50 ms |
| Loss terms | 0.50 ms | 0.86 ms (now holds the first wait after the forward) |
| Statistics between forward and backward | 1.3 ms | 0 (issued after the backward: 0.80 ms, while the card works) |

The step is three stretches. From the gradient norm to the first sweep block the card waits for
the host (optimiser, caps, average, loader, snapshot, prologue, twin refresh: about 5 ms now,
about 7.5 before). Then the sweep: the card works through 19.2 ms of recorded graphs while the
host has nothing to do for about 17 ms. Then the replay and the backward, bound by arithmetic.

## What would be next

In the order of what they are worth, with what was measured here:

- **The loader during the sweep** (0.9 ms). `next(loader)` is pure host work in the stretch
  where the card waits; the host idles 17 ms during the sweep. Fetching the next step's windows
  there needs the loader's state before the fetch kept for the snapshot and the checkpoint.
- **`W_out` into the unit bank's second layer** (about 0.8 ms: a matrix product and a gather per
  position). Not bit-identical: one bf16 rounding less between the two maps, so it needs
  accuracy runs like the merged products of P2.10 did.
- **The window prologue** (1.4 ms of host-bound model code with autograd: embeddings, `E`,
  scorer scalars, address features).
- **The divergence snapshot** (0.5 ms per step: RNG states, the persistent buffers through a
  `state_dict()` walk).
- **The caps' third norm** (0.25 ms): it only feeds statistics and could be issued with the
  next step's window statistics; the caps' counters would then be read through a flush.
- **The optimiser's host side** (0.2 ms of Python per step around four launches).

The sweep's small kernels are done: six are left per position (two activations with their
gathers, the scorer's features, the messages, the round's norm, one gather), each real work.
What is left in the sweep is the six matrix products (45 of about 70 µs per position) and the
per-kernel cost of a CUDA graph.

## Using it

```
python scripts/build_window_kernels.py     # once per checkout: window and trainer kernels
hardware.window=fused                      # as before; the folds and the trainer kernels need no key
HSN_TRAINER_KERNELS=0                      # the trainer's PyTorch passes (the reference)
python scripts/bench_window.py time --window fused|split|autograd
python scripts/bench_window.py phases --window fused
```

Without the build the trainer runs its PyTorch passes, as before this work.

## Files

- `kernels/cuda/hsn_trainer.cu`, `kernels/include/hsn_trainer_launch.h`,
  `python/hsn/kernels/trainer.py`, `python/hsn/kernels/window_ext.cpp` (`launch_table`,
  `launch_views`): the trainer kernels and their launcher.
- `python/hsn/train/lazy_adam.py`, `grad_norm.py`, `update_limit.py`, `weight_ema.py`,
  `route_guard.py`: the passes through the kernels; `tbptt.py`, `python/hsn/losses.py`
  (`terms_total`): the statistics after the backward pass.
- `kernels/cuda/hsn_window.cu`, `python/hsn/kernels/window.py` (`select_fold`, `update_fold`),
  `python/hsn/model/cell.py` (`sparse_products_raw`), `python/hsn/train/compile.py`
  (`_folded_tape_block`, `SWEEP_FOLD`, `SweepTwin`, `MERGED_READS`), `split.py`: the folded
  sweep and the twin's refresh.
- Tests: `tests/test_p2_10b_trainer_kernels.py`, `tests/test_p2_10b_folded_sweep.py`.
- Measurements (on the development machine, outside the repository):
  `/home/joe/hsn-experiments/p2.11-speed/` holds the scripts (`step_parts.py`,
  `step_kernels.py`, `tools/chain.py`, `tools/fold_check.py`, `tools/history.py`) and their
  outputs (`base/`: the base in the morning; `w1/`, `w1.log`: the first session, with the
  folds that did not pay; `w2/`, `w2.log`: the final numbers and the suites).
