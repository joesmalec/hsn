# Compiled baselines: `hardware.baseline_compile`

14 November 2026. Branch `claude/baseline-compile`, from `6bbf53d`. Written and tested on the
CPU only (torch 2.15.0+cu132, the card hidden): nothing here has run on a GPU, so this document
holds no step time. How to take one is at the end.

## What the key does

The HSN trains through a compiled window (`hardware.compile_mode`, P2.8); the baselines of the
generic trainer ran eager, so their step times (GRU 16.7 ms, Transformer 16.2 ms at B32, T256
on the RTX 3060) stood against the HSN's compiled ones. `hardware.baseline_compile` compiles a
baseline's windows with `torch.compile`, so that the comparison can be made like for like.

| Value | What runs |
| --- | --- |
| `none` (default) | The baseline eager, as before the key. |
| `default` | Compiled (inductor), no CUDA graphs. |
| `reduce-overhead` | Compiled, replayed as CUDA graphs on CUDA. The HSN's mode on the RTX 3060. |
| `max-autotune` | Compiled with inductor's autotuning, replayed as CUDA graphs on CUDA. |
| `max-autotune-no-cudagraphs` | Compiled with autotuning, no CUDA graphs. |

The value is passed to `torch.compile` as its mode. The key is invisible at its default: it is
left out of the configuration hash while it is `none` (`HASH_OMIT_AT_DEFAULT`) and out of the
memory key, so every existing configuration keeps its hash and every arm its id and its stored
memory and step-time estimates. A configuration that sets it is another configuration.

`hardware.compile_mode` keeps its meaning: it is the HSN's window, and the generic trainer
still ignores it. The sweep spec still refuses `hardware.compile_mode != none` on a baseline
arm; a compiled baseline arm pins `hardware.compile_mode=none` and sets
`hardware.baseline_compile`. An HSN configuration that sets the key is refused at load. HSN-T
without its memory (`model_name: hsn_t` with `model.memory: false`, family `hsn_t_nomem`,
`docs/hsn_t_no_memory.md`) is a stateless model on the generic trainer and takes the key;
HSN-T with its memory is an HSN and does not.

## What is compiled, and what is not

The compiled functions call the same Python the eager forward runs, with `fullgraph` (a graph
break raises) and static shapes. Nothing of a baseline's architecture changes.

| Family | Compiled | Stays eager |
| --- | --- | --- |
| `transformer`, `reader`, `hsn_t_nomem` | The whole forward of a window, as one graph, and its backward. | |
| `gru` | The embeddings before the recurrence and the read-out after it (LayerNorm and head), as two graphs, and their backward. | The recurrence, `torch.nn.GRU`: cuDNN on CUDA, called between the two graphs as before. |
| `selective_ssm` | Outside the scope: refused at load. | |

For every family the loss, the gradient clipping, the guards and the optimiser stay eager, as
under the HSN's compiled window. The harness's edit probes and trained-state pass run on the
eager model (the record's `probe_execution` says `eager`).

**The GRU.** `torch.compile` does not take `torch.nn.GRU`: dynamo breaks the graph at RNN
modules, and tracing through one (`torch._dynamo.config.allow_rnn`) replaces cuDNN's kernel by
an unrolled loop of GRU cells, 512 of them for two layers over 256 positions. cuDNN's
recurrence already is a vendor kernel, so it is left as PyTorch calls it. What the key compiles
of the GRU is a small part of its step: two embedding lookups and a sum before the recurrence,
a LayerNorm and a linear layer after it. Expect its step time to move little.

**The selective state-space model.** Its scan is a Python loop that writes every position's
state into one window-sized tensor in place (`hs[:, t].addcmul_(...)`), with a hand-written
backward that does the same. The whole window compiles and gives the eager values, but a
compiled graph has no in-place writes and the loop's cost then grows with the square of the
window: on the CPU (B2, 512 channels of 16 states) the scan's forward loop takes 4, 73 and
339 ms compiled at 32, 64 and 128 positions against 0.4, 1.0 and 3.2 ms eager, and one
training window at the default size (4 blocks, T256) 3.0 to 10.5 s against 0.5 s, after 4 to 6
minutes of compiling. That would be a "compiled Mamba" slower than the eager one, so the
family is outside the scope.

## How it runs

`hsn.train.baseline_compile.CompiledBaseline` stands in for the model in the trainer's
training windows and in its evaluation (`Trainer.compiled_baseline`).

- **One compiled function per variant**: batch shape, grad mode, training mode and autocast
  state, each with a code object of its own. The evaluation (`no_grad`, evaluation mode, the
  evaluation batch size) is a compiled function of its own, built at the first evaluation, and
  nothing recompiles once warm. A compiled graph is also specialised to its inputs' memory
  layout: the window's inputs are made contiguous, and the GRU's read-out has a function per
  layout of the recurrence's output (a transposed view after one cuDNN call, a contiguous
  tensor when a reset inside the window splits the recurrence).
- **Compile steps.** The first call of a function compiles; under CUDA graphs the second
  records. A step during which that happened is a compile step: its wall time goes to
  `compile_seconds`, not to `train_seconds` or `tokens_per_s`, as for the HSN.
- **CUDA graphs** (`reduce-overhead`, `max-autotune`, on CUDA): every window starts a new
  graph step (`torch.compiler.cudagraph_mark_step_begin`); a window's logits are graph memory,
  read before the next window; `train.grad_accum > 1` is refused, as for the HSN.

Refused when the trainer is built (`NotImplementedError`): dropout above 0 (a compiled random
draw differs from the eager one; every shipped baseline configuration has dropout 0) and
`train.grad_accum > 1` under CUDA graphs. On MPS a backend that cannot compile falls back to
the eager model with a logged notice, and the record then shows no compile.

## The run record

`History.measured` of a compiled run carries the HSN trainer's keys; an eager run keeps
`compiled_windows = 0` and has none of the other three.

| Key | Meaning |
| --- | --- |
| `compiled_windows` | Windows that ran through the compiled functions, evaluation windows included. |
| `compile_steps` | Training steps during which a function compiled or recorded. |
| `compile_seconds` | Wall time of those steps (kept out of `train_seconds` and `tokens_per_s`). |
| `window_compile_seconds` | Wall time of every window that compiled or recorded, evaluation windows included. |

The sweep harness's record keeps its `compile` block: `mode` and `compiled_window` are the
HSN's (`none` and `false` for a baseline), the four counts above are filled, and a compiled
baseline adds `compile.baseline`. `hsn run` writes the same report as
`metrics.json["baseline_compile"]`. An eager run has neither.

| Field of the report | Meaning |
| --- | --- |
| `mode`, `backend` | The value of the key; `inductor`. |
| `cuda_graphs` | Whether the mode replays the graphs as CUDA graphs on this device. |
| `cudagraph_skips` | Graphs PyTorch refused to replay as CUDA graphs since the trainer was built. Must be 0; PyTorch logs the reason of each. |
| `scope` | What of the family is compiled, in words. |
| `evaluation` | `compiled`. |
| `windows`, `functions`, `graphs` | Windows run, compiled functions built, graphs dynamo compiled. `graphs` equals `functions` unless something recompiled. |
| `compile_seconds` | As `window_compile_seconds` above. |

Where timing is concerned the harness treats a compiled baseline as a compiled HSN: compile
steps are excluded from the step-time guard, `walls.compile_s` is the compile time, and the
protocol's `gc_freeze` applies. `hsn run`'s profile pass runs the compiled windows too (the
first two windows of each pass are its `compile` segment).

## How to time it

1. Run the CUDA tests of this change first. They are skipped without a card and have never
   run: `pytest -m cuda tests/test_baseline_compile.py`. For each family they check that
   `reduce-overhead` (warm-up, recording, replay) and `default` equal the eager forward and
   backward in fp32, that no graph is refused as a CUDA graph, that a warm window compiles
   nothing, and that the trainer trains 40 steps with two evaluations under the profile's
   autocast.
2. Add the overrides to the eager timing's configuration:
   `hardware.compile_mode=none hardware.baseline_compile=reduce-overhead`. `reduce-overhead`
   is the mode the HSN's window runs in on the RTX 3060; `hardware.baseline_compile=default`
   separates what the compiled kernels give from what the CUDA graphs give.
3. Time steady-state steps only, as for the HSN: skip the compile steps (the first two
   training steps under CUDA graphs, and the step of the first evaluation). From a `History`,
   `train_seconds / (steps - compile_steps)` is the mean steady-state step.
4. Read the record before the number: `cuda_graphs` true, `cudagraph_skips` 0, `graphs` equal
   to `functions`.

In a sweep spec a compiled baseline arm is
`{name: gru_2m_compiled, family: gru, overrides: [hardware.compile_mode=none, hardware.baseline_compile=reduce-overhead]}`.

### HSN-T without its memory

The recipe is `configs/experiments/hsn_t_nomem.yaml`; the overrides of the comparison with
the baselines, and the key:

```
# reader width 64 (the recipe's size): 627,328 parameters
profile=rtx3060 hardware.compile_mode=none train.batch=32 train.micro_batch=32
hardware.baseline_compile=reduce-overhead

# reader width 224 (the GRU's size): 1,942,528 parameters
profile=rtx3060 hardware.compile_mode=none train.batch=32 train.micro_batch=32
model.reader_width=224 hardware.baseline_compile=reduce-overhead
```

Without `hardware.baseline_compile` the same lines are the eager runs. The window is one
graph (the reader's attention, `S` and `D` over all positions at once), compiled once for
training and once for the evaluation. On the CPU with inductor it equals the eager forward
and backward to 2.6e-7 of the largest logit and 2.9e-7 of the largest gradient at the test's
size, and to 8.3e-7 and 3.3e-7 at width 224 (windows of 16 positions, batch 3).
