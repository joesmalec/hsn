# HSN-T without its memory (`model.memory: false`)

Status: **an opt-in switch of HSN-T, implemented on 14 Nov 2026; first text and arithmetic
runs on 14 and 15 Nov ("Text: the gap to the baselines" below); not adopted.** Asked for by the
user on 14 Nov 2026: "I would like the memory option opt in and like
to see if we can beat gru (our original goal) on a compiled kernel", then "I'm looking to beat
gru on speed without memory, I know it can't be done with memory."

- `model.memory` (bool, default `true`), HSN-T only (`model_name: hsn_t`); every other
  `model_name` refuses `false`. Left out of the configuration hash at its default, so no
  existing configuration, hash or arm id changed.
- With `true` nothing changes: the class, the trainer, the parameters and every value are
  HSN-T's (`docs/hsn_t.md`).
- With `false` the model is **not an HSN and not the HSN-T that was run**: it has no cell, no
  state and no recall beyond the window. It is the path of HSN-T that does not pass through
  the memory, built so that its speed can be compared with the GRU baseline.

## What it computes

For a window `x_{1..T}` with context symbols `ctx_{1..T}`, all positions at once:

```
z      = E(emb(x))                    [B, T, d]
c      = R(x, reset)                  [B, T, w]     the reader of HSN-T, unchanged
u      = S([z ; 0 ; ctx ; c])         0 where HSN-T reads q_{t−1}
logits = D([0 ; u ; 0 ; c])           0 where HSN-T reads q_t and r_t
```

- **Built:** the byte and context embeddings, `E`, the reader, `S`, `D`. They are HSN-T's
  modules under HSN-T's names (`byte_emb`, `ctx_emb`, `encoder`, `reader`, `shared`,
  `decoder`), at HSN-T's widths.
- **Not built:** the cells, the router and its scorer, the unit bank, the graph, the
  predictor `P`, the halting head, the summary `q`, the selected-cell readout `r`. None counts
  as a parameter and none runs.
- **No loop over positions and no state.** Nothing is carried from one window to the next. A
  reset column starts a document for the reader, which reads nothing before it. `logits_t`
  depends on the document's bytes up to `t` inside the window (the last `1 + L (n − 1)` of them
  with `model.reader_span = n`) and on `ctx_t`.
- **The zero columns.** `S` keeps its `2d + context_dim + w` inputs and `D` its `3d + w`; the
  `d` columns of `q` in each and the `d` columns of `r` in `D` read zeros. This is what makes
  the model exactly HSN-T with every read of the state removed, and what lets HSN-T's weights
  load (below). It costs arithmetic that produces nothing. Counted as multiply-accumulates per
  position (a count, not a measurement):

  | Sizes | of `S` | of `D` | of the whole model (window 256) |
  | --- | --- | --- | --- |
  | Default (`w` 64, 2 blocks) | 20 % | 25 % | 15 % |
  | Reader widened to the GRU's size (below) | 13 to 15 % | 19 to 21 % | about 5 % |

  The request for this build put it at "a few percent"; that holds for the widened sizes and
  not for the default ones. Leaving the columns out of the arithmetic (the parameters keeping
  their shape) or out of the model would be a later, separate option; it is not built.

## Loading a trained HSN-T

`HSNTNoMemory.load_memory_on_state(state_dict)` takes the state dict of an HSN-T with its
memory (or a checkpoint's `model`) and loads the tensors of the six modules; the memory's
tensors (`predictor.*`, `cell.*`, `router.*`, `halting.*`) are left out, and a state dict that
lacks one of the six is refused. The logits are then the logits of that HSN-T's own window
with `S` reading zeros for `q_{t−1}` and `D` reading zeros for `q_t` and `r_t` (tested in
float64 to 1e-12, with the HSN-T running its cells as usual). `train.init_from` is not
extended: it still needs a checkpoint with exactly the model's keys.

A fresh model of seed `s` starts with HSN-T's draws for that seed where they do not depend on
the memory: the embeddings, `E`, the reader, and the input columns of `r` and `c`. The other
weights of `S` and `D` are other draws (HSN-T draws the predictor, the cell and the router
before them).

## Training

The model is stateless, so the generic trainer trains it
(`hsn.train.loop.Trainer`, as the GRU, Transformer and reader-alone models): `make_trainer`
gives the HSN's window trainer to an `HSN` instance and the generic one to everything else,
and `HSNTNoMemory` is a class of its own, not an `HSN`. The objective is the next-byte
cross-entropy alone.

Recipe: `configs/experiments/hsn_t_nomem.yaml`. Its training settings are the Transformer
baseline's (`configs/baselines/transformer_2m.yaml`: AdamW, 1e-3, cosine, warm-up 100), so
that the two differ in the model only.

In an experiment spec the family is **`hsn_t_nomem`**, a family of its own
(`hsn.experiments.spec.family_of`: `model_name: hsn_t` with `model.memory: false`). The
harness treats it as it treats a baseline: variant `none`, no `hardware.compile_mode`, the
verdict rules scoped to `hsn` or `hsn_t` do not apply, and reports keep it apart from the
HSN-T arms. An arm that names `family: hsn_t` with the memory off is refused, and the
reverse. The family of every existing arm is unchanged.

## What is refused, and what is not read

Refused at load (the training settings in one message that names each of them):

| Setting | Why |
| --- | --- |
| `model.memory: false` under any `model_name` but `hsn_t`, or without `model.reader` | the switch is HSN-T's |
| `model.selected_readout: false` (and so `model.routing: dense`, which A5 refuses) | HSN-T's rule; it keeps the `r` columns of `D` |
| `model.fast_cells`, `model.fast_readout` (A8) | cells of the memory and a read of them |
| `model.decoder_shortcut_dropout > 0` | it makes `D` read the selected cells |
| `train.lambda_p > 0` (the schema default is 1: a memory-off configuration sets 0) | `L_pred` reads the summary through `P` |
| `train.lambda_r`, `lambda_d`, `lambda_e`, `lambda_c`, `lambda_b`, `lambda_g`, `lambda_u`, `lambda_ponder`, `lambda_w` `> 0`; `train.aux_enabled` | the objective has no such term |
| `train.route_guard.enabled` | the router's units |
| `train.feedback_detach`, `train.state_grad_clip`, `train.state_grad_ratio` (A6) | cuts and limits of the state's gradient |
| `train.gate_mode` other than the default, `train.soft_until_step`, `train.unit_credit`, `train.ste_scope` | the routed selection |
| `train.temperature_*`, `train.epsilon_*`, `train.p_dense` | the selection's schedules |
| `train.cell_lr_mult`, `train.cell_weight_decay`, `train.router_lr_mult` | parameter groups of the memory |
| `train.address_gate_hold_step` | the address gate's weights |
| `train.optimizer: lazy_adamw` | the unit bank's groups (left unset, the optimiser resolves to `adamw`) |
| `train.update_limit` and `train.gradient_growth` groups with a selector on `cell.*`, `router.*`, `predictor.*` or `halting.*` | parameters of the memory (any other selector that matches nothing is refused by the limiter itself, as before) |
| `hardware.window: split` or `fused` | the HSN's windows (HSN-T's rule) |

Not read, and accepted:

- The `model.*` keys that describe the memory (`n_cells`, `n_units`, `k_select`, which
  routed mode `routing` names, `key_address`, `address_*`, `write_gate`, `cell_norm`,
  `cand_norm`, `predictor_target` and the like). A memory-off configuration may carry the
  values of the HSN-T it is the memory-less form of; two memory-off configurations that
  differ only in such a key compute the same model under two hashes and two arm ids.
- `train.q_resync_every`, `train.probe_fraction`, `train.n_probe`, `train.probe_horizon`,
  `train.ema_tau`, `train.variance_floor_gamma`, `train.reinforce_baseline_decay`: secondary
  values of settings that are refused above, or bookkeeping of the summary.

Available as for any model of the generic trainer: the spike guard and the rollback,
`train.weight_ema`, `train.adaptive_lr`, `train.grad_accum`, `train.answer_weight`,
`train.readout_lr_mult` (the decoder's group), and update-limit or gradient-growth groups on
the model's own parameter names. The weight average, the spike guard, `readout_lr_mult` and
an update limit on `reader.*` run in a test. The rollback, `adaptive_lr`, a gradient-growth
group, `grad_accum` and `answer_weight` were each run for twelve steps on the CPU during the
build and are not in a test.

`hardware.compile_mode` compiles the HSN's training window only. Under the generic trainer it
is ignored with a warning, and a spec refuses it for this family. The compile path for the
generic trainer's models is a separate change (`hardware.baseline_compile`, branch
`claude/baseline-compile`); on the integration branch `claude/nomem-compile` the two are
merged and the key compiles this model's window as one graph (`docs/baseline_compile.md`).
What this build guarantees is that the
forward compiles as one graph: no data-dependent branch, shape or host read
(`torch.compile(model, backend="aot_eager", fullgraph=True)`, forward and backward equal to
eager, the second window of the same shape reusing the graph; tested). Outside the tests the
`inductor` backend compiled it the same way on the CPU.

## Size

| | Parameters |
| --- | --- |
| Byte and context embeddings | 17,408 |
| `E` (64 → 512 → 64) | 66,112 |
| Reader (width 64, 2 blocks, 4 heads) | 116,480 |
| `S` (256 → 512 → 64) | 164,416 |
| `D` (256 → 512 → 256) | 262,912 |
| **HSN-T without memory, default sizes** | **627,328** |
| HSN-T with its memory, as run | 2,037,801 |
| GRU baseline (`gru_2m`) | 1,977,856 |
| Transformer baseline (`transformer_2m`) | 1,930,624 |

`HSNTNoMemory.param_report()` gives one row per module and
`hsn.train.memory_plan.memory_off_param_count` is the closed form. With the other sizes at
their defaults, a reader of width `w` and `L` blocks gives

```
445,312 + 1,282 w + L (12 w² + 13 w)
```

**To compare quality with the GRU at a matched parameter count** the reader would be widened
(`model.reader_width`, `model.reader_layers`, `model.reader_heads`; the width must be a
multiple of the heads with an even width per head). Three sizes near the GRU's 1.98 M, none
chosen and none run:

| `reader_width` | `reader_layers` | `reader_heads` | Parameters |
| --- | --- | --- | --- |
| 224 | 2 | 4 | 1,942,528 |
| 184 | 3 | 4 | 1,907,192 |
| 160 | 4 | 4 | 1,887,552 |

The speed question and the quality question use different sizes: the default model is a
third of the GRU's size, and a widened one does about three times the default's
multiply-accumulates per position.

## Text: the gap to the baselines (15 Nov 2026)

TinyStories (`tinystories_v1`), 9,000 steps, batch 32, learning rate 1e-3 unless noted, bits
per byte on the 2,000 dev stories (lower is better), uncompiled unless noted:

| Model | Parameters | Seed 0 | Seed 1 |
| --- | --- | --- | --- |
| Transformer baseline (192 × 4) | 1,930,624 | 0.919 | 0.925 |
| GRU baseline (learning rate 2e-3, its best) | 1,977,856 | 0.952 | 0.954 |
| Reader alone, 192 × 4 | 1,878,400 | 0.967 | |
| Without memory, reader 160 × 4 | 1,887,552 | 0.995 (compiled 0.983) | |
| Without memory, reader 184 × 3 | 1,907,192 | 0.996 | |
| Without memory, reader 224 × 2 (the shape chosen for speed) | 1,942,528 | 1.015 (compiled 1.021) | 1.027 |
| The same at learning rate 5e-4 / 2e-3 | | 1.048 / 1.078 | |

A compiled and an eager run of one configuration differ by up to 0.012 bits and two seeds by
0.012, so differences under about 0.015 between single runs are not read.

**Where the gap is** (the step-9,000 checkpoints scored on the CPU in float32; the totals
reproduce the runs). Documents are aligned to windows, so the position in the window is the
number of bytes the model can see:

| Window positions | Share of the bytes | Transformer | Reader 192 × 4 | No memory 160 × 4 | No memory 224 × 2 |
| --- | --- | --- | --- | --- | --- |
| 0 to 7 | 3.6 % | 1.477 | 1.495 | 1.507 | 1.510 |
| 8 to 31 | 10.7 % | 0.932 | 0.931 | 0.940 | 0.949 |
| 32 to 63 | 13.8 % | 0.907 | 0.922 | 0.939 | 0.960 |
| 64 to 127 | 26.0 % | 0.913 | 0.954 | 0.982 | 1.005 |
| 128 to 255 | 45.9 % | 0.880 | 0.954 | 0.993 | 1.014 |
| All | 100 % | 0.919 | 0.967 | 0.995 | 1.015 |

Three readings, each with what it rests on:

1. **The reader does not use the bytes far back, and the baseline does** (0.047 of the 0.096
   bits between the chosen shape and the baseline). `docs/hsn_t.md`, "The reader's positions",
   has the measurements: the same blocks with the baseline's position table in place of the
   rotation are the baseline; the reader's loss stops falling after about 64 bytes of context
   and no block of it attends far. The memory-off models inherit it: their loss rises from
   positions 32 to 63 to the end of the window where the baseline's falls. Three opt-in keys of
   the reader test the cause (`model.reader_rotary_pairs`, `model.reader_rotary_dtype`,
   `model.reader_position`).
2. **At one parameter count the model has a smaller reader than the reader alone** (0.028
   bits from the reader alone at 192 × 4 to the memory-off model at 160 × 4; one run each).
   `E`, `S` and `D` hold 657,280 of the chosen shape's 1,942,528 parameters and the reader
   1,267,840; the reader alone at 192 × 4 has 1,828,992 in the reader. The trained model does
   use them: with `u` held at zero in `D`'s input the chosen shape's loss on 300 dev stories
   rises from 1.016 to 1.420, with `z` at zero in `S`'s input to 1.188, with `c` at zero in
   `S`'s input to 1.234. Whether the same parameters would do more inside the reader is not
   measured; the arms of `text_gap.yaml` run the reader alone at 224 × 2 and a shape with a
   smaller `model.hidden` and a wider reader (an existing key: `S` and `D` keep their inputs
   and their place, their hidden layer narrows).
3. **Two blocks against four** (0.020 to 0.038 bits between 160 × 4 and 224 × 2 over the
   eager, second-seed and compiled runs; about 0.02 from position 32 on and less before it).
   This is the speed choice and is not changed here.

**Not built, by the architecture rule.** Nothing that removes or bypasses `S` or `D`, changes
`u = S([z ; q ; ctx ; c])` or `D([q ; u ; r ; c])`, or leaves the zero columns out. If the arms
show that the cost of point 2 is `S` and `D` themselves and not their size, that is a finding
to report and an amendment to ask for, not a switch.

Shapes near the same size and arithmetic, for point 2 (counts, not measurements; the
multiply-accumulates count the reader's attention at a window of 256):

| `model.hidden` | Reader | Parameters | In the reader | Multiply-accumulates per position |
| --- | --- | --- | --- | --- |
| 512 (default) | 224 × 2 | 1,942,528 | 1,267,840 | 2.09 M |
| 256 | 248 × 2 | 1,905,056 | 1,546,528 | 2.07 M |
| 128 | 264 × 2 | 1,939,936 | 1,747,680 | 2.12 M |

## How to time it

The coordinating session times it alone on the card with its own script. The configuration
of the comparison with the figures given with the request (alone on the RTX 3060, batch 32,
window 256: HSN-T 169 ms per step, of which the reader about 6.5 ms; GRU 16.7 ms and
Transformer 16.2 ms, both uncompiled):

```
configs/experiments/hsn_t_nomem.yaml
  profile=rtx3060  hardware.compile_mode=none  train.batch=32  train.micro_batch=32
```

and for a GRU-sized model the same with, for example, `model.reader_width=224`. The compiled
figure adds `hardware.baseline_compile=reduce-overhead` (branch `claude/nomem-compile`;
`docs/baseline_compile.md`, "HSN-T without its memory").

## Where it is in the code

- `python/hsn/config.py`: the key, its rules (`memory_off_refusals`), the hash registry.
- `python/hsn/model/hsn_t.py`: `HSNTNoMemory`; `HSN` and `HSNT` refuse a memory-off
  configuration.
- `python/hsn/runner.py`: `model_name: hsn_t` builds `HSNT` or, with the memory off,
  `HSNTNoMemory`.
- `python/hsn/train/memory_plan.py`: `memory_off_param_count`.
- `python/hsn/experiments/spec.py`: the family `hsn_t_nomem` and `family_of`; the memory
  key of the experiment database leaves the key out at its default, so the arms from before it
  keep their stored step-time and memory estimates.
- `configs/experiments/hsn_t_nomem.yaml`: the recipe.

## Tests (`tests/test_hsn_t_no_memory.py`)

The key is opt-in and leaves the hashes and the experiment database's memory keys alone;
with its memory the model is built and trained as before; stateless, with no memory module;
logits read no later byte, nothing before a reset, and the context at its own position; a
trained HSN-T's weights load and give its logits with the memory's reads at zero; the
starting weights; the closed form of the parameter count and the three widened sizes; the
refusals; the experiment family; thirty generic-trainer steps lower the loss; the generic
trainer's own controllers run; the forward and backward compile as one graph and match eager.

## Limits

- On text every shape run so far is behind the GRU baseline (0.952 / 0.954 bits) and the
  Transformer baseline ("Text: the gap to the baselines"). The step times are the
  coordinating session's and are not recorded here.
- `hsn run` labels its results by `model_name`, so a memory-off run made with it appears as
  `hsn_t` in the result store; the configuration hash and `model.memory` tell the two apart.
  The experiment harness uses the family.
