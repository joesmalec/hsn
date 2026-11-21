# The generic trainer's host reads: `train.host_sync` and `train.fused_adamw` (opt-in, 17 Nov 2026)

## Why

The generic trainer (`hsn.train.loop.Trainer`: HSN-T without its memory and the GRU, Transformer,
Mamba and reader baselines) read five values on the host in every step:

* `logits[loss_mask]` and `targets[loss_mask]`: a boolean index needs the number of selected rows;
* `loss.detach().cpu()`;
* `float(clip_grad_norm_(...))`;
* `float(correct)`.

Each waits for the card. The profile of the memory-off 160 x 3 recipe (`~/hsn-handoff/prof_160x3/`,
17 Nov 2026) found the host waiting in the clip-norm and loss reads for most of the step and the
card idle for about 3.4 ms of an 18 ms step; a copy of the step without the reads gave 15.6 ms
against 19.3 ms with them. AdamW was the foreach implementation; the per-step divergence snapshot
took 0.36 ms of host time.

## The keys

**`train.host_sync: per_step | stats`**, default `per_step`.

* `per_step`: the trainer as it was, bit for bit (the code path is unchanged; the gradient norm is
  read with the same call).
* `stats`: a step's loss, accuracy and gradient norm stay on the device.
  * The loss is `masked_cross_entropy` (`hsn.train.loop`): the cross-entropy summed over the loss
    positions (`ignore_index` elsewhere) divided by the host's count of them, and the correct count
    from the argmax over every position and the mask. The same mean as
    `F.cross_entropy(logits[mask], targets[mask])` to rounding (the sums run in another order);
    the same gradients; nothing reaches a position off the mask.
  * The loss sum is accumulated on the device in float64 (as the host accumulated it).
  * The optimiser is the fused AdamW (`train.fused_adamw: true` is required), and a step whose
    gradient norm is not finite is skipped **on the device**: the trainer sets the optimiser's
    `found_inf` to `(not isfinite(norm))`, and the fused kernel then returns without touching the
    weights or the moments and takes its step count back. That is what P1.12's host guard does
    (it never calls the step). The scheduler advances, as for a skip.
  * Each step stacks `[loss sum, correct, grad norm]`, copies it into pinned host memory without
    blocking and records an event (`_HostReads.push`). The host then waits for the event of **the
    step before** (`HOST_SYNC_LAG = 1`), never for the step it just queued, so the card always has
    a step of work queued and is never drained; and books that step (`_HostReads.settle`,
    `Trainer._book_host_values`): its record's loss, accuracy and norm, P1.12's skip count, the
    `skipped_steps` entry, the warning and the stop rule for a skipped step, the reset of the run of
    skips otherwise, and the non-finite-loss check of an applied step.
  * A statistics step (`train.stats_every`, and the last step), an evaluation, a checkpoint, an
    early stop and the end of `fit` book every queued step first: the log line, the evaluation and
    the checkpoint see exactly what `per_step` would have.
  * The step records are `_DeferredStepRecord`s: reading `loss`, `accuracy` or `grad_norm` of a
    record not yet booked waits for its copy and reads it then (the harness reads the train records
    at its early-stop steps); every other field (`seconds`, `lr`, `tokens`, ...) is the host's and
    reads without a wait. Pickled or copied, a record is a plain `StepRecord`.

**`train.fused_adamw: bool`**, default `false`: `torch.optim.AdamW(..., fused=True)`, one kernel for
every tensor. The same update to rounding (it is another implementation). Usable alone, with every
per-step feature.

Both keys are left out of the configuration hash at their defaults (`HASH_OMIT_AT_DEFAULT`) and of
the experiment memory key (`hsn.experiments.spec.memory_key`, which has them when they are set:
they change the step time).

## What keeps working, what is refused

| Feature | With `host_sync: stats` |
|---|---|
| P1.12 non-finite skip | on the device (fused `found_inf`): the same weights, moments and step count |
| P1.12 count, warning, `skipped_steps`, stop rule | the same, booked one step later (or at a stats, evaluation or checkpoint step); a run that must stop stops when the step is read, with the same message |
| the non-finite-loss check | the same, when the step is read |
| step records, `History`, log lines, evaluations, checkpoints | the same values (to the masked loss's rounding); a step is booked before any of them is written |
| early stop (the harness's callback) | reads the records, which are read on access |
| the step-time guard (`step_time.check`, `run_median_s`) | reads `seconds`, a host value. It includes the wait for the step before, so in steady state a step's `seconds` is the card's step period, not the host's launch time; stats steps (which book every step) are left out by the guard as before |
| `train.spike_factor`, `slow_burst_factor` | **refused**: the decision needs the norm against the host's history before the optimiser step |
| `train.rollback_every` | **refused**: the same |
| `train.adaptive_lr`, `train.gradient_growth`, `train.update_limit` | **refused**: each reads the step's norm or update on the host before the next step |
| `train.weight_ema` | **refused**: the average moves only after an applied step, which the host learns a step later |
| `train.divergence_dump` (on by default) | **refused**: the dump holds the weights as the skipped step left them, and a step later they have moved; set it `false` (healthy dumps need it too) |
| `train.answer_weight != 1` | **refused** (its loss indexes the loss positions) |
| `train.grad_norm_precision: float64` | **refused** (it reads the norm on the host itself) |
| `train.precision: fp16` | **refused** (the GradScaler reads its own found_inf) |
| `Trainer.grad_hook`, `Trainer.update_post_hooks` | **refused** at `fit` (they run inside a step on host values) |
| MPS | **refused** (no float64 on the device, no pinned copies) |
| `train.fused_adamw: false` | **refused** with `stats` (the device skip is the fused kernel's) |

The HSN trainer (`TBPTTTrainer`: the HSN and HSN-T with its memory) has a `fit` of its own; both
keys are refused there (`host_sync_refusals`), so that trainer is unchanged. They share only the
optimiser's construction, which builds the foreach AdamW without the key, as before.

What stays a host read: the GRU baseline's own forward reads the window's reset columns
(`reset.any(dim=0).nonzero()`, one sync per window, to split cuDNN's call); that is the model's,
not the trainer's, and is unchanged. The Mamba baseline's eager forward reads its reset columns too
(its compiled window does not).

## Tests (`tests/test_trainer_host_sync.py`)

* the defaults keep hashes and memory keys, named or not; each key changes both;
* the refusals of the table, the HSN trainer's, and the hooks at `fit`;
* `fused_adamw` builds fused groups with the same weight decays;
* the masked loss equals the indexed loss (float64 1e-13, float32 and bf16 1e-6), the same correct
  count and gradients, none off the mask;
* twenty steps of memory-off HSN-T and of the GRU (statistics every 5, evaluations at 10 and 20,
  windows with no loss position among them): `stats` gives `per_step`'s records, evaluations and
  log lines to 1e-5 (with the fused optimiser in both), and the fused optimiser the foreach one's
  to 1e-4;
* a callback that reads every record as it comes gets the values a run without it books;
  records pickle and copy as `StepRecord`s;
* fused AdamW with `found_inf = 1` leaves weights, moments and step count bit for bit;
* NaN logits at steps 4 and 9: the same two skips, `skipped_steps`, warnings, step counts (10 of 12)
  and weights as `per_step`; ten consecutive NaN steps: the same stop-rule message (`per_step`
  raises inside step 12, `stats` while it runs step 13);
* the non-finite-loss check of an applied step;
* CUDA only: the memory-off 160 x 3 model under CUDA graphs and bf16 runs steps 4 to 9 under
  `torch.cuda.set_sync_debug_mode("error")` (no host read, boolean index or blocking copy), with
  `per_step`'s losses and norms to bf16 rounding.

## Timings (RTX 3060, 17 Nov 2026)

400 steps alone on the card through `Trainer.fit` (evaluation and checkpoint stubbed), CUDA graphs
(`hardware.baseline_compile=reduce-overhead`). A run's step time is the median of its step seconds
over steps 21 to 400 without compile and stats steps (the step-time guard's rule); "wall" is the
synchronised mean of steps 21 to 399. Three runs of each arm in one session, the eight arms in
one order, then reversed, then the first order again; the table has the median of the three.
HSN-T: `fast_readers.yaml` `ts27_nomem_w160l3_top0_w1024` (memory-off 160 x 3, w1024 x 8). GRU:
`fast_timing.yaml` `time_gru_t256b32_graphs` (GRU 2M, 256 x 32). "No-sync" is `host_sync: stats`,
which needs `fused_adamw: true` and `divergence_dump: false`; "flash" is `model.reader_mask:
causal` (`docs/reader_causal_flash.md`).

| Model | Arm | ms / step (3 runs) | wall ms |
|---|---|---|---|
| HSN-T 160 x 3 | base (as configured) | **25.73** (24.22, 26.44, 25.73) | 25.94 |
| HSN-T 160 x 3 | + flash | **21.06** (19.64, 21.06, 21.55) | 21.58 |
| HSN-T 160 x 3 | + no-sync | **15.26** (15.23, 15.34, 15.26) | 15.42 |
| HSN-T 160 x 3 | + both | **10.97** (11.60, 10.95, 10.97) | 11.78 |
| HSN-T 160 x 3 | flash + fused AdamW + no dump, syncs kept | 19.00 (18.32, 19.00, 19.16) | 19.13 |
| GRU 2M | base (as configured) | **33.48** (31.40, 34.39, 33.48) | 34.07 |
| GRU 2M | + no-sync | **26.88** (20.64, 29.72, 26.88) | 29.06 |
| GRU 2M | fused AdamW + no dump, syncs kept | 28.30 (25.25, 28.30, 30.22) | 29.75 |

Every run: no CUDA-graph skip, no compile after step 20 (one graph for HSN-T, two for the GRU).
With no-sync the median and the wall agree (15.26 / 15.42, 10.97 / 11.78): a step's `seconds`,
which includes the wait for the step before, is the card's step period, so the step-time guard
reads what it read before. The no-sync lever is the syncs: against flash + fused AdamW + no dump
with the syncs kept (19.00), flash with no-sync gives 10.97.

**The machine was loaded**: other sessions' CPU scoring jobs (six processes of about four threads)
kept the load average at 21 to 23 on the 32 threads during the session, and every absolute number
is above the profile's of 13:22 (as configured 18.36 ms there, 25.73 here; the GRU 16.0 ms on
14 Nov, 33.5 here). The host-side work and its waits are what the load slows, so the levers that
remove host waits gain more here than they would on a quiet machine (the profile's prototype: 18.36
as configured, 13.30 flash, 12.36 flash + loss in the graph + fused AdamW with the syncs, 10.60
the same without them). The GRU, whose cuDNN recurrence runs eager between two compiled graphs and
whose forward reads its reset columns on the host once a window, is the most load-sensitive
(20.6 to 29.7 ms for one arm).
