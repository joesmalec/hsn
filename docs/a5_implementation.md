# P1.17 A5 implementation and validation

This implements the mechanism requested in the 8 November hand-off. It does not establish that
the model binds keys or beats gru_2m. The architecture amendment remains experimental.

## Controls and compatibility

`model.key_address` and `model.selected_readout` are independent booleans, both false by default.
Enable them only on routed HSN models. The reference dense model, baseline models, frozen data,
G1 task set, thresholds, and cell-state schema are unchanged. Existing hashes and model state-dict
keys are preserved when both controls are off. Each added feature can be tested by itself.

The scorer owns fixed, persistent `byte_codes[V,d]` and `cell_addresses[N,d]` only when addresses
are enabled. Their CPU generator uses the run seed and a fixed tag. Added scorer and decoder
columns also use isolated initialisation; common parameters retain their initial values. Address
dot products run in master precision, independently of autocast. Feature ablations can exclude
`a_cur` or `a_prev` by name.

`HSN.forward(..., prev_tokens=...)`, `HSN.step(..., prev_tokens=...)` and the compiled wrappers
accept the previous byte for each row. None means BOS. Resets replace that byte with BOS before
scoring. The trainer, evaluator, profiling windows and counterfactual probes carry it across
windows; A5 checkpoints and divergence snapshots save it. Rollback clears it and starts fresh
documents. It is deliberately outside `CellGraphState`: the state schema remains version 1.

## The readout's training gradient

Hard execution reads the mean of the updated valid selected cells. Padding contributes neither
state nor denominator. Dense-fallback rows read `q`, the mean over all cells they executed.
The masked path reads the normalised mask-weighted mean over all cells, including soft warm-up.

For STE, a gathered hard mask carries the existing straight-through derivative into the selected
scores. Its forward value is exactly one, so the numerical readout remains the specified selected
mean. This gives task loss a direct route to selection even when the cell's write gate is small.
Plain integer gathering alone would not provide that gradient. This surrogate is an implementation
choice using the existing STE discipline, not a new loss, an oracle route, or a binding result.

The model still uses persistent vector cells, the fixed sparse graph, shared units, prediction
error, convex tanh updates and `q`. It adds no sequence attention, matrix state, or weight writes
to the computation path.

## Experiment protocol

Expand the two-seed control, address-only, readout-only and combined arms:

```bash
python -m hsn.experiments expand configs/sweeps/p1_17_a5.yaml
```

Expansion creates no experiment and starts no training. The sweep records the documented exp-196
dimensions and schedule explicitly, including beta2=.999. The 9,000-step runs and CUDA step-time
measurement require the reference GPU. Existing step-time guards continue to apply.

After each run, use the saved step-9,000 checkpoint with the existing commands:

```bash
python -m hsn.experiments probe-rerun --exp N --key-alphabet train
python -m hsn.experiments route-check --exp N --split train
python -m hsn.experiments probe-rerun --exp N
python -m hsn.experiments route-check --exp N --split dev
```

Use the database/root arguments of the worker that ran the experiment. Require the hand-off's
trained-key binding result before interpreting unseen-key generalisation. The combined arm must
meet the unchanged swap and route-check criteria; later-query accuracy must beat the key-blind
0.623 reference. Measure matched shapes and precision on an idle GPU, reporting compilation
separately and scorer cost explicitly. The predicted +5% overhead is a hypothesis.

Do not promote a utility or warm-up switch from exps 210–213 without reading their results. This
change leaves those switches off; it implements the readout's direct gradient path. The new A5
spec sets `train.probe_horizon=null`, which probes to the current window's end if utility is
enabled. An integer retains the legacy fixed horizon. The trainer reports dependency eligibility
and warns when the chosen horizon cannot reach source positions, rather than silently treating
near-answer probes as supervision for memory writes. Null does not extend gradients or probes
across the TBPTT boundary. If both
combined seeds meet the draft's kill condition, run its oracle-gate diagnostic before further
architecture changes. Reweighting training data and adopting rollback as a default remain separate
decisions. The frozen benchmark must not be weakened.

Compare both gru_2m reference recipes: final score at lr=5e-4, beta2=.999, and learning speed at
lr=2e-3, beta2=.95. Equal task/data/search budgets, per-task quality, trained/unseen key binding,
onset, latency, energy, memory and total training cost must accompany a superiority claim.

## Stability follow-up to the 9 November results

The first A5 sweep lost both combined runs to collapse. The larger router gradient establishes
that the selected readout changes optimization; it does not establish binding or higher final
accuracy. The retention policy also deleted intermediate checkpoints after a normally completed
run, even when its accuracy had collapsed. The following controls address those failures without
changing the model equations, data, losses, benchmark metrics or existing default configurations.

| Control | Behavior |
| --- | --- |
| `train.slow_burst_factor` (default null) | In addition to the sharp spike guard, compares the median of the latest `spike_window` norms, including the candidate step, with a full window ending `slow_burst_window` applied steps earlier. Both the current norm and the recent median must exceed the factor times that older median. Requires `spike_factor`. |
| `train.slow_burst_window` (default 250) | Separation between the two median windows. History includes accepted steps only and survives checkpoint resume, divergence dumps and rollback. |
| `train.router_lr_mult` (default 1) | Scales the learning rate of `HSN.router` independently; requires learned routing. |
| `train.readout_lr_mult` (default 1) | Scales the whole `HSN.decoder`, including its shared input columns and output layer, not just the selected-cell columns. |
| Protocol `checkpoint: retained` | Retains intermediate evidence and pins the best evaluated checkpoint; final metrics remain final metrics. |

The new learning-rate groups work with AdamW and lazy AdamW, preserve cell-specific rates and
weight decay, and retain their ratios through warmup, decay, rollback rate cuts and resume.
Default groups and configuration hashes are unchanged.

A slow detector needs older recovery weights too. When it is enabled, the in-memory rollback ring
has `2 + ceil((spike_window + slow_burst_window) / rollback_every)` slots instead of two. Recovery
restores the oldest slot and its guard history. With 100/250/50 this is nine snapshots; memory
and copy overhead must be measured on the target GPU. Ring snapshots remain in-memory only, as
before: a resumed run starts filling a new ring. Failed candidates never enter guard history.

The first comparison is `configs/sweeps/p1_17_a5_b095.yaml`: control and full A5, seeds 0 and 1,
9,000 steps, changing only beta2 from 0.999 to 0.95. It saves every 500 steps with `keep_last: 18`
and `checkpoint: retained`. Do this before adding other stability settings.

If that comparison still collapses, `configs/sweeps/p1_17_a5_recovery.yaml` provides a matched
control, A5, router-rate 0.3 and decoder-rate 0.3, each on two seeds. All arms share spike factor
100, slow-burst factor 10 with lag 250, and rollback 50/10/0.5/200/5. The rate reductions are
separate ablations; do not infer their combination works from either alone. No default recipe
is promoted from a single successful seed.

For each completed run, measure trained-key swap binding and route following at the final step
and at the retained peak, then evaluate unseen keys. For example, for a retained step 2,000:

```bash
python -m hsn.experiments probe-rerun --exp N --checkpoint-step 2000 --key-alphabet train
python -m hsn.experiments route-check --exp N --checkpoint-step 2000 --split train
python -m hsn.experiments probe-rerun --exp N --checkpoint-step 2000
python -m hsn.experiments route-check --exp N --checkpoint-step 2000 --split dev
```

Report both seeds' final and best dev accuracy, later-query accuracy, skipped steps, rollbacks
and binding verdicts. A 0.2 swap follow is a screening threshold, not the `binds_keys` success
criterion: the existing preset requires follow at least 0.5, unchanged at most 0.1 and at least
20 eligible originally correct swap probes. Keep the route-check criterion and its swap control,
and the later-query key-blind reference 0.623. A post-hoc peak is diagnostic evidence and must
not replace a collapsed final result or count as an independent confirmatory run.

An offline check of the original `ff57fc8` traces (experiments 221 and 225) first flags the slow
growth at steps 2,253 and 6,036. The 100x sharp guard first flags steps 3,471 and 5,266 respectively,
so their combined first detections are 2,253 and 5,266. This reuses recorded gradients and cannot
predict the trajectory after a rollback. Factor 20 also fires during successful control training
(experiment 222, step 2,850); the fallback starts with factor 100 to reduce early interventions.
