# P1.17 A5 implementation and validation

This implements the mechanism requested in the 8 November hand-off. It does not establish that
the model binds keys or beats gru_2m. The architecture amendment remains experimental.

## Gradient norm precision

`train.grad_norm_precision=float64` is an optional numerical control for finite gradients whose
native L2 reduction overflows. It widens both the reduction and multiplication before copying
clipped gradients back in place. The default `native` retains historical clipping and hashes.
Actual nonfinite guards remain active. Details and validation requirements are in
[the numerical amendment](amendments/A5_gradient_norm_precision.md). This is not evidence that
the weight drift or key-binding deficit is fixed; learned performance and cost remain unvalidated.

## Adaptive learning rates (9 November follow-up)

`train.adaptive_lr.enabled=true` enables training-only feedback in both trainers. Defaults are
documented in `configs/_example.yaml`; disabled configurations retain their historical hashes,
optimizer groups and scheduler behavior. The new thresholds are hypotheses, not a validated
accuracy improvement. `configs/sweeps/p1_17_a5_adaptive.yaml` supplies matched seeds 0/1 and fixed
recovery controls. It preserves the existing WSD schedule, data, objectives and quality gates.

Every 100 **applied** updates, the controller evaluates the same seeded, stratified sample of
32 training documents (expanded if necessary to cover every task/variant). It uses fresh recurrent
state and an isolated fixed RNG seed, then restores training RNG state. It never reads dev/test
scores. The sample fingerprint is checked on
resume. Monitoring and parameter snapshots count toward training time; the existing step-time
guard remains in force.

The measurements are fixed-sample task loss, the median accepted raw gradient norm, and the actual
sampled parameter-change ratio `||after - before|| / ||after||`, with separate router and decoder
budgets. Sustained loss regression halves rates; regression with gradient growth relative to the
last healthy reference can react earlier. Sustained excess parameter changes reduce the affected
component. Guard skips/rollbacks also halve rates, cumulatively down to the floor. Controller state
is deliberately outside the in-memory rollback ring, so restoring model weights cannot undo a
cut. Use `rollback_lr_mult=1` and `rollback_lr_steps=0` to avoid a competing temporary recovery cut.

After cooldown, a healthy plateau can trial a 10% increase within configured bounds. The default
ceiling is the original rate, so a trial can recover part of a previous cut. A trial is retained
only after meaningful monitor-loss improvement with healthy update/gradient measurements. Rejection
restores rates, **not weights already updated during the trial**. Failed trials remain blocked until
new healthy improvement or a reduction. The selected schedule multiplies these scales; constant
schedule can be tested separately, while WSD still decays at its existing late-run boundary.

Checkpoints contain the full controller state, pending trial, partial gradient window and monitor
identity. Normal and failed experiment `record_json` entries in SQLite include `adaptive_lr`, with
every observation, decision and before/after scale. Standard logs also emit each decision. Runtime
rollback snapshots remain ephemeral as before; full checkpoint resume restores controller state.

A plateau alone cannot distinguish a small rate from a missing or poor learning signal. Existing
counterfactual write probes compare matched examples, state and RNG with a write added or removed;
a positive loss difference means that intervention helped prediction. Key-swap and route-check
probes test key use. Report the number of eligible examples and the distribution of paired loss
differences: zero effect can mean redundancy or a horizon that cannot reach the answer. These are
separate measurements from gradient size and are not dev/test feedback to this controller.

For a controller input, distinguish predictive value from intervention value. Its logged warning
should precede deterioration with useful lead time and few false alarms; a matched-seed ablation
then tests whether acting on it improves final quality and cost. Correlation alone cannot establish
that the warning or the resulting rate change helped. Loss, gradient and update observations are
stored separately to support this analysis without adjusting frozen quality thresholds.

### Live reporting and signal ablations

`python -m hsn.experiments status` shows controller state while an adaptive run is active.
The child writes compact SQLite progress after every controller observation or guard decision,
including observations between dev evaluations. It shows the latest decision, last actual rate
change, training-monitor loss, gradient median, cooldown, pending/blocked trial, and effective
shared/router/readout learning rates for the **next** update. The dev score retains its evaluation
step. The JSON snapshot also retains update ratios, scales, enabled signals and monitor identity;
it does not copy the growing observation history. Each new controller event is flushed to the
run log. The final/failed result retains the complete history as before.

Read existing results without running a model or modifying the database:

```bash
python -m hsn.experiments controller-report --exp 240 --out /tmp/controller-240.json
python -m hsn.experiments controller-report --exp FULL_ID --exp ABLATED_ID
```

The JSON report counts raw warnings followed by a monitor-loss increase greater than the configured
threshold at the next observation. It excludes intervals starting with warmup, baseline, cooldown,
trials or rate changes, and pairs interrupted by a guard. It lists the next loss change after rate
changes separately. Missing evidence is not counted as a successful warning; a signal with no
alerts has no follow-through fraction. These adjacent, correlated measurements are exploratory
training evidence, not causal usefulness or an independent quality score.

Four default-on switches under `train.adaptive_lr` support controlled experiments: `loss_cuts`,
`gradient_cuts`, `update_cuts`, and `trial_increases`. They gate sustained-loss cuts, the combined
loss/gradient cut, component update-budget cuts, and upward trials, respectively. Disabling a
gradient/update signal also removes its veto on healthy plateaus and trials. Raw measurements
continue to be collected. Loss remains the trial reward, and emergency guard/rollback cuts remain
active. Existing enabled-controller configuration hashes, arm identities and checkpoints remain
compatible when all switches retain their defaults.

`configs/sweeps/p1_17_a5_adaptive_ablations.yaml` defines full-controller and one-switch-off arms
for seeds 0/1, preserving the 9000-step recipe and quality gates. Its full-controller arms reuse
the original adaptive sweep's identities. The report pairs only completed, clean attempts with
matching source commit, seed, data, monitor identity, protocol and configuration apart from one
switch, with a final evaluation at the full training horizon. Accuracy deltas use fractions
(0.01 is one percentage point). Repeat across seeds and inspect cost and guard failures as well
as quality; an unpaired or failed run is still reported but does not produce a matched delta.

Implementation validation: 968 CPU tests passed in 71 seconds with eight workers; two focused
compiled CUDA tests passed, including monitoring across rollback and checkpoint resume.
Lint, pre-commit and strict typing passed (80 source files).
CPU continuation tests cover partial monitoring windows, an active upward trial and preserved
training trajectories when no rate changes are made. Reporting tests inspect SQLite while training
is still running, check old dev-score timestamps, guard events, full/failed histories and matching
rules. Experiment 240's saved configuration hash, arm identity and controller state remain compatible.
This validates mechanics, not final accuracy.

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
Both specs set `timing: true`, so the existing SQLite worker runs one arm at a time after its
idle GPU/CPU checks; no second worker or change to its concurrency setting is needed.

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

## Investigating spikes before further training

Experiment 241 completed with five rollbacks and 2,081 discarded updates. Its first saved spike
is concentrated in one example, but that observation does not identify the originating operation.
Raw gradient norms are measured before the existing clipping. A spike alone therefore does not
establish that its clipped optimizer update would damage learning.

The `spike_diagnostic` protocol runs through the ordinary SQLite worker. It uses saved weights,
optimizer moments, scheduler, RNG, loader position, recurrent state and previous byte. It verifies
the checkpoint SHA-256, configuration and regenerated windows before backward. No training
continuation, final evaluation or training checkpoint is produced. Each attempt records its parent
experiment, source checksum and commit, seed, data fingerprint, execution mode, intervention,
progress and results. Diagnostic wall time is not a training step-time measurement.

Run stages in order: `reproduce`, `isolate`, `trace`, `update`, then `compare` on other seeds and
non-A5 controls. Reproduction compares repeated eager/compiled passes, including routing choices.
Isolation separates objective terms (answer loss, prediction loss and any enabled variance term)
and optional document contributions, preserving the forward batch and original normalization.
For a batch-coupled variance loss, document attribution means its output-gradient contribution,
not an independent per-document scalar loss. The gradient partitions must reconstruct the baseline
within 1e-5 relative error in fp32 or 0.02 under bf16; outputs must remain identical.

Tracing clones each module's differentiable input to isolate its local VJP from other consumers.
It first verifies forward outputs and parameter gradients against an uninstrumented baseline.
Separate diagnostic cuts of h/q at position 31 preserve forward values; these are investigative
interventions, never changes to the model's training equations. An earlier retained checkpoint can
provide a reference window; its health must be assessed from the measurements, not its age.

The update stage takes one clipped optimizer update on a disposable copy, including lazy-AdamW
active-unit semantics and target EMA. It reports raw/clipped gradients, parameter displacement,
and loss before/after on a fixed stratified TRAIN monitor. Dev/test scores are not used to select
the update. One-step behavior cannot establish long-term accuracy or justify weakening guards.

Optional capture settings for subsequent training are `train.divergence_max_dumps=5`,
`train.divergence_healthy_every=50`, and `train.divergence_healthy_keep=2`. Defaults remain 1/0/2,
with unchanged legacy hashes and training behavior. Guard-accepted finite windows are saved before
their update; their JSON sidecars receive actual parameter displacement after the update. At each
spike, the current healthy references are frozen beside that event. Storage is bounded by the
event cap and reference count. `divergence` remains the first-event summary; `divergences` records
all captured events and survives normal checkpoint resume. Per-parameter norms in captures are
post-clip; reconstructed raw norms are explicitly labelled. Diagnostic backward records measure
raw parameter gradients directly.

Objective and document attribution also support `modes: [compiled]`. These interventions mask
only the loss's backward inputs, preserving the compiled forward. Reports identify each execution
mode separately and enforce the same repeatability, unchanged-forward and gradient-reconstruction
checks. An eager trace does not identify the compiled gradient path when the routes differ.

`reference_mode: same_window_weights` loads an earlier checkpoint's model parameters and buffers
onto the spike's exact saved stream, RNG, carried state and previous byte. The reference is
replayed twice in every requested mode, without any optimizer update. Both sides record the same
window fingerprint, weight source and whether every row resets at the first position. This is a
controlled weight intervention, not a reconstruction of the earlier training trajectory. The
default `reference_mode: next_window` instead uses the earlier checkpoint's own next window;
its fingerprint is reported separately, and a norm difference alone cannot isolate weight changes.

### Capturing a burst when no guard dump exists

The `spike_diagnostic` stage `capture` takes a retained checkpoint in `dump` (with its SHA-256)
and a `capture` mapping containing `steps`, the original `result.json` in `trace`, and
`trace_sha256`. It resumes disposable copies using the original schedule, execution mode,
RNG, optimizer, loader, carried state and previous byte. It currently accepts sources without
rollback or finite-gradient skipping. The checksummed trace supplies the original `arm.run_id`
for trainer resume; the diagnostic retains its separate database identity. The full resume
configuration check remains enforced, including every training setting.
Every replayed task loss and raw gradient norm must match
the source trace; `rtol` and `atol` default to zero and are explicit protocol settings. A mismatch
fails the diagnostic and pauses its queue. Do not weaken a tolerance to call a different trajectory
an exact reproduction.

Each requested step is captured after clipping and before its update; earlier replay steps keep
their normal updates and all documents. The final requested update is not applied. At least two
replays compare the complete captured training state exactly, including weights, optimizer,
scheduler, RNG, stream and recurrent state. Capture neither marks a skip nor modifies event
counters. The parent experiment and checkpoint remain unchanged. The artifact includes the
source trace checks, snapshot/window checksums, disposable update count, and a copied reference
checkpoint. Existing `reproduce`, `isolate` and `same_window_weights` diagnostics can use these
artifacts under the capture experiment as their recorded parent.

These are verified reconstructions, not original in-flight captures. Metric agreement is checked
against the original run; tensor equality across repeats establishes reproducibility of the
reconstruction, not equality to unsaved original tensors. A failed reconstruction remains evidence
of its own failure and must not be used to explain the original burst.

For `same_window_weights`, optional `reference_selectors` select parameter-name globs from the
reference checkpoint. Only those parameters are replaced; unselected parameters, persistent
buffers, stream, RNG and carried state remain from the source window. Every selector must match.
An empty selection retains the existing whole-model intervention (parameters and buffers).
The report names exactly which parameters changed and repeats the intervention to check it.
This can test whether a group affects sensitivity on a fixed window. Mixed checkpoints may be
unlike any learned model, so a changed norm alone does not establish which group initiated the
training failure or that a corresponding optimizer change will fix it.
