# Experiment sweeps (P1.11)

Status: P1.11, 7 Nov 2026; follow-ups P1.14, 7 Nov 2026, P1.15 and P1.16, 8 Nov 2026. The harness turns one YAML file into many experiments,
with no code change per experiment. Every experiment, past, present and future, is recorded in one SQLite
database, which is also the queue. Code: `python/hsn/experiments/`. Command line:
`python -m hsn.experiments <command>` or `hsn sweep <command>`.

P1.14 added:
- [failed arms keep their partial record](#failed-arms);
- [the 0.5 s step-time guard](#the-step-time-guard);
- [the mixed diagnostic cadence](#mixed-diagnostic-cadence) and [`min_samples`](#rules);
- [probe v2](#probe-v2-the-vocabulary-matched-edit), the vocabulary-matched edit;
- the write rate next to the gate quantiles.

P1.15 added:
- [the first/later split](#the-firstlater-split) of every reads_state evaluation, and a read-only
  [backfill](#reports-and-import) for stored runs;
- [probe v3](#probe-v3-key-addressed-edits): the value edit and the key swap of the later, key-addressed queries
  (reads_state `version: 3`);
- [the gradient-spike guard](#gradient-spikes-and-divergence-dumps) (`train.spike_factor`, `train.spike_window`)
  and the divergence dump with its deterministic replay (`train.divergence_dump`, on by default);
- [the presets](#rules) `memory_writes`, `reads_state_v3` and `binds_keys`, and `min_vocab` on
  `reads_state_v2`.

P1.16 added:
- [the rollback](#rollback) (`train.rollback_*`): a guard trigger restores an in-memory snapshot instead of
  skipping, and the run trains on;
- [kept checkpoints and partial results](#kept-checkpoints-and-partial-results) (protocol `keep_every`,
  `keep_last`): a failed arm is re-scored from its newest checkpoint before the failure, labelled
  `failed: … (from step X: …)`, and the `partial:` rule source;
- [route-check](#route-check-does-routing-follow-the-key), a read-only diagnostic of whether a routed model's
  routing follows the key;
- `probe-rerun --weights dump` (a failed run's divergence dump) and `--key-alphabet train` (a diagnostic remap of
  the dev keys), and `replay --continue-steps` with a remedy;
- [the HSN recipe file](#recipes) `configs/experiments/hsn_recipe_b095.yaml` (AdamW β2 0.95);
- [the preset](#rules) `memory_writes_run` (the written gates' 90th percentile and the cell's update ratio, each in
  some window of the run), calibrated on the stored runs;
- a first open's switch to WAL retries "database is locked".

Existing specs expand to the same arms with the same ids, and every existing configuration keeps its
`config_hash` (P1.16's keys are left out of the hash at their defaults).

## Quick start

```bash
python -m hsn.experiments expand  configs/sweeps/example_reads_state.yaml   # arms and ids, no database
python -m hsn.experiments enqueue configs/sweeps/example_reads_state.yaml   # queue them
setsid nohup python -m hsn.experiments worker > ~/hsn-experiments/worker.log 2>&1 &
python -m hsn.experiments status                                            # queue, running, recent
python -m hsn.experiments report --sweep example_reads_state                # one row per seed group
```

Adaptive runs publish controller observations to `status` between dev evaluations, with effective
rates for the next update, the latest decision and the last actual rate change. `dev @ step` identifies
the evaluation behind a displayed dev score. Full signal histories remain in each result's
`adaptive_lr` block. `python -m hsn.experiments controller-report --exp N [--exp M ...] [--out FILE]`
reads warning follow-through and strictly matched ablation comparisons without loading a model or
writing to SQLite. See [controller reporting and ablations](a5_implementation.md#live-reporting-and-signal-ablations).

For a spike investigation, prepare an ordered diagnostic spec without changing the database:

```bash
python scripts/prepare_spike_queue.py --db ~/hsn-experiments/experiments.sqlite \
  --parent 241 --target-doc 15751 --compare 235 --compare 239 --compare 233 --compare 234 \
  --out ~/hsn-experiments/a5-stability/spike-queue.yaml
python -m hsn.experiments enqueue ~/hsn-experiments/a5-stability/spike-queue.yaml
```

Enqueue and resume only in the explicitly owned compute turn. The spec inherits each parent's
configuration and seed, pins source hashes, marks every job exclusive (`timing: true`), and chains
prerequisite labels. `spike_diagnostic` runs strictly in insertion order. An exception, failed
repeatability/instrumentation check, cancellation or timeout pauses the queue; dependent jobs do
not silently continue. Results are diagnostic records, with no fabricated final dev score.

A worker configured with `--turn-file ~/hsn-handoff/run-turns.json --turn-owner A5` requires an
active, unreleased claim for that owner. It starts a bounded job only if its full timeout plus
30 seconds fits the turn. It signals a timed-out diagnostic, then kills its owned process group
after ten seconds if needed. A turn's last fifteen seconds are reserved for draining. Missing,
malformed, expired or another owner's claim blocks launches. Replace an idle existing worker to
enable these options; do not start a second one. Pending jobs may remain paused between turns;
coordinate them with any external idle guard before handing compute to another session.

See [spike diagnostics](a5_implementation.md#investigating-spikes-before-further-training) for
capture settings, numerical checks and limits on interpreting these results.

**Database location:** `~/hsn-experiments/experiments.sqlite`, here `/home/joe/hsn-experiments/`. Override it
with `--db` or `$HSN_EXPERIMENTS_DB`.

**Run outputs:** `~/hsn-experiments/runs/<sweep>/<run_id>/exp<N>/`. Override with `--root` or
`$HSN_EXPERIMENTS_ROOT`. Each run directory holds:
- `result.json`: the full record, with every probe;
- `run.log`;
- `config.resolved.yaml`;
- the final checkpoint (P1.16: and, with `keep_every`, the kept mid-run checkpoints until the run finishes);
- `divergence_<step>.pt` and its `.json` sidecar, when a guard skipped or rolled back a step (P1.15);
- `dirty.patch`, when the checkout had modified tracked files.

The database is the index; the JSON files are the full copies. A path under `/tmp` is refused, because `/tmp`
does not survive a reboot.

**Shared machine:** run every command from a checkout whose `hsn` you mean to test. Ask the GPU coordinator
before starting a worker that uses the card.

## The spec

```yaml
name: my_sweep                              # letters, digits, . _ -
description: what this sweep asks
base:                                       # every arm starts here
  config: configs/experiments/g1_ref_a1.yaml
  overrides: [profile=rtx3060, train.batch=32, train.micro_batch=32, train.steps=3000, ...]
families:                                   # optional per-family recipe, applied after base
  gru: {config: configs/baselines/gru_2m.yaml, overrides: [train.warmup_steps=200, ...]}
protocol: {name: reads_state, diag_every: 500, diag_docs: 32, eval_batch: 128, probe_batch: 64}
# P1.14, all optional:  version: 2 (probe v2)   diag_dense_every: 25 (mixed cadence)
#                       max_step_s: 2.0 + step_time_approval: "user, 7 Nov 18:19: E5 pair" (above 0.5 s)
# P1.15, optional:      version: 3 (probe v3, after v1 and v2); and, as config overrides,
#                       train.spike_factor=100 (the spike guard), train.divergence_dump=false (no dump)
timing: false                               # true: the arm runs alone, on an idle card and CPU
constants: {chance: 0.0208333}              # usable in rule values
seeds: [0, 1, 2]
rules: {verdicts: {...}, early_stop: [...]}
arms:
  - name: hsn_d128
    family: hsn                             # hsn | gru | transformer; checked against model_name
    variant: pdg                            # or {name: cell_offset, offset_std: 0.5}
    overrides: [model.d=128]
    grid: {train.lr: [0.001, 0.003]}        # optional: cartesian product, in declared order
    seeds: [0]                              # replaces the sweep's seeds
    protocol: {diag_every: 25}              # merged over the sweep's (unless it names another)
    timing: true
    constants: {ref_dev: 0.07}
```

**Override order.** Overrides apply in this order, the later winning:
1. base;
2. family;
3. arm;
4. grid point;
5. `seed=<s>`;
6. `run_id`, set by the harness.

`hsn.config.load_config` validates the result, so an unknown key raises. Configuration paths are relative to the
repository root.

**Expansion checks.** Expansion raises on:
- a CUDA arm that does not pin `hardware.compile_mode` in its overrides (`profile:rtx3060` compiles by
  default once P2.8 is merged; a pin keeps the arm's id and computation independent of that default);
- a compile mode other than `none` on a GRU or Transformer arm (P2.8 compiles the HSN window only; a baseline
  would train eager under a compiled label);
- a family that does not match `model_name`;
- a variant on a baseline;
- a variant combined with `hardware.compile_mode`;
- `routing=random` with the reads-state protocol;
- protocol steps beyond `train.steps`;
- a duplicate label;
- any rule that does not fit an arm's cadence or protocol.

**Labels and run ids.** Labels are `name[key=value,...]/s<seed>`. Run ids are `<name>-s<seed>-<arm id[:12]>`.

**Arm ids.** An arm id hashes what the arm computes:
- the resolved configuration's non-default values, without `run_id` and `notes`;
- the data (the manifest file);
- the family, the variant and its parameters;
- the protocol's name, version and science parameters;
- the early-stop rules, with the arm's constants substituted.

What leaves the id unchanged:
- YAML spelling, key order and the arm's name;
- the execution-only parameters `eval_batch`, `probe_batch` and `checkpoint`, which change speed and rounding
  only, P1.14's `max_step_s` and `step_time_approval`, which decide whether an arm may run, and P1.16's
  `keep_every` and `keep_last`, which keep checkpoints;
- verdict rules, which can be re-evaluated from the stored record;
- a new configuration key with a default;
- P1.14's `diag_dense_every` and `diag_windows` at their defaults (0 and none), and reads_state's
  `version: 1`. Versions 2 and 3 and a dense cadence give new ids, since they change what the arm records.

A changed default is a code change. Like every code change, it is tracked by the commit each run records.

**Group id.** The group id is the arm id without the seed; it groups seeds in reports.

## Protocols

**`reads_state`**: the P1.10 audit's reads-state check. It trains on the configuration's data; the spec's
overrides carry the 5(b) recipe. Then it runs, on the trained model:
- the cell-state statistics (`trained_state`);
- the 5(c) edit probe on every dev `delayed/in_range` document.

During training, the diagnostics run every `diag_every` steps on 4 dev documents:
- per-module update ratio and gradient share;
- cell diversity, `rank_pr`, saturation;
- gate quantiles, with the write rate (P1.14, `gate/write_rate`) and the mean gate (`gate/mean`).

The write rate is the share of the written cells' gate coordinates at or above 0.5. Gate p50 alone cannot tell
"no writes" from "few but strong writes": blocks B1 and B2 had p50 about 0 and p90 0.76 to 0.91, and still
learned. Both numbers are reported only; no rule or preset reads them.

The probe is batched:
- rows are truncated after the answer position;
- originals and edits share a batch;
- every row gets a fresh state.

It equals the audit's per-document probe exactly in fp64. In fp32 and bf16 a near-tie can flip, because
batching changes rounding; `probe-check --exp N` lists such flips with their top-2 margins. The probe raises
when no dev document is probed (P1.14).

### Probe v2: the vocabulary-matched edit

`protocol: {name: reads_state, version: 2}` runs v1's probe and then v2 on the same model. v1's `probe` stays in
the record, unchanged, and version 1 arms keep their ids and results.

At the old recipe, dense HSN answered with only 10 to 18 of the 48 values. For correctly answered documents,
v1's new value (the next value absent from the document) was in the model's answer vocabulary in only 0 to 11 %
of cases, so "follows the edit" was near 0 by construction. v2 writes a value the model can say:
- **Vocabulary:** the value symbols among v1's first-query answers on every probed dev document. A value counts
  once it is answered once; answers that are not value symbols are counted (`answers_outside_values`) and left
  out.
- **The edit:** the candidates are the vocabulary minus the document's own symbols, so never the old value.
  The new value is the candidate at `blake2b("hsn-probe-v2:0:<doc id>") mod len(candidates)`: seeded,
  reproducible, and independent of the other documents. A document with no candidate is unmatched and counted.
- **Batching:** v2 uses v1's documents, order and chunks, the v2 edit in place of v1's. The original rows'
  answers should equal v1's; any that differ are counted as `original_mismatches`. Under top-K routing the rows
  of a batch share the per-unit GEMMs, so an exact tie can flip.

`probe_v2` reports:
- follow / other / unchanged, as v1, over the matched documents and over those answered correctly before the
  edit;
- `vocab` and `vocab_size`;
- `matched_frac`, the share of documents where a matched edit was possible;
- the verdict of preset `reads_state_v2`, in this order:
  1. `no matched probes` at 0 correct matched documents;
  2. `too few matched probes` below 20 (`min_correct`);
  3. then v1's thresholds: reads state (follow ≥ 0.5 and unchanged ≤ 0.1), does not read state (follow < 0.2),
     partial otherwise.

`probe-rerun --exp N [--cpu]` runs v1 again on a stored checkpoint, compares it document by document with the
stored probes, then runs v2. On exps 46, 48 and 53 (CUDA, bf16) v1 reproduced every stored answer.

| exp | v1 correct: follow / other / unchanged | v2 vocabulary | v2 matched | v2 correct: follow / other / unchanged | v2 verdict |
| --- | --- | --- | --- | --- | --- |
| 46, eager bridge | 104: 1 / 88 / 15 | 18 | 100 % | 104: 29 / 64 / 11 (0.279) | partial |
| 48, compiled bridge | 59: 0 / 29 / 30 | 10 | 100 % | 59: 14 / 28 / 17 (0.237) | partial |
| 53, E4 top-K | 40: 0 / 15 / 25 | 10 | 100 % | 40: 5 / 13 / 22 (0.125) | does not read state |

For exp 53, 2 original answers were exact bf16 ties: one side's top-2 margin was 0.0.

### The first/later split

In a delayed/in_range document the first query always asks the first fact, so position alone answers it; the
later queries (slots 1 to 3) ask facts by key, and the dev keys (bytes 112 to 127) never occur in training. Every
strong model scores about 1.0 on first queries and about 0.60 on later ones, just under the audit's key-blind
bound (0.800 overall, 0.623 on later queries). One dev accuracy hides that, so every reads_state evaluation now
reports the split (`hsn.experiments.split`):
- each evaluation entry carries `split`: `first`, `later` (slots 1 to 3 together), `later/1`, `later/2`,
  `later/3`, each with `accuracy`, `loss` and `n`;
- `final` carries it flattened: `split/first/accuracy`, `split/later/accuracy`, `split/later/1/n`, ...

It is computed from the evaluation's own per-position values: the dev documents are parsed (each loss position
must follow `QRY` and be a key, and slot 0 must ask the first fact; anything else raises), and a second
accumulator averages the same values by slot. The frozen metric is computed exactly as before. The split is
checked against it: the slots' positions, correct answers and losses must sum to the delayed/in_range group's,
or the run raises. Rules read it as `eval:split/<g>/<metric>` (a step series) or `final:split/<g>/<metric>`.
The dev set has 2,126 answers: 1,000 first queries and 1,126 later ones (661 / 404 / 61 by slot).

### Probe v3: key-addressed edits

`protocol: {name: reads_state, version: 3}` runs v1 and v2 unchanged, then v3 on every *later* query `QRY k v`
of every dev delayed/in_range document. Each query gets two edits, each on its own copy of the document:
- **the value edit:** fact k's value becomes a value of the model's answer vocabulary absent from the document.
  The vocabulary is the value symbols the model answers at any query of the original documents, in v3's own
  pass. The draw is `blake2b("hsn-probe-v3:value:0:<doc id>:<slot>")`. The probe *follows* when the answer at
  that query changes to the new value. It tells whether the later answer is read from the state;
- **the key swap:** the keys of fact k and of a partner fact j trade places (j is not k, not asked before this
  slot, and has another value; drawn the same way). The probe *follows* when the answer changes to j's value.
  A reader that binds keys follows. A key-blind reader (positions and exclusion only) does not change, because
  no position or value moved. It tells whether the read is addressed by key.

The value edit alone cannot tell the two readers apart: a key-blind reader that guesses fact k by position also
reads k's value from its state, so it follows whenever its guess is right (36.6 % of later queries have a single
eligible fact). Every probe also records its key-blind ambiguity: the number of facts a key-blind reader may
assign to the slot (index w ≥ 1 with 2w ≥ 3·slot, not yet asked; the generator's rule and `ceiling_check.py`'s).

`probe_v3` reports, for both edits, follow / other / unchanged over the later queries answered correctly before
the edit, overall, by slot (`by_slot`) and by ambiguity (`by_ambiguity`: `single`, `multiple`), plus the
vocabulary, the probe pass's own first/later accuracy (`pass_accuracy`) and the verdicts:
- value edit (preset `reads_state_v3`): `no correct later queries` (none), `vocabulary too small` (below 20,
  `min_vocab`), `too few correct later queries` (below 20 edited, `min_correct`), then v1's thresholds,
  labelled for the later queries: `reads state (later queries)`, `does not read state (later queries)`,
  `partial (later queries)`. A later answer read from the state; it says nothing of key binding, which is the
  swap's verdict;
- key swap (preset `binds_keys`): `no swap probes`, `too few swap probes` (below 20), `binds keys` (follow ≥ 0.5
  and unchanged ≤ 0.1), `does not bind keys` (follow < 0.2), `partial`.

Batching is v1's (rows truncated after their last scored position, a fresh state per row, chunks by length);
`edit_probe_v3(..., reference=True)` is the per-row loop, equal to it exactly in fp64. The per-probe rows are in
`result.json` (`probes_v3`, `probes_v3_originals`), not in the database. Oracle models in the tests pin the
meaning: a reader that binds keys reads `binds keys`, a key-blind positional reader `does not bind keys`.

`probe-rerun --exp N` (P1.15) also compares v2 with the stored v2 probes and runs v3. On the stored checkpoints
(CUDA, bf16, `probe_batch` 64; 8 Nov 2026) v1 reproduced every stored answer of all four runs and v2 every
stored answer of exp 136. v3, over the later queries answered correctly before each edit (vocabulary 48 in all
four):

| exp | model | later-query accuracy (probe pass) | value edit: follow / other / unchanged | key swap: follow / other / unchanged | value verdict | swap verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 136 | HSN d=64 + A4, WSD 9k | 0.604 | 535 / 140 / 5 of 680 | 2 / 4 / 671 of 677 | reads state (later queries) | does not bind keys |
| 154 | gru_2m, WSD 9k at 5e-4 | 0.604 | 532 / 139 / 9 of 680 | 0 / 1 / 676 of 677 | reads state (later queries) | does not bind keys |
| 161 | gru_2m, WSD 15k at 5e-4 | 0.607 | 544 / 130 / 10 of 684 | 0 / 0 / 681 of 681 | reads state (later queries) | does not bind keys |
| 119 | transformer_2m, 3k of 9k at 1.25e-4 | 0.239 | 92 / 153 / 24 of 269 | 0 / 1 / 265 of 266 | partial (later queries) | does not bind keys |

Every model reads its later answers from its state (on queries with a single key-blind candidate the value edit
is followed 96 to 97 % of the time by HSN and gru_2m) and none binds keys: swapping two facts' keys changes
almost no answer. This is the audit's key-blind finding shown by an intervention.

**P1.16, `probe-rerun` options** (read-only database; nothing is written):
- `--weights dump` probes the parameters of the run's divergence dump (the state before its first guard event)
  instead of its last checkpoint: a run that failed without a checkpoint still gets its verdicts.
- A run without stored probes (a failed one) skips the comparison with them (`compared_with_stored: false`).
- Every rerun adds the dev evaluation with the first/later split (`eval`).
- `--key-alphabet train` remaps, in memory, the dev documents' key bytes 112 to 127 to the training keys 64 to
  79 (a bijection: no byte of 64 to 79 occurs in a dev document) before probing. It is a **diagnostic, not a
  metric** (the output says so; the frozen data and metrics are untouched): it tells "cannot bind keys" from
  "cannot bind keys never seen in training". After training, unused embedding rows have decayed by
  exp(−wd·Σlr) (×0.86 over 9k WSD steps at 2e-3), and dev keys keep random directions while trained keys share a
  learned one (P1.16 measurement, `docs/amendments/A5_key_address.md`).

**`--key-alphabet train` on the stored checkpoints** (8 Nov): no model binds even keys it was trained on. The
evaluation hardly changes: exp 136 scores 0.790 with the dev keys and with the remapped ones.

| exp | model | dev (remapped) | later | value edit | swap follow / unchanged | swap verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 136 | HSN d=64 + A4 | 0.790 | 0.604 | reads state (0.78) | 0.007 / 0.991 | does not bind keys |
| 154 | gru_2m | 0.790 | 0.603 | reads state (0.79) | 0.027 / 0.944 | does not bind keys |
| 161 | gru_2m, 15k | 0.790 | 0.605 | reads state (0.79) | 0.022 / 0.944 | does not bind keys |
| 119 | transformer_2m | 0.593 | 0.239 | partial (0.37) | 0.008 / 0.992 | does not bind keys |
| 194 | HSN, β2 0.95 | 0.787 | 0.598 | reads state (0.78) | 0.006 / 0.988 | does not bind keys |
| 195 | HSN, β2 0.95 | 0.781 | 0.586 | reads state (0.79) | 0.003 / 0.992 | does not bind keys |
| 196 | HSN top-K | 0.698 | 0.440 | reads state (0.66) | 0.008 / 0.954 | does not bind keys |
| 197 | HSN top-K | 0.309 | 0.251 | partial (0.40) | 0.014 / 0.964 | does not bind keys |
| 198 | HSN top-K | 0.564 | 0.344 | partial (0.39) | 0.008 / 0.974 | does not bind keys |

So the missing binding is not about keys never seen in training: within 9,000 steps the models never learn to bind
keys at all, and stay at the key-blind policy.

**`train_eval`**: train and evaluate, with optional diagnostics and `trained_state`. No probe.

For `train_eval` and `reads_state`, optional `n_train` / `n_dev` caps sample deterministically by
`run.seed`, stratified over task/variant groups. Every group receives at least one document and the
remaining budget follows group sizes; a cap smaller than the group count raises. Selected documents
keep their original relative order. Results carry `data_selection` with the full and effective
document, token and answer counts per group. Capped arms have a new identity for this policy;
re-scoring historical records preserves their original prefix selection. Zero still means the full
split. These caps select existing documents and do not change any frozen shard or split.

**`overfit`**: the G1 overfit recipe. The first `n_docs` (64) training documents are both train and dev, and the
record gives `first_ge_0.90` and `first_ge_0.99`.

**`dry_run`**: no training and no torch; a synthetic record. Use it to check a spec, its rules and the queue end
to end.

**Variants.** `pdg`, `gru_cell`, `gru_cell_linear_head`, `linear_head`, `decoder_fc1_fan_in`, `cell_offset`,
`cell_embed`, `cell_embed_linear_head`. They are installed on the built model by name, before the trainer (and
any compiled window) is built, and add no configuration keys. `gru_cell_direct_input` is retired and raises.

Variants run eager only. A variant patches the model, and a compiled path may not call what it patched (as
P1.10a's forward stopped calling what `gru_cell_direct_input` patched), so a variant with a compile mode other
than `none` is refused at expansion and again when the run starts. A test checks that every variant changes the
model's logits.

**Compiled step.** `hardware.compile_mode` is an ordinary override: `none` or `reduce-overhead` (P2.8), pinned in
every CUDA arm. Without P2.8, a checkout refuses any mode but `none` when the run starts, so the arm fails
loudly; once P2.8 is merged, requeue it. Each record holds the compile mode, whether a compiled window ran, and
P2.8's `compile_seconds` (training steps) and `window_compile_seconds` (every compiled variant, evaluation
included).

Compiled artifacts persist between arms: the worker gives every child `TORCHINDUCTOR_CACHE_DIR`, by default
`<db dir>/inductor-cache` (inductor's own default is under `/tmp`). Arms that differ only in learning rate, seed
or schedule share one compiled graph. The first arm of each memory key runs alone, so it compiles cold and fills
the cache for the arms after it.

### Mixed diagnostic cadence

Diagnostics every 25 steps cost 146 s of a 727 s top-K arm (exp 53), but the windowed rules need dense samples
only inside their windows. With `diag_dense_every: 25` the arm samples every 25 steps inside the windows its
rules read, and every `diag_every` steps elsewhere.
- **Windows:** taken at expansion from every `diag:` atom of the verdicts and early stops that apply to the arm.
  `steps: {at: n}` gives `[n, n]`; a span gives `[from, to]`, clipped to the run.
- **Explicit windows:** windows the spec gives in `diag_windows` are merged with the derived ones. The result is
  stored in the protocol and is part of the arm id.
- **Refused:**
  - an atom with no `steps`: it would read a mix of dense and sparse samples;
  - `diag_windows` without `diag_dense_every`;
  - a rule step outside the schedule, as with a single cadence.

An arm none of whose rules reads a window (a GRU arm beside HSN-only rules, say) keeps the single cadence. It is
recorded as `diag_dense_every: 0`, so its id is the plain arm's. Without `diag_dense_every`, nothing changes.

### The step-time guard

The user's rule (7 Nov): no experiment whose training step takes longer than 0.5 s. Measured at B32:
- compiled dense HSN: 0.244 s;
- compiled top-K: 0.111 s;
- GRUs: 0.004 to 0.017 s;
- eager HSN: 0.73 s (dense) to 1.7 s (top-K).

**At enqueue.** An arm that would be added is refused, and nothing of the spec is queued, when:
- it is an HSN arm on CUDA with `hardware.compile_mode=none` (the variants included), or its `max_step_s` is
  above 0.5, unless its protocol carries `step_time_approval: "<who approved, when>"`;
- the recorded step time of its memory key is above its `max_step_s`, approved or not, since the run-time guard
  would stop it.

The recorded step time is the median of the earlier runs' `step_time.run_median_s`. For records before P1.14 it
is `walls.train_s / (steps − compile steps)`: 0.732 s for exp 46, 0.245 s for exp 48. `enqueue` prints each added
arm's expected s/step. An arm skipped as already queued or done is never refused. GRU and Transformer arms are
never refused for being eager.

**At run time.** Steps that never count:
- the first 20 (warm-up);
- compile steps;
- evaluation steps and the step after each;
- diagnostics steps;
- trainer statistics steps.

Once 50 eligible steps exist, their median is compared with `max_step_s`, once. An arm above it fails early with
its partial record and the verdict `failed: step time 0.73 s > 0.5 s (median of 50 steps, 21-74) @ 74`.

Every record's `step_time` holds:
- the limit and the approval;
- the check: median, steps and n;
- the run's median over every eligible step;
- the exclusion counts.

The constants are module constants (`hsn.experiments.steptime`), as P1.12's stop rule.

### Failed arms

When an arm fails (the P1.12 stop, OOM, the step-time guard, any exception) or is cancelled (SIGTERM), the child
first keeps what existed. Into `result.json`, and into the results row in the same transaction as the final
status, it writes:
- the evaluations, the diagnostics and every completed step's train record (loss, gradient norm, lr; a
  non-finite value stays `null` and its kind is in `nonfinite`);
- the early-stop log and the walls;
- a `failure` block: `kind` (`nonfinite`, `step_time`, `oom`, `cancelled`, `interrupted` or `error`), `reason`,
  the error and the traceback's tail, the stage and the last completed step.

P1.15: a run whose skips included gradient spikes stops with the kind `spike` (the same P1.12 rule). The partial
record keeps `skipped_steps` and `divergence`.

P1.16: a run that keeps diverging after `train.rollback_max` rollbacks stops with the kind `rollback`. With the
protocol's `keep_every`, a run that fails on a guard (`nonfinite`, `spike`, `rollback`) is re-scored from its
newest checkpoint before the first guard event: see [kept checkpoints and partial
results](#kept-checkpoints-and-partial-results).

The status stays `failed` or `cancelled`, so the arm can be queued again. Every verdict that applies reads
`failed: <reason> @ <step>`, which is never a pass and never a finding. A registered early stop keeps its
`stopped early: ...` label and its full record, as before.

### Gradient spikes and divergence dumps

A divergence can be one finite jump. Exp 171 ran with gradient norms of 0.3 to 0.56, then one batch gave
2.2e10 at step 1,269; P1.12's guard skips only non-finite norms, so clipping applied it, and the model broke.
Finite spikes are common in the stored runs. On the 108 stored runs that did not fail, the guard (window 100)
would skip 55 steps in 17 runs at factor 100, 21 steps in 5 runs at 1,000, and 16 steps in 3 runs at 10^4. Many
of those runs learned anyway: gru_2m exp 161 (0.793) spiked 1.5e5 times its median at step 3,791. Others stayed
at chance after early spikes: the gru_2m seeds of exps 122, 123, 144 and 145, and HSN exp 170, whose burst
reached 5e14 at steps 1,345 to 1,363. In exp 171 every step before 1,269 is at most 4.5 times its median, so
every factor from 5 to 6e10 skips step 1,269 and nothing before it. After a first skip the stored norms are not
a counterfactual: a guarded run's later batches meet another model.

**Choosing the factor.** 100 sits at the edge of ordinary variation: it would skip 55 steps in 17 of the 108
runs, learners among them (exps 65, 154, 155 and 161 have steps of 180 to 572 times their median and learned).
Exp 171's jump (7e10) and the peak of exp 170's burst (5e14) are caught at any factor up to about 1e10. For
experiments, 1e3 to 1e4 is the safer choice: only 3 runs skip anything at 1e4. The default stays `None` (off).

**The spike guard** (config, off by default):
- `train.spike_factor` (None: off; 1e3 to 1e4 for experiments, 100 at the edge): skip a step whose finite
  pre-clip global norm exceeds the factor times the median of the last `train.spike_window` (default 100)
  applied norms. The guard decides once min(20, window) norms are known; skipped and non-finite norms never
  enter the history.
- A skip is P1.12's skip: gradients zeroed, no optimiser step, the scheduler advances. It is counted
  (`final.skipped_spike_steps`, None with the guard off), logged, listed in `skipped_steps` with its reason, and
  marked on its train entry (`skipped: spike`, the finite norm kept).
- Spike and non-finite skips count together in P1.12's stop rule (10 consecutive, or more than 1 % with at least
  10); the failure kind is then `spike`.
- On P2.8's compiled path the guard reads the norm the step already copies to the host, after the compiled
  backward: no device synchronisation and nothing changes in the captured graphs. It is refused under an fp16
  `GradScaler`.
- Cost, measured on the RTX 3060 (E1's compiled dense recipe, B32, median of steps 11 to 60, 3 processes each):
  defaults 0.2399 / 0.2391 / 0.2386 s per step against 0.2405 / 0.2395 / 0.2393 on the base commit; with the
  guard at 100, 0.2391 / 0.2388 / 0.2392; with the dump off, 0.2387. All within noise.

**The divergence dump** (`train.divergence_dump`, on by default): at the first step a guard skips (non-finite,
or a spike with the guard on) the trainer writes `divergence_<step>.pt` into the run directory, with a JSON
sidecar. It is a checkpoint at the step before: the parameters, optimiser and scheduler (still pre-step at that
point), and the step snapshot every step keeps (loader position, every generator, copies of the carried state
and the buffers, the counters). It adds the event, the step's windows with their document ids, the resolved
configuration, the versions and per-parameter gradient finiteness. It is 22.5 MB for ref_a1 at B32. The record's
`divergence` names it. A failing write is logged and recorded, and training goes on.

**Replay:**

```bash
python -m hsn.experiments replay --dump RUN_DIR/divergence_00001269.pt [--eager | --compiled] [--repeats 2] [--out FILE]
python -m hsn.experiments replay --exp N            # the experiment's first dump (read-only database)
```

It rebuilds the trainer from the dump's configuration, resumes from the dump, checks the windows it regenerates
against the stored ones, and runs the step again under deterministic kernels (and inductor's deterministic mode
when compiled), as P1.13's CUDA golden test does. It reports:
- the replayed loss and pre-clip norm against the recorded ones and against each other (deterministic replays
  agree bit for bit);
- the raw per-parameter gradient norms and the non-finite parameters;
- eager, with `model.cand_norm=rms`, the RMS of the candidate norm's input (the near-zero hypothesis of exp 171).

`--cpu` replays a CUDA dump in fp32 on the CPU; it is approximate (another device and dtype).

**A real one (8 Nov 2026).** Exp 170/171's arm (HSN d=64 + A4, WSD at 2e-3) was rerun for 1,500 steps with the guard
at 100 (scratch database, P1.15 evidence).
- **Seed 0:** step 908 gave a norm of 2.2e16 (4e16 times its median) at a normal loss (3.88). The guard skipped it
  and wrote the dump (22.5 MB). The next nine batches, with the weights frozen at step 907, gave 1e13 to 3.5e18
  at normal losses, and P1.12's stop ended the run at step 917 (kind `spike`).
- **Seed 1:** ran 1,500 steps without a spike.

**The replay of seed 0's dump:**
- compiled, under the deterministic modes, two replays agree bit for bit and give 2.27e16 (1.3 % from the
  recorded norm);
- eager they give 6.2e13: still a spike, and also identical twice;
- the candidate norm's input RMS is at least 2.41 (median 3.01): the near-zero RMSNorm hypothesis is refuted for
  this event;
- the exploding gradient is spread over the cell's gate, `W_out`, the byte embedding, the encoder and the
  shared transform, at a normal forward: it points at the backward through time, not at one singular operation.

So at that point the weights made many batches' backward explode, not one outlier batch.

### Rollback

P1.15's guard fails fast: a burst ramps over 5 to 40 steps under clipped updates, so by the trigger the weights
have already drifted and every later batch spikes too. Exp 189 lost a model at dev 0.755 that way (8 Nov: a
burst at step 5,214, 25 of 30 steps skipped, P1.12's stop at 5,243, no checkpoint). The rollback (P1.16,
`python/hsn/train/rollback.py`) is PaLM's remedy (restart about 100 steps before the spike and skip the batches
around it) in memory.

**The keys** (config, all off by default; the last four are read only with `rollback_every` set):

| key | default | meaning | recommended |
| --- | --- | --- | --- |
| `train.rollback_every` | `None` (off) | snapshot cadence in steps | 50 |
| `train.rollback_skip` | 0 | extra steps' worth of windows discarded after a rollback | 10 |
| `train.rollback_lr_mult` | 1.0 | learning-rate factor after a rollback, in (0, 1] | 0.5 |
| `train.rollback_lr_steps` | 0 | steps the factor lasts | 200 |
| `train.rollback_max` | 5 | the run stops (failure kind `rollback`) at its `rollback_max + 1`-th trigger | 5 |

**What it does.**
- Every `rollback_every` steps, after an applied optimiser step, the trainer copies every parameter and
  persistent buffer, the optimiser state (AdamW's and LazyAdamW's moments and step counters) and the spike
  history into a ring of two snapshots on the device (buffers allocated once; `torch._foreach_copy_`, no
  synchronisation).
- A step the guards would skip (non-finite, or a spike with `train.spike_factor` set) instead restores the
  **older** snapshot in place, so the rollback reaches back between `rollback_every` and twice that many steps;
  the newer one may sit inside the ramp and is dropped. P1.15's dump is written first, as for a skip.
- The windows between the snapshot and the trigger are never replayed; `rollback_skip` steps' worth more are
  discarded; every loader row starts a fresh document (the carried state resets at the next window's reset); the
  learning rate is cut for `rollback_lr_steps` steps. The step clock, the scheduler and the generators keep
  running, so records stay monotonic.
- A trigger before the first snapshot is P1.15's skip.

**The stop rule** (the user's decision Q2 = A, 8 Nov): a rollback is not a skip. It does not count in P1.12's
rule (10 consecutive, or more than 1 % with at least 10) and resets its run of consecutive skips; its own limit is
`rollback_max`. Skips without a snapshot count as before.

**Records:** `rollbacks` (step, kind, norm, median, restored step, lost steps, windows skipped), `final.rollbacks`
(the count; `None` with the rollback off), the trigger step marked `skipped: rollback` on its train entry, the
TBPTT stats' `rollbacks`. The events and the learning-rate cut are in the checkpoint; the snapshots are not (a
resumed run re-arms at its next multiple of the cadence).

**P2.8's compiled path:** the parameters and buffers are the CUDA Graphs' static inputs and are restored in
place, so their addresses stay valid; the optimiser runs eagerly; the trigger reads the norm the step already
copies to the host. Refused under an fp16 `GradScaler`.

**Cost:** two copies of the parameters and both Adam moments (44 MB for HSN at A.1), and one device copy of about
22 MB every `rollback_every` steps. Measured on the RTX 3060 (8 Nov; E1's compiled dense recipe, B32, bf16,
median of steps 11 to 60, one process each, P1.15's method):

| run | s per step | mean |
| --- | --- | --- |
| P1.15's defaults (the cited baseline; the training path is unchanged since, `git diff 2a25c35 80b893d`) | 0.2399 / 0.2391 / 0.2386 | 0.2392 |
| P1.16, defaults | 0.2400 / 0.2408 / 0.2401 | 0.2403 |
| P1.16, `rollback_every=10` (five snapshots in the 60 steps) | 0.2405 / 0.2402 / 0.2401 | 0.2403 |
| P1.16, kept checkpoints every 20 steps | 0.2401 | |

- **The defaults** have a mean of 0.2403, against P1.15's 0.2392: +0.46 %, about 1.1 ms. Their maximum is 0.2408,
  against a prior maximum of 0.2406 (P1.13 to P1.15 measured 0.2386 to 0.2406). At the defaults the step adds
  only a `None` check, the kept-checkpoint test and the schedule's unused multiplier, so this reads as session
  noise.
- **The rollback's snapshots** cost nothing measurable: a mean of 0.2403, +0.0 % against the defaults.

**On a dump.** `replay --dump FILE --continue-steps N --rollback-every 50 [--rollback-skip …] [--spike-factor F]`
keeps training past a dump with the remedy set on the resumed trainer (the dump's configuration and hash stay
its own). A dump holds only the state at its trigger's step before, inside the burst: its first snapshot is that
state, so a continuation shows whether that state recovers, not an N-step rollback.

**The stored dumps** (8 Nov; compiled, deterministic; 300 steps with the rollback, and again with lr × 0.1 for 200
steps): neither state recovers.
- **Exp 189** (from step 5,213): 5 of the 6 batches that followed spiked (norms 279 to 61,076, against a median
  of 1.04), and the continuation stopped at the rollback limit at step 5,220.
- **Exp 193** (from 818): 5 of 9 spiked (12 to 22, against a median of 0.48).
- **The learning-rate cut** changed nothing: each rollback restores the same weights.
- **What it shows:** the weights one step before the trigger are already inside the burst. So the rollback
  restores the older of two snapshots, between `rollback_every` and twice that many steps back, not the newest.
  Whether that rescues a live run is the next experiment: HSN at β2 0.999 and lr 2e-3 with the rollback on.

### Kept checkpoints and partial results

With `ckpt_every` 9,000 a run that fails at step 5,243 keeps no model. The protocol fields `keep_every` (0: off)
and `keep_last` (2), both execution-only:
- the trainer writes a checkpoint every `keep_every` steps (a harness setting of the trainer: the configuration,
  its hash and the training are unchanged);
- after each step the newest `keep_last` of those stay, and so does the newest one before the run's first guard
  event (a skipped or rolled-back step), which is never pruned;
- with the default `checkpoint: final`, a run that finishes deletes them;
- `checkpoint: retained` keeps them after completion and also pins the checkpoint with the best
  dev accuracy at its exact saved step, even if the guard never fires. Set `keep_every` to the
  evaluation cadence to capture every evaluated peak. `keep_last` bounds the recent checkpoints;
  the best and the pre-guard checkpoint are additional pins. The result records
  `retained_checkpoints` and `best_checkpoint` separately; final scores and verdicts still use the
  final weights. `checkpoint: none` still deletes all checkpoints after completion.

Both `probe-rerun` and `route-check` accept `--checkpoint-step N` to read one retained checkpoint.
It must exist in that experiment's run directory; a missing step is an error, never a fallback to
the final weights. Earlier weights are not compared with stored final probes, and the diagnostic
does not overwrite the experiment's results. `--checkpoint-step` cannot be combined with
`probe-rerun --weights dump`.

When a run fails on a guard (`nonfinite`, `spike`, `rollback`), the newest checkpoint on disk before the first
guard event (before the failure if there was none) is loaded into the model and evaluated (with the split), its
state statistics and the protocol version's probes (summaries) are taken, and the record gets `partial`:
`from_step`, `checkpoint`, `reason`, `final` (the `final` keys of that evaluation), `trained`, `probes` and
`seconds`. A failure while re-scoring is kept as `partial.error`.

The newest saved checkpoint is assessed separately as `latest_saved`, with the same fields. This
shows recent progress even if recovery started much earlier. It is diagnostic only: the run stays
failed, the primary final score stays NULL, and the conservative `partial` verdict is unchanged.
The failure column labels both checkpoint steps. Neither checkpoint is certified healthy by age.

Every verdict of such a record reads `failed: <reason> @ <step> (from step X: <the rule on the kept
checkpoint>)`: still a failure, never a pass and never a finding. Rules can read the kept checkpoint directly with
`partial:<final key>` (for example `partial:split/later/accuracy`) and `partial:from_step`. The report's failure
cell adds `; partial from step X (dev <acc>)`.

For a run stored before P1.16 that left a dump but no checkpoint (exp 189), `probe-rerun --exp N --weights dump`
gives the dump's parameters their probes and the dev evaluation with the split, as evidence (nothing is written to
the database).
- **Exp 189's dump** (step 5,213) scores dev 0.667 (first queries 0.845, later 0.508). It reads state (v1, v2 and
  v3's value edit, follow 0.72) and does not bind keys (swap follow 0.007). That is below its 0.755 at step 5,000,
  because the burst had already cost about 0.09 before the trigger. A checkpoint kept at 5,000 would have kept the
  better model.
- **Exp 193's dump** (step 818) is at chance (0.025).

### route-check: does routing follow the key?

`python -m hsn.experiments route-check --exp N [--split dev|train] [--n-docs 1000] [--cpu] [--out FILE]`
(read-only; `python/hsn/experiments/routing.py`). For a routed model's latest checkpoint it runs every delayed
document from a fresh state in evaluation mode and compares, for every query, the cells selected at the query
key's position with those selected at the asked fact's key position (`o_key`) and value position (`o_val`),
against the document's other facts (`o_other_*`). `d_key` and `d_val` are the differences, with 95 % bootstrap
CIs over documents. The key-swap control uses probe v3's swap documents: routing that follows the key follows it
to its new position (`d_swap`).

Verdict (thresholds of the diagnostic, not frozen metrics): `follows the key` when a CI's lower bound is at least
half a cell (0.5 / K) and the swap control agrees; `weakly key-dependent` when a CI excludes 0; else `does not
follow the key`. `--split train` reads the first training documents instead (keys seen in training, in-sample),
so that "follows trained keys" and "follows keys never seen" can be told apart. A dense model is refused.

**On the stored top-K runs** (8 Nov; their step-9,000 checkpoints; K = 4, so the threshold is 0.125):

| exp | scorer | split | o_key | o_val | o_other | d_key [95 % CI] | d_val [95 % CI] | d_swap | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 196 | linear | dev | 0.41 | 0.57 | 0.51 | −0.140 [−0.166, −0.115] | −0.004 [−0.025, +0.017] | −0.081 | does not follow the key |
| 196 | linear | train | 0.36 | 0.52 | 0.46 | −0.142 [−0.167, −0.117] | −0.022 [−0.044, −0.001] | −0.071 | does not follow the key |
| 197 | linear | dev | 0.33 | 0.35 | 0.34 | −0.027 [−0.047, −0.006] | −0.013 [−0.037, +0.011] | −0.098 | does not follow the key |
| 198 | linear | dev | 0.33 | 0.26 | 0.39 | −0.091 [−0.110, −0.072] | −0.002 [−0.026, +0.021] | +0.031 | does not follow the key |
| 202 | mlp | dev | 0.36 | 0.48 | 0.42 | −0.085 [−0.107, −0.064] | +0.007 [−0.014, +0.028] | −0.075 | does not follow the key |
| 202 | mlp | train | 0.33 | 0.46 | 0.38 | −0.077 [−0.098, −0.057] | +0.012 [−0.009, +0.033] | −0.079 | does not follow the key |
| 203 | mlp | dev | 0.09 | 0.06 | 0.10 | −0.015 [−0.027, −0.003] | +0.008 [−0.002, +0.019] | +0.018 | does not follow the key |
| 203 | mlp | train | 0.08 | 0.05 | 0.10 | −0.011 [−0.024, +0.001] | +0.004 [−0.007, +0.015] | −0.000 | does not follow the key |

- **No run routes by key,** neither on the dev keys nor on the trained keys in-sample, and the mlp scorer does not
  change that.
- **The cells follow the query's position, not its key.** `d_key` is negative: a query overlaps less with its
  fact's key position than with the other facts' positions. The swap control is negative or about 0: when the key
  moves, the query's cells stay with its old position.
- **Refresh probes** fill 15 to 28 % of the slots.

### Recipes

New HSN specs start from `configs/experiments/hsn_recipe_b095.yaml` (P1.16, the user's decision of 8 Nov): HSN
d=64 + A4 (`model.cand_norm=rms`), delayed/in_range, B32, WSD 0.2 over 9,000 steps at lr 2e-3, warm-up 200, clip
1.0, the spike guard at 1e4 and **AdamW β2 0.95**.

```yaml
base:
  config: configs/experiments/hsn_recipe_b095.yaml
  overrides: [profile=rtx3060, hardware.compile_mode=reduce-overhead]
```

- With these two overrides it resolves to exp 194's configuration exactly, so its arms have exp 194's arm id
  (arm ids hash the resolved configuration, not the file).
- β2 0.95 made the lr 2e-3 recipe reliable (exps 194 and 195: reads state, no burst, onset by step 1,000 to
  1,500; β2 0.999: 2 of 5).
- **gru_2m's two bars** (the user's decision, 8 Nov):
  - the final-score bar is gru_2m at 5e-4 with β2 0.999 (0.793, exp 154). At that lr β2 0.95 had no effect
    (exps 199 to 201: 0.771 / 0.776 / 0.787);
  - the learning-speed bar is gru_2m at 2e-3 with β2 0.95: dev ≥ 0.6 by step 2,500 to 3,000 in 2 of 3 runs
    (exps 204 to 206: final 0.786 and 0.776; one stayed at chance, with no burst).
- `train.betas`' default, `configs/ref_a1.yaml`, `g1_ref_a1.yaml` and the baselines are unchanged. A recipe file
  is never edited once a spec uses it: a new recipe gets a new file.

## Rules

Rules are written in the spec before any run, and `enqueue` stores them in the database. They are experiment
rules registered before the run. They are not gate thresholds: those live in `configs/metrics_freeze.json` and
are never changed here.

**Atoms.** An atom compares one measured value with a value:

```yaml
{metric: "diag:grad_share/cell", steps: {from: 400, to: 600, every: 25}, agg: median, op: ">=", value: 0.01}
```

**Metric sources**, written `source:key`:
- step series:
  - `eval:accuracy`, `eval:loss`, `eval:delayed/in_range/accuracy`; P1.15: `eval:split/later/accuracy` (and
    `first`, `later/1` to `later/3`);
  - `diag:<key>`;
  - `train:loss|accuracy|grad_norm`;
- scalars:
  - `trained:<key>`, e.g. `rank_pr/answer`, `cell_spread/answer`, `edge_logit/absmax`;
  - `probe:<key>`, e.g. `correct_n`, `correct_follow_frac`, `correct_unchanged_frac`, `first_query_accuracy`;
  - `probe_v2:<key>` (P1.14, reads_state version 2), e.g. `correct_n`, `correct_follow_frac`, `vocab_size`,
    `matched_frac`;
  - `probe_v3:<key>` (P1.15, reads_state version 3), e.g. `correct_later_n`, `vocab_size`, `value/correct_n`,
    `value/follow_frac`, `value/unchanged_frac`, `swap/correct_n`, `swap/follow_frac`, `swap/unchanged_frac`;
  - `overfit:<key>`;
  - `final:<key>`; P1.16: `final:rollbacks` (`None` with the rollback off);
  - `partial:<key>` (P1.16): the same keys of a failed arm's re-scored kept checkpoint, and `partial:from_step`;
    missing when there is none;
  - `verdict:<rule>`, another rule's label.

**Step selection.** `steps` selects samples: `{at: n}`, or `{from, to, every}` (multiples of `every`). Steps the
cadence does not produce raise at enqueue.

**`min_samples`** (P1.14) sets a minimum number of samples for an atom with a window (`from` and/or `to`).
- At enqueue, a cadence that puts fewer steps in the window raises.
- At evaluation, fewer recorded samples read `n/a: diag:gate/p50 3 samples < min_samples 5`; a step P1.12's guard
  skipped has no diagnostics sample.
- It has no default, so existing rules are unchanged. New windowed rules should set at least 5.
- `memory_open` takes it as a parameter.

**Operators and values:**
- `agg`: `all`, `any`, `median`, `mean`, `min`, `max`, `first`, `last`;
- `op`: `>=`, `>`, `<=`, `<`, `==`, `!=`, or `between: [lo, hi]`;
- `value`: a number, a label, a constant, or `"<constant> ± <number>"`.

**Combining atoms:** `all`, `any` and `not`.

**Verdicts.** `{cases: [{label, when}], default, applies_to: {family, protocol}, group: {min_agree: n}}`. The
first case that holds gives the label, the default otherwise. Special results:
- an arm outside `applies_to` reads `n/a (family gru)` or `n/a (protocol train_eval)`, never a pass or a fail;
- missing data reads `n/a: <metric> missing ...`;
- every verdict of an arm that stopped early reads `stopped early: <outcome> @ <step>`;
- every verdict of an arm that failed or was cancelled reads `failed: <reason> @ <step>` (P1.14).

`group.min_agree` marks a seed group's label as a *finding* in `report` when that many seeds share it.

**Presets**, expanded at enqueue:

| preset | meaning | applies to |
| --- | --- | --- |
| `reads_state` | the audit's verdict on the probes answered correctly before the edit: reads state when ≥ 50 % follow the edit and ≤ 10 % are unchanged; does not read state when < 20 % follow; partial otherwise | every family |
| `reads_state_v2` | P1.14: the same thresholds over probe v2's matched documents answered correctly before the edit, after `no matched probes` (0) and `too few matched probes` (< `min_correct`, 20). P1.15: `min_vocab: 20` adds `vocabulary too small` after `no matched probes` (absent: the expansion is P1.14's). Needs reads_state version 2 | every family |
| `reads_state_v3` | P1.15: probe v3's value edit over the correct later queries: `no correct later queries`, `vocabulary too small` (< `min_vocab`, 20), `too few correct later queries` (< `min_correct`, 20), then v1's thresholds labelled `reads state (later queries)`, `does not read state (later queries)`, `partial (later queries)`. Needs version 3 | every family |
| `binds_keys` | P1.15: probe v3's key swap: `no swap probes`, `too few swap probes` (< `min_correct`, 20), `binds keys` (follow ≥ 0.5 and unchanged ≤ 0.1), `does not bind keys` (follow < 0.2), `partial`. Needs version 3 | every family |
| `memory_writes_run` | P1.16: `writes` when some sliding window `[s - window, s + window]` (centres `s` at the multiples of `every`, 500, from `from_step`, 500, to `to_step`, required; `window` 250) has a median `gate_key` (`gate/p90`, the written gates' 90th percentile) ≥ `gate_min` (0.095), **and** some window has a median `update_ratio/cell` ≥ `update_ratio_min` (0.0027), each with `min_samples` (3); else `no writes`. Also `gate_key` (`gate/p90`, `gate/p50`, `gate/mean`, `gate/write_rate`) and `agg`. Reads the late and top-K learners that `memory_writes`' one early window misses. Calibrated below | hsn |
| `memory_writes` | P1.15: `writes` when, over the diagnostics in `[from_step, to_step]` (1,000 to 1,500), the median `gate/write_rate` ≥ `write_rate_min` (0.025) and the median `update_ratio/cell` ≥ `update_ratio_min` (0.001), each window with `min_samples` (3); else `no writes`. Parameters also `agg`. Provisional (see the caveat below) | hsn |
| `memory_open` | gate p50 ≥ 0.01 and `grad_share/cell` ≥ 0.01 at every multiple of 500 from step 500 (as registered for queue9). Parameters: `from_step`, `to_step`, `every`, `agg`, `gate_p50_min`, `cell_share_min`, `window` (with `to_step`: the median over ±window at each multiple), `min_samples` (P1.14) | hsn |
| `cells_apart` | `rank_pr/answer` ≥ 2.0 and `diversity_live/answer` ≥ 0.2 on the trained model | hsn |

**`memory_writes_run` calibration** (8 Nov 2026, the 34 stored HSN reads_state runs with write diagnostics,
read-only; `evidence/P1.16/calibration/`). A run is a learner when its v1 or v3 verdict reads state (16 runs),
dead when its final dev accuracy is at most 0.05 and it does not read state (9 runs):

| rule (each median over a sliding window with 3 samples) | learners "writes" | dead "no writes" |
| --- | --- | --- |
| P1.16 default: some window `gate/p90` ≥ 0.095, and some window `update_ratio/cell` ≥ 0.0027 | 16 of 16 | 9 of 9 |
| the same, both in one window | 13 of 16 (misses top-K 198, 202, 203) | 9 of 9 |
| some window `gate/write_rate` ≥ 0.02 and some window ratio ≥ 0.0027 | 14 of 16 (misses top-K 198, 203) | 9 of 9 |
| `memory_writes` (median write rate ≥ 0.025 and ratio ≥ 0.001 over 1,000 to 1,500) | misreads 190, 196, 198 | |

- **Why not the write rate.** It cannot separate top-K runs: learner exp 198 peaks at 0.009, while dead exp 192
  reaches 0.017. The written gates' 90th percentile does: top-K learners reach 0.10 to 0.57 and dead top-K runs at
  most 0.088.
- **Why separate windows.** The mlp-scorer learners (exps 202, 203) open their gates early (step 500) and their
  cell's update ratio grows later (1,500 to 2,000).
- **Thin margins.** Dead exp 192's p90 is 0.088 against learner exp 198's 0.102. Dead exp 171's update ratio is
  0.0024 against learner exp 173's 0.0029.
- Like `memory_writes`'s, the update-ratio threshold is calibrated for lr of about 1e-3 to 3e-3.
- For a cadence of 500, set `window: 500` so that each window holds three diagnostics.

**`memory_writes` caveat.** Its thresholds come from the 16 stored runs that record the write rate (exps 128 to
174), and the margins are thin: exp 174, a dead run, writes at 0.0275 against the 0.025 threshold, and the
lowest learner (exp 128) at 0.034. Under Adam `update_ratio/cell` scales with the learning rate, so the 0.001
threshold is calibrated for lr of about 1e-3 to 2.5e-3 only; set `update_ratio_min` for other rates.

**Early stops.** `{name, at_steps, outcome, when, applies_to}`. Each is evaluated right after the evaluation and
diagnostics of each listed step. It may read step series only, and never past its step. Every listed step must
be an evaluation step before the last.

A stopped run still runs the probe and `trained_state`. They are stored under `at_stop`, never used for a
verdict, and are there to check the stop rules themselves.

`grad_share/<module>` is a one-step sample: the bridge resample showed a 7x step-to-step spread. Rules on it
should use a median over a window, not one sample.

## The queue

`enqueue` skips an arm that already has a `queued`, `running`, `done` or `stopped_early` experiment, and lists
what it added and skipped. Two flags change that:
- `--force` queues the arm anyway;
- `--skip-imported` also skips arms that only have an imported run.

Imported runs never block a new run by default, because they ran on old code. `enqueue` notes every match.
Every attempt is a row; `requeue` adds a new one at the back of the queue. `enqueue` refuses arms under
[the step-time rule](#the-step-time-guard) and queues nothing of the spec then.

`worker` launches the oldest queued arm that fits, detached in its own process group. Each arm is a
`run-arm` child process, which writes its own results.

**When an arm fits:**
- `--max-concurrent` (default 3) arms of ours run at most;
- its memory estimate fits the card's free memory minus `--reserve-mib` (1024) and other processes' use;
- no other session's process is on the card.

**Memory estimate:** the row's `--mem-estimate-mib`, else the largest recorded peak of an arm with the same
memory key times 1.1. An arm with no estimate runs alone the first time.

**Timing arms** (`timing: true`) wait until nothing of ours runs and the card and CPU are idle at two looks 60 s
apart. Nothing else starts while one runs. Two rules count as idle:
- the card: no Python process, and nothing of 200 MiB or more;
- the CPU: no other process above 50 % in `top`.

The worker records each run's card status: `idle`, `shared (own: ...)` or `shared (foreign: ...)`. A shared run
is fine for a verdict; its times are not measurements.

**Finishing arms (P1.14).** An arm's child writes its final status and then exits; its CUDA context and inductor's
compile workers go last. Until then the worker still counts that process group as ours:
- it is never foreign (before P1.14, exps 56, 57, 72 and 73 were labelled `shared (foreign: ...)` by our own
  exiting children);
- it counts as `own` for the arms it overlaps;
- it keeps the card from being idle, so a timing arm never starts beside it.

**Starvation:** once the oldest arm has been passed 3 times, younger arms wait until it fits.

**Restarts.** Killing the worker or rebooting is safe:
- a new worker marks `running` rows with a dead, zombie or reused PID `failed`;
- with `--requeue-orphans`, it also queues them again;
- children that are still alive finish their own rows.

**Commands.** `pause` and `resume` stop and restart new launches. `cancel N` cancels a queued experiment, or
sends SIGTERM to a running one that is our `run-arm`.

## Reports and import

`report` reads only the database.

**Rows:**
- by default, one row per seed group;
- `--per-seed`: one row per finished attempt;
- `--format csv` for CSV.

**Filters:** `--sweep`, `--status`, `--family`, `--protocol`, `--imported` / `--no-imported`. With `--spec FILE`,
that spec's verdict rules are evaluated again on the stored records of its arms, with its constants.

**Columns:**
- verdict labels with seed counts;
- final dev accuracy and loss as mean [min, max];
- P1.15: the first/later split, `first_acc`, `later_acc`, `later_1` to `later_3`, and `split_src` (`record`, or
  `derived` / `recomputed` from a backfill file);
- probe fractions; with probe v2 also `v2_follow`, `v2_unchanged`, `v2_vocab` and `v2_matched`; with probe v3
  also `v3_follow`, `v3_unchanged`, `v3_vocab`, `swap_follow` and `swap_unchanged`;
- `write_rate` (the trained model's) and `step_s` (the median step time);
- early stops and failures (`<reason> @ <step>`);
- train, total and compile seconds;
- card and commits.

`status` shows a failed or cancelled row as `failed @ <step>: <reason>`, and (P1.15) the final first/later split.

**Split backfill (P1.15).** Runs stored before P1.15 have no split. `split-backfill [--exp N ...] [--recompute]
[--cpu] --out FILE` opens the database read-only and writes FILE (JSON) and a markdown table next to it:
- every stored reads_state record gets the derived split, the audit's hand reconstruction: first = the probe's
  first-query accuracy, later = (dev × 2,126 − first × 1,000) / 1,126. It is approximate (the probe and the
  evaluation round differently) and has no later slots;
- with `--recompute`, a run whose checkpoint is kept is evaluated again, eager, with the slots: the exact split
  with its later slots, and the recomputed overall accuracy against the stored one.

`report --split-backfill FILE` fills the split columns of those runs.

On 8 Nov 2026 (read-only snapshot of the database, CUDA) the backfill recomputed all 127 stored reads_state runs
with a kept checkpoint. 113 reproduce their stored dev accuracy exactly, and the rest are within 4 answers of it
(eager evaluation against the runs' compiled one). The derived split equals the audit's hand reconstruction
(AUDIT.md, 02:52 and later). The recomputed later-query accuracy is within 2 answers of it for 12 of the 13
runs listed there; exp 173 (d=128 at R*) is 7 answers off. Per later slot, HSN exp 136 scores 0.531 / 0.696 /
0.803, and gru_2m exp 154 0.513 / 0.728 / 0.869.

`import --audit-dir /home/joe/hsn-eff/evidence/EFF` backfills the audit's reads-state JSONs and its overfit
summaries (`hsn/`, `e2/`) as imported runs of the pseudo-sweep `imported_audit`. Per file:
- commits are kept where recorded, otherwise left unknown;
- the card status is kept as written;
- probe counts are recomputed with the partition, and files written before the partition fix are flagged;
- the steps and the diagnostics cadence are inferred from the records.

## Validation (`configs/sweeps/p1_11_validation.yaml`)

The protocol reproduces the verdicts of four audit runs:
- `gru384x2_b32_lr3e-3`, `gru128x1_b32_lr3e-3`: reads state;
- `gru64x1_b32_lr3e-3`: does not read state;
- `hsn_pdg_b32_lr1e-3_bridge`: does not read state, memory open from step 1,000.

Final dev accuracy must fall within ±0.03 of each reference. The bridge also has a compiled twin
(`hardware.compile_mode=reduce-overhead`), checked against the same rule once P2.8 is merged. The results and
the time per experiment are in the P1.11 pull request.

### Training measurement views and observational logging (10 November)

Install optional analytical views into an existing database:

```bash
python -m hsn.experiments.measurements --db ~/hsn-experiments/experiments.sqlite
```

This transaction only replaces the named views. It leaves schema version 2, experiment identities,
raw JSON records, statuses, and queue contents intact, so older workers remain compatible. Views
include future records automatically. `training_measurements` exposes task scope, data hash,
final overall/delayed/later accuracy, binding results, peak memory, and evaluation drawdown.
A failed run's partial score remains separate from the NULL final score. Drawdowns are in
percentage points and reflect only saved evaluation times; a single observation gives NULL.
Overall, delayed-task, and later-query drawdowns have separate columns.

```sql
SELECT exp_id, status, tasks_json, final_delayed_accuracy,
       max_delayed_drawdown_pp, final_later_accuracy, binding_verdict
FROM training_measurements WHERE exp_id >= 325;
SELECT * FROM training_diagnostics WHERE exp_id = 343 ORDER BY step;
SELECT * FROM training_gpu_samples WHERE exp_id = 343 ORDER BY sampled_at_unix;
```

`training_evaluations` and `training_steps` expose recorded time series. The diagnostic view keeps
`diagnostics_json` for less common existing fields. Missing old measurements remain NULL, and
old runs have no fabricated GPU samples. Raw records with invalid JSON are marked invalid in the
summary and omitted from JSON time-series views. These are analytical views, not indexed copies;
large whole-history queries may be slower than querying a small selection of experiments.

Opt-in protocol settings (unchanged defaults are omitted from arm identity):

```yaml
protocol:
  diag_stability: true
  diag_every: 250
  diag_sensitivity_epsilon: 0.001
  diag_sensitivity_position: 64
  diag_sensitivity_horizon: 8
```

At sampled accepted updates, after any projections, `stability/route_churn` measures the fraction
of valid positions whose selected cell set changed since the prior observation on the same fixed
diagnostic window. Slot order and invalid padding do not count. The reference step is saved;
the first observation and the first observation after restarting the observer have NULL churn.
The sensitivity fields measure one fixed-direction perturbation of h, preserving q's offset
from mean(h), over a configurable horizon. Rows containing resets/padding in that horizon are
excluded and the eligible count is saved. The model mode and RNG are restored, and no training
state, gradient, or optimizer update is changed. Diagnostics use eager execution without autocast;
the gain is not a Jacobian spectral norm or a replay of the compiled training backward. Diagnostic
time is charged to existing diagnostic overhead. Latest observations also appear in
`experiments.progress_json.stability` with their sample step; old dev scores retain `dev_step`.

Worker `--telemetry-max-samples 3600 --poll 1` enables a bounded series of device utilization,
device VRAM, and owned process VRAM, stored in `card_json.telemetry`. Samples have timestamps and
last-reported training steps; the latter are approximate alignment. Missing readings are NULL.
Old samples are discarded on reaching the cap and `telemetry_dropped` reports the count. The
actual intervals can exceed the poll time (for example while CPU interference is sampled).
Device utilization includes other processes, and samples cannot identify subsecond dips,
allocator reservation, host RAM, kernel bottlenecks, or energy. Adopted workers report incomplete
watch history. Default workers collect no series.

### A6 keys, compiled diagnostics and the phase report (10 November)

**Recipe keys** (`docs/amendments/A6_feedback_gradient.md`; all opt-in, omitted from arm identity
at their defaults, forward values unchanged):

```yaml
- train.state_grad_clip=4.0                               # per-row limit on dL/dq at every position (dL/dh: / sqrt(N))
- train.feedback_detach=[summary,eps,state,router]        # feedback reads held constant in the backward
- model.address_init=1.0                                  # start value of the scorer's two address columns (A5)
```

**Diagnostics and probes under a compiled trainer.** The state window of every diagnostic sample
used to run the eager routed forward and a batched GPU SVD (1.3 s per pass on `profile:rtx3060`
with the card near 18 %; two passes per sample with an update limiter), and the three end-of-run
probes ran eager too (85 s). Both now replay compiled graphs:

- the state window runs the compiled window's diagnostic variant (`CompiledWindow(...,
  capture=True)`, no gradients), compiled once in `Diagnostics.attach` before training
  (`phases.diagnostics_warm_s`); the participation ratio comes from the Gram matrix;
- the probes pad each batch to the evaluation window's shape and replay it
  (`hsn.experiments.probe.PaddedWindow`; the record's `probe_execution` says `compiled`).

Values agree with the eager ones to rounding (the answers of a probe are argmaxes; a near-tie can
differ). An eager trainer (`hardware.compile_mode=none`) is unchanged. `diag_stability` still
perturbs the state inside an eager forward and keeps its cost.

**Phases.** Every training record carries `phases`: wall-clock intervals (`[unix start, seconds]`)
of each evaluation, diagnostic sample (`[step, unix start, seconds]`) and checkpoint write, and the
start and end of the `setup`, `fit` and `probing` stages. With the worker's card time series
(`--telemetry-max-samples`), each utilisation sample can be attributed:

```bash
python scripts/gpu_phase_report.py --exp 346 --exp 364      # per phase: samples, mean %, share below 50 %
```

A run without `phases` (before 10 November) is split by its reported step against the diagnostic
and evaluation cadences, labelled as an estimate. Device utilisation counts every process on the
card: use a timing arm for a clean trace.
