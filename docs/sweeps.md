# Experiment sweeps (P1.11)

Status: P1.11, 7 Nov 2026; follow-ups P1.14, 7 Nov 2026, and P1.15, 8 Nov 2026. The harness turns one YAML file into many experiments,
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

Existing specs expand to the same arms with the same ids.

## Quick start

```bash
python -m hsn.experiments expand  configs/sweeps/example_reads_state.yaml   # arms and ids, no database
python -m hsn.experiments enqueue configs/sweeps/example_reads_state.yaml   # queue them
setsid nohup python -m hsn.experiments worker > ~/hsn-experiments/worker.log 2>&1 &
python -m hsn.experiments status                                            # queue, running, recent
python -m hsn.experiments report --sweep example_reads_state                # one row per seed group
```

**Database location:** `~/hsn-experiments/experiments.sqlite`, here `/home/joe/hsn-experiments/`. Override it
with `--db` or `$HSN_EXPERIMENTS_DB`.

**Run outputs:** `~/hsn-experiments/runs/<sweep>/<run_id>/exp<N>/`. Override with `--root` or
`$HSN_EXPERIMENTS_ROOT`. Each run directory holds:
- `result.json`: the full record, with every probe;
- `run.log`;
- `config.resolved.yaml`;
- the final checkpoint;
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
  only, and P1.14's `max_step_s` and `step_time_approval`, which decide whether an arm may run;
- verdict rules, which can be re-evaluated from the stored record;
- a new configuration key with a default;
- P1.14's `diag_dense_every` and `diag_windows` at their defaults (0 and none), and reads_state's
  `version: 1`. Version 2 and a dense cadence give new ids, since they change what the arm records.

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
  `min_vocab`), `too few correct later queries` (below 20 edited, `min_correct`), then v1's thresholds: reads
  state, does not read state, partial;
- key swap (preset `binds_keys`): `no swap probes`, `too few swap probes` (below 20), `binds keys` (follow ≥ 0.5
  and unchanged ≤ 0.1), `does not bind keys` (follow < 0.2), `partial`.

Batching is v1's (rows truncated after their last scored position, a fresh state per row, chunks by length);
`edit_probe_v3(..., reference=True)` is the per-row loop, equal to it exactly in fp64. The per-probe rows are in
`result.json` (`probes_v3`, `probes_v3_originals`), not in the database. Oracle models in the tests pin the
meaning: a reader that binds keys reads `binds keys`, a key-blind positional reader `does not bind keys`.

**`train_eval`**: train and evaluate, with optional diagnostics and `trained_state`. No probe.

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

The status stays `failed` or `cancelled`, so the arm can be queued again. Every verdict that applies reads
`failed: <reason> @ <step>`, which is never a pass and never a finding. A registered early stop keeps its
`stopped early: ...` label and its full record, as before.

### Gradient spikes and divergence dumps

A divergence can be one finite jump. Exp 171 ran with gradient norms of 0.3 to 0.56, then one batch gave
2.2e10 at step 1,269; P1.12's guard skips only non-finite norms, so clipping applied it, and the model broke.
Finite spikes are common in the stored runs: 17 of the 107 runs that did not fail have steps above 100 times
their running median (the measured would-skip table is in the P1.15 pull request).

**The spike guard** (config, off by default):
- `train.spike_factor` (None: off; recommended 100): skip a step whose finite pre-clip global norm exceeds the
  factor times the median of the last `train.spike_window` (default 100) applied norms. The guard decides once
  min(20, window) norms are known; skipped and non-finite norms never enter the history.
- A skip is P1.12's skip: gradients zeroed, no optimiser step, the scheduler advances. It is counted
  (`final.skipped_spike_steps`, None with the guard off), logged, listed in `skipped_steps` with its reason, and
  marked on its train entry (`skipped: spike`, the finite norm kept).
- Spike and non-finite skips count together in P1.12's stop rule (10 consecutive, or more than 1 % with at least
  10); the failure kind is then `spike`.
- On P2.8's compiled path the guard reads the norm the step already copies to the host, after the compiled
  backward: no device synchronisation and nothing changes in the captured graphs. It is refused under an fp16
  `GradScaler`.

**The divergence dump** (`train.divergence_dump`, on by default): at the first step a guard skips (non-finite,
or a spike with the guard on) the trainer writes `divergence_<step>.pt` into the run directory, with a JSON
sidecar. It is a checkpoint at the step before: the parameters, optimiser and scheduler (still pre-step at that
point), and the step snapshot every step keeps (loader position, every generator, copies of the carried state
and the buffers, the counters). It adds the event, the step's windows with their document ids, the resolved
configuration, the versions and per-parameter gradient finiteness. It is about 30 MB for ref_a1. The record's
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
  - `final:<key>`;
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
| `reads_state_v3` | P1.15: probe v3's value edit over the correct later queries: `no correct later queries`, `vocabulary too small` (< `min_vocab`, 20), `too few correct later queries` (< `min_correct`, 20), then v1's thresholds. Needs version 3 | every family |
| `binds_keys` | P1.15: probe v3's key swap: `no swap probes`, `too few swap probes` (< `min_correct`, 20), `binds keys` (follow ≥ 0.5 and unchanged ≤ 0.1), `does not bind keys` (follow < 0.2), `partial`. Needs version 3 | every family |
| `memory_writes` | P1.15: `writes` when, over the diagnostics in `[from_step, to_step]` (1,000 to 1,500), the median `gate/write_rate` ≥ `write_rate_min` (0.025) and the median `update_ratio/cell` ≥ `update_ratio_min` (0.001), each window with `min_samples` (3); else `no writes`. Parameters also `agg`. Provisional, from 16 stored runs | hsn |
| `memory_open` | gate p50 ≥ 0.01 and `grad_share/cell` ≥ 0.01 at every multiple of 500 from step 500 (as registered for queue9). Parameters: `from_step`, `to_step`, `every`, `agg`, `gate_p50_min`, `cell_share_min`, `window` (with `to_step`: the median over ±window at each multiple), `min_samples` (P1.14) | hsn |
| `cells_apart` | `rank_pr/answer` ≥ 2.0 and `diversity_live/answer` ≥ 0.2 on the trained model | hsn |

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
