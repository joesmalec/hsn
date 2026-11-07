# Experiment sweeps (P1.11)

Status: P1.11, 7 Nov 2026. The harness turns one YAML file into many experiments, with no code change per
experiment. Every experiment, past, present and future, is recorded in one SQLite database, which is also the
queue. Code: `python/hsn/experiments/`. Command line: `python -m hsn.experiments <command>` or
`hsn sweep <command>`.

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
  only;
- verdict rules, which can be re-evaluated from the stored record;
- a new configuration key with a default.

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
- gate quantiles.

The probe is batched:
- rows are truncated after the answer position;
- originals and edits share a batch;
- every row gets a fresh state.

It equals the audit's per-document probe exactly in fp64. In fp32 and bf16 a near-tie can flip, because
batching changes rounding; `probe-check --exp N` lists such flips with their top-2 margins.

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
  - `eval:accuracy`, `eval:loss`, `eval:delayed/in_range/accuracy`;
  - `diag:<key>`;
  - `train:loss|accuracy|grad_norm`;
- scalars:
  - `trained:<key>`, e.g. `rank_pr/answer`, `cell_spread/answer`, `edge_logit/absmax`;
  - `probe:<key>`, e.g. `correct_n`, `correct_follow_frac`, `correct_unchanged_frac`, `first_query_accuracy`;
  - `overfit:<key>`;
  - `final:<key>`;
  - `verdict:<rule>`, another rule's label.

**Step selection.** `steps` selects samples: `{at: n}`, or `{from, to, every}` (multiples of `every`). Steps the
cadence does not produce raise at enqueue.

**Operators and values:**
- `agg`: `all`, `any`, `median`, `mean`, `min`, `max`, `first`, `last`;
- `op`: `>=`, `>`, `<=`, `<`, `==`, `!=`, or `between: [lo, hi]`;
- `value`: a number, a label, a constant, or `"<constant> ± <number>"`.

**Combining atoms:** `all`, `any` and `not`.

**Verdicts.** `{cases: [{label, when}], default, applies_to: {family, protocol}, group: {min_agree: n}}`. The
first case that holds gives the label, the default otherwise. Special results:
- an arm outside `applies_to` reads `n/a (family gru)` or `n/a (protocol train_eval)`, never a pass or a fail;
- missing data reads `n/a: <metric> missing ...`;
- every verdict of an arm that stopped early reads `stopped early: <outcome> @ <step>`.

`group.min_agree` marks a seed group's label as a *finding* in `report` when that many seeds share it.

**Presets**, expanded at enqueue:

| preset | meaning | applies to |
| --- | --- | --- |
| `reads_state` | the audit's verdict on the probes answered correctly before the edit: reads state when ≥ 50 % follow the edit and ≤ 10 % are unchanged; does not read state when < 20 % follow; partial otherwise | every family |
| `memory_open` | gate p50 ≥ 0.01 and `grad_share/cell` ≥ 0.01 at every multiple of 500 from step 500 (as registered for queue9). Parameters: `from_step`, `to_step`, `every`, `agg`, `gate_p50_min`, `cell_share_min` | hsn |
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
Every attempt is a row; `requeue` adds a new one at the back of the queue.

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
- probe fractions and early stops;
- train, total and compile seconds;
- card and commits.

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
