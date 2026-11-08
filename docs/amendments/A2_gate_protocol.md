# Amendment A2: a recipe search of its own for every method in the gate

Status: **adopted with P1.16 (8 Nov 2026)**, implemented in `scripts/gate_g1.py`; no gate has run
under it. Drafted 5 Nov 2026 (commit 4a93278, branch `a2-gate-protocol`) to land once HSN trained on
delayed alone reads state. That condition is met: on the P1.11 harness, exps 136, 138, 139, 190,
194 and 195 get the verdict "reads state". The 8 Nov revision updates the evidence, the grid, the
horizon and the cost (sections below); the draft's principles (changes 1 to 4) are unchanged.

Amends the gate protocol of P1.9 (`scripts/gate_g1.py`, stage 5 "Recipe search on dev":
`DEFAULT_TRIALS`, `MATCHED_TRAIN_KEYS`, `--search-steps`) and makes explicit what the plan's "same
hyperparameter search budget" (P0.5 "Context you need"; P1.9 step 4) means. It changes no metric, no
threshold, no task and no seed count: `configs/metrics_freeze.json` and the frozen G1 task set are
untouched.

## Cause

The G1 search of 4 to 5 Nov 2026 (PR #22) gave every method 4 trials of 2,000 steps on seed 0
(`train.search_trials: 4`, `--search-steps` default 2000). Trial 0 was the experiment configuration;
trials 1 to 3 changed one factor each: `train.lr=1e-3`; `train.batch=32 train.micro_batch=32`;
`train.schedule=constant train.warmup_steps=0`. The GRU and Transformer configurations inherited
`MATCHED_TRAIN_KEYS` from the HSN experiment configuration, recipe keys included: lr 3e-3 (chosen for
HSN from its 64-document overfit) and warmup 200 in place of their own lr 1e-3 and warmup 100 (weight
decay 0.01, the cosine schedule and gradient clip 1.0 are the same in all three configurations). So
for the Transformer the search tried lr 3e-3 at batch 16 and 32 and lr 1e-3 only at batch 16; lr
1e-3 at batch 32 and lr 3e-4 at any batch never ran for any method.

What the search saw (`evidence/G1/G1_REPORT.md` section 9, mean dev accuracy over kv and delayed at
2,000 steps):

| Transformer trial | dev delayed | dev kv | mean |
| --- | --- | --- | --- |
| 0: inherited recipe, lr 3e-3, batch 16 | 0.020 | 0.160 | 0.090 |
| 1: lr 1e-3, batch 16 | 0.018 | 0.163 | 0.091 |
| 2: inherited lr 3e-3, batch 32 (selected) | 0.030 | 0.165 | 0.097 |
| 3: inherited lr 3e-3, constant schedule, no warmup | 0.020 | 0.160 | 0.090 |

The selection picked trial 2 over trial 1 on a difference of 0.006 in mean dev accuracy; all four
trials sit at the chance floor of delayed (1/48 = 0.0208) and at the majority floor of kv (0.160).

What the learning rate does (`evidence/G1/DIAGNOSIS.md` part II.6, worktree of branch
`g1-diagnosis`; runs of `scripts/g1_harness_dense_supervision.py` on that branch): the Transformer
baseline `transformer_2m` on the same harness, batch 32, bf16, its own recipe otherwise (cosine, 100
warmup steps, weight decay 0.01, gradient clip 1.0), seed 0, 3,000 steps, task loss mask, dev answer
accuracy over all queries:

| Transformer, batch 32 | step 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 |
| --- | --- | --- | --- | --- | --- | --- |
| lr 3e-3, mixed kv + delayed (the gate's stream): delayed | 0.021 | 0.020 | 0.019 | 0.028 | 0.025 | 0.025 |
| lr 1e-3, mixed kv + delayed (the gate's stream): delayed | 0.021 | 0.020 | 0.070 | 0.086 | 0.119 | 0.119 |
| lr 3e-3, delayed alone | 0.056 | 0.046 | 0.038 | 0.054 | 0.060 | 0.056 |
| lr 1e-3, delayed alone | 0.243 | 0.303 | 0.321 | 0.327 | 0.353 | 0.357 |

On the mixed stream kv ends at 0.162 with lr 3e-3 and 0.178 with lr 1e-3. At lr 3e-3 the Transformer
stays at the floor on both streams; at lr 1e-3 it learns, and on the gate's mixed stream it starts to
move only after 1,500 steps (0.020 at 1,000, 0.070 at 1,500). A 2,000-step search horizon ends 500
steps after that onset, and the one-factor-at-a-time grid never paired lr 1e-3 with batch 32. The
search therefore could not tell a method that does not learn from a recipe that was never tried.

### What 7 and 8 Nov added (P1.11 harness, delayed/in_range alone, B32 for every method)

- **Each method has its own learning-rate optimum, and they differ by a factor of four.**
  - gru_2m: 5e-4 on WSD over 9,000 steps (exp 154: dev 0.793; exp 186 repeats it at 0.784).
    At 1e-3 it stays at chance (exp 153, 0.020); at 2.5e-3 it bursts (exps 176 to 184) or ends
    lower (exp 143, 0.776).
  - HSN + A4 (`model.cand_norm=rms`): 2e-3 to 2.5e-3 on WSD.
    - At 2.5e-3: exps 136 / 138 / 139 reach 0.791 / 0.763 / 0.557.
    - At 2e-3 with β2 0.999: exps 172 and 190 reach 0.784 and 0.782.
    - At 2e-3 with β2 0.95: exps 194 and 195 reach 0.732 and 0.783.
  - The Transformer: 1e-3 (`DIAGNOSIS.md` II.6, above).
  - A grid that misses an optimum by a factor of two is a different method, not a worse one.
- **A short cosine horizon hides late onsets** (Block A, 7 Nov).
  - On a 3,000-step cosine schedule "no learning rate qualifies" for the GRU ladder, because the
    schedule shrinks the learning rate before the onset.
  - On 9,000 steps the same GRU (128 x 1) reads state at 2e-3 (0.756) and 2.5e-3 (0.767), and is at
    0.62 and 0.74 by step 3,000.
  - gru_2m's onset at its optimum comes at 3,500 to 4,500 steps (exp 154). No search of 3,000
    steps can see it learn.
  - The warmup-stable-decay schedule (P1.13's `train.schedule=wsd`) holds the peak learning rate
    until the last 20 %.
- **β2 is a recipe key, and it matters for HSN only.**
  - With AdamW β2 0.95 (large-model practice), HSN's 2e-3 recipe is reliable: exps 194 and 195
    read state with no burst and an onset by step 1,000 to 1,500. With β2 0.999 the same recipe met
    the same criteria in 2 of 5 runs (exps 170, 171, 172, 189 and 190).
  - gru_2m with β2 0.95 at its optimum shows no effect: exps 199 to 201 reach 0.771 / 0.776 /
    0.787, against 0.793 and 0.784.
  - So each method's own configuration sets β2: HSN's recipe file
    (`configs/experiments/hsn_recipe_b095.yaml`) uses 0.95, and gru_2m keeps 0.999.
- **Compiled HSN costs 0.240 s per step** (P2.8, `profile:rtx3060`, d = 64, B32), not the eager
  rate of the 4–5 Nov search; gru_2m costs 0.0167 s (eager cuDNN).

## The change

1. **A learning-rate search for every method.** HSN, the GRU and the Transformer each get their own
   learning-rate search over a grid of at least {3e-4, 1e-3, 3e-3}. Every grid value is an absolute
   override applied to the method's own configuration; no method's trial is "the other method's
   recipe".
2. **No recipe keys inherited across methods.** `lr`, `weight_decay`, `warmup_steps`, `schedule` and
   `grad_clip` come from each method's own configuration (or from its grid), never from the
   experiment configuration of another method. The keys that stay matched across methods are the
   budget keys: `steps`, `batch`, `micro_batch`, `unroll`, `search_trials`, the search horizon and
   seed, and the evaluation cadence (`eval_every`, `ckpt_every`, `stats_every`). `batch` and
   `micro_batch` count as budget keys because they set the tokens seen per step, i.e. the matched
   training compute of P0.5.
3. **Search horizon of 9,000 steps** (`--search-steps` default 9,000; the draft's "at least 3,000"
   is not enough, above). The seed runs keep their own length (12,000 steps for G1).
4. **The same number of trials for every method.** Every method runs the identical grid, so the
   trial count is equal by construction; a trial that fails or is excluded (non-finite gradient
   norm, memory) still counts against the method's budget, as now. `train.search_trials` equals the
   grid size for every configuration in the comparison.

**The grid** (revised 8 Nov, the user's decision Q6 of P1.16):
- **Learning rate:** {3e-4, 5e-4, 1e-3, 2e-3, 3e-3}, the same for every method. This brackets the
  three measured optima (5e-4, 1e-3, 2e-3 to 2.5e-3) with one point on either side.
- **Schedule:** every trial sets `train.schedule=wsd` and `train.decay_fraction=0.2`, so the
  learning rate holds through a late onset.
- **Trials:** 5 per method, each an absolute override set.
- **β2:** each method's own configuration sets it. It is not searched, because its effect is
  method-specific (above).
- **Batch:** a budget key (change 2), matched at the experiment configuration's value and not
  searched.
  - The 7 and 8 Nov evidence is all at batch 32, so the configuration of a rerun should set 32.
  - The draft's batch axis {16, 32} is dropped. Searching it would vary the matched training
    compute, and at a horizon that sees the onsets it no longer decides a search.
- **Dropped from the draft:** the constant-schedule trial of the 4–5 Nov grid stays out. It logged
  a non-finite gradient norm for HSN, its GRU run failed after 10 s, and it equalled trial 0 for the
  Transformer.

The selection rule is unchanged: the best trial on mean dev accuracy over the G1 tasks, then the
highest minimum, then the lowest dev loss; trials with a non-finite logged gradient norm or a peak
memory above the seed stage's per-run budget are excluded and reported.

**Cost** at the measured step times, one arm at a time on the RTX 3060 (the timing rule of 7 Nov):
- HSN: 5 trials x 9,000 steps x 0.240 s = 3.0 h;
- gru_2m: 5 x 9,000 x 0.0167 s = 13 min;
- the Transformer: measured at its first trial. The 4–5 Nov search gave 1.2 to 1.3 min per 2,000
  steps at batch 16 with two concurrent runs.
- Evaluation and probes add a few minutes per trial.
- The seed stage that follows (five seeds of 12,000 steps per method) costs about 4 h for HSN at
  the same rate.
- This replaces the draft's estimate of 90 min per HSN trial, which came from the uncompiled 4–5 Nov
  runs.

## Implementation (P1.16)

In `scripts/gate_g1.py`:
- `MATCHED_TRAIN_KEYS` keeps only the budget keys of change 2: `steps`, `batch`, `micro_batch`,
  `unroll`, `eval_every`, `ckpt_every` and `stats_every`.
- **`search_trials`.** Every configuration in the comparison, HSN's included, gets
  `train.search_trials=<grid size>` as a search override (`search_overrides`).
  - The configuration files are not edited: `configs/experiments/g1_ref_a1.yaml` and the baseline
    configurations are the bases of the sweep specs, and editing them would change their arm ids.
- `DEFAULT_TRIALS` is the grid above: each trial an absolute override set.
- **No implicit trial 0.** `trial_sets` gives the grid as it is, and `--trial` sets give the whole
  grid.
  - A grid value equal to a configuration's own value would otherwise resolve to the same
    configuration as trial 0, which the gate refuses.
- `--search-steps` defaults to 9,000.
- **The recipe is visible.** Every trial in `search.json` carries `recipe`: the method's own recipe
  keys as resolved (lr, betas, weight decay, warm-up, schedule, decay fraction, gradient clip). The
  log prints a search table per method with them.
- **Tests.** Two P1.9 tests changed with the protocol:
  - the end-to-end fixture's `--trial` list;
  - the search-budget assertion of `test_p1_9_experiment_config_matches_ref_a1`.

## What it applies to

Every gate that compares HSN with its baselines under the "same search budget" rule: the G1 rerun,
and later gates. The G1 decision of 5 Nov 2026 (FAIL, PR #22) was taken under the old search and
stands as that record.

**No G1 run now.** The user's rule "don't rerun the G1 gate" stands.
- G1 needs ≥ 95 % on in-range delayed (`configs/metrics_freeze.json`). Every model trained since
  sits at or under the task's key-blind ceiling of 0.800 overall (0.623 on later queries).
- The dev keys never occur in training, and probe v3's key swap reads "does not bind keys" on every
  run. So a rerun fails by construction until a model binds keys it has never seen.
- That is the subject of amendment A5 (draft, P1.17). The HSN configuration of a rerun (A4, the β2
  0.95 recipe) is the user's decision when one is scheduled.
