# Amendment A2: a recipe search of its own for every method in the gate (draft)

Status: draft of 5 Nov 2026, not adopted. It lands only after HSN's post-A1 step 5(c) passes, that is
when the amended HSN checkpoint trained on delayed alone gets the verdict "reads state" under the
decision rule fixed in `evidence/G1/POST_A1_CHECKS.md` before that test runs (of the dev documents
answered correctly before the edit, at least 50 % follow the edited value and at most 10 % are
unchanged). Until then it is not merged and no gate runs under it.

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
3. **Search horizon at least 3,000 steps** (`--search-steps` default 3,000). The seed runs keep their
   own length (12,000 steps for G1).
4. **The same number of trials for every method.** Every method runs the identical grid, so the
   trial count is equal by construction; a trial that fails or is excluded (non-finite gradient
   norm, memory) still counts against the method's budget, as now. `train.search_trials` equals the
   grid size for every configuration in the comparison.

The proposed grid for the G1 rerun is the learning-rate axis {3e-4, 1e-3, 3e-3} crossed with the
batch axis {16, 32} of the current grid: 6 trials per method. The batch axis stays because it decided
the GRU's search (trial 2, batch 32 at lr 3e-3: mean dev 0.552, delayed 0.703, kv 0.401, against 0.092
and 0.093 for its two batch-16 trials). The constant-schedule trial of the current grid is not carried
over: it logged a non-finite gradient norm for HSN, its GRU run failed after 10 s (`search.json` keeps
only the head of that run's log, not the cause), and it equalled trial 0 for the Transformer. Both
are choices of this draft, open to change before it lands.

The selection rule is unchanged: the best trial on mean dev accuracy over the G1 tasks, then the
highest minimum, then the lowest dev loss; trials with a non-finite logged gradient norm or a peak
memory above the seed stage's per-run budget are excluded and reported.

Measured wall time of one trial at 2,000 steps in the 4 to 5 Nov search (two concurrent runs on the
RTX 3060, pre-A1 model): HSN 59.2 to 59.7 min, GRU 1.6 to 1.7 min, Transformer 1.2 to 1.3 min.
Scaled to 3,000 steps, one HSN trial takes about 90 min, so 6 HSN trials run two at a time take about
3 x 90 min = 4.5 h; the 12 baseline trials add under 30 min. The A1 model's step time is not yet
measured.

## Implementation (not part of this draft's commit)

In `scripts/gate_g1.py`: `MATCHED_TRAIN_KEYS` keeps only the budget keys listed in change 2;
`DEFAULT_TRIALS` becomes the grid above, every trial an absolute override set with no "configuration
as is" trial 0 (a grid value equal to a configuration's own value would otherwise resolve to the same
configuration as trial 0, which the gate refuses); `--search-steps` defaults to 3,000; every
configuration in the comparison sets `train.search_trials` to the grid size. The search table of the
gate report gains the method's own recipe keys per trial so that the absence of inheritance is
visible.

## What it applies to

The G1 rerun on the A1 model and every later gate that compares HSN with its baselines under the
"same search budget" rule. The G1 decision of 5 Nov 2026 (FAIL, PR #22) was taken under the old
search and stands as that record.
