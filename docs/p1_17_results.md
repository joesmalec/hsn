# Baseline run notes

All figures below come from canonical SQLite `/home/joe/hsn-experiments/experiments.sqlite`.
Historical runs 331–343 use commit `d4bd65bcbd2f0436ffcebbf2efa4174a7808308b`;
logging-validation runs 344–347 use `6fe8e0b4ec62241970952ca175b789ae1cc104ea`.

**Comparison correction (10 November):** GRU and Transformer overall dev scores below combine
five tasks; HSN and A5 are delayed-only. The 27k delayed-task scores are **78.69% GRU** and
**70.23% Transformer**, rather than their 92% overall scores. See the
[training findings report](p1_17_training_findings_2026_11_10.md) for every task-specific score,
the three-seed cap comparison, and the remaining accuracy drops.

**Latest completed queue (14:09 MST, 10 November):** the 200-step logging check and
all three 9k runs finished and were saved to SQLite. Current-norm A5 seed 2 (345)
ended at **2.16% dev / 2.22% later**; anchored-router A5 seed 2 (346) ended at
**77.47% / 62.52%**, but suffered a **36.92 pp** cumulative decline before recovery.
Delayed-only GRU seed 0 (347) ended at **77.85% / 58.35%**, with a **1.27 pp**
worst decline. All three failed key binding. This new anchored repeat supersedes
any suggestion that its temporary declines are limited to the earlier 12 pp.
The task selection and data fingerprint now match across these runs; seeds and
other recipe settings still differ. See the report for diagnostic curves and limits.

## HSN and GRU 9k resource comparison

- HSN exp331 (9,000 steps): peak process GPU memory **8,156 MiB**; median step **208.92 ms**; measured training time **1,920.49 s**.
- GRU exp334 (9,000 steps): peak process GPU memory **710 MiB**; median step **15.96 ms**; measured training time **145.01 s**.
- On these runs the GRU used about **91% less peak GPU memory**, had about **13.1x lower median step time**, and used about **13.2x less measured training time**.

This records the user's observation that the GRU used much less memory and GPU compute than HSN. Treat the factors as run-specific measurements, not universal model ratios. The recipes differ: HSN uses the 64-wide dense HSN recipe on the delayed task; GRU uses the 2M-parameter baseline and the sweep's shared batch override. This comparison is therefore descriptive, not a controlled architecture-only benchmark.

Results so far:
- exp331 HSN 9k: final dev accuracy **79.82%**, later-query accuracy **61.99%**.
- exp332 HSN 12k: final dev accuracy **77.80%**, later-query accuracy **58.08%**.
- exp333 HSN 27k: canceled while queued at the user's request due to HSN run time.
- exp334 GRU 9k: final dev accuracy **81.85%**, later-query accuracy **60.66%**.

The baseline queue completed for the listed GRU and Transformer runs, followed by A5 experiments 340–343. Final metrics and run states are recorded in SQLite.

## Memory growth over a run

The user observed that GRU memory use increases gradually as training proceeds, while HSN reserves most of its GPU memory near startup. This is recorded as an operator observation; the historical baseline records provide start/end/peak memory snapshots rather than a full time series, so they do not independently establish the shape of the memory curve. Runs 344–347 now add process and device VRAM time series, but do not measure host RAM or allocator-reserved memory. Per-run peak memory above is still available in SQLite.

## GPU utilization pattern

The user also observed that HSN has frequent, deeper dips in GPU utilization, while GRU and Transformer runs appear to have fewer and smaller dips. Historical baseline snapshots were too sparse to quantify this. New runs 345–347 sample utilization about once per second: after reported step 200 and before 9,000, about 10.2% of A5 samples fell below 50% utilization, compared with none of the GRU samples. These samples include diagnostics, evaluation, and checkpoint activity; phase profiling is still needed to attribute the dips. No new Transformer trace was collected.

## Remaining baseline results

- GRU exp335, 12k: final dev **83.32%**, later-query **61.72%**.
- GRU exp336, 27k: final dev **92.00%**, later-query **59.86%**.
- Transformer exp337, 9k: final dev **77.60%**, later-query **20.96%**.
- Transformer exp338, 12k: final dev **88.24%**, later-query **23.89%**.
- Transformer exp339, 27k: final dev **92.74%**, later-query **50.71%**.

All completed baseline records and checkpoints are stored under experiment IDs 331, 332, and 334–339. HSN exp333 (27k) was canceled while queued per the user's request.

## A5 paired follow-up (experiments 340–343)

All four A5 follow-up attempts are recorded in canonical SQLite. Seed 0 control (exp342) failed at step 5,748 after gradients became non-finite at step 5,739; its diagnostic dump is saved in the run directory. The run stopped after ten consecutive skipped optimizer steps, as configured. The other three runs completed:

- Seed 2 current reference (exp340): final dev **2.16%**, later-query **2.22%**; peak dev **70.65%** at step 3,500.
- Seed 2 anchored router (exp341): final dev **76.72%**, later-query **57.10%**; peak dev **78.03%** at step 8,500.
- Seed 0 anchored router (exp343): final dev **73.85%**, later-query **57.82%**; peak dev **73.85%** at step 9,000.

All three completed runs reported no optimizer skips or rollbacks. These two seeds show a large difference between the current reference and anchored-router settings, but the seed 0 reference failed before its planned endpoint, and the result is not enough to establish a reliable general advantage. The control runs' strong declines and the captured non-finite event warrant examining the saved divergence diagnostics and checkpoint trajectory before drawing a cause.
