# Baseline run notes

All figures below come from canonical SQLite `/home/joe/hsn-experiments/experiments.sqlite` and run records on commit `d4bd65bcbd2f0436ffcebbf2efa4174a7808308b`.

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

The user observed that GRU memory use increases gradually as training proceeds, while HSN reserves most of its GPU memory near startup. This is recorded as an operator observation; the current experiment records provide start/end/peak memory snapshots rather than a full time series, so they do not independently establish the shape of the memory curve. Per-run peak memory above is still available in SQLite.

## GPU utilization pattern

The user also observed that HSN has frequent, deeper dips in GPU utilization, while GRU and Transformer runs appear to have fewer and smaller dips. The current sweep records utilization at coarse worker snapshots, not a sufficiently dense time series to quantify dip frequency or depth. Treat this as an operator observation pending sampled utilization traces.

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

Neither completed run reported optimizer skips or rollbacks. These two seeds show a large difference between the current reference and anchored-router settings, but the seed 0 reference failed before its planned endpoint, and the result is not enough to establish a reliable general advantage. The control runs' strong declines and the captured non-finite event warrant examining the saved divergence diagnostics and checkpoint trajectory before drawing a cause.
