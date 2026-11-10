# A6 results — the A5 accuracy drops, the GPU pauses and the GRU bar (10 Nov 2026)

Branch `claude/a5-fix` (from `codex/a5-norm` at `38123a1`). Every number comes from the canonical
SQLite database `/home/joe/hsn-experiments/experiments.sqlite`; the replay evidence is in
`/home/joe/hsn-experiments/a5-fix/probe/`. Runs are 9,000 steps of the A5 recipe (top-K routing,
key addresses, selected-cell readout, lr 2.5e-3, β2 0.95, WSD) on the delayed task, B32, compiled,
unguarded: no spike guard, no rollback, every document kept.

RESULTS_PLACEHOLDER
