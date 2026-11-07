# P1.11 test fixtures

Three result files of the P1.10 efficiency audit (branch `hsn-efficiency`, worktree `/home/joe/hsn-eff`,
`evidence/EFF/`), copied on 7 Nov 2026 with their content unchanged (re-serialised without indentation), for
`tests/test_p1_11_import.py` and `tests/test_p1_11_rules.py`:

- `reads_state/gru64x1_reads_state.json`: an early reads-state run (GRU 64x1, 6 Nov). It has no commit,
  diagnostics or trained_state, its card was shared, and its "all" counts predate the partition fix (they sum to
  1,025 of 1,000 probes).
- `reads_state/hsn_pdg_b32_lr1e-3_bridge_reads_state.json`: the bridge run (7 Nov, `f8884ec`). It has the commit,
  an absolute config path, diagnostics every 500 steps and trained_state, and its card was idle.
- `hsn/s2v1_cell_embed_linear_head_summary.json`: an overfit-recipe summary (6 Nov, variant
  `cell_embed_linear_head`). Its card was idle.

They are test data, not an evidence directory.
