# The untouched final test set

`configs/data/synthetic_final.yaml` (14 Nov 2026) is the final test set of the hold-out data:
fresh documents of the hold-out tasks with a key alphabet (bytes 32 to 47) that no run has been
trained or scored on. It exists because the dev and test keys of the other sets have been
scored many times while recipes and amendments were chosen.

**Rules**

- Never select a recipe, a checkpoint or a seed on it. Choose on the dev data; then score the
  chosen checkpoint here.
- It is scored once per gate claim, with `python -m hsn.eval.final_test RUN_DIR --claim "…"
  --reach N`. `--reach` is the complete receptive field of the model's reader in bytes
  (`1 + L (n − 1)` for `L` blocks of span `n`; 0 for a model without a reader).
- Every scoring appends a line to `docs/final_test_ledger.jsonl`: the claim, the run, the
  checkpoint's hash, the commit of the scoring code and every number. A checkpoint that is in
  the ledger is refused unless `--again "<reason>"` is given, and the reason is recorded.
- Commit the ledger with the change that makes the claim.

**What is reported.** For `kv`: training keys and final keys. For `delayed2h`: seen pairs, new
pairs of seen bytes, and the final bytes. Each in total and for the answers beyond the reader's
reach, with a 95 % interval from a bootstrap over documents. A claim that a memory was used
rests on the answers beyond the reach.

**Limits.** The set is defined for models trained on the hold-out data. For a model trained on
the five-task mix, bytes 32 to 39 are bracket symbols it has seen in another role. Documents
are in range (each fits one window), so nothing here tests recall across a window boundary.

The ledger is empty: nothing has been scored on this set.
