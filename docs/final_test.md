# The untouched final test set

`configs/data/synthetic_final.yaml` (14 Nov 2026) is the final test set of the hold-out data:
fresh documents of the hold-out tasks with a key alphabet (bytes 32 to 47) that no run has been
trained or scored on. It exists because the dev and test keys of the other sets have been
scored many times while recipes and amendments were chosen.

## Rules

The set is good for **one campaign**. The procedure follows the review amendments of
14 Nov 2026 ("lock the final evaluation campaign"):

1. **Select on the dev data only.** Never choose a recipe, a checkpoint, a seed, a mask or a
   threshold on these documents.
2. **Freeze the plan first.** Write a draft (JSON): `name`; `claims`, each with `id`, `text`
   and `pass` (its condition, in words and numbers); `candidates`, each with `run_dir`,
   `checkpoint` and `reach`; `analysis` (what is computed and how it is read). Then

   ```
   python -m hsn.eval.final_test freeze DRAFT.json --out docs/final_test_plans/NAME.json
   ```

   checks every candidate, records the checkpoints' hashes and writes the frozen plan with
   its own hash. `reach` is the complete receptive field of the candidate's reader in bytes
   (`1 + L (n − 1)` for `L` blocks of span `n`; 0 for a model without a reader).
3. **Commit the frozen plan** before anything is scored.
4. **Run once.**

   ```
   python -m hsn.eval.final_test run docs/final_test_plans/NAME.json
   ```

   scores every candidate of the plan in one go and appends one record to
   `docs/final_test_ledger.jsonl`. Commit the ledger with the change that reports the claim.
5. **The set is then spent.** No plan is frozen or scored on it again. A redesigned claim
   needs a new final set: a new dataset spec with another alphabet and other seeds.

What the tool refuses: a plan without claims, pass conditions or an analysis; a candidate whose
checkpoint is missing; a candidate that may have seen a final byte in training (below); a plan
changed after freezing; a checkpoint that is not the frozen file; a plan that is not committed
(`--allow-uncommitted` exists for tests and is written into the record); any plan once a
campaign is on the ledger.

## Exposure in training

A new alphabet that was never scored is not thereby absent from training. `freeze` reads each
candidate's training stream by role (keys, values, filler, markers, digits, symbols of every
task the run trained on) and refuses the candidate when a final byte occurs in any of them:

- a run trained on the hold-out data alone (`synthetic_hold`: `kv`, `delayed2h`) has never
  seen bytes 32 to 47: it passes;
- a run trained on the five-task mix has seen 32 to 39 as bracket symbols: refused;
- a run trained on text (TinyStories, the story-text facts sets) has seen all of them as
  ordinary text: refused;
- a warm-started run is refused, because the stream of its source is not known to the check.

So this set serves claims about models trained on the hold-out data. A claim about a model
that also trains on text needs a final set whose keys that model's text cannot contain.

## What is reported

| Task | Condition | What the keys are, relative to training |
| --- | --- | --- |
| `kv` | training keys | key bytes seen in training (fresh documents) |
| `kv` | final keys | key bytes absent from every training input |
| `delayed2h` | seen pairs | pairs of training bytes that training may contain |
| `delayed2h` | new pairs | unseen combinations of bytes seen in training |
| `delayed2h` | final bytes | key bytes absent from every training input |

Each condition in total and for the answers beyond the reader's reach, with the answer counts
and a 95 % interval from a bootstrap over documents. "Beyond the reach" says where the
designated source of an answer stands; by itself it does not show that the answer came from
the memory (an answer can still follow from a local duplicate, an earlier answer or a prior).

## Limits

- Documents are in range (each fits one window): nothing here tests recall across a window
  boundary.
- One-byte and two-byte keys only.

The ledger does not exist yet: nothing has been scored on this set.
