# Stage 5c-1 Plan — retrain on proto_core labels, 3 variants only

**Status:** frozen before launch (2026-04-20). This document is the
commitment artifact: changes after launch must be recorded as a
separate post-hoc note and any such deviation invalidates the run.

## Prerequisite

Stage 5c-0 validated `proto_core` as a legitimate internal-val GT proxy
for 9-genome tight-core GT: Spearman rank correlation with 9-genome
dedup_f1 flipped from **−0.80 (region-val)** to **+0.80 (core-val)**;
`tight_pos_099` rose from rank 9 to rank 2 on core-val; `baseline`
dropped from rank 1 to rank 7. See
`experiments/stage5c0_core_decoder_sweep_selected.csv`.

## Scope — exactly 3 variants

| variant id | label rule | rationale |
|---|---|---|
| `baseline_core_v1` | positive if window overlap ≥ 0.5 against proto_core | control — isolates the effect of switching GT from antiSMASH region to proto_core with nothing else changed |
| `tight_core_v1` | positive if window overlap ≥ 0.75 against proto_core | main candidate — 0.75 (not 0.99) because proto_core is already tight (median 2.5 kb); 0.99 risks killing nearly all short-core positives |
| `short_core_upweight_v1` | positive if overlap ≥ 0.5 against proto_core + ×4 loss weight on rows where the overlapping core is <10 kb | short-BGC candidate — tests whether core labels + upweighting finally move <10 kb recall. May still be limited by window=8192 |

No other variants. If Stage 5c-1 fails, do **not** add a fourth
variant under this stage; open Stage 5c-2 with fresh justification.

## Frozen selection rules

- Training inputs: `windows_metadata_core.csv` (rebuilt with proto_core
  overlap), warm-started from `H_hardneg_r10` (same as Stage 5a/5b).
- All training and decoder selection happens against **proto_core
  internal val GT** (`bgc_annotations_core.csv` restricted to val
  split).
- Primary ranking metric: `dedup_f1`.
- Fragmentation gate: **`dedup_precision / bench_precision ≥ 0.95`**
  (stricter than Stage 5b's 0.75, because core predictions should be
  tight enough that duplicates are genuinely rare).
- Tie-breaker order: `bench_precision`, `bench_recall`.
- 9-genome final eval runs on the **top 2** candidates by core-val
  `dedup_f1` only. No re-selection after 9-genome results arrive.

## Out of scope

- Stage 5d (short-window head, `window=2048`). Held until Stage 5c-1
  settles. Short-BGC recall is a separate problem and mixing it with
  the proxy-repair validation would confound interpretation.
- MIBiG positive injection. Held until 5c + 5d together are shown
  insufficient.

## Execution checklist

1. Rebuild `windows_metadata_core.csv` — recompute `max_bgc_overlap_ratio`,
   `nearest_bgc_distance`, `bgc_length_max`, `label` against proto_core
   coords. Keep `hard_negative` flag from stage 3 mining (mined against
   regions but still identifies high-score background windows).
2. Build stage5 label CSVs + patched meta_arrays for each of the 3
   variants via existing `build_stage5_labels.py` + `apply_stage5_meta.py`.
3. Train all 3 on 3 GPUs, same hyperparameters as Stage 5a (warm-start,
   60 epochs, hard_neg_weight 4.0).
4. Dump val predictions for the 3 checkpoints (cached embeddings, ~16 s
   each per Stage 5b-1).
5. Run decoder sweep against `bgc_annotations_core.csv` on the 3 new
   tags; apply the gate.
6. Freeze top-2 in `experiments/stage5c1-selection_decision.md`.
7. 9-genome inference + eval on the 2 winners under their selected
   decoders.
8. Write `docs/stage5c1-results.md` leading with the primary metric
   triple (`dedup_f1`, `bench_precision`, `bench_recall`) and including
   length-bucket recall/precision for `<5kb / 5-10kb / 10-20kb / 20-50kb
   / >=50kb`.

## Success criterion

At least one of the two 9-genome winners hits **`dedup_f1 ≥ 0.40`**
with **`frag_ratio ≥ 0.95`** — matching or exceeding Stage 5b's
diagnostic `tight_pos_099 + core-gate decoder` (0.398) under a
**legitimate selection protocol** (no 9-genome inspection).

Stretch: `dedup_f1 ≥ 0.44`, matching the 9-genome-tuned `defrag_v2`
upper bound.

Failure: both winners stay ≤ 0.35 dedup_f1 on 9-genome, or the gate
can't be met. Escalate to Stage 5d.
