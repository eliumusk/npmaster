# Evaluation Protocol

This doc fixes the evaluation contract across internal val and the 9-genome benchmark. Parameters for the two frozen operating points live in `configs/operating_points.yaml`; this doc is about the rules those parameters are chosen under.

## 1. Window-Level Score

Per-window BGC probability vector is `probs ∈ ℝ⁸` (7 BGC types + 1 non-BGC). Three score families are defined:

- `max_type = max(probs[:7])`
- `1_minus_isneg = 1 - probs[7]`
- `fused = 0.5 * max_type + 0.5 * (1 - probs[7])`

Per-window CSV writes `bgc_score = max_type` and `non_bgc_score = probs[7]`, so any family can be recomputed at eval time without re-inference.

`fused` is the default formal family because it has the best AUROC on both internal val and the 9-genome benchmark (diagnosis in git history, doc archived).

## 2. Threshold Selection Policy

- **Split**: internal val only. Test sets are never used for threshold selection.
- **Ambiguous windows**: excluded before threshold fitting.
- **Ground truth**: binary `is_bgc = 1` iff `label == "positive"`.
- **Objective**: F1 via `sklearn.metrics.precision_recall_curve(y_true, scores)` → pick `threshold = argmax(F1)`.

## 3. Region Decoder (v2)

Regions are produced by `scripts/postprocess_v2.py::decode_regions_hysteresis`:

- dual-threshold hysteresis (`start_threshold ≥ extend_threshold`)
- optional `min_support_windows`
- optional `length_cap`
- optional valley-based splitting on sustained weak bridges

Every region is traceable back to concrete window spans — no smoothing, no spline, no coordinate fiction.

## 4. 9-Genome Region-Level Metric (`bench`)

For the 9-genome benchmark the matching rule is **asymmetric 50% overlap**:

- **Recall TP**: a GT BGC is TP if any predicted region satisfies `overlap / gt_len ≥ 0.5`.
- **Precision TP**: a predicted region is TP if any GT BGC satisfies `overlap / pred_len ≥ 0.5`.

This is the `bench` mode in `scripts/evaluate_nine_genome.py`. The old `/data/syh/benchmark/nine_genome/evaluate.py` precision (any overlap > 0) is **deprecated** — too permissive, produces misleadingly high precision on long predictions.

For reviewer-facing tables, `bench` is the only precision number quoted. `loose / strict / iou` modes are also computed by the script but are for internal diagnosis.

### 4.1 Fragmentation Diagnostics

`bench` precision can be inflated if one true BGC is split into many short predictions. Therefore `scripts/evaluate_nine_genome.py` also reports a diagnostic `fragmentation` block:

- `unique_gt_precision_tp`: number of unique GT BGCs covered by precision-TP predictions
- `duplicate_precision_tp`: extra precision TPs caused by multiple predictions hitting the same GT
- `max_preds_per_gt`: worst-case number of precision-TP predictions assigned to one GT
- `dedup_precision = unique_gt_precision_tp / n_pred`
- `dedup_f1`: F1 using normal recall and `dedup_precision`

These diagnostics are not a replacement for the published `bench` metric, but any decoder/model sweep must report them. A configuration that improves raw precision while lowering `dedup_f1` is treated as fragmentation, not a real boundary-quality improvement.

Operational rule (2026-04-19): decoder sweeps must rank by `dedup_f1` (primary) with `bench_f1` as a tiebreaker, and the sweep CSV must include `dedup_precision`, `dedup_f1`, `unique_gt_precision_tp`, `duplicate_precision_tp`, `max_preds_per_gt`, and `median_pred_len`. Sweep CSVs must additionally honor `ranking.min_recall` / `ranking.max_n_pred` gates in their sort order (eligible trials first, ineligible follow). See `scripts/run_precision_constrained_decoder_sweep.py` for the reference implementation. Gating candidate operating points by `dedup_p / bench_p >= 0.75` catches the short-core hack; candidates that fail this gate are diagnostic upper bounds only, never formal operating points.

### 4.2 Length-Bucket Diagnostics

Every no_9gt_v1 report must include length-bucket metrics for GT and predicted regions:

- `<5kb`
- `5-10kb`
- `10-20kb`
- `20-50kb`
- `>=50kb`

The main `defrag_v2_ms10_s099` operating point uses `window=8192`, `stride=2048`, and `min_support_windows=10`, so its minimum practical span is about 26.6 kb. It must not be used as evidence of tight `<5kb` BGC boundary detection. Short-BGC claims require a separate length-bucket report and, if needed, a pre-registered small-BGC operating point selected without 9-genome tuning.

For small-BGC selection, `small_bgc_f1` means the harmonic mean of `<5kb` GT-bucket recall and `<5kb` pred-bucket precision.

## 5. Evaluation Flow

1. Train on internal training split.
2. Select threshold and decoder params on internal val (4 knobs: `score_family`, `start/extend_threshold`, `max_gap`, `length_cap`, `min_support_windows`).
3. Freeze into `configs/operating_points.yaml`.
4. Apply frozen point to 9-genome → report `bench` R / P / F1 + `#pred` + per-genome.

Do **not** tune thresholds directly on the 9-genome report set. Points selected on 9-genome are diagnostic upper bounds only (see `operating-points.md::tuned_v2`).

## 6. no_9gt_v1 No-9-Genome-Tuning Protocol

no_9gt_v1 treats all nine benchmark genomes as final external test only. The full embargo list is documented in `docs/no-9gt-v1.md` and should be encoded in `configs/splits/nine_genome_embargo_v1.json` before no_9gt_v1 execution.

For no_9gt_v1 candidates, the following are forbidden before candidate freeze:

- 9-genome training examples
- 9-genome validation examples
- 9-genome hard-negative or hard-positive mining
- 9-genome threshold fitting
- 9-genome decoder selection
- 9-genome hyperparameter selection
- 9-genome-based early stopping

The current `defrag_v2_ms10_s099` operating point is valid as a lockbox-validated 9-genome result and useful upper-bound target, but it must not be treated as evidence that a future no_9gt_v1 training/data method generalizes without 9-genome tuning. A no_9gt_v1 result is only formal if its model, labels, threshold, and decoder were selected entirely from internal non-9-genome data before the final 9-genome report was run.

no_9gt_v1 reports must also include the length-bucket diagnostics from Section 4.2. If a candidate has a separate small-BGC detector, it must be labelled as `secondary` or `gated_ensemble`; otherwise it must be labelled `none`.
