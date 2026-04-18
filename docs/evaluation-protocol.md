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

## 5. Evaluation Flow

1. Train on internal training split.
2. Select threshold and decoder params on internal val (4 knobs: `score_family`, `start/extend_threshold`, `max_gap`, `length_cap`, `min_support_windows`).
3. Freeze into `configs/operating_points.yaml`.
4. Apply frozen point to 9-genome → report `bench` R / P / F1 + `#pred` + per-genome.

Do **not** tune thresholds directly on the 9-genome report set. Points selected on 9-genome are diagnostic upper bounds only (see `operating-points.md::tuned_v2`).
