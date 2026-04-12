# WS0-T2: Metric Contract Inventory
**Date:** 2026-04-09
**Task ID:** WS0-T2

## Score Contract Mismatch (Critical Finding)

**Stage 3 binary BGC score** and **Stage 4 genome-scanning decision** use different signals from the same 8-output head:

| Aspect | Stage 3 (`evaluate.py:75`) | Stage 4 (`run_stage4.py:104,173`) |
|---|---|---|
| **Score used** | `bgc_score = probs[:, :7].max(axis=1)` | `pred[7] < is_neg_thr` (i.e., `is_negative` sigmoid output) |
| **Semantics** | "How confident is the max BGC-type head?" | "Is the is_negative head below a threshold?" |
| **Direction** | Higher = more BGC-like | Lower `is_negative` = more BGC-like |
| **Threshold** | No threshold; used as continuous score for AUROC | Learned from val PR-curve on `is_negative` column (step 4.1, line 79) |

**Why this matters:** The model's `bgc_binary_auroc` (Stage 3) is computed using `max(probs[:, :7])` as the ranking score, but Stage 4 decides whether a window is BGC by checking `probs[7] < threshold`. These two quantities are **not equivalent**:

- A window could have `max(probs[:, :7]) = 0.6` (decent BGC score in Stage 3) but `probs[7] = 0.3` (above a typical threshold like 0.128, so classified as negative in Stage 4).
- Conversely, `probs[7]` could be very low (strong BGC call) while all 7 type heads are individually low (low Stage 3 binary score).

The correlation between `max(probs[:, :7])` and `1 - probs[7]` depends entirely on how well the multi-label heads are calibrated relative to the `is_negative` head. Since BCE trains them independently, there is no guarantee they agree.

**Additional MIBiG inconsistency:** Step 4.4 (`run_stage4.py:465`) uses a **hardcoded** `is_negative < 0.5` threshold for MIBiG detection rate, ignoring the val-optimized threshold from Step 4.1.

## Metric Inventory Table

| # | Metric | Code Location | Formula | Split | Ambiguous? | Level | Checkpoint? | Stage 4? |
|---|---|---|---|---|---|---|---|---|
| 1 | `per_type_auroc` | `evaluate.py:63` | `roc_auc_score(labels[:, i], probs[:, i])` for i in 0..6 | val / test_id / test_ood | Excluded (dataset.py:28) | Window | No | No |
| 2 | `per_type_ap` | `evaluate.py:64` | `average_precision_score(labels[:, i], probs[:, i])` for i in 0..6 | val / test_id / test_ood | Excluded | Window | No | No |
| 3 | `per_type_f1_at_05` | `evaluate.py:65-68` | `f1_score(labels[:, i], (probs[:, i] >= 0.5))` for i in 0..6 | val / test_id / test_ood | Excluded | Window | No | No |
| 4 | `mean_auroc` | `evaluate.py:70-71` | `mean(per_type_auroc[i] for non-NaN i)` | val / test_id / test_ood | Excluded | Window | **YES** (train.py:209) | No |
| 5 | `bgc_binary_auroc` | `evaluate.py:74-76` | `roc_auc_score(1 - labels[:, 7], max(probs[:, :7], axis=1))` | val / test_id / test_ood | Excluded | Window | No (logged only) | No |
| 6 | `bgc_binary_prauc` | `evaluate.py:77` | `average_precision_score(1 - labels[:, 7], max(probs[:, :7], axis=1))` | val / test_id / test_ood | Excluded | Window | No | No |
| 7 | `per_phylum_auroc` | `evaluate.py:80-88` | Per-phylum `roc_auc_score` on `bgc_score` (min 10 samples) | val / test_id / test_ood | Excluded | Window | No | No |
| 8 | `dann_phylum_accuracy` | `evaluate.py:91` | `mean(dom_pred == phyla)` | val / test_id / test_ood | Excluded | Window | No | No |
| 9 | `L_cls` | `train.py:136` | `0.5 * (BCE(pred1, labels) + BCE(pred2, labels))` with pos_weight | train | Excluded | Window | No (loss) | No |
| 10 | `L_con` | `train.py:142` | `MultiLabelSupCon(z_cat, lbl_cat, phy_cat)` | train | Excluded | Window | No (loss) | No |
| 11 | `L_dann` | `train.py:147-149` | `0.5 * (CE(dom1, phyla) + CE(dom2, phyla))` | train | Excluded | Window | No (loss) | No |
| 12 | `L_total` | `train.py:153` | `alpha * L_cls + beta * L_con + gamma * L_dann` | train | Excluded | Window | No (loss) | No |
| 13 | 4.1 optimal thresholds | `run_stage4.py:77-87` | Per-column PR-curve, argmax F1 on val (non-ambiguous) | val | **Excluded** (line 71) | Window | No | Yes (threshold source) |
| 14 | 4.1 `is_negative` threshold | `run_stage4.py:104` | `pred[7] < is_neg_thr` to classify as BGC | test_id | Excluded (line 93) | Window | No | **YES** (binary call) |
| 15 | 4.2 region recall | `run_stage4.py:296` | `n_matched / n_antismash_bgcs` | test_id / test_ood | Excluded (line 148) | **Region** | No | Yes |
| 16 | 4.2 region precision | `run_stage4.py:299` | `n_matched / n_predicted_regions` | test_id / test_ood | Excluded | **Region** | No | Yes |
| 17 | 4.2 type concordance | `run_stage4.py:300` | `type_concordant / type_matched_count` (among matched regions only) | test_id / test_ood | Excluded | **Region** | No | Yes |
| 18 | 4.2 per-type recall | `run_stage4.py:301-303` | `type_matched[c] / type_total[c]` per antiSMASH type | test_id / test_ood | Excluded | **Region** | No | Yes |
| 19 | 4.2 overlap matching | `run_stage4.py:220-225` | `overlap / min(antiSMASH_len, pred_region_len) >= 0.5` | test_id / test_ood | Excluded | **Region** | No | Yes |
| 20 | 4.2 novel candidate count | `run_stage4.py:253-278` | Predicted regions not matched to any antiSMASH BGC | test_id / test_ood | Excluded | **Region** | No | Yes |
| 21 | 4.2 novel confidence | `run_stage4.py:264-267` | high: `max_prob > 0.8 AND length > 10kb AND dist > 20kb`; medium: `max_prob > 0.5 AND length > 5kb`; else low | test_id / test_ood | Excluded | **Region** | No | Yes |
| 22 | 4.4 MIBiG detection rate | `run_stage4.py:465-466` | `sum(preds[:, 7] < 0.5) / n_mibig` | MIBiG external | N/A | Window | No | Yes |
| 23 | 4.4 MIBiG per-type AUROC | `run_stage4.py:475-476` | `roc_auc_score(labels_7[:, j], preds[:, j])` (min 10 pos) | MIBiG external | N/A | Window | No | Yes |
| 24 | 4.4 MIBiG per-type recall@0.5 | `run_stage4.py:477` | `(preds[:, j] > 0.5) & (labels == 1)).sum / n_pos` | MIBiG external | N/A | Window | No | Yes |
| 25 | 4.4 MIBiG type concordance | `run_stage4.py:487-491` | `argmax(preds[i, :7])` in `mibig[i]["categories"]` / n_mibig | MIBiG external | N/A | Window | No | Yes |
| 26 | Error: BGC-level miss rate | `error_analysis.py:219-223` | BGC detected if ANY positive window has `is_neg < threshold` | test_id | Excluded (line 157) | **BGC** | No | Diagnostic |
| 27 | Error: miss rate by length bucket | `error_analysis.py:215-242` | Breakdown into <5kb, 5-10kb, 10-20kb, 20-50kb, >50kb | test_id | Excluded | **BGC** | No | Diagnostic |
| 28 | Error: miss rate by type | `error_analysis.py:246-274` | Per-class, same detection criterion | test_id | Excluded | **BGC** | No | Diagnostic |
| 29 | Error: window miss by overlap ratio | `error_analysis.py:308-333` | Positive windows bucketed by overlap ratio; miss = `is_neg >= thr` | test_id | Excluded | Window | No | Diagnostic |
| 30 | Error: FN score distribution | `error_analysis.py:337-368` | Distribution of `is_neg` scores for false-negative positive windows | test_id | Excluded | Window | No | Diagnostic |
| 31 | Error: miss rate by truncation | `error_analysis.py:374-390` | Complete vs truncated BGCs | test_id | Excluded | **BGC** | No | Diagnostic |
| 32 | Error: per-BGC best is_neg score | `error_analysis.py:394-409` | `min(is_neg_score)` across all positive windows of a BGC | test_id | Excluded | **BGC** | No | Diagnostic |

## Ambiguous Window Handling Map

| Evaluation Path | Ambiguous Excluded? | Where? |
|---|---|---|
| Stage 3 training (BGCDataset) | **YES** | `dataset.py:27-28` -- `mask & (label != "ambiguous")` |
| Stage 3 evaluation (`evaluate.py`) | **YES** | Via BGCDataset (same filter) |
| Stage 4 step 4.1 threshold selection | **YES** | `run_stage4.py:71` -- `r["label"] != "ambiguous"` |
| Stage 4 step 4.1 test_id confusion | **YES** | `run_stage4.py:93` -- `r["label"] != "ambiguous"` |
| Stage 4 step 4.2 genome scanning | **YES** | `run_stage4.py:148` -- `r["label"] != "ambiguous"` |
| Stage 4 step 4.4 MIBiG | **N/A** | MIBiG entries are all genuine BGCs; no ambiguous concept |
| Error analysis | **YES** | `error_analysis.py:157` -- `r["label"] != "ambiguous"` |

**Critical observation:** Stage 4 step 4.2 genome scanning (line 148) excludes ambiguous windows. This means ambiguous windows -- which in real genome scanning would be present and scored -- are silently dropped. In a true deployment, there would be no `label` column to filter on. The current evaluation overstates precision because ambiguous windows (partially overlapping BGCs) that might produce false positives are never scored.

## Hidden Assumptions

### 1. Checkpoint selection uses `mean_auroc`, not `bgc_binary_auroc`
- **Location:** `train.py:209` -- `if val_metrics["mean_auroc"] > best_val`
- **Impact:** Best checkpoint is chosen by average per-type AUROC across the 7 BGC types. This favors balanced type classification, not binary BGC detection. A model that detects BGCs well but confuses types could be discarded in favor of one with better type separation but worse binary detection.

### 2. `bgc_binary_auroc` uses `max(probs[:, :7])`, not `1 - probs[:, 7]`
- **Location:** `evaluate.py:75`
- **Impact:** The binary AUROC metric does NOT measure the discriminative power of the `is_negative` head that Stage 4 actually uses. It measures the max-over-types signal instead. High `bgc_binary_auroc` does NOT guarantee good Stage 4 recall.

### 3. Region merging uses a 3kb minimum length filter
- **Location:** `run_stage4.py:183,191,194` -- `cur_region["end"] - cur_region["start"] >= 3000`
- **Impact:** Predicted regions shorter than 3kb are silently discarded. This could miss small BGCs (e.g., some RiPPs < 5kb). The filter is applied three times: when starting a new region after a gap, when closing a region at a non-BGC window, and at end of contig.

### 4. Overlap matching uses `min(len)` denominator, not Jaccard
- **Location:** `run_stage4.py:221` -- `ov / min_len`
- **Impact:** `overlap / min(antiSMASH_len, pred_region_len) >= 0.5`. If a predicted region is much larger than the true BGC, the overlap fraction can be high even though Jaccard would be low. This inflates recall for over-predicted regions.

### 5. Type concordance only counts matched regions
- **Location:** `run_stage4.py:230-233`
- **Impact:** Type concordance is `concordant / matched_count`, not `concordant / total_antismash`. Undetected BGCs are excluded, so type concordance can look good even if recall is poor.

### 6. `primary_type` is argmax of per-window max-pooled probabilities
- **Location:** `run_stage4.py:198` -- `max(CLASSES, key=lambda c: reg["probs"][c])`
- **Where probs come from:** `run_stage4.py:181` -- per-window max pooling: `max(cur_region["probs"][c], float(pred[j]))`
- **Impact:** Each type's region-level score is the max across all constituent windows. The "primary type" is then the type with the highest such max. This favors types that spike in any single window rather than being consistently elevated.

### 7. MIBiG detection rate uses hardcoded 0.5, not the val-optimized threshold
- **Location:** `run_stage4.py:465` -- `bgc_detected = is_neg_pred < 0.5`
- **Impact:** Step 4.1 finds an optimal `is_negative` threshold (often much lower, e.g., 0.128 as the default in `error_analysis.py:128`). MIBiG detection rate uses 0.5 instead, making detection look artificially easy. If the same val-optimized threshold were used, detection rate would be lower.

### 8. `novel_candidate` confidence thresholds are arbitrary
- **Location:** `run_stage4.py:264-267`
- **Impact:** The high/medium/low confidence bins use hardcoded thresholds (`max_prob > 0.8`, `length > 10kb`, `dist > 20kb`) with no calibration or justification. These do not correspond to any false-discovery rate guarantee.

### 9. Error analysis default threshold differs from val-optimized threshold
- **Location:** `error_analysis.py:128` -- `--is_neg_threshold, default=0.128`
- **Impact:** The error analysis script uses a hardcoded default of 0.128 for `is_negative` threshold. If the user forgets to pass the actual val-optimized threshold from step 4.1, all miss-rate analyses use a potentially stale value.

### 10. No val/test split distinction for threshold selection
- **Location:** `run_stage4.py:71` -- threshold selection uses `split == "val"`
- **Impact:** The val split here refers to the same split used during Stage 3 training for checkpoint selection (train.py:175). This means the threshold is tuned on the same data that selected the checkpoint -- a minor but real double-dipping concern.
