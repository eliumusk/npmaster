# WS1-T1: Score Swap Benchmark Results

**Date:** 2026-04-09
**Task ID:** WS1-T1
**Checkpoint:** `B_contrastive/best_B_contrastive_s0.pt`
**Evaluation type:** Reduced proxy (ambiguous windows excluded)

---

## 1. Experiment Design

Three detection scores tested on the same checkpoint, same data, same region caller:

| Score | Formula | Description |
|-------|---------|-------------|
| `score_is_neg` | `1 - sigmoid(logit[7])` | Current Stage 4 score |
| `score_maxtype` | `max(sigmoid(logit[0:7]))` | Current Stage 3 BGC binary score |
| `score_fused` | `0.5 * score_is_neg + 0.5 * score_maxtype` | Simple equal-weight fusion |

All thresholds selected by F1-optimization on val set (same policy as current Stage 4).

---

## 2. Threshold Selection (Val Set)

| Score | Threshold | F1 | Precision | Recall |
|-------|-----------|-----|-----------|--------|
| `score_is_neg` | 0.9085 | 0.5114 | 0.4975 | 0.5260 |
| `score_maxtype` | 0.9720 | 0.5914 | 0.7115 | 0.5059 |
| `score_fused` | 0.9325 | 0.5620 | 0.5554 | 0.5688 |

**Note:** `score_is_neg` threshold of 0.9085 corresponds to `is_negative < 0.0915`, which is higher than the current Stage 4 threshold of 0.054. This means the F1-optimal threshold for the same score is already less conservative than the current operating point.

**Important context vs current Stage 4 results:**
The current Stage 4 recall of 0.2707 used `is_negative < 0.054`. This experiment's baseline (`score_is_neg` at threshold 0.9085, equivalent to `is_negative < 0.0915`) already yields test_id recall of 0.5086. This means **even without changing the score formula, just adjusting the threshold policy nearly doubles recall** (0.27 → 0.51).

---

## 3. Test Results

### 3.1 test_id

| Score | Recall | P(AS-count) | P(pred-level) | Regions | Unmatched | Clean Missed |
|-------|--------|-------------|---------------|---------|-----------|-------------|
| `score_is_neg` | **0.5086** | 0.3554 | 0.3673 | 4904 | 3202 | 2612 |
| `score_maxtype` | **0.5891** | 0.5411 | 0.5805 | 3731 | 1720 | 1387 |
| `score_fused` | **0.5894** | 0.4192 | 0.4374 | 4819 | 2830 | 2291 |

### 3.2 test_ood

| Score | Recall | P(AS-count) | P(pred-level) | Regions | Unmatched | Clean Missed |
|-------|--------|-------------|---------------|---------|-----------|-------------|
| `score_is_neg` | **0.3815** | 0.4129 | 0.4272 | 1945 | 1142 | 1078 |
| `score_maxtype` | **0.3677** | 0.6998 | 0.7251 | 1106 | 332 | 295 |
| `score_fused` | **0.3976** | 0.5849 | 0.6031 | 1431 | 595 | 554 |

### 3.3 Per-Type Recall (test_id)

| Type | score_is_neg | score_maxtype | score_fused |
|------|-------------|---------------|-------------|
| Polyketide | 0.786 | 0.757 | 0.773 |
| NRP | 0.778 | 0.786 | 0.801 |
| **Terpene** | 0.455 | **0.639** | 0.620 |
| **RiPP** | 0.461 | **0.556** | 0.556 |
| Saccharide | 0.941 | 0.706 | 0.941 |
| Alkaloid | 1.000 | 0.667 | 0.833 |
| Other | 0.381 | **0.453** | 0.447 |

### 3.4 Per-Type Recall (test_ood)

| Type | score_is_neg | score_maxtype | score_fused |
|------|-------------|---------------|-------------|
| Polyketide | 0.669 | 0.613 | 0.660 |
| NRP | 0.815 | 0.776 | 0.807 |
| Terpene | 0.140 | 0.127 | 0.145 |
| RiPP | 0.327 | 0.386 | 0.404 |
| Saccharide | 0.667 | 0.667 | 0.667 |
| Alkaloid | 0.667 | 0.167 | 0.333 |
| Other | 0.337 | 0.320 | 0.366 |

---

## 4. Unmatched Prediction Classification (WS4-T2 Integration)

Unmatched predictions are classified into three categories:
- **boundary_expansion**: within 5kb of an antiSMASH BGC (not truly novel — likely boundary overreach)
- **ambiguous_unmatched**: 5-20kb from nearest antiSMASH BGC
- **clean_missed**: >20kb from any antiSMASH BGC on same contig (strongest novel candidate signal)

### test_id

| Score | Total Unmatched | Boundary Expansion | Ambiguous | Clean Missed |
|-------|-----------------|--------------------|-----------|-------------|
| `score_is_neg` | 3202 | 99 (3.1%) | 491 (15.3%) | 2612 (81.6%) |
| `score_maxtype` | 1720 | 155 (9.0%) | 178 (10.3%) | 1387 (80.6%) |
| `score_fused` | 2830 | 119 (4.2%) | 420 (14.8%) | 2291 (80.9%) |

### test_ood

| Score | Total Unmatched | Boundary Expansion | Ambiguous | Clean Missed |
|-------|-----------------|--------------------|-----------|-------------|
| `score_is_neg` | 1142 | 28 (2.5%) | 36 (3.2%) | 1078 (94.4%) |
| `score_maxtype` | 332 | 28 (8.4%) | 9 (2.7%) | 295 (88.9%) |
| `score_fused` | 595 | 27 (4.5%) | 14 (2.4%) | 554 (93.1%) |

---

## 5. Precision Comparison (WS4-T2)

Two precision definitions compared:
- **P(AS-count)**: `n_matched_antismash_bgcs / n_predicted_regions` (current Stage 4 definition)
- **P(pred-level)**: `n_predictions_with_any_match / n_predicted_regions` (strict prediction-level)

The difference is small (1-2 percentage points), indicating the current matching is approximately 1-to-1 in practice.

---

## 6. Gate A Assessment

**Question:** Is score mismatch a first-order failure mode?

| Metric | Value |
|--------|-------|
| Baseline (score_is_neg) test_id recall | 0.5086 |
| Best variant (score_fused) test_id recall | 0.5894 |
| Delta | **+0.0808** |
| Gate A pass threshold | +0.05 |
| **Gate A passed** | **YES** |

**However**, the biggest finding is **not** which score formula wins. The biggest finding is:

> **Simply adjusting the threshold policy** (from the current `is_negative < 0.054` to the F1-optimal `score > 0.9085` on the same score) **nearly doubles test_id recall** from 0.2707 to 0.5086 — an improvement of **+0.24 absolute** — without changing any score formula at all.

This means the current Stage 4 threshold policy, not the score formula itself, is the dominant source of recall loss.

---

## 7. Key Findings

### 7.1 Threshold policy is a bigger problem than score formula

The recall gain from **threshold adjustment alone** (+0.24) dwarfs the gain from **score switching** (+0.08). This confirms Section 5.8 of the redesign memo: threshold policy is a first-class system contract issue.

### 7.2 score_maxtype has the best precision-recall tradeoff

On test_id, `score_maxtype` achieves nearly the same recall as `score_fused` (0.589 vs 0.589) but with much higher precision (0.54 vs 0.42) and far fewer unmatched predictions (1720 vs 2830).

On test_ood, `score_maxtype` is the most conservative (recall 0.37) but has the highest precision (0.70).

### 7.3 Terpene and RiPP recall dramatically improved with score_maxtype

- Terpene: 0.098 (old Stage 4) → 0.455 (is_neg F1-opt) → **0.639 (maxtype)**
- RiPP: 0.165 (old Stage 4) → 0.461 (is_neg F1-opt) → **0.556 (maxtype)**

These are the types that were nearly undetectable under the old operating point.

### 7.4 Discovery implication: unmatched prediction volume is very high

All variants produce 1000-3000+ unmatched predictions, with ~80-94% classified as "clean missed" (>20kb from any antiSMASH BGC). These are not all real novel BGCs — many are likely false positives. A fixed-budget ranking strategy (precision@K) is essential; raw unmatched count is not meaningful.

---

## 8. Open Questions

1. **Which score should become canonical?** `score_maxtype` has the best precision/recall profile, but it is conceptually odd for the "is this a BGC?" question to be answered by "is any type confident?" rather than "is it not negative?" Need to evaluate whether `score_maxtype` maintains its advantage under calibrated training (WS2).

2. **Full-window evaluation needed.** All results here are reduced-proxy (ambiguous excluded). Per the promotion rule, these findings are diagnostic but not promotion-grade.

3. **Discovery metrics missing.** This experiment did not include precision@K, de-duplicated family diversity, or enrichment vs null distribution. These must be added before any score choice is promoted.

---

## 9. Canonical Artifacts

| Artifact | Path |
|----------|------|
| Results JSON | `/data/muskliu/np_master/data/stage4_results/ws1_score_swap/score_swap_results.json` |
| Script | `scripts/execution/ws1_score_swap.py` |
| Log | `/data/muskliu/np_master/data/stage4_results/ws1_score_swap.log` |
