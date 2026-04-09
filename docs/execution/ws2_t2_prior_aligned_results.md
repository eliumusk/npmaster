# WS2-T2: Deployment-Prior Retraining Results

**Date:** 2026-04-09
**Task ID:** WS2-T2
**Config:** `E_prior_aligned` (pos_ratio=0.15, same architecture as Config B)
**Checkpoint:** `best_run_s0.pt`
**Evaluation:** Full-window, all 3 score variants

---

## 1. Training Summary

| Parameter | Config B (baseline) | Config E (prior-aligned) |
|-----------|--------------------|-----------------------|
| pos_ratio | 0.50 (64/64 per batch) | **0.15** (38/218 per batch) |
| Architecture | Same | Same |
| Epochs | 100 | 100 |
| Best val mAUROC | ~0.92 | 0.9235 |
| Best val bgcAUROC | ~0.88 | 0.8971 |
| Training time | ~11 min | ~11 min |

Window-level metrics are comparable, suggesting the model didn't lose discriminative power from the prior change.

---

## 2. Threshold Selection (val set, non-ambiguous)

| Score | Config B threshold | Config E threshold |
|-------|-------------------|-------------------|
| `score_is_neg` | 0.9085 | **0.7571** |
| `score_maxtype` | 0.9720 | **0.9006** |
| `score_fused` | 0.9325 | **0.8232** |

**Key observation:** All thresholds shifted lower (less conservative). This is the expected effect of training with deployment-realistic priors — the model is better calibrated, so the F1-optimal threshold is closer to the "natural" decision boundary rather than being pushed to an extreme by 50/50 training.

The `score_is_neg` threshold shifted the most (0.909 → 0.757), confirming that the `is_negative` head was most affected by the prior mismatch.

---

## 3. Head-to-Head Comparison (full-window, score_maxtype)

### test_id

| Metric | Config B | Config E | Delta | Direction |
|--------|----------|----------|-------|-----------|
| **Region recall** | 0.5929 | **0.6679** | **+0.075** | Better |
| Precision (pred-level) | 0.5429 | 0.5671 | +0.024 | Better |
| Precision (AS-count) | 0.5112 | 0.5466 | +0.035 | Better |
| Type concordance | 0.9365 | 0.9235 | -0.013 | ~Same |
| Predicted regions | 3975 | 4188 | +213 | More |
| Clean missed unmatched | 1387 | 1437 | +50 | More |

### test_ood

| Metric | Config B | Config E | Delta | Direction |
|--------|----------|----------|-------|-----------|
| **Region recall** | 0.3672 | **0.4589** | **+0.092** | Better |
| Precision (pred-level) | 0.7096 | 0.6204 | -0.089 | Worse |
| Precision (AS-count) | 0.6865 | 0.6041 | -0.082 | Worse |
| Type concordance | 0.9793 | 0.9607 | -0.019 | ~Same |
| Predicted regions | 1126 | 1599 | +473 | More |

### Per-Type Recall (test_id, score_maxtype)

| Type | Config B | Config E | Delta |
|------|----------|----------|-------|
| Polyketide | 0.757 | 0.795 | +0.038 |
| NRP | 0.786 | 0.816 | +0.030 |
| **Terpene** | 0.639 | **0.767** | **+0.128** |
| **RiPP** | 0.565 | **0.639** | **+0.074** |
| Saccharide | 0.706 | 0.941 | +0.235 |
| Alkaloid | 0.667 | 0.833 | +0.166 |
| Other | 0.460 | 0.518 | +0.058 |

### Per-Type Recall (test_ood, score_maxtype)

| Type | Config B | Config E | Delta |
|------|----------|----------|-------|
| Polyketide | 0.613 | 0.682 | +0.069 |
| NRP | 0.776 | 0.836 | +0.060 |
| **Terpene** | 0.127 | **0.258** | **+0.131** |
| **RiPP** | 0.384 | **0.452** | **+0.068** |
| Saccharide | 0.667 | 0.667 | +0.000 |
| Alkaloid | 0.167 | 0.500 | +0.333 |
| Other | 0.320 | 0.360 | +0.040 |

---

## 4. Key Findings

### 4.1 Prior-aligned training provides substantial recall gains

**test_id recall improved from 0.5929 to 0.6679 (+7.5 pp) AND precision also improved (+2.4 pp).** This is the rare outcome where both safety and discovery metrics improve simultaneously.

On test_ood, recall improved +9.2 pp but precision dropped -8.9 pp. This is the expected recall-precision tradeoff at a more aggressive operating point.

### 4.2 Terpene and RiPP continue to benefit the most

These previously-underdetected types show the largest gains:
- Terpene: 0.098 (old) → 0.639 (Config B + score fix) → **0.767** (Config E)
- RiPP: 0.165 (old) → 0.565 (Config B + score fix) → **0.639** (Config E)

### 4.3 Post-hoc calibration was insufficient, retraining was necessary

WS2-T1 showed that post-hoc calibration (temperature/Platt) cannot improve the F1-optimal operating point. WS2-T2 confirms that **retraining with deployment priors changes the learned ranking**, giving genuine recall improvement that monotone transforms cannot achieve.

### 4.4 The score gap between variants narrowed

With Config E, the `score_is_neg` baseline is stronger (0.632 vs 0.514 for Config B). The delta between `score_is_neg` and `score_maxtype` shrunk from +0.079 to +0.036. This suggests that better-calibrated training makes the `is_negative` head more aligned with `max(type_probs)`.

---

## 5. Cumulative Progress

| Stage | test_id Recall | test_id P(pred) | Key Change |
|-------|---------------|-----------------|------------|
| Original Stage 4 | 0.2707 | 0.5729 | `is_neg < 0.054` threshold |
| + F1-optimal threshold | 0.5086 | 0.3673 | Threshold policy fix |
| + score_maxtype | 0.5929 | 0.5429 | Score formula change |
| + prior-aligned training | **0.6679** | **0.5671** | Sampler: 50/50 → 15/85 |

**Total recall improvement: +0.397 (0.271 → 0.668) without any model architecture or representation change.**

---

## 6. Gate B Assessment

| Metric | Value |
|--------|-------|
| Baseline (Config B, score_maxtype, full-window) | 0.5929 |
| Config E (prior-aligned, score_maxtype, full-window) | 0.6679 |
| Delta | **+0.075** |
| **Gate B passed** | **YES** |

Prior mismatch is confirmed as a first-order problem. Retraining with deployment-realistic priors yields genuine gains that post-hoc calibration cannot replicate.

---

## 7. Canonical Artifacts

| Artifact | Path |
|----------|------|
| Checkpoint | `/data/muskliu/np_master/data/stage3_results/E_prior_aligned/best_run_s0.pt` |
| Training log | `/data/muskliu/np_master/data/stage4_results/ws2_t2_training.log` |
| Score swap results | `/data/muskliu/np_master/data/stage4_results/ws2_t2_eval/score_swap_results.json` |
| Config | `bgc-gcd/configs/ws2_t2_prior_aligned.yaml` |
| Modified sampler | `bgc-gcd/data/sampler.py` (pos_ratio parameter) |
| Modified train.py | `bgc-gcd/train.py` (passes pos_ratio from config) |
