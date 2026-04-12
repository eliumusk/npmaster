# WS2-T1: Calibration Benchmark Results
> **Archive note:** Config B era results. Current mainline is Config F_cap08.

**Date:** 2026-04-09
**Task ID:** WS2-T1
**Checkpoint:** `B_contrastive/best_B_contrastive_s0.pt`
**Evaluation type:** Reduced proxy (ambiguous windows excluded)

---

## 1. Experiment Design

Three calibration methods tested on frozen model outputs (no retraining):

| Method | Parameters | Description |
|--------|-----------|-------------|
| Uncalibrated | - | Original sigmoid output (baseline) |
| Temperature | T=10.0000 | Single temperature parameter fitted on val NLL |
| Platt | a=2.0198, b=5.6134 | Two-parameter logistic regression on val |

Score: `1 - is_negative` throughout (keeping score formula fixed to isolate calibration effects).
Threshold: F1-optimal on val for each calibration variant.

---

## 2. Calibration Parameters

| Parameter | Value | Interpretation |
|-----------|-------|---------------|
| Temperature T | **10.0** | Hit the upper bound of the optimization range. The model's raw logits are so overconfident that the optimizer wants to flatten them as much as possible. |
| Platt a | 2.02 | Logit slope — rescales the logit range |
| Platt b | 5.61 | Large positive bias — shifts the decision boundary substantially |

**Critical observation:** T=10 means the optimizer hit the bound. The model's `is_negative` logits have extreme dynamic range, consistent with the 50/50 training prior creating massive overconfidence on the negative class.

---

## 3. Results

### 3.1 F1-Optimal Operating Point

| Method | test_id Recall | test_id Precision | test_ood Recall | test_ood Precision |
|--------|---------------|-------------------|-----------------|-------------------|
| Uncalibrated | 0.5086 | 0.3554 | 0.3815 | 0.4129 |
| Temperature | 0.5086 | 0.3554 | 0.3815 | 0.4129 |
| Platt | 0.5086 | 0.3554 | 0.3815 | 0.4129 |

**At the F1-optimal operating point, all three methods produce identical results.** This is expected: post-hoc monotone calibration (temperature/Platt) does not change the ranking of scores, only their absolute values. Since F1-optimal threshold selection adapts to any monotone transformation, the operating point is the same.

### 3.2 Recall-Precision Tradeoff (test_id)

| Method | R@P>=0.30 | R@P>=0.40 | R@P>=0.50 | P@R>=0.40 | P@R>=0.50 | P@R>=0.60 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| Uncalibrated | **0.6528** | 0.3986 | 0.1325 | 0.3064 | 0.3064 | 0.3064 |
| Temperature | 0.6402 | 0.3715 | 0.2679 | 0.3568 | **0.3568** | 0.3104 |
| Platt | 0.6367 | 0.3709 | **0.2708** | **0.3980** | 0.3540 | **0.3127** |

### 3.3 Recall-Precision Tradeoff (test_ood)

| Method | R@P>=0.30 | R@P>=0.40 | R@P>=0.50 | P@R>=0.40 | P@R>=0.50 | P@R>=0.60 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| Uncalibrated | 0.3786 | **0.3786** | 0.2171 | 0.1955 | 0.1955 | 0.1391 |
| Temperature | 0.4271 | 0.3268 | 0.3268 | 0.3326 | 0.2210 | 0.1617 |
| Platt | **0.4404** | 0.3648 | **0.3302** | **0.3717** | **0.2220** | 0.1594 |

---

## 4. Key Findings

### 4.1 Post-hoc calibration does NOT change the F1-optimal operating point

This is the most important finding. Because all three calibration methods are monotone transformations of the same score, F1-threshold-optimization compensates perfectly. **There is no free recall to be gained from post-hoc calibration at the F1-optimal point.**

### 4.2 Calibration improves the recall-precision CURVE shape

While the optimal point is the same, the curve shape differs:
- **Platt scaling improves test_ood R@P>=0.30 by +6%** (0.378 → 0.440)
- **Platt scaling improves test_ood P@R>=0.40 by +18%** (0.196 → 0.372)

This means Platt-calibrated scores give more stable behavior across different operating points. If the system needs to operate at a non-F1-optimal point (e.g., high-recall mode for safety), Platt calibration helps.

### 4.3 Temperature hit the upper bound — model is severely overconfident

T=10 means the raw `is_negative` logits have 10x too much dynamic range. This confirms the 50/50 sampling mismatch diagnosis from the redesign memo (Section 5.4):
- The model was trained with 50% positive / 50% negative
- Real genomes have ~5-15% BGC windows
- This prior mismatch inflates logit magnitudes
- Post-hoc calibration can partially compensate but cannot fix the ranking

### 4.4 Calibration is NOT the bottleneck — Gate B result

| Metric | Value |
|--------|-------|
| Baseline recall | 0.5086 |
| Best calibrated recall | 0.5086 |
| Delta | **+0.0000** |

**Gate B interpretation:** Post-hoc calibration does not improve the F1-optimal recall. This means the calibration/prior mismatch is NOT a first-order bottleneck at the F1-optimal operating point. However, it IS a concern for operating-point stability, and re-training with deployment-realistic priors (WS2-T2) may yield different results than post-hoc correction.

---

## 5. Implications for the Roadmap

1. **Post-hoc calibration alone is not enough.** The ranking order of windows is baked in by training. Monotone transforms cannot change it.

2. **Retraining with deployment priors (WS2-T2) is still worth pursuing.** Changing the training distribution can change the learned ranking, which post-hoc calibration cannot.

3. **Platt scaling is useful for multi-operating-point systems.** If the system needs both "high-recall mode" and "high-precision mode," Platt-calibrated scores give more predictable behavior across the full threshold range.

4. **The score_maxtype finding from WS1 is more important.** Since calibration cannot improve the F1-optimal recall, the +0.08 recall gain from switching to `max(type_probs)` (WS1-T1) represents a genuine ranking improvement, not just a calibration artifact.

---

## 6. Canonical Artifacts

| Artifact | Path |
|----------|------|
| Results JSON | `/data/muskliu/np_master/data/stage4_results/ws2_calibration/calibration_results.json` |
| Script | `scripts/execution/ws2_calibration.py` |
| Log | `/data/muskliu/np_master/data/stage4_results/ws2_calibration.log` |
