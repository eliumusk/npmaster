# WS1-T2: Threshold-Selection Contract

**Date:** 2026-04-09  
**Version:** 2.1 (updated for H_hardneg_r10)  
**Task ID:** WS1-T2  
**Supersedes:** v2.0 (Config E, threshold 0.9006), v1.0 (Config B, threshold 0.9720)

---

## 1. Canonical Detection Score

The detection score for a single window is:

```
score_maxtype = max(sigmoid(logit[0:7]))
```

- `logit[0:7]` — the 7 raw logits corresponding to the 7 BGC type heads (indices 0 through 6 inclusive).
- `sigmoid` — element-wise standard sigmoid: `1 / (1 + exp(-x))`.
- `max` — scalar maximum over the 7 resulting probabilities.

Interpretation: higher `score_maxtype` means the window is more likely to contain a BGC. The score is bounded in (0, 1).

This is the **only** score used for binary BGC detection. All other candidate scores (e.g., `score_entropy`, `score_margin`) are rejected per WS1-T1 results.

---

## 2. Threshold Selection Policy

| Parameter | Value |
|---|---|
| **Split** | Validation set only. The test set is never used for threshold selection. |
| **Ambiguous windows** | **Excluded.** Windows labeled `"ambiguous"` have no reliable ground truth and must be removed before threshold fitting. |
| **Ground truth** | Binary: `is_bgc = 1` if `label == "positive"`, else `is_bgc = 0`. |
| **Objective** | Maximize F1 score. |
| **Method** | `sklearn.metrics.precision_recall_curve(y_true, scores)` → compute F1 at each threshold → select `threshold = argmax(F1)`. |
| **Current optimal threshold** | **0.8635** (H_hardneg_r10 s0; range 0.8577–0.8670 across 3 seeds) |

### Implementation pseudocode

```python
from sklearn.metrics import precision_recall_curve

# Filter out ambiguous windows
mask = labels != "ambiguous"
y_true = (labels[mask] == "positive").astype(int)
scores = score_maxtype[mask]

precision, recall, thresholds = precision_recall_curve(y_true, scores)
f1 = 2 * precision * recall / (precision + recall + 1e-12)
optimal_idx = f1.argmax()
threshold = thresholds[optimal_idx]
# threshold == 0.9006 (Config E)
```

### Val-set performance at threshold 0.8635 (H_hardneg_r10 s0)

| Metric | Value |
|---|---|
| F1 | 0.6486 |
| Precision | 0.8287 |
| Recall | 0.5328 |

### Threshold history

| Version | Config | Threshold | Val F1 |
|---|---|---|---|
| v1.0 | Config B | 0.9720 | 0.5914 |
| v2.0 | Config E (prior-aligned) | 0.9006 | 0.6463 |
| **v2.1** | **H_hardneg_r10** | **0.8635** | **0.6486** |

The threshold has progressively decreased as calibration improved. H_hardneg_r10's lower threshold reflects tighter decision boundaries from hard-negative training.

---

## 3. Window-to-Region Mapping

Given a set of scored windows on a contig, the mapping from window scores to called BGC regions proceeds as follows:

### Step 1: Binary window call

For each window `w`:

```
call(w) = 1  if score_maxtype(w) >= threshold
call(w) = 0  otherwise
```

### Step 2: Merge adjacent BGC windows

Adjacent windows (on the same contig, in genomic order) with `call = 1` are merged into contiguous regions. Two windows are "adjacent" if and only if they are consecutive in the contig's window tiling (no gap windows between them).

### Step 3: Minimum region length filter

Merged regions shorter than **3000 bp** are discarded. Region length is measured from the start coordinate of the first constituent window to the end coordinate of the last constituent window.

### Step 4: Per-region type assignment

For each surviving region:

1. Collect the sigmoid probability vectors `sigmoid(logit[0:7])` from all constituent windows.
2. Max-pool across windows: for each of the 7 type indices, take the maximum probability across all windows in the region.
3. The region's predicted BGC type is `argmax` of the 7-element max-pooled vector.

```python
# type_probs shape: (n_windows_in_region, 7)
pooled = type_probs.max(axis=0)        # shape: (7,)
predicted_type = pooled.argmax()        # int in [0, 6]
```

---

## 4. Operating Points

The threshold 0.8635 is the **F1-optimal operating point** for H_hardneg_r10 (s0). It represents the single default threshold used in all standard evaluations. Across 3 seeds, the threshold ranges from 0.8577 to 0.8670.

No alternative operating points (e.g., high-recall mode, high-precision mode) are defined in this contract version. If alternative operating points are needed, they must be specified in a separate contract or a future version of this one.

---

## 5. What This Contract Does Not Cover

- **Novel-candidate ranking score.** The detection score `score_maxtype` is for binary BGC-vs-nonBGC classification. A separate score for ranking novel BGC candidates (not in training data) is outside this contract's scope.
- **Operating-point selection for specific use cases.** This contract defines the F1-optimal threshold only. Use-case-specific thresholds (e.g., genome mining with high recall, annotation with high precision) require separate specification.
- **Full-window vs reduced-proxy choice.** Whether the model uses full 10 kb windows or a reduced proxy representation is orthogonal to the threshold policy defined here. Both representations use the same `score_maxtype` formula and the same threshold.

---

## 6. Evidence Base

| Source | Path |
|---|---|
| WS1-T1 Config B results | `/data/muskliu/np_master/data/stage4_results/ws1_score_swap_fullwindow/score_swap_results.json` |
| WS2-T2 Config E results | `/data/muskliu/np_master/data/stage4_results/ws2_t2_eval/score_swap_results.json` |
| **H_hardneg_r10 results** | **`/data/muskliu/np_master/data/stage4_results/p2_t2_hardneg_r10_eval/score_swap_results.json`** |
| Stability confirmation | `/data/muskliu/np_master/data/stage4_results/p2_stability_s{0,1,2}/` |

### Key results (H_hardneg_r10, score_maxtype, full-window, 3-seed mean±std)

| Split | Recall | Precision (pred-level) |
|---|---|---|
| **test_id** | **0.731 ± 0.012** | **0.650 ± 0.032** |
| test_ood | 0.566 ± 0.041 | 0.531 ± 0.080 |

### Cumulative improvement history

| Stage | test_id Recall | Key Change |
|---|---|---|
| Original Stage 4 | 0.271 | is_neg < 0.054 |
| + F1-optimal threshold | 0.509 | threshold policy fix |
| + score_maxtype (Config B) | 0.593 | score formula |
| + Config E (prior-aligned) | 0.668 | sampler 50/50 → 15/85 |
| + F_cap08 (cap=0.8) | 0.692 | length-adaptive labels |
| + G_edgepad (edge padding) | 0.737 | padded windows |
| **+ H_hardneg_r10** | **0.719 (0.731±0.012)** | **hard-neg upsampling** |

---

## 7. Change Control

This is **v2.1** of the threshold-selection contract.

### v2.1 changes from v2.0
- Threshold updated from 0.9006 (Config E) to **0.8635** (H_hardneg_r10)
- Evidence base updated with Phase 1/2 results and 3-seed stability confirmation
- Score formula and selection policy unchanged

### v2.0 changes from v1.0
- Threshold updated from 0.9720 (Config B) to 0.9006 (Config E, prior-aligned)
- Evidence base updated to include WS2-T2 results
- Checkpoint now uses `bgc_binary_auroc` for selection (per Task A)

Any of the following changes **requires a new version** (v1.1, v2.0, etc.) of this contract:

1. Change to the score formula (e.g., different logit indices, different aggregation, different activation).
2. Change to the threshold objective (e.g., switching from F1-optimal to precision-at-recall-X).
3. Change to the threshold split (e.g., using test set, using cross-validation).
4. Change to the ambiguous-window policy (e.g., including ambiguous windows with soft labels).
5. Change to the minimum region length filter (e.g., 5000 bp instead of 3000 bp).
6. Change to the region type assignment method (e.g., mean-pool instead of max-pool).

The new version must document: (a) what changed, (b) why, (c) updated performance numbers, and (d) a diff against the previous version.
