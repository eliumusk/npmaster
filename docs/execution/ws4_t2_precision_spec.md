# WS4-T2: Prediction-Level Precision Specification and Results

**Date:** 2026-04-09
**Task ID:** WS4-T2
**Implementation:** Integrated into `scripts/execution/ws1_score_swap.py`

---

## 1. Problem Statement

The current Stage 4 precision metric is:

```
precision_as_count = n_matched_antismash_bgcs / n_predicted_regions
```

This is an antiSMASH-count-based metric. It counts how many antiSMASH BGCs were matched by predictions, divided by the number of predicted regions. Under this definition:

- One oversized predicted region that matches 3 antiSMASH BGCs counts as 3 matches but only 1 prediction
- This can inflate precision when regions are over-merged

## 2. New Metric: Prediction-Level Precision

```
precision_pred_level = n_predictions_with_any_match / n_predicted_regions
```

For each predicted region, we independently check: does it overlap with ANY antiSMASH BGC at the matching threshold (overlap/min_len >= 0.5)? Each prediction is scored yes/no.

This is a strict prediction-level metric: every prediction is accountable, no double-counting.

## 3. Implementation

```python
def compute_prediction_level_precision(regions, as_bgcs_by_loc, overlap_threshold=0.5):
    n_with_match = 0
    for reg in regions:
        genome_id = reg["genome_id"]
        contig = reg["contig"]
        as_list = as_bgcs_by_loc.get((genome_id, contig), [])
        for ab in as_list:
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            ab_len = ab_e - ab_s
            ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
            min_len = min(ab_len, reg["length"])
            if min_len > 0 and ov / min_len >= overlap_threshold:
                n_with_match += 1
                break
    return n_with_match, len(regions)
```

Code location: `scripts/execution/ws1_score_swap.py:212-228`

## 4. Comparison Results

From WS1-T1 experiment (test_id, reduced proxy):

| Score | P(AS-count) | P(pred-level) | Difference |
|-------|-------------|---------------|------------|
| score_is_neg | 0.3554 | 0.3673 | +0.012 |
| score_maxtype | 0.5411 | 0.5805 | +0.039 |
| score_fused | 0.4192 | 0.4374 | +0.018 |

From WS1-T1 experiment (test_ood, reduced proxy):

| Score | P(AS-count) | P(pred-level) | Difference |
|-------|-------------|---------------|------------|
| score_is_neg | 0.4129 | 0.4272 | +0.014 |
| score_maxtype | 0.6998 | 0.7251 | +0.025 |
| score_fused | 0.5849 | 0.6031 | +0.018 |

## 5. Findings

1. **The two precision definitions are close** (1-4 percentage point difference). This means the current region merging does not produce many oversized regions that match multiple antiSMASH BGCs.

2. **Prediction-level precision is consistently higher.** This is because P(pred-level) counts each prediction once (denominator and numerator on the same units), while P(AS-count) can have fewer matches than predictions when one prediction matches zero or one antiSMASH BGC.

3. **For future use, both metrics should be reported.** P(AS-count) tells us "how much of antiSMASH do we recover per prediction volume." P(pred-level) tells us "what fraction of our predictions are real."

## 6. Recommendation

- **P(pred-level) should become the primary precision metric** for deployment-facing evaluation
- **P(AS-count) should be retained** as a secondary metric for backward compatibility
- Both must be reported in all future benchmarks
