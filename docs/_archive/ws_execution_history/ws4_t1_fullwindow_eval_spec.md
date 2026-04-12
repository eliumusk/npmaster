# WS4-T1: Full-Window Deployment Evaluation Specification
**Date:** 2026-04-09
**Task ID:** WS4-T1

## 1. Current Reduced Proxy Definition

### What is excluded

Every window with `label == "ambiguous"` is dropped before any evaluation logic runs. This affects:

| Step | Code location | Filter expression |
|------|--------------|-------------------|
| 4.1 threshold selection (val) | `run_stage4.py:71-73` | `r["split"] == "val" and r["label"] != "ambiguous"` |
| 4.1 confusion analysis (test_id) | `run_stage4.py:93-97` | `r["split"] == "test_id" and r["label"] != "ambiguous"` |
| 4.2 genome-level region merging + matching (test_id, test_ood) | `run_stage4.py:148-149` | `r["split"] == split and r["label"] != "ambiguous"` |
| Training dataset | `dataset.py:28` | `df["label"].values != "ambiguous"` |

### How "ambiguous" is defined

Source: `scripts/stage1/05_windows_and_labels.py:131-156`

A window (W=16384, S=8192) is labeled ambiguous in two cases:

1. **Partial-overlap boundary windows**: The window overlaps at least one BGC, but `max_overlap_ratio < 0.5` (lines 143-147). These sit at BGC boundaries; they contain some BGC sequence but not enough to be called positive.

2. **Buffer-zone negative-adjacent windows**: The window has zero overlap with any BGC, but the nearest BGC is within the 10 kb buffer (`min_dist < 10000`; lines 153-156). These are "near-BGC" windows that might contain flanking biosynthetic genes.

In both cases the label vector is all-zeros with `is_negative=0`, making them have no valid training signal.

### Why this is a proxy, not deployment-realistic

In a real genome scan, the model receives every window. There is no oracle to pre-filter ambiguous windows. The current evaluation:
- Inflates precision: boundary windows that would be FP in deployment are removed.
- Inflates recall: partial-overlap windows that might confuse the classifier are absent.
- Distorts region merging: removing boundary windows creates artificial gaps between positive runs, changing merged-region boundaries and lengths.
- Makes novel-candidate counts unreliable: boundary-adjacent FPs that would appear as novel candidates are invisible.

### When reduced proxy is appropriate

- Early-stage model comparison (relative ranking is preserved if exclusion is consistent).
- Hyperparameter sweeps where speed matters.
- NOT appropriate for: final metric reporting, promotion decisions, novel candidate enumeration, or any claim about deployment performance.

## 2. Full-Window Deployment Approximation Definition

### Core principle

**Full-window evaluation means: every window that exists in the metadata receives a model prediction and participates in region merging.** No window is dropped based on its label.

### Precise definition

A full-window evaluation run differs from the reduced proxy in exactly these ways:

1. **Prediction scope**: Model predicts on ALL windows in each split (val, test_id, test_ood), including `label == "ambiguous"`.

2. **Threshold selection (step 4.1)**: Two options (see section 3 for trade-offs):
   - **Option A (recommended)**: Select thresholds on val split using only non-ambiguous windows (same as now), then APPLY those thresholds to all windows including ambiguous. Rationale: ambiguous windows have no reliable ground truth, so they cannot guide threshold optimization.
   - **Option B**: Select thresholds on the full val set. This requires assigning a ground-truth label to ambiguous windows (see open questions).

3. **Confusion analysis (step 4.1)**: Report TWO confusion tables:
   - Table 1: non-ambiguous windows only (for comparison with reduced proxy).
   - Table 2: all windows, with ambiguous windows classified under a separate column showing how many ambiguous windows the model calls BGC vs non-BGC.

4. **Region merging (step 4.2)**: All windows participate. This is the critical change: ambiguous boundary windows will now fill gaps between positive runs, producing longer merged regions that better approximate real scanning.

5. **Genome-level matching (step 4.2)**: Matching logic unchanged (overlap >= 50% of shorter span). But merged regions will differ because boundary windows are included.

6. **Novel candidate enumeration (step 4.2/4.3)**: Runs on the full-window merged regions.

### What "full-window" does NOT mean

- It does NOT mean training on ambiguous windows. Training exclusion (`dataset.py:28`) stays.
- It does NOT mean counting ambiguous windows toward window-level precision/recall (they lack ground truth).
- It does NOT mean changing the labeling logic in `05_windows_and_labels.py`.

## 3. Metric Behavior Under Full-Window

### Window-level metrics

| Metric | Reduced proxy | Full-window | Expected direction |
|--------|--------------|-------------|-------------------|
| Precision (is_negative) | Computed on clean pos/neg only | Same denominator (only non-ambiguous have labels), but model sees more context | Likely similar |
| Recall (is_negative) | Same | Same | Likely similar |

Window-level P/R/F1 **should still be computed on non-ambiguous windows only** because ambiguous windows have no ground truth label. The difference is that the model must still produce predictions for them; we just don't score them.

### Region-level (genome-level) metrics

| Metric | Reduced proxy | Full-window | Expected direction |
|--------|--------------|-------------|-------------------|
| Region count | Fewer, shorter regions | More, longer regions (boundary windows fill gaps) | Increase |
| antiSMASH recall | Potentially lower (gaps may prevent overlap threshold) | Likely higher (longer regions overlap more with antiSMASH) | Increase |
| antiSMASH precision | Inflated (no boundary FPs) | More realistic (boundary FPs appear) | Decrease |
| Novel candidates | Under-counted | More candidates, including boundary-expansion artifacts | Increase |
| Type concordance | Computed on clean matches | Same computation but over different matched set | Uncertain |

### Key new metrics to report in full-window mode

1. **Ambiguous-window BGC call rate**: fraction of ambiguous windows predicted as BGC. Split by ambiguous sub-type (partial-overlap vs buffer-zone). Expected: partial-overlap windows should have higher BGC call rate than buffer-zone windows.

2. **Region length delta**: distribution of `full_region_length - proxy_region_length` for matched antiSMASH BGCs. Shows how much boundary windows extend predicted regions.

3. **Boundary-expansion novel candidates vs clean novel candidates**: Novel candidates that exist ONLY because boundary windows were included (absent in reduced proxy) vs novel candidates present in both. Only "clean" novel candidates should be promoted.

## 4. Promotion Criteria

The following decisions REQUIRE full-window evaluation results. Reduced proxy results alone are NOT sufficient.

### Hard gates (must pass before any promotion)

| Decision | Required evidence | Threshold |
|----------|------------------|-----------|
| Claim genome-level recall | Full-window antiSMASH recall on test_id AND test_ood | >= 0.60 on test_id |
| Claim genome-level precision | Full-window precision on test_id | >= 0.30 on test_id |
| Rank novel candidates for wet-lab | Full-window novel candidate list with confidence tiers | Only "high" confidence candidates from full-window run |
| Compare to antiSMASH | Full-window region merging (since antiSMASH sees all windows) | Side-by-side on same genomes |

### Decisions where reduced proxy suffices

| Decision | Justification |
|----------|--------------|
| Model architecture comparison (A vs B) | Relative ranking preserved under consistent exclusion |
| Hyperparameter selection | Same |
| Training convergence monitoring | Uses training/val loss, not deployment metrics |

### Promotion workflow

```
1. Train model (ambiguous excluded from training)
2. Run reduced-proxy Stage 4 (fast, for iteration)
3. When reduced-proxy metrics look promising:
   a. Run full-window Stage 4
   b. Compare region-level metrics between proxy and full-window
   c. If delta is large (recall drops >5pp or precision drops >10pp), investigate
4. Only promote claims based on full-window numbers
5. Novel candidates: intersect full-window and reduced-proxy candidate lists;
   candidates appearing only in full-window are "boundary-expansion" artifacts
   and must be flagged separately
```

## 5. Implementation Plan

### Change 1: Add `--include-ambiguous` flag to `run_stage4.py`

**File:** `scripts/stage4/run_stage4.py`

Add CLI argument:
```python
ap.add_argument("--include-ambiguous", action="store_true",
                help="Include ambiguous windows in prediction and region merging (full-window mode)")
```

Pass `include_ambiguous` to `step_4_1`, `step_4_2`.

### Change 2: Modify `step_4_1` (threshold selection + confusion)

**Lines to change:** 71-73, 93-97

```python
# Threshold selection: always on non-ambiguous val (even in full-window mode)
val_mask = np.array([r["split"] == "val" and r["label"] != "ambiguous" for r in meta])

# Confusion analysis: controlled by flag
if include_ambiguous:
    tid_mask = np.array([r["split"] == "test_id" for r in meta])
else:
    tid_mask = np.array([r["split"] == "test_id" and r["label"] != "ambiguous" for r in meta])
```

When `include_ambiguous=True`, emit a separate JSON block `ambiguous_window_stats`:
```json
{
  "n_ambiguous": 1234,
  "n_ambiguous_called_bgc": 456,
  "ambiguous_bgc_rate": 0.369,
  "by_subtype": {
    "partial_overlap": {"n": 800, "called_bgc": 400},
    "buffer_zone": {"n": 434, "called_bgc": 56}
  }
}
```

To distinguish sub-types, check `max_bgc_overlap_ratio`: if > 0 the window is partial-overlap; if == 0 it is buffer-zone.

### Change 3: Modify `step_4_2` (genome-level)

**Lines to change:** 148-149

```python
if include_ambiguous:
    mask = np.array([r["split"] == split for r in meta])
    s_meta = [r for r in meta if r["split"] == split]
else:
    mask = np.array([r["split"] == split and r["label"] != "ambiguous" for r in meta])
    s_meta = [r for r in meta if r["split"] == split and r["label"] != "ambiguous"]
```

No other changes needed: region merging, matching, and novel candidate logic operate on predictions and are label-agnostic.

### Change 4: Output directory convention

When `--include-ambiguous` is set, append `_fullwindow` to the output directory:
```python
out_suffix = "_fullwindow" if args.include_ambiguous else ""
# step_4_1 → f"{args.out}/step4_1{out_suffix}"
```

This keeps reduced-proxy and full-window results side by side for comparison.

### Change 5: Comparison report script (new)

Create `scripts/stage4/compare_proxy_vs_fullwindow.py` that:
1. Loads `step4_2/genome_level_metrics.json` and `step4_2_fullwindow/genome_level_metrics.json`
2. Computes deltas for recall, precision, novel candidate count
3. Identifies novel candidates unique to full-window (boundary-expansion artifacts)
4. Outputs `proxy_vs_fullwindow_comparison.json`

### Change 6: Contig-edge handling

Current `05_windows_and_labels.py` line 116: `while pos + W <= clen` -- windows shorter than W at contig ends are already dropped (no padding). This is correct and should remain unchanged. In deployment, the same policy applies: if a contig tail is < W, it is not scanned. This is documented but does not change between proxy and full-window modes.

If future work adds padding for short tails, those padded windows should be:
- Predicted (model receives them)
- Flagged as `edge_padded=True` in metadata
- Excluded from window-level metric computation
- Included in region merging with a confidence penalty (e.g., multiply predicted probability by `actual_length / W`)

### Estimated effort

| Change | Lines of code | Risk |
|--------|--------------|------|
| CLI flag + plumbing | ~10 | Low |
| step_4_1 conditional + ambiguous stats | ~30 | Low |
| step_4_2 conditional | ~5 | Low |
| Output directory convention | ~5 | Low |
| Comparison script | ~80 | Low |
| **Total** | **~130** | **Low** |

## 6. Open Questions

1. **Should ambiguous windows participate in window-level metric computation?**
   Recommendation: No. They have no ground truth. But report their prediction distribution separately (Change 2 above). If a future labeling pass assigns soft labels to ambiguous windows, revisit.

2. **What is the correct ground truth for a partial-overlap ambiguous window?**
   Currently: all-zeros with `is_negative=0`. This means they are neither positive nor negative. In full-window mode, we predict on them but do not evaluate them. If we wanted to include them in metrics, we would need to define a soft label (e.g., `is_positive = overlap_ratio`) -- but this is out of scope for this spec.

3. **Should threshold selection ever use ambiguous windows?**
   Recommendation: No (Option A in section 2). Ambiguous windows lack clean labels; including them in PR-curve computation would bias thresholds toward the boundary region where labels are unreliable.

4. **How to handle the boundary-expansion artifact in novel candidates?**
   Proposed: Tag each novel candidate with `appears_in_proxy: bool`. Candidates with `appears_in_proxy=False` are boundary-expansion artifacts. For promotion, only consider candidates with `appears_in_proxy=True` OR candidates where the full-window region is far (> 20 kb) from any boundary-expansion source.

5. **Should full-window mode affect MIBiG validation (step 4.4)?**
   No. MIBiG validation uses separate embeddings from MIBiG entries, not genome windows. It is unaffected by the ambiguous window question.

6. **Training impact: should a future iteration train on ambiguous windows with soft labels?**
   Out of scope for this spec. If pursued, this would be a separate experiment requiring its own evaluation. The current spec assumes training exclusion is fixed.
