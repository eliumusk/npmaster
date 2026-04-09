# WS3-MVP: Length-Adaptive Overlap Threshold Results

**Date:** 2026-04-09
**Task ID:** WS3-MVP
**Config:** F (Config E + length-adaptive labels, capture threshold 0.5)
**Detection Score:** score_maxtype, full-window

---

## 1. What Changed

**Single change:** Windows that capture >= 50% of any overlapping BGC's length are now labeled positive, even if `overlap/window_size < 0.5`.

This supplements the original rule (`overlap/W >= 0.5`) with a BGC-centric rule (`overlap/BGC_length >= 0.5`).

### Relabeling Impact

| Metric | Before | After |
|--------|--------|-------|
| Positive windows | 66,820 | 68,212 (+1,392) |
| Ambiguous windows | 59,885 | 58,493 (-1,392) |
| Negative windows | unchanged | unchanged |
| Flipped types | - | RiPP: +763, Other: +602, Terpene: +26, Alkaloid: +1 |
| Flipped BGC lengths | - | 5.1-16kb (mean 11kb) |

### What Was NOT Changed
- Window size (16,384bp) — unchanged
- Stride (8,192bp) — unchanged
- Negative label rule — unchanged
- Ambiguous windows in training — still excluded
- Model architecture — unchanged
- Embeddings — reused (same window positions)

### Root Cause Finding: <5kb BGCs Are Not Solvable by Label Threshold

During implementation, we discovered that **all <5kb BGCs sit on contigs shorter than 16,384bp**. The sliding window loop (`while pos + W <= clen`) never executes on these contigs, so **no windows exist at all**, regardless of label threshold.

This means:
- <5kb BGC recall **cannot improve** without changing window size or adding edge padding
- The WS3-MVP relabeling only helps 5-16kb BGCs whose windows were previously labeled ambiguous

---

## 2. Cumulative Progress (test_id, score_maxtype, full-window)

| Stage | Recall | Precision (pred) | Key Change |
|-------|--------|-------------------|------------|
| Original Stage 4 | 0.2707 | 0.5729 | is_neg < 0.054 |
| + threshold fix | 0.5086 | 0.3673 | F1-optimal threshold |
| + score_maxtype | 0.5929 | 0.5429 | score formula |
| + prior-aligned (Config E) | 0.6679 | 0.5671 | sampler 50/50 → 15/85 |
| **+ WS3-MVP (Config F)** | **0.7205** | **0.5825** | **length-adaptive labels** |

**Total recall improvement: 0.2707 → 0.7205 (+0.450), no architecture change.**

---

## 3. Per-Type Recall (test_id, score_maxtype, full-window)

| Type | Config E | Config F | Delta |
|------|----------|----------|-------|
| Polyketide | 0.795 | 0.816 | +0.021 |
| NRP | 0.816 | 0.821 | +0.005 |
| **Terpene** | 0.767 | **0.812** | +0.045 |
| **RiPP** | 0.639 | **0.703** | **+0.064** |
| Saccharide | 0.941 | 0.941 | +0.000 |
| Alkaloid | 0.833 | 0.500 | -0.333 |
| **Other** | 0.518 | **0.622** | **+0.104** |

RiPP and Other gained the most — consistent with the relabeling adding 763 RiPP and 602 Other positive windows.

---

## 4. test_ood Results

| Metric | Config E | Config F | Delta |
|--------|----------|----------|-------|
| Recall | 0.4589 | **0.5648** | **+0.106** |
| Precision | 0.6204 | 0.4205 | **-0.200** |
| Terpene recall | 0.258 | **0.439** | +0.181 |
| RiPP recall | 0.452 | **0.540** | +0.088 |

**Recall improved substantially (+10.6pp) but precision dropped sharply (-20pp).** This suggests the model became more aggressive on OOD data, producing many more false positive regions (2911 regions in Config F vs 1599 in Config E).

This is a known risk of lowering label stringency — the model learns a more permissive decision boundary that may overcall on unseen phyla.

---

## 5. Gate Assessment

| Condition | Result | Status |
|-----------|--------|--------|
| short-BGC recall improvement | RiPP +6.4pp, Other +10.4pp | **PASS** |
| test_id overall recall no drop | 0.668 → 0.721 (+5.3pp) | **PASS** |
| test_id precision drop <= 0.05 | 0.567 → 0.583 (+1.5pp) | **PASS** |
| test_ood precision concern | 0.620 → 0.421 (-20pp) | **FLAG** |

**Overall: PASS on test_id. FLAG on test_ood precision for investigation.**

---

## 6. Canonical Artifacts

| Artifact | Path |
|----------|------|
| Checkpoint | `/data/muskliu/np_master/data/stage3_results/F_ws3_mvp/best_run_s0.pt` |
| New labels | `/data/muskliu/np_master/data/windows/windows_metadata_ws3mvp.csv` |
| Relabel summary | `/data/muskliu/np_master/data/windows/windows_metadata_ws3mvp_relabel_summary.json` |
| Eval results | `/data/muskliu/np_master/data/stage4_results/ws3_mvp_eval/score_swap_results.json` |
| Config | `bgc-gcd/configs/ws3_mvp_config_f.yaml` |
| Relabel script | `scripts/execution/ws3_mvp_relabel.py` |

---

## 7. Open Questions

1. **test_ood precision drop:** Is this a generalization issue from the softer labels, or does it indicate the capture=0.5 threshold is too aggressive? Could try capture=0.65 as a middle ground.

2. **<5kb BGC blind spot remains.** These BGCs require either smaller window size or edge padding. This is a geometry change (WS3 spec Option 3.1 or 3.4), not a label change.

3. **Alkaloid recall dropped** (0.833 → 0.500 on test_id). Only 12 Alkaloid BGCs in test, so this is high-variance. Not actionable without more data.
