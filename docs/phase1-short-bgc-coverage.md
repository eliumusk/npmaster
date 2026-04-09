# Phase 1 Summary: Short-BGC Coverage

**Date:** 2026-04-09  
**Verdict:** G_edgepad was Phase 1 output, now superseded by H_hardneg_r10 (Phase 2)  
**Current mainline:** see [phase2-ood-robustness.md](phase2-ood-robustness.md)

---

## Phase 1 Decision (historical)

**G_edgepad was promoted at the end of Phase 1.** It was later superseded by H_hardneg_r10 after Phase 2 added hard-negative training.

Canonical artifacts:
- Checkpoint: `/data/muskliu/np_master/data/stage3_results/G_edgepad/best_G_edgepad.pt`
- Labels: `/data/muskliu/np_master/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv`
- Embeddings: `/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy`
- Eval: `/data/muskliu/np_master/data/stage4_results/p1_t3_edgepad_eval/`

---

## What Changed

Edge-padding: short contigs (< 16,384 bp) and contig tails now generate N-padded windows with Evo2 embeddings. 116,206 new windows added (of 123,953 total; GPU4 shard pending). 1,550 new positive windows.

---

## Gains

| Metric | F_cap08 | G_edgepad | Delta |
|--------|---------|-----------|-------|
| test_id recall | 0.692 | **0.737** | **+0.045** |
| test_ood recall | 0.542 | **0.578** | **+0.036** |
| <5kb BGC recall | 0.000 | **0.244** | from zero |
| 5-10kb BGC recall | ~0 | **0.382** | from near-zero |
| Truncated BGC recall | — | 0.638 | first measurement |
| Signature diversity@50 | 35 | **37** | +2 |

## Costs (documented, accepted)

| Metric | F_cap08 | G_edgepad | Delta |
|--------|---------|-----------|-------|
| test_id P(pred) | 0.650 | 0.587 | **-0.063** |
| test_ood P(pred) | 0.492 | 0.430 | **-0.062** |
| Sig enrichment@20 | 2.7x | 1.6x | **-1.1x** |
| Sig precision@20 | 95% | 90% | -5pp |

Precision drop is the primary Phase 2 target.

---

## Cumulative Progress (test_id, score_maxtype, full-window)

| Stage | Recall | P(pred) | Key Change |
|-------|--------|---------|------------|
| Original Stage 4 | 0.271 | 0.573 | is_neg < 0.054 |
| + score_maxtype | 0.593 | 0.581 | score formula |
| + prior-aligned | 0.668 | 0.567 | 15/85 sampler |
| + cap=0.8 labels | 0.692 | 0.650 | length-adaptive |
| **+ edge padding** | **0.737** | **0.587** | **padded windows** |

---

## Phase 1 Question Answered

> How much of the remaining known-BGC gap is still caused by short-BGC support defects?

**Answer:** Short-BGC support defects account for ~4.5pp of the remaining recall gap. Edge padding brings previously invisible BGCs (<5kb, 5-10kb) into the trainable regime, but their recall remains limited (24-38%) because N-padded windows dilute the BGC signal. Further short-BGC gains would require smaller window size (expensive: full re-extraction).

The remaining gap is now primarily:
1. OOD precision degradation (Phase 2 target)
2. 10-20kb BGCs at 59% recall (boundary difficulty)
3. Representation ceiling for short, diluted windows
