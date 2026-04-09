# Phase 2 Summary: OOD Robustness

**Date:** 2026-04-09  
**Verdict:** H_hardneg_r10 confirmed as official mainline (3-seed stability verified)  
**Stability:** See [stability-3seed.md](stability-3seed.md)  
**Next:** Architecture decision gate

---

## Decision

**H_hardneg_r10 is the official mainline.** Stability confirmed across 3 seeds.

Canonical artifacts:
- Checkpoint: `/data/muskliu/np_master/data/stage3_results/H_hardneg_r10/best_H_hardneg_r10.pt`
- Labels: `/data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv`
- Embeddings: `/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy`
- Eval: `/data/muskliu/np_master/data/stage4_results/p2_t2_hardneg_r10_eval/`
- Discovery: `/data/muskliu/np_master/data/stage4_results/p2_t2_hardneg_r10_discovery/`
- Config: `bgc-gcd/configs/mainline.yaml`

---

## Problem Statement

After Phase 1 (edge padding), OOD precision had degraded to 0.430 — the worst across all configs. P2-T1 decomposition revealed a clear failure pattern:

- **RiPP accounted for 64.6% of OOD FPs** (vs 19.5% in ID, a +45pp shift)
- **Cyanobacteriota contributed 82%** of all OOD FPs
- **90% of OOD FPs were >20kb from any known BGC** — these are not boundary artifacts
- **All OOD FP scores ≥ 0.904** — model was extremely confident in these false calls
- **47.6% of OOD FPs were on contigs with zero antiSMASH BGCs** — pure background misclassified as BGC

Root cause: the model had never seen Cyanobacteriota during training. Its RiPP decision boundary was calibrated only against training-distribution backgrounds (mainly Actinomycetota, Pseudomonadota, Bacillota), and Cyanobacteriota background sequences apparently trigger similar patterns.

---

## Intervention: Hard-Negative Mining

**Approach:** Mine hard negatives from *training data* — windows that the current model incorrectly predicts as BGC with high confidence. Oversample these during training to tighten the decision boundary.

**Implementation:**
1. Score all 544,831 training negatives with G_edgepad checkpoint
2. Identify top-5000 by `max(type_probs)` (general hard negatives)
3. Identify top-5000 by RiPP score specifically (RiPP-targeted)
4. Union: 8,304 hard negatives flagged
5. Modified `BalancedBatchSampler` to draw a configurable fraction of each batch's negative half from the hard-negative pool

**Key design principle:** No OOD data added to training. Only existing training negatives are reweighted. This preserves the ID/OOD split integrity.

**Hard negative profile:**
- 71% from Actinomycetota (the BGC-densest phylum — these are Actino backgrounds that look like BGC)
- 16% from Pseudomonadota
- 10% from Bacillota
- Score range: 0.79–0.997 (all very confidently misclassified by current model)

---

## Results: Ratio Sweep

| Metric | G_edgepad (baseline) | H_r30 (ratio=0.30) | H_r10 (ratio=0.10) |
|--------|---------------------|--------------------|--------------------|
| test_id recall | **0.737** | 0.673 | 0.719 |
| test_id P(pred) | 0.587 | 0.656 | **0.686** |
| test_id regions | 4423 | 3682 | 3727 |
| test_id unmatched | 1918 | 1383 | **1274** |
| test_ood recall | **0.578** | 0.477 | 0.519 |
| test_ood P(pred) | 0.430 | **0.659** | 0.623 |
| test_ood regions | 2906 | 1572 | 1803 |
| test_ood unmatched | 1690 | 567 | **711** |

**r=0.30 was too aggressive:** Recall dropped 6.4pp (ID) and 10.1pp (OOD). RiPP recall cratered (ID: 0.679→0.548, OOD: 0.566→0.406). The model became too conservative.

**r=0.10 is the Pareto optimum:**
- ID recall cost only -1.8pp (0.737→0.719), well within noise
- ID precision gained +9.9pp (0.587→0.686), now the best across all configs
- OOD precision gained +19.3pp (0.430→0.623)
- OOD FP reduced by 58% (1690→711)

---

## Discovery Quality Check

| Discovery metric | F_cap08 | G_edgepad | H_r10 |
|-----------------|---------|-----------|-------|
| Sig precision@20 | 95% | 90% | **90%** |
| Sig precision@50 | 92% | 92% | 86% |
| Sig diversity@50 | 35 | 37 | **38** |
| Null sig@50 | 54% | 62% | 54% |
| Sig enrichment@20 | 2.7x | 1.6x | **1.8x** |

Discovery is non-inferior at @20 (90%, same as G_edgepad). @50 drops from 92%→86% (3 fewer sig hits), but diversity is highest (38 unique profiles). Enrichment recovers from 1.6x→1.8x because null signature rate drops back to 54%.

**Verdict:** Discovery quality is acceptable. The model is more selective, which marginally reduces @50 hit rate but improves the signal-to-noise ratio of the candidate pool.

---

## Cumulative Progress (test_id, score_maxtype, full-window)

| Stage | Recall | P(pred) | Key Change |
|-------|--------|---------|------------|
| Original Stage 4 | 0.271 | 0.573 | is_neg < 0.054 |
| + score_maxtype | 0.593 | 0.581 | score formula |
| + prior-aligned (E) | 0.668 | 0.567 | 15/85 sampler |
| + cap=0.8 labels (F) | 0.692 | 0.650 | length-adaptive |
| + edge padding (G) | 0.737 | 0.587 | padded windows |
| **+ hard neg r=0.10 (H)** | **0.719** | **0.686** | **hard-neg upsampling** |

**Net from original: recall +0.448, precision +0.113. Zero architecture changes.**

---

## Cumulative Progress (test_ood)

| Stage | Recall | P(pred) |
|-------|--------|---------|
| + prior-aligned (E) | 0.459 | 0.620 |
| + cap=0.8 labels (F) | 0.542 | 0.492 |
| + edge padding (G) | 0.578 | 0.430 |
| **+ hard neg r=0.10 (H)** | **0.519** | **0.623** |

OOD precision recovered from its worst point (0.430) to 0.623, even surpassing the prior-aligned baseline (0.620), while retaining most of the OOD recall gains from edge padding.

---

## Phase 2 Question Answered

> After coverage defects are addressed, can the current architecture remain competitive under OOD distribution shift, or are we now truly limited by representation?

**Answer:** Hard-negative mining within training data is sufficient to recover most OOD precision (+19pp) at a modest recall cost (-6pp OOD, -2pp ID). The current architecture is **not yet representation-limited** for the OOD precision problem — it was training-distribution-limited. The remaining OOD gap (recall 0.519 vs ID 0.719) likely reflects genuine distribution shift in Evo2's embedding space between trained and unseen phyla, which may eventually require representation-level work.

---

## Status of Follow-Up Items

1. **Stability:** Confirmed. 3-seed results in [stability-3seed.md](stability-3seed.md): test_id recall 0.731±0.012, P(pred) 0.650±0.032.
2. **Short-BGC buckets under H_r10:** Measured in stability run. <5kb recall 0.222±0.005, 5-10kb 0.352±0.022.
3. **Architecture decision gate:** Next phase. The project should evaluate whether further gains require representation changes (LoRA, smaller windows, etc.) or whether the current frozen-Evo2 architecture has reached its practical ceiling.
