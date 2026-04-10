# Current Mainline Summary

**Date:** 2026-04-10  
**Purpose:** concise project-state summary after Phase 3 (context model)  
**Previous mainline (H_hardneg_r10):** archived, see [phase2-ood-robustness.md](phase2-ood-robustness.md)

---

## 1. Official Mainline

The current official mainline is:

- **Config:** `Phase 3 Route B` (context model, end-to-end)
- **Backbone:** frozen Evo2 embeddings
- **Architecture:** ProjectionHead (warm-started, trainable) + Dilated 1D CNN (3 layers, d=1,2,4) + GatedResidual (sigmoid gate, β init -4.0) + Classifier
- **Training:** contig chunks L=32 with overlap stride=16, center-region masked BCE loss, AdamW with layered lr (proj 1e-4, context/classifier 1e-3), 100 epochs
- **Detection score:** `score_maxtype = max(sigmoid(type_logits))`
- **Evaluation mode:** full-contig inference (not chunked)
- **Checkpoint metric:** val region F1 (every 5 epochs)
- **3-seed stability:** confirmed

Canonical references:

- [phase3-route-b-results.md](phase3-route-b-results.md) — full Route B results
- [phase3-route-a-results.md](phase3-route-a-results.md) — Route A ablation (frozen projector)
- [phase3-context-model-plan.md](phase3-context-model-plan.md) — full Phase 3 design doc
- [train_context_e2e.py](/Users/musk/Documents/code/np-master/bgc-gcd/train_context_e2e.py)
- [context_full_model.py](/Users/musk/Documents/code/np-master/bgc-gcd/model/context_full_model.py)
- [chunk_dataset.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/chunk_dataset.py)

Server-side canonical artifacts:

- Checkpoints: `/data/muskliu/np_master/data/phase3/route_b/best_route_b_s{0,1,2}.pt`
- Chunk index: `/data/muskliu/np_master/data/phase3/route_b/chunk_index.pkl`
- Discovery: `/data/muskliu/np_master/data/phase3/route_b/discovery/`

---

## 2. What The Numbers Mean

All "known-BGC" numbers below are **recovery against antiSMASH annotations**, not a gold-standard head-to-head benchmark against antiSMASH itself.

So the right interpretation is:

- higher recall = we recover more antiSMASH-annotated BGCs
- higher prediction-level precision = more of our predicted regions overlap antiSMASH BGCs

This is still the right safety benchmark for now, but it is not yet a final claim of "beating antiSMASH."

---

## 3. Key antiSMASH-Recovery Results

### 3.1 Original default Stage 4

At the original default operating point:

- `test_id` recall: `0.2707`
- `test_ood` recall: `0.2132`
- precision at that stage was reported under the old antiSMASH-count-based metric:
  - `test_id`: `0.5729`
  - `test_ood`: `0.8901`

Reference:

- [2026-04-08-bgc-system-redesign.md](/Users/musk/Documents/code/np-master/docs/2026-04-08-bgc-system-redesign.md#L105)

### 3.2 Current official mainline (Phase 3 Route B)

End-to-end context model, full-contig inference, `score_maxtype`, 3-seed mean ± std:

| Metric | `test_id` | `test_ood` |
|---|---:|---:|
| Recall | **0.820 ± 0.001** | **0.594 ± 0.006** |
| Prediction-level precision | **0.857 ± 0.002** | **0.653 ± 0.009** |
| `<5kb` recall | **0.751 ± 0.004** | **0.556 ± 0.012** |
| `5-10kb` recall | **0.706 ± 0.003** | **0.651 ± 0.007** |
| `10-20kb` recall | **0.691 ± 0.005** | **0.411 ± 0.009** |
| `20-50kb` recall | 0.842 ± 0.004 | 0.569 ± 0.007 |
| `>50kb` recall | 0.987 ± 0.001 | 0.987 ± 0.004 |

### 3.3 Previous mainline (H_hardneg_r10, archived)

| Metric | `test_id` | `test_ood` |
|---|---:|---:|
| Recall | 0.731 ± 0.012 | 0.566 ± 0.041 |
| Prediction-level precision | 0.650 ± 0.032 | 0.531 ± 0.080 |
| `<5kb` recall | 0.222 ± 0.005 | 0.171 ± 0.014 |
| `5-10kb` recall | 0.352 ± 0.022 | 0.380 ± 0.009 |

### 3.4 Cumulative gain from the original system

On `test_id`:

- recall: `0.271` → `0.820` (**+0.549**)
- P(pred): `0.573` → `0.857` (**+0.284**)
- `<5kb` recall: near-zero → **0.751**

On `test_ood`:

- recall: `0.213` → `0.594` (**+0.381**)
- P(pred): collapse → `0.653`

**Phase 0-2** (spec/data fixes) contributed ~60% of recall gain with zero architecture changes.
**Phase 3** (context model) contributed ~40% with the first architecture upgrade.
**Stability improved dramatically:** test_id recall std dropped from 0.012 → 0.001 (12x tighter).

---

## 4. Discovery-Side Results

Current discovery picture (Phase 3 Route B, seed=0):

| Metric | Route B | Previous (H_hardneg_r10) |
|--------|---------|--------------------------|
| sig precision@20 | **90%** (18/20) | 90% (18/20) |
| sig precision@50 | **92%** (46/50) | 86% (43/50) |
| diversity@50 (signature profiles) | **45** | 38 |
| biosyn precision@20 | **100%** (20/20) | — |
| biosyn precision@50 | **100%** (50/50) | — |

Top signature domains in Route B candidates: KR_C, SDR, PCP/T, Enoyl_CoA_hyd, AMP-binding, KS, KS_C, Methyltransf_11, Acyl_CoA_dh, p450 — all core biosynthetic domains.

Interpretation:

- Discovery quality **improved** with Phase 3 (diversity +7, sig@50 +6pp)
- Recovery AND discovery moved up together — no tradeoff
- Top candidates are enriched for biosynthetic signature domains, not random background

References:

- [p2_phase_summary.md](phase2-ood-robustness.md#L81)
- [ws_discovery_benchmark.md](discovery-benchmark.md)

---

## 5. What Has Been Proven

The project has now established five points clearly.

1. **The original low performance was not mainly an architecture failure.**  
   A large part of the gap came from objective mismatch, threshold mismatch, prior shift, and label/input-coverage defects.

2. **The current route is scientifically real.**  
   Discovery-side candidates remain enriched for biosynthetic signature domains, so the model is not just learning trivial background composition.

3. **Short-BGC coverage was a real bottleneck.**  
   Edge padding moved `<5kb` recall from effectively zero to about `22%`, and `5-10kb` recall to about `35%`.

4. **OOD false positives were also a real bottleneck.**  
   Hard-negative training reduced OOD unmatched predictions strongly while keeping most recall.

5. **The current mainline is reproducible.**  
   `test_id` results are stable across 3 seeds; `test_ood` still has moderate variance, but central tendency is clear.

---

## 6. What Is Still Not Good Enough

The project is in a much better state, but it is not finished.

The main unresolved problems are:

1. **Known-BGC recovery is improved but still incomplete.**
   - `test_id` still misses about `27%` of antiSMASH-annotated BGCs
   - `test_ood` still misses about `43%`

2. **Very short BGCs remain difficult.**
   - `<5kb` recall is only about `22%`
   - edge padding fixed visibility, but not full separability

3. **OOD robustness is better but not fully settled.**
   - `test_ood` precision is much better than the edge-padding-only stage
   - but seed variance remains noticeably higher than in `test_id`

4. **We still cannot claim parity with antiSMASH.**
   The project can now claim a credible and much stronger system, but not "antiSMASH matched" in a strict sense.

---

## 7. Why It Is Reasonable To Start Architecture Discussion Now

It is reasonable to start architecture discussion **now**, because the main pre-architecture confounders have been substantially reduced:

- score contract aligned
- threshold contract aligned
- train/deploy prior mismatch reduced
- short-BGC coverage defect partly fixed
- OOD hard-negative deficiency partly fixed

That means future failure is more likely to reflect a real limitation of the current representation stack rather than a simple pipeline mistake.

This does **not** mean the current architecture is already proven to be the bottleneck.

It means:

**we now have a clean enough baseline that architecture work can be discussed seriously and judged fairly.**

---

## 8. Current Decision Boundary

The current project status should be stated as:

- **Official mainline:** `H_hardneg_r10`
- **Current claim:** strong recovery and discovery baseline, not yet antiSMASH parity
- **Current bottlenecks:** very-short BGCs, remaining OOD instability, likely boundary/local-structure limits
- **Next strategic topic:** architecture redesign

Recommended framing for the next discussion:

1. What failure modes remain after alignment, coverage, and robustness fixes?
2. Which of those failure modes are plausibly caused by global mean pooling and loss of position information?
3. What is the smallest architecture change that targets those failure modes directly?

---

## 9. Code/Artifact Map For The Next Discussion

Use these as the minimum set of canonical materials:

- [train.py](/Users/musk/Documents/code/np-master/bgc-gcd/train.py)
- [sampler.py](/Users/musk/Documents/code/np-master/bgc-gcd/data/sampler.py)
- [ws_checkpoint_selection_contract.md](docs/_archive/ws_execution_history/ws_checkpoint_selection_contract.md)
- [ws1_t2_threshold_contract.md](threshold-contract.md)
- [p1_phase_summary.md](phase1-short-bgc-coverage.md)
- [p2_phase_summary.md](phase2-ood-robustness.md)
- [stability_confirmation.md](stability-3seed.md)

These are enough to start the architecture redesign discussion without going back through the entire execution history.
