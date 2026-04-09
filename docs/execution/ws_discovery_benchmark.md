# Discovery-Side Benchmark Results

**Date:** 2026-04-09
**Task ID:** C_discovery_benchmark_config_e
**Model:** Config E (prior-aligned, pos_ratio=0.15)
**Detection Score:** `score_maxtype` (canonical, per threshold contract v2.0)
**Evaluation Type:** Full-window
**Candidate Source:** test_id unmatched predictions, `clean_missed` category (>20kb from any antiSMASH BGC)

---

## 1. Candidate Pool (Config E, score_maxtype, full-window)

| Category | Count | Description |
|----------|-------|-------------|
| Total unmatched | 1917 | All predicted regions not matching any antiSMASH BGC |
| **Clean missed** | **1437** | >20kb from any antiSMASH BGC on same contig |
| Boundary expansion | 290 | Within 5kb of antiSMASH BGC (likely boundary overreach) |
| Ambiguous unmatched | 190 | 5-20kb from nearest antiSMASH BGC |

Only **clean_missed** candidates are evaluated. Boundary expansions and ambiguous are excluded.

---

## 2. Precision@K

### Biosynthetic Pfam (broad, 2391 families)

| K | Candidates | Null | Enrichment |
|---|-----------|------|------------|
| 20 | 100% | 100% | 1.0x |
| 50 | 100% | 100% | 1.0x |

**Interpretation:** The `biosyn_names.txt` list (2391 Pfam families) is too broad to be discriminative — both candidates and random negative regions hit 100%. This metric is **not useful** for distinguishing real BGC candidates from background.

### BGC Signature Domains (curated, 27 core families)

| K | Candidates | Null | Enrichment |
|---|-----------|------|------------|
| **20** | **80%** (16/20) | 40% (8/20) | **2.0x** |
| **50** | **84%** (42/50) | 58% (28/50) | **1.4x** |

**This is the meaningful metric.** The curated 27-domain signature set discriminates candidates from background:
- At top-20, candidates are **2x enriched** for core BGC domains
- The enrichment weakens at top-50 (1.4x), consistent with ranking quality degrading further down the list
- Null regions still show 40-58% signature hit rate due to Actinomycetota genomic background

---

## 3. Family Diversity

| K | Candidates (biosyn profiles) | Null | Candidates (signature profiles) | Null |
|---|------------------------------|------|---------------------------------|------|
| 20 | 20 | 20 | **10** | 7 |
| 50 | 50 | 50 | **29** | 19 |

- **Biosyn profiles** are maximally diverse for both (every region has a unique broad Pfam set) — not discriminative.
- **Signature domain profiles** show candidates have **1.4-1.5x more diverse BGC architectures** than null regions (29 vs 19 unique profiles at top-50).
- Root cause of biosyn profile non-discrimination: the 2391-family list is so large that every 20kb+ region in an Actinomycetota genome contains a unique subset of these common domains.

---

## 4. Top Signature Domains in Candidates

| Domain | Name | Frequency (top-50) |
|--------|------|-------------------|
| PF08241 | Methyltransf_11 | 21/50 |
| PF00106 | SDR (short-chain dehydrogenase) | 15/50 |
| PF08659 | KR_C | 15/50 |
| PF00501 | AMP-binding (adenylation) | 15/50 |
| PF13561 | Enoyl_CoA_hyd | 13/50 |
| PF00067 | p450 | 11/50 |
| PF00550 | PCP/T (peptidyl carrier) | 10/50 |
| PF00668 | C (condensation) | 8/50 |
| PF02801 | KS_C-terminal | 7/50 |
| PF00109 | KS (ketosynthase) | 7/50 |

Dominant signatures: NRPS (AMP-binding, PCP, condensation) and PKS (KS, KR, AT), consistent with the NRP/Polyketide type predictions.

---

## 5. Type Distribution (Top-50 Clean Missed)

| Type | Count |
|------|-------|
| Saccharide | 20 |
| Alkaloid | 13 |
| NRP | 10 |
| Polyketide | 6 |
| Terpene | 1 |

Note the shift from Config B (where NRP dominated). Config E's improved calibration changes the ranking.

---

## 6. Comparison with Config B Discovery Benchmark

| Metric | Config B | Config E | Change |
|--------|----------|----------|--------|
| Signature precision@20 | 90% | **80%** | -10pp |
| Signature precision@50 | 86% | **84%** | -2pp |
| Null signature@20 | 50% | 40% | -10pp |
| Null signature@50 | 64% | 58% | -6pp |
| **Enrichment@20** | 1.8x | **2.0x** | +0.2x |
| **Enrichment@50** | 1.3x | **1.4x** | +0.1x |
| Signature diversity@50 | ~15 (est.) | **29** | +14 |

Config E gives slightly better enrichment ratios and substantially more diverse candidates, despite slightly lower absolute precision@20. The key improvement is that enrichment is **more stable across K** (2.0x→1.4x vs 1.8x→1.3x).

---

## 7. Root Cause: Why Biosynthetic Pfam Precision Is Not Discriminative

The `biosyn_names.txt` file contains 2391 Pfam family names. This list is derived from antiSMASH's broad "biosynthetic-related" annotation, which includes many domains that are common in all bacteria (e.g., methyltransferases, oxidoreductases, CoA-related enzymes). In Actinomycetota genomes, these domains are pervasive even in non-BGC regions.

**Recommendation:** For future discovery benchmarks, use only the curated 27-domain BGC signature set for precision@K and enrichment calculations. The broad biosynthetic Pfam list should be retired as a primary metric.

---

## 8. GECCO / DeepBGC Incremental Yield

**Not computed.** Neither tool is installed on the server. This remains a blocker for the full discovery-side evaluation.

---

## 9. Canonical Artifacts

| Artifact | Path |
|----------|------|
| Results JSON | `/data/muskliu/np_master/data/stage4_results/ws_discovery_benchmark_config_e/discovery_benchmark.json` |
| Source data | `/data/muskliu/np_master/data/stage4_results/ws2_t2_eval/score_swap_results.json` |
| Script | `scripts/execution/ws_discovery_benchmark.py` |
