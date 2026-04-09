# Discovery-Side Benchmark Results

**Date:** 2026-04-09  
**Current mainline:** H_hardneg_r10  
**Detection Score:** `score_maxtype` (canonical)  
**Evaluation Type:** Full-window  
**Candidate Source:** test_id unmatched predictions, `clean_missed` category (>20kb from any antiSMASH BGC)

---

## 1. Cross-Config Comparison

| Metric | Config E | F_cap08 | G_edgepad | **H_hardneg_r10** |
|--------|----------|---------|-----------|-------------------|
| Sig precision@20 | 80% | 95% | 90% | **90%** |
| Sig precision@50 | 84% | 92% | 92% | **86%** |
| Sig enrichment@20 | 2.0x | 2.7x | 1.6x | **1.8x** |
| Sig enrichment@50 | 1.4x | 1.7x | 1.0x | **1.0x** |
| Sig diversity@20 | — | 13 | 12 | **16** |
| Sig diversity@50 | — | 35 | 37 | **38** |
| Null sig@20 | — | 35% | 55% | **50%** |
| Null sig@50 | — | 54% | 62% | **54%** |
| Clean missed pool | — | 1419 | 1470 | **934** |

**Interpretation:** H_hardneg_r10 is more selective (934 clean candidates vs 1419–1470 in earlier configs). Top-20 precision holds at 90%. @50 drops to 86% (3 fewer sig hits than F_cap08). But diversity is the highest (38 profiles) — the candidate pool is more varied, not just repeated NRPS/PKS clusters. Enrichment recovers from G_edgepad's dip because null signature rate returns to 54%.

---

## 2. Current Mainline: H_hardneg_r10 Detail

### Candidate Pool

| Category | Count |
|----------|-------|
| Total unmatched | 1274 |
| Clean missed (>20kb) | 934 |
| Boundary expansion (<5kb) | 230 |
| Ambiguous (5-20kb) | 110 |

### Precision@K (BGC Signature Domains)

Curated 27-domain signature set (KS, AT, TE, C, A, PCP, terpene synthase, etc.)

| K | Candidates | Null | Enrichment |
|---|-----------|------|------------|
| **20** | **90%** (18/20) | 50% (10/20) | **1.8x** |
| **50** | **86%** (43/50) | 54% (27/50) | **1.6x** |

### Family Diversity

| K | Candidates (sig profiles) | Null (sig profiles) | Ratio |
|---|--------------------------|---------------------|-------|
| 20 | **16** | 10 | 1.6x |
| 50 | **38** | 21 | 1.8x |

38 unique signature profiles at top-50.

### Top Signature Domains (top-50 candidates)

| Domain | Name | Frequency |
|--------|------|-----------|
| PF00550 | PCP/T (peptidyl carrier) | 26/50 |
| PF00501 | AMP-binding (adenylation) | 21/50 |
| PF00067 | p450 | 12/50 |
| PF08659 | KR_C | 12/50 |
| PF08241 | Methyltransf_11 | 11/50 |
| PF00109 | KS (ketosynthase) | 11/50 |
| PF02801 | KS_C-terminal | 11/50 |
| PF00668 | C (condensation) | 11/50 |
| PF00106 | SDR (short-chain dehydrogenase) | 11/50 |
| PF13561 | Enoyl_CoA_hyd | 9/50 |

### Type Distribution (top-50 clean missed)

| Type | Count |
|------|-------|
| NRP | 17 |
| Polyketide | 17 |
| Terpene | 8 |
| RiPP | 5 |
| Alkaloid | 2 |
| Saccharide | 1 |

---

## 3. Historical: F_cap08 (for reference)

F_cap08 had the highest top-end purity before hard-negative training.

| K | Candidates | Null | Enrichment |
|---|-----------|------|------------|
| 20 | 95% (19/20) | 35% (7/20) | 2.7x |
| 50 | 92% (46/50) | 54% (27/50) | 1.7x |

Diversity@50: 35 profiles. Top domains: PF00550 (21/50), PF00501 (18/50), PF08241 (15/50).

---

## 4. Known Limitations

- **Biosynthetic Pfam (2391 families) is not discriminative** — both candidates and null hit ~100%. Only the curated 27-domain signature set discriminates.
- **Null distribution is genome-matched but not phylum-controlled.**
- **Only 50 candidates validated per config.** Full coverage would provide more robust statistics.
- **GECCO/DeepBGC incremental yield not computed** — tools not installed on server.
- **Null signature rates vary across runs** (35%–62%) due to random negative-region sampling. This affects enrichment ratio stability.

---

## 5. Canonical Artifacts

| Config | Results JSON |
|--------|-------------|
| F_cap08 | `/data/muskliu/np_master/data/stage4_results/ws_discovery_benchmark_cap08/discovery_benchmark.json` |
| G_edgepad | `/data/muskliu/np_master/data/stage4_results/p1_t3_discovery_benchmark/discovery_benchmark.json` |
| **H_hardneg_r10** | **`/data/muskliu/np_master/data/stage4_results/p2_t2_hardneg_r10_discovery/discovery_benchmark.json`** |
| Script | `scripts/eval/ws_discovery_benchmark.py` |
