# Baseline Table — NP-Master BGC Detection

Originally frozen 2026-04-23; updated 2026-04-26 with per-token U-Net E10/E11 readouts. Methods (rows) x datasets (column blocks). Hand-consolidated from source artifacts in `baseline_table.json._meta._sources`. Any candidate architecture must be added as a new row, not edited over these numbers.

Primary metrics (per `memory/feedback_primary_metrics.md`): `dedup_f1`, `bench_precision`, `bench_recall`. `bench_f1` alone is fragmentation-gameable and kept only as a secondary diagnostic.

## 9-genome (isolate, held-out lockbox; do NOT tune on this)

| Method              | n_pred | dedup_F1 (CI95)          | bench_prec | bench_rec | strict_F1 (CI95)        | recall <5kb | 5–10kb | 10–20kb | 20–50kb | >=50kb |
|---------------------|-------:|--------------------------|-----------:|----------:|-------------------------|-------------|--------|---------|---------|--------|
| v4.1-hysteresis     | 246    | **0.4878** (0.345–0.609) | 0.5976     | 0.4590    | 0.4299 (0.284–0.564)    | 0.146       | 0.393  | 0.435   | 0.689   | 0.864  |
| v4.1-learned        | 363    | **0.3323** (0.224–0.437) | 0.2617     | 0.5279    | 0.2455 (0.155–0.339)    | 0.167       | 0.483  | 0.482   | 0.787   | 0.955  |
| per_token_v1 (thr=0.4) | 462 | **0.1936** (0.115–0.262) | 0.6407     | 0.1443    | 0.1147                  | 0.083       | 0.180  | 0.106   | 0.148   | 0.273  |
| E10 U-Net (thr=0.3, OER recall-first) | 333 | **0.4561** (0.316–0.562) | 0.4444 | 0.4984 | 0.4044 | 0.146 | 0.472 | 0.482 | 0.721 | 0.818 |
| E10 U-Net (thr=0.4, OER conservative) | 314 | **0.4664** (0.324–0.573) | 0.4809 | 0.4852 | 0.4136 | 0.104 | 0.472 | 0.471 | 0.705 | 0.818 |
| E10 U-Net (thr=0.5, default reference) | 303 | **0.4751** (0.331–0.583) | 0.4983 | 0.4852 | 0.4243 | 0.104 | 0.472 | 0.471 | 0.705 | 0.818 |
| E11 U-Net (thr=0.3, comparison) | 321 | **0.4470** (0.300–0.553) | 0.4424 | 0.4852 | 0.3898 | 0.125 | 0.449 | 0.471 | 0.721 | 0.818 |
| E11 U-Net (thr=0.4, comparison) | 308 | **0.4500** (0.311–0.554) | 0.4870 | 0.4623 | 0.3850 | 0.125 | 0.449 | 0.412 | 0.689 | 0.818 |
| E11 U-Net (thr=0.5, default reference) | 298 | **0.4544** (0.317–0.556) | 0.5067 | 0.4492 | 0.3914 | 0.125 | 0.438 | 0.400 | 0.656 | 0.818 |
| antiSMASH           | —      | — (skipped; training signal) | —      | —         | —                       | —           | —      | —       | —       | —      |

E10/E11 rows above use `scripts/evaluate/per_token_readout.py` and canonical `benchmark.nine_genome.eval_regions`.
Thresholds 0.3 and 0.4 are not chosen from 9g: they are the OER recall-first and OER conservative points, respectively.
The 0.5 rows are default-threshold references. The checkpoint/config family came from a round2 grid that was later read out on 9g, so these rows are threshold-clean diagnostics, not yet fully pre-registered formal candidates.

## OER004256 (metagenome, contigs >= 10 kb)

| Method          | n_regions | median_len_bp | AS-overlap count | AS-overlap rate (AS-recall) | n_as_missed | per-type count (Polyketide / NRP / Terpene / RiPP / Saccharide / Alkaloid / Other) |
|-----------------|----------:|--------------:|-----------------:|---------------------------:|------------:|------------------------------------------------------------------------------------|
| v4.1-hysteresis | 254       | 12,288        | 239 *            | **0.12** (AS-recall)        | 15          | 150 / 78 / 17 / 4 / 1 / 0 / 4  (caveat: partial AS coverage at wrap-up — 1 of 7 samples had antiSMASH at that time) |
| v4.1-learned    | 1,230     | 5,974         | 801              | **0.37** (AS-recall)        | 429         | 200 / 125 / 752 / 74 / 2 / 0 / 77  (Alkaloid=0; 6/7 types represented; Terpene-dominant due to short-arm) |
| antiSMASH       | 1,976     | 13,573        | 1,976            | 1.00 (self)                 | 0           | 326 / 215 / 802 / 351 / 36 / 8 / 238  (all 7 types present; sum=1976; multi-product regions collapsed via priority PKS>NRP>RiPP>Terpene>Sacch>Alk>Other) |

\* For v4.1-hysteresis on OER: the 239/254 figure is *v4.1 regions that overlap any antiSMASH region* (the inverse of AS-recall). The 0.12 AS-recall number (fraction of antiSMASH regions hit) is the comparable cross-method metric; see `experiments/OER004256/v4_1_learned/README.md` headline table.

### Per-token U-Net OER readout (2026-04-26)

These rows use `scripts/evaluate/oer_per_token_compare.py`, which reports antiSMASH recall under asymmetric 50% overlap (`overlap / AS_region_len >= 0.5`) plus reverse-direction novel counts. Matching 9g readouts are now listed above; promotion still requires a fully pre-registered checkpoint+threshold selection rule that does not use 9g.

| Method | threshold | n_pred | AS TP / 1,976 | AS-recall | novel | pred AS-overlap rate | median_len_bp | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E10 U-Net | 0.3 | 1,377 | 616 | **0.3117** | 470 | 0.6587 | 9,828 | `experiments/step2_per_token/OER_UNET_READOUT.md` |
| E10 U-Net | 0.4 | 1,271 | 531 | 0.2687 | 388 | 0.6947 | 9,384 | same |
| E10 U-Net | 0.5 | 1,194 | 491 | 0.2485 | 325 | 0.7278 | 8,192 | same |
| E11 U-Net | 0.3 | 1,154 | 541 | 0.2738 | 382 | 0.6690 | 10,240 | same |
| E11 U-Net | 0.4 | 1,083 | 472 | 0.2389 | 321 | 0.7036 | 10,033 | same |
| E11 U-Net | 0.5 | 1,007 | 334 | 0.1690 | 265 | 0.7368 | 6,182 | same |
| v4.1-learned, recomputed in same script | production OP | 1,230 | 408 | 0.2065 | 429 | 0.6512 | 5,973 | same |
| v4.1-hysteresis, recomputed in same script | frozen OP | 254 | 187 | 0.0946 | 15 | 0.9409 | 12,288 | same |

The legacy v4.1-learned OER headline (`0.37`) and this same-script readout (`0.2065`) are not interchangeable until the metric/path difference is audited. For the E10/E11 comparison above, use only the same-script rows.

## TBD cells (resolved 2026-04-23)

- **9g v4.1-learned** — FILLED. n_pred=363, dedup_f1=0.3323 (CI 0.224–0.437). Note: the earlier "0.461" entry in `docs/REGRESSION_LEDGER.md #16` / `full_pipeline_report.md` is **not** comparable — it came from `full_pipeline_eval.py`'s greedy 50%-either-side `dedup_metrics`, while this table's `dedup_f1` uses `benchmark.nine_genome.eval_regions` (unique-GT-TP / n_pred denominator), the same code path that produced v4.1-hysteresis' 0.4878. Apples-to-apples, v4.1-learned is ~0.15 F1 **below** v4.1-hysteresis, driven by low bench precision (0.26 vs 0.60) at comparable recall. The learned decoder over-predicts short regions. `bucket_recall` shows modest gains on 5–20 kb but <5 kb still floors at 0.167 (same blind spot as hysteresis).
- **OER v4.1-hysteresis per-type count**: values from `wrap_up_report.json` are kept as-is — caveat about partial AS coverage at wrap-up time preserved in the JSON.
- **OER antiSMASH per-type** — FILLED. 326 / 215 / 802 / 351 / 36 / 8 / 238. All 7 types represented. Terpene dominates (40.6 %) driven by `terpene` + `terpene-precursor` products. Parser rule + product→type map stored in `baseline_table.json -> antismash.oer004256_ge10kb.per_type_count._product_to_type_map`.
- **OER antiSMASH median_len_bp** — FILLED. 13,573 bp (vs v4.1-hysteresis' 12,288 and v4.1-learned's 5,974).

## Pass/fail gates for future candidate architectures

A candidate row must satisfy ALL of:

- **9g `dedup_F1` >= 0.458** (v4.1-mainline 0.4878 minus 0.03 tolerance; outside the CI lower bound is not acceptable as "matching").
- **OER `n_regions` >= 1,400** (v4.1-learned ceiling minus ~13%; fewer regions = loss of metagenomic recall).
- **OER AS-overlap (AS-recall) >= 0.50** (must exceed v4.1-learned's 0.37 meaningfully).
- **All 7 types represented** in OER per-type count (Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other); zero counts in any type is an automatic fail.

## Prior null result (rejected; do not revisit without new evidence)

Zero-shot Evo2 LL (8 scoring variants, w=16384/s=8192) produces best 9g dedup_F1=0.1296 — 3.8x worse than v4.1-mainline. Pure LL signal rejected. See `experiments/evo2_first_principles/9g_eval_ll_zscore_pos/sweep_summary.json`.
