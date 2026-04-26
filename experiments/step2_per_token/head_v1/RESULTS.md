# Per-Token BGC Head v1 — 9-genome Readout

**Date**: 2026-04-24
**Checkpoint**: `experiments/step2_per_token/head_v1/runs/ckpn3_h64_seed0/best.pt`
**Training**: PerTokenHead (1D CNN, 3 dilated blocks, hidden=64, ~49K params), seed 0, best epoch 8. Val token F1=0.302, precision=0.82, recall=0.18.
**Features**: per-token Evo2 embeddings, 9g w=8192/s=2048, T_latent=1024 @ K=8 bp/token.
**Decode**: `src/per_token/decode.py::decode_all` (per-nt track via nearest-neighbour upsample of token probs × K=8, contig-level threshold + min_len_bp=2000 + max_gap_bp=500).
**Evaluator**: `scripts/evaluate/per_token_readout.py` (canonical `benchmark.nine_genome.eval_regions`, bench mode = asymmetric 50%).
**GT**: `/data/syh/benchmark/nine_genome/ground_truth.json` (305 GT regions across 9 genomes).

## Threshold sweep

| threshold | n_pred | dedup_F1 (CI95)        | bench_P | bench_R | strict_F1 | <5kb  | 5–10kb | 10–20kb | 20–50kb | >=50kb | max_preds_per_GT |
|-----------|-------:|------------------------|--------:|--------:|----------:|-------|--------|---------|---------|--------|-----------------:|
| 0.3       | 1421   | 0.1661 (0.122–0.207)   | 0.2829  | 0.2361  | 0.0823    | 0.229 | 0.292  | 0.141   | 0.279   | 0.273  | 17               |
| 0.4       | 462    | 0.1936 (0.115–0.262)   | 0.6407  | 0.1443  | 0.1147    | 0.083 | 0.180  | 0.106   | 0.148   | 0.273  | 15               |
| 0.5       | 346    | 0.1752 (0.096–0.241)   | 0.7832  | 0.1148  | 0.1075    | 0.083 | 0.112  | 0.071   | 0.148   | 0.273  | 14               |
| 0.7       | 282    | 0.1559 (0.086–0.213)   | 0.8121  | 0.0984  | 0.1022    | 0.021 | 0.090  | 0.071   | 0.148   | 0.273  | 13               |

## Baselines (for reference)

| method             | n_pred | dedup_F1 | bench_P | bench_R | strict_F1 |
|--------------------|-------:|---------:|--------:|--------:|----------:|
| v4.1-hysteresis    | 246    | 0.4878   | 0.5976  | 0.4590  | 0.4299    |
| v4.1-learned       | 363    | 0.3323   | 0.2617  | 0.5279  | 0.2455    |
| **per_token_v1 (best, thr=0.4)** | **462** | **0.1936** | **0.6407** | **0.1443** | **0.1147** |

## Honest assessment

**FAIL.** Per-token v1 does **not** pass the gate (dedup_F1 ≥ 0.458). Best operating point (thr=0.4) is dedup_F1=0.194, roughly **2.5× worse than v4.1-hysteresis** (0.488) and **1.7× worse than v4.1-learned** (0.332). The 95% CIs at every threshold sit entirely below the gate line.

**Diagnosis.** The failure mode is not a coord-mapping bug (regions are produced at every threshold, of sensible lengths, spread across genomes). It is the **recall ceiling** inherited from the training checkpoint:

- Val token recall was 0.18; here bench recall ranges 0.24 → 0.10 across thresholds. The head simply does not fire on enough positive tokens.
- Bench precision is actually *strong* at high threshold (0.81 @ thr=0.7, beats v4.1-hysteresis' 0.60) — the model is right when it commits, but it commits too rarely.
- Heavy fragmentation: `max_preds_per_GT` = 13–17 means single GT regions are being split into many small calls, so `dedup_precision` (0.13–0.38) is far below `bench_precision` (0.28–0.81). The `dedup_p / bench_p` ratio is 0.45–0.48 at thr ≥ 0.4, which fails the ≥ 0.75 short-core-hack gate — even the apparent precision is inflated by duplication.
- Length-bucket recall collapses at high threshold (e.g. <5 kb falls from 0.229 at thr=0.3 to 0.021 at thr=0.7), and is dominated by >= 50 kb (0.273 is constant — a few long regions anchor what little recall there is). Compare to v4.1-hysteresis which gets 0.864 on >= 50 kb and 0.689 on 20–50 kb.

**Verdict.** This checkpoint (val F1 0.302, recall 0.18) is **undertrained** for downstream decode. The ceiling is set by the training objective, not decoder parameters — no threshold choice can recover recall that the head didn't learn. Do not promote. Next step: retrain with higher recall (class-balanced loss, longer schedule, or a larger head) before re-running this pipeline.

## Files

- Probs: `experiments/step2_per_token/head_v1/runs/ckpn3_h64_seed0/9g_probs/*.probs.npz`
- Regions: `experiments/step2_per_token/head_v1/9g_regions/regions_thr{0.3,0.4,0.5,0.7}.csv`
- Readouts: `experiments/step2_per_token/head_v1/9g_readouts/readout_thr{0.3,0.4,0.5,0.7}.json`
