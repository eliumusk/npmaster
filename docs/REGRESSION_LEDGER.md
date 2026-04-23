# Regression Ledger

The numbers any refactor must NOT break. Frozen 2026-04-23 before the scripts/src reorganization.

If a commit after this date reports a number outside the listed CI (or changes a point estimate by more than rounding), that is a regression — revert.

## Primary detection metrics (9-genome)

| # | Metric | Value | Source artifact | Reproduce script |
|---|---|---|---|---|
| 1 | v4.1 multi-scale ensemble dedup_F1 | **0.4878** (CI95 0.345–0.609) | `experiments/evo2_first_principles/v4_1_final_readout.json` | `scripts/evaluate/v4_1_multiscale.py` |
| 2 | v4.1 bench_F1 / precision / recall | 0.5192 / 0.5976 / 0.4590 | same | same |
| 3 | v4.1 strict_F1 | 0.4299 (CI95 0.284–0.564) | same | same |
| 4 | v4 single-scale (w=8192) dedup_F1 | 0.4869 (CI95 0.344–0.611) | `v4_1_final_readout.json` → `primary_only` block | `scripts/evaluate/honest_readout.py` |
| 5 | defrag_v2_ms10_s099 dedup_F1 (w=8192/s=2048, start=0.99/extend=0.95/ms=10) | **0.4385** (precision 0.4714, n_pred 140) | `configs/operating_points.yaml` lines 58–79 | `scripts/evaluate/reproduce_defrag_v2_nine_genome.py` |

Frozen operating points: `configs/experiments/evo2_first_principles_v4_1_frozen.yaml`, `configs/operating_points.yaml`.

## Classification metrics (9-genome vs antiSMASH, IoU ≥ 0.3)

| # | Metric | Value | Source |
|---|---|---|---|
| 6 | defrag_v2 CNN per-type top-1 | **0.827** (top-any 0.971, n=139 pairs) | `experiments/nine_genome/defrag_v2_classification.json` |
| 7 | v4.1 Stage-2 LR per-type top-1 | ~0.705 (top-any 0.827) | eval script output |
| 8 | defrag_v2 per-type detail (top 4) | Terpene F1=0.973 · Polyketide 0.843 · RiPP 0.833 · NRP 0.78 | same as #6 |

Note on #6: class-order bug fixed 2026-04-23 across all CNN inference scripts. See `memory/reference_cnn_class_order.md`.

## OER004256 handoff (wet-lab deliverable)

| # | Metric | Value |
|---|---|---|
### v4.1 hysteresis (frozen baseline, 2026-04-23)

| # | Metric | Value |
|---|---|---|
| 9 | v4.1 regions detected | 254 (median 12.3 kb) |
| 10 | v4.1 type distribution | Polyketide 150 · NRP 78 · Terpene 17 · RiPP 4 · Other 4 · Saccharide 1 |
| 11 | defrag_v2 regions detected | 93 (median 26.6 kb) |
| 12 | Inter-tool type agreement | 87.4% (222 / 254) |
| 13 | Wet-lab candidates (both = Terpene or RiPP) | **18** |

Artifacts: `experiments/OER004256/v41_regions_fully_typed.csv`, `experiments/OER004256/OER004256_wrap_up_report.json`.

### v4.1-learned decoder (preferred, 2026-04-23)

| # | Metric | Value |
|---|---|---|
| 14 | v4.1-learned regions detected (prim_thr=0.95, short_thr=0.98) | **1,230** (primary 531 + short 699) |
| 15 | AS-recall on OER | **37%** (vs 12% hysteresis) |
| 16 | 9g dedup F1 vs GT (learned multi-scale) | 0.461 (vs 0.541 hysteresis — regression accepted for metagenomic recall) |
| 17 | 15 original novel candidates preserved | 15 / 15 |
| 18 | Novel (antiSMASH-missed) candidates | **429** |
| 19 | Novel passing HMM+MIBiG quality filter | 263 (61%) |
| 20 | Top wet-lab cohort (STRONG × NOVEL/LIKELY_NOVEL) | **86** |
| 21 | Inter-tool type agreement (v4.1 ∩ defrag_v2) | 80% (987 / 1,230) |
| 22 | Original 15 — HMM STRONG verdict | 14 / 15 |
| 23 | Original 15 — MIBiG novelty verdicts | 0 KNOWN · 3 LIKELY_NOVEL · 12 VARIANT |
| 24 | Prior FDR on 9g (≥0.98 score, AS-missed) — P(GT-hit) | ≥ 0.53 (lower bound) |

Decoder ckpt: `experiments/evo2_first_principles/decoder_v1/runs/c00_cnn3_h32_antismash_bce/decoder.pt` (server-side).
Artifacts: `experiments/OER004256/v4_1_learned/regions_fully_typed.csv`, `experiments/OER004256/v4_1_learned/novel_triage/novel_triage_full.csv`.
See: `docs/learned-decoder.md`, `docs/oer004256-novel-triage.md`.

## Known blind spot (do not attempt to "fix" as part of refactor)

- GCA_000154945.1 <5 kb recall = 7/48 (0.146) across all models. Root-cause in `docs/reports/genome_154945_diagnostic.md`. Needs native w=2048 training (~20 A800-h), not an architecture change.

## Baselines (for comparison context)

- BGC-Prophet 9-genome: dedup_F1 0.3911 · strict_F1 0.3018 · bench_F1 0.3933
- antiSMASH / DeepBGC / GECCO: live in `scripts/evaluate/baselines_nine_genome.py` outputs (server-side data).

## Reproduce commands (post-refactor script paths will change; content must still produce these numbers)

1. **v4.1 honest readout** → produces #1–3
   ```
   python scripts/evaluate/v4_1_multiscale.py \
     --primary-preds experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
     --short-preds   experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
     --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
     --out-json experiments/evo2_first_principles/v4_1_final_readout.json
   ```
2. **defrag_v2 9-genome** → produces #5
   ```
   python scripts/evaluate/reproduce_defrag_v2_nine_genome.py \
     --checkpoint /data/muskliu/npmaster/data/phase3/stage3_hardneg_v1/best/best_tk100-ms0-hnw4.pt \
     --window 8192 --stride 2048 \
     --start-threshold 0.99 --extend-threshold 0.95 --min-support-windows 10 \
     --gt-json /data/syh/benchmark/nine_genome/ground_truth.json
   ```
3. **defrag_v2 classification** → produces #6, #8
   ```
   python scripts/evaluate/classification_defrag_v2_nine_genome.py \
     --v4-1-primary-preds experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
     --v4-1-short-preds   experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
     --defrag-v2-preds-dir /data/muskliu/npmaster/data/phase4/defrag_v2_ms10_s099_preds_stride2048 \
     --antismash-dir /data/syh/benchmark/nine_genome/results/antismash \
     --bgc-annotations /data/muskliu/npmaster/data/bgc_annotations.csv \
     --agg max
   ```
4. **v4.1 classification** → produces #7 (same flags as above, different script name).
5. **OER004256 labeling** → produces #9–13 (both v4.1 and defrag_v2 label scripts; server-side FASTA).

## Verification protocol after refactor

Since #1–8 require server-side data, local verification after the reorganization is limited to:

- `python -m py_compile` on every moved script → must succeed.
- `--help` on each renamed entrypoint → must return without error.
- `pytest tests/` (if tests exist) → must pass.
- OER004256 CSV outputs (#9–13) are committed to the repo — diffable locally.

Full-number regression is deferred to the next server session; any discrepancy there is a blocker.
