# NP-Master

Biosynthetic Gene Cluster (BGC) detection and discovery on bacterial genomes and metagenomes. Frozen Evo2 7B embeddings + a light scorer (LR_A1 logistic regression) + a multi-scale decoder (primary w=8192 + short-arm w=4096, union-dedup at IoU=0.3).

**Two decoders, two OPs** — pick by target:

| target | operating point | decoder | headline metric | entrypoint |
|---|---|---|---|---|
| **9-genome benchmark** (paper results) | `v4.1 multi-scale hysteresis` | hand-tuned rule | dedup F1 = **0.4878** (9g) | `scripts/decode/v4_1.py`, `scripts/evaluate/v4_1_multiscale.py` |
| **Metagenomes** (OER004256 + similar) | **`v4.1-learned`** (2026-04-23) | 1D CNN, trained | 1,230 OER regions (vs hysteresis 254), 37% AS-recall (vs 12%) | `scripts/decode/v4_1_learned.py` |

The learned decoder closes the ~8× antiSMASH:v4.1 region-count gap that the hand-tuned `min_support_windows=3` rule opens on short metagenomic contigs. Under the canonical 9g metric it is a metagenome-only tradeoff: 9g dedup_F1 drops from v4.1-hysteresis 0.488 to 0.332. The older 0.461 vs 0.541 numbers use a different greedy-pairing metric and are not comparable to the formal table. See [`docs/learned-decoder.md`](docs/learned-decoder.md).

For BGC-type classification, `defrag_v2`'s CNN head (top-1 82.7% on 9g) is stronger than v4.1's Stage-2 LR (70.5%) — both type annotations are merged via the existing `scripts/classify/{v4_1,defrag_v2}_regions.py` scripts. defrag_v2 detection is **retired** (redundant vs v4.1-learned's 1,230 ≫ 93 regions); its CNN remains in the pipeline only as a second-opinion type labeler.

## Current Status (9-genome, bench mode)

**9-genome is the formal benchmark** (human-curated GT from BGC-Prophet paper, no 9g-derived tuning). Two families of NP-Master candidates are frozen in `configs/operating_points.yaml` — the Evo2-LR multi-scale ensemble (**v4.1**, mainline since 2026-04-22) and the Route-B context CNN (**defrag_v2**, retained for stronger type classification).

Full 9-genome head-to-head (honest readout, no 9g tuning for v4.1):

| Tool / candidate | bench R | bench P | **bench F1** | dedup F1 | strict F1 | n_pred | median pred len |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 | 0.3911 | 0.3018 | 170 | 22.7 kb |
| **NP-Master `v4.1 multi-scale`** ⭐ | **0.4590** | **0.5976** | **0.5192** | **0.4878** | **0.4299** | 246 | — |
| NP-Master `v4-single (w=8192)` | — | — | — | 0.4869 | — | — | — |
| NP-Master `defrag_v2_ms10_s099` | 0.4098 | 0.4714 | 0.4385 | 0.4385 | 0.287 | 140 | 38.9 kb |
| NP-Master `defrag_v2_ms10_s0995` | 0.3934 | 0.4851 | 0.4345 | 0.4345 | 0.292 | 134 | 38.9 kb |

**v4.1 is the mainline for benchmarking and reporting** (dedup F1 **0.488**, bench F1 **0.519**, strict F1 **0.430** — each the highest honest number we have on 9g). Point estimates with 95% bootstrap CIs in `experiments/evo2_first_principles/v4_1_final_readout.json`. Source: `docs/REGRESSION_LEDGER.md` lines 1-5.

`defrag_v2_ms10_s099` remains the second-strongest detector and its CNN type classifier (top-1 82.7%) is stronger than v4.1's Stage-2 LR (70.5%); the OER004256 pipeline uses **v4.1 for detection** and **defrag_v2 for type labels** as a cross-validation signal.

Lockbox-only check (defrag_v2, 4 held-out genomes, frozen OP):

| Tool / candidate | bench R | bench P | **bench F1** | dedup F1 | n_pred |
|---|---:|---:|---:|---:|---:|
| BGC-Prophet | 0.3630 | 0.4205 | 0.3896 | 0.3896 | 88 |
| **NP-Master `defrag_v2_ms10_s099`** | **0.4370** | **0.4697** | **0.4528** | **0.4528** | 66 |
| NP-Master `defrag_v2_ms10_s0995` | 0.4222 | 0.4688 | 0.4443 | 0.4443 | 64 |

Head-to-head against all three external BGC tools on the same dev split (170 GT regions across 5 genomes, stride-2048 inference, decoder frozen per operating point):

| Tool / candidate | bench R | bench P | **bench F1** | **loose F1** | strict F1 | dedup F1 | n_pred | median pred len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGC-Prophet                    | 0.335 | 0.488 | 0.397 | 0.438 | **0.302** | 0.393 | 82  | 21.4 kb |
| antiSMASH                      | 0.541 | 0.161 | 0.249 | 0.522 | 0.154     | 0.249 | 155 | 29.8 kb |
| DeepBGC                        | 0.671 | 0.070 | 0.126 | 0.272 | 0.089     | 0.126 | 574 | 5.3 kb  |
| NP-Master Stage3 default       | 0.541 | 0.282 | 0.371 | —     | —         | 0.271 | 255 | —       |
| **NP-Master `defrag_v2_ms10_s099`**  | 0.388 | 0.473 | **0.426** | **0.519** | 0.287 | 0.426 | 74  | 36.9 kb |
| **NP-Master `defrag_v2_ms10_s0995`** | 0.371 | 0.500 | **0.426** | 0.504     | 0.292 | 0.426 | 70  | 38.9 kb |

Reading, without cherry-picking:

- **bench F1** (the BGC-Prophet paper's headline metric, asymmetric 50% overlap): NP-Master > BGC-Prophet > antiSMASH > DeepBGC. Both `defrag_v2_ms10_*` tie at 0.426.
- **loose F1** (symmetric 50% with `min(pred,gt)` denominator): NP-Master ms10/s099 (0.519) ≈ antiSMASH (0.522) in a near tie, both clearly above BGC-Prophet (0.438) and DeepBGC (0.272).
- **strict / IoU F1** (symmetric 50% with `max(pred,gt)` denominator): BGC-Prophet (0.302) narrowly beats NP-Master (0.287 / 0.292). NP-Master's predictions are wider than the tight-core GT, which is penalised harder here than under bench.
- **recall**: DeepBGC (0.671) and antiSMASH (0.541) out-recall NP-Master (0.388) and BGC-Prophet (0.335), but their n_pred is 574 and 155 respectively versus our 74 — they pay for recall with precision that collapses under bench.

`bench` = asymmetric 50% overlap (both recall and precision sides; see `docs/evaluation-protocol.md`). GT comes from the BGC-Prophet 9-genome test set and uses a gene-level "tight core" convention (median dev GT length ≈ 9.9 kb, 9-genome median ≈ 10.9 kb, no flanks). Tools whose native output includes flanks (antiSMASH, DeepBGC) are structurally penalised on bench precision — see the boundary plots below and the loose-F1 column for a length-normalised view.

`dedup F1` de-duplicates multiple precision true positives assigned to the same GT BGC. This is now mandatory because an earlier strict-decoder sweep inflated raw precision by fragmenting regions. Both `defrag_v2_ms10_*` candidates have `duplicate_precision_tp=0` and `max_preds_per_gt=1` on dev, so the precision gain is not coming from repeated short fragments. The gain comes from requiring sustained evidence across `min_support_windows=10` at `window=8192 / stride=2048`, i.e. a minimum prediction span of 26,624 bp.

**Honest caveats:**

- The 5-genome dev-split comparison above is an older (defrag_v2-era) snapshot kept for context and baseline reproducibility. The **all-9 table at the top is the live formal readout**; v4.1 multi-scale is the current promoted mainline.
- `defrag_v2_ms10_s099` and `_s0995` remain frozen in `configs/operating_points.yaml` as precision-oriented alternates and as the source of the stronger type classifier (CNN top-1 82.7%).
- Boundary tightness is still a known weakness across both families: NP-Master predictions are wider than BGC-Prophet's under strict/IoU. v4.1 improves all-9 strict F1 to 0.430 (vs 0.302 BGC-Prophet), narrowing this gap but not fully closing it.
- NP-Master misses `GCA_000154945.1` almost entirely (R ≈ 0.03) under every decoder setting, and BGC-Prophet misses it completely (R ≈ 0.03, F1 = 0). This genome is a shared model-domain blind spot, not a NP-Master-specific regression. Root-cause: short (<5 kb) BGCs concentrated in one genome; needs native w=2048 training. Deferred per `docs/next-phase.md`.

### Boundary plots (dev split, longest contig per genome)

Each strip is one tool's BGC calls along the longest contig, with GT in black. Source: `scripts/plot/genome_bgc_boundaries.py`. PNGs live under `figures/dev_boundaries/`.

| Genome | Plot |
|---|---|
| GCA_000154945.1 (collapse — shared blind spot with BGC-Prophet) | [`figures/dev_boundaries/GCA_000154945.1.png`](figures/dev_boundaries/GCA_000154945.1.png) |
| GCA_000156475.1 | [`figures/dev_boundaries/GCA_000156475.1.png`](figures/dev_boundaries/GCA_000156475.1.png) |
| GCA_000158875.1 | [`figures/dev_boundaries/GCA_000158875.1.png`](figures/dev_boundaries/GCA_000158875.1.png) |
| GCA_000158915.1 (most crowded GT) | [`figures/dev_boundaries/GCA_000158915.1.png`](figures/dev_boundaries/GCA_000158915.1.png) |
| GCA_000568915.1 | [`figures/dev_boundaries/GCA_000568915.1.png`](figures/dev_boundaries/GCA_000568915.1.png) |

## Layout

See `docs/CONVENTIONS.md` for the rules; here is the summary.

```
npmaster/
├── configs/                      # YAML/JSON run parameters
│   ├── mainline.yaml             # Route B training
│   ├── operating_points.yaml     # frozen decoder OPs for 9-genome
│   └── experiments/              # frozen first-principles configs
├── src/                          # IMPORTABLE LIBRARY (no CLIs, no argparse)
│   ├── train.py  evaluate.py  utils.py
│   ├── model/                    # ContextFullModel + dilated CNN + projector
│   ├── data/                     # ChunkDataset + ContextDataset
│   ├── evo2_lr/                  # Evo2 embedding + mean-pool LR pipeline
│   ├── decoding/                 # region decoders (hysteresis, v4.1 ensemble)
│   ├── benchmark/                # 9-genome eval + OP sweep math
│   └── classification/           # per-type Stage-2 LR + region-pairing helpers
├── scripts/                      # CLI ENTRY POINTS (one verb per dir)
│   ├── extract/  inference/  decode/  classify/
│   ├── evaluate/  train/  plot/  diagnose/  utils/
│   └── baselines/                # antiSMASH / DeepBGC / GECCO / BGC-Prophet parsers
├── experiments/                  # OUTPUTS (JSON / CSV / plots, per-target)
│   ├── nine_genome/              # benchmark artifacts
│   ├── evo2_first_principles/    # v4.1 design campaign
│   └── OER004256/                # wet-lab handoff package
├── tests/
├── docs/
│   ├── CONVENTIONS.md            # layout + naming rules
│   ├── REGRESSION_LEDGER.md      # frozen numbers any refactor must preserve
│   ├── mainline.md  operating-points.md  evaluation-protocol.md  ...
│   └── reports/
├── _backups/                     # local-only ckpt backups (never deleted, never pushed)
└── _archive/                     # retired code/results (gitignored, kept for provenance)
```

Data, checkpoints, wandb, raw results are **server-only** (gitignored). Local clone is code + configs + docs + promoted JSON/CSV outputs.

## Quick Start (server-side)

All commands assume `conda activate bgc` and you are in the repo root.

**Train mainline (seed 0):**

```bash
python -u src/train.py --config configs/mainline.yaml --seed 0 \
    --data-dir data/mibig_v2 --out-dir data/phase3/route_b/s0
```

**Run stride inference on a FASTA dir for the current Stage 3/3b candidates:**

```bash
python scripts/inference/stride.py \
    --fna-dir /data/syh/benchmark/nine_genome/fna \
    --ckpt /data/muskliu/npmaster/data/phase3/stage3_hardneg_v1/best/best_tk100-ms0-hnw4.pt \
    --window 8192 \
    --stride 2048 \
    --out-dir /tmp/npmaster_stage3_tk100_w8192_s2048
```

**Evaluate the promoted 9-genome operating point:**

```bash
python scripts/evaluate/nine_genome.py \
    --base /data/syh/benchmark/nine_genome \
    --results-dir /tmp/npmaster_stage3_tk100_w8192_s2048 \
    --operating-point defrag_v2_ms10_s099 \
    --out /tmp/defrag_v2_ms10_s099_all9.json
```

Use explicit flags with `--start-threshold 0.995` only for the precision-oriented alternate `defrag_v2_ms10_s0995`.

**Regenerate the full baseline comparison table:**

```bash
python scripts/evaluate/baselines_nine_genome.py \
    --antismash-dir    /data/syh/benchmark/nine_genome/results/antismash \
    --deepbgc-dir      /data/syh/benchmark/nine_genome/results/deepbgc \
    --bgc-prophet-json /data/syh/benchmark/nine_genome/results/bgc_prophet/bgc_prophet_regions.json \
    --npmaster-results-dir /tmp/npmaster_stride8192_s0 \
    --operating-points val_frozen_fused,tuned_v2 \
    --out     data/phase4/nine_genome_comparison/results.json \
    --out-md  data/phase4/nine_genome_comparison/results.md
```

**v4.1 multi-scale ensemble (primary mainline since 2026-04-22):**

```bash
python scripts/evaluate/v4_1_multiscale.py \
    --primary-preds experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
    --short-preds   experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
    --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
    --out-json experiments/evo2_first_principles/v4_1_final_readout.json
```

**v4.1-learned decoder on OER004256 (metagenomic):**

```bash
# (1) decode with learned decoder
python scripts/decode/v4_1_learned.py \
    --primary-preds experiments/OER004256/v41_primary_preds \
    --short-preds   experiments/OER004256/v41_short_preds \
    --decoder-ckpt  experiments/evo2_first_principles/decoder_v1/runs/c00_cnn3_h32_antismash_bce/decoder.pt \
    --out-dir       experiments/OER004256/v4_1_learned

# (2) add v4.1 + defrag_v2 per-type classification (chain)
python scripts/classify/v4_1_regions.py     ...
python scripts/classify/defrag_v2_regions.py ...

# (3) triage antiSMASH-missed candidates (HMM biosynth + MIBiG novelty)
python scripts/diagnose/oer_novel_candidates.py ...
python scripts/classify/hmm_biosynth_scan.py
python scripts/classify/mibig_novelty.py
```

See `docs/REGRESSION_LEDGER.md` for canonical numbers and reproduce commands. See `docs/learned-decoder.md` for learned-decoder motivation + training protocol, `docs/oer004256-novel-triage.md` for the three-layer in-silico filter that produced the 86-candidate wet-lab shortlist.

External baseline directories are read-only inputs; all parsing logic lives in `scripts/baselines/`.

## OER004256 metagenomic discovery (2026-04-23)

v4.1-learned on 1.48 Gb assembly (7 samples, contigs ≥ 10 kb):

| output | count | notes |
|---|---:|---|
| v4.1-learned regions | **1,230** | 531 primary + 699 short-rescue |
| antiSMASH regions (same data) | 1,976 | same order of magnitude |
| v4.1-learned ∩ antiSMASH | 801 | AS-recall 37% (hysteresis: 12%) |
| antiSMASH-missed "novel" candidates | **429** | of which 86 pass STRONG biosynth × (NOVEL or LIKELY_NOVEL) filter |
| v4.1 ∩ defrag_v2 type agreement | 80% | cross-validation signal |
| Original 15 novel candidates | 15 / 15 preserved | unchanged by decoder upgrade |

All 15 of the original novel-candidate cohort (from hysteresis) have STRONG or MODERATE biosynthetic HMM support and **0 are classified as KNOWN** (≥2 ORFs at ≥70% pident) against MIBiG 4.0. The 86-candidate expanded shortlist includes novel jamaicamide-, rifamycin-, and quinolidomicin-like scaffolds. Artifacts under `experiments/OER004256/v4_1_learned/`.

## Next Step

1. **Wet-lab** — send the 86 STRONG × (NOVEL/LIKELY_NOVEL) candidates to experimental screening. Covers 4+ chemistry families (PKS, NRPS, NRPS-PKS hybrids, Terpene-NRPS, AurF N-oxidation).
2. **Training-side improvements** — the learned decoder plateaus at ~42% AS-recall on OER; the remaining gap requires a retrained scorer (joint training + metagenomic augmentation). Deferred until wet-lab returns initial labels — enables active-learning retrain on real confirmations.
3. **Legacy items** — `GCA_000154945.1` <5kb recall and boundary overshoot are still open (see `docs/route-b-hardneg-v1.md`, `docs/next-phase.md`) but lower priority than wet-lab confirmation loop.
