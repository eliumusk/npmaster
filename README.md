# NP-Master

Biosynthetic Gene Cluster (BGC) detection and discovery on bacterial genomes. Frozen Evo2 7B embeddings + a light context model (dilated CNN + gated residual). Trained against antiSMASH annotations, formally evaluated on the 9-genome external benchmark.

## Current Status (9-genome, bench mode)

Current best candidates are from the Stage 3 hard-negative checkpoint trained on antiSMASH-derived labels. Decoder parameters were selected on the 5-genome dev split, then frozen and evaluated on the 4-genome lockbox. No lockbox retuning was done.

Full 9-genome head-to-head:

| Tool / candidate | bench R | bench P | **bench F1** | dedup P | dedup F1 | n_pred | median pred len |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 | 0.4471 | 0.3911 | 170 | 22.7 kb |
| **NP-Master `defrag_v2_ms10_s099`** | **0.4098** | 0.4714 | **0.4385** | 0.4714 | **0.4385** | 140 | 38.9 kb |
| NP-Master `defrag_v2_ms10_s0995` | 0.3934 | **0.4851** | 0.4345 | **0.4851** | 0.4345 | 134 | 38.9 kb |

Lockbox-only check:

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

- `defrag_v2_ms10_s099` is now promoted to `configs/operating_points.yaml` because the frozen lockbox check held. `defrag_v2_ms10_s0995` remains a precision-oriented alternate, not the default, because it gives slightly lower recall/F1.
- Boundary tightness is still the main weakness: although all-9 strict/IoU F1 is now slightly above BGC-Prophet, NP-Master regions remain wider (median ≈ 38.9 kb vs BGC-Prophet ≈ 22.7 kb) and dev strict/IoU was still slightly behind. Treat boundary refinement as the next quality target.
- NP-Master misses `GCA_000154945.1` almost entirely (R ≈ 0.03) under every decoder setting, and BGC-Prophet misses it completely (R ≈ 0.03, F1 = 0). This genome is a shared model-domain blind spot, not a NP-Master-specific regression.

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

See `docs/REGRESSION_LEDGER.md` for all canonical reproduce commands.

External baseline directories are read-only inputs; all parsing logic lives in `scripts/baselines/`.

## Next Step

Next work should move back to training/data: fix the `GCA_000154945.1` domain miss and reduce boundary overshoot under strict/IoU. See `docs/route-b-hardneg-v1.md` and `docs/next-phase.md`.
