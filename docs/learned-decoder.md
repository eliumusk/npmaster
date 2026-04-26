# Learned decoder — motivation, training, operating point

## Problem

The v4.1 multi-scale pipeline has three layers:

1. Frozen Evo2 7B embedding (representation)
2. Logistic regression `LR_A1` on mean-pooled embeddings (scorer, learned)
3. **Hysteresis decoder** with `start=0.95, extend=0.80, min_support_windows=3, max_gap=0` (**hand-tuned**)

The hand-tuned OP was picked on 9-genome isolate contigs (median contig length ≫ Mb). On metagenomic short contigs (median ~15 kb at OER004256, ≈3–5 windows at stride 2048), the `min_support_windows=3` rule **structurally rejects** BGCs the underlying scorer is actually detecting:

- **OER v4.1 hysteresis output: 254 regions**
- **antiSMASH on same data: 1,976 regions** (7.8× gap)
- Per-window v4.1 score on the 1,739 "antiSMASH-missed" regions: **p50 = 0.85, p90 = 0.98** — the model sees them, the decoder kills them.
- **0 AS_ONLY regions had zero v4.1 coverage** — this is not a scorer gap, it is a post-processing gap.

Same symptom on 9g (F1=0.54 decoder, R=0.47) scales linearly with 1.2× antiSMASH:v4.1, but on OER the symptom blows up to 7.8×. Classic **post-processing distribution shift** — representation layer generalizes, hand-tuned rule does not.

## Solution: learn the decoder

Replace the hysteresis rule with a **small 1D CNN** that takes the per-window LR score sequence along a contig as input and predicts per-window P(inside a real BGC region). Decoded region = maximal run of windows with P ≥ threshold. No hard `min_support_windows`.

Module: [`src/decoding/learned.py`](../src/decoding/learned.py) — `Decoder1DCNN(depth=3, hidden=32)`, ~50K parameters.

```python
# Architecture
input:  [B, T] per-window LR score ∈ [0, 1]
  ↓ Conv1d(1→32, k=3)         (local context)
  ↓ GELU
  ↓ Conv1d(32→32, k=3, d=1)
  ↓ GELU
  ↓ Conv1d(32→32, k=3, d=2)   (dilated — 3 windows radius)
  ↓ GELU
  ↓ Conv1d(32→32, k=3, d=4)   (broader context)
  ↓ GELU
  ↓ Conv1d(32→1, k=1)          (per-window logit)
output: [B, T] logit  →  sigmoid → threshold → decode
```

## Training data

Source: training-pool LR_A1 scores + antiSMASH-matched window labels.

1. Apply LR_A1 to `embeddings_merged.npy` (1.064M training windows) → per-window scores (`scripts/train/decoder_data.py`).
2. Group by `(genome_id, contig_id)`, keep contigs with **≥ 10 windows** (18,280 contigs, 877K windows), sort by `window_start`.
3. Split exactly as LR training (train / val / test_id / test_ood).
4. Per-window label: `1` if the window is marked `positive` in windows_metadata (antiSMASH overlap ≥ threshold), `0` if `negative`, `-1` if `ambiguous` (masked out of loss).

Script: `scripts/train/decoder_data.py` → `decoder_data.pkl`.

## Training protocol

16-config grid, 1 seed each, parallelized across node7 + node8 (16 A800 GPUs). Each run ≈ 1–2 min.

| axis | values |
|---|---|
| depth | 3 (CNN-3L), 5 (CNN-5L) |
| hidden | 32, 64 |
| label_src | `antismash` (use metadata `label` column), `tight` (overlap_ratio ≥ 0.7) |
| loss | BCE (pos_weight=3), Focal (γ=2, pos_weight=3) |

- Optimizer: Adam, lr=1e-3
- Batch: 32 contigs with random crops to max 512 windows
- Early stop: patience=5 on val AP
- Loss mask: ambiguous windows (`label=-1`) excluded

Script: `scripts/train/decoder.py --config-id K --gpu G --out-root <dir>`.

Launch both nodes:
```bash
# node7 (local)
for i in 0..7; do nohup python scripts/train/decoder.py --config-id $i --gpu $i --out-root $OUT &; done
# node8 (over SSH)
for i in 8..15; do ssh node8 "nohup python ... --config-id $i --gpu $((i-8)) ... &"; done
```

## Evaluation

Three sequential evals (`scripts/evaluate/learned_decoder_{sweep,threshold_sweep,multiscale}.py`):

1. **sweep** — 16 configs at fixed threshold=0.5 on 9g GT + OER regions. Picks best config family.
2. **threshold_sweep** — for top-4 configs, sweep threshold ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98}. Picks best operating point on Pareto frontier.
3. **multiscale** — for chosen config, grid over primary × short thresholds. Selects final frozen OP.

## Results

### 16-config sweep at threshold=0.5 (primary arm only)

All 16 configs converged to val AP 0.79–0.82 in <2 min. `c00` (cnn3, hidden=32, antismash label, BCE loss) selected for novel preservation robustness — it holds all 15 original novel candidates across thresholds up to 0.98; tight-label variants start dropping novel at thr ≥ 0.8.

### Threshold sweep, c00, primary arm only

| threshold | 9g F1 | 9g P | 9g R | OER regions | OER AS-recall | novel15 |
|---:|---:|---:|---:|---:|---:|---:|
| baseline (hysteresis) | **0.539** | 0.640 | 0.466 | 254 | 12% | — |
| 0.98 | **0.538** ≈ baseline | 0.656 | 0.456 | 406 | 17% | 15/15 |
| 0.95 | 0.512 | 0.553 | 0.475 | 745 | 23% | 15/15 |
| 0.90 | 0.493 | 0.495 | 0.492 | 1,316 | 31% | 15/15 |
| 0.80 | 0.456 | 0.405 | 0.521 | 2,307 | 38% | 15/15 |

### Multi-scale grid (c00, primary × short)

**Frozen operating point: primary=0.95, short=0.98.**

| primary | short | 9g F1 | 9g P | OER regions | OER AS-recall | novel15 |
|---:|---:|---:|---:|---:|---:|---:|
| baseline (multi-scale hysteresis) | — | **0.541** | 0.606 | 254 | 12% | — |
| 0.95 | **0.98** | **0.461** | 0.424 | **1,230** | 37% | 15/15 |
| 0.95 | 0.95 | 0.428 | 0.368 | 1,773 | 42% | 15/15 |
| 0.95 | 0.90 | 0.394 | 0.312 | 2,857 | 45% | 15/15 |

At `(0.95, 0.98)`: 9g F1 drops 0.08 from baseline (0.541 → 0.461) in exchange for **4.8× more OER regions** (254 → 1,230), **3× AS-recall** (12% → 37%), 0 loss of the 15 original novel candidates.

> **⚠ Metric caveat added 2026-04-24**: the 0.541/0.461 numbers above come from `full_pipeline_eval.py`'s custom greedy-pairing `dedup_metrics`. Under the **canonical** `benchmark/nine_genome.eval_regions` metric (same one used for published v4.1-hysteresis=0.488), v4.1-learned scores **0.332** on 9g — a **0.156** drop from hysteresis, not 0.08. Both numbers are "correct" within their own metric; the mistake was mixing them across tables. For honest cross-comparison with v4.1-hysteresis (0.488), use 0.332. See `memory/project_v41_learned_9g_metric_confusion.md` and `experiments/eval_harness/baseline_table.md`.

## Production entrypoint

```bash
python scripts/decode/v4_1_learned.py \
    --primary-preds <dir w=8192/s=2048> \
    --short-preds   <dir w=4096/s=1024> \
    --decoder-ckpt  <path to c00 decoder.pt> \
    --out-dir       <output dir>
```

Add per-type classification with the existing `scripts/classify/v4_1_regions.py` (v4.1 Stage-2 LR) and optionally `scripts/classify/defrag_v2_regions.py` (defrag_v2 CNN, stronger classifier at 82.7% top-1 on 9g).

## Honest caveats

- **9g F1 drops 0.08 vs hysteresis** (under `full_pipeline_eval.py` metric). Under canonical bench-mode metric (`benchmark/nine_genome.eval_regions`), the drop is actually **0.156** (hysteresis 0.488 → learned 0.332, bench_P collapses from 0.60 to 0.26 — heavy over-prediction on 9g's long isolate contigs). Acceptable *only* if metagenomic recall is worth more than ~3× on the precision-honest 9g scoreboard.
- **Short arm is the dominant contributor** (699 of 1,230 OER regions on short arm, 531 on primary). Short arm was trained on primary-arm scores — OOD transfer. Works but is the component most likely to surface false positives.
- **AS-recall plateaus at ~42%** even at aggressive thresholds. The remaining ~1,100 antiSMASH regions the learned decoder cannot recover are where LR_A1 does not fire at all — a scorer-level limit, not a decoder-level one. Closing that gap needs a retrained scorer or joint training.
- The 15 original novel candidates are preserved at every operating point tested. Downstream HMM + MIBiG filtering of the expanded 429-novel cohort produces **86 "STRONG biosynth × novel"** candidates (`experiments/OER004256/v4_1_learned/novel_triage/novel_triage_full.csv`); these include previously undiscovered jamaicamide-like, rifamycin-like, and quinolidomicin-like scaffolds.
