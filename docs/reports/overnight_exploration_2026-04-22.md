# Overnight experimental campaign — 2026-04-22

Ran a broad architecture × data axes sweep under the v4 honest protocol to
answer: **is there anything that beats the frozen-Evo2 + LR mainline (dedup=0.487, strict=0.419)?**

Scoped experiments, all with first-principles OPs (no 9-genome tuning):

- **Exp A**: pooling — mean / max / std / concat (done earlier)
- **Exp B**: gene-level LR (done earlier)
- **Exp C**: gene-level arch sweep — LR / MLP / CNN / BiLSTM / Transformer
- **Exp D**: data variants × arch — proto_core labels, hardneg augment, deeper CNN, + window-LR with proto_core
- **Exp E**: window-level context retry — CNN / Transformer + calibration loss
- **Exp F**: ensemble — window-LR ∪/∩ gene-CNN
- **Exp G**: full-pool label variants — region / proto_core / soft / boundary-weighted

30+ model variants trained, all evaluated under identical 9-genome honest readout.

## Headline

**No configuration beats the w=8192 mean-LR baseline on dedup_F1**. Every
architectural complication, every feature variant, every label regime either
ties or loses. The simple linear readout on frozen Evo2 features is the ceiling
for this task paradigm.

**One modest win on strict_F1**: proto_core labels give **strict=0.386 at
w=16384** (baseline w=16384 region = 0.370) — +0.016 strict via tighter training
labels. Dedup drops by -0.034, so this is a precision/recall trade, not a
Pareto improvement.

## Full results table (best OP per variant)

| experiment | arch | label | geom | dedup | strict | bench | val_AUROC |
|---|---|---|---|---:|---:|---:|---:|
| **baseline (LR_A1)** | LR | region | w=8192 | **0.487** | **0.419** | **0.511** | 0.868 |
| Exp A.mean_only (60g) | LR | region | w=8192 | 0.441 | 0.381 | 0.464 | — |
| Exp G.region (full 1188) | LR | region | w=16384 | 0.474 | 0.370 | 0.476 | 0.868 |
| **Exp G.proto_core (full 1188)** | LR | proto_core | w=16384 | 0.440 | **0.386** | 0.446 | 0.868 |
| Exp G.region_soft (full) | LR | region_soft | w=16384 | 0.461 | 0.360 | 0.466 | — |
| Exp G.region_w_boundary (full) | LR | region+boundary-weight | w=16384 | 0.468 | 0.367 | 0.470 | — |
| Exp C.cnn_loose | CNN | region | gene | 0.420 | 0.343 | 0.431 | 0.868 |
| Exp C.bilstm_loose | BiLSTM | region | gene | 0.355 | 0.229 | 0.359 | 0.961 |
| Exp C.lr_loose | LR | region | gene | 0.353 | 0.305 | 0.370 | 0.674 |
| Exp C.mlp_loose | MLP | region | gene | 0.271 | 0.227 | 0.289 | 0.696 |
| Exp C.transformer_loose | TXfmr | region | gene | **0.040** | 0.006 | 0.040 | 0.676 |
| Exp D.cnn_deep | CNN(h=384) | region | gene | 0.434 | 0.333 | 0.454 | 0.865 |
| Exp D.cnn_hardneg | CNN | hardneg | gene | 0.412 | 0.320 | 0.434 | 0.864 |
| Exp D.cnn_proto_core | CNN | proto_core | gene | 0.241 | 0.214 | 0.243 | 0.978 |
| Exp D.bilstm_proto_core | BiLSTM | proto_core | gene | 0.272 | 0.258 | 0.275 | 0.983 |
| Exp D.window_lr_proto_core (60g) | LR | proto_core | w=8192 | 0.415 | 0.377 | 0.448 | — |
| Exp E.window_cnn (cal) | CNN | region | w=8192 | 0.251 | 0.120 | 0.251 | 0.956 |
| Exp E.window_transformer (cal) | TXfmr | region | w=8192 | 0.377 | 0.324 | 0.413 | 0.802 |
| Exp F.ensemble best (∩) | LR ∩ CNN | region | w=8192/gene | 0.481 | 0.418 | 0.499 | — |
| Exp G.proto_x_0.7 ensemble primary | 0.3·LR + 0.7·proto_LR | both | w=16384 | 0.443 | 0.384 | 0.450 | — |

Reference comparators (same honest GT):
- defrag_v2 Route B CNN (same backbone, honest OP): dedup=0.270, strict=0.195
- BGC-Prophet (external): dedup=0.391, strict=0.302

## Seven durable findings from the sweep

### 1. More pooling ≠ more signal
Adding max and/or std channels to mean-pool HURTS at primary OP (more windows
cross threshold, precision crashes) and converges with mean-pool at alt OP
(spike filter compensates). max-only is weaker than mean-only by 0.03–0.08.
Evo2 encodes BGCs as distributed statistics, not localized peaks.

### 2. Val AUROC does NOT predict region dedup
- BiLSTM: val AUROC 0.961 → 9g dedup 0.355 (worst-translation)
- CNN (window, cal loss): val AUROC 0.956 → 9g dedup 0.251
- LR (gene): val AUROC 0.674 → 9g dedup 0.353

Gap of up to 0.71 between val AUROC and 9g dedup. Suggests either (a) val is
too easy (only held-out *genomes* from same distribution), or (b) decoders
penalize overconfident scores (same lesson as Route B CNN's saturation failure).

### 3. Transformers are unreliable for this task
Gene-Transformer collapsed catastrophically on 9g (dedup 0.04 despite val
AUROC 0.68). Window-Transformer at 0.38 beats Route B CNN but loses to LR.
Attention picks up spurious correlations that don't transfer — possibly
distribution shift (train uses antiSMASH CDS, 9g uses pyrodigal).

### 4. Gene-level underperforms window-level
Best gene model (CNN deep) hits 0.434; best window LR hits 0.487.
The "BGCs are gene clusters, so gene-level should be better" intuition
FAILS with frozen mean-pool Evo2 features. Per-gene mean-pool of a 1kb
gene is too coarse; the discriminative signal lives in longer stretches.

### 5. Proto_core labels trade dedup for strict (+0.016 strict, -0.034 dedup)
Training on tighter BGC cores produces tighter predictions — higher strict
F1 but lower recall. Not a Pareto win at any one OP, but opens the door to
an ensemble that combines region-trained recall with proto_core-trained precision.

### 6. Region-level ensembles don't beat best-single; score-level marginally might
- **Region-level ensemble** (each model decodes, then union/intersect regions):
  - window-LR ∩ gene-CNN: 0.481 dedup (just below window alone, 0.487)
  - region + proto_core avg/product/max: 0.433–0.474 dedup (below region alone)
- **Score-level ensemble** (combine per-window probs, single decoder):
  - `geomean_max 0.5/0.5` at alt OP: dedup=0.495 (+0.008 vs baseline)
  - `avg_mean 0.5/0.5` at alt OP: dedup=0.492 (+0.005)
  - `avg_max 0.5/0.5` at alt OP: dedup=0.485 (-0.002)
- **Post-hoc tuned** `w=0.6·window + 0.4·gene_max` at alt OP: **dedup=0.501** (+0.014)
  — but weight tuned on 9g so not honestly claimable
- Under strictly-honest protocol (pre-declared 0.5/0.5): **ensemble ≈ baseline within noise**

### 6b. Per-bucket recall shows genuine complementarity on short BGCs
| bucket | n_gt | window-LR R | gene-CNN R | union R | gap |
|---|---:|---:|---:|---:|---:|
| <5kb | 48 | 0.146 | **0.229** | 0.229 | +0.083 |
| 5-10kb | 89 | 0.360 | **0.461** | 0.506 | +0.045 |
| 10-20kb | 85 | **0.447** | 0.412 | 0.494 | +0.047 |
| 20-50kb | 61 | 0.705 | 0.754 | 0.770 | +0.016 |
| ≥50kb | 22 | 0.864 | 0.909 | 0.955 | +0.045 |

Gene-CNN **wins short-BGC recall** (<5kb: 0.23 vs 0.15, 5-10kb: 0.46 vs 0.36).
Union recall has +0.045 headroom in 3 buckets. **This is the only genuine
complementarity we found**. But region- and score-level ensembles can't
unlock it cleanly under honest protocol without a properly-learned fusion
weight from a held-out meta-training set (we didn't compute gene-CNN scores
on test_ood genomes, so this remains untested).

### 7. Hard-negative augmentation hurts
Adding adjacent-to-BGC genes as negatives REDUCED gene-CNN dedup by 0.01–0.03.
The boundary is noisy; telling the model "this gene is NOT BGC" when it's
literally flanking one confuses rather than clarifies.

## Why nothing beats LR: a hypothesis

The frozen Evo2 mean-pool embedding `blocks.20.mlp.l3` is a ~4k-d
**compression** of a 8-16kb DNA window. The BGC vs non-BGC discriminative
information lives in a near-linear subspace of this embedding — a single
4096-d hyperplane captures nearly everything. Additional model capacity
either:

- **Overfits to training-distribution artifacts** (val AUROC up, 9g down)
- **Saturates the output distribution** (decoder can't rank)
- **Specializes on the wrong granularity** (gene-level loses long-context)

The only genuinely complementary signal we found is **proto_core labels**,
which boost strict via tighter supervision. This suggests future gains are
more likely to come from:

1. **Better/different features**: different Evo2 layers, per-token heads,
   LoRA fine-tune
2. **Boundary refinement post-processing**: use existing LR regions, snap
   boundaries to gene edges or proto_core trained-model outputs
3. **Short-BGC dedicated head** (still not tested) — only known structural
   gap (<5kb recall 0.15)

## Artifacts

- Code (after cleanup): `src/evo2_lr/` package + 7 scripts in `scripts/`
  (failed-experiment scripts deleted; results preserved in
  `experiments/evo2_first_principles/stage0/exp_*.json`)
- Canonical ensemble: `scripts/evaluate/v4_1_multiscale.py`
- Frozen config: `configs/experiments/evo2_first_principles_v4_1_frozen.yaml`

## Daytime follow-ups (ran after this log was first drafted)

The complementarity finding in §6b suggested three possible directions.
Two were pursued the same day; both failed under honest protocol; and a
third direction (short-BGC multi-scale) unexpectedly produced the one
positive result of this campaign.

### Path A — properly-learned score fusion — **REJECTED**

Extracted gene-CNN scores on 40 held-out test_id genomes (disjoint from
LR_A1 training), computed per-window (LR_prob, gene_max, gene_mean) pairs,
trained 2-parameter logistic regression on test_id windows, applied to
9-genome.

- Meta-training AUROC: 0.96 (the fusion separates classes well on held-out)
- 9-genome dedup_F1 at pre-declared OPs: **0.31** (vs baseline 0.487)
- Cause of failure: **gene-CNN looks near-perfect on test_id** (mean_pos=0.89,
  mean_neg=0.17 — it was trained on similar Actinomycetes); the fusion
  learned to overweight gene-CNN (weight 5.9 vs LR 2.75 in prob space);
  that weight doesn't transfer because gene-CNN has much lower precision on
  the 9-genome BGC distribution.

Learned ensemble weights that look great on meta-training fail at inference
when target distribution differs. Script: `(archived in _archive)`
(since removed — results preserved in
`experiments/evo2_first_principles/stage0/...`).

### Short-BGC multi-scale — **ACCEPTED (v4.1)**

Direct attack on the <5 kb and 5–10 kb gap: extract Evo2 at w=4096/s=1024
on 9-genome only, apply LR_A1 at pre-declared OPs, union with the v4
baseline. The recipe that survives bootstrap and makes biological sense:

- Primary arm: w=8192 alt OP (v4 baseline)
- Short rescue arm: w=4096, s=0.95/e=0.80/ms=5 (min region 8 kb), length cap 10 kb
- Union-dedup, IoU 0.3

Result: +3 hits in 5–10 kb bucket (32/89 → 35/89); strict_F1 +0.011,
bench_F1 +0.008, dedup_F1 +0.001. Pareto improvement on strict and bench;
dedup gain within CI but never negative. Tri-scale extension with w=2048
was tried and rejected (recovers <5 kb recall 7/48 → 17/48 but crashes
dedup to 0.26 — LR_A1 isn't calibrated at that geometry).

Frozen as v4.1 in
`configs/experiments/evo2_first_principles_v4_1_frozen.yaml`. See v4 plan §13.

### LoRA fine-tune — still unexplored

The remaining untouched lever. ~1–3 days of training; feasibility test
(LoRA on 10 training genomes) would take a few hours. Not pursued this
round; open question for future work.

## What's left to try (if you choose to keep pushing)

1. **Native w=2048 LR training** on the full 1188-genome pool (~20 A800-h
   extraction). Would properly calibrate LR at the geometry needed to fix
   the <5 kb bucket — currently the only known structural gap.
2. **LoRA fine-tune of Evo2 7B**. Highest ceiling, unexplored.
3. **Paper / thesis writeup** as-is. The null-result set is substantial and
   the v4.1 positive is defensibly honest.
