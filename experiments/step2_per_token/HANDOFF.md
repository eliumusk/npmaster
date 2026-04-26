# Per-Token BGC Detection — Handoff for External Review

**Date**: 2026-04-24 (overnight experiment)
**Author**: Claude (working with a CS student on BGC detection thesis)
**Purpose**: get a second opinion on (a) whether our diagnosis of why per-token v1 failed is correct, and (b) which of three proposed next steps is right.

---

## 0. Current update — OER U-Net readout (2026-04-26)

The OER-first branch of the plan has now completed. Full OER004256 per-token features were extracted for all 16 shards (`755,512` windows, `w=8192/s=2048`), and two Round-2 U-Net checkpoints were evaluated:

| method | threshold | n_pred | AS TP / 1,976 | AS-recall | novel | median_len_bp |
|---|---:|---:|---:|---:|---:|---:|
| E10 U-Net | 0.3 | 1,377 | 616 | **0.3117** | 470 | 9,828 |
| E10 U-Net | 0.4 | 1,271 | 531 | 0.2687 | 388 | 9,384 |
| E10 U-Net | 0.5 | 1,194 | 491 | 0.2485 | 325 | 8,192 |
| E11 U-Net | 0.3 | 1,154 | 541 | 0.2738 | 382 | 10,240 |
| E11 U-Net | 0.4 | 1,083 | 472 | 0.2389 | 321 | 10,033 |
| E11 U-Net | 0.5 | 1,007 | 334 | 0.1690 | 265 | 6,182 |
| v4.1-learned | production OP | 1,230 | 408 | 0.2065 | 429 | 5,973 |
| v4.1-hysteresis | frozen OP | 254 | 187 | 0.0946 | 15 | 12,288 |

**Interpretation**: E10 is the only promising OER checkpoint. `E10 thr=0.3` is the recall-first point (+10.5 AS-recall points over v4.1-learned), while `E10 thr=0.4` is the cleaner candidate-count tradeoff. E11 is a useful comparison but should not be promoted.

**Important caveat**: this is an antiSMASH-overlap readout, not true OER F1/AUC. "Novel" means antiSMASH-missed, not necessarily false positive. The candidate has 9-genome readout now, but still needs a pre-registered checkpoint+threshold selection rule before it can be called a formal general detector.

Full record: `experiments/step2_per_token/OER_UNET_READOUT.md`.

**9-genome update**: E10/E11 have now been read out on 9-genome with canonical `benchmark.nine_genome.eval_regions`. E10 is close to v4.1 without selecting thresholds from 9g: `thr=0.3` (OER recall-first) gives dedup_F1 0.4561, `thr=0.4` (OER conservative) gives 0.4664, and default `thr=0.5` gives 0.4751. v4.1-hysteresis remains 0.4878. These are threshold-clean diagnostics; a fully formal claim still needs pre-registering the checkpoint/config selection rule before looking at 9g.

---

## 1. Problem context (1 paragraph)

We detect bacterial Biosynthetic Gene Clusters (BGCs) on genomes and metagenomes.
Current mainline **v4.1** = frozen Evo2 7B embeddings → **mean-pool over 8192 bp windows** → logistic regression → hysteresis decoder → regions. It scores `dedup_F1 = 0.488` on the 9-genome held-out benchmark (v4.1-hysteresis), which is SOTA among honest/no-9g-tuning methods. But on metagenomes (OER004256, median contig ~15 kb) it only recovers ~12% of antiSMASH regions because the `min_support_windows=3` rule structurally kills short-contig BGCs. A learned CNN decoder (v4.1-learned) lifts OER AS-recall to 37% but drops 9g `dedup_F1` to 0.332 (-0.156 — canonical bench metric, NOT the 0.461 misreported earlier under a different metric).

**Hypothesis driving per-token**: mean-pool dilutes the localized BGC signature (e.g. a PKS KS domain is ~1.5 kb inside an 8 kb window; mean-pool dilutes that 5×). Per-token prediction should both (a) preserve short-BGC signal and (b) collapse the scorer+decoder into one shallow Conv1D over per-nt probabilities, removing most v4.1 "patches" (multi-scale, short-arm, union-dedup, hysteresis).

## 2. What we built tonight

**Extraction**: Evo2 `blocks.20.mlp.l3` per-token hidden states, projected 4096 → 128 via fixed Gaussian random matrix (JL-scaled, seed=0xE2E2), then mean-pooled over every 8 tokens (K=8), stored fp16. Validated byte-exact (`max_abs=0.0`, 10/10 windows) against a pre-existing path that sliced from genome FASTA with N-pad.

**Head**: `PerTokenHead` — 3× dilated Conv1D (d=1, 2, 4), hidden=64, ~49K params. Intentionally tiny — prior overnight campaign showed complex heads on frozen Evo2 features overfit (val AUROC 0.96 → 9g dedup_F1 0.25 pattern).

**Training**:
- Data: 355K train + 48K val windows (33% of full pool — extraction still running at time of training).
- Labels: per-nt label = 1 if inside any antiSMASH region on same contig, 0 otherwise, -1 if upstream "ambiguous" window; downsampled by K=8 (rule: 1 if any subunit is 1, -1 if any is -1, else 0).
- BCE with `pos_weight=3.0` (positive rate is ~6%).
- 10 epochs, batch 512, Adam 1e-3, AdamW no schedule.

**Inference on 9g** (same head, length-agnostic Conv1D handles 1024-token input from w=8192):
- For each of 4 thresholds {0.3, 0.4, 0.5, 0.7}: per-token probs → per-nt track → runs of `prob ≥ threshold` → filter `length ≥ 2 kb` → fill gaps `< 500 bp`.

## 3. Results

### Training

| epoch | val loss | val token P | val token R | val token F1 |
|---|---:|---:|---:|---:|
| 0 | 0.463 | 0.78 | 0.13 | 0.22 |
| 4 | 0.445 | 0.89 | 0.16 | 0.27 |
| 9 (best) | **0.436** | 0.84 | 0.18 | **0.30** |

Loss monotonically decreasing, val loss still falling at epoch 9.

### 9-genome region-level (canonical `bench` metric, same as v4.1-hysteresis published)

| method | n_pred | dedup_F1 (CI95) | bench_P | bench_R | strict_F1 | `<5kb` bucket R |
|---|---:|---:|---:|---:|---:|---:|
| v4.1-hysteresis | 246 | **0.488** (0.345–0.609) | 0.598 | 0.459 | 0.430 | 0.146 |
| v4.1-learned | 363 | 0.332 (0.224–0.437) | 0.262 | 0.528 | 0.246 | 0.167 |
| per-token thr=0.3 | 1421 | 0.166 | 0.283 | 0.236 | 0.082 | **0.229** |
| per-token thr=0.4 | 462 | **0.194** (0.115–0.262) | 0.641 | 0.144 | 0.115 | 0.083 |
| per-token thr=0.5 | 346 | 0.175 | 0.783 | 0.115 | 0.108 | 0.083 |
| per-token thr=0.7 | 282 | 0.156 | 0.812 | 0.098 | 0.102 | 0.021 |

**Pass/fail gate**: `dedup_F1 ≥ 0.458`. Per-token **FAILS at every threshold**. Upper CI95 bound at best threshold is 0.262, nowhere near the gate.

## 4. Diagnosis

### What went wrong

1. **Token-level recall ceiling 0.18** — trained model never learned to fire on most BGC nucleotides. pos_weight=3 at ~6% positive rate appears too low (Youden-like calculation says pos_weight ≈ 15 would balance). Val token precision climbs to 0.84 but recall stalls at 0.18.
2. **Fragmentation**: `max_preds_per_GT = 13–17` at thr≥0.4. Each true BGC is split into many short predicted regions. `dedup_P / bench_P ≈ 0.46` (fails the 0.75 short-core-hack gate). Root cause: token-level probs are "bursty" — spike above threshold, drop below, spike again — so a single BGC makes 15 disjoint runs.
3. **Partial training data**: 33% of pool used (extraction was still running). Not clear how much this matters vs. the loss/threshold issues.

### What went right

- **Representation is not the problem**. Even at threshold 0.4, bench_P=0.64 exceeds v4.1-hysteresis (0.60). At thr=0.7, bench_P=0.81. When the model DOES predict a region, it's accurate. This rules out "per-token features don't encode BGC" — the Johnson-Lindenstrauss projection + mean-pool-over-K=8 preserves the BGC signature.
- **One genuine architectural win**: at thr=0.3, per-token recovers <5kb bucket recall of **0.229 vs 0.146 (hysteresis) and 0.167 (learned)** — the known blind spot that motivated this whole direction. The per-nt resolution removes the "mean-pool dilution" failure mode on short BGCs. But every other length bucket crashes under this threshold, so the total dedup_F1 is dragged down.

## 5. Three proposed next steps (want review on which is right)

### A. Recall-first retune (1 hour, today)

- Keep architecture, keep 33% training data.
- Change: `pos_weight = 3 → 10`. Optionally Focal loss γ=2.
- Re-decode at thr={0.3, 0.4, 0.5} + add aggressive gap-fill (`max_gap_bp=2000` instead of 500) to glue fragments within same BGC.
- **Hypothesis**: move the recall/precision operating point without architectural change.
- **Risk**: may just trade one failure mode for another; fragmentation persists if burstiness is inherent to the scorer.

### B. Full-data retrain (6 hour wait + 1 hour train)

- Wait for extraction to finish (~4 more hours from now).
- Train on full 1.064M windows with default pos_weight=3, 10 epochs.
- Compare to v1's 33% run to determine if data volume is the blocker vs. loss formulation.
- **Value**: isolates variable. If val recall doesn't budge from 0.18 even at full data, we know it's loss/model, not data.

### C. OER-first (parallel, 4 hour extraction + 1 hour infer)

- Skip further 9g optimization. Extract OER per-token features, run current ckpt, score on OER AS-overlap.
- **Rationale**: per-token's ONE win was `<5kb` bucket — OER median contig is ~15 kb with many `<10 kb` BGCs. Per-token may be genuinely better on metagenomes even while failing 9g.
- **Risk**: if per-token fails on OER too, we've burned 5 hours to confirm a dead end.

### My recommendation (want critique)

Run **A and C in parallel tonight**:
- A takes 1h on 1 GPU (GPU 1 on node8, already freed). Costs almost nothing.
- C starts OER extraction on GPUs that finish their training shards (watcher daemon handles this automatically as they free up).
- Defer B until we see whether A alone lifts recall materially.

If A succeeds (dedup_F1 > 0.30, close to v4.1-learned territory), per-token is a live mainline candidate. If A fails and C succeeds, we have an OER-only story ("per-token better on metagenomes, v4.1-hysteresis better on isolates — use both"). If both fail, we've learned per-token needs LoRA / fine-tuning Evo2 itself — i.e. much bigger investment.

## 6. Open questions for reviewer

1. **Is the "recall from pos_weight" diagnosis right?** Or could it be something else (label quality, per-nt vs per-segment loss, BCE vs Dice for sparse positives)?
2. **Is the "fragmentation = bursty head output" intuition right?** Alternative: decoder gap-fill is too conservative. What's the fastest discriminator between these two?
3. **Is pure per-token over-simplified?** The user's direction was "大道至简, 从 Evo2 入手". But maybe we need a tiny BiLSTM or a segment-level CRF to get non-bursty outputs. Where's the right simplicity tradeoff?
4. **Is the 9g metric gate too strict?** v4.1-learned already fails it (0.332). Reasonable to loosen for metagenomic-facing candidates, or is OER-first a cop-out?
5. **Is the `<5kb` bucket win (0.229 vs 0.146/0.167) meaningful?** Small sample (48 GTs), bootstrap CI wasn't computed per-bucket. Could be noise.

## 7. Full artifact pointers

Server (`node7 = Voc_gpu_node7_a800`):
- `/data/muskliu/npmaster/experiments/step2_per_token/head_v1/runs/ckpn3_h64_seed0/best.pt` — trained ckpt
- `.../train_log.jsonl` — epoch-level metrics
- `.../9g_regions_t{0.3,0.5,0.7}.csv` — decoded 9g regions per threshold
- `.../RESULTS.md` — human-readable summary

Local (`/Users/musk/Documents/code/npmaster/`):
- `src/per_token/{labels,model,decode}.py` — library code
- `scripts/train/per_token/train_shards.py` — training entry
- `scripts/inference/per_token/infer_shards.py` — 9g inference
- `scripts/decode/per_token_v1.py` — decode driver
- `scripts/evaluate/per_token_readout.py` — canonical bench eval
- `experiments/eval_harness/baseline_table.md` — unified comparison
- `experiments/eval_harness/NIGHT_LOG.md` — full running log of tonight

Extraction still running at ~60% at time of writing; full-pool training data available in ~4 hours. OER extraction not yet started.

---

## Metric-confusion note for the reviewer

The published "v4.1-learned 9g dedup_F1 = 0.461" (`docs/learned-decoder.md`, REGRESSION_LEDGER #16) uses a DIFFERENT metric definition (`full_pipeline_eval.py`'s custom greedy pairing) than the canonical `benchmark/nine_genome.eval_regions` used for v4.1-hysteresis's 0.488. Same predictions, different F1 formula. Under the canonical metric, v4.1-learned = 0.332. All numbers in this handoff use the canonical metric. See `memory/project_v41_learned_9g_metric_confusion.md`.
