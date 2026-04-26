# Multi-layer Evo2 Probe — Verdict (2026-04-25)

**Question**: Does concat(L13, L20, L27) Evo2 mean-pool features beat L20-only on 9-genome dedup_F1, in the same conditions?

**Answer**: No. Layers are redundant; concat regresses. **Commit to LoRA path.**

## Setup

- 200K stratified training subset (train.pos=12533 / train.neg=161146 / val.pos=2143 / val.neg=24178), `windows_metadata_hardneg.csv`, `split ∈ {train, val}`, `label ∈ {positive, negative}`.
- Per-layer standardization on train, applied at infer.
- LR: sklearn LBFGS, C=1, class_weight="balanced", max_iter=500.
- 9g eval: canonical `benchmark.nine_genome.eval_regions`, v4.1 published OP `start=0.95 / extend=0.80 / ms=3 / max_gap=0`, 1000-sample bootstrap CI.
- All four LRs trained + scored under identical conditions — internal comparison is fair.
- L20 9g features bit-identical to v4.1 reference (`mean_abs_diff=0` at fp16; 9 / 9 genomes).

## Results (v4.1 OP, alt = `s=0.95/e=0.80/ms=3`)

| combo | dedup_F1 | CI95 | bench_P | bench_R | n_pred | val AP |
|---|---:|---|---:|---:|---:|---:|
| L13 | 0.3888 | 0.274–0.499 | 0.329 | 0.574 | 398 | 0.539 |
| **L20** | **0.3892** | 0.275–0.489 | 0.342 | 0.567 | 395 | 0.555 |
| L27 | 0.3840 | 0.284–0.477 | 0.325 | 0.521 | 375 | 0.594 |
| concat | 0.2337 | 0.171–0.309 | 0.172 | 0.600 | 758 | 0.489 |

Reference: **v4.1-hysteresis (full 1.064M training, layer 20 only) = 0.488 (CI95 0.345–0.609)**.

## Three findings

1. **Layers are redundant.** L13 / L20 / L27 cluster at 0.384–0.389. CI95 intervals fully overlap across all three. Even val window-level AUROC differs by < 0.02 (0.812–0.827). **No single Evo2 layer carries unique BGC signal that the others lack at the same window granularity.**

2. **Concat regresses, doesn't help.** At 200K samples, the 12288-d concat LR scores 0.234 — **0.155 below L20-only**. Likely cause: triple feature dim with limited data → LR overfits despite L2; concat over-predicts (n_pred=758 vs ~395), bench_P collapses to 0.17. Could potentially recover at 1.064M, but the *direction* is opposite of what the layer-fusion hypothesis predicts.

3. **Subset penalty: 0.10 F1.** L20-only at 200K → 0.389, vs published L20-only at 1.064M → 0.488. So the absolute number gap is the data-scale effect, NOT a methodology bug. Bootstrap CIs on 200K (~0.27–0.49) overlap the published 0.488 — within noise of the 1.064M baseline.

## Verdict

> **Frozen Evo2 layers saturate. Multi-layer fusion is a dead end.**

The original hypothesis ("AS-recall plateaus at 42% because layer 20 alone is too narrow → adding L13/L27 should lift it") is **falsified at 200K**. L13 and L27 do not encode complementary BGC signal at this window granularity; the linear probe on any single layer reaches the same ceiling.

## Next step — LoRA fine-tune Evo2

Don't spend 22h on full-1.064M concat extraction. The concat trajectory (0.234 < 0.389 single-layer) is so far below the threshold that even +0.10 from data scaling wouldn't put it above L20-only (still ~0.34). Diminishing returns are not the issue; **layers are the same**.

Commit the next budget to:

- **LoRA on Evo2** (window-level antiSMASH binary classification, target rank 16–32 on attn + MLP modules). Single A800 windowed forward + LoRA backward fits in 80GB at w=8192. ~20–50 A800-h depending on epochs.
- Re-evaluate `dedup_F1` and OER AS-recall after fine-tuning; gate against `≥ 0.488` (9g) and `≥ 0.42` (OER, beats v4.1-learned).

If LoRA also can't move the needle, the bottleneck is task definition (window-level binary BGC label is too noisy / too coarse) rather than representation, and the next axis is **ORF-level pooling** with antiSMASH gene-cluster as the natural target.

## Artifacts

- Server: `/data/muskliu/npmaster/experiments/multilayer_probe/`
  - `train_subset_200k/` — 16 shards, 200K windows × 3 layers × fp16 × 4096 = 4.9 GB
  - `9g/` — 9 genome × 3 layers × fp16
  - `readout/lrs/lr_{L13,L20,L27,concat}.npz` — trained probes + scalers
  - `readout/9g_preds_{L13,L20,L27,concat}/` — per-genome predictions.csv
  - `readout/multilayer_probe_readout.{json,md}` — full bootstrap CI tables (rsynced to local `experiments/multilayer_probe/readout/`)
- Local (this file): `experiments/multilayer_probe_verdict.md`

## Reproduce

```bash
ssh Voc_gpu_node7_a800
cd /data/muskliu/npmaster
PATH=/root/miniconda3/envs/bgc/bin:$PATH bash scripts/evaluate/multilayer_probe_pipeline.sh
```
