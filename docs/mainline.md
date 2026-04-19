# Current Mainline

Last updated: 2026-04-18

## Model

Phase 3 Route B context model:

- Backbone: frozen Evo2 7B embeddings (layer `blocks.20.mlp.l3`, mean-pooled per 16 384 bp window).
- Head: `ProjectionHead (warm-started) + DilatedContextCNN (d=1,2,4) + GatedResidual (β_init=-4) + Classifier`.
- Loss: per-window BCE on center region of L=32 chunks (stride 16), with optional hard-negative weighting.
- Optim: AdamW, layered lr (proj 1e-4, context/classifier 1e-3), 100 epochs, checkpoint by val region F1 every 5 epochs.

Config: `configs/mainline.yaml`.
Entry points: `src/train.py`, `src/evaluate.py`.

## Checkpoints

Mainline: `H_hardneg_r10_s{0,1,2}` (3-seed replication). Server: `/data/muskliu/npmaster/data/stage3_results/H_hardneg_r10_s*/`.

## Internal antiSMASH-Proxy Benchmark (3-seed mean ± std)

| Metric | `test_id` | `test_ood` |
|---|---:|---:|
| Recall | 0.820 ± 0.001 | 0.594 ± 0.006 |
| Prediction-level precision | 0.857 ± 0.002 | 0.653 ± 0.009 |
| <5 kb recall | 0.751 ± 0.004 | 0.556 ± 0.012 |
| 5–10 kb recall | 0.706 ± 0.003 | 0.651 ± 0.007 |

## 9-Genome External Benchmark (bench mode, frozen `val_frozen_fused`)

Recall 0.5934, precision 0.1649, F1 0.2580 (n_pred = 461). Full comparison table and diagnostic upper bound in `operating-points.md`. Metric contract in `evaluation-protocol.md`.

## Discovery (seed=0)

| Metric | Route B |
|---|---:|
| sig precision@20 | 90% |
| sig precision@50 | 92% |
| diversity@50 (signature profiles) | 45 |
| biosyn precision@50 | 100% |

Top signature domains in candidates: KR_C, SDR, PCP/T, Enoyl_CoA_hyd, AMP-binding, KS, KS_C, p450.

## Known Limitations

- `test_ood` seed variance still noticeably higher than `test_id`.
- Very short BGCs (`<5 kb`) recover only ~22% in pre-context model; Route B fixes visibility but not full separability.
- 9-genome precision gap vs BGC-Prophet — see `next-phase.md` for direction.

## Next Step

Hard-negative mining under `route-b-hardneg-v1.md`. Decoder work is considered closed; further gains go through training data / sampling.
