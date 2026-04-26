# Full-Data DDP16 U-Net Readout

**Date**: 2026-04-26
**Status**: completed; diagnostic
**Run**: `full_e10style_ddp16_gb32_seed0`

## Training

This run repeats the E10 U-Net recipe without train/val subsampling.

- Train windows: 608,721
- Val windows: 92,351
- Model: 1D U-Net, hidden=64
- Loss: `dice_bce`
- Optimizer: AdamW, lr=1e-3
- Batch: 2 per rank, 16 ranks, global batch=32
- Hardware: node7 + node8, 16x A800
- Best epoch: 27
- Best val loss: 0.957645
- Best epoch token P/R/F1: 0.8451 / 0.7084 / 0.7707

Artifacts:

- Server checkpoint: `/data/muskliu/npmaster/experiments/evo2_per_token_unet/train_runs/full_e10style_ddp16_gb32_seed0/best.pt`
- Local training log: `experiments/evo2_per_token_unet/train_runs/full_e10style_ddp16_gb32_seed0/train_log.jsonl`
- Local config: `experiments/evo2_per_token_unet/train_runs/full_e10style_ddp16_gb32_seed0/config.json`

## 9-Genome Readout

Same decoder as E10: `min_len_bp=2000`, `max_gap_bp=500`, thresholds 0.3/0.4/0.5.

| threshold | dedup_F1 | bench_P | bench_R | n_pred | median_len |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.3247 | 0.2427 | 0.5869 | 548 | 10,240 |
| 0.4 | 0.3485 | 0.2735 | 0.5574 | 501 | 10,240 |
| 0.5 | 0.3771 | 0.3112 | 0.5475 | 466 | 10,240 |

Compared with old E10 (`dedup_F1=0.4561/0.4664/0.4751` at 0.3/0.4/0.5), full-data training increases recall but hurts precision and dedup_F1. The model is overcalling and/or fragmenting on 9-genome under the current decoder.

## OER004256 Readout

Reference: antiSMASH regions under asymmetric 50% overlap.

| threshold | n_pred | AS-recall | novel | median_len | 10-20kb | 20-50kb | >=50kb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 3,297 | 0.7050 | 1,613 | 8,678 | 0.798 | 0.615 | 0.454 |
| 0.4 | 2,893 | 0.6437 | 1,223 | 8,192 | 0.731 | 0.557 | 0.415 |
| 0.5 | 2,594 | 0.5931 | 948 | 8,184 | 0.687 | 0.491 | 0.376 |

For comparison, old E10 at threshold 0.3 had AS-recall 0.3117 with 1,377 predictions and 470 novel calls. The full-data model is much more sensitive on OER, but the candidate set is substantially larger and needs precision/triage control before wet-lab handoff.

## Interpretation

Full-data training is not a clean win. It strongly improves OER antiSMASH recall, especially on 10-50 kb contigs, but it damages the held-out 9-genome region metric because the current token-to-region decoder produces too many calls. The next step should not be another architecture change. It should be a decoder/validation pass: control fragmentation, widen/narrow behavior, and candidate burden while preserving the OER recall gain.
