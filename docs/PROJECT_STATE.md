# Project State

NP-Master currently has two active technical directions.

## 1. `evo2_lr_multiscale`

This is the renamed original v4.1 line. It uses frozen Evo2 mean-pooled embeddings, an LR scorer, and a multi-scale region decoder. This is the formal 9-genome benchmark line.

- 9-genome formal artifact: `experiments/evo2_lr_multiscale/nine_genome_readout.json`
- OER learned-decoder handoff: `experiments/evo2_lr_multiscale/oer004256_learned_decoder/`
- Config: `configs/evo2_lr_multiscale.yaml`

## 2. `evo2_per_token_unet`

This is the per-token U-Net candidate branch. It is useful for short metagenomic contigs and currently has promising OER readouts, but it is not a formal 9-genome claim because the model-family selection history saw 9-genome results.

- Readout: `experiments/evo2_per_token_unet/OER_UNET_READOUT.md`
- Config: `configs/evo2_per_token_unet.yaml`
- Status: diagnostic until a clean pre-registered checkpoint and threshold rule is executed.

## Archived Work

Historical explorations, old naming, defrag detection, multilayer probes, stage reports, and sweep debris were moved to `_archive/2026-04-26-mainline-reset/`. That directory is intentionally gitignored.
