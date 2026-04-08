# Stage 3: BGC-GCD training pipeline

Implements Stage 3 per `stage3_detailed.md`:
- MLP projection head (4096 → 2048 → 256, L2-normalized)
- Multi-label classifier (256 → 8 sigmoid logits)
- DANN domain classifier with GRL and λ warmup (0→1 over first 20 epochs)
- Multi-label-aware Supervised Contrastive loss (positive pair = share ≥1 type bit)
- BalancedBatchSampler (50/50 pos/neg, round-robin across 7 types for positives)
- Embedding-space augmentation (Gaussian noise / random dim dropout)

## Quick start (diagnostic data)

On the server at `/data/muskliu/np_master/bgc-gcd/`:

```bash
# 1) adapt diagnostic-day1 pickles -> embeddings.npy + labels.csv
python -m data.adapt_diagnostic --src_dir /data/muskliu/np_master/data --out_dir data

# 2) smoke test (1 epoch, C_full config)
python train.py --ablation configs/ablations/C_full.yaml --epochs 1 --tag smoke

# 3) full ablation sweep
python run_ablation.py --seeds 0 1 2 --skip_d

# 4) generate loss curves
python plot_curves.py --logs_dir results --out_dir results/figures
```

## Pseudo-phylum on diagnostic data

The diagnostic pkl files do not preserve the `phylum` field from their
source. For code-validation purposes we assign a synthetic 2-class
pseudo-phylum based on window-id prefix:
- `BGC*` → `MIBiG_src` (MIBiG BGC windows)
- `NEG*` → `Background_src` (background genome windows)

This exactly mirrors the known source bias. DANN's job here is to make
the projection no longer distinguish "MIBiG source vs. background source",
i.e. to remove the very spurious signal the source-bias diagnosis flagged
in earlier work. Real phylum assignment will be swapped in once Stage 1
produces `windows_metadata.csv`.
