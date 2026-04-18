# NP-Master

Biosynthetic Gene Cluster (BGC) detection and discovery on bacterial genomes. Frozen Evo2 7B embeddings + a light context model (dilated CNN + gated residual). Trained against antiSMASH annotations, formally evaluated on the 9-genome external benchmark.

## Current Status (9-genome, bench mode)

| Tool | R | P | F1 |
|---|---:|---:|---:|
| NP-Master (orig baseline) | 0.6361 | 0.0171 | 0.0334 |
| DeepBGC | 0.7213 | 0.0636 | 0.1169 |
| antiSMASH | 0.5574 | 0.1661 | 0.2559 |
| **NP-Master `val_frozen_fused`** | 0.5934 | 0.1649 | 0.2580 |
| **NP-Master `tuned_v2` (diagnostic)** | 0.5311 | 0.2513 | 0.3412 |
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 |

`bench` = asymmetric 50% overlap (both recall and precision sides; see `docs/evaluation-protocol.md`).

`val_frozen_fused` = operating point selected on internal val. `tuned_v2` = selected directly on 9-genome, diagnostic only.

## Layout

```
npmaster/
├── configs/
│   ├── mainline.yaml          # Phase 3 Route B training config
│   └── operating_points.yaml  # frozen decoder points for 9-genome
├── src/
│   ├── train.py               # end-to-end Route B training
│   ├── evaluate.py            # contig-aware region-level eval
│   ├── utils.py
│   ├── model/                 # ContextFullModel + context CNN + projector
│   └── data/                  # ChunkDataset + ContextDataset
├── scripts/
│   ├── build_chunks.py
│   ├── inference_stride.py    # Evo2 embed + model inference on FASTA
│   ├── postprocess_v2.py      # hysteresis region decoder (library)
│   ├── evaluate_nine_genome.py
│   ├── mine_hard_negatives.py
│   └── baselines/             # parse antiSMASH/DeepBGC/GECCO outputs
├── tests/
└── docs/
    ├── mainline.md
    ├── operating-points.md
    ├── evaluation-protocol.md
    ├── route-b-hardneg-v1.md
    └── next-phase.md
```

Data, checkpoints, wandb, results are **server-only** (gitignored). Local clone is code + configs + docs.

## Quick Start (server-side)

All commands assume `conda activate bgc` and you are in the repo root.

**Train mainline (seed 0):**

```bash
python -u src/train.py --config configs/mainline.yaml --seed 0 \
    --data-dir data/mibig_v2 --out-dir data/phase3/route_b/s0
```

**Run stride inference on a FASTA dir:**

```bash
python scripts/inference_stride.py \
    --fna-dir /data/syh/benchmark/nine_genome/fna \
    --ckpt data/stage3_results/H_hardneg_r10_s1/best.pt \
    --stride 8192 \
    --out-dir /tmp/npmaster_stride8192_s1
```

**Evaluate against 9-genome under a frozen operating point:**

```bash
python scripts/evaluate_nine_genome.py \
    --operating-point val_frozen_fused \
    --results-dir /tmp/npmaster_stride8192_s1 \
    --out /tmp/val_frozen_fused.json
```

`summary.bench` in the JSON matches `metrics` in `configs/operating_points.yaml`.

## Next Step

Hard-negative mining + training-side precision work. See `docs/route-b-hardneg-v1.md` and `docs/next-phase.md`. Decoder tuning is considered closed.
