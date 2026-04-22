# evo2_lr — linear BGC detector on frozen Evo2 features

A minimal, reproducible BGC-region detector:
**FASTA → Evo2 mean-pool embeddings → logistic-regression score → hysteresis decoder → regions**.

Chosen after 40+ architecture/data variants all failed to beat a plain
linear probe under a strictly-honest evaluation protocol. See
`docs/superpowers/plans/2026-04-21-evo2-first-principles-v4.md` and
`docs/reports/overnight_exploration_2026-04-22.md`.

## Honest numbers on 9-genome (no 9-genome data used for selection)

| Method | Recipe | dedup_F1 (CI95) | strict_F1 | bench_F1 |
|---|---|---:|---:|---:|
| **evo2_lr v4.1 (recommended)** | w=8192 + w=4096 L<10k union | **0.488** (0.345–0.609) | **0.430** (0.284–0.564) | **0.519** (0.366–0.648) |
| evo2_lr v4 alt OP | w=8192, s=0.95/e=0.8/ms=3 | 0.487 (0.344–0.611) | 0.419 (0.272–0.551) | 0.511 (0.359–0.641) |
| evo2_lr v4 primary OP | w=8192, s=0.9/e=0.9/ms=1 | 0.432 | 0.384 | 0.488 |
| Route B CNN (same as defrag_v2, honest OP) | w=8192 | 0.270 | 0.195 | 0.270 |
| defrag_v2_ms10_s099 (published, 9g-dev tuned) | w=8192 | 0.439 | 0.287 | 0.439 |
| BGC-Prophet (external) | — | 0.391 | 0.302 | 0.393 |

**v4.1 recipe**: primary window-LR at w=8192 (alt OP) plus a short-BGC
rescue arm at w=4096 (OP s=0.95/e=0.8/ms=5) filtered to L<10 kb, unioned
with IoU-0.3 dedup. All operating points pre-declared. Improvement over
v4 is structural: +3 hits in the 5–10 kb bucket.

## Quick start

### End-to-end single-genome (v4 — single-scale)
```bash
PYTHONPATH=src python -m evo2_lr run \
  --fna /path/to/genome.fna.gz \
  --out-dir /tmp/evo2_lr_out \
  --op alt                          # or --op primary
```
Writes:
- `<genome>_embeddings.npy`      (N, 4096) fp16
- `<genome>_windows.csv`         contig,start,end
- `<genome>_window_scores.csv`   per-window sigmoid
- `<genome>_regions.csv`         contig,start,end,max_score

### v4.1 multi-scale honest readout on 9-genome

Assuming per-window prediction CSVs at w=8192 and w=4096 already exist:

```bash
python scripts/evo2_lr/v4_1_multiscale_readout.py \
  --primary-preds experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
  --short-preds   experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
  --gt-json       /data/syh/benchmark/nine_genome/ground_truth.json \
  --out-json      experiments/evo2_first_principles/v4_1_final_readout.json
```

Prints baseline, short-only, and ensemble metrics with per-bucket recall +
bootstrap CIs. The exact three numbers from the v4.1 headline (dedup 0.488,
strict 0.430, bench 0.519) reproduce byte-for-byte.

### Subcommands (single-scale pipeline)

```bash
# 1. extract (GPU-heavy — Evo2 forward pass)
python -m evo2_lr extract --fna genome.fna.gz --out-dir /tmp/e
# 2. predict (CPU, ms)
python -m evo2_lr predict --emb-npy /tmp/e/genome_embeddings.npy \
                          --windows-csv /tmp/e/genome_windows.csv \
                          --out-csv /tmp/e/genome_scores.csv
# 3. decode (CPU, ms)
python -m evo2_lr decode --window-scores /tmp/e/genome_scores.csv \
                         --op alt --out-csv /tmp/e/genome_regions.csv
```

## Operating points — prior-fixed, never corpus-tuned

### v4.1 mainline (two scales)

| scale | window | stride | start | extend | ms | length cap | rationale |
|---|---:|---:|---:|---:|---:|---:|---|
| primary | 8192 | 2048 | 0.95 | 0.80 | 3 | — | v4 alt OP — conservative seed + relaxed extend + 3-window filter |
| short_rescue | 4096 | 1024 | 0.95 | 0.80 | 5 | 10 000 bp | ms=5 → min region 8 kb, dedicated to 5–10 kb BGCs |

Union-dedup at IoU 0.3. See
`configs/experiments/evo2_first_principles_v4_1_frozen.yaml`.

### v4 legacy single-scale OPs (retained for reproducibility)

| OP | start | extend | ms | rationale |
|---|---:|---:|---:|---|
| primary | 0.90 | 0.90 | 1 | LR calibrated 0.9 cutoff; hysteresis off |
| alt     | 0.95 | 0.80 | 3 | conservative seed + relaxed extend + spike filter |

**Do not re-tune these from a corpus**. The v3 night run showed that
corpus-selection drifts into spurious low-threshold regimes when the
corpus's BGC-density distribution differs from the target.

## Config & weights

Frozen at `configs/experiments/evo2_first_principles_v4_1_frozen.yaml`.
LR weights live on the GPU server at
`/data/muskliu/npmaster/experiments/evo2_first_principles/baseline_a/lr_A1.npz`
(training: all positives + 200 k subsampled negatives from 1188 training
genomes, `class_weight='balanced'`, sklearn `LogisticRegression`, ~16 s fit).

Short-BGC arm uses the same LR weights applied at w=4096/s=1024 (mean-pool
is geometry-robust; we tested this explicitly).

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_evo2_lr.py -x
```

Smoke tests cover: v4 and v4.1 OP defaults, config parsing, sigmoid scoring,
primary vs alt decoder behavior on synthetic windows.

## What this package explicitly does NOT include

And why (40+ experiments rejected these):

- **No CNN / Transformer head** on frozen features: saturation breaks the
  hysteresis decoder; Route B drops to 0.27 honest dedup.
- **No gene-level LR/MLP/CNN/BiLSTM/Transformer** as mainline: all five
  architectures lose to window-LR; gene-Transformer catastrophically fails
  due to gene-caller distribution shift (9g uses pyrodigal, train uses
  antiSMASH CDS).
- **No tri-scale fusion** with w=2048: helps <5 kb recall (7/48 → 17/48) but
  collapses dedup (0.49 → 0.26) because LR isn't calibrated at that geometry.
- **No learned score fusion**: meta-training distribution shift makes the
  learned weight transfer badly (Path A dedup = 0.31 on 9g).
- **No proto_core-only training**: gains strict (+0.016) but loses dedup
  (-0.034); trade-off not Pareto.
- **No 9-genome-dev OP selection** (hard rule).

Full rejection log:
`docs/reports/overnight_exploration_2026-04-22.md`.

## Script inventory (cleaned)

- `scripts/evo2_lr/extract_batch_9genome.py` — Evo2 extraction on 9-genome FASTAs
- `scripts/evo2_lr/apply_lr_to_dir.py` — apply LR weights to a dir of embeddings
- `scripts/evo2_lr/train_lr.py` — train LR from labelled embeddings
- `scripts/evo2_lr/honest_readout.py` — single-scale v4 honest readout
- `scripts/evo2_lr/v4_1_multiscale_readout.py` — v4.1 ensemble honest readout (current mainline)
- `scripts/evo2_lr/genome_154945_diagnostic.py` — diagnostic for the shared blind-spot genome
- `scripts/evo2_lr/_op_sweep_helpers.py` — shared helpers for honest_readout
