# Next Phase: Training-Side Precision

## Problem Statement

After `val_frozen_fused` and `tuned_v2`, the 9-genome `bench` F1 is ~0.26 formal / ~0.34 diagnostic. BGC-Prophet sits at F1 ≈ 0.39 on the same metric. The gap is almost entirely on the **precision** side (we lead on recall on all but BGC-Prophet, and we're not far behind there).

Postprocess sweep is done (v1→v2 gain was only ΔF1 ≈ 0.004). The window-level score study shows:

- Internal val negatives are well separated (AUROC 0.98+).
- 9-genome negatives have a saturated tail — ~12% of "negative" windows score ≥ 0.95.
- Of 90 high-score 9-genome negatives, 89 are also called by at least one external tool (antiSMASH / DeepBGC / BGC-Prophet).

That last bullet is the headline: most of our precision loss on this benchmark is not model-random-error, it's **benchmark-GT narrowness**. The model is finding BGC-like regions the benchmark GT doesn't list.

## Route 1: Hard-Negative Mining (active — `route-b-hardneg-v1.md`)

Mine top-K high-scoring windows from training negatives, weight their loss, retrain Route B. Minimal intervention:

- mining: `scripts/mine_hard_negatives.py` (formerly `mine_route_b_hardneg.py`)
- training: `src/train.py` + `src/data/chunk_dataset.py` accept the mined flag
- decision rule for 3-seed promotion: see `route-b-hardneg-v1.md::Decision Rule`

## Route 2: Data-Side Grid (after Route 1)

Only start if Route 1 shows movement. Knobs to sweep:

- **Window length** — current 16384 bp. Try 8192 / 32768.
- **Stride** — current 8192. Try 4096 / full-overlap.
- **Negative sampling ratio** — current train mix. Try reducing negatives or re-weighting by phylum.
- **Edge padding** — keep `edge_padding_filtered` windows; test phase-aware padding variants.

Evaluation protocol frozen (see `evaluation-protocol.md`). No decoder re-tuning — if a data change doesn't move `bench F1` under the frozen operating point, it's not real signal.

## Route 3: Benchmark GT Re-examination

Low priority but cheap. Take the 89 / 90 supported-by-other-tools high-score negatives from the score-family diagnosis. Two candidate outcomes:

- They're genuinely BGC-like → reclassify → we're underestimating our own precision.
- They're not → harvest as hard-negatives for Route 1.

Either way this feeds back into Route 1.

## What We Are NOT Doing Yet

Skip until Routes 1–2 are played out:

- CNN depth / dilation changes
- Chunk length changes
- Different Evo2 layer / different backbone
- Loss function redesign (focal, contrastive, etc.)

These are model-architecture changes. The data evidence says the bottleneck isn't there yet.
