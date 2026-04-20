# Stage 5c-1 Selection Decision — Freeze

**Protocol:** all 3 variants trained on `windows_metadata_core.csv`
labels (proto_core overlap ≥ 0.5 / ≥ 0.75), 60 epochs warm-started
from `H_hardneg_r10`. Decoder swept against core-val GT
(`bgc_annotations_core.csv` on val-split genomes). Primary metric
`dedup_f1`, fragmentation gate `dedup_precision / bench_precision ≥ 0.95`.
**No 9-genome data inspected during selection.**

Source: `experiments/stage5c1_core_sweep_selected.csv`.

## Frozen candidates (top-2 for 9-genome final)

| rank | tag | decoder (start / ext / ms / cap) | core-val dedup_f1 | bench_P | bench_R | frag_ratio |
|---:|---|---|---:|---:|---:|---:|
| 1 | short_core_upweight_v1_s0 | 0.97 / 0.95 / 3 / None | 0.3202 | 0.322 | 0.318 | 1.000 |
| 2 | baseline_core_v1_s0 | 0.90 / 0.85 / 3 / None | 0.3175 | 0.374 | 0.276 | 1.000 |

Not evaluated on 9-genome (rank 3): `tight_core_v1_s0`
(dedup_f1=0.309, start=0.99 ms=1 — hitting the ms=1 minimum suggests
it wants an even less strict decoder, but the gate excludes such
cells).

## Training peak metrics (internal-val region F1, diagnostic only)

| trial | peak val_region_f1 | peak auroc |
|---|---:|---:|
| baseline_core_v1 | 0.8609 | 0.9815 |
| short_core_upweight_v1 | 0.8553 | 0.9824 |
| tight_core_v1 | 0.8495 | 0.9811 |

## Rules for 9-genome final eval

1. Inference + eval on the 2 checkpoints above, each under its frozen
   decoder — no decoder re-tuning on 9-genome.
2. Report primary metrics (`dedup_f1`, `bench_precision`,
   `bench_recall`) first. Length-bucket recall + precision for
   `<5kb / 5-10kb / 10-20kb / 20-50kb / >=50kb` as secondary.
3. After 9-genome numbers arrive, this file is locked. No new
   variants may be added to Stage 5c-1.

## Reference points for interpretation

| method | dedup_f1 | source |
|---|---:|---|
| `defrag_v2_ms10_s099` | 0.439 | 9-genome-tuned (upper bound) |
| BGC-Prophet | 0.391 | external baseline |
| Stage 5b diagnostic (tight_pos_099 + core decoder) | 0.398 | not legitimate selection |
| Stage 5b selected winner (baseline_v1, region-val) | 0.256 | picked on broken proxy |

Success criterion per `docs/stage5c1-plan.md`: `dedup_f1 ≥ 0.40` on
9-genome for at least one winner. Stretch: ≥ 0.44.
