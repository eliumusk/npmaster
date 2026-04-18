# Operating Points

Two frozen region-decoding operating points. Parameters in `configs/operating_points.yaml`. Metrics reported under `bench` mode (asymmetric 50% overlap, see `evaluation-protocol.md`). Both assume `H_hardneg_r10_s1` checkpoint and `stride=8192` inference.

## `val_frozen_fused` — formal reporting

Selected on internal val (generalization-safe, never sees 9-genome).

| field | value |
|---|---|
| score | `fused` |
| start_threshold / extend_threshold | 0.98 / 0.95 |
| max_gap | 0 |
| length_cap | 50 000 bp |
| min_support_windows | 1 |

9-genome bench: **R = 0.5934, P = 0.1649, F1 = 0.2580** (n_pred = 461).

Use this point for any paper/reviewer-facing claim.

## `tuned_v2` — diagnostic upper bound

Selected directly on 9-genome. Shows what the current model + decoder can achieve when the benchmark is used for hyperparameter fitting. **Do not quote as a formal result.**

| field | value |
|---|---|
| score | `max_type` |
| start_threshold / extend_threshold | 0.95 / 0.95 |
| max_gap | 0 |
| length_cap | 30 000 bp |
| min_support_windows | 3 |

9-genome bench: **R = 0.5311, P = 0.2513, F1 = 0.3412** (n_pred = 975).

## Reproducing

```bash
# on server (after stage 3 clone)
cd /data/muskliu/npmaster
python scripts/evaluate_nine_genome.py \
    --operating-point val_frozen_fused \
    --results-dir /tmp/npmaster_stride8192_s0_merged \
    --out /tmp/val_frozen_fused.json

python scripts/evaluate_nine_genome.py \
    --operating-point tuned_v2 \
    --results-dir /tmp/npmaster_stride8192_s0_merged \
    --out /tmp/tuned_v2.json
```

`bench` in `summary` will match the metrics above exactly.

## Context — Baselines on Same Metric

| Tool | R | P | F1 |
|---|---:|---:|---:|
| NP-Master (orig default) | 0.6361 | 0.0171 | 0.0334 |
| DeepBGC | 0.7213 | 0.0636 | 0.1169 |
| antiSMASH | 0.5574 | 0.1661 | 0.2559 |
| **NP-Master val_frozen_fused** | 0.5934 | 0.1649 | 0.2580 |
| **NP-Master tuned_v2** | 0.5311 | 0.2513 | 0.3412 |
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 |

Decoder work is ~done. The remaining precision gap vs BGC-Prophet is expected to close via training-side work, not further decoder tuning — see `next-phase.md`.
