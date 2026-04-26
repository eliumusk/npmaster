# Operating Points

Two frozen region-decoding operating points. Parameters in `configs/operating_points.yaml`. Metrics reported under `bench` mode (asymmetric 50% overlap, see `evaluation-protocol.md`). Both assume the Phase 3 Route B checkpoint `best_route_b_s0.pt` (seed 0) and `stride=8192` inference. Seeds 1 and 2 have not been re-inferred at stride 8192 yet.

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

## `v4_1_learned` — metagenomic recall-oriented (2026-04-23)

Replaces the hand-tuned hysteresis decoder with a learned 1D CNN decoder (`src/decoding/learned.py`). Tuned to close the 7.8× antiSMASH:v4.1 recall gap on metagenomic short contigs that `min_support_windows=3` structurally rejects. See `docs/learned-decoder.md`.

| field | value |
|---|---|
| decoder | `Decoder1DCNN(depth=3, hidden=32)` ckpt `c00_cnn3_h32_antismash_bce/decoder.pt` |
| primary arm | w=8192 / s=2048 |
| short arm | w=4096 / s=1024 |
| primary threshold | 0.95 |
| short threshold | 0.98 |
| short length cap | 10,000 bp |
| union-dedup IoU | 0.3 |

9-genome dedup F1 vs GT: **0.332** under the canonical `benchmark.nine_genome.eval_regions` metric (vs v4.1-hysteresis **0.488** — a -0.156 F1 trade for metagenomic recall). The older **0.461 vs 0.541** pair used `full_pipeline_eval.py`'s greedy-pairing metric and must not be compared to the formal 0.488 v4.1 number.
OER004256: **1,230 regions** (vs hysteresis 254), AS-recall **37%** (vs 12%), 15/15 original novel candidates preserved.

Use this point for metagenomic applications (OER004256, environmental samples). For 9-genome benchmarking (paper tables), continue using `val_frozen_fused` / v4.1 multi-scale hysteresis.

Entrypoint: `scripts/decode/v4_1_learned.py`.

## Reproducing

```bash
# on server (after stage 3 clone)
cd /data/muskliu/npmaster
python scripts/evaluate/nine_genome.py \
    --operating-point val_frozen_fused \
    --results-dir /tmp/npmaster_stride8192_s0_merged \
    --out /tmp/val_frozen_fused.json

python scripts/evaluate/nine_genome.py \
    --operating-point tuned_v2 \
    --results-dir /tmp/npmaster_stride8192_s0_merged \
    --out /tmp/tuned_v2.json
```

`bench` in `summary` will match the metrics above exactly.

## Context — Baselines on Same Metric

| Tool | R | P | F1 | median pred len |
|---|---:|---:|---:|---:|
| DeepBGC | 0.7213 | 0.0636 | 0.1169 | 4.9 kb |
| antiSMASH | 0.5574 | 0.1661 | 0.2559 | 29.8 kb |
| **NP-Master val_frozen_fused** | 0.5934 | 0.1649 | 0.2580 | — |
| NP-Master tuned_v2 *(diagnostic)* | 0.5311 | 0.2513 | 0.3412 | — |
| BGC-Prophet | 0.3475 | 0.4529 | 0.3933 | 22.7 kb |

Reproduce with `scripts/evaluate/baselines_nine_genome.py` (see README Quick Start).

Caveats when reading this table:

- GT comes from the BGC-Prophet 9-genome test set, gene-level "tight core" convention. Median GT length ≈ 10.9 kb, no flanks. Tools whose native output includes flanking regions (antiSMASH ≈ 30 kb median) are mechanically penalised on precision by the asymmetric 50% rule — low antiSMASH precision is largely a length-convention artifact, not detection quality.
- BGC-Prophet's top F1 is partly explained by convention alignment between its output and the GT it authored.
- Decoder work is ~done. The remaining precision gap vs BGC-Prophet is expected to close via training-side work, not further decoder tuning — see `next-phase.md`.
