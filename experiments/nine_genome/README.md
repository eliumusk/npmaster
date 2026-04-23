# nine_genome/

9-genome benchmark results (dev + lockbox split, 305 GT regions, 9 bacterial genomes).

Always honest — no operating points tuned on these. OPs are picked on internal-val or declared from first principles, then evaluated once here.

## Files

| File | What | Source script |
|---|---|---|
| `defrag_v2_classification.json` | defrag_v2 CNN per-type top-1 = 0.827 (top-any 0.971) vs antiSMASH gold. Fixed 2026-04-23 for CNN class-order bug. | `scripts/evaluate/classification_defrag_v2_nine_genome.py` |
| `defrag_v2_op_sweep/summary.json` | Exhaustive OP sweep on 9-dev that selected `defrag_v2_ms10_s099` (start=0.99, extend=0.95, ms=10). 815 KB. | (historical — sweep infra retired, see `_archive/scripts/run_stage*`) |
| `stage2b_decoder_on_top_tilings_v1/summary.{csv,json}` | Stage 2b decoder-OP sweep across 96 trials × 3 tilings. Per-trial dirs archived under `_archive/experiments/`. | (historical) |
| `stage3b_defrag_v2/summary.csv` | Stage 3b defrag_v2 hard-negative sweep summary. | (historical) |

## Canonical benchmark numbers

Frozen in `docs/REGRESSION_LEDGER.md`. Any value here that contradicts the ledger without a matching commit is a regression — check git log.

## Raw per-window preds

`{genome}_predictions.csv` files live on the GPU server under `/data/muskliu/npmaster/`. Not committed — too large.
