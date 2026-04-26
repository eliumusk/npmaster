# eval_harness

Baseline freeze for architecture comparison. Rerun any candidate's row with `scripts/evaluate/unified_readout.py --method NAME` once that script exists — for now the table is hand-consolidated from the sources listed in `baseline_table.json._meta._sources`.

## Files

- `baseline_table.json` — structured dict keyed on method name (`v4_1_hysteresis`, `v4_1_learned`, `antismash`). Each method has a `nine_genome` block and an `oer004256_ge10kb` block. `_meta` carries source pointers, gates, and the prior null result.
- `baseline_table.md` — human-readable comparison table with the same data, grouped by dataset.

## Adding a new candidate

1. Produce a 9g readout JSON via the v4.1-style multiscale pipeline (so `dedup_f1` / `bench_precision` / `bench_recall` / `strict_f1` / bucket-recall are directly comparable).
2. Produce an OER regions CSV with the same column schema as `experiments/OER004256/v4_1_learned/regions_fully_typed.csv`.
3. Add a new top-level key to `baseline_table.json` and a new row to `baseline_table.md`.
4. Check against `_meta.gates_for_future_candidates` before declaring the candidate viable.

Never overwrite the three existing rows in-place — they are the frozen reference.
