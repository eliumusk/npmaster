# Stage 5b Selection Decision — Freeze

**Protocol:** candidates selected on shared internal val GT (= antiSMASH
regions from `bgc_annotations.csv` on val-split genomes, 165 genomes,
2012 BGCs). Primary metric `dedup_f1`, tie-break `bench_precision`,
`bench_recall`. Decoder cells must pass
`dedup_precision / bench_precision ≥ 0.75` (fragmentation
consistency gate). **9-genome data was not inspected during selection.**

See `experiments/stage5b_val_decoder_sweep_all_cells.csv` for the full
sweep and `experiments/stage5b_val_decoder_sweep_selected.csv` for the
per-checkpoint winner list.

## Frozen candidates for 9-genome final eval

| rank | tag | variant | decoder (start / ext / ms / cap) | val dedup_f1 | val bench_P | val bench_R | val frag_ratio |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | baseline_v1_s0 | baseline_v1 | 0.99 / 0.85 / 1 / None | 0.7333 | 0.7793 | 0.7187 | 0.961 |
| 2 | short_bgc_upweight_v1_s0 | short_bgc_upweight_v1 | 0.99 / 0.95 / 1 / None | 0.7315 | 0.7889 | 0.7112 | 0.955 |
| 3 | tight_pos_075_v1_s0 | tight_pos_075_v1 | 0.90 / 0.85 / 2 / None | 0.7161 | 0.9428 | 0.5810 | 0.991 |
| 4 | tight_pos_099_v1_s0 | tight_pos_099_v1 | 0.90 / 0.85 / 2 / None | 0.5865 | 0.9667 | 0.4230 | 0.989 |

### Rationale per candidate

- **baseline_v1_s0** — top internal-val dedup_f1. This is the honest
  "if Stage 5b selection is trusted" candidate.
- **short_bgc_upweight_v1_s0** — statistically tied with baseline on
  internal val, but with markedly better `<5kb` (0.66) and `5-10kb`
  (0.68) length-bucket recall. Needed to test whether short-BGC
  upweighting helps on 9-genome's 48 `<5kb` + 89 `5-10kb` GTs.
- **tight_pos_075_v1_s0** — top tight variant on internal val. Precise
  (bench_P 0.94) but misses all short BGCs.
- **tight_pos_099_v1_s0** — same checkpoint as Stage 5a's top pick,
  but paired with its own Stage 5b-selected decoder (start=0.90, ms=2,
  cap=None) instead of val_frozen_fused. Purpose: test whether the
  fragmentation problem (Stage 5a dedup_f1 0.365) disappears once the
  decoder is picked under the gate.

## Rules for 9-genome eval

1. Run inference + eval on ONLY the 4 checkpoints above, each paired
   with the decoder listed in its row. No further sweeping.
2. Metrics to report, in order: `dedup_f1`, `bench_precision`,
   `bench_recall`, then length-bucket recall/precision for `<5kb`,
   `5-10kb`, `10-20kb`, `20-50kb`, `>=50kb`.
3. Frame the result report as a diagnostic — "raw F1 high but
   fragmented" / "internal val proxy inverted" — not as a claim of
   beating any baseline.
4. After 9-genome eval completes, this file is locked. Do not re-run
   selection with new information.

## Reference points (from prior sweeps, for comparison only)

| method | bench_f1 | dedup_f1 | n_pred | notes |
|---|---:|---:|---:|---|
| `defrag_v2_ms10_s099` | 0.439 | **0.439** | 140 | 9-genome-tuned upper bound |
| BGC-Prophet | 0.393 | **0.391** | 170 | external baseline |
| Stage 5a tight_pos_099 + val_frozen_fused | 0.439 | 0.365 | 399 | fragmented reward-hack |
| Stage 5a baseline + val_frozen_fused | 0.258 | 0.183 | 461 | pre-Stage-5 status quo |
