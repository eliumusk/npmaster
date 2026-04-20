# Stage 5 Retrospective

_2026-04-20. Stage 5 concluded without a clear performance win; all
infrastructure retired. This doc preserves the diagnoses for future
reference; detailed numbers are in the three sibling docs linked below._

## One-paragraph TL;DR

Stage 5 set out to prove that NP-Master could match the 9-genome-tuned
`defrag_v2_ms10_s099` upper bound (`dedup_f1 = 0.4385`) under a
no-9-genome-tuning protocol. After 27 training runs across three
sub-stages (5a / 5b / 5c-1), the best legitimate result was
`baseline_core_v1` at **`dedup_f1 = 0.3995`** — a tie with BGC-Prophet
(0.3911), still 0.039 below the tuned upper bound. Because the gain
over BGC-Prophet is within seed noise and short-BGC recall never
moved, Stage 5 was retired without promoting any new operating point.

The **operating point in production is still `defrag_v2_ms10_s099`**
(unchanged throughout Stage 5).

## What the three stages found, in one line each

- **Stage 5a** — `bench_f1` alone is reward-hackable via fragmentation;
  `tight_pos_099`'s apparent 0.4385 bench F1 was actually 0.365 dedup.
  See `docs/stage5b-diagnostic-report.md`.
- **Stage 5b** — using antiSMASH `region` coords as internal val GT
  gives Spearman **−0.80** with 9-genome dedup F1. The proxy is
  actively backwards. See `docs/stage5b-diagnostic-report.md`.
- **Stage 5c-0 / 5c-1** — using antiSMASH `proto_core` coords flips
  the Spearman to **+0.80**. Training on proto_core labels lifts
  legitimate dedup_f1 from 0.256 (Stage 5b winner) to 0.400, but
  plateaus there. See `docs/stage5c1-results.md`.

## Durable lessons (worth keeping even after retiring Stage 5)

1. **Primary metric is `dedup_f1` + `bench_precision` + `bench_recall`.**
   `bench_f1` alone double-counts predictions that overlap the same GT
   and rewards fragmentation. This is now a project-wide convention —
   see `feedback_primary_metrics.md` in memory.

2. **Internal val GT must be in the same coordinate regime as the final
   evaluation.** `bgc_annotations.csv` uses antiSMASH `region` (wide,
   +10-20 kb flank); 9-genome GT is human-curated tight core.
   Selecting a model on region-val actively picks the model *least*
   likely to generalise to 9-genome.

3. **Short-BGC recall (`<5kb`, `<10kb`) is a structural window-size
   problem, not a label-reweighting problem.** With `window=8192`, a
   3 kb core has window-side overlap ≈ 0.37, so every overlap
   threshold ≥ 0.5 masks it. `short_bgc_upweight_v1`,
   `short_core_upweight_v1`, and tighter thresholds all failed to move
   `<5kb` recall past ~0.30. The only plausible fix is a
   `window=2048` short-window head (deferred, never built).

4. **antiSMASH JSONs already contain tight-core coordinates** —
   `proto_core`, `cand_cluster` — that we hadn't been using. We were
   training against the widest layer (`region`) and evaluating against
   tight human GT. `proto_core` is the right layer to use when
   generalisation target is tight.

5. **Training pool is 100 % antiSMASH.** Zero MIBiG clusters in
   `bgc_annotations.csv`. MIBiG injection remains unexplored.

## Artefacts retained

- `docs/stage5-data-audit.md` — data layer audit (training pool
  composition, label rule reverse-engineering, prior tuning soft-leak
  audit, architecture feasibility notes).
- `docs/stage5b-diagnostic-report.md` — the fragmentation + proxy
  inversion diagnosis with plots.
- `docs/stage5c1-results.md` — the final 9-genome numbers for Stage 5c-1
  candidates, including the `baseline_core_v1 = 0.3995` figure and its
  caveats.
- Memory: `feedback_primary_metrics.md`, `user_data_bottleneck.md`.

## Artefacts retired

- All `scripts/build_stage5_labels.py`, `apply_stage5_meta.py`,
  `run_stage5_data_sweep.py`, `stage5b_val_decoder_sweep.py`,
  `parse_antismash_core_coords.py`, `rebuild_windows_metadata_core.py`,
  `dump_val_predictions.py`, `plot_stage5_fragmentation.py`,
  launchers, orchestrator — deleted.
- All `configs/sweeps/stage5_*.yaml`, `configs/splits/nine_genome_embargo_v1.json` — deleted.
- All `experiments/stage5_*`, `experiments/stage5b_*`,
  `experiments/stage5c0_*`, `experiments/stage5c1-*` — deleted locally
  and on server (`/data/muskliu/npmaster/experiments/stage5*`,
  `/data/muskliu/npmaster/data/stage5/`,
  `/data/muskliu/npmaster/data/bgc_annotations_core.csv`).
- Planning docs (`stage5-no-nine-genome-tuning.md`, `stage5c1-plan.md`,
  `stage5-overnight-status.md`) — deleted.

See commit retiring this work for the full deletion list.

## If Stage 5 is revisited

Don't redo Stage 5a (bench_f1 sweeping) or Stage 5b (region-val
selection) — they are both proven wrong above. Start from:

1. **Use `proto_core` as internal val GT.** Re-parse antiSMASH JSONs
   if needed; the parser was ~150 lines and retired with the other
   infrastructure, but the structure is simple (`proto_core` features
   have `location` in the form `[start:end](strand)`).
2. **Build a short-window (`window=2048`) head** before any more
   label variants. Short-BGC recall is the dominant missed-recall
   bucket once you're past the proxy problem.
3. **Respect the primary metric convention.** Any future claim must
   hold on `dedup_f1` before it counts.

The `defrag_v2_ms10_s099` operating point in
`configs/operating_points.yaml` remains the production point.
