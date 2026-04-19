# Project Timeline

Serial dependency chain of NP-Master experiments. Read top-to-bottom: each node
consumes the previous node's winning artifacts. When a new sweep is planned,
append it here — the filename/config/script never needs to encode the ordering
(content-name + version is enough).

## Serial chain

| # | Name | What it does | Depends on |
|---|---|---|---|
| 1 | `decoder_v1` | decoder-only micro-sweep on frozen Route B predictions | `phase3/route_b/best_route_b_s0.pt` |
| 2 | `tiling_v1` | window/stride tiling sweep (inference-time) | `decoder_v1` top decoder params |
| 3 | `tiling_decoder_v1` | decoder re-sweep on top-3 `tiling_v1` predictions (CPU-only) | `tiling_v1` top-3 tilings |
| 4 | `hardneg_v1` | hard-negative mining + training warm-started from Route B | `tiling_decoder_v1` winning tiling (w8192/s2048) + `warm_starts/H_hardneg_r10/best_H_hardneg_r10.pt` |
| 5 | `defrag_v2` | de-fragmentation decoder sweep on `hardneg_v1` predictions (CPU-only) | `hardneg_v1` winner `best_tk100-ms0-hnw4.pt` |
| 6 | `training_sweep_v1` | training-hyperparam sweep over the `hardneg_v1` mining output | `hardneg_v1` winner mining `meta_arrays.pkl` (**currently not on disk — needs re-run**) |
| 7 | `no_9gt_v1` | no-9-genome-tuning protocol (plan in `docs/no-9gt-v1.md`) | Nothing yet (greenfield; will consume `warm_starts/` + `training_labels/`) |

## Current live operating point

`defrag_v2_ms10_s099` — selected from `defrag_v2` on 5-genome dev, frozen,
lockbox-validated on the other 4 genomes. All parameters in
`configs/operating_points.yaml`. Backing checkpoint:
`data/phase3/hardneg_sweep_v1/best/best_tk100-ms0-hnw4.pt` (from `hardneg_v1`).

## Naming rules (2026-04-20)

- **Phase N** — stable project milestones (Phase 3 = training mainline,
  Phase 4 = benchmark evaluation infra). Used in `data/phase3/`, `data/phase4/`,
  and `src/**/*.py` docstrings.
- **Stage / numbered sweeps** — deprecated in filenames. If you need to talk
  about ordering, use this timeline. Every sweep has a content name + version.
- **Content name + version** — `hardneg_v1`, `defrag_v2`, `no_9gt_v1`, etc.
  Applied uniformly to sweep yamls (`configs/sweeps/<name>.yaml`), scripts
  (`scripts/run_<name>_sweep.py`), tests (`tests/test_<name>_sweep.py`),
  experiment output dirs (`experiments/<name>-*/`), and promoted artifacts
  (`data/phase3/hardneg_sweep_v1/best/...`).
- **Role-named data dirs** — `data/warm_starts/`, `data/training_labels/` —
  pre-refactor artifacts still feeding current training, named by role rather
  than stage-of-origin.
