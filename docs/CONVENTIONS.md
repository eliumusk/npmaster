# npmaster Conventions

Rules for the active tree after the 2026-04-26 mainline reset.

## Directory Responsibilities

| Dir | Contains | Does NOT contain |
|---|---|---|
| `src/` | Importable library code for active pipelines. | CLI entrypoints, argparse dispatch, output files. |
| `scripts/` | CLI entrypoints grouped by verb. | Shared logic that belongs in `src/`. |
| `experiments/` | Promoted active JSON/CSV/MD/plots. | Code, raw sweep debris, retired outputs. |
| `configs/` | Active YAML configs. | Code, credentials, stale experiment configs. |
| `docs/` | Current human-facing docs. | Historical stage reports. |
| `_archive/` | Retired local provenance, gitignored. | Active claims. |

## Naming

- Active original-v4.1 line is named `evo2_lr_multiscale`.
- Active per-token U-Net line is named `evo2_per_token_unet`.
- Use `nine_genome`, not `9genome`, in committed names.
- Do not add committed iteration suffixes: `_FIXED`, `_v1`, `_v2`, `_tmp`, `_old`, `_new`, `_debug`, `_final`, `_final2`.
- Hyperparameter tags may appear only in result subdirectories or config fields, not module paths.

## Mainline Scope

Only two detection directions are active:

- `evo2_lr_multiscale`: formal 9-genome mainline and OER learned-decoder handoff.
- `evo2_per_token_unet`: exploratory candidate, not formal until a clean pre-registered selection rule is run.

Everything else belongs in `_archive/2026-04-26-mainline-reset/` unless explicitly revived.

## Scripts

- Group by verb: `extract/`, `inference/`, `decode/`, `classify/`, `evaluate/`, `train/`, `utils/`.
- Scripts may insert `src/` onto `sys.path` once near the top.
- Scripts must not import other scripts.

## Experiments

Active result directories:

- `experiments/evo2_lr_multiscale/`
- `experiments/evo2_per_token_unet/`

Every active result should be documented in `docs/REGRESSION_LEDGER.md` or a local README.
