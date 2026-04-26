# scripts/

CLI entrypoints grouped by verb. Active scripts support only the two current mainlines.

| Directory | Active purpose |
|---|---|
| `extract/` | Feature extraction launchers for active pipelines. |
| `inference/` | Apply active models to cached features. |
| `decode/` | Convert scores/probabilities into BGC region calls. |
| `evaluate/` | Evaluate active mainline outputs. |
| `train/` | Train active LR / learned-decoder / U-Net components. |
| `classify/` | Add optional type labels to predicted regions. |
| `utils/` | Sync and housekeeping. |

Retired diagnostics, sweeps, plots, and one-off scripts are in `_archive/2026-04-26-mainline-reset/scripts/`.
