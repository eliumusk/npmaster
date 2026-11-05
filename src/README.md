# src/

Importable library code for the active pipelines.

| Package | Purpose |
|---|---|
| `evo2_lr/` | Frozen Evo2 mean-pool + LR scorer utilities. |
| `evo2_per_token/` | Per-token labels, U-Net model, and token-to-region decoding. |
| `decoding/` | Shared region postprocessing, `evo2_lr_multiscale`, and learned decoder. |
| `benchmark/` | 9-genome region-level metrics. |
| `classification/` | Optional per-type classification helpers. |
| `data/` | Minimal package marker; old Route-B datasets are archived. |

No CLI argparse belongs in `src/`; use `scripts/` for entrypoints.
