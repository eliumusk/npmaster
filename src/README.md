# src/

Importable library code. **No CLIs, no argparse, no `__main__` blocks that do real work.**

Scripts in `scripts/` import from here. Cross-src imports are fine.

## Subpackages

| Package | Purpose |
|---|---|
| `data/` | `ChunkDataset`, `ContextDataset` — training-time data loaders. |
| `model/` | `ContextFullModel` (Route B backbone), projector, context CNN. |
| `evo2_lr/` | Evo2 7B extraction + mean-pool LR pipeline. Includes `extract_batch.py` (batched + layer-truncated FASTA→embedding). |
| `decoding/` | Region decoders. `postprocess.py` = hysteresis threshold+merge. `v4_1.py` = multi-scale ensemble logic (primary w=8192 + short rescue w=4096, union-dedup). |
| `benchmark/` | 9-genome benchmark math. `nine_genome.py` = region-level F1 under bench/loose/strict/iou rules. `op_sweep.py` = internal-val OP grid. |
| `classification/` | `per_type.py` — Stage-2 LR helpers, antiSMASH product→7-class map, region IoU pairing. |
| `baselines/` | Third-party tool parsers (antiSMASH / DeepBGC / GECCO / BGC-Prophet). Read-only output readers. |
| `plotting/` | Region-track and boundary comparison plots. |
| `experiments/` | Sweep runtime infrastructure (`layout`, `manifest`, `runbook`). Historically used by the run_stage* launchers (now archived). |
| `evaluate.py` | Training-loop evaluation hooks (not to be confused with `benchmark/`). |
| `train.py` | Route B end-to-end training entry. |
| `utils.py` | Seeding, logging helpers. |

## Import convention

Callers (scripts/, tests/) add `src/` to `sys.path` at the top:

```python
REPO_ROOT = Path(__file__).resolve().parents[N]
sys.path.insert(0, str(REPO_ROOT / "src"))
from decoding.postprocess import WindowScore, decode_regions_hysteresis
from benchmark.nine_genome import eval_regions
```

Intra-src imports follow the same flat form (`from data.chunk_dataset import ...`, `from model.context_full_model import ...`). This is the sole allowed `sys.path.insert`.

## What does not belong here

- CLI argparse dispatching — goes in `scripts/<verb>/`.
- Experiment outputs — go in `experiments/<target>/`.
- Hardcoded absolute paths (`/data/...`, `/root/...`).
