"""evo2_lr — a minimal linear-probe BGC detector on frozen Evo2 features.

Pipeline: FASTA → Evo2 mean-pool embeddings → logistic-regression probability →
hysteresis region decoder → BGC regions.

Entry points:
  - evo2_lr.extract.extract_embeddings(fna_path, window, stride)
  - evo2_lr.predict.score_windows(embeddings, lr_weights)
  - evo2_lr.decode.decode_regions(windows, start, extend, ms)
  - evo2_lr.cli.main            (python -m evo2_lr <subcommand>)

Frozen config: configs/experiments/evo2_first_principles_v4_frozen.yaml.
Protocol and honest-readout numbers: docs/superpowers/plans/2026-04-21-evo2-first-principles-v4.md.
"""
from .decode import decode_regions, WindowScore
from .predict import score_windows_from_embeddings, load_lr_weights
from .config import FrozenConfig, load_frozen_config, PRIMARY_OP, ALT_OP

__all__ = [
    "decode_regions", "WindowScore",
    "score_windows_from_embeddings", "load_lr_weights",
    "FrozenConfig", "load_frozen_config",
    "PRIMARY_OP", "ALT_OP",
]
