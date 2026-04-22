"""End-to-end: FASTA → LR scores → decoded BGC regions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import extract, predict, decode
from .config import FrozenConfig, PRIMARY_OP, ALT_OP


def _windows_from_scores(dfw: pd.DataFrame, scores: np.ndarray
                         ) -> list[decode.WindowScore]:
    return [
        decode.WindowScore(str(c), int(s), int(e), float(bs))
        for c, s, e, bs in zip(dfw["contig"], dfw["start"], dfw["end"], scores)
    ]


def apply_from_embeddings(
    emb_path: Path | str,
    windows_csv: Path | str,
    lr_weights_path: Path | str,
    op: dict | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, int, int, float]]]:
    """Second half of the pipeline when embeddings are already extracted.

    Returns (per-window score DataFrame, list of (contig,start,end,score) regions).
    """
    if op is None:
        op = ALT_OP
    emb = np.load(emb_path)
    dfw = pd.read_csv(windows_csv)
    W, b = predict.load_lr_weights(lr_weights_path)
    probs = predict.score_windows_from_embeddings(emb, W, b)
    windows = _windows_from_scores(dfw, probs)
    regions = decode.decode_regions(
        windows,
        start_threshold=op["start_threshold"],
        extend_threshold=op["extend_threshold"],
        min_support_windows=op["min_support_windows"],
        max_gap=op["max_gap"],
    )
    score_df = pd.DataFrame({
        "contig": dfw["contig"], "start": dfw["start"].astype(int),
        "end": dfw["end"].astype(int),
        "bgc_score": probs.astype(np.float32),
        "non_bgc_score": (1 - probs).astype(np.float32),
    })
    return score_df, regions


def run_pipeline(
    fna_path: Path | str,
    out_dir: Path | str,
    *,
    frozen_cfg: FrozenConfig | None = None,
    op: dict | None = None,
) -> dict:
    """FASTA → embeddings → scores → regions. Writes all intermediate artifacts."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if frozen_cfg is None:
        from .config import load_frozen_config
        frozen_cfg = load_frozen_config()
    if op is None:
        op = frozen_cfg.alt_op

    genome = Path(fna_path).name
    for ext in (".fna", ".fna.gz", ".fasta", ".fasta.gz"):
        if genome.endswith(ext):
            genome = genome[: -len(ext)]
            break

    emb_path = out_dir / f"{genome}_embeddings.npy"
    win_path = out_dir / f"{genome}_windows.csv"
    if not (emb_path.exists() and win_path.exists()):
        extract.extract_embeddings(
            fna_path, emb_path, win_path,
            window=frozen_cfg.window, stride=frozen_cfg.stride,
            layer=frozen_cfg.layer,
        )

    score_df, regions = apply_from_embeddings(
        emb_path, win_path, frozen_cfg.lr_weights_path, op=op
    )
    score_df.to_csv(out_dir / f"{genome}_window_scores.csv", index=False)
    regs_df = pd.DataFrame(regions, columns=["contig", "start", "end", "max_score"])
    regs_df.to_csv(out_dir / f"{genome}_regions.csv", index=False)
    return {
        "genome": genome,
        "embeddings": str(emb_path),
        "window_scores_csv": str(out_dir / f"{genome}_window_scores.csv"),
        "regions_csv": str(out_dir / f"{genome}_regions.csv"),
        "n_windows": int(len(score_df)),
        "n_regions": int(len(regions)),
        "op": op,
    }
