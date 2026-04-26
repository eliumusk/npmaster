"""Per-nucleotide BGC labels from antiSMASH region coordinates.

Training labels live in `data/bgc_annotations.csv` as (genome, contig, start_bp, end_bp)
spans. Training windows live in `data/windows/edge_padding/windows_metadata_edgepad_filtered.csv`
as (genome, contig, window_start, window_end) rows with a window-level `label`
(positive/negative/ambiguous).

For per-token (dense) prediction we need, for each training window, a per-nucleotide
label vector of length `T == window_end - window_start`:

    y[i] = 1  if (win_start + i) is inside any antiSMASH region on this contig
    y[i] = 0  otherwise
    y[i] = -1 for ambiguous — windows where the existing window-level label is
           'ambiguous'; we exclude the entire window from dense loss to match
           the per-window training protocol (`stage5-data-audit.md`).

Evo2 tokenizes bytes 1:1 so token index == nucleotide offset inside the cleaned
window sequence. For edge-padded windows we additionally mask the padded tail
so loss only flows through real nucleotides.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Window:
    window_id: str
    genome_id: str
    contig_id: str
    start: int
    end: int
    label: str           # positive/negative/ambiguous
    split: str           # train/val/test_id/test_ood
    actual_length: int   # unpadded length


def load_bgc_annotations(path: str | Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Return {(genome_id, contig_id): sorted list of (start_bp, end_bp)}."""
    out: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (row["genome_id"], row["contig_id"])
            out[key].append((int(row["start_bp"]), int(row["end_bp"])))
    for k in out:
        out[k].sort()
    return dict(out)


def load_windows(path: str | Path) -> list[Window]:
    out: list[Window] = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append(Window(
                window_id=row["window_id"],
                genome_id=row["genome_id"],
                contig_id=row["contig_id"],
                start=int(row["window_start"]),
                end=int(row["window_end"]),
                label=row["label"],
                split=row["split"],
                actual_length=int(row["actual_length"]),
            ))
    return out


def per_nt_labels(
    windows: list[Window],
    bgc_map: dict[tuple[str, str], list[tuple[int, int]]],
    ambiguous_as_mask: bool = True,
) -> np.ndarray:
    """Produce dense per-nucleotide labels for a list of windows.

    Returns int8 array of shape (N_windows, max_T) with values in {-1, 0, 1}.
    All windows are assumed to have the same length (max_T = windows[0].end - windows[0].start);
    padded positions (idx >= window.actual_length) are set to -1.

    -1 = mask / ambiguous / padded
     0 = non-BGC nucleotide
     1 = BGC nucleotide (inside an antiSMASH region)
    """
    if not windows:
        return np.zeros((0, 0), dtype=np.int8)
    T = windows[0].end - windows[0].start
    for w in windows:
        if (w.end - w.start) != T:
            raise ValueError(f"heterogeneous window length: {w.window_id} {w.end - w.start} vs {T}")

    y = np.zeros((len(windows), T), dtype=np.int8)
    for i, w in enumerate(windows):
        # Padded-tail mask.
        if w.actual_length < T:
            y[i, w.actual_length:] = -1

        # Ambiguous entire window → mask all real positions.
        if ambiguous_as_mask and w.label == "ambiguous":
            y[i, :w.actual_length] = -1
            continue

        # Paint BGC regions inside the window.
        for (bs, be) in bgc_map.get((w.genome_id, w.contig_id), []):
            if be <= w.start or bs >= w.end:
                continue
            lo = max(bs, w.start) - w.start
            hi = min(be, w.end) - w.start
            if lo < hi:
                y[i, lo:hi] = np.where(y[i, lo:hi] == -1, -1, 1)
    return y


def summary_stats(y: np.ndarray) -> dict:
    """Quick sanity stats on a dense-label array."""
    total = y.size
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    mask = int((y == -1).sum())
    return {
        "n_windows": int(y.shape[0]),
        "per_window_len": int(y.shape[1]),
        "n_positions": int(total),
        "positive": pos,
        "negative": neg,
        "masked": mask,
        "positive_rate": round(pos / max(1, pos + neg), 4),
    }
