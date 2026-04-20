#!/usr/bin/env python3
"""Rebuild windows_metadata with overlap computed against proto_core coords.

Reads the existing ``windows_metadata_hardneg.csv`` and
``bgc_annotations_core.csv`` and emits a new CSV with the same rows
but with the overlap-derived columns recomputed against core coords:

  * ``max_bgc_overlap_ratio`` — max over overlapping cores of
    ``max(overlap / window_length, overlap / core_length)``. The
    asymmetric max captures "window mostly contains a core" cases
    that the pure window-side ratio would miss for short cores
    (median proto_core length is 2.5 kb, well below the 8 kb window
    so pure window-side ratio tops out at ~0.3).
  * ``nearest_bgc_distance`` — min start-to-end gap to any core on
    the same contig (``inf`` if none).
  * ``bgc_length_max`` — length of the core that achieves the max
    overlap ratio (NaN if window has zero overlap with any core).
  * ``overlaps_truncated_bgc`` — set to 0 (truncation flag not tracked
    for proto_core; downstream variants that need it fall back gracefully).
  * ``label`` — ``positive`` if overlap ratio ≥ 0.5, ``ambiguous`` if
    0 < ratio < 0.5 or within 10 kb of a core, else ``negative``.

All other columns (``genome_id``, ``contig_id``, ``window_start``,
``window_end``, ``Polyketide`` ... ``Other``, ``is_negative``,
``phylum``, ``split``, ``edge_padded``, ``actual_length``,
``hard_negative``) are passed through unchanged. Row order is
preserved so the output is drop-in for ``apply_stage5_meta.py``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd


def _process_contig(
    group: tuple[tuple[str, str], pd.DataFrame],
    core_by_key: dict[tuple[str, str], np.ndarray],
    dist_threshold_bp: int,
    positive_threshold: float,
) -> pd.DataFrame:
    key, sub = group
    sub = sub.copy()

    cores = core_by_key.get(key)
    if cores is None or len(cores) == 0:
        sub["max_bgc_overlap_ratio"] = 0.0
        sub["nearest_bgc_distance"] = np.inf
        sub["bgc_length_max"] = np.nan
        sub["overlaps_truncated_bgc"] = 0
        sub["label"] = "negative"
        return sub

    starts = sub["window_start"].to_numpy()
    ends = sub["window_end"].to_numpy()

    core_starts = cores[:, 0]
    core_ends = cores[:, 1]
    core_lens = cores[:, 2]

    # Broadcast-compute per-window max asymmetric overlap ratio.
    # Shape: (n_windows, n_cores)
    ov_lo = np.maximum(starts[:, None], core_starts[None, :])
    ov_hi = np.minimum(ends[:, None], core_ends[None, :])
    ov = np.clip(ov_hi - ov_lo, 0, None).astype(np.float64)

    win_len = np.maximum(1, ends - starts).astype(np.float64)
    core_len_bc = np.maximum(1, core_lens)[None, :].astype(np.float64)

    ratio = np.maximum(ov / win_len[:, None], ov / core_len_bc)
    # Best-matching core index per window (only meaningful when ov>0).
    best_idx = ratio.argmax(axis=1)
    best_ratio = ratio.max(axis=1)
    sub["max_bgc_overlap_ratio"] = best_ratio

    # Nearest core distance: 0 if overlap, else min of (win_end < core_start) or (core_end < win_start).
    overlaps_any = ov.max(axis=1) > 0
    dist = np.full(len(sub), np.inf, dtype=np.float64)
    # Without overlap: min distance from interval endpoints.
    if not overlaps_any.all():
        not_ov = ~overlaps_any
        # distance from window to each core: max(0, max(core_start-win_end, win_start-core_end))
        d_left = core_starts[None, :] - ends[:, None]  # positive if core to the right
        d_right = starts[:, None] - core_ends[None, :]  # positive if core to the left
        dmat = np.maximum(0, np.maximum(d_left, d_right))
        dist[not_ov] = dmat[not_ov].min(axis=1)
    dist[overlaps_any] = 0
    sub["nearest_bgc_distance"] = dist

    # bgc_length_max: length of the best-matching core when overlap>0, else NaN.
    blen = np.where(overlaps_any, core_lens[best_idx], np.nan)
    sub["bgc_length_max"] = blen

    sub["overlaps_truncated_bgc"] = 0

    label = np.where(
        best_ratio >= positive_threshold,
        "positive",
        np.where(
            (best_ratio > 0) | (dist < dist_threshold_bp),
            "ambiguous",
            "negative",
        ),
    )
    sub["label"] = label
    return sub


def rebuild(
    windows_csv: Path,
    bgc_core_csv: Path,
    out_csv: Path,
    positive_threshold: float = 0.5,
    dist_threshold_bp: int = 10_000,
    n_workers: int = max(1, mp.cpu_count() // 2),
) -> None:
    print(f"[rebuild] loading {windows_csv}")
    w = pd.read_csv(windows_csv)
    print(f"[rebuild]   rows={len(w)}")

    # Preserve original row order for alignment with meta_arrays.pkl.
    w["__orig_order"] = np.arange(len(w))

    print(f"[rebuild] loading {bgc_core_csv}")
    b = pd.read_csv(bgc_core_csv)
    b = b[b["bgc_length"] > 0].copy()
    print(f"[rebuild]   cores={len(b)}")

    # Group cores by (genome_id, contig_id).
    core_by_key: dict[tuple[str, str], np.ndarray] = {
        k: sub[["start_bp", "end_bp", "bgc_length"]].to_numpy(dtype=np.int64)
        for k, sub in b.groupby(["genome_id", "contig_id"], sort=False)
    }
    print(f"[rebuild]   core-keyed contigs={len(core_by_key)}")

    # Split windows by (genome, contig) for embarrassingly-parallel processing.
    groups = list(w.groupby(["genome_id", "contig_id"], sort=False))
    print(
        f"[rebuild] window contigs={len(groups)}, workers={n_workers}"
    )

    worker = partial(
        _process_contig,
        core_by_key=core_by_key,
        dist_threshold_bp=dist_threshold_bp,
        positive_threshold=positive_threshold,
    )
    with mp.Pool(n_workers) as pool:
        chunks = pool.map(worker, groups, chunksize=16)
    merged = pd.concat(chunks, axis=0, ignore_index=False)
    merged = merged.sort_values("__orig_order").drop(columns="__orig_order")

    # Summary.
    print("[rebuild] label distribution:")
    print(merged["label"].value_counts())
    print(
        "[rebuild] positives per length bucket (of source core):"
    )
    pos = merged[merged["label"] == "positive"]
    for lo, hi, name in [
        (0, 5000, "<5kb"),
        (5000, 10000, "5-10kb"),
        (10000, 20000, "10-20kb"),
        (20000, 50000, "20-50kb"),
        (50000, 10**9, ">=50kb"),
    ]:
        n = pos[
            (pos["bgc_length_max"].fillna(-1) >= lo)
            & (pos["bgc_length_max"].fillna(-1) < hi)
        ].shape[0]
        print(f"[rebuild]   {name}: {n}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[rebuild] wrote {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path(
            "/data/muskliu/npmaster/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv"
        ),
    )
    parser.add_argument(
        "--bgc-core-csv",
        type=Path,
        default=Path(
            "/data/muskliu/npmaster/data/bgc_annotations_core.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/data/muskliu/npmaster/data/stage5/windows_metadata_core.csv"
        ),
    )
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--dist-threshold-bp", type=int, default=10_000)
    parser.add_argument(
        "--n-workers", type=int, default=max(1, mp.cpu_count() // 2)
    )
    args = parser.parse_args()
    rebuild(
        windows_csv=args.windows_csv,
        bgc_core_csv=args.bgc_core_csv,
        out_csv=args.out,
        positive_threshold=args.positive_threshold,
        dist_threshold_bp=args.dist_threshold_bp,
        n_workers=args.n_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
