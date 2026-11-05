"""Per-token probabilities → BGC regions.

For each (contig, window) we have a (T,) array of P(BGC). To decode into
region coordinates we:

1. Convert per-window per-token probs into a per-contig per-nt probability
   track by placing each window's probs at its original (start, end) and
   averaging where windows overlap (stride < window).
2. Threshold the track. Merge runs of `prob >= threshold` into candidate
   regions. Drop runs shorter than `min_len_bp` (default 2 kb — half of the
   minimum antiSMASH region length in training).
3. Optional: fill single-nt gaps of length < `max_gap_bp` (default 500 bp).

The per-nt track handles overlap correctly by construction (same token
covered by multiple windows gets averaged), which is exactly what per-token
prediction was supposed to buy us — no hysteresis / min_support hacks.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class TokenRegion:
    contig: str
    start: int
    end: int
    score: float  # mean P inside the region


def assemble_per_contig_track(
    window_probs: dict[str, np.ndarray],       # window_id → (T,) probs
    window_coords: dict[str, tuple[str, int, int]],  # window_id → (contig, start, end)
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {contig: (sum_probs, counts)} so final track = sum / count."""
    by_contig: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for wid, p in window_probs.items():
        c, s, e = window_coords[wid]
        # Trim to real length (in case T > e-s due to padding).
        L = e - s
        p = p[:L] if p.shape[0] >= L else np.concatenate([p, np.zeros(L - p.shape[0], dtype=p.dtype)])
        by_contig[c].append((s, e, p.astype(np.float32)))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c, items in by_contig.items():
        max_end = max(e for (_, e, _) in items)
        acc = np.zeros(max_end, dtype=np.float32)
        cnt = np.zeros(max_end, dtype=np.int32)
        for (s, e, p) in items:
            acc[s:e] += p
            cnt[s:e] += 1
        out[c] = (acc, cnt)
    return out


def decode_per_contig(
    track_sum: np.ndarray,
    track_count: np.ndarray,
    threshold: float = 0.5,
    min_len_bp: int = 2_000,
    max_gap_bp: int = 500,
) -> list[tuple[int, int, float]]:
    """Threshold + merge. Returns list of (start, end, mean_prob)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        track = np.where(track_count > 0, track_sum / np.maximum(track_count, 1), 0.0)

    above = (track >= threshold).astype(np.int8)
    if above.sum() == 0:
        return []

    # Close short gaps: fill any run of 0s shorter than max_gap_bp.
    diff = np.diff(above, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if max_gap_bp > 0:
        merged_s, merged_e = [starts[0]], [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - merged_e[-1] < max_gap_bp:
                merged_e[-1] = e
            else:
                merged_s.append(s); merged_e.append(e)
        starts, ends = np.array(merged_s), np.array(merged_e)

    regions: list[tuple[int, int, float]] = []
    for s, e in zip(starts, ends):
        if e - s < min_len_bp:
            continue
        seg = track[s:e]
        regions.append((int(s), int(e), float(seg.mean())))
    return regions


def decode_all(
    window_probs: dict[str, np.ndarray],
    window_coords: dict[str, tuple[str, int, int]],
    threshold: float = 0.5,
    min_len_bp: int = 2_000,
    max_gap_bp: int = 500,
) -> list[TokenRegion]:
    tracks = assemble_per_contig_track(window_probs, window_coords)
    out: list[TokenRegion] = []
    for c, (s, cnt) in tracks.items():
        for (rs, re_, sc) in decode_per_contig(s, cnt, threshold, min_len_bp, max_gap_bp):
            out.append(TokenRegion(c, rs, re_, sc))
    return out
