"""Region decoder — wraps the hysteresis decoder from scripts/postprocess_v2.

Public API:
  WindowScore(contig, start, end, score)
  decode_regions(windows, start_threshold, extend_threshold, min_support_windows, max_gap)
    → list[tuple[contig, start, end, score]]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from decoding.postprocess import WindowScore, decode_regions_hysteresis  # noqa: E402


def decode_regions(
    windows: list[WindowScore],
    start_threshold: float,
    extend_threshold: float,
    min_support_windows: int,
    max_gap: int = 0,
) -> list[tuple[str, int, int, float]]:
    """Decode BGC regions from per-window scores under a hysteresis policy.

    Parameters mirror `configs/experiments/evo2_first_principles_v4_frozen.yaml`.
    Returns list of (contig, start, end, max_score_within_region).
    """
    raw = decode_regions_hysteresis(
        windows,
        start_threshold=start_threshold,
        extend_threshold=extend_threshold,
        min_support_windows=min_support_windows,
        max_gap=max_gap,
    )
    # Re-attach per-region max score for downstream ranking/fusion.
    out = []
    for reg in raw:
        contig, rs, re_ = reg[0], int(reg[1]), int(reg[2])
        mx = 0.0
        for w in windows:
            if w.contig == contig and w.end > rs and w.start < re_:
                mx = max(mx, w.score)
        out.append((contig, rs, re_, mx))
    return out
