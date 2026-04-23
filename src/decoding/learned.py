"""Learned 1D CNN decoder — replaces hysteresis `postprocess.decode_regions_hysteresis`.

Takes a per-window bgc_score sequence along a contig and predicts per-window
P(inside a real BGC), trained on antiSMASH-matched labels from the training
pool. Sequence decoding = threshold + merge consecutive windows meeting
threshold (no hard min_support_windows requirement).

Motivation: hysteresis OP (`start=0.95, min_support_windows=3, extend=0.80`)
was tuned on 9-genome isolate contigs. On metagenomic short contigs (10-20kb,
~3-5 windows at stride 2048), the 3-window minimum structurally rejects BGCs
the underlying LR_A1 scorer IS detecting (per-window p50 = 0.85 on
antiSMASH-called OER regions our hysteresis missed). A learned decoder over
the window-score sequence adapts to the real score distribution without
hard-coded length assumptions.

See `docs/learned-decoder.md` for training protocol + OP selection.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --- Frozen operating point for v4.1-learned multi-scale ---------------------
# Selected from primary × short 2×4 sweep (2026-04-23). At this OP:
#   9g vs GT dedup F1 = 0.461 (vs hysteresis 0.541, acceptable)
#   OER: 1230 regions (vs hysteresis 254, antiSMASH 1976)
#   OER AS-recall = 37% (vs hysteresis 12%)
#   15/15 original novel candidates preserved
PRIMARY_THRESHOLD = 0.95
SHORT_THRESHOLD = 0.98
SHORT_LENGTH_CAP = 10_000   # match v4.1 hysteresis: short arm only rescues <10 kb regions
IOU_DEDUP = 0.3             # match v4.1 hysteresis union-dedup


# --- Decoder architecture ----------------------------------------------------
class Decoder1DCNN(nn.Module):
    """Small dilated 1D CNN over per-window LR scores.

    Input:  [B, T] float scores ∈ [0, 1]
    Output: [B, T] logits (apply sigmoid for probabilities)

    Frozen config for v4.1-learned: ``depth=3, hidden=32`` (c00 in sweep).
    """

    def __init__(self, depth: int = 3, hidden: int = 32):
        super().__init__()
        dilations = [2**i for i in range(depth)]
        layers: list[nn.Module] = [nn.Conv1d(1, hidden, 3, padding=1), nn.GELU()]
        for d in dilations:
            layers += [nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d), nn.GELU()]
        layers.append(nn.Conv1d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        return self.net(x.unsqueeze(1)).squeeze(1)


def load_decoder(ckpt_path: Path, device: torch.device | str = "cuda:0",
                 depth: int = 3, hidden: int = 32) -> Decoder1DCNN:
    model = Decoder1DCNN(depth=depth, hidden=hidden).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# --- Region emission ---------------------------------------------------------
@torch.no_grad()
def decode_contig(
    model: Decoder1DCNN,
    starts: np.ndarray,
    ends: np.ndarray,
    raw_scores: np.ndarray,
    threshold: float,
    device: torch.device | str = "cuda:0",
) -> list[tuple[int, int, float]]:
    """Apply decoder to one contig's window sequence.

    Returns a list of (region_start, region_end, max_raw_score) intervals where
    probs >= threshold. No min_support_windows constraint — a single
    qualifying window can produce a region.
    """
    if len(raw_scores) == 0:
        return []
    x = torch.from_numpy(raw_scores.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x).squeeze(0).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    mask = probs >= threshold
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    rs = np.where(diff == 1)[0]
    re_ = np.where(diff == -1)[0]
    out: list[tuple[int, int, float]] = []
    for i, j in zip(rs, re_):
        out.append((int(starts[i]), int(ends[j - 1]), float(raw_scores[i:j].max())))
    return out


# --- Directory-level decode --------------------------------------------------
def decode_dir(
    preds_dir: Path,
    model: Decoder1DCNN,
    threshold: float,
    device: torch.device | str = "cuda:0",
) -> dict[str, list[tuple]]:
    """Decode regions from a directory of per-genome/shard window CSVs.

    Each CSV: columns ``contig, start, end, bgc_score``.
    Returns ``{genome: [(contig, start, end, max_score), ...]}`` (identical to
    :func:`decoding.v4_1.decode_dir` schema, swap-in compatible).
    """
    import pandas as pd  # local import to keep cold-start cheap

    out: dict[str, list[tuple]] = {}
    for csv_path in sorted(Path(preds_dir).glob("*_predictions.csv")):
        genome = csv_path.name[: -len("_predictions.csv")]
        df = pd.read_csv(csv_path, usecols=["contig", "start", "end", "bgc_score"])
        df = df.sort_values(["contig", "start"]).reset_index(drop=True)
        regions: list[tuple] = []
        for contig, sub in df.groupby("contig", sort=False):
            starts = sub["start"].to_numpy(np.int32)
            ends = sub["end"].to_numpy(np.int32)
            scores = sub["bgc_score"].to_numpy(np.float32)
            for (s, e, sc) in decode_contig(model, starts, ends, scores, threshold, device):
                regions.append((str(contig), s, e, sc))
        out[genome] = regions
    return out


# --- Multi-scale (primary + short) union-dedup -------------------------------
def iou_1d(a: tuple[int, int], b: tuple[int, int]) -> float:
    o = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return o / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def union_dedup(
    primary_regions: list[tuple],    # [(contig, s, e, score)]
    short_regions: list[tuple],
    short_length_cap: int = SHORT_LENGTH_CAP,
    iou_thresh: float = IOU_DEDUP,
) -> list[tuple]:
    """Union-dedup primary + short regions, keeping higher score on overlap.

    Short regions are filtered to length < short_length_cap before dedup
    (short arm is a rescue for 5-10 kb short BGCs; at full length it overlaps
    with primary redundantly).
    """
    short_filt = [r for r in short_regions if (r[2] - r[1]) < short_length_cap]
    tagged = [(*r, "primary") for r in primary_regions] + [(*r, "short_rescue") for r in short_filt]

    per_contig: dict = defaultdict(list)
    for r in tagged:
        per_contig[r[0]].append(r)

    kept: list[tuple] = []
    for contig, lst in per_contig.items():
        lst_sorted = sorted(lst, key=lambda r: (-r[3], 0 if r[4] == "primary" else 1))
        picked: list[tuple] = []
        for r in lst_sorted:
            if any(iou_1d((r[1], r[2]), (k[1], k[2])) >= iou_thresh for k in picked):
                continue
            picked.append(r)
        kept.extend(picked)
    return kept
