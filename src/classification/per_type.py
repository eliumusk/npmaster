"""Shared per-type classification helpers (v4.1 Stage-2 LR + region pairing).

Used by:
  - scripts/evaluate/classification_v4_1_nine_genome.py
  - scripts/evaluate/classification_defrag_v2_nine_genome.py
  - scripts/classify/v4_1_regions.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Canonical 7-class ordering for training annotations + Stage-2 per-type LRs.
# NOT the same as the CNN head column order (which is in memory/reference_cnn_class_order.md).
BGC_TYPES = ["Alkaloid", "Terpene", "NRP", "Polyketide", "RiPP", "Saccharide", "Other"]


def build_product_to_type_map(annotations_csv: Path) -> dict[str, str]:
    """Derive raw-product → canonical 7-class map from training annotations.

    `bgc_types` is ';'-separated when hybrid (e.g. 'NRP;Polyketide') — take the
    first listed as canonical primary, group by `bgc_product`, majority-vote.
    """
    df = pd.read_csv(annotations_csv, usecols=["bgc_types", "bgc_product"])
    df = df.dropna(subset=["bgc_types", "bgc_product"])
    df["primary_type"] = df["bgc_types"].str.split(";").str[0]
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, r in df.iterrows():
        counts[str(r["bgc_product"]).strip()][r["primary_type"]] += 1
    return {prod: c.most_common(1)[0][0] for prod, c in counts.items()}


def map_antismash_products(products_str: str, prod_to_type: dict[str, str]) -> list[str]:
    """Map an antiSMASH area's `products` (comma-joined) to unique 7-classes.

    Unknown products fall back to 'Other' so nothing is silently discarded.
    """
    out = []
    for p in (products_str or "").split(","):
        p = p.strip()
        if not p:
            continue
        t = prod_to_type.get(p, "Other")
        if t not in out:
            out.append(t)
    return out


def load_type_lrs(type_lr_dir: Path) -> dict[str, dict]:
    """Load the 7 per-type LR heads. Each .npz contains at least 'W' and 'b'."""
    lrs = {}
    for t in BGC_TYPES:
        p = type_lr_dir / f"lr_type_{t}.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing per-type LR: {p}")
        z = np.load(p)
        lrs[t] = dict(z)
    return lrs


def type_lr_forward(emb: np.ndarray, weights: dict) -> np.ndarray:
    """Apply a binary LR to (N, D) embeddings → probability-of-positive (N,).

    Handles sklearn-style (coef_/intercept_) and npz-custom (W/b or w/b).
    """
    if "coef_" in weights and "intercept_" in weights:
        W = weights["coef_"].reshape(-1)
        b = float(weights["intercept_"].ravel()[0])
    elif "W" in weights and "b" in weights:
        W = weights["W"].reshape(-1)
        b = float(np.ravel(weights["b"])[0])
    elif "w" in weights and "b" in weights:
        W = weights["w"].reshape(-1)
        b = float(np.ravel(weights["b"])[0])
    else:
        raise KeyError(f"unknown LR npz layout; keys={list(weights.keys())}")
    z = emb @ W + b
    return 1.0 / (1.0 + np.exp(-z))


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def iou_pair(a, b) -> float:
    """a, b are (contig, start, end, ...)."""
    if a[0] != b[0]:
        return 0.0
    ov = overlap_len(a[1], a[2], b[1], b[2])
    if ov == 0:
        return 0.0
    return ov / max(1, max(a[2], b[2]) - min(a[1], b[1]))
