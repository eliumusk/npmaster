"""Smoke tests for evo2_lr — uses synthetic data; no Evo2 model needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evo2_lr.config import (  # noqa: E402
    PRIMARY_OP, ALT_OP, V4_1_SCALES, V4_1_IOU_DEDUP, load_frozen_config,
)
from evo2_lr.decode import WindowScore, decode_regions  # noqa: E402
from evo2_lr.predict import score_windows_from_embeddings  # noqa: E402


def test_v4_legacy_ops_are_prior_declared():
    """v4 baseline OPs kept as legacy constants for reproducibility."""
    assert PRIMARY_OP["start_threshold"] == 0.90
    assert PRIMARY_OP["extend_threshold"] == 0.90
    assert PRIMARY_OP["min_support_windows"] == 1
    assert PRIMARY_OP["max_gap"] == 0
    assert ALT_OP["start_threshold"] == 0.95
    assert ALT_OP["extend_threshold"] == 0.80
    assert ALT_OP["min_support_windows"] == 3
    assert ALT_OP["max_gap"] == 0


def test_v4_1_scales_are_prior_declared():
    assert V4_1_SCALES["primary"]["window"] == 8192
    assert V4_1_SCALES["primary"]["stride"] == 2048
    assert V4_1_SCALES["primary"]["operating_point"]["start_threshold"] == 0.95
    assert V4_1_SCALES["short_rescue"]["window"] == 4096
    assert V4_1_SCALES["short_rescue"]["stride"] == 1024
    assert V4_1_SCALES["short_rescue"]["operating_point"]["min_support_windows"] == 5
    assert V4_1_SCALES["short_rescue"]["length_filter_max"] == 10000
    assert V4_1_IOU_DEDUP == 0.3


def test_load_frozen_config_parses_v4_1():
    cfg = load_frozen_config()
    # v4.1 yaml: primary scale is w=8192 alt OP (what v4 called alt)
    assert cfg.window == 8192
    assert cfg.stride == 2048
    assert cfg.primary_op["start_threshold"] == 0.95
    assert cfg.primary_op["extend_threshold"] == 0.80
    assert cfg.primary_op["min_support_windows"] == 3
    # short rescue scale
    assert "short_rescue" in cfg.scales
    assert cfg.scales["short_rescue"].window == 4096
    assert cfg.scales["short_rescue"].length_filter_max == 10000
    assert cfg.iou_dedup == 0.3


def test_score_windows_sigmoid():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 4096)).astype(np.float32)
    W = rng.standard_normal(4096).astype(np.float32)
    b = 0.0
    probs = score_windows_from_embeddings(emb, W, b)
    assert probs.shape == (100,)
    assert probs.dtype == np.float32
    assert (probs >= 0).all() and (probs <= 1).all()


def test_decode_primary_op_picks_high_score_contiguous_regions():
    # One high-score run on contig A; an isolated spike on contig B
    wins = [
        WindowScore("A", 0, 8192, 0.3),
        WindowScore("A", 2048, 10240, 0.95),
        WindowScore("A", 4096, 12288, 0.98),
        WindowScore("A", 6144, 14336, 0.96),
        WindowScore("A", 8192, 16384, 0.4),
        WindowScore("B", 0, 8192, 0.99),  # isolated spike
        WindowScore("B", 2048, 10240, 0.1),
    ]
    regs = decode_regions(wins, 0.90, 0.90, 1, 0)
    # Primary OP (ms=1) → isolated B spike kept
    contigs = {r[0] for r in regs}
    assert "A" in contigs
    assert "B" in contigs


def test_decode_alt_op_filters_isolated_spikes():
    # Same setup, alt OP requires 3 consecutive windows → B's single spike dropped
    wins = [
        WindowScore("A", 0, 8192, 0.3),
        WindowScore("A", 2048, 10240, 0.96),
        WindowScore("A", 4096, 12288, 0.98),
        WindowScore("A", 6144, 14336, 0.96),
        WindowScore("A", 8192, 16384, 0.4),
        WindowScore("B", 0, 8192, 0.99),  # isolated → dropped by ms=3
        WindowScore("B", 2048, 10240, 0.1),
    ]
    regs = decode_regions(wins, 0.95, 0.80, 3, 0)
    contigs = {r[0] for r in regs}
    assert "A" in contigs
    assert "B" not in contigs
