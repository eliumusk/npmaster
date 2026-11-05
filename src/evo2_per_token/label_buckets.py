"""PU-style target and weight construction for per-token BGC training."""
from __future__ import annotations

import numpy as np


def _validate_mask(name: str, mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match labels shape {shape}")
    return arr


def _soften_positive_edges(targets: np.ndarray, labels: np.ndarray, tokens: int, value: float) -> None:
    if tokens <= 0:
        return
    for row_idx in range(labels.shape[0]):
        row = labels[row_idx]
        pos = row == 1
        if not pos.any():
            continue
        diff = np.diff(pos.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            left_end = min(end, start + tokens)
            right_start = max(start, end - tokens)
            targets[row_idx, start:left_end] = value
            targets[row_idx, right_start:end] = value


def build_pu_targets(
    labels: np.ndarray,
    trusted_negative_mask: np.ndarray | None = None,
    hard_negative_mask: np.ndarray | None = None,
    pos_weight: float = 1.0,
    weak_negative_weight: float = 0.2,
    trusted_negative_weight: float = 1.0,
    hard_negative_weight: float = 0.7,
    boundary_soften_tokens: int = 0,
    boundary_target: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float targets and per-token weights for PU supervision.

    Input labels follow the existing convention: ``1`` positive, ``0`` unlabeled
    background, ``-1`` masked. Optional masks strengthen selected background
    positions without changing the target class.
    """
    y = np.asarray(labels)
    if y.ndim != 2:
        raise ValueError(f"labels must be rank-2, got shape {y.shape}")

    trusted = _validate_mask("trusted_negative_mask", trusted_negative_mask, y.shape)
    hard = _validate_mask("hard_negative_mask", hard_negative_mask, y.shape)

    targets = (y == 1).astype(np.float32)
    weights = np.zeros(y.shape, dtype=np.float32)
    weights[y == 1] = pos_weight
    weights[y == 0] = weak_negative_weight
    weights[(y == 0) & trusted] = trusted_negative_weight
    weights[(y == 0) & hard] = hard_negative_weight

    _soften_positive_edges(targets, y, boundary_soften_tokens, boundary_target)
    return targets.astype(np.float32), weights.astype(np.float32)
