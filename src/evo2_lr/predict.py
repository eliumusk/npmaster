"""Apply the frozen LR probe to mean-pool Evo2 embeddings → per-window scores.

The LR is a single dot-product + sigmoid; we use NumPy, no sklearn at inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_lr_weights(path: Path | str) -> tuple[np.ndarray, float]:
    d = np.load(Path(path))
    W = d["W"].astype(np.float32)
    b = float(d["b"])
    return W, b


def score_windows_from_embeddings(emb: np.ndarray, W: np.ndarray, b: float
                                    ) -> np.ndarray:
    """emb: (N, 4096) fp16 or fp32. Returns (N,) sigmoid probabilities in fp32.

    Uses a numerically stable sigmoid so very negative logits don't overflow.
    """
    x = emb.astype(np.float32, copy=False)
    logits = x @ W + b
    # Stable sigmoid: positive branch uses 1/(1+exp(-z)); negative branch uses exp(z)/(1+exp(z))
    out = np.empty_like(logits, dtype=np.float32)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    e = np.exp(logits[~pos])
    out[~pos] = e / (1.0 + e)
    return out
