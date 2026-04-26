"""Reusable datasets and sampling helpers for per-token training."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def deterministic_subset(total: int, requested: int, seed: int) -> np.ndarray | None:
    """Return a stable sorted subset, or ``None`` when no subsampling is needed."""
    if requested <= 0 or requested >= total:
        return None
    rng = np.random.RandomState(seed)
    return np.sort(rng.permutation(total)[:requested])


class PreparedArrayDataset(Dataset):
    """Dataset over prepared feature/label arrays, including mmap-backed arrays."""

    def __init__(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        if feats.shape[0] != labels.shape[0]:
            raise ValueError(f"feature/label row mismatch: {feats.shape[0]} != {labels.shape[0]}")
        self.feats = feats
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else int(self.feats.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[i]) if self.indices is not None else i
        x = np.array(self.feats[row], dtype=np.float32, copy=True)
        y = np.array(self.labels[row], dtype=np.int64, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)
