import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelBCE(nn.Module):
    """BCE-with-logits for multi-label classification.

    Optional per-class pos_weight to mitigate class imbalance.
    weight[i] = N_total / (n_classes * N_positive_for_class_i)
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.register_buffer(
            "pos_weight", pos_weight.float() if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight
        )

    @staticmethod
    def compute_pos_weight(labels: "np.ndarray", eps: float = 1.0) -> torch.Tensor:
        """labels: (N, C) binary numpy array."""
        import numpy as np
        n_total, n_classes = labels.shape
        pos = labels.sum(axis=0).astype("float64")
        neg = n_total - pos
        # classic pos_weight for BCEWithLogitsLoss is neg/pos
        pos = np.maximum(pos, eps)
        return torch.tensor(neg / pos, dtype=torch.float32)
