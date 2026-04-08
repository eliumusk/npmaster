import torch
import torch.nn as nn


class MultiLabelClassifier(nn.Module):
    """256 -> 8, independent per-dim sigmoid (BCEWithLogitsLoss expects logits)."""

    def __init__(self, in_dim: int = 256, n_labels: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_labels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # returns logits; caller applies sigmoid for probabilities
        return self.fc(z)
