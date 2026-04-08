import torch
import torch.nn as nn


class DANNLoss(nn.Module):
    """Cross-entropy over phylum targets. GRL handles reversal upstream."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target.long())
