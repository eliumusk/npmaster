import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """4096 -> 2048 -> 256, L2-normalized output."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 2048,
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.drop(h)
        z = self.fc2(h)
        z = F.normalize(z, p=2, dim=1)
        return z
