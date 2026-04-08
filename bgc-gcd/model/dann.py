import torch
import torch.nn as nn
from torch.autograd import Function


class _GRLFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def set_lambda(self, lambd: float):
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFn.apply(x, self.lambd)


class DomainClassifier(nn.Module):
    """GRL -> Linear(256,128) -> ReLU -> Linear(128, n_phyla).

    Returns logits. Apply softmax/cross-entropy outside.
    """

    def __init__(self, n_phyla: int, in_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.grl = GradientReversalLayer(lambd=0.0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_phyla),
        )

    def set_lambda(self, lambd: float):
        self.grl.set_lambda(lambd)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_rev = self.grl(z)
        return self.net(z_rev)
