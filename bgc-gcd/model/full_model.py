import torch
import torch.nn as nn
from .projector import ProjectionHead
from .classifier import MultiLabelClassifier
from .dann import DomainClassifier


class BGCModel(nn.Module):
    def __init__(self, n_phyla: int, input_dim: int = 4096,
                 proj_hidden: int = 2048, proj_out: int = 256,
                 dropout: float = 0.3, dann_hidden: int = 128,
                 n_labels: int = 8):
        super().__init__()
        self.projector = ProjectionHead(input_dim, proj_hidden, proj_out, dropout)
        self.classifier = MultiLabelClassifier(proj_out, n_labels)
        self.dann = DomainClassifier(n_phyla, proj_out, dann_hidden)

    def set_dann_lambda(self, lambd: float):
        self.dann.set_lambda(lambd)

    def forward(self, x: torch.Tensor):
        z = self.projector(x)
        pred_logits = self.classifier(z)
        domain_logits = self.dann(z)
        return z, pred_logits, domain_logits
