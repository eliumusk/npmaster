"""
Phase 3 Route B: End-to-end context model.

4096 -> ProjectionHead (warm start) -> DilatedContextCNN -> GatedResidual -> Classifier

ProjectionHead is trainable (with lower learning rate).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .projector import ProjectionHead
from .context_model import DilatedContextCNN, GatedResidual


class ContextFullModel(nn.Module):
    """End-to-end model: ProjectionHead + Context + Classifier.

    Input: (B, L, 4096) raw Evo2 embeddings
    Output: (B, L, 8) per-window logits
    """

    def __init__(self, input_dim=4096, proj_hidden=2048, proj_out=256,
                 proj_dropout=0.3, n_labels=8,
                 cnn_dropout=0.1, init_beta=-4.0):
        super().__init__()
        self.projector = ProjectionHead(input_dim, proj_hidden, proj_out, proj_dropout)
        self.context = DilatedContextCNN(
            input_dim=proj_out + 1,  # +1 for valid_mask channel
            hidden_dim=proj_out,
            dropout=cnn_dropout,
        )
        self.gate = GatedResidual(init_beta=init_beta)
        self.classifier = nn.Linear(proj_out, n_labels)
        self.proj_out = proj_out

    def forward(self, x, valid_mask):
        """
        Args:
            x: (B, L, 4096) raw embeddings
            valid_mask: (B, L) bool
        Returns:
            z_final: (B, L, proj_out) context-enhanced representations
            logits: (B, L, n_labels)
        """
        B, L, D = x.shape

        # Project each window independently
        # Reshape to (B*L, D), forward, reshape back
        x_flat = x.reshape(B * L, D)
        z_flat = self.projector(x_flat)  # (B*L, proj_out), L2-normalized
        z_proj = z_flat.reshape(B, L, self.proj_out)

        # Zero out padding positions in projection
        z_proj = z_proj * valid_mask.unsqueeze(-1).float()

        # Context module with mask channel
        mask_channel = valid_mask.unsqueeze(-1).float()
        ctx_input = torch.cat([z_proj, mask_channel], dim=-1)  # (B, L, proj_out+1)
        z_context = self.context(ctx_input, valid_mask)

        # Gated residual
        z_final = self.gate(z_proj, z_context)
        z_final = F.normalize(z_final, p=2, dim=-1)
        z_final = z_final * valid_mask.unsqueeze(-1).float()

        # Classify
        logits = self.classifier(z_final)

        return z_final, logits

    def param_groups(self, lr_proj=1e-4, lr_context=1e-3, lr_classifier=1e-3):
        """Return parameter groups with layered learning rates."""
        return [
            {"params": self.projector.parameters(), "lr": lr_proj},
            {"params": self.context.parameters(), "lr": lr_context},
            {"params": self.gate.parameters(), "lr": lr_context},
            {"params": self.classifier.parameters(), "lr": lr_classifier},
        ]
