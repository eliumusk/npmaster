"""
Phase 3 Context Model: Dilated 1D CNN with Gated Residual.

Architecture:
  Input: (B, K, 257) = 256-dim projection + 1-dim valid_window_mask
  Conv1d(257, 256, k=3, d=1) -> LayerNorm -> GELU -> Dropout
  Conv1d(256, 256, k=3, d=2) -> LayerNorm -> GELU -> Dropout
  Conv1d(256, 256, k=3, d=4) -> LayerNorm -> GELU -> Dropout
  Output: (B, K, 256) context-enhanced representation

  Receptive field: 1 + 2*(1+2+4) = 15 windows = 120kb

GatedResidual:
  alpha = sigmoid(beta), beta init = -4.0 -> alpha ~= 0.018
  z_final = z_proj + alpha * z_context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedContextCNN(nn.Module):
    """3-layer dilated 1D CNN for window-sequence context modeling."""

    def __init__(self, input_dim=257, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, dilation in enumerate([1, 2, 4]):
            in_c = input_dim if i == 0 else hidden_dim
            padding = dilation  # same padding for kernel=3
            self.layers.append(nn.ModuleDict({
                "conv": nn.Conv1d(in_c, hidden_dim, kernel_size=3,
                                  dilation=dilation, padding=padding),
                "norm": nn.LayerNorm(hidden_dim),
                "drop": nn.Dropout(dropout),
            }))

    def forward(self, x, valid_mask=None):
        """
        Args:
            x: (B, L, C_in) where C_in = 257 (256 proj + 1 mask channel)
            valid_mask: (B, L) bool, True for valid windows
        Returns:
            (B, L, 256) context features
        """
        # Conv1d expects (B, C, L)
        h = x.transpose(1, 2)

        for layer in self.layers:
            h = layer["conv"](h)
            # Back to (B, L, C) for LayerNorm
            h = h.transpose(1, 2)
            h = layer["norm"](h)
            h = F.gelu(h)
            h = layer["drop"](h)
            # Back to (B, C, L) for next conv
            h = h.transpose(1, 2)

        # Final output (B, L, C)
        h = h.transpose(1, 2)

        # Zero out padding positions
        if valid_mask is not None:
            h = h * valid_mask.unsqueeze(-1).float()

        return h


class GatedResidual(nn.Module):
    """Sigmoid-gated residual: z_final = z_proj + sigmoid(beta) * z_context.

    beta init = -4.0 -> alpha ~= 0.018, ensuring:
      - Initial context contribution is tiny (doesn't disrupt warm-started classifier)
      - Gradients always flow through context module (fixes alpha=0 zero-gradient bug)
      - alpha has natural upper bound of 1.0 (sigmoid saturation)
    """

    def __init__(self, init_beta=-4.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    @property
    def alpha(self):
        return torch.sigmoid(self.beta)

    def forward(self, z_proj, z_context):
        alpha = torch.sigmoid(self.beta)
        return z_proj + alpha * z_context


class ContextBGCModel(nn.Module):
    """Route A context model: operates on pre-extracted 256-dim projections.

    Input: (B, K, 256) projections + (B, K) valid_mask
    Output: (B, K, 8) logits per window position
    """

    def __init__(self, proj_dim=256, n_labels=8, cnn_dropout=0.1, init_beta=-4.0):
        super().__init__()
        # Context module: input is 256 + 1 (valid_mask channel) = 257
        self.context = DilatedContextCNN(
            input_dim=proj_dim + 1,
            hidden_dim=proj_dim,
            dropout=cnn_dropout,
        )
        self.gate = GatedResidual(init_beta=init_beta)
        self.classifier = nn.Linear(proj_dim, n_labels)

    def forward(self, z_proj, valid_mask):
        """
        Args:
            z_proj: (B, L, 256) pre-extracted L2-normalized projections
            valid_mask: (B, L) bool, True for real windows, False for padding
        Returns:
            z_final: (B, L, 256) context-enhanced representations
            logits: (B, L, 8) classification logits
        """
        # Concatenate valid_mask as channel
        mask_channel = valid_mask.unsqueeze(-1).float()  # (B, L, 1)
        ctx_input = torch.cat([z_proj, mask_channel], dim=-1)  # (B, L, 257)

        # Context module
        z_context = self.context(ctx_input, valid_mask)  # (B, L, 256)

        # Gated residual
        z_final = self.gate(z_proj, z_context)  # (B, L, 256)
        z_final = F.normalize(z_final, p=2, dim=-1)

        # Zero out padding
        z_final = z_final * valid_mask.unsqueeze(-1).float()

        # Per-window classification
        logits = self.classifier(z_final)  # (B, L, 8)

        return z_final, logits
