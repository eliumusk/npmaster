"""Per-token BGC heads for 1D semantic segmentation.

All heads: (B, T, D_in) → (B, T) logits, variable-length T.

Two architectures:
  - PerTokenHead: shallow dilated Conv1D, ~50K params, RF ~120 bp.
  - UNetHead:     1D U-Net encoder-decoder with skip connections,
                  ~150K params, effective RF ~16 kb (covers typical BGCs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Original shallow head (baseline)
# ---------------------------------------------------------------------------

class PerTokenHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: int = 64,
        kernel: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = d_in
        for d in dilations:
            layers.append(nn.Conv1d(
                c_in, hidden, kernel_size=kernel,
                padding=d * (kernel // 2), dilation=d,
            ))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            c_in = hidden
        self.body = nn.Sequential(*layers)
        self.readout = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T) logits."""
        x = x.transpose(1, 2)          # (B, D, T)
        x = self.body(x)
        x = self.readout(x).squeeze(1)  # (B, T)
        return x


# ---------------------------------------------------------------------------
# 1D U-Net head
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(1, c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class UNetHead(nn.Module):
    """1D U-Net for per-token BGC segmentation.

    Encoder compresses by ``pool`` at each level; decoder upsamples and
    concatenates skip connections.  All-convolutional → handles any T.

    With defaults (depth=4, pool=4, hidden=64, K=8 bp/token):
      bottleneck token covers 4^4 * 8 = 2048 bp;
      k=5 at bottleneck → RF ≈ 5 * 2048 = 10 kb per token.
    """

    def __init__(
        self,
        d_in: int = 128,
        hidden: int = 64,
        depth: int = 4,
        pool: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.pool = pool
        self.factor = pool ** depth

        self.enc = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = d_in
        for i in range(depth):
            k = 7 if i == 0 else 5
            self.enc.append(_ConvBlock(c, hidden, k, dropout))
            self.pools.append(nn.AvgPool1d(pool))
            c = hidden

        self.bottleneck = _ConvBlock(hidden, hidden, 5, dropout)

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth):
            self.ups.append(nn.Upsample(scale_factor=pool, mode="nearest"))
            k = 7 if i == depth - 1 else 5
            self.dec.append(_ConvBlock(hidden * 2, hidden, k, dropout))

        self.readout = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T) logits."""
        B, T, D = x.shape
        x = x.transpose(1, 2)  # (B, D, T)

        # Pad T to multiple of pool^depth so every pool is exact.
        T_pad = ((T + self.factor - 1) // self.factor) * self.factor
        if T_pad > T:
            x = F.pad(x, (0, T_pad - T))

        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.enc, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[2] > s.shape[2]:
                x = x[:, :, :s.shape[2]]
            elif x.shape[2] < s.shape[2]:
                x = F.pad(x, (0, s.shape[2] - x.shape[2]))
            x = dec(torch.cat([x, s], dim=1))

        return self.readout(x).squeeze(1)[:, :T]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "pertoken": PerTokenHead,
    "unet": UNetHead,
}


def build_model(model_type: str, **kwargs) -> nn.Module:
    cls = MODEL_REGISTRY[model_type]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_bce_loss(
    logits: torch.Tensor,       # (B, T)
    labels: torch.Tensor,       # (B, T) values in {-1, 0, 1}
    pos_weight: float = 3.0,
) -> torch.Tensor:
    mask = (labels >= 0).float()
    y = labels.clamp(min=0).float()
    pw = torch.tensor(pos_weight, device=logits.device)
    loss_el = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pw,
    )
    denom = mask.sum().clamp(min=1.0)
    return (loss_el * mask).sum() / denom


def dice_bce_loss(
    logits: torch.Tensor,       # (B, T)
    labels: torch.Tensor,       # (B, T) values in {-1, 0, 1}
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """Dice + BCE combo.  Dice handles class imbalance and encourages
    contiguous predictions; BCE stabilises per-token calibration."""
    mask = (labels >= 0).float()
    y = labels.clamp(min=0).float()

    p = torch.sigmoid(logits) * mask
    ym = y * mask
    inter = (p * ym).sum()
    union = p.sum() + ym.sum()
    dice = 1.0 - 2.0 * inter / (union + 1e-6)

    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1.0)

    return dice + bce_weight * bce


def focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Masked focal BCE for dense labels."""
    mask = (labels >= 0).float()
    y = labels.clamp(min=0).float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(y == 1, p, 1.0 - p)
    alpha_t = torch.where(y == 1, torch.full_like(y, alpha), torch.full_like(y, 1.0 - alpha))
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def tversky_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """Tversky + BCE, weighting false negatives more than false positives by default."""
    mask = (labels >= 0).float()
    y = labels.clamp(min=0).float()
    p = torch.sigmoid(logits) * mask
    ym = y * mask
    tp = (p * ym).sum()
    fp = (p * (1.0 - ym) * mask).sum()
    fn = ((1.0 - p) * ym).sum()
    tversky = 1.0 - tp / (tp + alpha * fp + beta * fn + 1e-6)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1.0)
    return tversky + bce_weight * bce


def weak_negative_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    neg_weight: float = 0.2,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """PU-style weak-negative BCE.

    antiSMASH positives stay fully positive; unlabeled/background tokens are
    treated as weak negatives rather than certain negatives.
    """
    mask = (labels >= 0).float()
    y = labels.clamp(min=0).float()
    weights = torch.where(y == 1, torch.full_like(y, pos_weight), torch.full_like(y, neg_weight))
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    denom = (weights * mask).sum().clamp(min=1.0)
    return (loss * weights * mask).sum() / denom


LOSS_REGISTRY: dict[str, callable] = {
    "bce": masked_bce_loss,
    "dice_bce": dice_bce_loss,
    "focal_bce": focal_bce_loss,
    "tversky_bce": tversky_bce_loss,
    "weak_negative_bce": weak_negative_bce_loss,
}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
