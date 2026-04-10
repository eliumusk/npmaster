import torch
import torch.nn.functional as F


class EmbeddingAugmentation:
    """Two-view augmentation on (B, D) embedding tensors.

    Mode per view is sampled independently:
      - 50%: add Gaussian noise N(0, noise_std)
      - 50%: random dim dropout (rate=dropout_rate), then re-L2-normalize

    NOTE: raw Evo2 embeddings are not L2-normalized. We still re-L2-normalize
    after the dropout branch to preserve scale within that branch; Gaussian
    branch leaves scale alone.
    """

    def __init__(self, noise_std: float = 0.05, dropout_rate: float = 0.1,
                 enabled: bool = True):
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.enabled = enabled

    def _one_view(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        B = x.size(0)
        # per-sample mode choice: 1 = noise, 0 = dropout
        mode = torch.randint(0, 2, (B, 1), device=x.device).float()
        # noise view
        noise = torch.randn_like(x) * self.noise_std
        noise_view = x + noise
        # dropout view
        mask = (torch.rand_like(x) > self.dropout_rate).float()
        drop_view = x * mask
        # renormalize drop_view to preserve original per-sample L2 norm
        orig_norm = x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        new_norm = drop_view.norm(dim=1, keepdim=True).clamp_min(1e-8)
        drop_view = drop_view * (orig_norm / new_norm)
        return mode * noise_view + (1 - mode) * drop_view

    def __call__(self, x: torch.Tensor):
        return self._one_view(x), self._one_view(x)
