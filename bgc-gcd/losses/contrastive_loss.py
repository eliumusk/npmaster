import torch
import torch.nn as nn


class MultiLabelSupCon(nn.Module):
    """Multi-label-aware supervised contrastive loss.

    Positive pair (i, j): i != j AND labels_i and labels_j share at least
    one of the 7 type bits (any common type label). Both-negative windows
    (is_negative=1 for both) are skipped — they don't carry type info.

    Negatives for the denominator: all other samples in the batch (i != j)
    that are NOT positive partners of i.

    Optional: cross-phylum boost. When a positive pair spans two phyla,
    its positive-weight is multiplied by `cross_phylum_weight`.

    Inputs:
        z:      (B, D) L2-normalized projections
        labels: (B, 8) multi-label; last column is is_negative
        phyla:  (B,) int64 phylum codes
    """

    def __init__(self, tau: float = 0.07, cross_phylum_weight: float = 1.5):
        super().__init__()
        self.tau = tau
        self.cross_phylum_weight = cross_phylum_weight

    def forward(self, z, labels, phyla):
        device = z.device
        B = z.size(0)

        # Cosine similarity. z is already L2-normalized, so dot product is cosine.
        sim = z @ z.t() / self.tau  # (B, B)

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # ---- Mask construction ----
        type_labels = labels[:, :7].float()  # (B, 7)
        is_neg = labels[:, 7].bool()  # (B,)

        # Shared type: (B, B) bool — True if pair shares >=1 type bit.
        shared_type = (type_labels @ type_labels.t()) > 0  # (B, B)

        # Exclude self
        eye = torch.eye(B, dtype=torch.bool, device=device)
        shared_type = shared_type & ~eye

        # Skip pairs where BOTH are negative windows (no type info either side).
        # A shared-type pair among type_labels==0 for both is impossible by
        # construction (0 @ 0 = 0), so shared_type already excludes them.
        # Positive pair = shared_type.
        pos_mask = shared_type.float()  # (B, B)

        # Cross-phylum weighting
        if self.cross_phylum_weight != 1.0:
            diff_phylum = (phyla.unsqueeze(0) != phyla.unsqueeze(1)).float()
            pos_weight = 1.0 + (self.cross_phylum_weight - 1.0) * diff_phylum
            pos_mask = pos_mask * pos_weight

        # Valid anchors: at least one positive pair
        has_pos = pos_mask.sum(dim=1) > 0  # (B,)
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Logits denominator: exclude self, include both-negative pairs as
        # valid "negatives" for the contrastive denominator (they give the
        # anchor something to push away from).
        logits_mask = (~eye).float()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives (weighted)
        pos_sum = pos_mask.sum(dim=1)  # (B,)
        pos_sum_safe = torch.where(
            pos_sum > 0, pos_sum, torch.ones_like(pos_sum)
        )
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_sum_safe

        loss = -mean_log_prob_pos[has_pos].mean()
        return loss

    @torch.no_grad()
    def pair_stats(self, labels):
        """Diagnostic: returns pos-pair statistics for a batch.

        Returns dict with:
          n_pos_pairs, n_total_pairs, frac_pos, type_combo_counts
        """
        B = labels.size(0)
        type_labels = labels[:, :7].float()
        shared_type = (type_labels @ type_labels.t()) > 0
        eye = torch.eye(B, dtype=torch.bool, device=labels.device)
        pos_mask = shared_type & ~eye

        n_pos = pos_mask.sum().item() // 2  # symmetric
        n_total = B * (B - 1) // 2

        # Type combo counts: for each positive pair, which bits do they share?
        combos = {}
        type_names = ["Polyketide", "NRP", "Terpene", "RiPP",
                      "Saccharide", "Alkaloid", "Other"]
        # pair-wise shared bits
        # restrict to upper triangle
        iu, ju = torch.triu_indices(B, B, offset=1)
        pmask = pos_mask[iu, ju]
        shared_bits = (
            type_labels[iu].bool() & type_labels[ju].bool()
        )  # (P, 7)
        for k in range(pmask.size(0)):
            if not pmask[k]:
                continue
            bits = shared_bits[k].nonzero(as_tuple=True)[0].tolist()
            key = "+".join(type_names[b] for b in bits) or "none"
            combos[key] = combos.get(key, 0) + 1

        return {
            "n_pos_pairs": n_pos,
            "n_total_pairs": n_total,
            "frac_pos": n_pos / max(n_total, 1),
            "type_combo_counts": combos,
        }
