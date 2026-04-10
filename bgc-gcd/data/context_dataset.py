"""
Phase 3 Context Dataset: returns K-window neighborhoods for center windows.

For Route A:
  - Loads pre-extracted 256-dim projections
  - Returns (K, 256) neighborhood + valid_mask + center label/phylum
  - Ambiguous windows included as context but excluded from loss
  - Padding uses zeros with valid_mask=0
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    """Dataset that returns K-window neighborhoods centered on sampled windows.

    For each center window, loads K//2 neighbors on each side from the same contig.
    Short contigs / contig edges are zero-padded with valid_mask=False.
    """

    def __init__(self, projections_path, meta_arrays_path, contig_index_path,
                 split="train", K=11, exclude_ambiguous_centers=True):
        """
        Args:
            projections_path: path to projections_256.npy (N, 256)
            meta_arrays_path: path to meta_arrays.pkl
            contig_index_path: path to contig_index.pkl
            split: train/val/test_id/test_ood
            K: context window size (odd number)
            exclude_ambiguous_centers: if True, only non-ambiguous windows as centers
        """
        # Load projections (memory-mapped for efficiency)
        self.projections = np.load(projections_path, mmap_mode="r")
        self.proj_dim = self.projections.shape[1]

        # Load metadata
        with open(meta_arrays_path, "rb") as f:
            meta = pickle.load(f)
        self.all_labels = meta["labels"]  # (N, 8) float32
        self.all_splits = meta["split"]
        self.all_label_strs = meta["label"]
        self.all_phyla = meta["phylum"]

        # Load contig index
        with open(contig_index_path, "rb") as f:
            self.contig_index = pickle.load(f)

        # Build reverse index: global_idx -> (contig_key, position_in_contig)
        self.window_to_contig = {}
        for key, indices in self.contig_index.items():
            for pos, global_idx in enumerate(indices):
                self.window_to_contig[global_idx] = (key, pos, len(indices))

        self.K = K
        self.half_K = K // 2

        # Build phylum encoding (consistent across splits)
        all_phyla_unique = sorted(set(str(p) for p in self.all_phyla))
        self.phylum_to_code = {p: i for i, p in enumerate(all_phyla_unique)}

        # Select center windows for this split
        split_mask = self.all_splits == split
        if exclude_ambiguous_centers:
            non_amb_mask = self.all_label_strs != "ambiguous"
            split_mask = split_mask & non_amb_mask

        self.center_indices = np.where(split_mask)[0]

        # For BalancedBatchSampler compatibility: expose labels of center windows
        self.labels = self.all_labels[self.center_indices]  # (n_centers, 8)

        # Build hard_neg indices relative to center_indices
        if "hard_negative" in meta:
            all_hn = meta["hard_negative"]
            self.hard_neg_mask = all_hn[self.center_indices].astype(bool)
        else:
            self.hard_neg_mask = np.zeros(len(self.center_indices), dtype=bool)

    def __len__(self):
        return len(self.center_indices)

    def _get_neighborhood(self, global_idx):
        """Get K-window neighborhood around global_idx.

        Returns:
            proj_neighborhood: (K, proj_dim) float32
            valid_mask: (K,) bool
            label_neighborhood: (K, 8) float32
        """
        contig_key, pos_in_contig, contig_len = self.window_to_contig[global_idx]
        contig_indices = self.contig_index[contig_key]

        proj_nbr = np.zeros((self.K, self.proj_dim), dtype=np.float32)
        valid = np.zeros(self.K, dtype=bool)
        label_nbr = np.zeros((self.K, 8), dtype=np.float32)

        for k_offset in range(-self.half_K, self.half_K + 1):
            pos = pos_in_contig + k_offset
            k_idx = k_offset + self.half_K  # position in the K-window

            if 0 <= pos < contig_len:
                g_idx = contig_indices[pos]
                proj_nbr[k_idx] = self.projections[g_idx]
                valid[k_idx] = True
                label_nbr[k_idx] = self.all_labels[g_idx]

        return proj_nbr, valid, label_nbr

    def __getitem__(self, idx):
        """
        Args:
            idx: index into center_indices (not global index)
        Returns:
            proj_nbr: (K, proj_dim) float32 tensor
            valid_mask: (K,) bool tensor
            center_label: (8,) float32 tensor
            center_phylum: int
        """
        global_idx = self.center_indices[idx]
        proj_nbr, valid_mask, _ = self._get_neighborhood(global_idx)

        center_label = self.all_labels[global_idx]
        center_phylum = self.phylum_to_code[str(self.all_phyla[global_idx])]

        return (
            torch.from_numpy(proj_nbr),
            torch.from_numpy(valid_mask),
            torch.from_numpy(center_label),
            int(center_phylum),
        )

    def get_contig_data(self, contig_key):
        """Get full contig data for inference.

        Returns:
            projections: (L, proj_dim) tensor
            valid_mask: (L,) bool tensor — all True for real contigs
            labels: (L, 8) tensor
            global_indices: list of global indices
        """
        indices = self.contig_index[contig_key]
        L = len(indices)
        proj = np.stack([self.projections[i] for i in indices])
        labels = np.stack([self.all_labels[i] for i in indices])
        valid = np.ones(L, dtype=bool)
        return (
            torch.from_numpy(proj),
            torch.from_numpy(valid),
            torch.from_numpy(labels),
            indices,
        )
