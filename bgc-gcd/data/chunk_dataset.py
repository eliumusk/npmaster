"""
Phase 3 Route B: Chunk Dataset for end-to-end training.

Loads overlapping contig chunks of shape (L_chunk, 4096).
Full epoch traversal, no weighted sampling.
Loss mask = valid_window_mask & non-ambiguous & in center loss region.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    """Dataset returning contig chunks for Route B end-to-end training.

    Each chunk is (L_chunk, 4096) with valid_mask and loss_mask.
    Chunks are pre-computed with overlap to avoid boundary artifacts.
    """

    def __init__(self, embeddings_path, chunk_index_path, meta_arrays_path,
                 split="train"):
        # Load embeddings (memory-mapped)
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.input_dim = self.embeddings.shape[1]

        # Load chunk index
        with open(chunk_index_path, "rb") as f:
            chunk_data = pickle.load(f)
        self.chunks = chunk_data["chunks"][split]
        self.L_chunk = chunk_data["L_chunk"]

        # Load metadata for phylum info
        with open(meta_arrays_path, "rb") as f:
            meta = pickle.load(f)
        self.all_labels = meta["labels"]
        self.all_phyla = meta["phylum"]

        # Build phylum encoding
        all_phyla_unique = sorted(set(str(p) for p in self.all_phyla))
        self.phylum_to_code = {p: i for i, p in enumerate(all_phyla_unique)}

        # Compute class statistics for pos_weight
        n_pos = np.zeros(8)
        n_total = 0
        for chunk in self.chunks:
            for pos, has_loss in enumerate(chunk["loss_mask"]):
                if has_loss:
                    g_idx = chunk["global_indices"][pos]
                    n_pos += self.all_labels[g_idx]
                    n_total += 1
        self.n_loss_windows = n_total
        self.n_pos_per_class = n_pos
        print(f"  ChunkDataset: {len(self.chunks)} chunks, {n_total} loss windows, "
              f"pos/class: {n_pos.astype(int).tolist()}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        L = self.L_chunk
        D = self.input_dim

        emb = np.zeros((L, D), dtype=np.float32)
        labels = np.zeros((L, 8), dtype=np.float32)
        valid_mask = np.array(chunk["valid_mask"], dtype=bool)
        loss_mask = np.array(chunk["loss_mask"], dtype=bool)

        for pos, g_idx in enumerate(chunk["global_indices"]):
            if g_idx == -1:
                continue
            emb[pos] = self.embeddings[g_idx]
            labels[pos] = self.all_labels[g_idx]

        return (
            torch.from_numpy(emb),          # (L, 4096)
            torch.from_numpy(valid_mask),    # (L,) bool
            torch.from_numpy(loss_mask),     # (L,) bool
            torch.from_numpy(labels),        # (L, 8)
        )
