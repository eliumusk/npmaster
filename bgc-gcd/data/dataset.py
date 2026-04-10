from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TYPE_COLS = [
    "Polyketide", "NRP", "Terpene", "RiPP",
    "Saccharide", "Alkaloid", "Other", "is_negative",
]


class BGCDataset(Dataset):
    def __init__(self, embeddings_path, labels_path, split: str = "train",
                 phylum_to_code: dict | None = None, nll_path: str | None = None):
        emb = np.load(embeddings_path)
        df = pd.read_csv(labels_path)

        # Optionally concatenate NLL features
        if nll_path is not None and Path(nll_path).exists():
            nll = np.load(nll_path)
            emb = np.concatenate([emb, nll], axis=1)  # (N, 4104)

        mask = (df["split"].values == split)
        # Exclude ambiguous samples (label==[0]*8, neither positive nor negative)
        if "label" in df.columns:
            mask = mask & (df["label"].values != "ambiguous")
        self.embeddings = emb[mask].astype(np.float32)
        self.df = df[mask].reset_index(drop=True)

        self.labels = self.df[TYPE_COLS].values.astype(np.float32)  # (N, 8)

        # Consistent phylum encoding across splits.
        if phylum_to_code is None:
            phyla_sorted = sorted(df["phylum"].astype(str).unique().tolist())
            phylum_to_code = {p: i for i, p in enumerate(phyla_sorted)}
        self.phylum_to_code = phylum_to_code
        self.phyla = np.array(
            [phylum_to_code[str(p)] for p in self.df["phylum"].values],
            dtype=np.int64,
        )

        self.window_ids = self.df["window_id"].astype(str).values

    @property
    def n_phyla(self) -> int:
        return len(self.phylum_to_code)

    @property
    def input_dim(self) -> int:
        return self.embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.from_numpy(self.labels[idx]),
            int(self.phyla[idx]),
        )


def build_phylum_map(labels_path) -> dict:
    df = pd.read_csv(labels_path)
    if "label" in df.columns:
        df = df[df["label"] != "ambiguous"]
    phyla_sorted = sorted(df["phylum"].astype(str).unique().tolist())
    return {p: i for i, p in enumerate(phyla_sorted)}
