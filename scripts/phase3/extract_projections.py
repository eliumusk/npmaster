#!/usr/bin/env python3
"""
P3-T0: Extract 256-dim projections from H_hardneg_r10 ProjectionHead
and build contig index sorted by genomic coordinate.

Outputs:
  projections_256.npy   (1064568, 256) float32 ~1GB
  contig_index.pkl      {(genome_id, contig_id): [array indices sorted by window_start]}
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add bgc-gcd to path
BGC_GCD = str(Path(__file__).resolve().parents[2] / "bgc-gcd")
if BGC_GCD not in sys.path:
    sys.path.insert(0, BGC_GCD)

from model.projector import ProjectionHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="H_hardneg_r10 checkpoint path")
    p.add_argument("--emb", required=True, help="embeddings_merged.npy path")
    p.add_argument("--meta", required=True, help="windows_metadata_hardneg.csv path")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # --- Load ProjectionHead from checkpoint ---
    print("Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    proj = ProjectionHead(
        input_dim=cfg["data"].get("input_dim", 4096),
        hidden_dim=cfg["model"]["proj_hidden"],
        out_dim=cfg["model"]["proj_out"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Extract projector state dict
    proj_state = {}
    for k, v in ckpt["model"].items():
        if k.startswith("projector."):
            proj_state[k.replace("projector.", "")] = v
    proj.load_state_dict(proj_state)
    proj.eval()
    print(f"  ProjectionHead loaded: {cfg['data'].get('input_dim', 4096)} -> {cfg['model']['proj_out']}")

    # --- Load embeddings ---
    print(f"Loading embeddings from {args.emb}...")
    emb = np.load(args.emb)
    N = emb.shape[0]
    print(f"  Shape: {emb.shape} ({emb.nbytes / 1e9:.1f} GB)")

    # --- Forward pass in batches ---
    print("Extracting projections...")
    projections = np.empty((N, cfg["model"]["proj_out"]), dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(emb).float())
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    with torch.no_grad():
        offset = 0
        for batch_idx, (x,) in enumerate(loader):
            z = proj(x.to(device))
            bs = z.size(0)
            projections[offset:offset + bs] = z.cpu().numpy()
            offset += bs
            if (batch_idx + 1) % 100 == 0:
                print(f"  {offset}/{N} ({offset / N * 100:.1f}%)")

    assert offset == N, f"Expected {N}, got {offset}"

    # Save projections
    proj_path = out_dir / "projections_256.npy"
    np.save(proj_path, projections)
    print(f"  Saved {proj_path} shape={projections.shape} ({projections.nbytes / 1e9:.2f} GB)")

    # --- Verify: spot-check 10 random windows ---
    print("Spot-checking 10 random windows...")
    rng = np.random.default_rng(42)
    check_idx = rng.choice(N, size=10, replace=False)
    with torch.no_grad():
        x_check = torch.from_numpy(emb[check_idx]).float().to(device)
        z_check = proj(x_check).cpu().numpy()
    max_err = np.abs(projections[check_idx] - z_check).max()
    print(f"  Max error: {max_err:.2e} (should be < 1e-5)")
    assert max_err < 1e-4, f"Spot check failed: max_err={max_err}"

    # --- Build contig index ---
    print(f"Building contig index from {args.meta}...")
    df = pd.read_csv(args.meta)
    assert len(df) == N, f"Metadata rows {len(df)} != embeddings {N}"

    contig_index = defaultdict(list)
    for idx, row in df.iterrows():
        key = (row["genome_id"], row["contig_id"])
        contig_index[key].append((int(row["window_start"]), idx))

    # Sort each contig by window_start (genomic coordinate)
    for key in contig_index:
        contig_index[key].sort(key=lambda x: x[0])
        # Store only the array indices (sorted by position)
        contig_index[key] = [idx for _, idx in contig_index[key]]

    contig_index = dict(contig_index)
    n_contigs = len(contig_index)
    n_windows_check = sum(len(v) for v in contig_index.values())
    assert n_windows_check == N, f"Contig index windows {n_windows_check} != {N}"

    idx_path = out_dir / "contig_index.pkl"
    with open(idx_path, "wb") as f:
        pickle.dump(contig_index, f)
    print(f"  Saved {idx_path}: {n_contigs} contigs, {n_windows_check} windows")

    # Also save metadata split/label info for quick lookup during training
    meta_arrays = {
        "split": df["split"].values,
        "label": df["label"].values,
        "genome_id": df["genome_id"].values,
        "contig_id": df["contig_id"].values,
        "window_start": df["window_start"].values.astype(np.int64),
        "phylum": df["phylum"].values,
        "edge_padded": df["edge_padded"].values.astype(np.int32) if "edge_padded" in df.columns else np.zeros(N, dtype=np.int32),
    }
    # Save label columns
    type_cols = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other", "is_negative"]
    meta_arrays["labels"] = df[type_cols].values.astype(np.float32)

    if "hard_negative" in df.columns:
        meta_arrays["hard_negative"] = df["hard_negative"].values.astype(np.int32)

    meta_path = out_dir / "meta_arrays.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta_arrays, f)
    print(f"  Saved {meta_path}")

    print("\nDone! Summary:")
    print(f"  projections: {proj_path}")
    print(f"  contig_index: {idx_path} ({n_contigs} contigs)")
    print(f"  meta_arrays: {meta_path}")


if __name__ == "__main__":
    main()
