#!/usr/bin/env python3
"""One-time data preparation: load shards + build labels → save to /dev/shm.

All subsequent training processes mmap the same files = zero extra RAM.

Usage:
  python scripts/train/per_token/prepare_data.py \
      --features-dir experiments/step2_per_token/train_w16384_s8192 \
      --labels-csv   data/windows/edge_padding/windows_metadata_edgepad_filtered.csv \
      --bgc-annot    data/bgc_annotations.csv \
      --out-dir      /dev/shm/per_token_data

Output:
  train_feats.npy   (N_train, T, D) fp16
  train_labels.npy  (N_train, T) int8
  val_feats.npy     (N_val, T, D) fp16
  val_labels.npy    (N_val, T) int8
  meta.json         {n_train, n_val, t_latent, d_in, downsample_k}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from per_token.labels import Window, load_bgc_annotations, load_windows, per_nt_labels


def scan_shards(features_dir: Path) -> dict[str, tuple[Path, int]]:
    import csv
    index: dict[str, tuple[Path, int]] = {}
    csvs = sorted(features_dir.glob("*.part*_windows.csv"))
    for csv_path in csvs:
        npy_path = csv_path.with_name(csv_path.name.replace("_windows.csv", ".npy"))
        if not npy_path.exists():
            continue
        with open(csv_path) as fh:
            for row_idx, row in enumerate(csv.DictReader(fh)):
                wid = row["window_id"]
                if wid not in index:
                    index[wid] = (npy_path, row_idx)
    return index


def downsample_labels(y_raw: np.ndarray, K: int) -> np.ndarray:
    N, T_raw = y_raw.shape
    T_latent = T_raw // K
    y_block = y_raw[:, :T_latent * K].reshape(N, T_latent, K)
    has_mask = (y_block == -1).any(axis=2)
    has_pos = (y_block == 1).any(axis=2)
    out = np.zeros((N, T_latent), dtype=np.int8)
    out[has_pos] = 1
    out[has_mask] = -1
    return out


def preload_paired(
    train_w, val_w, shard_index, d_in, t_latent,
):
    by_shard: dict[Path, tuple[list, list]] = {}
    for i, w in enumerate(train_w):
        entry = shard_index.get(w.window_id)
        if entry is None:
            continue
        by_shard.setdefault(entry[0], ([], []))[0].append((i, entry[1]))
    for i, w in enumerate(val_w):
        entry = shard_index.get(w.window_id)
        if entry is None:
            continue
        by_shard.setdefault(entry[0], ([], []))[1].append((i, entry[1]))

    n_train = sum(len(t[0]) for t in by_shard.values())
    n_val = sum(len(t[1]) for t in by_shard.values())

    train_feats = np.empty((n_train, t_latent, d_in), dtype=np.float16)
    val_feats = np.empty((n_val, t_latent, d_in), dtype=np.float16)
    train_keep = [-1] * n_train
    val_keep = [-1] * n_val
    w_tr = 0; w_va = 0
    n_shards = len(by_shard)
    t0 = time.time()
    for k, (shard_path, (tr_items, va_items)) in enumerate(sorted(by_shard.items())):
        arr = np.load(shard_path)
        if tr_items:
            tr_items.sort(key=lambda t: t[1])
            rows = [r for (_, r) in tr_items]
            block = arr[rows, :t_latent, :].astype(np.float16, copy=False)
            train_feats[w_tr:w_tr + len(rows)] = block
            for (win_i, _), wr in zip(tr_items, range(w_tr, w_tr + len(rows))):
                train_keep[wr] = win_i
            w_tr += len(rows)
        if va_items:
            va_items.sort(key=lambda t: t[1])
            rows = [r for (_, r) in va_items]
            block = arr[rows, :t_latent, :].astype(np.float16, copy=False)
            val_feats[w_va:w_va + len(rows)] = block
            for (win_i, _), wr in zip(va_items, range(w_va, w_va + len(rows))):
                val_keep[wr] = win_i
            w_va += len(rows)
        del arr
        if (k + 1) % 50 == 0 or (k + 1) == n_shards:
            elapsed = time.time() - t0
            print(f"  preload [{k+1}/{n_shards}] shards  "
                  f"train={w_tr}/{n_train}  val={w_va}/{n_val}  "
                  f"{elapsed:.0f}s")
    return train_feats, val_feats, train_keep, val_keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--labels-csv", required=True, type=Path)
    p.add_argument("--bgc-annot", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--d-reduced", type=int, default=128)
    p.add_argument("--downsample-k", type=int, default=8)
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["val"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("scanning shards...")
    shard_index = scan_shards(args.features_dir)
    print(f"  {len(shard_index)} windows across "
          f"{len(set(v[0] for v in shard_index.values()))} npy files")

    print("loading canonical labels csv...")
    all_windows = load_windows(args.labels_csv)
    window_by_id = {w.window_id: w for w in all_windows}
    bgc_map = load_bgc_annotations(args.bgc_annot)

    train_w, val_w = [], []
    for wid in shard_index:
        w = window_by_id.get(wid)
        if w is None:
            continue
        if w.split in args.train_splits:
            train_w.append(w)
        elif w.split in args.val_splits:
            val_w.append(w)
    print(f"train: {len(train_w)}  val: {len(val_w)}")

    t_raw = args.downsample_k * 2048
    train_w = [replace(w, end=w.start + t_raw) if (w.end - w.start) != t_raw else w for w in train_w]
    val_w = [replace(w, end=w.start + t_raw) if (w.end - w.start) != t_raw else w for w in val_w]

    print("building raw dense labels...")
    y_train_raw = per_nt_labels(train_w, bgc_map)
    y_val_raw = per_nt_labels(val_w, bgc_map)
    print(f"  train: {y_train_raw.shape}  val: {y_val_raw.shape}")

    print(f"downsampling labels by K={args.downsample_k}...")
    y_train = downsample_labels(y_train_raw, args.downsample_k)
    y_val = downsample_labels(y_val_raw, args.downsample_k)
    del y_train_raw, y_val_raw
    t_latent = y_train.shape[1]
    pos_rate = (y_train == 1).sum() / max(1, (y_train >= 0).sum())
    print(f"  train: {y_train.shape} pos_rate={pos_rate:.4f}")
    print(f"  val: {y_val.shape}")

    print("preloading features (one pass over all shards)...")
    train_feats, val_feats, train_keep, val_keep = preload_paired(
        train_w, val_w, shard_index, args.d_reduced, t_latent,
    )
    train_labels = y_train[train_keep]
    val_labels = y_val[val_keep]

    print(f"saving to {args.out_dir}...")
    np.save(args.out_dir / "train_feats.npy", train_feats)
    np.save(args.out_dir / "train_labels.npy", train_labels)
    np.save(args.out_dir / "val_feats.npy", val_feats)
    np.save(args.out_dir / "val_labels.npy", val_labels)
    meta = {
        "n_train": int(train_feats.shape[0]),
        "n_val": int(val_feats.shape[0]),
        "t_latent": int(t_latent),
        "d_in": int(args.d_reduced),
        "downsample_k": int(args.downsample_k),
        "pos_rate": float(pos_rate),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_gb = (train_feats.nbytes + val_feats.nbytes + train_labels.nbytes + val_labels.nbytes) / 1e9
    print(f"done: {total_gb:.1f} GB written. "
          f"train={train_feats.shape}, val={val_feats.shape}")


if __name__ == "__main__":
    main()
