#!/usr/bin/env python3
"""Train the per-token dense BGC head on cached per-token Evo2 features.

Input:
  --features-dir   directory of `(window_id).npy` (each shape (T, D_reduced), fp16)
                   OR a single merged .npz / .npy memmap
  --labels-csv     same CSV as mainline training
                   (data/windows/edge_padding/windows_metadata_edgepad_filtered.csv)
  --bgc-annot      data/bgc_annotations.csv
  --splits         which splits to use for train vs val (default train/val)

Outputs:
  runs/<run_name>/best.pt         checkpoint with lowest val loss
  runs/<run_name>/train_log.jsonl
  runs/<run_name>/config.json

Deliberately kept minimal. No WandB. No multi-node DDP. Single GPU per run;
launch N independent runs across GPUs/nodes if sweeping.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from per_token.labels import (load_bgc_annotations, load_windows,  # noqa: E402
                              per_nt_labels)
from per_token.model import PerTokenHead, count_params, masked_bce_loss  # noqa: E402


class PerTokenWindowDataset(Dataset):
    """Lazy per-window loader. Features live as per-window .npy files."""

    def __init__(self, windows, labels_dense, features_dir: Path, d_reduced: int):
        assert len(windows) == labels_dense.shape[0]
        self.windows = windows
        self.labels = labels_dense
        self.features_dir = Path(features_dir)
        self.d_reduced = d_reduced

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        feat_path = self.features_dir / f"{w.window_id}.npy"
        x = np.load(feat_path).astype(np.float32)   # (T, D)
        y = self.labels[i].astype(np.int64)          # (T,)
        if x.shape[1] != self.d_reduced:
            raise ValueError(f"{w.window_id}: expected D={self.d_reduced}, got {x.shape[1]}")
        # Pad/truncate to fixed T if needed (labels define T).
        T = y.shape[0]
        if x.shape[0] < T:
            x = np.concatenate([x, np.zeros((T - x.shape[0], x.shape[1]), dtype=np.float32)], axis=0)
        elif x.shape[0] > T:
            x = x[:T]
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--labels-csv", required=True, type=Path)
    p.add_argument("--bgc-annot", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--d-reduced", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--pos-weight", type=float, default=3.0)
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["val"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsample-train", type=int, default=0,
                   help="If >0, sample this many train windows (for quick smoke runs).")
    return p.parse_args()


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("loading labels...")
    all_windows = load_windows(args.labels_csv)
    bgc_map = load_bgc_annotations(args.bgc_annot)

    train_w = [w for w in all_windows if w.split in args.train_splits]
    val_w = [w for w in all_windows if w.split in args.val_splits]
    if args.subsample_train and args.subsample_train < len(train_w):
        rnd = random.Random(args.seed); rnd.shuffle(train_w)
        train_w = train_w[:args.subsample_train]
    print(f"train: {len(train_w)}  val: {len(val_w)}")

    print("building dense labels (train)...")
    y_train = per_nt_labels(train_w, bgc_map)
    print("building dense labels (val)...")
    y_val = per_nt_labels(val_w, bgc_map)

    train_ds = PerTokenWindowDataset(train_w, y_train, args.features_dir, args.d_reduced)
    val_ds = PerTokenWindowDataset(val_w, y_val, args.features_dir, args.d_reduced)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda")
    model = PerTokenHead(
        d_in=args.d_reduced, hidden=args.hidden, dilations=tuple(args.dilations),
    ).to(device)
    print("params:", count_params(model))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    log_path = args.out_dir / "train_log.jsonl"

    with open(args.out_dir / "config.json", "w") as fh:
        json.dump({**vars(args), "param_count": count_params(model)}, fh, indent=2, default=str)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tr_loss = 0.0; tr_n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = masked_bce_loss(logits, y, pos_weight=args.pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.shape[0]; tr_n += x.shape[0]

        model.eval()
        va_loss = 0.0; va_n = 0
        va_pos_preds = 0; va_pos_true = 0; va_tp = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                va_loss += masked_bce_loss(logits, y, pos_weight=args.pos_weight).item() * x.shape[0]
                va_n += x.shape[0]
                preds = (torch.sigmoid(logits) > 0.5).int()
                mask = (y >= 0)
                tp = ((preds == 1) & (y == 1) & mask).sum().item()
                va_tp += tp
                va_pos_preds += ((preds == 1) & mask).sum().item()
                va_pos_true += ((y == 1) & mask).sum().item()

        tr_loss /= max(1, tr_n); va_loss /= max(1, va_n)
        prec = va_tp / max(1, va_pos_preds); rec = va_tp / max(1, va_pos_true)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        elapsed = time.time() - t0
        rec_line = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                    "val_token_precision": prec, "val_token_recall": rec,
                    "val_token_f1": f1, "elapsed_s": elapsed}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec_line) + "\n")
        print(rec_line)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_loss": va_loss, "epoch": epoch},
                       args.out_dir / "best.pt")

    print("done.")


if __name__ == "__main__":
    main()
