"""
Phase 3 Route A: Train context model on pre-extracted 256-dim projections.

Center-only BCE loss. BalancedBatchSampler (same as mainline).
Classifier warm-started from H_hardneg_r10.
Checkpoint selection by val region F1 (every 5 epochs).
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.context_dataset import ContextDataset
from data.sampler import BalancedBatchSampler
from model.context_model import ContextBGCModel
from losses.classification_loss import MultiLabelBCE
from evaluate_context import evaluate_context_model, evaluate_region_level
from utils import set_seed, save_json


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"route_a_s{args.seed}"

    # --- Data ---
    print(f"[{tag}] Loading data...")
    train_ds = ContextDataset(
        args.projections, args.meta_arrays, args.contig_index,
        split="train", K=args.K,
    )
    val_ds = ContextDataset(
        args.projections, args.meta_arrays, args.contig_index,
        split="val", K=args.K,
    )
    print(f"[{tag}] train={len(train_ds)} val={len(val_ds)} K={args.K} proj_dim={train_ds.proj_dim}")

    # Hard-negative indices (relative to center_indices)
    hard_neg_indices = np.where(train_ds.hard_neg_mask)[0] if train_ds.hard_neg_mask.any() else None
    if hard_neg_indices is not None:
        print(f"[{tag}] Hard negatives: {len(hard_neg_indices)} flagged, ratio={args.hard_neg_ratio}")

    sampler = BalancedBatchSampler(
        train_ds.labels, batch_size=args.batch_size, seed=args.seed,
        pos_ratio=args.pos_ratio,
        hard_neg_indices=hard_neg_indices,
        hard_neg_ratio=args.hard_neg_ratio,
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    print(f"[{tag}] pos_ratio={args.pos_ratio} n_pos={sampler.n_pos} n_neg={sampler.n_neg}")

    # --- Model ---
    model = ContextBGCModel(
        proj_dim=train_ds.proj_dim,
        n_labels=8,
        cnn_dropout=args.cnn_dropout,
        init_beta=args.init_beta,
    ).to(device)

    # Warm-start classifier from mainline checkpoint
    if args.ckpt:
        print(f"[{tag}] Warm-starting classifier from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cls_state = {}
        for k, v in ckpt["model"].items():
            if k.startswith("classifier."):
                # Mainline classifier is MultiLabelClassifier with .fc
                # Our model.classifier is nn.Linear directly
                new_key = k.replace("classifier.fc.", "").replace("classifier.", "")
                cls_state[new_key] = v
        model.classifier.load_state_dict(cls_state)
        print(f"[{tag}] Classifier loaded: {list(cls_state.keys())}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] Model params: {n_params:,}")

    # --- Loss ---
    pos_weight = MultiLabelBCE.compute_pos_weight(train_ds.labels).to(device).clamp(max=20.0)
    print(f"[{tag}] pos_weight: {pos_weight.tolist()}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training log ---
    log_csv = out_dir / f"train_log_{tag}.csv"
    fieldnames = ["epoch", "L_bce", "val_bgc_auroc", "val_region_f1", "alpha", "lr", "sec"]
    with open(log_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    center_idx = args.K // 2  # center position in K-window

    best_region_f1 = -1.0
    best_auroc = -1.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for proj_nbr, valid_mask, center_label, center_phylum in train_loader:
            proj_nbr = proj_nbr.to(device, non_blocking=True)       # (B, K, 256)
            valid_mask = valid_mask.to(device, non_blocking=True)    # (B, K)
            center_label = center_label.to(device, non_blocking=True)  # (B, 8)

            z_final, logits = model(proj_nbr, valid_mask)

            # Center-only loss (Rule 4)
            center_logits = logits[:, center_idx, :]  # (B, 8)
            loss = loss_fn(center_logits, center_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        scheduler.step()
        dt = time.time() - t0
        avg_loss = total_loss / max(n_steps, 1)
        alpha_val = model.gate.alpha.item()
        lr_now = optimizer.param_groups[0]["lr"]

        # --- Validation ---
        val_metrics = evaluate_context_model(model, val_ds, device, K=args.K)
        val_auroc = val_metrics["bgc_binary_auroc"]

        # Region-level eval every 5 epochs (Rule 8)
        val_region_f1 = float("nan")
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            region_metrics = evaluate_region_level(
                model, val_ds, device,
                bgc_csv_path=args.bgc_csv,
                split="val",
            )
            val_region_f1 = region_metrics.get("region_f1", float("nan"))

            # Checkpoint by region F1
            if not np.isnan(val_region_f1) and val_region_f1 > best_region_f1:
                best_region_f1 = val_region_f1
                ckpt_data = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_region_f1": val_region_f1,
                    "val_bgc_auroc": val_auroc,
                    "alpha": alpha_val,
                }
                torch.save(ckpt_data, out_dir / f"best_{tag}.pt")
                print(f"[{tag} ep{epoch:03d}] * new best region_f1={val_region_f1:.4f}")

        # Also track best AUROC as fallback
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            ckpt_data = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_bgc_auroc": val_auroc,
                "alpha": alpha_val,
            }
            torch.save(ckpt_data, out_dir / f"best_auroc_{tag}.pt")

        # Log
        row = {
            "epoch": epoch, "L_bce": round(avg_loss, 6),
            "val_bgc_auroc": round(val_auroc, 4),
            "val_region_f1": round(val_region_f1, 4) if not np.isnan(val_region_f1) else "",
            "alpha": round(alpha_val, 6), "lr": round(lr_now, 6),
            "sec": round(dt, 1),
        }
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"[{tag} ep{epoch:03d}] L={avg_loss:.4f} "
              f"val_auroc={val_auroc:.4f} region_f1={val_region_f1:.4f} "
              f"alpha={alpha_val:.4f} lr={lr_now:.4g} ({dt:.1f}s)")

        # Save last checkpoint
        torch.save({
            "model": model.state_dict(), "epoch": epoch,
            "val_bgc_auroc": val_auroc, "alpha": alpha_val,
        }, out_dir / f"last_{tag}.pt")

    # --- Final summary ---
    print(f"\n[{tag}] Training complete.")
    print(f"  Best region_f1: {best_region_f1:.4f}")
    print(f"  Best AUROC: {best_auroc:.4f}")
    print(f"  Final alpha: {alpha_val:.4f}")

    summary = {
        "seed": args.seed, "K": args.K, "epochs": args.epochs,
        "best_region_f1": best_region_f1, "best_auroc": best_auroc,
        "final_alpha": alpha_val,
    }
    save_json(summary, out_dir / f"summary_{tag}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projections", required=True)
    p.add_argument("--meta-arrays", required=True)
    p.add_argument("--contig-index", required=True)
    p.add_argument("--bgc-csv", required=True, help="BGC annotations CSV for region-level eval")
    p.add_argument("--ckpt", default=None, help="Mainline checkpoint for classifier warm start")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--K", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pos-ratio", type=float, default=0.15)
    p.add_argument("--hard-neg-ratio", type=float, default=0.10)
    p.add_argument("--cnn-dropout", type=float, default=0.1)
    p.add_argument("--init-beta", type=float, default=-4.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
