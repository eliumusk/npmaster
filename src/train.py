"""
Phase 3 Route B: End-to-end context model training.

Key differences from Route A:
  - Input: 4096-dim raw embeddings (not frozen 256-dim projections)
  - ProjectionHead trainable (warm-started, lower lr)
  - Contig chunks with overlap, loss on center region only
  - Full epoch traversal, class balance via loss-level pos_weight
  - Checkpoint by val region F1 (every 5 epochs)
"""
from __future__ import annotations
import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.chunk_dataset import ChunkDataset
from data.context_dataset import ContextDataset
from model.context_full_model import ContextFullModel
from evaluate import evaluate_context_model, evaluate_region_level
from utils import set_seed, save_json


def apply_hard_negative_weights(
    per_pos_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    hard_neg_mask: torch.Tensor,
    hard_neg_weight: float,
) -> tuple[torch.Tensor, int]:
    """Average masked loss with optional extra weight on flagged hard negatives."""
    masked = loss_mask.bool()
    n_hard = int((masked & hard_neg_mask.bool()).sum().item())
    if hard_neg_weight <= 1.0:
        denom = masked.float().sum().clamp_min(1.0)
        return (per_pos_loss * masked.float()).sum() / denom, n_hard

    weight = torch.ones_like(per_pos_loss)
    weight = torch.where(masked & hard_neg_mask.bool(), hard_neg_weight, weight)
    weight = weight * masked.float()
    denom = weight.sum().clamp_min(1.0)
    return (per_pos_loss * weight).sum() / denom, n_hard


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"route_b_s{args.seed}"

    # --- Data ---
    print(f"[{tag}] Loading training data...")
    train_ds = ChunkDataset(
        args.embeddings, args.chunk_index, args.meta_arrays, split="train",
    )

    # Val dataset uses Route A style (context neighborhoods) for eval
    print(f"[{tag}] Loading val data...")
    val_ds = ContextDataset(
        args.projections, args.meta_arrays, args.contig_index,
        split="val", K=11,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"[{tag}] train chunks={len(train_ds)} val={len(val_ds)} "
          f"L_chunk={train_ds.L_chunk} loss_windows={train_ds.n_loss_windows}")
    print(f"[{tag}] hard_neg_loss_windows={train_ds.n_hard_loss_windows} "
          f"hard_neg_weight={args.hard_neg_weight}")

    # --- Model ---
    model = ContextFullModel(
        input_dim=train_ds.input_dim,
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
        proj_dropout=args.proj_dropout,
        n_labels=8,
        cnn_dropout=args.cnn_dropout,
        init_beta=args.init_beta,
    ).to(device)

    # Warm-start from mainline checkpoint
    if args.ckpt:
        print(f"[{tag}] Warm-starting from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

        # Load projector
        proj_state = {}
        for k, v in ckpt["model"].items():
            if k.startswith("projector."):
                proj_state[k.replace("projector.", "")] = v
        model.projector.load_state_dict(proj_state)
        print(f"[{tag}]   ProjectionHead loaded")

        # Load classifier (mainline uses MultiLabelClassifier.fc)
        cls_state = {}
        for k, v in ckpt["model"].items():
            if k.startswith("classifier.fc."):
                cls_state[k.replace("classifier.fc.", "")] = v
            elif k.startswith("classifier."):
                cls_state[k.replace("classifier.", "")] = v
        model.classifier.load_state_dict(cls_state)
        print(f"[{tag}]   Classifier loaded")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] Total params: {n_params:,}")

    # --- Loss ---
    # pos_weight: for each of 8 columns, weight = n_neg / n_pos (clamped at 20)
    # Column 7 (is_negative) is inverted: most windows are negative, so pos_weight should be < 1
    n_pos = train_ds.n_pos_per_class.copy()
    n_total = float(train_ds.n_loss_windows)
    pw = np.ones(8)
    for c in range(8):
        if n_pos[c] > 0:
            pw[c] = (n_total - n_pos[c]) / n_pos[c]
    pos_weight = torch.from_numpy(pw).float().to(device).clamp(max=20.0)
    print(f"[{tag}] pos_weight: {[round(x, 3) for x in pos_weight.tolist()]}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # --- Optimizer with layered lr ---
    param_groups = model.param_groups(
        lr_proj=args.lr_proj,
        lr_context=args.lr_context,
        lr_classifier=args.lr_classifier,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)

    # Cosine with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training log ---
    log_csv = out_dir / f"train_log_{tag}.csv"
    fieldnames = ["epoch", "L_bce", "val_bgc_auroc", "val_region_f1",
                  "alpha", "lr_proj", "lr_ctx", "hard_neg_frac", "sec"]
    with open(log_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    save_json({
        "tag": tag,
        "seed": args.seed,
        "embeddings": args.embeddings,
        "projections": args.projections,
        "chunk_index": args.chunk_index,
        "meta_arrays": args.meta_arrays,
        "contig_index": args.contig_index,
        "bgc_csv": args.bgc_csv,
        "ckpt": args.ckpt,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_proj": args.lr_proj,
        "lr_context": args.lr_context,
        "lr_classifier": args.lr_classifier,
        "wd": args.wd,
        "warmup_epochs": args.warmup_epochs,
        "proj_hidden": args.proj_hidden,
        "proj_out": args.proj_out,
        "proj_dropout": args.proj_dropout,
        "cnn_dropout": args.cnn_dropout,
        "init_beta": args.init_beta,
        "hard_neg_weight": args.hard_neg_weight,
        "n_train_chunks": len(train_ds),
        "n_train_loss_windows": train_ds.n_loss_windows,
        "n_train_hard_neg_loss_windows": train_ds.n_hard_loss_windows,
    }, out_dir / f"config_{tag}.json")

    # For val eval, load contig index and metadata
    with open(args.contig_index, "rb") as f:
        contig_index = pickle.load(f)
    with open(args.meta_arrays, "rb") as f:
        meta_arrays = pickle.load(f)
    embeddings_path = args.embeddings  # raw 4096-dim for full model eval

    best_region_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_loss_positions = 0
        n_hard_positions = 0
        t0 = time.time()

        for emb, valid_mask, loss_mask, hard_neg_mask, labels in train_loader:
            emb = emb.to(device, non_blocking=True)            # (B, L, 4096)
            valid_mask = valid_mask.to(device, non_blocking=True)  # (B, L)
            loss_mask = loss_mask.to(device, non_blocking=True)    # (B, L)
            hard_neg_mask = hard_neg_mask.to(device, non_blocking=True)  # (B, L)
            labels = labels.to(device, non_blocking=True)          # (B, L, 8)

            _, logits = model(emb, valid_mask)  # (B, L, 8)

            # Masked loss: only on loss_mask positions
            per_pos_loss = loss_fn(logits, labels)  # (B, L, 8)
            # Average over 8 classes per position, then mask
            per_pos_loss = per_pos_loss.mean(dim=-1)  # (B, L)
            n_valid = int(loss_mask.sum().item())
            loss, n_hard = apply_hard_negative_weights(
                per_pos_loss,
                loss_mask,
                hard_neg_mask,
                hard_neg_weight=args.hard_neg_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item() * n_valid
            n_loss_positions += n_valid
            n_hard_positions += n_hard

        scheduler.step()
        dt = time.time() - t0
        avg_loss = total_loss / max(n_loss_positions, 1)
        alpha_val = model.gate.alpha.item()

        # --- Val evaluation: contig-level with full model ---
        val_auroc, val_region_f1 = _eval_val_full_model(
            model, contig_index, meta_arrays, embeddings_path, device,
            do_region=(((epoch + 1) % 5 == 0) or epoch == args.epochs - 1),
        )

        lr_p = optimizer.param_groups[0]["lr"]
        lr_c = optimizer.param_groups[1]["lr"]

        row = {
            "epoch": epoch, "L_bce": round(avg_loss, 6),
            "val_bgc_auroc": round(val_auroc, 4),
            "val_region_f1": round(val_region_f1, 4) if not np.isnan(val_region_f1) else "",
            "alpha": round(alpha_val, 6),
            "lr_proj": f"{lr_p:.2e}", "lr_ctx": f"{lr_c:.2e}",
            "hard_neg_frac": round(n_hard_positions / max(n_loss_positions, 1), 4),
            "sec": round(dt, 1),
        }
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        # Checkpoint by region F1
        if not np.isnan(val_region_f1) and val_region_f1 > best_region_f1:
            best_region_f1 = val_region_f1
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_region_f1": val_region_f1, "val_bgc_auroc": val_auroc,
                "alpha": alpha_val,
            }, out_dir / f"best_{tag}.pt")
            print(f"[{tag} ep{epoch:03d}] * best region_f1={val_region_f1:.4f}")

        print(f"[{tag} ep{epoch:03d}] L={avg_loss:.4f} "
              f"auroc={val_auroc:.4f} rf1={val_region_f1:.4f} "
              f"a={alpha_val:.4f} lr_p={lr_p:.2e} "
              f"hard={n_hard_positions/max(n_loss_positions,1):.4f} ({dt:.1f}s)")

        torch.save({
            "model": model.state_dict(), "epoch": epoch,
            "alpha": alpha_val,
        }, out_dir / f"last_{tag}.pt")

    print(f"\n[{tag}] Done. Best region_f1={best_region_f1:.4f} alpha={alpha_val:.4f}")
    save_json({"seed": args.seed, "best_region_f1": best_region_f1,
               "final_alpha": alpha_val}, out_dir / f"summary_{tag}.json")


@torch.no_grad()
def _eval_val_full_model(model, contig_index, meta_arrays, embeddings_path,
                         device, do_region=False):
    """Evaluate full model on val using contig-level inference with raw 4096-dim embeddings."""
    model.eval()
    embeddings = np.load(embeddings_path, mmap_mode="r")

    splits = meta_arrays["split"]
    label_arr = meta_arrays["labels"]
    label_strs = meta_arrays["label"]

    all_scores = []
    all_is_bgc = []

    for contig_key, indices in contig_index.items():
        val_indices = [i for i in indices
                      if splits[i] == "val" and label_strs[i] != "ambiguous"]
        if not val_indices:
            continue

        L = len(indices)
        emb = np.stack([embeddings[i] for i in indices])
        emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)
        valid = torch.ones(1, L, dtype=torch.bool, device=device)
        _, logits = model(emb_t, valid)
        probs = torch.sigmoid(logits[0]).cpu().numpy()

        val_set = set(val_indices)
        for pos, g_idx in enumerate(indices):
            if g_idx not in val_set:
                continue
            score = probs[pos, :7].max()
            is_bgc = 1 - int(label_arr[g_idx, 7])
            all_scores.append(score)
            all_is_bgc.append(is_bgc)

    if not all_scores:
        return float("nan"), float("nan")

    all_scores = np.array(all_scores)
    all_is_bgc = np.array(all_is_bgc)

    from sklearn.metrics import roc_auc_score, precision_recall_curve
    auroc = float(roc_auc_score(all_is_bgc, all_scores))

    region_f1 = float("nan")
    if do_region:
        prec, rec, thr = precision_recall_curve(all_is_bgc, all_scores)
        f1s = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1s)
        region_f1 = float(f1s[best_idx])

    return auroc, region_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, help="Raw 4096-dim embeddings")
    p.add_argument("--projections", required=True, help="256-dim projections (for val eval fallback)")
    p.add_argument("--chunk-index", required=True)
    p.add_argument("--meta-arrays", required=True)
    p.add_argument("--contig-index", required=True)
    p.add_argument("--bgc-csv", required=True)
    p.add_argument("--ckpt", default=None, help="Mainline checkpoint for warm start")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr-proj", type=float, default=1e-4)
    p.add_argument("--lr-context", type=float, default=1e-3)
    p.add_argument("--lr-classifier", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--proj-out", type=int, default=256)
    p.add_argument("--proj-dropout", type=float, default=0.3)
    p.add_argument("--cnn-dropout", type=float, default=0.1)
    p.add_argument("--init-beta", type=float, default=-4.0)
    p.add_argument("--hard-neg-weight", type=float, default=1.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
