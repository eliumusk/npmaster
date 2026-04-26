#!/usr/bin/env python3
"""DDP training for the per-token U-Net on prepared mmap-backed arrays."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evo2_per_token.model import (LOSS_REGISTRY, build_model,  # noqa: E402
                                  count_params, masked_bce_loss)
from evo2_per_token.training_data import (PreparedArrayDataset,  # noqa: E402
                                          deterministic_subset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP train per-token U-Net from prepared arrays")
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--d-reduced", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--model", choices=["pertoken", "unet"], default="unet")
    parser.add_argument(
        "--loss",
        choices=["bce", "dice_bce", "focal_bce", "tversky_bce", "weak_negative_bce"],
        default="dice_bce",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=8, help="Per-rank batch size.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--neg-weight", type=float, default=0.2,
                        help="Background/unlabeled weight for weak_negative_bce.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample-train", type=int, default=0)
    parser.add_argument("--subsample-val", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=150.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        return False, 0, 0, 1, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size, torch.device("cuda", local_rank)


def reduce_sums(values: list[float], device: torch.device, distributed: bool) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def is_main(rank: int) -> bool:
    return rank == 0


def load_prepared_dataset(
    prepared_dir: Path,
    split: str,
    subsample: int,
    seed: int,
) -> PreparedArrayDataset:
    feats = np.load(prepared_dir / f"{split}_feats.npy", mmap_mode="r")
    labels = np.load(prepared_dir / f"{split}_labels.npy", mmap_mode="r")
    indices = deterministic_subset(total=int(feats.shape[0]), requested=subsample, seed=seed)
    return PreparedArrayDataset(feats, labels, indices=indices)


def compute_loss(loss_fn, loss_name: str, logits: torch.Tensor, y: torch.Tensor, args) -> torch.Tensor:
    if loss_name == "bce":
        return loss_fn(logits, y, pos_weight=args.pos_weight)
    if loss_name == "weak_negative_bce":
        return loss_fn(logits, y, neg_weight=args.neg_weight, pos_weight=1.0)
    return loss_fn(logits, y)


def main() -> None:
    args = parse_args()
    distributed, rank, _local_rank, world_size, device = setup_distributed()
    set_seed(args.seed)

    if is_main(rank):
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[device.index])

    meta = json.loads((args.prepared_dir / "meta.json").read_text())
    args.d_reduced = int(meta["d_in"])

    train_ds = load_prepared_dataset(
        args.prepared_dir,
        "train",
        subsample=args.subsample_train,
        seed=args.seed,
    )
    val_ds = load_prepared_dataset(
        args.prepared_dir,
        "val",
        subsample=args.subsample_val,
        seed=args.seed + 1,
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model_kwargs: dict = {"d_in": args.d_reduced, "hidden": args.hidden}
    if args.model == "pertoken":
        model_kwargs["dilations"] = tuple(args.dilations)
    model = build_model(args.model, **model_kwargs).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])

    loss_fn = LOSS_REGISTRY[args.loss]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = args.out_dir / "train_log.jsonl"
    if is_main(rank):
        log_path.write_text("")
        config = {
            **vars(args),
            "model_type": args.model,
            "param_count": count_params(model.module if distributed else model),
            "n_train_windows": len(train_ds),
            "n_val_windows": len(val_ds),
            "t_latent": int(meta["t_latent"]),
            "downsample_k": int(meta["downsample_k"]),
            "world_size": world_size,
            "per_rank_batch": args.batch,
            "global_batch": args.batch * world_size,
            "data_loading": "np.load(..., mmap_mode='r')",
        }
        with open(args.out_dir / "config.json", "w") as fh:
            json.dump(config, fh, indent=2, default=str)
        print(
            f"model={args.model} loss={args.loss} params={config['param_count']} "
            f"train={len(train_ds)} val={len(val_ds)} world={world_size} "
            f"batch_per_rank={args.batch} global_batch={args.batch * world_size}",
            flush=True,
        )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    t_global = time.time()

    for epoch in range(args.epochs):
        wall_min = (time.time() - t_global) / 60.0
        if wall_min > args.max_wall_minutes:
            if is_main(rank):
                print(f"[wall {wall_min:.1f}min > budget {args.max_wall_minutes}]: stopping.")
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_n = 0
        n_steps = len(train_loader)
        log_every = max(1, n_steps // 5)
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = compute_loss(loss_fn, args.loss, logits, y, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * x.shape[0]
            train_n += x.shape[0]
            if is_main(rank) and (step % log_every == 0 or step == n_steps - 1):
                print(
                    f"  ep{epoch} step {step}/{n_steps} loss={loss.item():.4f} "
                    f"wall={((time.time() - t_global) / 60):.0f}m",
                    flush=True,
                )

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_pos_preds = 0
        val_pos_true = 0
        val_tp = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                val_loss = compute_loss(loss_fn, args.loss, logits, y, args)
                val_loss_sum += val_loss.item() * x.shape[0]
                val_n += x.shape[0]
                preds = (torch.sigmoid(logits) > 0.5).int()
                mask = y >= 0
                val_tp += ((preds == 1) & (y == 1) & mask).sum().item()
                val_pos_preds += ((preds == 1) & mask).sum().item()
                val_pos_true += ((y == 1) & mask).sum().item()

        train_loss_sum, train_n, val_loss_sum, val_n, val_tp, val_pos_preds, val_pos_true = reduce_sums(
            [
                train_loss_sum,
                float(train_n),
                val_loss_sum,
                float(val_n),
                float(val_tp),
                float(val_pos_preds),
                float(val_pos_true),
            ],
            device,
            distributed,
        )
        train_loss = train_loss_sum / max(1.0, train_n)
        val_loss = val_loss_sum / max(1.0, val_n)
        precision = val_tp / max(1.0, val_pos_preds)
        recall = val_tp / max(1.0, val_pos_true)
        f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
        rec_line = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_token_precision": precision,
            "val_token_recall": recall,
            "val_token_f1": f1,
            "elapsed_s": time.time() - t0,
            "wall_min_total": (time.time() - t_global) / 60.0,
        }

        if is_main(rank):
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec_line) + "\n")
            print(rec_line, flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if is_main(rank):
                state_model = model.module if distributed else model
                torch.save(
                    {
                        "model": state_model.state_dict(),
                        "args": vars(args),
                        "model_type": args.model,
                        "val_loss": val_loss,
                        "epoch": epoch,
                    },
                    args.out_dir / "best.pt",
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                if is_main(rank):
                    print(f"early stop: no improvement for {args.patience} epochs.")
                break

    if is_main(rank):
        print(f"done. best_epoch={best_epoch} best_val_loss={best_val:.6f}", flush=True)
    if distributed:
        dist.barrier(device_ids=[device.index])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
