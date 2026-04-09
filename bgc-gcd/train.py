"""Training loop for Stage 3 BGC-GCD model."""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import BGCDataset, build_phylum_map
from data.sampler import BalancedBatchSampler
from data.augmentation import EmbeddingAugmentation
from model.full_model import BGCModel
from losses.classification_loss import MultiLabelBCE
from losses.contrastive_loss import MultiLabelSupCon
from losses.dann_loss import DANNLoss
from evaluate import evaluate_model
from utils import load_config, set_seed, dann_lambda, autodetect_device, save_json


def _build_train_loader(ds: BGCDataset, batch_size: int, seed: int, workers: int,
                         pos_ratio: float = 0.5):
    sampler = BalancedBatchSampler(ds.labels, batch_size=batch_size, seed=seed,
                                   pos_ratio=pos_ratio)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=workers,
                        pin_memory=True)
    return loader, sampler


def train(cfg: dict, seed: int = 0, log_tag: str = "run"):
    set_seed(seed)
    device = autodetect_device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["log"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    phy_map = build_phylum_map(cfg["data"]["labels_path"])
    nll_path = cfg["data"].get("nll_path") if cfg["data"].get("input_dim", 4096) > 4096 else None
    train_ds = BGCDataset(cfg["data"]["embeddings_path"],
                          cfg["data"]["labels_path"], split="train",
                          phylum_to_code=phy_map, nll_path=nll_path)
    val_ds = BGCDataset(cfg["data"]["embeddings_path"],
                        cfg["data"]["labels_path"], split="val",
                        phylum_to_code=phy_map, nll_path=nll_path)

    n_phyla = len(phy_map)
    input_dim = train_ds.input_dim
    print(f"[{log_tag}] device={device} n_phyla={n_phyla} "
          f"train={len(train_ds)} val={len(val_ds)} input_dim={input_dim}")

    pos_ratio = cfg["data"].get("pos_ratio", 0.5)
    train_loader, sampler = _build_train_loader(
        train_ds, cfg["data"]["batch_size"], seed,
        cfg["data"]["num_workers"], pos_ratio=pos_ratio)
    print(f"[{log_tag}] pos_ratio={pos_ratio} "
          f"(n_pos={sampler.n_pos}, n_neg={sampler.n_neg} per batch)")

    model = BGCModel(
        n_phyla=n_phyla, input_dim=input_dim,
        proj_hidden=cfg["model"]["proj_hidden"],
        proj_out=cfg["model"]["proj_out"],
        dropout=cfg["model"]["dropout"],
        dann_hidden=cfg["model"]["dann_hidden"],
    ).to(device)

    # class weights for BCE
    pos_weight = None
    if cfg["train"]["use_class_weight"]:
        pos_weight = MultiLabelBCE.compute_pos_weight(train_ds.labels)
        pos_weight = pos_weight.to(device)
        # clamp extreme ratios (rare classes can explode)
        pos_weight = pos_weight.clamp(max=20.0)
        print(f"[{log_tag}] pos_weight: {pos_weight.tolist()}")

    loss_cls_fn = MultiLabelBCE(pos_weight=pos_weight).to(device)
    loss_con_fn = MultiLabelSupCon(
        tau=cfg["train"]["tau"],
        cross_phylum_weight=cfg["train"]["cross_phylum_weight"],
    )
    loss_dann_fn = DANNLoss()

    aug = EmbeddingAugmentation(
        noise_std=cfg["train"]["noise_std"],
        dropout_rate=cfg["train"]["dropout_rate"],
        enabled=cfg["train"]["augmentation"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    n_epochs = cfg["train"]["n_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)

    alpha = cfg["train"]["alpha"]
    beta = cfg["train"]["beta"]
    gamma = cfg["train"]["gamma"]
    grad_clip = cfg["train"]["grad_clip"]

    # training log
    log_csv = out_dir / f"train_log_{log_tag}.csv"
    fieldnames = ["epoch", "lambda_dann", "L_total", "L_cls", "L_con", "L_dann",
                  "val_mean_auroc", "val_bgc_auroc", "val_dann_acc", "lr",
                  "sec"]
    with open(log_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    debug_epochs = set(cfg["log"].get("debug_pair_stats_epochs", []))

    # Checkpoint selection metric: deployment-facing by default
    ckpt_metric = cfg["train"].get("checkpoint_metric", "bgc_binary_auroc")
    print(f"[{log_tag}] checkpoint_metric={ckpt_metric}")

    best_val = -1.0
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        lambd = dann_lambda(epoch, cfg["train"]["dann_warmup_epochs"])
        model.set_dann_lambda(lambd if gamma > 0 else 0.0)
        model.train()

        sums = {"L_total": 0.0, "L_cls": 0.0, "L_con": 0.0, "L_dann": 0.0}
        n_steps = 0
        t0 = time.time()

        # debug: dump pos-pair stats for first batch of selected epochs
        pair_stat_dump = None

        for step, (emb, labels, phyla) in enumerate(train_loader):
            emb = emb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            phyla = phyla.to(device, non_blocking=True)

            if cfg["train"]["augmentation"]:
                v1, v2 = aug(emb)
            else:
                v1 = v2 = emb

            z1, pred1, dom1 = model(v1)
            z2, pred2, dom2 = model(v2)

            L_cls = 0.5 * (loss_cls_fn(pred1, labels) + loss_cls_fn(pred2, labels))

            if beta > 0:
                z_cat = torch.cat([z1, z2], dim=0)
                lbl_cat = torch.cat([labels, labels], dim=0)
                phy_cat = torch.cat([phyla, phyla], dim=0)
                L_con = loss_con_fn(z_cat, lbl_cat, phy_cat)
            else:
                L_con = torch.tensor(0.0, device=device)

            if gamma > 0:
                L_dann = 0.5 * (
                    loss_dann_fn(dom1, phyla) + loss_dann_fn(dom2, phyla)
                )
            else:
                L_dann = torch.tensor(0.0, device=device)

            L_total = alpha * L_cls + beta * L_con + gamma * L_dann

            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            sums["L_total"] += L_total.item()
            sums["L_cls"] += L_cls.item()
            sums["L_con"] += L_con.item()
            sums["L_dann"] += L_dann.item()
            n_steps += 1

            if step == 0 and epoch in debug_epochs:
                pair_stat_dump = loss_con_fn.pair_stats(labels)

        scheduler.step()
        dt = time.time() - t0
        for k in sums:
            sums[k] /= max(n_steps, 1)

        # validate
        val_metrics = evaluate_model(model, val_ds, device,
                                     batch_size=cfg["data"]["batch_size"])
        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch, "lambda_dann": lambd,
            "L_total": sums["L_total"], "L_cls": sums["L_cls"],
            "L_con": sums["L_con"], "L_dann": sums["L_dann"],
            "val_mean_auroc": val_metrics["mean_auroc"],
            "val_bgc_auroc": val_metrics["bgc_binary_auroc"],
            "val_dann_acc": val_metrics["dann_phylum_accuracy"],
            "lr": lr_now, "sec": round(dt, 2),
        }
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"[{log_tag} ep{epoch:03d}] λ={lambd:.2f} "
              f"L={sums['L_total']:.4f} (cls={sums['L_cls']:.4f} "
              f"con={sums['L_con']:.4f} dann={sums['L_dann']:.4f}) "
              f"val_mAUROC={val_metrics['mean_auroc']:.4f} "
              f"val_bgcAUROC={val_metrics['bgc_binary_auroc']:.4f} "
              f"dann_acc={val_metrics['dann_phylum_accuracy']:.3f} "
              f"lr={lr_now:.4g} ({dt:.1f}s)")

        if pair_stat_dump is not None:
            print(f"[{log_tag} ep{epoch:03d}] pair_stats: "
                  f"n_pos={pair_stat_dump['n_pos_pairs']}/"
                  f"{pair_stat_dump['n_total_pairs']} "
                  f"({pair_stat_dump['frac_pos']:.3f}); "
                  f"combos={pair_stat_dump['type_combo_counts']}")

        # checkpoint — select by deployment-facing metric
        ckpt = {"model": model.state_dict(), "epoch": epoch,
                "config": cfg, "val": val_metrics,
                "checkpoint_metric": ckpt_metric}
        torch.save(ckpt, out_dir / f"last_{log_tag}.pt")
        current_metric = val_metrics.get(ckpt_metric, val_metrics["mean_auroc"])
        if current_metric > best_val:
            best_val = current_metric
            torch.save(ckpt, out_dir / f"best_{log_tag}.pt")
            print(f"[{log_tag} ep{epoch:03d}] ★ new best {ckpt_metric}={current_metric:.4f}")

    # final eval across all splits
    results = {"seed": seed, f"best_val_{ckpt_metric}": best_val,
               "checkpoint_metric": ckpt_metric}
    best = torch.load(out_dir / f"best_{log_tag}.pt", map_location=device)
    model.load_state_dict(best["model"])
    for split in ("val", "test_id", "test_ood"):
        try:
            ds = BGCDataset(cfg["data"]["embeddings_path"],
                            cfg["data"]["labels_path"], split=split,
                            phylum_to_code=phy_map, nll_path=nll_path)
        except Exception:
            continue
        if len(ds) == 0:
            continue
        m = evaluate_model(model, ds, device,
                           batch_size=cfg["data"]["batch_size"])
        m["split"] = split
        save_json(m, out_dir / f"eval_{split}_{log_tag}.json")
        results[f"{split}_mean_auroc"] = m["mean_auroc"]
        results[f"{split}_bgc_auroc"] = m["bgc_binary_auroc"]
        results[f"{split}_dann_acc"] = m["dann_phylum_accuracy"]
    save_json(results, out_dir / f"summary_{log_tag}.json")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--ablation", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="override n_epochs (useful for smoke tests)")
    args = p.parse_args()
    cfg = load_config(args.config, args.ablation)
    if args.epochs is not None:
        cfg["train"]["n_epochs"] = args.epochs
    tag = args.tag or (f"{cfg.get('name', 'run')}_s{args.seed}")
    train(cfg, seed=args.seed, log_tag=tag)


if __name__ == "__main__":
    main()
