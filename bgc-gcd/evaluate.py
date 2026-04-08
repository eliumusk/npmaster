"""Evaluation utilities and CLI."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score
)
from torch.utils.data import DataLoader

from data.dataset import BGCDataset, TYPE_COLS, build_phylum_map
from model.full_model import BGCModel
from utils import load_config, save_json, autodetect_device

TYPE_NAMES_7 = TYPE_COLS[:7]  # first 7 are BGC types


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_probs, all_labels, all_phyla, all_dom = [], [], [], []
    for emb, lbl, phy in loader:
        emb = emb.to(device)
        z, pred_logits, dom_logits = model(emb)
        probs = torch.sigmoid(pred_logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(lbl.numpy())
        all_phyla.append(np.asarray(phy))
        all_dom.append(dom_logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_phyla),
        np.concatenate(all_dom),
    )


def _safe_auroc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def evaluate_model(model, dataset, device, batch_size: int = 256) -> Dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    probs, labels, phyla, dom_pred = _predict(model, loader, device)

    # Per-type metrics on the 7 BGC classes
    per_type_auroc, per_type_ap, per_type_f1 = {}, {}, {}
    for i, name in enumerate(TYPE_NAMES_7):
        y_true = labels[:, i].astype(int)
        y_score = probs[:, i]
        per_type_auroc[name] = _safe_auroc(y_true, y_score)
        per_type_ap[name] = _safe_ap(y_true, y_score)
        per_type_f1[name] = (
            float(f1_score(y_true, (y_score >= 0.5).astype(int),
                           zero_division=0))
            if y_true.sum() > 0 else float("nan")
        )
    type_aurocs = [v for v in per_type_auroc.values() if not np.isnan(v)]
    mean_auroc = float(np.mean(type_aurocs)) if type_aurocs else float("nan")

    # BGC binary: is_bgc = 1 - is_negative
    is_bgc_true = 1 - labels[:, 7].astype(int)
    bgc_score = probs[:, :7].max(axis=1)  # max over 7 type probs
    bgc_auroc = _safe_auroc(is_bgc_true, bgc_score)
    bgc_prauc = _safe_ap(is_bgc_true, bgc_score)

    # Per-phylum BGC AUROC
    per_phylum_auroc = {}
    rev_map = {v: k for k, v in dataset.phylum_to_code.items()}
    for code in np.unique(phyla):
        mask = (phyla == code)
        if mask.sum() < 10:
            continue
        per_phylum_auroc[rev_map[code]] = _safe_auroc(
            is_bgc_true[mask], bgc_score[mask]
        )

    # DANN accuracy (proxy for phylum invariance)
    dann_acc = float((dom_pred == phyla).mean())

    return {
        "n_samples": int(len(labels)),
        "per_type_auroc": per_type_auroc,
        "per_type_ap": per_type_ap,
        "per_type_f1_at_05": per_type_f1,
        "mean_auroc": mean_auroc,
        "bgc_binary_auroc": bgc_auroc,
        "bgc_binary_prauc": bgc_prauc,
        "per_phylum_auroc": per_phylum_auroc,
        "dann_phylum_accuracy": dann_acc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--ablation", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val",
                   choices=["train", "val", "test_id", "test_ood"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config, args.ablation)
    device = autodetect_device(cfg.get("device", "cuda"))

    phy_map = build_phylum_map(cfg["data"]["labels_path"])
    dataset = BGCDataset(
        cfg["data"]["embeddings_path"], cfg["data"]["labels_path"],
        split=args.split, phylum_to_code=phy_map,
    )
    model = BGCModel(
        n_phyla=len(phy_map),
        input_dim=cfg["data"]["input_dim"],
        proj_hidden=cfg["model"]["proj_hidden"],
        proj_out=cfg["model"]["proj_out"],
        dropout=cfg["model"]["dropout"],
        dann_hidden=cfg["model"]["dann_hidden"],
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    metrics = evaluate_model(model, dataset, device,
                             batch_size=cfg["data"]["batch_size"])
    metrics["split"] = args.split
    print("Eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    out = args.out or Path(cfg["log"]["out_dir"]) / f"eval_{args.split}.json"
    save_json(metrics, out)


if __name__ == "__main__":
    main()
