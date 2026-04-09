#!/usr/bin/env python3
"""
P2-T2: Hard-Negative Mining for OOD Robustness
===============================================
Problem: RiPP accounts for 64.6% of OOD FPs, concentrated in Cyanobacteriota.
The model has never seen Cyano during training, so its RiPP decision boundary
is uncalibrated for Cyano-like sequence composition.

Strategy: Mine hard negatives from TRAINING data — windows that the current
model incorrectly predicts as RiPP with high confidence. Oversample these
during training to tighten the RiPP boundary.

Implementation:
  1. Score all training negatives with current model
  2. Identify top-K that score highest on RiPP (hard negatives)
  3. Create a weighted sampler that upsamples these hard negatives
  4. Retrain with the same architecture

This does NOT add OOD data to training — it only reweights existing training data.

Usage:
  python p2_t2_hard_negative_mining.py \
    --ckpt /data/muskliu/np_master/data/stage3_results/G_edgepad/best_G_edgepad.pt \
    --emb /data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy \
    --meta /data/muskliu/np_master/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv \
    --out /data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg \
    --top-k 5000
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["Polyketide","NRP","Terpene","RiPP","Saccharide","Alkaloid","Other"]


def load_model(ckpt_path, device="cpu"):
    bgc_gcd_path = str(Path(__file__).resolve().parents[2] / "bgc-gcd")
    if bgc_gcd_path not in sys.path:
        sys.path.insert(0, bgc_gcd_path)
    from model.full_model import BGCModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dann_keys = [k for k in ckpt["model"] if "dann" in k and "weight" in k and ckpt["model"][k].dim()==2]
    n_phyla = ckpt["model"][sorted(dann_keys)[-1]].shape[0] if dann_keys else 32
    m = BGCModel(n_phyla=n_phyla, input_dim=cfg["data"].get("input_dim",4096),
                 proj_hidden=cfg["model"]["proj_hidden"], proj_out=cfg["model"]["proj_out"],
                 dropout=cfg["model"]["dropout"], dann_hidden=cfg["model"]["dann_hidden"])
    m.load_state_dict(ckpt["model"]); m.to(device).eval()
    return m


def predict_batched(model, embeddings, device, batch_size=1024):
    preds = []
    ds = TensorDataset(torch.from_numpy(embeddings).float())
    for (x,) in DataLoader(ds, batch_size=batch_size):
        with torch.no_grad():
            _, pred, _ = model(x.to(device))
            preds.append(torch.sigmoid(pred).cpu().numpy())
    return np.vstack(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=5000,
                    help="Number of hardest negatives to flag")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ─── Load ───
    print("Loading model and data...")
    model = load_model(args.ckpt, device)
    emb = np.load(args.emb)
    meta = list(csv.DictReader(open(args.meta)))
    assert len(meta) == emb.shape[0]

    # ─── Find training negatives ───
    print("Finding training negatives...")
    train_neg_indices = [i for i, r in enumerate(meta)
                        if r["split"] == "train" and r["label"] == "negative"]
    print(f"  Training negatives: {len(train_neg_indices)}")

    # ─── Score them ───
    print("Scoring training negatives...")
    neg_emb = emb[train_neg_indices]
    neg_preds = predict_batched(model, neg_emb, device)
    # neg_preds shape: (N, 8) — 7 types + is_negative

    # ─── Analyze per-type FP scores ───
    print("\nPer-type score distribution on training negatives:")
    type_stats = {}
    for ti, t in enumerate(CLASSES):
        scores = neg_preds[:, ti]
        high = (scores >= 0.5).sum()
        type_stats[t] = {
            "mean": float(scores.mean()),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
            "n_above_0.5": int(high),
        }
        print(f"  {t:>12s}: mean={scores.mean():.4f} p90={np.percentile(scores,90):.4f} "
              f"p95={np.percentile(scores,95):.4f} p99={np.percentile(scores,99):.4f} "
              f"n>0.5={high}")

    # ─── Identify hard negatives: highest max(type_probs) ───
    max_type_scores = neg_preds[:, :7].max(axis=1)
    ripp_scores = neg_preds[:, 3]  # RiPP column

    # Strategy: flag negatives that are hard on ANY type (general), plus RiPP-specific
    general_hard_k = args.top_k
    ripp_hard_k = args.top_k

    # General hard negatives: highest max(type_probs)
    general_order = np.argsort(-max_type_scores)
    general_hard_indices = [train_neg_indices[i] for i in general_order[:general_hard_k]]

    # RiPP-specific hard negatives: highest RiPP score
    ripp_order = np.argsort(-ripp_scores)
    ripp_hard_indices = [train_neg_indices[i] for i in ripp_order[:ripp_hard_k]]

    # Union
    all_hard = set(general_hard_indices) | set(ripp_hard_indices)
    print(f"\nHard negatives identified:")
    print(f"  General (top-{general_hard_k} by max_type_score): {len(general_hard_indices)}")
    print(f"  RiPP-specific (top-{ripp_hard_k} by RiPP score): {len(ripp_hard_indices)}")
    print(f"  Union: {len(all_hard)}")

    # Score stats for hard negatives
    hard_gen_scores = max_type_scores[general_order[:general_hard_k]]
    hard_ripp_scores = ripp_scores[ripp_order[:ripp_hard_k]]
    print(f"\n  General hard neg score range: {hard_gen_scores.min():.4f} - {hard_gen_scores.max():.4f}")
    print(f"  RiPP hard neg score range:    {hard_ripp_scores.min():.4f} - {hard_ripp_scores.max():.4f}")

    # Phylum distribution of hard negatives
    hard_phyla = Counter(meta[i].get("phylum","") for i in all_hard)
    print(f"\n  Hard negative phylum distribution (top 5):")
    for p, c in hard_phyla.most_common(5):
        print(f"    {p}: {c}")

    # ─── Write hard-negative weight column ───
    # Create a new metadata CSV with a 'hard_negative' column
    print("\nWriting updated metadata with hard_negative flag...")
    out_meta = os.path.join(args.out, "windows_metadata_hardneg.csv")
    with open(args.meta) as fi, open(out_meta, "w", newline="") as fo:
        reader = csv.DictReader(fi)
        fieldnames = list(reader.fieldnames) + ["hard_negative"]
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        n_flagged = 0
        for i, row in enumerate(reader):
            row["hard_negative"] = "1" if i in all_hard else "0"
            writer.writerow(row)
            if i in all_hard:
                n_flagged += 1
    print(f"  Flagged {n_flagged} hard negatives in {out_meta}")

    # ─── Save analysis ───
    summary = {
        "task_id": "P2-T2",
        "description": "Hard-negative mining for OOD robustness",
        "checkpoint": args.ckpt,
        "n_train_negatives": len(train_neg_indices),
        "general_hard_k": general_hard_k,
        "ripp_hard_k": ripp_hard_k,
        "n_hard_union": len(all_hard),
        "type_score_stats": type_stats,
        "hard_neg_phylum_dist": dict(hard_phyla.most_common(10)),
        "general_score_range": [float(hard_gen_scores.min()), float(hard_gen_scores.max())],
        "ripp_score_range": [float(hard_ripp_scores.min()), float(hard_ripp_scores.max())],
        "output_metadata": out_meta,
    }
    with open(os.path.join(args.out, "hard_neg_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.out}/")
    print(f"\nNext: retrain with hard-negative upsampling using this metadata")


if __name__ == "__main__":
    main()
