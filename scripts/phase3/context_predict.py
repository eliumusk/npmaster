#!/usr/bin/env python3
"""
Phase 3: Contig-level prediction with context model.

Produces per-window predictions for ALL windows in specified splits,
then runs the existing ws1_score_swap evaluation pipeline.

Usage:
  python context_predict.py \
    --ckpt /data/.../best_route_a.pt \
    --projections /data/.../projections_256.npy \
    --meta-arrays /data/.../meta_arrays.pkl \
    --contig-index /data/.../contig_index.pkl \
    --emb /data/.../embeddings_merged.npy \
    --meta /data/.../windows_metadata_hardneg.csv \
    --bgc-csv /data/.../bgc_annotations.csv \
    --out /data/.../phase3/route_a/eval
"""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve

BGC_GCD = str(Path(__file__).resolve().parents[2] / "bgc-gcd")
if BGC_GCD not in sys.path:
    sys.path.insert(0, BGC_GCD)

EVAL_DIR = str(Path(__file__).resolve().parents[1] / "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from model.context_model import ContextBGCModel
from ws1_score_swap import (
    call_regions, match_regions, compute_prediction_level_precision,
    classify_unmatched, find_optimal_threshold, NumpyEncoder, CLASSES,
)


def load_context_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ContextBGCModel(proj_dim=256, n_labels=8)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    alpha = model.gate.alpha.item()
    print(f"  Loaded model: alpha={alpha:.4f}, epoch={ckpt.get('epoch', '?')}")
    return model, ckpt


@torch.no_grad()
def predict_contig_level(model, projections, contig_index, meta_arrays,
                         split, device, include_ambiguous=True):
    """Run contig-level inference for a split.

    Returns:
        preds_dict: {global_idx: (8,) probability array}
    """
    model.eval()
    splits = meta_arrays["split"]
    preds_dict = {}

    for contig_key, indices in contig_index.items():
        # Check if contig has any windows in this split
        split_indices = [i for i in indices if splits[i] == split]
        if not split_indices:
            continue

        # Load full contig projections
        L = len(indices)
        proj = np.stack([projections[i] for i in indices])
        proj_t = torch.from_numpy(proj).unsqueeze(0).to(device)  # (1, L, 256)
        valid = torch.ones(1, L, dtype=torch.bool, device=device)

        _, logits = model(proj_t, valid)
        probs = torch.sigmoid(logits[0]).cpu().numpy()  # (L, 8)

        for pos, g_idx in enumerate(indices):
            if splits[g_idx] != split:
                continue
            if not include_ambiguous and meta_arrays["label"][g_idx] == "ambiguous":
                continue
            preds_dict[g_idx] = probs[pos]

    return preds_dict


def build_window_list(preds_dict, meta_arrays):
    """Build sorted window list compatible with ws1_score_swap."""
    windows = []
    for g_idx, preds in preds_dict.items():
        w = {
            "genome_id": str(meta_arrays["genome_id"][g_idx]),
            "contig_id": str(meta_arrays["contig_id"][g_idx]),
            "window_start": int(meta_arrays["window_start"][g_idx]),
            "window_end": int(meta_arrays["window_start"][g_idx]) + 16384,
            "preds": [float(preds[j]) for j in range(8)],
            "label": str(meta_arrays["label"][g_idx]),
        }
        windows.append((g_idx, w))

    # Sort by genome, contig, position (CRITICAL: genomic coordinate order)
    windows.sort(key=lambda x: (x[1]["genome_id"], x[1]["contig_id"], x[1]["window_start"]))

    indices_sorted = [g_idx for g_idx, _ in windows]
    windows_sorted = [w for _, w in windows]
    preds_sorted = np.array([preds_dict[g_idx] for g_idx in indices_sorted])

    return windows_sorted, preds_sorted


def evaluate_split(model, projections, contig_index, meta_arrays,
                   as_bgcs_by_loc, threshold, split, device):
    """Full evaluation for one split."""
    print(f"\n=== Evaluating {split} ===")

    # Predict all windows (including ambiguous for region merging)
    preds_dict = predict_contig_level(
        model, projections, contig_index, meta_arrays,
        split, device, include_ambiguous=True,
    )
    print(f"  Predicted {len(preds_dict)} windows")

    windows_sorted, preds_sorted = build_window_list(preds_dict, meta_arrays)
    scores = preds_sorted[:, :7].max(axis=1)

    # Call regions
    regions = call_regions(windows_sorted, scores, threshold)

    # Count antiSMASH BGCs
    split_genomes = set(w["genome_id"] for w in windows_sorted)
    n_as_total = 0
    for (gid, cid), bgc_list in as_bgcs_by_loc.items():
        if gid in split_genomes:
            n_as_total += len(bgc_list)

    # Match
    matched_as, matched_pred, match_pairs = match_regions(regions, as_bgcs_by_loc)
    n_pred_matched, n_pred_total = compute_prediction_level_precision(regions, as_bgcs_by_loc)

    recall = len(matched_as) / max(1, n_as_total)
    precision_pred = n_pred_matched / max(1, n_pred_total)

    # Type concordance
    concordant = sum(1 for m in match_pairs if m["type_concordant"])
    type_conc = concordant / max(1, len(match_pairs))

    # Unmatched classification
    unmatched = classify_unmatched(regions, matched_pred, as_bgcs_by_loc)
    from collections import Counter
    cat_counts = Counter(u["category"] for u in unmatched)

    # Per-type recall
    type_matched = Counter()
    type_total = Counter()
    for (gid, cid), bgc_list in as_bgcs_by_loc.items():
        if gid not in split_genomes:
            continue
        for ab in bgc_list:
            ab_key = (gid, int(ab["start_bp"]), int(ab["end_bp"]))
            for t in ab["bgc_types"].split(";"):
                type_total[t] += 1
                if ab_key in matched_as:
                    type_matched[t] += 1
    per_type_recall = {c: round(type_matched.get(c, 0) / max(1, type_total.get(c, 0)), 4)
                       for c in CLASSES}

    result = {
        "split": split,
        "threshold": threshold,
        "n_antismash_bgcs": n_as_total,
        "n_matched": len(matched_as),
        "recall": round(recall, 4),
        "n_predicted_regions": len(regions),
        "precision_pred_level": round(precision_pred, 4),
        "type_concordance": round(type_conc, 4),
        "per_type_recall": per_type_recall,
        "n_unmatched": len(unmatched),
        "unmatched_categories": {
            "boundary_expansion": cat_counts.get("boundary_expansion", 0),
            "ambiguous_unmatched": cat_counts.get("ambiguous_unmatched", 0),
            "clean_missed": cat_counts.get("clean_missed", 0),
        },
    }

    print(f"  Recall: {recall:.4f} ({len(matched_as)}/{n_as_total})")
    print(f"  P(pred): {precision_pred:.4f}")
    print(f"  Regions: {len(regions)}")
    print(f"  Unmatched: {len(unmatched)} (clean_missed={cat_counts.get('clean_missed', 0)})")
    print(f"  Per-type: {per_type_recall}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--projections", required=True)
    p.add_argument("--meta-arrays", required=True)
    p.add_argument("--contig-index", required=True)
    p.add_argument("--meta", required=True, help="Original windows_metadata CSV")
    p.add_argument("--bgc-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Load model
    print("Loading model...")
    model, ckpt_data = load_context_model(args.ckpt, device)

    # Load data
    print("Loading projections...")
    projections = np.load(args.projections, mmap_mode="r")
    with open(args.meta_arrays, "rb") as f:
        meta_arrays = pickle.load(f)
    with open(args.contig_index, "rb") as f:
        contig_index = pickle.load(f)

    # Load BGC annotations
    print("Loading BGC annotations...")
    as_bgcs_by_loc = defaultdict(list)
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            as_bgcs_by_loc[(row["genome_id"], row["contig_id"])].append(row)

    # --- Find threshold on val ---
    print("\n=== Threshold Selection (val, non-ambiguous) ===")
    val_preds = predict_contig_level(
        model, projections, contig_index, meta_arrays,
        "val", device, include_ambiguous=False,
    )
    val_scores = []
    val_labels = []
    for g_idx, preds in val_preds.items():
        score = preds[:7].max()
        is_bgc = 1 - int(meta_arrays["labels"][g_idx, 7])
        val_scores.append(score)
        val_labels.append(is_bgc)

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    thr_info = find_optimal_threshold(val_scores, val_labels)
    threshold = thr_info["threshold"]
    print(f"  Threshold: {threshold:.4f} (F1={thr_info['f1']:.4f} P={thr_info['precision']:.4f} R={thr_info['recall']:.4f})")

    # --- Evaluate test_id and test_ood ---
    all_results = {"threshold": thr_info, "alpha": model.gate.alpha.item()}
    for split in ["test_id", "test_ood"]:
        result = evaluate_split(
            model, projections, contig_index, meta_arrays,
            as_bgcs_by_loc, threshold, split, device,
        )
        all_results[split] = result

    # --- Save ---
    out_path = os.path.join(args.out, "context_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
