#!/usr/bin/env python3
"""
Error analysis: diagnose WHY the model misses known BGCs.

Analyses:
  1. BGC-level miss rate by length bucket
  2. BGC-level miss rate by type (+ type×length cross-table)
  3. Window-level miss rate by overlap ratio
  4. FN score distribution (is the model close or far from threshold?)
  5. BGC-level miss rate by is_truncated

A BGC is "detected" if at least one of its overlapping positive windows
is predicted as BGC (is_negative < threshold).
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── Setup ───

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
ALL_COLS = CLASSES + ["is_negative"]

LENGTH_BUCKETS = [
    ("<5kb",    0,     5000),
    ("5-10kb",  5000,  10000),
    ("10-20kb", 10000, 20000),
    ("20-50kb", 20000, 50000),
    (">50kb",   50000, float("inf")),
]

OVERLAP_BUCKETS = [
    ("50-60%", 0.50, 0.60),
    ("60-80%", 0.60, 0.80),
    ("80-100%", 0.80, 1.01),
]


def load_model(ckpt_path, device="cpu"):
    bgc_gcd_path = str(Path(__file__).resolve().parents[2] / "bgc-gcd")
    if bgc_gcd_path not in sys.path:
        sys.path.insert(0, bgc_gcd_path)
    from model.full_model import BGCModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    n_phyla = 32
    dann_keys = sorted([k for k in ckpt["model"] if "dann" in k and "weight" in k and ckpt["model"][k].dim() == 2])
    if dann_keys:
        n_phyla = ckpt["model"][dann_keys[-1]].shape[0]
    input_dim = cfg["data"].get("input_dim", 4096)
    m = BGCModel(n_phyla=n_phyla, input_dim=input_dim,
                 proj_hidden=cfg["model"]["proj_hidden"],
                 proj_out=cfg["model"]["proj_out"],
                 dropout=cfg["model"]["dropout"],
                 dann_hidden=cfg["model"]["dann_hidden"])
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()
    return m


def predict_batched(model, embeddings, device, batch_size=512):
    preds = []
    ds = TensorDataset(torch.from_numpy(embeddings).float())
    loader = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for (x,) in loader:
            _, pred, _ = model(x.to(device))
            pred = torch.sigmoid(pred)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def load_bgc_annotations(bgc_csv):
    """Load BGC annotations keyed by bgc_id."""
    bgcs = {}
    with open(bgc_csv) as f:
        for row in csv.DictReader(f):
            bgcs[row["bgc_id"]] = row
    return bgcs


def map_windows_to_bgcs(meta, bgc_annotations):
    """For each positive window in test_id, find which BGC(s) it overlaps with."""
    # Build BGC interval index per (genome, contig)
    bgc_by_loc = defaultdict(list)
    for bgc_id, bgc in bgc_annotations.items():
        key = (bgc["genome_id"], bgc["contig_id"])
        bgc_by_loc[key].append(bgc)

    # For each positive test_id window, find overlapping BGCs
    window_bgc_map = defaultdict(list)  # window_idx -> list of bgc_ids
    bgc_window_map = defaultdict(list)  # bgc_id -> list of window_idx

    for i, r in enumerate(meta):
        if r["split"] != "test_id" or r["label"] != "positive":
            continue
        ws = int(r["window_start"])
        we = int(r["window_end"])
        key = (r["genome_id"], r["contig_id"])
        for bgc in bgc_by_loc.get(key, []):
            bs = int(bgc["start_bp"])
            be = int(bgc["end_bp"])
            overlap = max(0, min(we, be) - max(ws, bs))
            if overlap > 0:
                window_bgc_map[i].append(bgc["bgc_id"])
                bgc_window_map[bgc["bgc_id"]].append(i)

    return window_bgc_map, bgc_window_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--nll", default=None)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--bgc_csv", required=True)
    parser.add_argument("--is_neg_threshold", type=float, default=0.128)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ─── Load data ───
    print("Loading model...")
    model = load_model(args.ckpt, device)

    print("Loading embeddings...")
    emb = np.load(args.embeddings)
    if args.nll and Path(args.nll).exists():
        nll = np.load(args.nll)
        emb = np.concatenate([emb, nll], axis=1)
        print(f"  Concatenated NLL: {emb.shape}")

    print("Loading metadata...")
    with open(args.metadata) as f:
        reader = csv.DictReader(f)
        meta = list(reader)
    assert len(meta) == emb.shape[0], f"Metadata rows {len(meta)} != embeddings {emb.shape[0]}"

    print("Loading BGC annotations...")
    bgc_annotations = load_bgc_annotations(args.bgc_csv)
    print(f"  {len(bgc_annotations)} BGCs total")

    # ─── Predict on test_id ───
    print("Predicting on test_id...")
    tid_mask = np.array([r["split"] == "test_id" and r["label"] != "ambiguous" for r in meta])
    tid_indices = np.where(tid_mask)[0]
    tid_emb = emb[tid_mask]
    tid_meta = [meta[i] for i in tid_indices]
    tid_preds = predict_batched(model, tid_emb, device)
    print(f"  {len(tid_meta)} test_id windows (non-ambiguous)")

    is_neg_thr = args.is_neg_threshold

    # ─── Map windows to BGCs ───
    print("Mapping windows to BGCs...")
    # We need global indices for the mapping, so work with full meta
    # but only analyze test_id windows
    _, bgc_window_map_global = map_windows_to_bgcs(meta, bgc_annotations)

    # Build test_id specific indices
    # Map global meta index -> tid_preds index
    global_to_tid = {}
    for tid_idx, global_idx in enumerate(tid_indices):
        global_to_tid[global_idx] = tid_idx

    # For each BGC, check if it's in a test_id genome and has positive windows
    test_id_genomes = set(r["genome_id"] for r in tid_meta)
    test_id_bgcs = {bid: bgc for bid, bgc in bgc_annotations.items()
                    if bgc["genome_id"] in test_id_genomes}
    print(f"  {len(test_id_bgcs)} BGCs in test_id genomes")

    # Re-do mapping using only test_id positive windows with their tid_preds indices
    # Build BGC -> list of (tid_pred_index, overlap_ratio)
    bgc_windows = defaultdict(list)  # bgc_id -> [(tid_idx, overlap_ratio, is_negative_score)]

    for tid_idx, r in enumerate(tid_meta):
        if r["label"] != "positive":
            continue
        ws = int(r["window_start"])
        we = int(r["window_end"])
        window_len = we - ws
        key = (r["genome_id"], r["contig_id"])

        for bgc in [b for b in bgc_annotations.values()
                     if b["genome_id"] == r["genome_id"] and b["contig_id"] == r["contig_id"]]:
            bs = int(bgc["start_bp"])
            be = int(bgc["end_bp"])
            overlap = max(0, min(we, be) - max(ws, bs))
            if overlap > 0:
                overlap_ratio = overlap / window_len
                is_neg_score = float(tid_preds[tid_idx, 7])
                bgc_windows[bgc["bgc_id"]].append({
                    "tid_idx": tid_idx,
                    "overlap_ratio": overlap_ratio,
                    "is_neg_score": is_neg_score,
                    "pred_bgc": is_neg_score < is_neg_thr,
                    "window_id": r["window_id"],
                })

    # ─── Analysis 1: BGC-level miss rate by length bucket ───
    print("\n=== Analysis 1: Miss rate by BGC length ===")
    length_results = {}
    for bucket_name, lo, hi in LENGTH_BUCKETS:
        bucket_bgcs = [bid for bid, bgc in test_id_bgcs.items()
                       if lo <= int(bgc["bgc_length"]) < hi]
        n_total = len(bucket_bgcs)
        n_detected = sum(1 for bid in bucket_bgcs
                        if any(w["pred_bgc"] for w in bgc_windows.get(bid, [])))
        n_no_windows = sum(1 for bid in bucket_bgcs if len(bgc_windows.get(bid, [])) == 0)
        n_missed = n_total - n_detected
        miss_rate = n_missed / n_total if n_total > 0 else 0

        # Average is_neg score for missed BGCs' best window
        missed_best_scores = []
        for bid in bucket_bgcs:
            wins = bgc_windows.get(bid, [])
            if wins and not any(w["pred_bgc"] for w in wins):
                best = min(w["is_neg_score"] for w in wins)
                missed_best_scores.append(best)

        length_results[bucket_name] = {
            "n_bgcs": n_total,
            "n_detected": n_detected,
            "n_missed": n_missed,
            "n_no_positive_windows": n_no_windows,
            "miss_rate": round(miss_rate, 4),
            "missed_best_is_neg_mean": round(float(np.mean(missed_best_scores)), 4) if missed_best_scores else None,
        }
        print(f"  {bucket_name:>8s}: {n_total:4d} BGCs, {n_detected:4d} detected, "
              f"{n_missed:4d} missed ({miss_rate:.1%}), {n_no_windows} without positive windows")

    # ─── Analysis 2: Miss rate by BGC type ───
    print("\n=== Analysis 2: Miss rate by BGC type ===")
    type_results = {}
    for cls in CLASSES:
        typed_bgcs = [bid for bid, bgc in test_id_bgcs.items()
                      if cls.lower() in bgc["bgc_types"].lower() or
                      (cls == "NRP" and "nrp" in bgc["bgc_types"].lower())]
        n_total = len(typed_bgcs)
        n_detected = sum(1 for bid in typed_bgcs
                        if any(w["pred_bgc"] for w in bgc_windows.get(bid, [])))
        n_missed = n_total - n_detected
        miss_rate = n_missed / n_total if n_total > 0 else 0

        # Avg length of missed vs detected
        missed_lens = [int(test_id_bgcs[bid]["bgc_length"]) for bid in typed_bgcs
                       if not any(w["pred_bgc"] for w in bgc_windows.get(bid, []))]
        detected_lens = [int(test_id_bgcs[bid]["bgc_length"]) for bid in typed_bgcs
                        if any(w["pred_bgc"] for w in bgc_windows.get(bid, []))]

        type_results[cls] = {
            "n_bgcs": n_total,
            "n_detected": n_detected,
            "n_missed": n_missed,
            "miss_rate": round(miss_rate, 4),
            "avg_length_missed": int(np.mean(missed_lens)) if missed_lens else None,
            "avg_length_detected": int(np.mean(detected_lens)) if detected_lens else None,
        }
        ml = str(int(np.mean(missed_lens))) if missed_lens else "N/A"
        dl = str(int(np.mean(detected_lens))) if detected_lens else "N/A"
        print(f"  {cls:>12s}: {n_total:4d} BGCs, miss rate {miss_rate:.1%}, "
              f"avg len missed={ml:>6s}, detected={dl:>6s}")

    # ─── Analysis 2b: Type × Length cross-table ───
    print("\n=== Analysis 2b: Type × Length cross-table (miss rates) ===")
    cross_table = {}
    header = f"{'Type':>12s}"
    for bucket_name, _, _ in LENGTH_BUCKETS:
        header += f" | {bucket_name:>8s}"
    print(header)
    print("-" * len(header))

    for cls in CLASSES:
        typed_bgcs = [bid for bid, bgc in test_id_bgcs.items()
                      if cls.lower() in bgc["bgc_types"].lower() or
                      (cls == "NRP" and "nrp" in bgc["bgc_types"].lower())]
        cross_table[cls] = {}
        row = f"{cls:>12s}"
        for bucket_name, lo, hi in LENGTH_BUCKETS:
            bucket_typed = [bid for bid in typed_bgcs
                           if lo <= int(test_id_bgcs[bid]["bgc_length"]) < hi]
            n = len(bucket_typed)
            if n == 0:
                cross_table[cls][bucket_name] = {"n": 0, "miss_rate": None}
                row += f" | {'---':>8s}"
            else:
                n_missed = sum(1 for bid in bucket_typed
                              if not any(w["pred_bgc"] for w in bgc_windows.get(bid, [])))
                mr = n_missed / n
                cross_table[cls][bucket_name] = {"n": n, "miss_rate": round(mr, 4)}
                row += f" | {mr:>7.1%} ({n})"
        print(row)

    # ─── Analysis 3: Window-level miss rate by overlap ratio ───
    print("\n=== Analysis 3: Window miss rate by overlap ratio ===")
    overlap_results = {}
    all_positive_windows = []
    for tid_idx, r in enumerate(tid_meta):
        if r["label"] == "positive":
            overlap_ratio = float(r.get("max_bgc_overlap_ratio", 0))
            is_neg_score = float(tid_preds[tid_idx, 7])
            pred_bgc = is_neg_score < is_neg_thr
            all_positive_windows.append({
                "overlap_ratio": overlap_ratio,
                "is_neg_score": is_neg_score,
                "pred_bgc": pred_bgc,
            })

    for bucket_name, lo, hi in OVERLAP_BUCKETS:
        bucket_wins = [w for w in all_positive_windows if lo <= w["overlap_ratio"] < hi]
        n = len(bucket_wins)
        n_missed = sum(1 for w in bucket_wins if not w["pred_bgc"])
        miss_rate = n_missed / n if n > 0 else 0
        avg_score = float(np.mean([w["is_neg_score"] for w in bucket_wins])) if bucket_wins else None
        overlap_results[bucket_name] = {
            "n_windows": n,
            "n_missed": n_missed,
            "miss_rate": round(miss_rate, 4),
            "avg_is_neg_score": round(avg_score, 4) if avg_score else None,
        }
        print(f"  {bucket_name:>8s}: {n:5d} windows, {n_missed:5d} missed ({miss_rate:.1%}), "
              f"avg is_neg={avg_score:.3f}" if avg_score else "")

    # ─── Analysis 4: FN score distribution ───
    print("\n=== Analysis 4: FN score distribution ===")
    fn_scores = []
    for w in all_positive_windows:
        if not w["pred_bgc"]:
            fn_scores.append(w["is_neg_score"])

    fn_scores = np.array(fn_scores)
    if len(fn_scores) > 0:
        score_bins = [
            (f"[{is_neg_thr:.3f}, 0.2)", is_neg_thr, 0.2),
            ("[0.2, 0.3)", 0.2, 0.3),
            ("[0.3, 0.5)", 0.3, 0.5),
            ("[0.5, 0.7)", 0.5, 0.7),
            ("[0.7, 1.0]", 0.7, 1.01),
        ]
        fn_distribution = {}
        for name, lo, hi in score_bins:
            n = int(np.sum((fn_scores >= lo) & (fn_scores < hi)))
            fn_distribution[name] = n
            print(f"  {name:>16s}: {n:5d} ({n/len(fn_scores):.1%})")

        fn_stats = {
            "total_fn_windows": len(fn_scores),
            "mean": round(float(fn_scores.mean()), 4),
            "median": round(float(np.median(fn_scores)), 4),
            "p25": round(float(np.percentile(fn_scores, 25)), 4),
            "p75": round(float(np.percentile(fn_scores, 75)), 4),
            "distribution": fn_distribution,
        }
        print(f"\n  Mean is_neg score of FN: {fn_scores.mean():.4f}")
        print(f"  Median: {np.median(fn_scores):.4f}")
        print(f"  25th-75th percentile: [{np.percentile(fn_scores, 25):.4f}, {np.percentile(fn_scores, 75):.4f}]")
    else:
        fn_stats = {"total_fn_windows": 0}
        print("  No FN windows!")

    # ─── Analysis 5: Miss rate by truncation status ───
    print("\n=== Analysis 5: Miss rate by truncation ===")
    trunc_results = {}
    for trunc_status, label in [(0, "complete"), (1, "truncated")]:
        trunc_bgcs = [bid for bid, bgc in test_id_bgcs.items()
                      if int(bgc.get("is_truncated", 0)) == trunc_status]
        n = len(trunc_bgcs)
        n_detected = sum(1 for bid in trunc_bgcs
                        if any(w["pred_bgc"] for w in bgc_windows.get(bid, [])))
        n_missed = n - n_detected
        miss_rate = n_missed / n if n > 0 else 0
        trunc_results[label] = {
            "n_bgcs": n,
            "n_detected": n_detected,
            "n_missed": n_missed,
            "miss_rate": round(miss_rate, 4),
        }
        print(f"  {label:>10s}: {n:4d} BGCs, miss rate {miss_rate:.1%}")

    # ─── Analysis 6: BGC-level best-window score distribution ───
    print("\n=== Analysis 6: Per-BGC best (lowest) is_neg score ===")
    bgc_best_scores = {}
    for bid in test_id_bgcs:
        wins = bgc_windows.get(bid, [])
        if wins:
            best_score = min(w["is_neg_score"] for w in wins)
            n_wins = len(wins)
        else:
            best_score = 1.0  # no positive windows at all
            n_wins = 0
        bgc_best_scores[bid] = {
            "best_is_neg": best_score,
            "n_positive_windows": n_wins,
            "detected": best_score < is_neg_thr,
            "type": test_id_bgcs[bid]["bgc_types"],
            "length": int(test_id_bgcs[bid]["bgc_length"]),
        }

    detected = [v for v in bgc_best_scores.values() if v["detected"]]
    missed = [v for v in bgc_best_scores.values() if not v["detected"]]

    if missed:
        missed_scores = [v["best_is_neg"] for v in missed]
        print(f"  Detected BGCs: {len(detected)}, Missed BGCs: {len(missed)}")
        print(f"  Missed BGCs best score - mean: {np.mean(missed_scores):.4f}, "
              f"median: {np.median(missed_scores):.4f}")

        # How many missed BGCs are "close" (would be detected at threshold 0.2)?
        close = sum(1 for s in missed_scores if s < 0.2)
        medium = sum(1 for s in missed_scores if 0.2 <= s < 0.5)
        far = sum(1 for s in missed_scores if s >= 0.5)
        print(f"  Close to threshold (<0.2): {close} ({close/len(missed):.1%})")
        print(f"  Medium (0.2-0.5): {medium} ({medium/len(missed):.1%})")
        print(f"  Far from threshold (≥0.5): {far} ({far/len(missed):.1%})")

    # ─── Save all results ───
    results = {
        "threshold": is_neg_thr,
        "n_test_id_bgcs": len(test_id_bgcs),
        "n_detected": len(detected),
        "n_missed": len(missed),
        "overall_bgc_miss_rate": round(len(missed) / len(test_id_bgcs), 4) if test_id_bgcs else 0,
        "by_length": length_results,
        "by_type": type_results,
        "type_x_length": cross_table,
        "by_overlap_ratio": overlap_results,
        "fn_score_distribution": fn_stats,
        "by_truncation": trunc_results,
        "bgc_level_score_summary": {
            "n_close_to_threshold": close if missed else 0,
            "n_medium": medium if missed else 0,
            "n_far": far if missed else 0,
        },
    }

    out_path = os.path.join(args.output_dir, "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save per-BGC details for further investigation
    bgc_details_path = os.path.join(args.output_dir, "bgc_detection_details.json")
    with open(bgc_details_path, "w") as f:
        json.dump(bgc_best_scores, f, indent=2)
    print(f"Per-BGC details saved to {bgc_details_path}")


if __name__ == "__main__":
    main()
