#!/usr/bin/env python3
"""
WS2-T1: Calibration Benchmark
===============================
Estimate how much recall can be recovered without retraining,
using post-hoc calibration on frozen model outputs.

Methods:
  1. Temperature scaling (single parameter)
  2. Platt scaling (per-head logistic regression)
  3. Threshold sweep (no calibration, just threshold grid)

For each method:
  - Fit on val set
  - Evaluate on test_id and test_ood
  - Report safety metrics (recall, precision) and discovery metrics
  - Compare against uncalibrated baseline

Usage:
  python ws2_calibration.py \
    --ckpt /data/muskliu/np_master/data/stage3_results/B_contrastive/best_B_contrastive_s0.pt \
    --emb /data/muskliu/np_master/data/embeddings/embeddings.npy \
    --meta /data/muskliu/np_master/data/windows/windows_metadata.csv \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --out /data/muskliu/np_master/data/stage4_results/ws2_calibration
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, log_loss
from scipy.optimize import minimize_scalar, minimize

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
ALL_COLS = CLASSES + ["is_negative"]


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
    preds_logits = []
    preds_probs = []
    ds = TensorDataset(torch.from_numpy(embeddings).float())
    loader = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for (x,) in loader:
            _, pred_logits, _ = model(x.to(device))
            preds_logits.append(pred_logits.cpu().numpy())
            preds_probs.append(torch.sigmoid(pred_logits).cpu().numpy())
    return np.vstack(preds_logits), np.vstack(preds_probs)


# ─── calibration methods ───

def temperature_scaling_fit(logits, labels_binary, score_idx=7):
    """
    Fit a single temperature parameter on the is_negative logit.
    Minimizes NLL on val set.
    """
    logits_head = logits[:, score_idx]
    # For is_negative: label=1 means negative (background), label=0 means BGC
    # We want to calibrate so that sigmoid(logit/T) gives well-calibrated probabilities

    def nll(T):
        T = max(T, 0.01)
        scaled = 1.0 / (1.0 + np.exp(-logits_head / T))
        # Clip for numerical stability
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return -np.mean(labels_binary * np.log(scaled) + (1 - labels_binary) * np.log(1 - scaled))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def platt_scaling_fit(logits, labels_binary, score_idx=7):
    """
    Fit Platt scaling: sigmoid(a * logit + b) on val set.
    """
    logits_head = logits[:, score_idx]

    def nll(params):
        a, b = params
        scaled = 1.0 / (1.0 + np.exp(-(a * logits_head + b)))
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return -np.mean(labels_binary * np.log(scaled) + (1 - labels_binary) * np.log(1 - scaled))

    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return result.x


def apply_temperature(logits, T, score_idx=7):
    """Apply temperature scaling and return calibrated probabilities."""
    calibrated = logits.copy()
    calibrated[:, score_idx] = logits[:, score_idx] / T
    return 1.0 / (1.0 + np.exp(-calibrated))


def apply_platt(logits, a, b, score_idx=7):
    """Apply Platt scaling and return calibrated probabilities."""
    calibrated_logits = logits.copy()
    calibrated_logits[:, score_idx] = a * logits[:, score_idx] + b
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


# ─── region calling (same as ws1) ───

def call_regions(windows, scores, threshold, min_length=3000):
    regions = []
    cur_region = None
    for w, score in zip(windows, scores):
        is_bgc = score >= threshold
        if is_bgc:
            contig = w["contig_id"]
            ws = int(w["window_start"])
            we = int(w["window_end"])
            if cur_region and cur_region["contig"] == contig and ws <= cur_region["end"] + 1:
                cur_region["end"] = max(cur_region["end"], we)
                cur_region["n_windows"] += 1
                cur_region["max_score"] = max(cur_region["max_score"], score)
            else:
                if cur_region and cur_region["end"] - cur_region["start"] >= min_length:
                    regions.append(cur_region)
                cur_region = {
                    "contig": contig, "start": ws, "end": we, "n_windows": 1,
                    "max_score": score, "genome_id": w["genome_id"],
                }
        else:
            if cur_region and cur_region["end"] - cur_region["start"] >= min_length:
                regions.append(cur_region)
            cur_region = None
    if cur_region and cur_region["end"] - cur_region["start"] >= min_length:
        regions.append(cur_region)
    for reg in regions:
        reg["length"] = reg["end"] - reg["start"]
    return regions


def match_and_evaluate(regions, as_bgcs_by_loc, split_genomes):
    """Compute recall and precision for a set of regions."""
    n_as_total = 0
    matched_as = set()
    for (gid, cid), bgc_list in as_bgcs_by_loc.items():
        if gid not in split_genomes:
            continue
        n_as_total += len(bgc_list)
        for ab in bgc_list:
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            ab_len = ab_e - ab_s
            ab_key = (gid, ab_s, ab_e)
            for reg in regions:
                if reg["genome_id"] != gid or reg["contig"] != ab["contig_id"]:
                    continue
                ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
                min_len = min(ab_len, reg["length"])
                if min_len > 0 and ov / min_len >= 0.5:
                    matched_as.add(ab_key)
                    break

    recall = len(matched_as) / max(1, n_as_total)
    precision = len(matched_as) / max(1, len(regions))
    return {
        "n_antismash": n_as_total,
        "n_matched": len(matched_as),
        "recall": round(recall, 4),
        "n_regions": len(regions),
        "precision_as_count": round(precision, 4),
    }


# ─── threshold sweep for recall-precision curve ───

def sweep_thresholds(windows, scores, as_bgcs_by_loc, split_genomes, n_points=20):
    """Sweep detection threshold and report recall/precision at each point."""
    thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), n_points)
    curve = []
    for t in thresholds:
        regions = call_regions(windows, scores, t)
        metrics = match_and_evaluate(regions, as_bgcs_by_loc, split_genomes)
        metrics["threshold"] = round(float(t), 6)
        curve.append(metrics)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--bgc-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    print("Loading model...")
    model = load_model(args.ckpt, device)

    print("Loading data...")
    emb = np.load(args.emb)
    meta = []
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            meta.append(row)
    assert len(meta) == emb.shape[0]

    print("Loading BGC annotations...")
    as_bgcs_by_loc = defaultdict(list)
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            as_bgcs_by_loc[(row["genome_id"], row["contig_id"])].append(row)

    # ─── predict ───
    print("Predicting on val...")
    val_mask = np.array([r["split"] == "val" and r["label"] != "ambiguous" for r in meta])
    val_emb = emb[val_mask]
    val_meta = [r for r in meta if r["split"] == "val" and r["label"] != "ambiguous"]
    val_logits, val_probs = predict_batched(model, val_emb, device)
    val_is_neg_labels = np.array([1 if r["label"] == "negative" else 0 for r in val_meta], dtype=np.float32)

    # ─── fit calibration ───
    print("\n=== Fitting calibration ===")

    # Temperature scaling
    T = temperature_scaling_fit(val_logits, val_is_neg_labels, score_idx=7)
    print(f"  Temperature: T={T:.4f}")

    # Platt scaling
    a, b = platt_scaling_fit(val_logits, val_is_neg_labels, score_idx=7)
    print(f"  Platt: a={a:.4f}, b={b:.4f}")

    # ─── define calibration variants ───
    calibration_variants = {
        "uncalibrated": {
            "description": "Original sigmoid output (baseline)",
            "apply": lambda logits: 1.0 / (1.0 + np.exp(-logits)),
        },
        "temperature": {
            "description": f"Temperature scaling T={T:.4f}",
            "apply": lambda logits: apply_temperature(logits, T),
        },
        "platt": {
            "description": f"Platt scaling a={a:.4f} b={b:.4f}",
            "apply": lambda logits: apply_platt(logits, a, b),
        },
    }

    # ─── evaluate each variant ───
    all_results = {}
    for split in ["test_id", "test_ood"]:
        print(f"\n=== {split} ===")
        s_mask = np.array([r["split"] == split and r["label"] != "ambiguous" for r in meta])
        s_emb = emb[s_mask]
        s_meta = [r for r in meta if r["split"] == split and r["label"] != "ambiguous"]
        s_logits, _ = predict_batched(model, s_emb, device)
        split_genomes = set(w["genome_id"] for w in s_meta)

        split_results = {}
        for vname, vdef in calibration_variants.items():
            print(f"\n  --- {vname}: {vdef['description']} ---")
            cal_probs = vdef["apply"](s_logits)

            # Detection score: 1 - is_negative
            detection_scores = 1.0 - cal_probs[:, 7]

            # Find F1-optimal threshold on val for this calibration
            val_cal_probs = vdef["apply"](val_logits)
            val_det_scores = 1.0 - val_cal_probs[:, 7]
            val_is_bgc = 1 - val_is_neg_labels
            prec, rec, thr = precision_recall_curve(val_is_bgc.astype(int), val_det_scores)
            f1s = 2 * prec * rec / (prec + rec + 1e-10)
            best_idx = np.argmax(f1s)
            opt_threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5

            # Prepare sorted windows
            windows_with_meta = list(s_meta)
            order = sorted(range(len(windows_with_meta)),
                           key=lambda i: (windows_with_meta[i]["genome_id"],
                                          windows_with_meta[i]["contig_id"],
                                          int(windows_with_meta[i]["window_start"])))
            windows_sorted = [windows_with_meta[i] for i in order]
            scores_sorted = detection_scores[order]

            # Region calling at F1-optimal threshold
            regions = call_regions(windows_sorted, scores_sorted, opt_threshold)
            metrics = match_and_evaluate(regions, as_bgcs_by_loc, split_genomes)

            # Threshold sweep for recall-precision curve
            curve = sweep_thresholds(windows_sorted, scores_sorted, as_bgcs_by_loc, split_genomes)

            # Find recall at specific precision targets
            recall_at_prec = {}
            for target_p in [0.30, 0.40, 0.50]:
                best_r = 0
                for pt in curve:
                    if pt["precision_as_count"] >= target_p and pt["recall"] > best_r:
                        best_r = pt["recall"]
                recall_at_prec[f"recall_at_prec_{target_p}"] = round(best_r, 4)

            # Find precision at specific recall targets
            prec_at_recall = {}
            for target_r in [0.40, 0.50, 0.60]:
                best_p = 0
                for pt in curve:
                    if pt["recall"] >= target_r and pt["precision_as_count"] > best_p:
                        best_p = pt["precision_as_count"]
                prec_at_recall[f"prec_at_recall_{target_r}"] = round(best_p, 4)

            split_results[vname] = {
                "description": vdef["description"],
                "optimal_threshold": round(opt_threshold, 6),
                **metrics,
                **recall_at_prec,
                **prec_at_recall,
                "recall_precision_curve": curve,
            }

            print(f"    Threshold: {opt_threshold:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    Precision: {metrics['precision_as_count']:.4f}")
            print(f"    Regions:   {metrics['n_regions']}")
            for k, v in {**recall_at_prec, **prec_at_recall}.items():
                print(f"    {k}: {v}")

        all_results[split] = split_results

    # ─── comparison table ───
    print("\n\n" + "="*80)
    print("CALIBRATION COMPARISON")
    print("="*80)
    for split in ["test_id", "test_ood"]:
        print(f"\n{split}:")
        print(f"  {'Method':>15s} | {'Threshold':>10s} | {'Recall':>8s} | {'Prec':>8s} | {'Regions':>8s} | {'R@P0.3':>8s} | {'R@P0.4':>8s} | {'P@R0.5':>8s}")
        print("  " + "-"*90)
        for vname in calibration_variants:
            r = all_results[split][vname]
            print(f"  {vname:>15s} | {r['optimal_threshold']:>10.4f} | {r['recall']:>8.4f} | {r['precision_as_count']:>8.4f} | {r['n_regions']:>8d} | "
                  f"{r.get('recall_at_prec_0.3', 0):>8.4f} | {r.get('recall_at_prec_0.4', 0):>8.4f} | {r.get('prec_at_recall_0.5', 0):>8.4f}")

    # ─── Gate B assessment ───
    baseline_recall = all_results["test_id"]["uncalibrated"]["recall"]
    best_method = max(calibration_variants.keys(),
                      key=lambda m: all_results["test_id"][m]["recall"])
    best_recall = all_results["test_id"][best_method]["recall"]

    gate_b = {
        "baseline_recall": baseline_recall,
        "best_method": best_method,
        "best_recall": best_recall,
        "delta": round(best_recall - baseline_recall, 4),
        "note": "Gate B evaluates whether calibration/prior mismatch is a first-order problem. "
                "Reduced-proxy result (ambiguous excluded). "
                "Substantial improvement suggests 50/50 sampling is a major issue."
    }

    # ─── save ───
    output = {
        "task_id": "WS2-T1",
        "checkpoint": args.ckpt,
        "evaluation_type": "reduced_proxy (ambiguous windows excluded)",
        "calibration_params": {
            "temperature": round(T, 4),
            "platt_a": round(a, 4),
            "platt_b": round(b, 4),
        },
        "results": all_results,
        "gate_b": gate_b,
    }

    out_path = os.path.join(args.out, "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
