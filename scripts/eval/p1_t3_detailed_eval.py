#!/usr/bin/env python3
"""
P1-T3: Detailed evaluation for edge-padding — short-BGC bucket analysis + threshold sweep.

Reports:
  1. Per-length-bucket recall (before vs after)
  2. Truncated vs non-truncated recall
  3. Threshold operating points (recall≈60/70/80%)

Usage:
  python p1_t3_detailed_eval.py \
    --ckpt-new /data/muskliu/np_master/data/stage3_results/G_edgepad/best_G_edgepad.pt \
    --emb-new /data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy \
    --meta-new /data/muskliu/np_master/data/windows/edge_padding/windows_metadata_edgepad_filtered.csv \
    --baseline-eval /data/muskliu/np_master/data/stage4_results/ws3_cap08_eval/score_swap_results.json \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --out /data/muskliu/np_master/data/stage4_results/p1_t3_edgepad_eval
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve

CLASSES = ["Polyketide","NRP","Terpene","RiPP","Saccharide","Alkaloid","Other"]

LENGTH_BUCKETS = [
    ("<5kb", 0, 5000),
    ("5-10kb", 5000, 10000),
    ("10-20kb", 10000, 20000),
    ("20-50kb", 20000, 50000),
    (">50kb", 50000, float("inf")),
]

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

def predict_batched(model, embeddings, device, batch_size=512):
    preds = []
    ds = TensorDataset(torch.from_numpy(embeddings).float())
    for (x,) in DataLoader(ds, batch_size=batch_size):
        with torch.no_grad():
            _, pred, _ = model(x.to(device))
            preds.append(torch.sigmoid(pred).cpu().numpy())
    return np.vstack(preds)

def call_and_match(windows, scores, threshold, as_bgcs_by_loc, min_length=3000):
    """Call regions, match to antiSMASH, return per-BGC match status."""
    # Sort by genome/contig/position
    order = sorted(range(len(windows)), key=lambda i: (
        windows[i]["genome_id"], windows[i]["contig_id"], int(windows[i]["window_start"])))
    ws = [windows[i] for i in order]
    sc = scores[order]

    # Call regions
    regions = []
    cur = None
    for i, (w, s) in enumerate(zip(ws, sc)):
        if s >= threshold:
            contig = w["contig_id"]; wstart = int(w["window_start"]); wend = int(w["window_end"])
            if cur and cur["contig"] == contig and wstart <= cur["end"] + 1:
                cur["end"] = max(cur["end"], wend); cur["n_windows"] += 1
            else:
                if cur and cur["end"]-cur["start"] >= min_length: regions.append(cur)
                cur = {"contig": contig, "start": wstart, "end": wend, "n_windows": 1, "genome_id": w["genome_id"]}
        else:
            if cur and cur["end"]-cur["start"] >= min_length: regions.append(cur)
            cur = None
    if cur and cur["end"]-cur["start"] >= min_length: regions.append(cur)
    for r in regions: r["length"] = r["end"]-r["start"]

    # Match each antiSMASH BGC
    split_genomes = set(w["genome_id"] for w in windows)
    bgc_status = {}  # (genome_id, start, end) -> matched bool
    for (gid, cid), bgc_list in as_bgcs_by_loc.items():
        if gid not in split_genomes: continue
        for ab in bgc_list:
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            ab_len = ab_e - ab_s
            ab_key = (gid, ab_s, ab_e)
            best_ov = 0
            for reg in regions:
                if reg["genome_id"] != gid or reg["contig"] != cid: continue
                ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
                min_len = min(ab_len, reg["length"])
                if min_len > 0: best_ov = max(best_ov, ov/min_len)
            bgc_status[ab_key] = best_ov >= 0.5

    # Prediction-level precision
    n_pred_match = 0
    for reg in regions:
        gid = reg["genome_id"]; cid = reg["contig"]
        for ab in as_bgcs_by_loc.get((gid, cid), []):
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
            min_len = min(ab_e-ab_s, reg["length"])
            if min_len > 0 and ov/min_len >= 0.5:
                n_pred_match += 1; break

    return bgc_status, len(regions), n_pred_match

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-new", required=True)
    ap.add_argument("--emb-new", required=True)
    ap.add_argument("--meta-new", required=True)
    ap.add_argument("--baseline-eval", required=True, help="F_cap08 score_swap_results.json")
    ap.add_argument("--bgc-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ─── Load BGC annotations with length/truncation info ───
    print("Loading BGC annotations...")
    as_bgcs_by_loc = defaultdict(list)
    bgc_info = {}  # (gid, start, end) -> {length, truncated, types, phylum}
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            key = (row["genome_id"], row["contig_id"])
            as_bgcs_by_loc[key].append(row)
            bgc_key = (row["genome_id"], int(row["start_bp"]), int(row["end_bp"]))
            bgc_info[bgc_key] = {
                "length": int(row["end_bp"]) - int(row["start_bp"]),
                "truncated": row.get("is_truncated","0") == "1",
                "types": row["bgc_types"],
                "phylum": row.get("phylum",""),
                "genome_id": row["genome_id"],
            }

    # ─── Load new model + data ───
    print("Loading model and data...")
    model = load_model(args.ckpt_new, device)
    emb = np.load(args.emb_new)
    meta = list(csv.DictReader(open(args.meta_new)))
    assert len(meta) == emb.shape[0]

    # ─── Predict on val + test splits ───
    print("Predicting...")
    val_mask = np.array([r["split"]=="val" and r["label"]!="ambiguous" for r in meta])
    val_preds = predict_batched(model, emb[val_mask], device)
    val_meta = [r for r in meta if r["split"]=="val" and r["label"]!="ambiguous"]
    val_is_bgc = np.array([1 if r["label"]=="positive" else 0 for r in val_meta])
    val_scores = val_preds[:,:7].max(axis=1)  # score_maxtype

    # F1-optimal threshold
    prec_arr, rec_arr, thr_arr = precision_recall_curve(val_is_bgc, val_scores)
    f1s = 2*prec_arr*rec_arr/(prec_arr+rec_arr+1e-10)
    best_idx = np.argmax(f1s)
    f1_threshold = float(thr_arr[best_idx]) if best_idx < len(thr_arr) else 0.5
    print(f"  F1-optimal threshold: {f1_threshold:.4f}")

    # ─── Threshold sweep on val ───
    print("\nThreshold sweep (val, score_maxtype)...")
    sweep_thresholds = np.percentile(val_scores, np.linspace(1, 99, 50))
    sweep_results = []
    for thr in sweep_thresholds:
        tp = ((val_scores >= thr) & (val_is_bgc == 1)).sum()
        fp = ((val_scores >= thr) & (val_is_bgc == 0)).sum()
        fn = ((val_scores < thr) & (val_is_bgc == 1)).sum()
        p = tp / max(1, tp+fp); r = tp / max(1, tp+fn)
        sweep_results.append({"threshold": float(thr), "precision": float(p), "recall": float(r)})

    # Find recall-targeted thresholds
    recall_targets = {0.60: None, 0.70: None, 0.80: None}
    for sr in sorted(sweep_results, key=lambda x: x["threshold"], reverse=True):
        for target in recall_targets:
            if recall_targets[target] is None and sr["recall"] >= target:
                recall_targets[target] = sr

    print("  Recall-targeted operating points (val):")
    for target, sr in sorted(recall_targets.items()):
        if sr:
            print(f"    recall≈{target:.0%}: threshold={sr['threshold']:.4f} P={sr['precision']:.3f} R={sr['recall']:.3f}")

    # ─── Evaluate on test_id and test_ood at F1-optimal ───
    results_by_split = {}
    for split in ["test_id", "test_ood"]:
        print(f"\n=== {split} (score_maxtype, full-window, threshold={f1_threshold:.4f}) ===")
        s_mask = np.array([r["split"]==split for r in meta])
        s_meta = [r for r in meta if r["split"]==split]
        s_preds = predict_batched(model, emb[s_mask], device)
        s_scores = s_preds[:,:7].max(axis=1)

        bgc_matched, n_regions, n_pred_match = call_and_match(s_meta, s_scores, f1_threshold, as_bgcs_by_loc)

        # Overall metrics
        split_genomes = set(r["genome_id"] for r in s_meta)
        total_bgcs = sum(1 for k,v in bgc_info.items() if v["genome_id"] in split_genomes)
        n_matched = sum(1 for v in bgc_matched.values() if v)
        recall = n_matched / max(1, total_bgcs)
        precision = n_pred_match / max(1, n_regions)
        print(f"  Overall: recall={recall:.4f} ({n_matched}/{total_bgcs}), P(pred)={precision:.4f}, regions={n_regions}")

        # Per-length-bucket
        bucket_results = {}
        for bname, blo, bhi in LENGTH_BUCKETS:
            bucket_bgcs = [(k,v) for k,v in bgc_info.items()
                          if v["genome_id"] in split_genomes and blo <= v["length"] < bhi]
            n_total = len(bucket_bgcs)
            n_hit = sum(1 for k,_ in bucket_bgcs if bgc_matched.get(k, False))
            bucket_results[bname] = {
                "n_total": n_total,
                "n_matched": n_hit,
                "recall": round(n_hit/max(1,n_total), 4),
            }
            print(f"  {bname:>8s}: {n_hit}/{n_total} = {bucket_results[bname]['recall']:.3f}")

        # Truncated vs non-truncated
        trunc_bgcs = [(k,v) for k,v in bgc_info.items() if v["genome_id"] in split_genomes and v["truncated"]]
        nontrunc_bgcs = [(k,v) for k,v in bgc_info.items() if v["genome_id"] in split_genomes and not v["truncated"]]
        trunc_hit = sum(1 for k,_ in trunc_bgcs if bgc_matched.get(k, False))
        nontrunc_hit = sum(1 for k,_ in nontrunc_bgcs if bgc_matched.get(k, False))
        trunc_results = {
            "truncated": {"n": len(trunc_bgcs), "matched": trunc_hit,
                         "recall": round(trunc_hit/max(1,len(trunc_bgcs)),4)},
            "non_truncated": {"n": len(nontrunc_bgcs), "matched": nontrunc_hit,
                             "recall": round(nontrunc_hit/max(1,len(nontrunc_bgcs)),4)},
        }
        print(f"  Truncated:     {trunc_hit}/{len(trunc_bgcs)} = {trunc_results['truncated']['recall']:.3f}")
        print(f"  Non-truncated: {nontrunc_hit}/{len(nontrunc_bgcs)} = {trunc_results['non_truncated']['recall']:.3f}")

        # Per-type recall
        type_results = {}
        for t in CLASSES:
            t_bgcs = [(k,v) for k,v in bgc_info.items()
                      if v["genome_id"] in split_genomes and t in v["types"].split(";")]
            t_hit = sum(1 for k,_ in t_bgcs if bgc_matched.get(k, False))
            type_results[t] = {"n": len(t_bgcs), "matched": t_hit,
                              "recall": round(t_hit/max(1,len(t_bgcs)),4)}

        results_by_split[split] = {
            "recall": round(recall, 4),
            "precision_pred": round(precision, 4),
            "n_regions": n_regions,
            "n_matched": n_matched,
            "n_total_bgcs": total_bgcs,
            "by_length_bucket": bucket_results,
            "by_truncation": trunc_results,
            "by_type": type_results,
        }

    # ─── Load baseline for comparison ───
    print("\n\n=== BEFORE vs AFTER COMPARISON ===")
    # We need baseline per-bucket data. Run the same analysis on baseline eval.
    # Actually, baseline eval doesn't have per-bucket data in the JSON.
    # We'll compare what we can from the new eval only, and note the baseline overall.
    with open(args.baseline_eval) as f:
        baseline = json.load(f)
    baseline_recall_id = baseline["results"]["test_id"]["score_maxtype"]["recall"]
    baseline_prec_id = baseline["results"]["test_id"]["score_maxtype"]["precision_pred_level"]
    baseline_recall_ood = baseline["results"]["test_ood"]["score_maxtype"]["recall"]
    baseline_prec_ood = baseline["results"]["test_ood"]["score_maxtype"]["precision_pred_level"]

    new_recall_id = results_by_split["test_id"]["recall"]
    new_prec_id = results_by_split["test_id"]["precision_pred"]
    new_recall_ood = results_by_split["test_ood"]["recall"]
    new_prec_ood = results_by_split["test_ood"]["precision_pred"]

    print(f"  {'':>20s} | {'F_cap08':>10s} | {'G_edgepad':>10s} | {'Delta':>8s}")
    print("  " + "-"*58)
    print(f"  {'test_id recall':>20s} | {baseline_recall_id:>10.4f} | {new_recall_id:>10.4f} | {new_recall_id-baseline_recall_id:>+8.4f}")
    print(f"  {'test_id P(pred)':>20s} | {baseline_prec_id:>10.4f} | {new_prec_id:>10.4f} | {new_prec_id-baseline_prec_id:>+8.4f}")
    print(f"  {'test_ood recall':>20s} | {baseline_recall_ood:>10.4f} | {new_recall_ood:>10.4f} | {new_recall_ood-baseline_recall_ood:>+8.4f}")
    print(f"  {'test_ood P(pred)':>20s} | {baseline_prec_ood:>10.4f} | {new_prec_ood:>10.4f} | {new_prec_ood-baseline_prec_ood:>+8.4f}")

    # ─── Save ───
    output = {
        "task_id": "P1-T3-detailed",
        "checkpoint": args.ckpt_new,
        "detection_score": "score_maxtype",
        "evaluation_mode": "full_window",
        "f1_optimal_threshold": f1_threshold,
        "recall_targeted_thresholds": {
            f"recall_{int(t*100)}": v for t,v in recall_targets.items() if v
        },
        "threshold_sweep": sweep_results,
        "results": results_by_split,
        "comparison_vs_baseline": {
            "baseline": "F_cap08",
            "test_id": {
                "recall": {"before": baseline_recall_id, "after": new_recall_id,
                          "delta": round(new_recall_id-baseline_recall_id, 4)},
                "precision": {"before": baseline_prec_id, "after": new_prec_id,
                             "delta": round(new_prec_id-baseline_prec_id, 4)},
            },
            "test_ood": {
                "recall": {"before": baseline_recall_ood, "after": new_recall_ood,
                          "delta": round(new_recall_ood-baseline_recall_ood, 4)},
                "precision": {"before": baseline_prec_ood, "after": new_prec_ood,
                             "delta": round(new_prec_ood-baseline_prec_ood, 4)},
            },
        },
    }
    out_path = os.path.join(args.out, "p1_t3_detailed_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
