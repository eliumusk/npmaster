#!/usr/bin/env python3
"""
WS1-T1: Score Swap Benchmark
=============================
Test whether score mismatch is a first-order failure mode.

Runs region calling with 3 detection scores on the SAME checkpoint:
  1. score_is_neg:  1 - is_negative  (current Stage 4 score)
  2. score_maxtype: max(type_probs)  (current Stage 3 BGC binary score)
  3. score_fused:   0.5*(1-is_neg) + 0.5*max(type_probs)

For each score:
  - Find optimal threshold on val set (F1-based, same policy as current)
  - Run region calling on test_id and test_ood
  - Report region-level recall, type concordance
  - Report BOTH antiSMASH-count-based precision AND strict prediction-level precision
  - Report novel candidate counts and confidence breakdown
  - Classify unmatched predictions into: boundary-expansion / clean antiSMASH-missed

WS4-T2 (Prediction-level precision) is integrated here.

Usage:
  python ws1_score_swap.py \
    --ckpt /data/muskliu/np_master/data/stage3_results/B_contrastive/best_B_contrastive_s0.pt \
    --emb /data/muskliu/np_master/data/embeddings/embeddings.npy \
    --meta /data/muskliu/np_master/data/windows/windows_metadata.csv \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --out /data/muskliu/np_master/data/stage4_results/ws1_score_swap
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
from sklearn.metrics import precision_recall_curve

# ─── constants ───

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
ALL_COLS = CLASSES + ["is_negative"]

SCORE_VARIANTS = {
    "score_is_neg": {
        "description": "1 - is_negative (current Stage 4 score)",
        "compute": lambda preds: 1.0 - preds[:, 7],
    },
    "score_maxtype": {
        "description": "max(type_probs) (current Stage 3 BGC binary score)",
        "compute": lambda preds: preds[:, :7].max(axis=1),
    },
    "score_fused": {
        "description": "0.5*(1-is_neg) + 0.5*max(type_probs)",
        "compute": lambda preds: 0.5 * (1.0 - preds[:, 7]) + 0.5 * preds[:, :7].max(axis=1),
    },
}


# ─── model ───

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


# ─── threshold selection ───

def find_optimal_threshold(val_scores, val_labels_binary):
    """Find F1-optimal threshold on val set."""
    prec, rec, thr = precision_recall_curve(val_labels_binary, val_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best = np.argmax(f1s)
    return {
        "threshold": float(thr[best]) if best < len(thr) else 0.5,
        "precision": float(prec[best]),
        "recall": float(rec[best]),
        "f1": float(f1s[best]),
    }


# ─── region calling ───

def call_regions(windows, scores, threshold, min_length=3000):
    """
    Call regions from window-level scores.
    windows: list of dicts with contig_id, window_start, window_end, + 8-dim preds
    scores: 1D array of detection scores (higher = more BGC-like)
    threshold: score threshold
    min_length: minimum region length in bp
    """
    regions = []
    cur_region = None

    for i, (w, score) in enumerate(zip(windows, scores)):
        is_bgc = score >= threshold
        if is_bgc:
            contig = w["contig_id"]
            ws = int(w["window_start"])
            we = int(w["window_end"])
            if cur_region and cur_region["contig"] == contig and ws <= cur_region["end"] + 1:
                cur_region["end"] = max(cur_region["end"], we)
                cur_region["n_windows"] += 1
                for j, c in enumerate(CLASSES):
                    cur_region["probs"][c] = max(cur_region["probs"][c], w["preds"][j])
                cur_region["max_score"] = max(cur_region["max_score"], score)
            else:
                if cur_region and cur_region["end"] - cur_region["start"] >= min_length:
                    regions.append(cur_region)
                cur_region = {
                    "contig": contig, "start": ws, "end": we, "n_windows": 1,
                    "probs": {c: w["preds"][j] for j, c in enumerate(CLASSES)},
                    "max_score": score,
                    "genome_id": w["genome_id"],
                }
        else:
            if cur_region and cur_region["end"] - cur_region["start"] >= min_length:
                regions.append(cur_region)
            cur_region = None

    if cur_region and cur_region["end"] - cur_region["start"] >= min_length:
        regions.append(cur_region)

    for reg in regions:
        reg["primary_type"] = max(CLASSES, key=lambda c: reg["probs"][c])
        reg["length"] = reg["end"] - reg["start"]

    return regions


# ─── matching ───

def match_regions(regions, as_bgcs_by_loc, overlap_threshold=0.5):
    """
    Match predicted regions against antiSMASH BGCs.
    Returns: matched_as, matched_pred, match_pairs, novel_candidates
    """
    matched_as = set()
    matched_pred = set()
    match_pairs = []

    # group regions by genome
    regions_by_genome = defaultdict(list)
    for ri, reg in enumerate(regions):
        regions_by_genome[reg["genome_id"]].append((ri, reg))

    for genome_id, genome_regions in regions_by_genome.items():
        # get all antiSMASH BGCs for this genome's contigs
        contigs_in_genome = set(reg["contig"] for _, reg in genome_regions)
        genome_as = []
        for contig in contigs_in_genome:
            genome_as.extend([(ab_i, ab) for ab_i, ab in enumerate(as_bgcs_by_loc.get((genome_id, contig), []))])

        for ab_i, ab in genome_as:
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            ab_len = ab_e - ab_s
            ab_key = (genome_id, ab_s, ab_e)
            best_ov = 0
            best_ri = -1

            for ri, reg in genome_regions:
                if reg["contig"] != ab["contig_id"]:
                    continue
                ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
                min_len = min(ab_len, reg["length"])
                if min_len > 0 and ov / min_len > best_ov:
                    best_ov = ov / min_len
                    best_ri = ri

            if best_ov >= overlap_threshold:
                matched_as.add(ab_key)
                matched_pred.add(best_ri)
                pred_type = regions[best_ri]["primary_type"]
                ab_types = ab["bgc_types"].split(";")
                match_pairs.append({
                    "genome_id": genome_id,
                    "as_key": ab_key,
                    "pred_ri": best_ri,
                    "overlap_frac": best_ov,
                    "type_concordant": pred_type in ab_types,
                })

    return matched_as, matched_pred, match_pairs


def compute_prediction_level_precision(regions, as_bgcs_by_loc, overlap_threshold=0.5):
    """
    WS4-T2: Strict prediction-level precision.

    For each predicted region, check if it overlaps with ANY antiSMASH BGC
    at the required threshold. This is 1-to-many on the prediction side:
    each prediction is independently evaluated.

    Returns: n_pred_with_match / n_pred_total
    """
    n_with_match = 0
    for reg in regions:
        genome_id = reg["genome_id"]
        contig = reg["contig"]
        as_list = as_bgcs_by_loc.get((genome_id, contig), [])
        has_match = False
        for ab in as_list:
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            ab_len = ab_e - ab_s
            ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
            min_len = min(ab_len, reg["length"])
            if min_len > 0 and ov / min_len >= overlap_threshold:
                has_match = True
                break
        if has_match:
            n_with_match += 1
    return n_with_match, len(regions)


# ─── novel candidate classification ───

def classify_unmatched(regions, matched_pred, as_bgcs_by_loc):
    """
    Classify unmatched predictions into:
    - boundary_expansion: within 5kb of an antiSMASH BGC
    - clean_missed: >20kb from any antiSMASH BGC on same contig
    - ambiguous_unmatched: in between
    """
    results = []
    for ri, reg in enumerate(regions):
        if ri in matched_pred:
            continue
        genome_id = reg["genome_id"]
        contig = reg["contig"]
        as_list = as_bgcs_by_loc.get((genome_id, contig), [])

        # compute interval distance (overlap-aware)
        min_dist = float("inf")
        for ab in as_list:
            ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
            # interval distance: 0 if overlapping, else gap size
            if reg["start"] >= ab_e:
                d = reg["start"] - ab_e
            elif reg["end"] <= ab_s:
                d = ab_s - reg["end"]
            else:
                d = 0  # overlapping
            min_dist = min(min_dist, d)

        if min_dist == float("inf"):
            # no antiSMASH BGCs on this contig at all
            category = "clean_missed"
        elif min_dist <= 5000:
            category = "boundary_expansion"
        elif min_dist > 20000:
            category = "clean_missed"
        else:
            category = "ambiguous_unmatched"

        results.append({
            "region_idx": ri,
            "genome_id": genome_id,
            "contig": contig,
            "start": reg["start"],
            "end": reg["end"],
            "length": reg["length"],
            "primary_type": reg["primary_type"],
            "max_score": reg["max_score"],
            "dist_to_nearest_bgc": min_dist if min_dist < 1e9 else -1,
            "category": category,
        })
    return results


# ─── main evaluation ───

def evaluate_score_variant(
    score_name, score_func, threshold_info,
    split_windows, split_preds, as_bgcs_by_loc
):
    """Run full evaluation for one score variant on one split."""
    scores = score_func(split_preds)
    threshold = threshold_info["threshold"]

    # prepare windows with predictions
    windows_with_preds = []
    for i, w in enumerate(split_windows):
        ww = dict(w)
        ww["preds"] = [float(split_preds[i, j]) for j in range(8)]
        windows_with_preds.append(ww)

    # sort by contig + position
    order = sorted(range(len(windows_with_preds)),
                   key=lambda i: (windows_with_preds[i]["genome_id"],
                                  windows_with_preds[i]["contig_id"],
                                  int(windows_with_preds[i]["window_start"])))
    windows_sorted = [windows_with_preds[i] for i in order]
    scores_sorted = scores[order]

    # call regions
    regions = call_regions(windows_sorted, scores_sorted, threshold)

    # count antiSMASH BGCs in this split
    split_genomes = set(w["genome_id"] for w in split_windows)
    n_as_total = 0
    for (gid, cid), bgc_list in as_bgcs_by_loc.items():
        if gid in split_genomes:
            n_as_total += len(bgc_list)

    # match
    matched_as, matched_pred, match_pairs = match_regions(regions, as_bgcs_by_loc)

    # antiSMASH-count-based precision (current definition)
    precision_as_count = len(matched_as) / max(1, len(regions))

    # prediction-level precision (WS4-T2)
    n_pred_matched, n_pred_total = compute_prediction_level_precision(regions, as_bgcs_by_loc)
    precision_pred_level = n_pred_matched / max(1, n_pred_total)

    # type concordance
    concordant = sum(1 for m in match_pairs if m["type_concordant"])
    type_concordance = concordant / max(1, len(match_pairs))

    # classify unmatched
    unmatched = classify_unmatched(regions, matched_pred, as_bgcs_by_loc)
    cat_counts = Counter(u["category"] for u in unmatched)

    # per-type recall
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
    per_type_recall = {c: round(type_matched.get(c, 0) / max(1, type_total.get(c, 0)), 4) for c in CLASSES}

    return {
        "score_name": score_name,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "n_antismash_bgcs": n_as_total,
        "n_matched": len(matched_as),
        "n_missed": n_as_total - len(matched_as),
        "recall": round(len(matched_as) / max(1, n_as_total), 4),
        "n_predicted_regions": len(regions),
        "precision_as_count": round(precision_as_count, 4),
        "precision_pred_level": round(precision_pred_level, 4),
        "type_concordance": round(type_concordance, 4),
        "per_type_recall": per_type_recall,
        "n_unmatched": len(unmatched),
        "unmatched_categories": {
            "boundary_expansion": cat_counts.get("boundary_expansion", 0),
            "ambiguous_unmatched": cat_counts.get("ambiguous_unmatched", 0),
            "clean_missed": cat_counts.get("clean_missed", 0),
        },
        "unmatched_details": unmatched,  # all unmatched for discovery benchmark
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--bgc-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fullwindow", action="store_true",
                    help="Full-window mode: include ambiguous windows in prediction and region merging")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # load model and data
    print("Loading model...")
    model = load_model(args.ckpt, device)

    print("Loading embeddings...")
    emb = np.load(args.emb)
    meta = []
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            meta.append(row)
    assert len(meta) == emb.shape[0], f"meta {len(meta)} != emb {emb.shape[0]}"

    print("Loading BGC annotations...")
    as_bgcs_by_loc = defaultdict(list)
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            as_bgcs_by_loc[(row["genome_id"], row["contig_id"])].append(row)

    # ─── predict on all splits ───
    print("Predicting on val (for threshold)...")
    val_mask = np.array([r["split"] == "val" and r["label"] != "ambiguous" for r in meta])
    val_emb = emb[val_mask]
    val_meta = [r for r in meta if r["split"] == "val" and r["label"] != "ambiguous"]
    val_preds = predict_batched(model, val_emb, device)
    val_is_bgc = np.array([1 if r["label"] == "positive" else 0 for r in val_meta])

    # ─── find thresholds for each score variant ───
    print("\n=== Threshold Selection ===")
    thresholds = {}
    for sname, sdef in SCORE_VARIANTS.items():
        val_scores = sdef["compute"](val_preds)
        thr = find_optimal_threshold(val_scores, val_is_bgc)
        thresholds[sname] = thr
        print(f"  {sname}: threshold={thr['threshold']:.4f}  F1={thr['f1']:.4f}  P={thr['precision']:.4f}  R={thr['recall']:.4f}")

    # ─── evaluate on test_id and test_ood ───
    eval_mode = "full_window" if args.fullwindow else "reduced_proxy"
    print(f"\nEvaluation mode: {eval_mode}")
    if args.fullwindow:
        print("  -> Ambiguous windows INCLUDED in prediction and region merging")
        print("  -> Threshold selection still uses non-ambiguous val only (Option A)")

    all_results = {}
    ambiguous_stats = {}
    for split in ["test_id", "test_ood"]:
        print(f"\n=== Evaluating {split} ===")
        if args.fullwindow:
            # Full-window: include ALL windows in this split
            s_mask = np.array([r["split"] == split for r in meta])
            s_meta = [r for r in meta if r["split"] == split]
        else:
            # Reduced proxy: exclude ambiguous
            s_mask = np.array([r["split"] == split and r["label"] != "ambiguous" for r in meta])
            s_meta = [r for r in meta if r["split"] == split and r["label"] != "ambiguous"]
        s_emb = emb[s_mask]
        s_preds = predict_batched(model, s_emb, device)

        # Ambiguous window stats (full-window mode)
        if args.fullwindow:
            amb_indices = [i for i, r in enumerate(s_meta) if r["label"] == "ambiguous"]
            n_amb = len(amb_indices)
            # Compute how many ambiguous windows each score variant calls as BGC
            amb_stats_split = {"n_ambiguous": n_amb, "n_total_windows": len(s_meta)}
            for sname, sdef in SCORE_VARIANTS.items():
                all_scores = sdef["compute"](s_preds)
                thr = thresholds[sname]["threshold"]
                n_amb_bgc = sum(1 for i in amb_indices if all_scores[i] >= thr)
                # Sub-type: partial_overlap vs buffer_zone
                n_partial = 0; n_partial_bgc = 0
                n_buffer = 0; n_buffer_bgc = 0
                for i in amb_indices:
                    overlap_ratio = float(s_meta[i].get("max_bgc_overlap_ratio", 0))
                    is_bgc = all_scores[i] >= thr
                    if overlap_ratio > 0:
                        n_partial += 1
                        if is_bgc: n_partial_bgc += 1
                    else:
                        n_buffer += 1
                        if is_bgc: n_buffer_bgc += 1
                amb_stats_split[sname] = {
                    "n_called_bgc": n_amb_bgc,
                    "bgc_rate": round(n_amb_bgc / max(1, n_amb), 4),
                    "partial_overlap": {"n": n_partial, "called_bgc": n_partial_bgc,
                                        "rate": round(n_partial_bgc / max(1, n_partial), 4)},
                    "buffer_zone": {"n": n_buffer, "called_bgc": n_buffer_bgc,
                                    "rate": round(n_buffer_bgc / max(1, n_buffer), 4)},
                }
            ambiguous_stats[split] = amb_stats_split
            print(f"  Windows: {len(s_meta)} total ({n_amb} ambiguous)")
        else:
            print(f"  Windows: {len(s_meta)} (ambiguous excluded)")

        split_results = {}
        for sname, sdef in SCORE_VARIANTS.items():
            print(f"\n  --- {sname}: {sdef['description']} ---")
            result = evaluate_score_variant(
                sname, sdef["compute"], thresholds[sname],
                s_meta, s_preds, as_bgcs_by_loc
            )
            split_results[sname] = result
            print(f"    Recall:           {result['recall']:.4f} ({result['n_matched']}/{result['n_antismash_bgcs']})")
            print(f"    Prec(AS-count):   {result['precision_as_count']:.4f}")
            print(f"    Prec(pred-level): {result['precision_pred_level']:.4f}")
            print(f"    Type concordance: {result['type_concordance']:.4f}")
            print(f"    Regions:          {result['n_predicted_regions']}")
            print(f"    Unmatched:        {result['n_unmatched']}")
            cats = result["unmatched_categories"]
            print(f"      boundary_expansion:   {cats['boundary_expansion']}")
            print(f"      ambiguous_unmatched:   {cats['ambiguous_unmatched']}")
            print(f"      clean_missed:          {cats['clean_missed']}")
            print(f"    Per-type recall:  {result['per_type_recall']}")

        all_results[split] = split_results

    # ─── comparison summary ───
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    for split in ["test_id", "test_ood"]:
        print(f"\n{split}:")
        print(f"  {'Score':>20s} | {'Recall':>8s} | {'P(AS)':>8s} | {'P(pred)':>8s} | {'Regions':>8s} | {'Unmatched':>10s} | {'Clean missed':>14s}")
        print("  " + "-"*95)
        for sname in SCORE_VARIANTS:
            r = all_results[split][sname]
            cats = r["unmatched_categories"]
            print(f"  {sname:>20s} | {r['recall']:>8.4f} | {r['precision_as_count']:>8.4f} | {r['precision_pred_level']:>8.4f} | {r['n_predicted_regions']:>8d} | {r['n_unmatched']:>10d} | {cats['clean_missed']:>14d}")

    # ─── Gate A assessment ───
    baseline_recall = all_results["test_id"]["score_is_neg"]["recall"]
    best_variant = max(SCORE_VARIANTS.keys(),
                       key=lambda s: all_results["test_id"][s]["recall"])
    best_recall = all_results["test_id"][best_variant]["recall"]
    delta = best_recall - baseline_recall

    gate_a = {
        "baseline_score": "score_is_neg",
        "baseline_recall": baseline_recall,
        "best_variant": best_variant,
        "best_recall": best_recall,
        "delta": round(delta, 4),
        "gate_pass_threshold": 0.05,
        "gate_passed": delta >= 0.05,
        "note": f"Gate A evaluates whether score mismatch is a first-order problem. "
                f"Pass condition: test_id recall improves by >= +0.05 absolute. "
                f"Evaluation mode: {eval_mode}."
    }

    print(f"\n\n=== GATE A ASSESSMENT ===")
    print(f"  Baseline (score_is_neg):  recall={baseline_recall:.4f}")
    print(f"  Best variant ({best_variant}): recall={best_recall:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"  Gate A passed: {gate_a['gate_passed']}")

    # ─── save ───
    output = {
        "task_id": "WS1-T1",
        "checkpoint": args.ckpt,
        "evaluation_type": eval_mode,
        "fullwindow": args.fullwindow,
        "threshold_selection": "non-ambiguous val only (Option A)" if args.fullwindow else "non-ambiguous val only",
        "thresholds": thresholds,
        "results": all_results,
        "ambiguous_stats": ambiguous_stats if args.fullwindow else None,
        "gate_a": gate_a,
        "precision_definitions": {
            "precision_as_count": "n_matched_antismash / n_predicted_regions (current Stage 4 definition)",
            "precision_pred_level": "n_predictions_with_any_match / n_predicted_regions (WS4-T2 strict definition)",
        },
        "unmatched_categories": {
            "boundary_expansion": "unmatched prediction within 5kb of an antiSMASH BGC",
            "ambiguous_unmatched": "unmatched prediction 5-20kb from nearest antiSMASH BGC",
            "clean_missed": "unmatched prediction >20kb from any antiSMASH BGC on same contig, or no antiSMASH BGCs on contig",
        },
    }

    out_path = os.path.join(args.out, "score_swap_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
