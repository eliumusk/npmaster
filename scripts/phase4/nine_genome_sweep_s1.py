"""
Diagnostic sweep on zhanglin's s1 ckpt (MIBiG v2 trained) on 9-genome benchmark.

Step 1: Load s1 ckpt, run context inference on 9 cached embeddings -> save bgc_score.npz
Step 2: Sweep (threshold x max_gap x length_cap) x metric (loose/strict/iou)
Step 3: Report window-level AUROC as diagnostic (not for tuning)

NOTE: numbers reported here are DIAGNOSTIC — we sweep on the test set to understand
the model, not to pick a final config for reporting. Any config claimed on 9-genome
as a benchmark number would need held-out tuning.
"""
import argparse
import json
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve

BGC_GCD = "/data/zhanglin/np_master/bgc-gcd"
if BGC_GCD not in sys.path:
    sys.path.insert(0, BGC_GCD)

from model.context_full_model import ContextFullModel  # noqa: E402

NINE_GENOME_DIR = "/data/syh/benchmark/nine_genome"
GT_PATH = os.path.join(NINE_GENOME_DIR, "ground_truth.json")
EMB_CACHE_DIR = "/data/zhanglin/np_master/data/mibig_v2/nine_genome_eval"

GENOME_INDEX = [
    ("GCA_000158915.1", "SSOG"), ("GCA_000158875.1", "SSMG"),
    ("GCA_000154945.1", "SSDG"), ("GCA_000158895.1", "SSNG"),
    ("GCA_000156435.1", "SSFG"), ("GCA_000568915.1", "MCBG"),
    ("GCA_000158815.1", "MCAG"), ("GCA_000568255.1", "KUTG"),
    ("GCA_000156475.1", "SSHG"),
]


def overlap_length(s1, e1, s2, e2):
    return max(0, min(e1, e2) - max(s1, s2))


# ------------------ Step 1: inference + cache ------------------

def run_inference_and_cache(ckpt_path, out_dir, device="cuda"):
    """Run context inference on all 9 genomes, cache window scores."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "window_scores.pkl")
    if os.path.exists(cache_path):
        print(f"Cached window scores found at {cache_path}, skipping inference.")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Loading ckpt from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ContextFullModel(
        input_dim=4096, proj_hidden=2048, proj_out=256,
        proj_dropout=0.0, n_labels=8, cnn_dropout=0.0, init_beta=-4.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    alpha = model.gate.alpha.item()
    print(f"  alpha={alpha:.4f}")

    all_data = {"alpha": alpha, "per_genome": {}}

    with torch.no_grad():
        for gca, tag in GENOME_INDEX:
            emb_path = os.path.join(EMB_CACHE_DIR, f"{gca}_embeddings.npz")
            data = np.load(emb_path, allow_pickle=True)
            emb_array = data["embeddings"]
            meta = json.loads(str(data["meta"]))

            contig_windows = defaultdict(list)
            for i, m in enumerate(meta):
                contig_windows[m["contig"]].append((i, m))

            # windows in original order
            per_window = [None] * len(meta)
            for contig_id, win_list in contig_windows.items():
                indices = [w[0] for w in win_list]
                emb = emb_array[indices]
                emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)
                valid = torch.ones(1, len(indices), dtype=torch.bool, device=device)
                _, logits = model(emb_t, valid)
                probs = torch.sigmoid(logits[0]).cpu().numpy()  # (L, 8)
                for j, (idx, m) in enumerate(win_list):
                    per_window[idx] = {
                        "contig": m["contig"],
                        "start": int(m["start"]),
                        "end": int(m["end"]),
                        "probs": probs[j].astype(np.float32),
                    }

            all_data["per_genome"][gca] = {"tag": tag, "windows": per_window}
            print(f"  {tag} ({gca}): {len(per_window)} windows scored")

    with open(cache_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Cached to {cache_path}")
    return all_data


# ------------------ Step 2: sweep ------------------

def call_regions(windows, scores, threshold, max_gap, length_cap=None):
    """Merge consecutive positive windows; optionally split at length_cap."""
    positive = [(w["contig"], w["start"], w["end"])
                for w, s in zip(windows, scores) if s >= threshold]
    if not positive:
        return []
    positive.sort()

    raw = []
    cur_c, cur_s, cur_e = positive[0]
    for c, s, e in positive[1:]:
        if c == cur_c and s <= cur_e + max_gap:
            cur_e = max(cur_e, e)
        else:
            raw.append((cur_c, cur_s, cur_e))
            cur_c, cur_s, cur_e = c, s, e
    raw.append((cur_c, cur_s, cur_e))

    if length_cap is None:
        return raw
    out = []
    for c, s, e in raw:
        L = e - s
        if L <= length_cap:
            out.append((c, s, e))
        else:
            # split into roughly equal chunks not exceeding length_cap
            n_chunks = int(np.ceil(L / length_cap))
            chunk = int(np.ceil(L / n_chunks))
            for k in range(n_chunks):
                cs = s + k * chunk
                ce = min(e, s + (k + 1) * chunk)
                if ce > cs:
                    out.append((c, cs, ce))
    return out


def eval_regions(gt_regions, pred_regions, overlap_frac=0.5):
    """Return loose/strict/iou precision+recall."""
    gt_by_contig = defaultdict(list)
    for c, s, e in gt_regions:
        gt_by_contig[c].append((s, e))
    pred_by_contig = defaultdict(list)
    for c, s, e in pred_regions:
        pred_by_contig[c].append((s, e))

    # Recall: per GT, best pred by each metric
    tp_r = {"loose": 0, "strict": 0, "iou": 0}
    for c, gs, ge in gt_regions:
        glen = ge - gs
        if glen <= 0:
            continue
        best = {"loose": 0.0, "strict": 0.0, "iou": 0.0}
        for ps, pe in pred_by_contig.get(c, []):
            plen = pe - ps
            ov = overlap_length(gs, ge, ps, pe)
            if ov == 0:
                continue
            loose = ov / max(1, min(plen, glen))
            strict = ov / max(1, max(plen, glen))
            iou = ov / max(1, plen + glen - ov)
            best["loose"] = max(best["loose"], loose)
            best["strict"] = max(best["strict"], strict)
            best["iou"] = max(best["iou"], iou)
        for k in tp_r:
            if best[k] >= overlap_frac:
                tp_r[k] += 1

    # Precision: per pred, best GT by each metric
    tp_p = {"loose": 0, "strict": 0, "iou": 0}
    for c, ps, pe in pred_regions:
        plen = pe - ps
        if plen <= 0:
            continue
        best = {"loose": 0.0, "strict": 0.0, "iou": 0.0}
        for gs, ge in gt_by_contig.get(c, []):
            glen = ge - gs
            ov = overlap_length(gs, ge, ps, pe)
            if ov == 0:
                continue
            loose = ov / max(1, min(plen, glen))
            strict = ov / max(1, max(plen, glen))
            iou = ov / max(1, plen + glen - ov)
            best["loose"] = max(best["loose"], loose)
            best["strict"] = max(best["strict"], strict)
            best["iou"] = max(best["iou"], iou)
        for k in tp_p:
            if best[k] >= overlap_frac:
                tp_p[k] += 1

    n_gt = len(gt_regions)
    n_pred = len(pred_regions)
    out = {"n_gt": n_gt, "n_pred": n_pred}
    for k in ["loose", "strict", "iou"]:
        r = tp_r[k] / max(1, n_gt)
        p = tp_p[k] / max(1, n_pred)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[f"recall_{k}"] = r
        out[f"precision_{k}"] = p
        out[f"f1_{k}"] = f1
    return out


def run_sweep(all_data, gt_data, score_fn_name="max_type"):
    """Sweep params and return table."""
    score_fns = {
        "max_type": lambda p: p[:, :7].max(axis=-1),
        "one_minus_isneg": lambda p: 1.0 - p[:, 7],
        "fused": lambda p: 0.5 * (1.0 - p[:, 7]) + 0.5 * p[:, :7].max(axis=-1),
    }
    score_fn = score_fns[score_fn_name]

    # Precompute scores + flatten for quick sweep
    gca_to_windows = {}
    gca_to_scores = {}
    for gca, tag in GENOME_INDEX:
        wins = all_data["per_genome"][gca]["windows"]
        probs = np.stack([w["probs"] for w in wins])
        scores = score_fn(probs)
        gca_to_windows[gca] = [{"contig": w["contig"], "start": w["start"], "end": w["end"]} for w in wins]
        gca_to_scores[gca] = scores

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
    max_gaps = [0, 16384, 32768]
    length_caps = [None, 30000, 50000, 80000]

    results = []
    for thr in thresholds:
        for gap in max_gaps:
            for cap in length_caps:
                agg = {"n_gt": 0, "n_pred": 0}
                for k in ["loose", "strict", "iou"]:
                    agg[f"tp_r_{k}"] = 0
                    agg[f"tp_p_{k}"] = 0

                for gca, tag in GENOME_INDEX:
                    windows = gca_to_windows[gca]
                    scores = gca_to_scores[gca]
                    gt_regions = [(c, s, e) for c, s, e in gt_data.get(gca, [])]
                    pred_regions = call_regions(windows, scores, thr, gap, cap)
                    m = eval_regions(gt_regions, pred_regions)
                    agg["n_gt"] += m["n_gt"]
                    agg["n_pred"] += m["n_pred"]
                    for k in ["loose", "strict", "iou"]:
                        agg[f"tp_r_{k}"] += int(round(m[f"recall_{k}"] * m["n_gt"]))
                        agg[f"tp_p_{k}"] += int(round(m[f"precision_{k}"] * m["n_pred"]))

                row = {
                    "threshold": thr, "max_gap": gap,
                    "length_cap": cap if cap is not None else 0,
                    "score_fn": score_fn_name,
                    "n_gt": agg["n_gt"], "n_pred": agg["n_pred"],
                }
                for k in ["loose", "strict", "iou"]:
                    r = agg[f"tp_r_{k}"] / max(1, agg["n_gt"])
                    p = agg[f"tp_p_{k}"] / max(1, agg["n_pred"])
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    row[f"recall_{k}"] = round(r, 4)
                    row[f"precision_{k}"] = round(p, 4)
                    row[f"f1_{k}"] = round(f1, 4)
                results.append(row)
    return results


# ------------------ Step 3: window-level AUROC (diagnostic) ------------------

def window_auroc(all_data, gt_data, score_fn_name="max_type"):
    """Window-level AUROC on 9 genomes. A window is positive if it overlaps any GT BGC."""
    score_fns = {
        "max_type": lambda p: p[:, :7].max(axis=-1),
        "one_minus_isneg": lambda p: 1.0 - p[:, 7],
        "fused": lambda p: 0.5 * (1.0 - p[:, 7]) + 0.5 * p[:, :7].max(axis=-1),
    }
    score_fn = score_fns[score_fn_name]

    all_scores = []
    all_labels = []
    for gca, tag in GENOME_INDEX:
        wins = all_data["per_genome"][gca]["windows"]
        probs = np.stack([w["probs"] for w in wins])
        scores = score_fn(probs)
        gt_regions = gt_data.get(gca, [])
        gt_by_contig = defaultdict(list)
        for c, s, e in gt_regions:
            gt_by_contig[c].append((s, e))
        for w, s in zip(wins, scores):
            is_pos = 0
            for gs, ge in gt_by_contig.get(w["contig"], []):
                if overlap_length(w["start"], w["end"], gs, ge) > 0:
                    is_pos = 1
                    break
            all_scores.append(float(s))
            all_labels.append(is_pos)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_scores)
    prec, rec, thr = precision_recall_curve(all_labels, all_scores)
    # sweep-free F1-optimal at window level (still on 9-genome — diagnostic only)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best = int(np.argmax(f1s))
    return {
        "window_auroc": float(auroc),
        "n_windows": int(len(all_labels)),
        "n_positive": int(all_labels.sum()),
        "pos_rate": float(all_labels.mean()),
        "window_f1_best": float(f1s[best]),
        "window_thr_at_best_f1": float(thr[best]) if best < len(thr) else None,
        "window_precision_at_best": float(prec[best]),
        "window_recall_at_best": float(rec[best]),
    }


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-fn", default="max_type",
                    choices=["max_type", "one_minus_isneg", "fused"])
    ap.add_argument("--ckpt", required=True,
                    help="Path to ContextFullModel checkpoint")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for cache + results")
    ap.add_argument("--tag", default="run",
                    help="Tag used in output filename")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load GT
    with open(GT_PATH) as f:
        gt_data = json.load(f)["ground_truth"]

    # Step 1: inference (cached)
    all_data = run_inference_and_cache(args.ckpt, args.out_dir)

    # Step 2: sweep
    print("\n=== Sweep ===")
    results = run_sweep(all_data, gt_data, score_fn_name=args.score_fn)

    # Find top configs per metric
    print("\nTop configs by F1_strict (top 10):")
    top_strict = sorted(results, key=lambda r: r["f1_strict"], reverse=True)[:10]
    header = "thr   gap    cap     n_pred  R_l    P_l    R_s    P_s    F1_s   R_iou  P_iou  F1_iou"
    print(header)
    for r in top_strict:
        cap_s = f"{r['length_cap']/1000:.0f}k" if r['length_cap'] else "none"
        print(f"{r['threshold']:.2f}  {r['max_gap']//1024:>3}k   {cap_s:<6} "
              f"{r['n_pred']:>4}    {r['recall_loose']:.3f}  {r['precision_loose']:.3f}  "
              f"{r['recall_strict']:.3f}  {r['precision_strict']:.3f}  {r['f1_strict']:.3f}  "
              f"{r['recall_iou']:.3f}  {r['precision_iou']:.3f}  {r['f1_iou']:.3f}")

    print("\nTop configs by F1_iou (top 10):")
    top_iou = sorted(results, key=lambda r: r["f1_iou"], reverse=True)[:10]
    print(header)
    for r in top_iou:
        cap_s = f"{r['length_cap']/1000:.0f}k" if r['length_cap'] else "none"
        print(f"{r['threshold']:.2f}  {r['max_gap']//1024:>3}k   {cap_s:<6} "
              f"{r['n_pred']:>4}    {r['recall_loose']:.3f}  {r['precision_loose']:.3f}  "
              f"{r['recall_strict']:.3f}  {r['precision_strict']:.3f}  {r['f1_strict']:.3f}  "
              f"{r['recall_iou']:.3f}  {r['precision_iou']:.3f}  {r['f1_iou']:.3f}")

    # Reference: zhanglin baseline was thr=0.173, max_gap=16k, no cap
    print("\nZhanglin's s1 baseline (thr=0.173, gap=16k, no cap):")
    for r in results:
        if r["threshold"] == 0.20 and r["max_gap"] == 16384 and r["length_cap"] == 0:
            print(f"  (close: thr=0.20) {r}")

    # Step 3: AUROC
    print("\n=== Window-level diagnostic metrics on 9-genome ===")
    auroc_info = window_auroc(all_data, gt_data, score_fn_name=args.score_fn)
    for k, v in auroc_info.items():
        print(f"  {k}: {v}")

    out = {
        "tag": args.tag,
        "ckpt": args.ckpt,
        "score_fn": args.score_fn,
        "alpha": all_data["alpha"],
        "window_diagnostic": auroc_info,
        "sweep_results": results,
    }
    out_path = os.path.join(args.out_dir, f"sweep_{args.tag}_{args.score_fn}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
