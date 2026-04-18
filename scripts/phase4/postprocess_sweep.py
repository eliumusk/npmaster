"""
P0 sweep: window-level AUC + region-calling parameter grid search on Route B.

Caches per-window predictions once, then sweeps:
  - score formula: max_type / 1-is_neg / fused
  - threshold:     0.50 .. 0.99
  - max_gap:       0 / 1 / 2 negative windows tolerated
  - length_cap:    None / 30k / 50k / 80k bp (split at score local minimum)

For each (split, score, config) reports:
  - recall @ overlap/min_len >= 0.5  (loose)
  - precision (loose, strict ov/max>=0.5, IoU>=0.5)
  - region count + length stats

Outputs:
  - cached_preds_<seed>.npz : per-window scores + labels (run once per ckpt)
  - sweep_results.json      : full grid + AUC + best operating points
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# repo paths
REPO = Path("/data/muskliu/np_master/code/np-master")
sys.path.insert(0, str(REPO / "bgc-gcd"))
from model.context_full_model import ContextFullModel  # noqa: E402

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
WINDOW_BP = 16384
STRIDE_BP = 8192


# --------------------------------------------------------------- inference
@torch.no_grad()
def predict_all_windows(ckpt_path, embeddings_path, meta_arrays_path,
                        contig_index_path, splits, device="cuda"):
    """Run Route B inference contig-by-contig. Returns dict per split."""
    print(f"[load] ckpt = {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ContextFullModel(input_dim=4096, proj_hidden=2048, proj_out=256,
                             proj_dropout=0.3, n_labels=8,
                             cnn_dropout=0.1, init_beta=-4.0)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"  alpha={model.gate.alpha.item():.4f}")

    print("[load] embeddings (mmap)")
    emb = np.load(embeddings_path, mmap_mode="r")

    print("[load] meta arrays + contig index")
    with open(meta_arrays_path, "rb") as f:
        meta = pickle.load(f)
    with open(contig_index_path, "rb") as f:
        contig_idx = pickle.load(f)

    split_arr = meta["split"]
    label_arr = meta["label"]      # 'positive' / 'negative' / 'ambiguous'
    labels_8 = meta["labels"]       # (N, 8) one-hot incl is_negative col 7
    genome_arr = meta["genome_id"]
    contig_arr = meta["contig_id"]
    wstart_arr = meta["window_start"]

    out_per_split = {}
    for split in splits:
        print(f"\n[infer] split={split}")
        scores_buf = []   # (8,) probs per window
        gidx_buf = []
        n_contigs = 0
        t0 = time.time()
        for ckey, indices in contig_idx.items():
            in_split = [i for i in indices if split_arr[i] == split]
            if not in_split:
                continue
            n_contigs += 1
            L = len(indices)
            x = np.asarray(emb[indices], dtype=np.float32)        # (L, 4096)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)     # (1, L, 4096)
            valid = torch.ones(1, L, dtype=torch.bool, device=device)
            _, logits = model(x_t, valid)
            probs = torch.sigmoid(logits[0]).cpu().numpy()        # (L, 8)
            for pos, g in enumerate(indices):
                if split_arr[g] != split:
                    continue
                scores_buf.append(probs[pos])
                gidx_buf.append(g)
            if n_contigs % 200 == 0:
                print(f"  {n_contigs} contigs done in {time.time()-t0:.0f}s")

        gidx = np.array(gidx_buf, dtype=np.int64)
        probs = np.stack(scores_buf, axis=0).astype(np.float32)   # (N, 8)
        out_per_split[split] = {
            "g_idx": gidx,
            "probs": probs,
            "genome_id": np.array([str(genome_arr[g]) for g in gidx]),
            "contig_id": np.array([str(contig_arr[g]) for g in gidx]),
            "window_start": np.array([int(wstart_arr[g]) for g in gidx], dtype=np.int64),
            "label": np.array([str(label_arr[g]) for g in gidx]),
            "is_bgc": (1 - labels_8[gidx, 7]).astype(np.int8),
        }
        print(f"  total windows = {len(gidx)}, contigs = {n_contigs}, elapsed = {time.time()-t0:.0f}s")
    return out_per_split


def cache_predictions(out_path, ckpt, emb, meta_arr, contig_idx, splits, device):
    if os.path.exists(out_path):
        print(f"[cache] reuse {out_path}")
        d = np.load(out_path, allow_pickle=True)
        return {s: {k: d[f"{s}__{k}"] for k in [
            "g_idx", "probs", "genome_id", "contig_id",
            "window_start", "label", "is_bgc"]} for s in splits}
    res = predict_all_windows(ckpt, emb, meta_arr, contig_idx, splits, device)
    save = {}
    for s, d in res.items():
        for k, v in d.items():
            save[f"{s}__{k}"] = v
    np.savez_compressed(out_path, **save)
    print(f"[cache] saved {out_path}")
    return res


# --------------------------------------------------------------- AUC (window-level)
SCORE_FNS = {
    "max_type": lambda p: p[:, :7].max(axis=1),
    "1_minus_isneg": lambda p: 1.0 - p[:, 7],
    "fused": lambda p: 0.5 * (1.0 - p[:, 7]) + 0.5 * p[:, :7].max(axis=1),
}


def window_auc(split_data):
    """AUROC + AUPRC on non-ambiguous windows."""
    keep = split_data["label"] != "ambiguous"
    y = split_data["is_bgc"][keep].astype(np.int32)
    out = {}
    for name, fn in SCORE_FNS.items():
        s = fn(split_data["probs"])[keep]
        out[name] = {
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
        }
    return out


# --------------------------------------------------------------- region calling
def call_regions(score_arr, window_meta, threshold, max_gap=0,
                 min_length_bp=3000, length_cap_bp=None):
    """
    Build regions from per-window scores along sorted contig positions.

    window_meta: list of dicts with genome_id, contig_id, window_start
    score_arr:   1D scores aligned with window_meta
    max_gap:     allow up to N consecutive sub-threshold windows inside a region
    length_cap_bp: if a region exceeds this, split at lowest-score window inside
    """
    regions = []
    cur = None
    gap_cnt = 0

    def close(cur_):
        if cur_ is None:
            return
        if cur_["end"] - cur_["start"] >= min_length_bp:
            cur_["windows"] = list(cur_["windows"])  # copy
            regions.append(cur_)

    for i, (w, s) in enumerate(zip(window_meta, score_arr)):
        same_contig = (cur is not None
                       and cur["contig"] == w["contig_id"]
                       and cur["genome_id"] == w["genome_id"])
        is_pos = s >= threshold
        ws, we = int(w["window_start"]), int(w["window_start"]) + WINDOW_BP

        if not same_contig:
            close(cur); cur = None; gap_cnt = 0

        if is_pos:
            if cur is None:
                cur = {"genome_id": w["genome_id"], "contig": w["contig_id"],
                       "start": ws, "end": we,
                       "windows": [(ws, we, float(s), i)],
                       "max_score": float(s)}
            else:
                cur["end"] = max(cur["end"], we)
                cur["windows"].append((ws, we, float(s), i))
                cur["max_score"] = max(cur["max_score"], float(s))
            gap_cnt = 0
        else:
            if cur is not None:
                if gap_cnt < max_gap:
                    # tolerate; extend region but don't add window
                    gap_cnt += 1
                    cur["end"] = max(cur["end"], we)
                else:
                    close(cur); cur = None; gap_cnt = 0
    close(cur)

    # length cap: split oversized regions at score local minima (recursive)
    if length_cap_bp:
        regions = split_long_regions(regions, length_cap_bp, min_length_bp)

    # final per-region annotation
    for r in regions:
        r["length"] = r["end"] - r["start"]
        # primary type from windows
        type_max = np.zeros(7)
        for (_, _, s, idx) in r["windows"]:
            # we only kept score, not type vector here; primary_type via window_meta below
            pass
        r["n_windows"] = len(r["windows"])
    return regions


def split_long_regions(regions, cap, min_len):
    out = []
    stack = list(regions)
    while stack:
        r = stack.pop()
        if r["end"] - r["start"] <= cap or len(r["windows"]) < 3:
            out.append(r)
            continue
        # find lowest-score window not at edges
        ws_sorted = sorted(r["windows"], key=lambda t: t[0])
        scores = [t[2] for t in ws_sorted]
        # restrict split to interior (skip first/last)
        interior = scores[1:-1]
        if not interior:
            out.append(r); continue
        cut_local = int(np.argmin(interior)) + 1
        left_ws = ws_sorted[:cut_local]
        right_ws = ws_sorted[cut_local + 1:]   # drop the split window itself
        if not left_ws or not right_ws:
            out.append(r); continue
        for sub in (left_ws, right_ws):
            sub_start = sub[0][0]
            sub_end = max(t[1] for t in sub)
            if sub_end - sub_start < min_len:
                continue
            stack.append({"genome_id": r["genome_id"], "contig": r["contig"],
                          "start": sub_start, "end": sub_end,
                          "windows": sub,
                          "max_score": max(t[2] for t in sub)})
    return out


# --------------------------------------------------------------- matching + metrics
def overlap_metrics(p_s, p_e, g_s, g_e):
    ov = max(0, min(p_e, g_e) - max(p_s, g_s))
    plen = p_e - p_s
    glen = g_e - g_s
    return {
        "loose":  ov / max(1, min(plen, glen)),     # ov / min_len
        "strict": ov / max(1, max(plen, glen)),     # ov / max_len
        "iou":    ov / max(1, plen + glen - ov),
    }


def evaluate_regions(regions, gt_by_loc, ov_thr=0.5):
    """Compute recall + 3 precision variants. One-to-one match per GT (best loose)."""
    matched_gt = {"loose": set(), "strict": set(), "iou": set()}
    matched_pred_per_metric = {"loose": set(), "strict": set(), "iou": set()}

    by_genome = defaultdict(list)
    for ri, r in enumerate(regions):
        by_genome[r["genome_id"]].append((ri, r))

    for genome_id, regs in by_genome.items():
        contigs = {r["contig"] for _, r in regs}
        gt_list = []
        for c in contigs:
            for ab in gt_by_loc.get((genome_id, c), []):
                gt_list.append(ab)
        for ab in gt_list:
            ab_s = int(ab["start_bp"]); ab_e = int(ab["end_bp"])
            ab_key = (genome_id, ab["contig_id"], ab_s, ab_e)
            best_ri = {"loose": -1, "strict": -1, "iou": -1}
            best_v = {"loose": 0.0, "strict": 0.0, "iou": 0.0}
            for ri, r in regs:
                if r["contig"] != ab["contig_id"]:
                    continue
                m = overlap_metrics(r["start"], r["end"], ab_s, ab_e)
                for k in m:
                    if m[k] > best_v[k]:
                        best_v[k] = m[k]; best_ri[k] = ri
            for k in ("loose", "strict", "iou"):
                if best_v[k] >= ov_thr:
                    matched_gt[k].add(ab_key)
                    matched_pred_per_metric[k].add(best_ri[k])

    # prediction-level precision: for each pred, per metric independently, does ANY GT pass?
    pred_match = {"loose": 0, "strict": 0, "iou": 0}
    for ri, r in enumerate(regions):
        gt_list = gt_by_loc.get((r["genome_id"], r["contig"]), [])
        matched_any = {"loose": False, "strict": False, "iou": False}
        for ab in gt_list:
            m = overlap_metrics(r["start"], r["end"],
                                int(ab["start_bp"]), int(ab["end_bp"]))
            for k in matched_any:
                if m[k] >= ov_thr:
                    matched_any[k] = True
        for k, v in matched_any.items():
            if v:
                pred_match[k] += 1
    n_pred = max(1, len(regions))
    return {
        "n_regions": len(regions),
        "matched_gt":   {k: len(v) for k, v in matched_gt.items()},
        "precision":    {k: pred_match[k] / n_pred for k in pred_match},
        "n_pred_matched_per_metric": pred_match,
    }


# --------------------------------------------------------------- driver
def load_gt(bgc_csv):
    by = defaultdict(list)
    with open(bgc_csv) as f:
        for row in csv.DictReader(f):
            by[(row["genome_id"], row["contig_id"])].append(row)
    return by


def make_window_meta(split_data):
    """Sort windows by genome -> contig -> start. Return sorted indices + meta list."""
    n = len(split_data["g_idx"])
    keys = [(split_data["genome_id"][i], split_data["contig_id"][i],
             int(split_data["window_start"][i])) for i in range(n)]
    order = sorted(range(n), key=lambda i: keys[i])
    meta = [{"genome_id": split_data["genome_id"][i],
             "contig_id": split_data["contig_id"][i],
             "window_start": int(split_data["window_start"][i])} for i in order]
    return order, meta


def run_split(split_data, score_name, score, gt_by_loc, configs):
    order, win_meta = make_window_meta(split_data)
    score_sorted = score[order]
    split_genomes = set(split_data["genome_id"].tolist())
    n_gt = sum(len(v) for (g, _), v in gt_by_loc.items() if g in split_genomes)

    rows = []
    for cfg in configs:
        regions = call_regions(score_sorted, win_meta,
                               threshold=cfg["threshold"],
                               max_gap=cfg["max_gap"],
                               length_cap_bp=cfg["length_cap"])
        ev = evaluate_regions(regions, gt_by_loc)
        recall = {k: ev["matched_gt"][k] / max(1, n_gt) for k in ("loose", "strict", "iou")}
        # length stats
        lens = [r["end"] - r["start"] for r in regions]
        rows.append({
            "score": score_name,
            "threshold": cfg["threshold"],
            "max_gap": cfg["max_gap"],
            "length_cap": cfg["length_cap"],
            "n_regions": ev["n_regions"],
            "median_len_kb": float(np.median(lens) / 1000) if lens else 0.0,
            "p95_len_kb": float(np.percentile(lens, 95) / 1000) if lens else 0.0,
            "recall_loose": recall["loose"],
            "recall_strict": recall["strict"],
            "recall_iou": recall["iou"],
            "p_loose": ev["precision"]["loose"],
            "p_strict": ev["precision"]["strict"],
            "p_iou": ev["precision"]["iou"],
        })
    return rows, n_gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/data/muskliu/np_master/data/phase3/route_b/best_route_b_s0.pt")
    p.add_argument("--emb", default="/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy")
    p.add_argument("--meta", default="/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl")
    p.add_argument("--contig-index", default="/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl")
    p.add_argument("--bgc-csv", default="/data/muskliu/np_master/data/bgc_annotations.csv")
    p.add_argument("--cache", default="/data/muskliu/np_master/data/phase4/postprocess/cached_preds_s0.npz")
    p.add_argument("--out", default="/data/muskliu/np_master/data/phase4/postprocess/sweep_results.json")
    p.add_argument("--splits", default="test_id,test_ood")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    splits = args.splits.split(",")

    pred = cache_predictions(args.cache, args.ckpt, args.emb, args.meta,
                             args.contig_index, splits, args.device)
    gt = load_gt(args.bgc_csv)

    # AUC (window-level)
    auc = {s: window_auc(pred[s]) for s in splits}
    print("\n=== Window-level AUROC / AUPRC (non-ambiguous) ===")
    for s in splits:
        print(f"  {s}:")
        for sn, m in auc[s].items():
            print(f"    {sn:>15s}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  pos/neg={m['n_pos']}/{m['n_neg']}")

    # config grid
    configs = []
    for thr in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]:
        for mg in [0, 1, 2]:
            for cap in [None, 30000, 50000, 80000]:
                configs.append({"threshold": thr, "max_gap": mg, "length_cap": cap})

    all_rows = {}
    for s in splits:
        all_rows[s] = {}
        for sn, fn in SCORE_FNS.items():
            score_vals = fn(pred[s]["probs"])
            print(f"\n=== sweep on {s} / score={sn} ({len(configs)} configs) ===")
            t0 = time.time()
            rows, n_gt = run_split(pred[s], sn, score_vals, gt, configs)
            all_rows[s][sn] = {"n_gt": n_gt, "rows": rows}
            # report best by f1(strict)
            for r in rows:
                p_, r_ = r["p_strict"], r["recall_strict"]
                r["f1_strict"] = 2 * p_ * r_ / (p_ + r_ + 1e-9)
                p2, r2 = r["p_loose"], r["recall_loose"]
                r["f1_loose"] = 2 * p2 * r2 / (p2 + r2 + 1e-9)
            best_strict = max(rows, key=lambda r: r["f1_strict"])
            best_loose = max(rows, key=lambda r: r["f1_loose"])
            best_iou = max(rows, key=lambda r: 2 * r["p_iou"] * r["recall_iou"] / (r["p_iou"] + r["recall_iou"] + 1e-9))
            print(f"  best F1(strict): thr={best_strict['threshold']} gap={best_strict['max_gap']} "
                  f"cap={best_strict['length_cap']}  R={best_strict['recall_strict']:.3f} "
                  f"P={best_strict['p_strict']:.3f} F1={best_strict['f1_strict']:.3f} "
                  f"(median len={best_strict['median_len_kb']:.1f}kb, n={best_strict['n_regions']})")
            print(f"  best F1(loose):  thr={best_loose['threshold']} gap={best_loose['max_gap']} "
                  f"cap={best_loose['length_cap']}  R={best_loose['recall_loose']:.3f} "
                  f"P={best_loose['p_loose']:.3f} F1={best_loose['f1_loose']:.3f} "
                  f"(median len={best_loose['median_len_kb']:.1f}kb, n={best_loose['n_regions']})")
            f1_iou = 2 * best_iou["p_iou"] * best_iou["recall_iou"] / (best_iou["p_iou"] + best_iou["recall_iou"] + 1e-9)
            print(f"  best F1(iou):    thr={best_iou['threshold']} gap={best_iou['max_gap']} "
                  f"cap={best_iou['length_cap']}  R={best_iou['recall_iou']:.3f} "
                  f"P={best_iou['p_iou']:.3f} F1={f1_iou:.3f} "
                  f"(median len={best_iou['median_len_kb']:.1f}kb, n={best_iou['n_regions']})")
            # baseline: current mainline default (thr=0.86, no gap, no cap)
            base = [r for r in rows if r["threshold"] == 0.85 and r["max_gap"] == 0 and r["length_cap"] is None]
            if base:
                b = base[0]
                print(f"  baseline (thr=0.85, no gap, no cap): "
                      f"R_loose={b['recall_loose']:.3f} P_loose={b['p_loose']:.3f} "
                      f"R_strict={b['recall_strict']:.3f} P_strict={b['p_strict']:.3f}")
            print(f"  elapsed {time.time()-t0:.0f}s")

    out = {"auc": auc, "sweep": all_rows,
           "metric_defs": {
               "loose":  "overlap / min(pred_len, gt_len) >= 0.5  (current)",
               "strict": "overlap / max(pred_len, gt_len) >= 0.5",
               "iou":    "intersection-over-union >= 0.5"}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
