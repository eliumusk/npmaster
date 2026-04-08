#!/usr/bin/env python3
"""
Stage 4: Full evaluation pipeline.
  4.1 Window-level: threshold selection + confusion analysis
  4.2 Genome-level: region merging + antiSMASH matching
  4.3 Disagreement: FN analysis + novel candidate ranking
  4.4 MIBiG external validation
"""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── helpers ───

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
ALL_COLS = CLASSES + ["is_negative"]

def load_model(ckpt_path, device="cpu"):
    # Add bgc-gcd to path for model imports
    bgc_gcd_path = str(Path(__file__).resolve().parents[2] / "bgc-gcd")
    if bgc_gcd_path not in sys.path:
        sys.path.insert(0, bgc_gcd_path)
    from model.full_model import BGCModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # infer n_phyla from DANN output layer (last linear in dann)
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
    """Return (N, 8) sigmoid-applied predictions."""
    preds = []
    ds = TensorDataset(torch.from_numpy(embeddings).float())
    loader = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for (x,) in loader:
            _, pred, _ = model(x.to(device))
            pred = torch.sigmoid(pred)  # model outputs raw logits
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


# ─── 4.1 thresholds + confusion ───

def step_4_1(model, emb, meta, out_dir, device):
    print("\n=== Step 4.1: Threshold selection + confusion ===")
    os.makedirs(out_dir, exist_ok=True)

    # val set
    val_mask = np.array([r["split"] == "val" and r["label"] != "ambiguous" for r in meta])
    val_emb = emb[val_mask]
    val_labels = np.array([[int(r[c]) for c in ALL_COLS] for r in meta if r["split"] == "val" and r["label"] != "ambiguous"], dtype=np.float32)
    val_preds = predict_batched(model, val_emb, device)

    from sklearn.metrics import precision_recall_curve, f1_score
    thresholds = {}
    for i, c in enumerate(ALL_COLS):
        prec, rec, thr = precision_recall_curve(val_labels[:, i], val_preds[:, i])
        f1s = 2 * prec * rec / (prec + rec + 1e-10)
        best = np.argmax(f1s)
        thresholds[c] = {
            "threshold": float(thr[best]) if best < len(thr) else 0.5,
            "precision": float(prec[best]),
            "recall": float(rec[best]),
            "f1": float(f1s[best]),
        }
    with open(f"{out_dir}/optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds: { {c: round(v['threshold'],3) for c,v in thresholds.items()} }")

    # confusion on test_id
    tid_mask = np.array([r["split"] == "test_id" and r["label"] != "ambiguous" for r in meta])
    tid_emb = emb[tid_mask]
    tid_labels = np.array([[int(r[c]) for c in ALL_COLS] for r in meta if r["split"] == "test_id" and r["label"] != "ambiguous"], dtype=np.float32)
    tid_preds = predict_batched(model, tid_emb, device)
    tid_meta = [r for r in meta if r["split"] == "test_id" and r["label"] != "ambiguous"]

    # FP/FN analysis
    fp_rows = []
    fn_rows = []
    is_neg_thr = thresholds["is_negative"]["threshold"]
    for i, r in enumerate(tid_meta):
        pred_bgc = tid_preds[i, 7] < is_neg_thr  # is_negative below threshold → BGC
        true_bgc = r["label"] == "positive"
        if pred_bgc and not true_bgc:
            fp_rows.append({
                "window_id": r["window_id"], "genome_id": r["genome_id"],
                "phylum": r["phylum"],
                "pred_probs": {c: round(float(tid_preds[i, j]), 3) for j, c in enumerate(ALL_COLS)},
            })
        elif not pred_bgc and true_bgc:
            fn_rows.append({
                "window_id": r["window_id"], "genome_id": r["genome_id"],
                "phylum": r["phylum"], "true_types": [c for c in CLASSES if r[c] == "1"],
            })

    with open(f"{out_dir}/false_positive_analysis.json", "w") as f:
        json.dump({"n_fp": len(fp_rows), "examples": fp_rows[:100]}, f, indent=2)
    with open(f"{out_dir}/false_negative_analysis.json", "w") as f:
        json.dump({"n_fn": len(fn_rows), "examples": fn_rows[:100]}, f, indent=2)
    print(f"test_id: {len(fp_rows)} FP, {len(fn_rows)} FN")

    return thresholds


# ─── 4.2 genome-level ───

def step_4_2(model, emb, meta, bgc_csv, thresholds, out_dir, device):
    print("\n=== Step 4.2: Genome-level evaluation ===")
    os.makedirs(out_dir, exist_ok=True)

    is_neg_thr = thresholds["is_negative"]["threshold"]

    # load antiSMASH BGC annotations
    bgcs = defaultdict(list)
    with open(bgc_csv) as f:
        for row in csv.DictReader(f):
            bgcs[(row["genome_id"], row["contig_id"])].append(row)

    results = {"test_id": {}, "test_ood": {}}
    all_predicted = []
    all_matching = []
    all_novel = []
    per_genome_rows = []

    for split in ["test_id", "test_ood"]:
        mask = np.array([r["split"] == split and r["label"] != "ambiguous" for r in meta])
        s_meta = [r for r in meta if r["split"] == split and r["label"] != "ambiguous"]
        s_emb = emb[mask]
        s_preds = predict_batched(model, s_emb, device)

        # group by genome
        genome_windows = defaultdict(list)
        for i, r in enumerate(s_meta):
            genome_windows[r["genome_id"]].append((r, s_preds[i]))

        n_as_total = n_matched = n_missed = n_pred_total = n_novel = 0
        type_matched = Counter()
        type_total = Counter()
        type_concordant = 0
        type_matched_count = 0

        for genome_id, wins in genome_windows.items():
            # sort windows by contig + position
            wins.sort(key=lambda x: (x[0]["contig_id"], int(x[0]["window_start"])))

            # merge adjacent BGC-predicted windows into regions
            regions = []
            cur_region = None
            for r, pred in wins:
                is_bgc = pred[7] < is_neg_thr
                if is_bgc:
                    contig = r["contig_id"]
                    ws = int(r["window_start"])
                    we = int(r["window_end"])
                    if cur_region and cur_region["contig"] == contig and ws <= cur_region["end"] + 1:
                        cur_region["end"] = max(cur_region["end"], we)
                        cur_region["n_windows"] += 1
                        for j, c in enumerate(CLASSES):
                            cur_region["probs"][c] = max(cur_region["probs"][c], float(pred[j]))
                    else:
                        if cur_region and cur_region["end"] - cur_region["start"] >= 3000:
                            regions.append(cur_region)
                        cur_region = {
                            "genome_id": genome_id, "contig": contig,
                            "start": ws, "end": we, "n_windows": 1,
                            "probs": {c: float(pred[j]) for j, c in enumerate(CLASSES)},
                        }
                else:
                    if cur_region and cur_region["end"] - cur_region["start"] >= 3000:
                        regions.append(cur_region)
                    cur_region = None
            if cur_region and cur_region["end"] - cur_region["start"] >= 3000:
                regions.append(cur_region)

            for reg in regions:
                reg["primary_type"] = max(CLASSES, key=lambda c: reg["probs"][c])
                reg["length"] = reg["end"] - reg["start"]
                all_predicted.append(reg)

            # match vs antiSMASH
            phylum = wins[0][0]["phylum"] if wins else ""
            as_bgcs = []
            for contig_id in set(r["contig_id"] for r, _ in wins):
                as_bgcs.extend(bgcs.get((genome_id, contig_id), []))

            matched_as = set()
            matched_pred = set()
            for ai, ab in enumerate(as_bgcs):
                ab_s, ab_e = int(ab["start_bp"]), int(ab["end_bp"])
                ab_len = ab_e - ab_s
                ab_types = ab["bgc_types"].split(";")
                type_total.update(ab_types)
                best_ov = 0
                best_ri = -1
                for ri, reg in enumerate(regions):
                    if reg["contig"] != ab["contig_id"]:
                        continue
                    ov = max(0, min(reg["end"], ab_e) - max(reg["start"], ab_s))
                    min_len = min(ab_len, reg["length"])
                    if min_len > 0 and ov / min_len > best_ov:
                        best_ov = ov / min_len
                        best_ri = ri
                if best_ov >= 0.5:
                    matched_as.add(ai)
                    matched_pred.add(best_ri)
                    type_matched.update(ab_types)
                    # type concordance
                    pred_type = regions[best_ri]["primary_type"]
                    if pred_type in ab_types:
                        type_concordant += 1
                    type_matched_count += 1
                    all_matching.append({
                        "genome_id": genome_id, "phylum": phylum,
                        "as_start": ab_s, "as_end": ab_e,
                        "as_types": ab["bgc_types"],
                        "as_truncated": ab.get("is_truncated", "0"),
                        "pred_start": regions[best_ri]["start"],
                        "pred_end": regions[best_ri]["end"],
                        "pred_type": pred_type,
                        "overlap_frac": round(best_ov, 3),
                    })

            n_as = len(as_bgcs)
            nm = len(matched_as)
            n_as_total += n_as
            n_matched += nm
            n_missed += (n_as - nm)
            n_pred_total += len(regions)

            # novel candidates
            for ri, reg in enumerate(regions):
                if ri not in matched_pred:
                    # find distance to nearest antiSMASH BGC
                    min_dist = float("inf")
                    for ab in as_bgcs:
                        if ab["contig_id"] == reg["contig"]:
                            d = min(abs(reg["start"] - int(ab["end_bp"])),
                                    abs(reg["end"] - int(ab["start_bp"])))
                            min_dist = min(min_dist, d)
                    prob_max = max(reg["probs"].values())
                    conf = "low"
                    if prob_max > 0.8 and reg["length"] > 10000 and min_dist > 20000:
                        conf = "high"
                    elif prob_max > 0.5 and reg["length"] > 5000:
                        conf = "medium"
                    all_novel.append({
                        "genome_id": genome_id, "phylum": phylum,
                        "contig": reg["contig"],
                        "start": reg["start"], "end": reg["end"],
                        "length": reg["length"],
                        "primary_type": reg["primary_type"],
                        "max_prob": round(prob_max, 3),
                        "dist_to_nearest_bgc": min_dist if min_dist < 1e9 else -1,
                        "confidence": conf,
                        "split": split,
                    })
                    n_novel += 1

            recall_g = nm / n_as if n_as > 0 else float("nan")
            prec_g = nm / len(regions) if regions else float("nan")
            per_genome_rows.append({
                "genome_id": genome_id, "phylum": phylum, "split": split,
                "n_antismash_bgcs": n_as, "n_predicted_regions": len(regions),
                "n_matched": nm, "n_missed": n_as - nm,
                "n_novel_candidates": sum(1 for ri in range(len(regions)) if ri not in matched_pred),
                "recall": round(recall_g, 3) if n_as > 0 else "",
                "precision": round(prec_g, 3) if regions else "",
            })

        results[split] = {
            "n_antismash_bgcs": n_as_total,
            "n_matched": n_matched,
            "n_missed": n_missed,
            "recall": round(n_matched / max(1, n_as_total), 4),
            "n_predicted_regions": n_pred_total,
            "n_novel_candidates": n_novel,
            "precision": round(n_matched / max(1, n_pred_total), 4),
            "type_concordance": round(type_concordant / max(1, type_matched_count), 4),
            "per_type_recall": {
                c: round(type_matched.get(c, 0) / max(1, type_total.get(c, 0)), 4) for c in CLASSES
            },
        }
        print(f"  {split}: recall={results[split]['recall']:.3f} precision={results[split]['precision']:.3f} "
              f"concordance={results[split]['type_concordance']:.3f} "
              f"novel={n_novel}")

    with open(f"{out_dir}/genome_level_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{out_dir}/per_genome_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_genome_rows[0].keys()))
        w.writeheader()
        for r in per_genome_rows:
            w.writerow(r)
    with open(f"{out_dir}/predicted_regions.json", "w") as f:
        json.dump(all_predicted, f, indent=2)
    with open(f"{out_dir}/matching_results.json", "w") as f:
        json.dump(all_matching, f, indent=2)
    with open(f"{out_dir}/novel_candidates.json", "w") as f:
        json.dump(all_novel, f, indent=2)

    return results, all_novel, all_matching, per_genome_rows


# ─── 4.3 disagreement analysis ───

def step_4_3(all_novel, all_matching, per_genome_rows, bgc_csv, out_dir):
    print("\n=== Step 4.3: Disagreement analysis ===")
    os.makedirs(f"{out_dir}/false_negative_analysis", exist_ok=True)
    os.makedirs(f"{out_dir}/novel_candidates", exist_ok=True)

    # load BGC annotations for FN analysis
    bgc_info = {}
    with open(bgc_csv) as f:
        for row in csv.DictReader(f):
            bgc_info[row["bgc_id"]] = row

    # FN by type from matching results
    matched_bgcs = set()
    for m in all_matching:
        matched_bgcs.add((m["genome_id"], m["as_start"], m["as_end"]))

    fn_by_type = Counter()
    fn_by_truncation = {"truncated": {"missed": 0, "total": 0}, "intact": {"missed": 0, "total": 0}}
    fn_by_length = defaultdict(lambda: {"missed": 0, "total": 0})
    fn_by_phylum = defaultdict(lambda: {"missed": 0, "total": 0})

    for bgc in bgc_info.values():
        g = bgc["genome_id"]
        s, e = int(bgc["start_bp"]), int(bgc["end_bp"])
        types = bgc["bgc_types"].split(";")
        length = e - s
        trunc = "truncated" if bgc.get("is_truncated") == "1" else "intact"
        phy = bgc.get("phylum", "?")

        is_matched = (g, s, e) in matched_bgcs
        length_bin = "<5kb" if length < 5000 else "<10kb" if length < 10000 else "<20kb" if length < 20000 else "<50kb" if length < 50000 else ">=50kb"

        if not is_matched:
            for t in types:
                fn_by_type[t] += 1
        fn_by_truncation[trunc]["total"] += 1
        if not is_matched:
            fn_by_truncation[trunc]["missed"] += 1
        fn_by_length[length_bin]["total"] += 1
        if not is_matched:
            fn_by_length[length_bin]["missed"] += 1
        fn_by_phylum[phy]["total"] += 1
        if not is_matched:
            fn_by_phylum[phy]["missed"] += 1

    fn_summary = {
        "fn_by_type": dict(fn_by_type),
        "fn_by_truncation": fn_by_truncation,
        "fn_by_length": {k: {**v, "miss_rate": round(v["missed"]/max(1,v["total"]),3)} for k,v in sorted(fn_by_length.items())},
        "fn_by_phylum_top10": {k: {**v, "miss_rate": round(v["missed"]/max(1,v["total"]),3)}
                               for k,v in sorted(fn_by_phylum.items(), key=lambda x:-x[1]["total"])[:10]},
    }
    with open(f"{out_dir}/false_negative_analysis/fn_summary.json", "w") as f:
        json.dump(fn_summary, f, indent=2)
    print(f"  FN by type: {dict(fn_by_type)}")
    print(f"  FN by truncation: {fn_by_truncation}")

    # Novel candidate analysis
    conf_counts = Counter(n["confidence"] for n in all_novel)
    type_counts = Counter(n["primary_type"] for n in all_novel)
    phy_counts = Counter(n["phylum"] for n in all_novel)
    dists = [n["dist_to_nearest_bgc"] for n in all_novel if n["dist_to_nearest_bgc"] >= 0]

    novel_summary = {
        "total": len(all_novel),
        "by_confidence": dict(conf_counts),
        "by_type": dict(type_counts),
        "by_phylum_top10": dict(Counter(n["phylum"] for n in all_novel).most_common(10)),
        "dist_to_nearest_bgc": {
            "median": int(np.median(dists)) if dists else 0,
            "mean": int(np.mean(dists)) if dists else 0,
            "pct_within_20kb": round(sum(1 for d in dists if d < 20000)/max(1,len(dists)),3),
        },
    }
    with open(f"{out_dir}/novel_candidates/candidate_summary.json", "w") as f:
        json.dump(novel_summary, f, indent=2)

    # ranked list
    ranked = sorted(all_novel, key=lambda x: (-{"high":3,"medium":2,"low":1}[x["confidence"]], -x["max_prob"]))
    with open(f"{out_dir}/novel_candidates/candidates_ranked.csv", "w", newline="") as f:
        if ranked:
            w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
            w.writeheader()
            for r in ranked:
                w.writerow(r)

    print(f"  Novel candidates: {len(all_novel)} (high={conf_counts['high']}, med={conf_counts['medium']}, low={conf_counts['low']})")
    print(f"  Top high-conf:")
    for n in ranked[:5]:
        print(f"    {n['genome_id']} {n['contig']}:{n['start']}-{n['end']} {n['primary_type']} prob={n['max_prob']} dist={n['dist_to_nearest_bgc']}")

    return fn_summary, novel_summary


# ─── 4.4 MIBiG validation ───

def step_4_4(model, mibig_pkl, emb_dir, out_dir, device):
    print("\n=== Step 4.4: MIBiG external validation ===")
    os.makedirs(out_dir, exist_ok=True)

    # load MIBiG metadata
    mibig = pickle.load(open(mibig_pkl, "rb"))
    print(f"  MIBiG entries: {len(mibig)}")

    # load embeddings from shards
    shard_files = sorted(Path(emb_dir).glob("shard_*.pkl"))
    all_emb = []
    for sp in shard_files:
        d = pickle.load(open(sp, "rb"))
        all_emb.append(d["embeddings"])
    mibig_emb = np.vstack(all_emb)
    print(f"  Total shard embeddings: {mibig_emb.shape}")
    # Shards contain MIBiG + background; MIBiG entries are first
    mibig_emb = mibig_emb[:len(mibig)]
    print(f"  MIBiG embeddings (sliced): {mibig_emb.shape}")

    # predict
    preds = predict_batched(model, mibig_emb, device)

    # build labels
    from sklearn.metrics import roc_auc_score
    labels_7 = np.zeros((len(mibig), 7), dtype=np.float32)
    mapped = 0
    for i, entry in enumerate(mibig):
        cats = entry["categories"]
        found = False
        for j, c in enumerate(CLASSES):
            if c in cats:
                labels_7[i, j] = 1
                found = True
        if found:
            mapped += 1

    print(f"  Mapped: {mapped}/{len(mibig)}")

    # is_negative threshold from training
    is_neg_pred = preds[:, 7]
    bgc_detected = is_neg_pred < 0.5  # conservative threshold
    detection_rate = bgc_detected.sum() / len(mibig)
    print(f"  BGC detection rate: {detection_rate:.3f}")

    # per-type AUROC (only for types with ≥10 samples)
    per_type_auroc = {}
    per_type_recall = {}
    for j, c in enumerate(CLASSES):
        pos = labels_7[:, j].sum()
        if pos >= 10:
            auc = roc_auc_score(labels_7[:, j], preds[:, j])
            per_type_auroc[c] = round(float(auc), 4)
            rec = ((preds[:, j] > 0.5) & (labels_7[:, j] == 1)).sum() / max(1, int(pos))
            per_type_recall[c] = round(float(rec), 4)
        else:
            per_type_auroc[c] = None
            per_type_recall[c] = None

    mean_auroc = np.mean([v for v in per_type_auroc.values() if v is not None])

    # type concordance
    concordant = 0
    for i in range(len(mibig)):
        pred_type = CLASSES[np.argmax(preds[i, :7])]
        if pred_type in mibig[i]["categories"]:
            concordant += 1
    type_conc = concordant / len(mibig)

    metrics = {
        "n_total": len(mibig),
        "n_mapped": mapped,
        "bgc_detection_rate": round(float(detection_rate), 4),
        "per_type_auroc": per_type_auroc,
        "per_type_recall_at_05": per_type_recall,
        "mean_auroc": round(float(mean_auroc), 4),
        "type_concordance": round(float(type_conc), 4),
    }
    with open(f"{out_dir}/mibig_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


# ─── main ───

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Config B best checkpoint path")
    ap.add_argument("--emb", required=True, help="embeddings.npy")
    ap.add_argument("--meta", required=True, help="windows_metadata.csv")
    ap.add_argument("--bgc-csv", required=True, help="bgc_annotations.csv")
    ap.add_argument("--mibig-pkl", default="", help="mibig_variable.pkl")
    ap.add_argument("--mibig-emb-dir", default="", help="dir with shard_*.pkl")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # load model
    model = load_model(args.ckpt, device)
    print("Model loaded.")

    # load data
    print("Loading embeddings...")
    emb = np.load(args.emb)
    meta = []
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            meta.append(row)
    assert len(meta) == emb.shape[0]
    print(f"Loaded {emb.shape[0]} windows.")

    # 4.1
    thresholds = step_4_1(model, emb, meta, f"{args.out}/step4_1", device)

    # 4.2
    results, all_novel, all_matching, per_genome_rows = step_4_2(
        model, emb, meta, args.bgc_csv, thresholds, f"{args.out}/step4_2", device)

    # 4.3
    fn_summary, novel_summary = step_4_3(
        all_novel, all_matching, per_genome_rows, args.bgc_csv, f"{args.out}/step4_3")

    # 4.4
    if args.mibig_pkl and args.mibig_emb_dir:
        mibig_metrics = step_4_4(model, args.mibig_pkl, args.mibig_emb_dir,
                                  f"{args.out}/step4_4", device)
    else:
        print("\nSkipping 4.4 (no MIBiG paths provided)")
        mibig_metrics = None

    # final summary
    summary = {
        "thresholds": thresholds,
        "genome_level": results,
        "fn_summary": fn_summary,
        "novel_summary": novel_summary,
        "mibig": mibig_metrics,
    }
    with open(f"{args.out}/stage4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Stage 4 complete. Results in {args.out}/ ===")


if __name__ == "__main__":
    main()
