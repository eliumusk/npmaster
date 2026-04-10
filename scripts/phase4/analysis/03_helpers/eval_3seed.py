"""Evaluate Route B across 3 seeds, compute mean/std with per-bucket metrics."""
import csv, json, pickle, sys, os
from collections import defaultdict, Counter
import numpy as np
import torch

sys.path.insert(0, "/data/muskliu/np_master/code/np-master/bgc-gcd")
sys.path.insert(0, "/data/muskliu/np_master/code/np-master/scripts/eval")

from model.context_full_model import ContextFullModel
from ws1_score_swap import (call_regions, match_regions, compute_prediction_level_precision,
                            find_optimal_threshold, CLASSES, NumpyEncoder)

device = "cuda"
emb = np.load("/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy", mmap_mode="r")
with open("/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl", "rb") as f:
    meta = pickle.load(f)
with open("/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl", "rb") as f:
    contig_index = pickle.load(f)

as_bgcs = defaultdict(list)
with open("/data/muskliu/np_master/data/bgc_annotations.csv") as f:
    for row in csv.DictReader(f):
        as_bgcs[(row["genome_id"], row["contig_id"])].append(row)

all_results = {}
for seed in [0, 1, 2]:
    ckpt_path = f"/data/muskliu/np_master/data/phase3/route_b/best_route_b_s{seed}.pt"
    print(f"\n{'='*60}\n=== Seed {seed} ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ContextFullModel(input_dim=4096, proj_hidden=2048, proj_out=256, proj_dropout=0.3)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"alpha={model.gate.alpha.item():.4f} epoch={ckpt.get('epoch','?')}")

    all_preds = {}
    with torch.no_grad():
        for contig_key, indices in contig_index.items():
            e = np.stack([emb[i] for i in indices])
            e_t = torch.from_numpy(e).unsqueeze(0).to(device)
            v = torch.ones(1, len(indices), dtype=torch.bool, device=device)
            _, logits = model(e_t, v)
            probs = torch.sigmoid(logits[0]).cpu().numpy()
            for pos, g_idx in enumerate(indices):
                all_preds[g_idx] = probs[pos]

    val_scores, val_labels = [], []
    for g_idx, preds in all_preds.items():
        if meta["split"][g_idx] != "val" or meta["label"][g_idx] == "ambiguous":
            continue
        val_scores.append(preds[:7].max())
        val_labels.append(1 - int(meta["labels"][g_idx, 7]))
    thr_info = find_optimal_threshold(np.array(val_scores), np.array(val_labels))
    threshold = thr_info["threshold"]
    print(f"Threshold: {threshold:.4f}")

    seed_results = {"threshold": threshold, "alpha": float(model.gate.alpha.item())}

    for split in ["test_id", "test_ood"]:
        windows = []
        for g_idx, preds in all_preds.items():
            if meta["split"][g_idx] != split:
                continue
            windows.append({
                "genome_id": str(meta["genome_id"][g_idx]),
                "contig_id": str(meta["contig_id"][g_idx]),
                "window_start": int(meta["window_start"][g_idx]),
                "window_end": int(meta["window_start"][g_idx]) + 16384,
                "preds": [float(preds[j]) for j in range(8)],
            })
        windows.sort(key=lambda w: (w["genome_id"], w["contig_id"], w["window_start"]))
        scores = np.array([w["preds"][:7] for w in windows]).max(axis=1)
        regions = call_regions(windows, scores, threshold)

        split_genomes = set(w["genome_id"] for w in windows)
        n_as = sum(len(bl) for (g,c), bl in as_bgcs.items() if g in split_genomes)
        matched_as, matched_pred, pairs = match_regions(regions, as_bgcs)
        n_pm, n_pt = compute_prediction_level_precision(regions, as_bgcs)

        # Per-bucket
        bucket_m, bucket_t = Counter(), Counter()
        for (g,c), bl in as_bgcs.items():
            if g not in split_genomes: continue
            for ab in bl:
                k = (g, int(ab["start_bp"]), int(ab["end_bp"]))
                l = int(ab["end_bp"])-int(ab["start_bp"])
                b = "<5kb" if l<5000 else "5-10kb" if l<10000 else "10-20kb" if l<20000 else "20-50kb" if l<50000 else ">50kb"
                bucket_t[b] += 1
                if k in matched_as: bucket_m[b] += 1
        pbr = {b: bucket_m[b]/max(1,bucket_t[b]) for b in ["<5kb","5-10kb","10-20kb","20-50kb",">50kb"]}

        recall = len(matched_as)/max(1,n_as)
        ppred = n_pm/max(1,n_pt)
        seed_results[split] = {
            "recall": recall, "p_pred": ppred, "n_regions": len(regions),
            "n_as": n_as, "n_matched": len(matched_as),
            "per_bucket": pbr,
        }
        print(f"  {split}: recall={recall:.4f} p(pred)={ppred:.4f} regions={len(regions)}")
        print(f"    buckets: {' '.join(f'{k}={v:.3f}' for k,v in pbr.items())}")

    all_results[f"s{seed}"] = seed_results

# Compute mean/std
print("\n" + "="*60)
print("=== 3-seed Summary ===")
print("="*60)
for split in ["test_id", "test_ood"]:
    print(f"\n{split}:")
    for metric in ["recall", "p_pred"]:
        vals = [all_results[f"s{s}"][split][metric] for s in [0,1,2]]
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (s0={vals[0]:.4f} s1={vals[1]:.4f} s2={vals[2]:.4f})")
    for bucket in ["<5kb","5-10kb","10-20kb","20-50kb",">50kb"]:
        vals = [all_results[f"s{s}"][split]["per_bucket"][bucket] for s in [0,1,2]]
        print(f"  {bucket}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (s0={vals[0]:.4f} s1={vals[1]:.4f} s2={vals[2]:.4f})")

# Save
os.makedirs("/data/muskliu/np_master/data/phase3/route_b/eval", exist_ok=True)
with open("/data/muskliu/np_master/data/phase3/route_b/eval/3seed_results.json", "w") as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
print("\nSaved to /data/muskliu/np_master/data/phase3/route_b/eval/3seed_results.json")
