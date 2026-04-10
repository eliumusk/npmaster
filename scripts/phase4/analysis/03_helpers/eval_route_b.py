"""Route B full evaluation: contig-level inference with end-to-end model."""
import csv, json, pickle, sys
from collections import defaultdict, Counter
import numpy as np
import torch

sys.path.insert(0, "/data/muskliu/np_master/code/np-master/bgc-gcd")
sys.path.insert(0, "/data/muskliu/np_master/code/np-master/scripts/eval")

from model.context_full_model import ContextFullModel
from ws1_score_swap import (call_regions, match_regions, compute_prediction_level_precision,
                            classify_unmatched, find_optimal_threshold, CLASSES, NumpyEncoder)

device = "cuda"
print("Loading model...")
ckpt = torch.load("/data/muskliu/np_master/data/phase3/route_b/best_route_b_s0.pt",
                  map_location=device, weights_only=False)
model = ContextFullModel(input_dim=4096, proj_hidden=2048, proj_out=256, proj_dropout=0.3)
model.load_state_dict(ckpt["model"])
model.to(device).eval()
alpha = model.gate.alpha.item()
print(f"  alpha={alpha:.4f}, epoch={ckpt.get('epoch','?')}, rf1={ckpt.get('val_region_f1','?')}")

print("Loading data...")
emb = np.load("/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy", mmap_mode="r")
with open("/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl", "rb") as f:
    meta = pickle.load(f)
with open("/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl", "rb") as f:
    contig_index = pickle.load(f)

as_bgcs = defaultdict(list)
bgc_lengths = {}
with open("/data/muskliu/np_master/data/bgc_annotations.csv") as f:
    for row in csv.DictReader(f):
        as_bgcs[(row["genome_id"], row["contig_id"])].append(row)
        key = (row["genome_id"], int(row["start_bp"]), int(row["end_bp"]))
        bgc_lengths[key] = int(row["end_bp"]) - int(row["start_bp"])

# Predict all windows contig-by-contig
print("Predicting...")
all_preds = {}
with torch.no_grad():
    for ci, (contig_key, indices) in enumerate(contig_index.items()):
        L = len(indices)
        e = np.stack([emb[i] for i in indices])
        e_t = torch.from_numpy(e).unsqueeze(0).to(device)
        v = torch.ones(1, L, dtype=torch.bool, device=device)
        _, logits = model(e_t, v)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        for pos, g_idx in enumerate(indices):
            all_preds[g_idx] = probs[pos]
        if (ci+1) % 10000 == 0:
            print(f"  {ci+1}/{len(contig_index)} contigs")

print(f"Predicted {len(all_preds)} windows")

# Threshold on val
val_scores, val_labels = [], []
for g_idx, preds in all_preds.items():
    if meta["split"][g_idx] != "val" or meta["label"][g_idx] == "ambiguous":
        continue
    val_scores.append(preds[:7].max())
    val_labels.append(1 - int(meta["labels"][g_idx, 7]))
thr_info = find_optimal_threshold(np.array(val_scores), np.array(val_labels))
threshold = thr_info["threshold"]
print(f"Threshold: {threshold:.4f} (F1={thr_info['f1']:.4f} P={thr_info['precision']:.4f} R={thr_info['recall']:.4f})")

results = {"threshold": thr_info, "alpha": alpha, "epoch": ckpt.get("epoch", -1)}

for split in ["test_id", "test_ood"]:
    print(f"\n{'='*60}\n=== {split} ===")
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
    recall = len(matched_as)/max(1,n_as)
    ppred = n_pm/max(1,n_pt)
    unmatched = classify_unmatched(regions, matched_pred, as_bgcs)
    cats = Counter(u["category"] for u in unmatched)

    # Per-type recall
    type_m, type_t = Counter(), Counter()
    for (g,c), bl in as_bgcs.items():
        if g not in split_genomes: continue
        for ab in bl:
            k = (g, int(ab["start_bp"]), int(ab["end_bp"]))
            for t in ab["bgc_types"].split(";"):
                type_t[t] += 1
                if k in matched_as: type_m[t] += 1
    ptr = {c: round(type_m.get(c,0)/max(1,type_t.get(c,0)),4) for c in CLASSES}

    # Per-bucket recall
    bucket_m, bucket_t = Counter(), Counter()
    for (g,c), bl in as_bgcs.items():
        if g not in split_genomes: continue
        for ab in bl:
            k = (g, int(ab["start_bp"]), int(ab["end_bp"]))
            l = int(ab["end_bp"])-int(ab["start_bp"])
            b = "<5kb" if l<5000 else "5-10kb" if l<10000 else "10-20kb" if l<20000 else "20-50kb" if l<50000 else ">50kb"
            bucket_t[b] += 1
            if k in matched_as: bucket_m[b] += 1
    pbr = {b: round(bucket_m[b]/max(1,bucket_t[b]),4) for b in ["<5kb","5-10kb","10-20kb","20-50kb",">50kb"]}

    print(f"  Recall: {recall:.4f} ({len(matched_as)}/{n_as})")
    print(f"  P(pred): {ppred:.4f}")
    print(f"  Regions: {len(regions)}")
    print(f"  Unmatched: {len(unmatched)} (clean={cats.get('clean_missed',0)})")
    print(f"  Per-type: {ptr}")
    print(f"  Per-bucket: {pbr}")

    results[split] = {
        "recall": round(recall,4), "precision_pred_level": round(ppred,4),
        "n_regions": len(regions), "n_as": n_as, "n_matched": len(matched_as),
        "per_type_recall": ptr, "per_bucket_recall": pbr,
        "unmatched": {"boundary": cats.get("boundary_expansion",0),
                      "ambiguous": cats.get("ambiguous_unmatched",0),
                      "clean": cats.get("clean_missed",0)},
    }

import os
os.makedirs("/data/muskliu/np_master/data/phase3/route_b/eval", exist_ok=True)
with open("/data/muskliu/np_master/data/phase3/route_b/eval/results.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print("\nSaved to /data/muskliu/np_master/data/phase3/route_b/eval/results.json")
