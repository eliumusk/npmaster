"""Route B evaluation producing ws1_score_swap-compatible JSON for discovery benchmark."""
import csv, json, pickle, sys, os
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/data/muskliu/np_master/code/np-master/bgc-gcd")
sys.path.insert(0, "/data/muskliu/np_master/code/np-master/scripts/eval")

from model.context_full_model import ContextFullModel
from ws1_score_swap import (call_regions, match_regions, compute_prediction_level_precision,
                            classify_unmatched, find_optimal_threshold, CLASSES, NumpyEncoder)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

device = "cuda"
ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
model = ContextFullModel(input_dim=4096, proj_hidden=2048, proj_out=256, proj_dropout=0.3)
model.load_state_dict(ckpt["model"])
model.to(device).eval()
print(f"alpha={model.gate.alpha.item():.4f}")

emb = np.load("/data/muskliu/np_master/data/embeddings/edge_padding/embeddings_merged.npy", mmap_mode="r")
with open("/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl", "rb") as f:
    meta = pickle.load(f)
with open("/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl", "rb") as f:
    contig_index = pickle.load(f)

as_bgcs = defaultdict(list)
with open("/data/muskliu/np_master/data/bgc_annotations.csv") as f:
    for row in csv.DictReader(f):
        as_bgcs[(row["genome_id"], row["contig_id"])].append(row)

# Predict all windows
print("Predicting...")
all_preds = {}
with torch.no_grad():
    for ci, (contig_key, indices) in enumerate(contig_index.items()):
        e = np.stack([emb[i] for i in indices])
        e_t = torch.from_numpy(e).unsqueeze(0).to(device)
        v = torch.ones(1, len(indices), dtype=torch.bool, device=device)
        _, logits = model(e_t, v)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        for pos, g_idx in enumerate(indices):
            all_preds[g_idx] = probs[pos]

# Threshold
val_scores, val_labels = [], []
for g_idx, preds in all_preds.items():
    if meta["split"][g_idx] != "val" or meta["label"][g_idx] == "ambiguous":
        continue
    val_scores.append(preds[:7].max())
    val_labels.append(1 - int(meta["labels"][g_idx, 7]))
thr_info = find_optimal_threshold(np.array(val_scores), np.array(val_labels))
threshold = thr_info["threshold"]
print(f"Threshold: {threshold:.4f}")

# Build ws1_score_swap-compatible output
output = {
    "evaluation_type": "phase3_route_b",
    "checkpoint": args.ckpt,
    "alpha": float(model.gate.alpha.item()),
    "thresholds": {"score_maxtype": thr_info},
    "results": {},
}

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

    # Add missing fields expected by discovery benchmark
    for reg in regions:
        reg["genome_id"] = reg["genome_id"]

    split_genomes = set(w["genome_id"] for w in windows)
    n_as = sum(len(bl) for (g,c), bl in as_bgcs.items() if g in split_genomes)
    matched_as, matched_pred, pairs = match_regions(regions, as_bgcs)
    n_pm, n_pt = compute_prediction_level_precision(regions, as_bgcs)
    unmatched = classify_unmatched(regions, matched_pred, as_bgcs)

    output["results"][split] = {
        "score_maxtype": {
            "threshold": threshold,
            "n_antismash_bgcs": n_as,
            "n_matched": len(matched_as),
            "recall": round(len(matched_as)/max(1,n_as), 4),
            "n_predicted_regions": len(regions),
            "precision_pred_level": round(n_pm/max(1,n_pt), 4),
            "n_unmatched": len(unmatched),
            "unmatched_details": unmatched,  # CRITICAL: all unmatched with full info
        }
    }
    print(f"{split}: recall={output['results'][split]['score_maxtype']['recall']} "
          f"p(pred)={output['results'][split]['score_maxtype']['precision_pred_level']} "
          f"regions={len(regions)} unmatched={len(unmatched)}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"Saved to {args.out}")
