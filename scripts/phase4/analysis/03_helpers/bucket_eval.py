"""Quick per-bucket recall analysis for Phase 3 Route A."""
import csv, json, pickle, sys
from collections import defaultdict, Counter
import numpy as np
import torch

sys.path.insert(0, "/data/muskliu/np_master/code/np-master/bgc-gcd")
sys.path.insert(0, "/data/muskliu/np_master/code/np-master/scripts/eval")

from model.context_model import ContextBGCModel
from ws1_score_swap import call_regions, match_regions, CLASSES, find_optimal_threshold

# Load model
device = "cuda"
ckpt = torch.load("/data/muskliu/np_master/data/phase3/route_a/best_route_a_s0.pt", map_location=device, weights_only=False)
model = ContextBGCModel(proj_dim=256, n_labels=8)
model.load_state_dict(ckpt["model"])
model.to(device).eval()
print(f"Loaded model: alpha={model.gate.alpha.item():.4f}")

# Load data
proj = np.load("/data/muskliu/np_master/data/phase3/route_a/projections_256.npy", mmap_mode="r")
with open("/data/muskliu/np_master/data/phase3/route_a/meta_arrays.pkl", "rb") as f:
    meta = pickle.load(f)
with open("/data/muskliu/np_master/data/phase3/route_a/contig_index.pkl", "rb") as f:
    contig_index = pickle.load(f)

# Load BGC annotations
as_bgcs = defaultdict(list)
bgc_lengths = {}
with open("/data/muskliu/np_master/data/bgc_annotations.csv") as f:
    for row in csv.DictReader(f):
        as_bgcs[(row["genome_id"], row["contig_id"])].append(row)
        key = (row["genome_id"], int(row["start_bp"]), int(row["end_bp"]))
        bgc_lengths[key] = int(row["end_bp"]) - int(row["start_bp"])

# Predict all windows contig-by-contig
all_preds = {}
with torch.no_grad():
    for contig_key, indices in contig_index.items():
        p = np.stack([proj[i] for i in indices])
        p_t = torch.from_numpy(p).unsqueeze(0).to(device)
        v = torch.ones(1, len(indices), dtype=torch.bool, device=device)
        _, logits = model(p_t, v)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        for pos, g_idx in enumerate(indices):
            all_preds[g_idx] = probs[pos]

print(f"Predicted {len(all_preds)} windows")

# Find threshold on val
val_scores, val_labels = [], []
for g_idx, preds in all_preds.items():
    if meta["split"][g_idx] != "val" or meta["label"][g_idx] == "ambiguous":
        continue
    val_scores.append(preds[:7].max())
    val_labels.append(1 - int(meta["labels"][g_idx, 7]))
thr_info = find_optimal_threshold(np.array(val_scores), np.array(val_labels))
threshold = thr_info["threshold"]
print(f"Threshold: {threshold:.4f} (val F1={thr_info['f1']:.4f})")

# Evaluate per split
for split in ["test_id", "test_ood"]:
    print(f"\n{'='*60}")
    print(f"=== {split} ===")

    # Build window list sorted by position
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
            "g_idx": g_idx,
        })
    windows.sort(key=lambda w: (w["genome_id"], w["contig_id"], w["window_start"]))
    scores = np.array([w["preds"][:7] for w in windows]).max(axis=1)

    regions = call_regions(windows, scores, threshold)
    matched_as, matched_pred, pairs = match_regions(regions, as_bgcs)

    # Count BGCs by split
    split_genomes = set(w["genome_id"] for w in windows)
    n_total = 0
    for (gid, cid), bgc_list in as_bgcs.items():
        if gid in split_genomes:
            n_total += len(bgc_list)

    recall = len(matched_as) / max(1, n_total)
    from ws1_score_swap import compute_prediction_level_precision
    n_pm, n_pt = compute_prediction_level_precision(regions, as_bgcs)
    ppred = n_pm / max(1, n_pt)

    print(f"  Overall recall: {recall:.4f} ({len(matched_as)}/{n_total})")
    print(f"  P(pred): {ppred:.4f}")
    print(f"  Regions: {len(regions)}")

    # Per-bucket recall
    bucket_matched = Counter()
    bucket_total = Counter()
    for (gid, cid), bgc_list in as_bgcs.items():
        if gid not in split_genomes:
            continue
        for ab in bgc_list:
            ab_key = (gid, int(ab["start_bp"]), int(ab["end_bp"]))
            bgc_len = int(ab["end_bp"]) - int(ab["start_bp"])
            if bgc_len < 5000: b = "<5kb"
            elif bgc_len < 10000: b = "5-10kb"
            elif bgc_len < 20000: b = "10-20kb"
            elif bgc_len < 50000: b = "20-50kb"
            else: b = ">50kb"
            bucket_total[b] += 1
            if ab_key in matched_as:
                bucket_matched[b] += 1

    print(f"\n  Per-bucket recall:")
    for b in ["<5kb", "5-10kb", "10-20kb", "20-50kb", ">50kb"]:
        m = bucket_matched[b]
        t = bucket_total[b]
        r = m / max(1, t)
        print(f"    {b:>8s}: {r:.4f} ({m}/{t})")
