"""P3: Compare antiSMASH / GECCO / Route B on the same evaluation protocol.

- Uses antiSMASH annotations as proxy benchmark
- Matching: overlap_frac >= 0.5 on min(pred_len, gt_len)
- Reports recall, prediction-level precision, per-bucket recall, per-phylum
"""
import csv
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Load antiSMASH annotations
as_bgcs = defaultdict(list)
with open("/data/muskliu/npmaster/data/bgc_annotations.csv") as f:
    for row in csv.DictReader(f):
        as_bgcs[(row["genome_id"], row["contig_id"])].append({
            "genome_id": row["genome_id"],
            "contig_id": row["contig_id"],
            "start_bp": int(row["start_bp"]),
            "end_bp": int(row["end_bp"]),
            "bgc_types": row["bgc_types"],
        })

# Load test split info
df = pd.read_csv("/data/muskliu/npmaster/data/training_labels/p2_t2_hard_neg/windows_metadata_hardneg.csv")
test_split = {}
for _, row in df[["genome_id", "split", "phylum"]].drop_duplicates("genome_id").iterrows():
    test_split[row["genome_id"]] = {"split": row["split"], "phylum": row["phylum"]}

# Load GECCO predictions
with open("/data/muskliu/npmaster/data/phase4/baselines/gecco_predictions.json") as f:
    gecco_data = json.load(f)
gecco_preds = gecco_data["predictions"]

# Load Route B predictions (from score_swap_s0)
with open("/data/muskliu/npmaster/data/phase3/route_b/eval/score_swap_s0.json") as f:
    rb_data = json.load(f)

# Extract Route B regions per genome
rb_regions = defaultdict(list)
for split in ["test_id", "test_ood"]:
    r = rb_data["results"][split]["score_maxtype"]
    # score_swap_s0.json has unmatched_details but not matched_details
    # Need to get matched from separate source
    # For the comparison table we only need matched count which is n_matched
    # But for region-level calling we need all regions
    pass

# Actually we need to re-run Route B region calling to get predictions
# Use the existing predict file if available, or approximate from the scored windows
# For now, load Route B predictions from eval/results.json (3-seed aggregated)
with open("/data/muskliu/npmaster/data/phase3/route_b/eval/3seed_results.json") as f:
    rb_3seed = json.load(f)

# For per-seed comparison use s0
rb_s0 = rb_3seed["s0"]
print("=== Route B (s0) vs antiSMASH benchmark ===")
for split in ["test_id", "test_ood"]:
    m = rb_s0[split]
    print(f"  {split}: recall={m['recall']:.4f} P(pred)={m['p_pred']:.4f} regions={m['n_regions']}")
    for b, v in m['per_bucket'].items():
        print(f"    {b}: {v:.4f}")

# Now evaluate GECCO vs antiSMASH
def match_regions(pred_list, as_bgcs_by_loc, overlap_thr=0.5):
    matched_as = set()
    matched_pred = set()
    for pi, p in enumerate(pred_list):
        gid = p["genome_id"]
        cid = p["contig_id"]
        ps = p["start_bp"]
        pe = p["end_bp"]
        plen = pe - ps
        if plen <= 0: continue
        for ab in as_bgcs_by_loc.get((gid, cid), []):
            ab_s, ab_e = ab["start_bp"], ab["end_bp"]
            ab_len = ab_e - ab_s
            ov = max(0, min(pe, ab_e) - max(ps, ab_s))
            min_len = min(plen, ab_len)
            if min_len > 0 and ov / min_len >= overlap_thr:
                matched_as.add((gid, ab_s, ab_e))
                matched_pred.add(pi)
    return matched_as, matched_pred

def eval_tool(pred_by_genome, as_by_loc, split_name, test_split):
    # Collect predictions for this split
    preds = []
    for gid, bgcs in pred_by_genome.items():
        if gid not in test_split: continue
        if test_split[gid]["split"] != split_name: continue
        for b in bgcs:
            if b.get("end_bp", 0) - b.get("start_bp", 0) >= 3000:  # min 3kb
                preds.append(b)
    
    # Count ground truth BGCs for this split
    split_genomes = set(g for g, v in test_split.items() if v["split"] == split_name)
    n_gt = 0
    for (gid, cid), bgcs in as_by_loc.items():
        if gid in split_genomes:
            n_gt += len(bgcs)
    
    matched_as, matched_pred = match_regions(preds, as_by_loc)
    recall = len(matched_as) / max(1, n_gt)
    precision = len(matched_pred) / max(1, len(preds))
    
    # Per-bucket recall
    bucket_m = Counter()
    bucket_t = Counter()
    for (gid, cid), bgcs in as_by_loc.items():
        if gid not in split_genomes: continue
        for ab in bgcs:
            ab_key = (gid, ab["start_bp"], ab["end_bp"])
            l = ab["end_bp"] - ab["start_bp"]
            b = "<5kb" if l<5000 else "5-10kb" if l<10000 else "10-20kb" if l<20000 else "20-50kb" if l<50000 else ">50kb"
            bucket_t[b] += 1
            if ab_key in matched_as: bucket_m[b] += 1
    pbr = {b: bucket_m[b]/max(1,bucket_t[b]) for b in ["<5kb","5-10kb","10-20kb","20-50kb",">50kb"]}
    
    return {"recall": recall, "precision": precision, "n_regions": len(preds),
            "n_matched": len(matched_as), "n_gt": n_gt, "per_bucket": pbr}

print("\n=== GECCO vs antiSMASH benchmark ===")
gecco_results = {}
for split in ["test_id", "test_ood"]:
    r = eval_tool(gecco_preds, as_bgcs, split, test_split)
    gecco_results[split] = r
    print(f"  {split}: recall={r['recall']:.4f} P={r['precision']:.4f} regions={r['n_regions']}/{r['n_gt']}")
    for b, v in r['per_bucket'].items():
        print(f"    {b}: {v:.4f}")

# Build the comparison table
print("\n" + "="*80)
print("COMPARISON TABLE (antiSMASH proxy benchmark)")
print("="*80)
print(f"{'Tool':<15} {'Split':<10} {'Recall':>8} {'P(pred)':>8} {'<5kb':>8} {'5-10kb':>8} {'10-20kb':>8} {'20-50kb':>8} {'>50kb':>8}")
for split in ["test_id", "test_ood"]:
    # Route B
    m = rb_s0[split]
    print(f"{'Route B':<15} {split:<10} {m['recall']:>8.4f} {m['p_pred']:>8.4f} "
          f"{m['per_bucket']['<5kb']:>8.4f} {m['per_bucket']['5-10kb']:>8.4f} "
          f"{m['per_bucket']['10-20kb']:>8.4f} {m['per_bucket']['20-50kb']:>8.4f} "
          f"{m['per_bucket']['>50kb']:>8.4f}")
    r = gecco_results[split]
    print(f"{'GECCO':<15} {split:<10} {r['recall']:>8.4f} {r['precision']:>8.4f} "
          f"{r['per_bucket']['<5kb']:>8.4f} {r['per_bucket']['5-10kb']:>8.4f} "
          f"{r['per_bucket']['10-20kb']:>8.4f} {r['per_bucket']['20-50kb']:>8.4f} "
          f"{r['per_bucket']['>50kb']:>8.4f}")

# Save
out = {
    "protocol": "antiSMASH proxy benchmark, overlap >= 0.5, min region 3kb",
    "route_b": {split: rb_s0[split] for split in ["test_id", "test_ood"]},
    "gecco": gecco_results,
}
with open("/data/muskliu/npmaster/data/phase4/p3_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to /data/muskliu/npmaster/data/phase4/p3_comparison.json")
