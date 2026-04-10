"""D2 analysis: compute signature density for candidates vs matched null."""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

NULL_HITS = Path("/data/muskliu/np_master/data/phase4/d2_matched_null/null_hits")
OUT_DIR = "/data/muskliu/np_master/data/phase4/d2_matched_null"

# Load signature domain list
sig_domains = {
    "PF00109", "PF02801", "PF00698", "PF00975", "PF14765", "PF08659",
    "PF00106", "PF00668", "PF00501", "PF00550", "PF13193", "PF00667",
    "PF01397", "PF03936", "PF00494", "PF13575", "PF04055", "PF00881",
    "PF00535", "PF00534", "PF00067", "PF00891", "PF08241", "PF00155",
    "PF13561", "PF00441", "PF02771",
}

# Parse hmmscan tbl output for each null region
def parse_tbl(tbl_path):
    if not tbl_path.exists():
        return []
    pfams = set()
    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 2:
                pfam_acc = parts[1].split(".")[0]  # remove version
                pfams.add(pfam_acc)
    return pfams

# Load null index
with open(f"{OUT_DIR}/null_index.json") as f:
    null_index = json.load(f)

# For each null, parse hmmscan + count
null_results = []
for n in null_index:
    hits_dir = NULL_HITS / n["null_id"]
    tbl = hits_dir / "pfam.tbl"
    pfams = parse_tbl(tbl)
    sig_count = len(pfams & sig_domains)
    biosyn_count = len(pfams)  # biosyn_pfam is the whole DB, so all hits count
    length_kb = n["length"] / 1000
    null_results.append({
        "null_id": n["null_id"],
        "cand_idx": n["cand_idx"],
        "length_kb": length_kb,
        "sig_count": sig_count,
        "biosyn_count": biosyn_count,
        "sig_density": sig_count / max(length_kb, 1),
        "biosyn_density": biosyn_count / max(length_kb, 1),
    })

print(f"Null hits analyzed: {len(null_results)}")

# Load candidates with their existing sig counts
with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    disc = json.load(f)
candidates = [c for c in disc["candidate_results"] if not c.get("skip", False)][:50]

cand_results = []
for i, c in enumerate(candidates):
    length_kb = c["length"] / 1000
    cand_results.append({
        "cand_idx": i,
        "length_kb": length_kb,
        "sig_count": c.get("n_signature", 0),
        "biosyn_count": c.get("n_biosyn", 0),
        "sig_density": c.get("n_signature", 0) / max(length_kb, 1),
        "biosyn_density": c.get("n_biosyn", 0) / max(length_kb, 1),
        "genome": c["genome_id"],
        "start": c["start"],
        "end": c["end"],
        "max_score": c["max_score"],
    })

# Aggregate comparison
c_sig_density = np.array([c["sig_density"] for c in cand_results])
c_sig_count = np.array([c["sig_count"] for c in cand_results])
n_sig_density = np.array([n["sig_density"] for n in null_results])
n_sig_count = np.array([n["sig_count"] for n in null_results])

print("\n" + "="*60)
print("D2 MATCHED NULL RESULTS")
print("="*60)
print(f"\nCandidates: n={len(c_sig_density)}")
print(f"Matched null controls: n={len(n_sig_density)}")

print(f"\nSignature count:")
print(f"  Candidates: mean={c_sig_count.mean():.2f}, median={np.median(c_sig_count):.0f}, max={c_sig_count.max()}")
print(f"  Matched null: mean={n_sig_count.mean():.2f}, median={np.median(n_sig_count):.0f}, max={n_sig_count.max()}")

print(f"\nSignature density (domains/kb):")
print(f"  Candidates: mean={c_sig_density.mean():.4f}, median={np.median(c_sig_density):.4f}")
print(f"  Matched null: mean={n_sig_density.mean():.4f}, median={np.median(n_sig_density):.4f}")

# Enrichment
if np.median(n_sig_density) > 0:
    enrichment_median = np.median(c_sig_density) / np.median(n_sig_density)
else:
    enrichment_median = float('inf') if np.median(c_sig_density) > 0 else 0
if n_sig_density.mean() > 0:
    enrichment_mean = c_sig_density.mean() / n_sig_density.mean()
else:
    enrichment_mean = float('inf') if c_sig_density.mean() > 0 else 0
print(f"\nEnrichment (candidates / matched null):")
print(f"  By median: {enrichment_median:.2f}x")
print(f"  By mean:   {enrichment_mean:.2f}x")

# Mann-Whitney U test
from scipy.stats import mannwhitneyu
try:
    stat, p = mannwhitneyu(c_sig_density, n_sig_density, alternative="greater")
    print(f"\nMann-Whitney U test (one-sided, candidates > null):")
    print(f"  U = {stat:.1f}, p = {p:.4e}")
    significant = p < 0.01
    print(f"  Significant at α=0.01: {significant}")
except Exception as e:
    print(f"Stats failed: {e}")

# Per-candidate z-scores
# Each candidate's 10 nulls are grouped by cand_idx
null_by_cand = {}
for n in null_results:
    ci = n["cand_idx"]
    null_by_cand.setdefault(ci, []).append(n["sig_density"])

z_scores = []
for c in cand_results:
    ci = c["cand_idx"]
    nulls = null_by_cand.get(ci, [])
    if len(nulls) < 2:
        continue
    mu = np.mean(nulls)
    sd = np.std(nulls)
    if sd > 0:
        z = (c["sig_density"] - mu) / sd
    else:
        z = (c["sig_density"] - mu) / 0.01  # pseudo-std for zero variance
    z_scores.append({"cand_idx": ci, "z_score": float(z), "n_nulls": len(nulls),
                     "cand_density": c["sig_density"], "null_mean": mu, "null_std": sd})

if z_scores:
    z_arr = np.array([z["z_score"] for z in z_scores])
    print(f"\nPer-candidate z-scores (n={len(z_scores)}):")
    print(f"  mean z: {z_arr.mean():.2f}, median: {np.median(z_arr):.2f}")
    print(f"  z > 0 (above null mean): {(z_arr > 0).sum()}/{len(z_arr)} ({(z_arr > 0).mean()*100:.0f}%)")
    print(f"  z > 1: {(z_arr > 1).sum()}/{len(z_arr)} ({(z_arr > 1).mean()*100:.0f}%)")
    print(f"  z > 2: {(z_arr > 2).sum()}/{len(z_arr)} ({(z_arr > 2).mean()*100:.0f}%)")
    print(f"  z > 3: {(z_arr > 3).sum()}/{len(z_arr)} ({(z_arr > 3).mean()*100:.0f}%)")

# Save
output = {
    "candidates": cand_results,
    "null_results": null_results,
    "z_scores": z_scores,
    "summary": {
        "n_candidates": len(cand_results),
        "n_null": len(null_results),
        "cand_sig_density_median": float(np.median(c_sig_density)),
        "null_sig_density_median": float(np.median(n_sig_density)),
        "enrichment_median": float(enrichment_median),
        "enrichment_mean": float(enrichment_mean),
        "mann_whitney_p": float(p) if 'p' in dir() else None,
        "frac_z_gt_2": float((z_arr > 2).mean()) if z_scores else None,
        "frac_z_gt_3": float((z_arr > 3).mean()) if z_scores else None,
    }
}
with open(f"{OUT_DIR}/d2_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUT_DIR}/d2_results.json")
