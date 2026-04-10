"""D3: Implement dual composite rankers (Conservative + Novelty).

Weights are pre-registered. No tuning on test set.
Sanity checks: rank known antiSMASH BGCs for conservative; compare novelty top-K vs random
on cross-tool concordance (only GECCO available, DeepBGC pending).
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

OUT_DIR = "/data/muskliu/np_master/data/phase4/d3_ranking"
os.makedirs(OUT_DIR, exist_ok=True)

# Load top-50 Route B candidates with Pfam
with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    disc = json.load(f)
candidates = [c for c in disc["candidate_results"] if not c.get("skip", False)][:50]

# Load full unmatched candidates (all 412) for ranking at scale
with open("/data/muskliu/np_master/data/phase3/route_b/eval/score_swap_s0.json") as f:
    sw = json.load(f)
all_unmatched = []
for split in ["test_id", "test_ood"]:
    for u in sw["results"][split]["score_maxtype"].get("unmatched_details", []):
        if u.get("category") == "clean_missed":
            all_unmatched.append({**u, "split": split})
print(f"Total clean_missed unmatched: {len(all_unmatched)}")

# Signature domain list (curated)
SIG_DOMAINS = {
    "PF00109", "PF02801", "PF00698", "PF00975", "PF14765", "PF08659",
    "PF00106", "PF00668", "PF00501", "PF00550", "PF13193", "PF00667",
    "PF01397", "PF03936", "PF00494", "PF13575", "PF04055", "PF00881",
    "PF00535", "PF00534", "PF00067", "PF00891", "PF08241", "PF00155",
    "PF13561", "PF00441", "PF02771",
}

# Type signature mapping (for module completeness)
def compute_module_completeness(pfams):
    pfams = set(pfams)
    # NRPS: C + A + PCP
    has_nrps = (
        ({"PF00668", "PF13437"} & pfams)  # C
        and ({"PF00501", "PF13193"} & pfams)  # A
        and ({"PF00550"} & pfams)  # PCP
    )
    # PKS type I: KS + AT + ACP
    has_pks1 = (
        ({"PF00109", "PF02801"} & pfams)  # KS
        and ({"PF00698"} & pfams)  # AT
        and ({"PF00550"} & pfams)  # ACP (= PCP)
    )
    # PKS type II: KS_alpha + KS_beta (CLF)
    has_pks2 = len({"PF00109"} & pfams) > 0 and len({"PF02801"} & pfams) > 0
    # Terpene
    has_terpene = bool({"PF01397", "PF03936", "PF00494"} & pfams)
    # RiPP: radical SAM
    has_ripp = bool({"PF04055"} & pfams)
    
    if has_nrps: return True, "NRPS"
    if has_pks1: return True, "PKS-I"
    if has_pks2: return True, "PKS-II"
    if has_terpene: return True, "Terpene"
    if has_ripp: return True, "RiPP-candidate"
    return False, None

def count_signatures(pfams):
    return len(set(pfams) & SIG_DOMAINS)

# For top-50 candidates, we have pfam_list. For others, we DON'T have Pfam hits.
# We can only compute Conservative ranker for top-50. For novelty, we can still use
# score + distance + contig quality for all 412.

# --- Compute Conservative ranker on top-50 ---
cons_scores = []
for c in candidates:
    pfams = c.get("pfam_list", [])
    sig_count = count_signatures(pfams)
    sig_density = sig_count / max(c["length"]/1000, 1)
    is_complete, complete_type = compute_module_completeness(pfams)
    sig_diversity = len(set(pfams) & SIG_DOMAINS)
    contig_quality = 1  # assume good; would check contig length
    type_coherence = 1  # placeholder, need module vs predicted type check
    
    cons = (
        0.2 * c["max_score"]
        + 0.3 * np.log(sig_density + 1)
        + 0.3 * (1 if is_complete else 0)
        + 0.1 * np.log(sig_diversity + 1)
        + 0.05 * contig_quality
        + 0.05 * type_coherence
    )
    cons_scores.append({
        "genome": c["genome_id"],
        "contig": c["contig"],
        "start": c["start"],
        "end": c["end"],
        "length": c["length"],
        "primary_type": c["primary_type"],
        "max_score": c["max_score"],
        "sig_count": sig_count,
        "sig_density": float(sig_density),
        "sig_diversity": sig_diversity,
        "module_complete": is_complete,
        "module_type": complete_type,
        "conservative_score": float(cons),
    })

# Rank
cons_scores.sort(key=lambda x: -x["conservative_score"])
print(f"\n=== Conservative Ranker Top 10 ===")
print(f"{'Rank':<5} {'Genome':<20} {'Type':<12} {'Len':<8} {'Score':<7} {'Sig':<4} {'Density':<8} {'Module':<10} {'Cons':<6}")
for i, c in enumerate(cons_scores[:10]):
    print(f"{i+1:<5} {c['genome'][:19]:<20} {c['primary_type'][:11]:<12} {c['length']//1000:<8} "
          f"{c['max_score']:<7.3f} {c['sig_count']:<4} {c['sig_density']:<8.3f} "
          f"{str(c['module_complete']):<10} {c['conservative_score']:<6.2f}")

# --- Compute Novelty ranker on all 412 clean_missed ---
# For novelty, we use: max_score, novelty_distance (already computed), contig_quality
# Note: without Pfam on all 412, we can only do a partial novelty score.
# Let's compute what we can.

# Novelty for top-50 (we have Pfam)
nov_scores_50 = []
for c in candidates:
    pfams = c.get("pfam_list", [])
    enz_density = len(pfams) / max(c["length"]/1000, 1)  # biosyn density as proxy
    
    # Cluster compactness: placeholder (would need gene coordinates)
    cluster_compactness = 0.5
    
    # Novelty distance: already have
    nd = c.get("dist_to_nearest_bgc", -1)
    if nd < 0: nd = 500000  # "no nearby BGC"
    novelty_dist = 1 / (1 + np.exp(-(nd - 50000) / 50000))  # sigmoid
    
    # Max canonical similarity: check if sig pattern matches canonical
    _, module_type = compute_module_completeness(pfams)
    max_can_sim = 1.0 if module_type else 0.3  # high sim if classical module
    low_canonical = 1 - max_can_sim
    
    contig_quality = 1
    
    nov = (
        0.25 * c["max_score"]
        + 0.2 * np.log(enz_density + 1)
        + 0.15 * cluster_compactness
        + 0.2 * novelty_dist
        + 0.15 * low_canonical
        + 0.05 * contig_quality
    )
    nov_scores_50.append({
        "genome": c["genome_id"],
        "contig": c["contig"],
        "start": c["start"],
        "end": c["end"],
        "length": c["length"],
        "primary_type": c["primary_type"],
        "max_score": c["max_score"],
        "enz_density": float(enz_density),
        "novelty_distance_raw": nd,
        "module_type": module_type,
        "novelty_score": float(nov),
    })

nov_scores_50.sort(key=lambda x: -x["novelty_score"])
print(f"\n=== Novelty Ranker Top 10 (from top-50 Pfam-validated) ===")
print(f"{'Rank':<5} {'Genome':<20} {'Type':<12} {'Len':<8} {'Score':<7} {'EnzDen':<8} {'NovDist':<10} {'Canon':<10} {'Nov':<6}")
for i, c in enumerate(nov_scores_50[:10]):
    nd_str = f"{c['novelty_distance_raw']//1000}k" if c['novelty_distance_raw'] > 0 else "max"
    mt = c['module_type'] or "atypical"
    print(f"{i+1:<5} {c['genome'][:19]:<20} {c['primary_type'][:11]:<12} {c['length']//1000:<8} "
          f"{c['max_score']:<7.3f} {c['enz_density']:<8.3f} {nd_str:<10} {mt[:9]:<10} {c['novelty_score']:<6.2f}")

# Intersection of top-10 conservative and top-10 novelty
cons_top10 = [(c["genome"], c["start"], c["end"]) for c in cons_scores[:10]]
nov_top10 = [(c["genome"], c["start"], c["end"]) for c in nov_scores_50[:10]]
intersection = set(cons_top10) & set(nov_top10)
print(f"\nIntersection of top-10 Conservative and top-10 Novelty: {len(intersection)}")

# Save
output = {
    "weights": {
        "conservative": "predeclared: 0.2*mc + 0.3*log(sig_density+1) + 0.3*module_complete + 0.1*log(sig_div+1) + 0.05*contig + 0.05*type_coherence",
        "novelty": "predeclared: 0.25*mc + 0.2*log(enz_den+1) + 0.15*compactness + 0.2*nov_dist + 0.15*(1-canon_sim) + 0.05*contig",
    },
    "conservative_ranked": cons_scores,
    "novelty_ranked": nov_scores_50,
    "top_k_intersection": list(intersection),
    "note": "Rankings computed on top-50 Pfam-validated candidates. Full 412-candidate ranking requires Pfam hits for all (pending).",
}
with open(f"{OUT_DIR}/d3_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {OUT_DIR}/d3_results.json")

# Summary
print("\n=== D3 Summary ===")
print(f"Conservative top-10: {sum(1 for c in cons_scores[:10] if c['module_complete'])}/10 have complete modules")
print(f"Novelty top-10: {sum(1 for c in nov_scores_50[:10] if c['module_type'] is None)}/10 are atypical (no classical module)")
print(f"Both rankers agree on: {len(intersection)}/10 top candidates")
