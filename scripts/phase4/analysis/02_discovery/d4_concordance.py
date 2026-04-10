"""D4: Multi-tool concordance (cross-tool support).

For each Route B top candidate, check if GECCO (and DeepBGC if available) flags
the same region.
"""
import json
from collections import defaultdict

# Load Route B candidates + rankings
with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    disc = json.load(f)
candidates = [c for c in disc["candidate_results"] if not c.get("skip", False)][:50]

# Load GECCO predictions
with open("/data/muskliu/np_master/data/phase4/baselines/gecco_predictions.json") as f:
    gecco = json.load(f)

# Load DeepBGC predictions if available
deepbgc_preds = {}
import os
if os.path.exists("/data/muskliu/np_master/data/phase4/baselines/deepbgc_predictions.json"):
    with open("/data/muskliu/np_master/data/phase4/baselines/deepbgc_predictions.json") as f:
        deepbgc_preds = json.load(f).get("predictions", {})

# Build GECCO lookup by (genome, contig)
gecco_by_loc = defaultdict(list)
for gid, bgcs in gecco["predictions"].items():
    for b in bgcs:
        gecco_by_loc[(gid, b["contig_id"])].append(b)

deepbgc_by_loc = defaultdict(list)
for gid, bgcs in deepbgc_preds.items():
    for b in bgcs:
        deepbgc_by_loc[(gid, b["contig_id"])].append(b)

def overlap_frac(s1, e1, s2, e2):
    ov = max(0, min(e1, e2) - max(s1, s2))
    return ov / max(1, min(e1 - s1, e2 - s2))

# For each Route B candidate, check concordance
concordance_results = []
for c in candidates:
    gid = c["genome_id"]
    contig = c["contig"]
    start = c["start"]
    end = c["end"]
    
    # Check GECCO
    gecco_hit = False
    gecco_ov = 0
    for gb in gecco_by_loc.get((gid, contig), []):
        ov = overlap_frac(start, end, gb["start_bp"], gb["end_bp"])
        if ov >= 0.5:
            gecco_hit = True
            gecco_ov = max(gecco_ov, ov)
    
    # Check DeepBGC
    deepbgc_hit = False
    deepbgc_ov = 0
    for db in deepbgc_by_loc.get((gid, contig), []):
        ov = overlap_frac(start, end, db["start_bp"], db["end_bp"])
        if ov >= 0.5:
            deepbgc_hit = True
            deepbgc_ov = max(deepbgc_ov, ov)
    
    # Tier assignment
    # Note: DeepBGC is pending, so for now treat unavailable DeepBGC as unknown
    # Tier A: both GECCO and DeepBGC agree
    # Tier B: GECCO or DeepBGC agree
    # Tier C: only Route B, but strong structural evidence
    # Tier D: only Route B, weak evidence
    
    sig_count = c.get("n_signature", 0)
    
    if deepbgc_preds:
        # Full tier assignment
        if gecco_hit and deepbgc_hit:
            tier = "A"
        elif gecco_hit or deepbgc_hit:
            tier = "B"
        elif sig_count >= 5:
            tier = "C"
        else:
            tier = "D"
    else:
        # Provisional: use GECCO only; DeepBGC pending
        if gecco_hit:
            tier = "B (GECCO agrees; DeepBGC pending)"
        elif sig_count >= 5:
            tier = "C (structural support only)"
        else:
            tier = "D (weak)"
    
    concordance_results.append({
        "genome": gid, "contig": contig, "start": start, "end": end,
        "length": c["length"], "primary_type": c["primary_type"],
        "max_score": c["max_score"], "sig_count": sig_count,
        "gecco_hit": gecco_hit, "gecco_overlap": gecco_ov,
        "deepbgc_hit": deepbgc_hit, "deepbgc_overlap": deepbgc_ov,
        "tier": tier,
    })

# Summary
tier_counts = defaultdict(int)
for r in concordance_results:
    t = r["tier"].split()[0]
    tier_counts[t] += 1

print("=== D4: Multi-tool Concordance (Route B top-50) ===")
print(f"DeepBGC data available: {'yes' if deepbgc_preds else 'NO (still running)'}")
print(f"\nTier distribution:")
for t in sorted(tier_counts.keys()):
    print(f"  Tier {t}: {tier_counts[t]}")

print("\nGECCO concordance alone:")
gecco_concordant = sum(1 for r in concordance_results if r["gecco_hit"])
print(f"  Route B candidates also flagged by GECCO: {gecco_concordant}/50 ({gecco_concordant/50*100:.0f}%)")
print(f"  Candidates only Route B flags: {50 - gecco_concordant}/50")
print(f"  Of 'only Route B' candidates with sig_count ≥ 5: {sum(1 for r in concordance_results if not r['gecco_hit'] and r['sig_count']>=5)}")

print("\nTop candidates by tier:")
for tier_name in ["A", "B", "C"]:
    tier_cands = [r for r in concordance_results if r["tier"].startswith(tier_name)]
    if tier_cands:
        print(f"\n  Tier {tier_name} ({len(tier_cands)} candidates):")
        for r in tier_cands[:5]:
            print(f"    {r['genome']} | {r['primary_type']} | len={r['length']//1000}k | sig={r['sig_count']} | score={r['max_score']:.3f}")

# Save
with open("/data/muskliu/np_master/data/phase4/d4_concordance.json", "w") as f:
    json.dump({
        "deepbgc_available": bool(deepbgc_preds),
        "tier_counts": dict(tier_counts),
        "gecco_concordance_rate": gecco_concordant/50,
        "results": concordance_results,
    }, f, indent=2)
print(f"\nSaved to /data/muskliu/np_master/data/phase4/d4_concordance.json")
