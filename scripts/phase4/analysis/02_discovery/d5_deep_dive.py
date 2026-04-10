"""D5: Deep dive on top candidates (Track 1 Conservative + Track 2 Novelty-first)."""
import json
import os
from collections import Counter

OUT_DIR = "/data/muskliu/np_master/data/phase4/d5_deep_dive"
os.makedirs(OUT_DIR, exist_ok=True)

# Load all relevant data
with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    disc = json.load(f)
candidates = {i: c for i, c in enumerate(
    [c for c in disc["candidate_results"] if not c.get("skip", False)][:50]
)}

with open("/data/muskliu/np_master/data/phase4/d3_ranking/d3_results.json") as f:
    d3 = json.load(f)

with open("/data/muskliu/np_master/data/phase4/d4_concordance.json") as f:
    d4 = json.load(f)

# Build index by (genome, start, end) 
d4_by_key = {(r["genome"], r["start"], r["end"]): r for r in d4["results"]}

# ======================================
# Track 1: Conservative (top + Tier B/C)
# ======================================
# Take top conservative candidates with any structure
track1 = []
for c in d3["conservative_ranked"]:
    key = (c["genome"], c["start"], c["end"])
    d4r = d4_by_key.get(key)
    if not d4r: continue
    tier = d4r["tier"].split()[0]
    # Track 1: at least Tier B (GECCO agrees) OR Tier C with module complete
    if tier in ("A", "B") or (tier == "C" and c["module_complete"]):
        c2 = dict(c)
        c2["tier"] = tier
        c2["gecco_hit"] = d4r["gecco_hit"]
        track1.append(c2)
    if len(track1) >= 5:
        break

# ======================================
# Track 2: Novelty-first  
# ======================================
track2 = []
for c in d3["novelty_ranked"]:
    key = (c["genome"], c["start"], c["end"])
    d4r = d4_by_key.get(key)
    if not d4r: continue
    tier = d4r["tier"].split()[0]
    # Track 2: atypical structure + at least 5 sig domains (strong structural) + not module-complete
    if c["module_type"] is None:
        # We need the original candidate sig_count
        # Look up in candidates by same coordinates
        match_c = next((k for k, v in candidates.items() 
                       if v["genome_id"]==c["genome"] and v["start"]==c["start"]), None)
        if match_c is None: continue
        sig = candidates[match_c].get("n_signature", 0)
        if sig >= 4:  # strong enough structural
            c2 = dict(c)
            c2["tier"] = tier
            c2["sig_count"] = sig
            c2["module_type_full"] = "atypical"
            track2.append(c2)
    if len(track2) >= 5:
        break

print("="*70)
print("TRACK 1: CONSERVATIVE (wet-lab safety, module-complete + cross-tool)")
print("="*70)
for i, c in enumerate(track1, 1):
    print(f"\n#{i}  [{c['tier']}] {c['genome']}")
    print(f"    Location:  {c['contig']}:{c['start']}-{c['end']} ({c['length']/1000:.0f} kb)")
    print(f"    Type:      {c['primary_type']}")
    print(f"    Module:    {c['module_type']} (complete)")
    print(f"    Evidence:  sig={c['sig_count']} domains, density={c['sig_density']:.3f}/kb")
    print(f"    Score:     Route B={c['max_score']:.3f}, Conservative={c['conservative_score']:.2f}")
    print(f"    GECCO:     {'YES' if c['gecco_hit'] else 'no'}")
    # Full candidate data for more context
    match_c = next((k for k, v in candidates.items()
                   if v["genome_id"]==c["genome"] and v["start"]==c["start"]), None)
    if match_c is not None:
        pfams = set(candidates[match_c].get("pfam_list", []))
        sig_pfams = pfams & {"PF00109","PF02801","PF00698","PF00975","PF14765","PF08659",
                             "PF00106","PF00668","PF00501","PF00550","PF13193","PF00667",
                             "PF01397","PF03936","PF00494","PF13575","PF04055","PF00881"}
        print(f"    Signature domains present: {sorted(sig_pfams)}")

print("\n" + "="*70)
print("TRACK 2: NOVELTY-FIRST (exploratory, atypical architecture)")
print("="*70)
for i, c in enumerate(track2, 1):
    print(f"\n#{i}  [{c['tier']}] {c['genome']}")
    print(f"    Location:  {c['contig']}:{c['start']}-{c['end']} ({c['length']/1000:.0f} kb)")
    print(f"    Type:      {c['primary_type']}")
    print(f"    Structure: ATYPICAL (no canonical NRPS/PKS/Terpene/RiPP module detected)")
    print(f"    Evidence:  sig={c['sig_count']} signature domains, enz_density={c['enz_density']:.2f}/kb")
    print(f"    Score:     Route B={c['max_score']:.3f}, Novelty={c['novelty_score']:.2f}")
    print(f"    Novelty distance: {c['novelty_distance_raw']/1000:.0f}kb from nearest antiSMASH BGC")

# Save full deep dive report
output = {
    "track1_conservative": track1,
    "track2_novelty": track2,
    "summary": {
        "track1_count": len(track1),
        "track2_count": len(track2),
        "track1_description": "Module-complete + cross-tool support (GECCO agrees) or strong sig evidence",
        "track2_description": "Atypical architecture + ≥4 signature domains",
    }
}
with open(f"{OUT_DIR}/top_candidates.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"Track 1 Conservative: {len(track1)} top candidates")
print(f"Track 2 Novelty:      {len(track2)} top candidates")
print(f"Saved to {OUT_DIR}/top_candidates.json")
