"""P1: MIBiG external validation with species-level matching.

Tier 1 (assembly-resolved) requires MIBiG metadata JSON not available locally.
Fall back to Tier 2 (species-level) matching as approximate support.
"""
import os
import re
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

MIBIG_DIR = "/data/zhanglin/EvoBGC/data/mibig"
META_CSV = "/data/muskliu/np_master/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv"
OUT_DIR = "/data/muskliu/np_master/data/phase4/mibig_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# Parse MIBiG gbk files: extract BGC id, organism, length, type
print("Parsing MIBiG gbk files...")
mibig_entries = []
gbks = sorted(Path(MIBIG_DIR).glob("*.gbk"))
print(f"Total MIBiG gbk files: {len(gbks)}")

for gbk in gbks:
    with open(gbk) as f:
        content = f.read()
    bgc_id = gbk.stem
    # Parse organism
    m_org = re.search(r"^\s*ORGANISM\s+([^\n]+)", content, re.MULTILINE)
    organism = m_org.group(1).strip() if m_org else ""
    # Parse length
    m_len = re.search(r"LOCUS\s+\S+\s+(\d+)\s+bp", content)
    length = int(m_len.group(1)) if m_len else 0
    # Parse BGC types - from COMMENT or keywords
    types = []
    # Try to parse from antiSMASH comment
    m_types = re.findall(r"/product=\"([^\"]+)\"", content)
    types = list(set(m_types))
    mibig_entries.append({
        "bgc_id": bgc_id,
        "organism": organism,
        "length": length,
        "types": types,
    })

print(f"Parsed {len(mibig_entries)} MIBiG entries")

# Get test genome organism info from GTDB or our metadata
# Unfortunately our metadata only has genome_id and phylum, not species.
# Use GTDB taxonomy from somewhere, or parse genome FNA headers
print("\nLoading test genome list...")
df = pd.read_csv(META_CSV)
test_df = df[df["split"].isin(["test_id", "test_ood"])].copy()
test_genomes_info = test_df[["genome_id", "phylum"]].drop_duplicates()
print(f"Test genomes: {len(test_genomes_info)}")

# Try to get species info from genome FNA headers (they usually have organism name in defline)
# This is cheap: zcat first line
import gzip

print("\nExtracting species from FNA headers...")
genome_species = {}
for gid in sorted(test_genomes_info["genome_id"]):
    fna_path = f"/data/muskliu/np_master/data/genomes/{gid}.fna.gz"
    if not os.path.exists(fna_path):
        continue
    try:
        with gzip.open(fna_path, "rt") as f:
            first_line = f.readline()
        # e.g. ">CP001673.1 Streptomyces violaceusniger Tu 4113 chromosome, complete genome"
        m = re.match(r">\S+\s+(.+?)(?:,|$)", first_line)
        if m:
            header_text = m.group(1).strip()
            # Get first 2-3 words as species
            words = header_text.split()
            if len(words) >= 2:
                species = " ".join(words[:2])
                genome_species[gid] = {"species": species, "full": header_text}
    except Exception as e:
        pass

print(f"Got species info for {len(genome_species)} test genomes")

# Match MIBiG organism to test genomes by species
print("\nMatching MIBiG to test genomes (species-level)...")
# Build lookup: species name -> list of test genomes
species_to_genomes = defaultdict(list)
for gid, info in genome_species.items():
    sp = info["species"]
    species_to_genomes[sp].append(gid)

# Match each MIBiG entry
mibig_matches = []
for entry in mibig_entries:
    org = entry["organism"]
    if not org:
        continue
    # Try exact species match
    words = org.split()
    if len(words) < 2:
        continue
    sp = " ".join(words[:2])
    matched = species_to_genomes.get(sp, [])
    if matched:
        mibig_matches.append({
            "bgc_id": entry["bgc_id"],
            "organism": org,
            "species": sp,
            "length": entry["length"],
            "types": entry["types"],
            "matched_genomes": matched,
            "match_type": "species_level",
        })

print(f"MIBiG entries with species-level match to test genomes: {len(mibig_matches)}")

# Breakdown
sp_matches = defaultdict(list)
for m in mibig_matches:
    sp_matches[m["species"]].append(m["bgc_id"])
print(f"\nTop matched species:")
for sp, bgcs in sorted(sp_matches.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  {sp}: {len(bgcs)} MIBiG BGCs")

# Length distribution
if mibig_matches:
    lengths = [m["length"] for m in mibig_matches]
    import numpy as np
    print(f"\nMatched MIBiG length: min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}")

# Save
with open(f"{OUT_DIR}/mibig_matches.json", "w") as f:
    json.dump({
        "mibig_total": len(mibig_entries),
        "test_genomes_with_species": len(genome_species),
        "species_level_matches": len(mibig_matches),
        "matches": mibig_matches,
        "note": "Species-level (Tier 2) matching only. Tier 1 (assembly-resolved) requires MIBiG metadata JSON not available locally.",
    }, f, indent=2)

print(f"\nSaved to {OUT_DIR}/mibig_matches.json")
print("\n*** LIMITATION: This is species-level (Tier 2) matching only. ***")
print("*** Tier 1 (assembly-resolved coordinate-exact) matching requires MIBiG metadata JSON ***")
print("*** Interpret results as 'support evidence' not 'primary benchmark'. ***")
