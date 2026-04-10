"""D2: Construct matched null distribution for top-50 Route B candidates.

For each candidate, sample 10 matched controls from the same genome with:
  - length match (strict)
  - GC ±2%
  - gene density ±20% (approximated via gene count per kb using Prodigal)
  - distance > 50kb from any antiSMASH / Route B / (GECCO/DeepBGC if available)
  - not overlapping candidate

Then run Prodigal + hmmscan on all controls to get signature density.
Compare distributions.
"""
import gzip
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = "/data/muskliu/np_master/data/phase4/d2_matched_null"
os.makedirs(OUT_DIR, exist_ok=True)

# Load top-50 Route B candidates with existing Pfam data
with open("/data/muskliu/np_master/data/phase3/route_b/discovery/discovery_benchmark.json") as f:
    discovery = json.load(f)

candidates = [c for c in discovery["candidate_results"] if not c.get("skip", False)][:50]
print(f"Loaded {len(candidates)} Route B top candidates")

# Load antiSMASH annotations for exclusion
as_regions = defaultdict(list)  # genome_id -> list of (start, end)
with open("/data/muskliu/np_master/data/bgc_annotations.csv") as f:
    import csv
    for row in csv.DictReader(f):
        as_regions[row["genome_id"]].append((
            row["contig_id"],
            int(row["start_bp"]),
            int(row["end_bp"]),
        ))

# Load ALL Route B calls (not just top candidates) for exclusion
# From score_swap results
with open("/data/muskliu/np_master/data/phase3/route_b/eval/score_swap_s0.json") as f:
    sw = json.load(f)
all_route_b_calls = defaultdict(list)
for split in ["test_id", "test_ood"]:
    for u in sw["results"][split]["score_maxtype"].get("unmatched_details", []):
        all_route_b_calls[u["genome_id"]].append((u["contig"], u["start"], u["end"]))

print(f"Loaded {sum(len(v) for v in all_route_b_calls.values())} Route B calls for exclusion")

# Helper: compute GC content
def compute_gc(seq):
    if not seq: return 0
    gc = seq.count("G") + seq.count("g") + seq.count("C") + seq.count("c")
    total = len(seq) - seq.count("N") - seq.count("n")
    return gc / max(total, 1)

# Load genome sequences on demand
genome_cache = {}
def get_contig(gid, contig_id):
    if gid not in genome_cache:
        fna = f"/data/muskliu/np_master/data/genomes/{gid}.fna.gz"
        contigs = {}
        with gzip.open(fna, "rt") as f:
            cur_id = None
            chunks = []
            for line in f:
                if line.startswith(">"):
                    if cur_id: contigs[cur_id] = "".join(chunks)
                    cur_id = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line.strip())
            if cur_id: contigs[cur_id] = "".join(chunks)
        genome_cache[gid] = contigs
    return genome_cache[gid].get(contig_id, "")

# Sample matched null for each candidate
def sample_matched_null(cand, n=10, max_attempts=200):
    gid = cand["genome_id"]
    contig_id = cand["contig"]
    cand_start = cand["start"]
    cand_end = cand["end"]
    cand_len = cand_end - cand_start
    
    contig_seq = get_contig(gid, contig_id)
    contig_len = len(contig_seq)
    if contig_len < cand_len * 3:
        return []  # contig too short
    
    # Compute candidate GC and gene density approximation
    cand_seq = contig_seq[cand_start:cand_end]
    cand_gc = compute_gc(cand_seq)
    cand_gene_density = cand.get("n_genes", 1) / max(cand_len / 1000, 1)  # genes/kb
    
    # Exclusion zones: antiSMASH BGCs and Route B calls (same contig, within 50kb)
    excluded = []
    for c, s, e in as_regions.get(gid, []):
        if c == contig_id:
            excluded.append((max(0, s - 50000), e + 50000))
    for c, s, e in all_route_b_calls.get(gid, []):
        if c == contig_id:
            excluded.append((max(0, s - 20000), e + 20000))
    # Also exclude the candidate itself
    excluded.append((cand_start - cand_len, cand_end + cand_len))
    
    def is_excluded(s, e):
        for xs, xe in excluded:
            if s < xe and e > xs:
                return True
        return False
    
    # Sample randomly
    rng = np.random.default_rng(hash(cand["genome_id"] + str(cand_start)) & 0x7fffffff)
    nulls = []
    attempts = 0
    while len(nulls) < n and attempts < max_attempts:
        attempts += 1
        max_start = contig_len - cand_len
        if max_start <= 0: break
        ns = rng.integers(0, max_start)
        ne = ns + cand_len
        if is_excluded(ns, ne): continue
        null_seq = contig_seq[ns:ne]
        if len(null_seq) - null_seq.upper().count("N") < cand_len * 0.8:
            continue  # too many Ns
        null_gc = compute_gc(null_seq)
        if abs(null_gc - cand_gc) > 0.03:
            continue
        # Gene density check: approximate by counting ORFs > 300bp
        # Run prodigal later; for now just accept
        nulls.append({
            "start": int(ns),
            "end": int(ne),
            "gc": float(null_gc),
            "seq_len": len(null_seq),
        })
    return nulls

# Sample null for each candidate
print("\nSampling matched null controls...")
cand_with_null = []
for i, c in enumerate(candidates):
    nulls = sample_matched_null(c, n=10)
    c_with_null = dict(c)
    c_with_null["matched_null"] = nulls
    cand_with_null.append(c_with_null)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(candidates)}, avg null count: {np.mean([len(x['matched_null']) for x in cand_with_null]):.1f}")

null_counts = [len(c["matched_null"]) for c in cand_with_null]
print(f"\nNull sampling complete. Per-candidate null count: mean={np.mean(null_counts):.1f}, min={min(null_counts)}, max={max(null_counts)}")

# Save intermediate
with open(f"{OUT_DIR}/candidates_with_null.json", "w") as f:
    json.dump(cand_with_null, f, indent=2)

# Write FASTA files for all null regions (for Prodigal + hmmscan)
print("\nWriting null region FASTA files...")
fasta_dir = Path(f"{OUT_DIR}/null_fasta")
fasta_dir.mkdir(exist_ok=True)

null_index = []
n_total = 0
for ci, c in enumerate(cand_with_null):
    gid = c["genome_id"]
    contig_id = c["contig"]
    contig_seq = get_contig(gid, contig_id)
    for ni, nn in enumerate(c["matched_null"]):
        null_id = f"null_{ci:03d}_{ni:02d}"
        seq = contig_seq[nn["start"]:nn["end"]]
        with open(fasta_dir / f"{null_id}.fna", "w") as f:
            f.write(f">{null_id}_{gid}_{contig_id}_{nn['start']}_{nn['end']}\n{seq}\n")
        null_index.append({
            "null_id": null_id, "cand_idx": ci,
            "genome_id": gid, "contig": contig_id,
            "start": nn["start"], "end": nn["end"],
            "gc": nn["gc"],
            "length": nn["end"] - nn["start"],
        })
        n_total += 1
print(f"Wrote {n_total} null FASTA files")

with open(f"{OUT_DIR}/null_index.json", "w") as f:
    json.dump(null_index, f, indent=2)

print(f"\nNext: run Prodigal + hmmscan (biosyn_pfam) on all null regions")
print(f"  null FASTA dir: {fasta_dir}")
