"""Parse GECCO batch output to unified BGC format."""
import json
from collections import defaultdict
from pathlib import Path

GECCO_DIR = "/data/muskliu/npmaster/data/phase4/baselines/gecco"
OUT = "/data/muskliu/npmaster/data/phase4/baselines/gecco_predictions.json"

predictions = defaultdict(list)
n_total = 0
n_bgcs = 0
n_with_any = 0

for gdir in sorted(Path(GECCO_DIR).iterdir()):
    if not gdir.is_dir(): continue
    gid = gdir.name
    # Check if the run actually completed (features file exists)
    if not (gdir / f"{gid}.features.tsv").exists(): continue
    n_total += 1
    
    cluster_tsv = gdir / f"{gid}.clusters.tsv"
    if not cluster_tsv.exists():
        predictions[gid] = []  # ran but found 0 BGCs
        continue
    
    with open(cluster_tsv) as f:
        lines = f.readlines()
    if len(lines) < 2:
        predictions[gid] = []
        continue
    n_with_any += 1
    headers = lines[0].strip().split("\t")
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        row = dict(zip(headers, parts))
        try:
            bgc = {
                "genome_id": gid,
                "contig_id": row.get("sequence_id", ""),
                "start_bp": int(row.get("start", 0)) - 1,
                "end_bp": int(row.get("end", 0)),
                "predicted_type": row.get("type", "Unknown"),
                "score": float(row.get("max_p", 0)),
                "cluster_id": row.get("cluster_id", ""),
            }
            predictions[gid].append(bgc)
            n_bgcs += 1
        except ValueError:
            continue

print(f"Genomes processed: {n_total}")
print(f"Genomes with >=1 GECCO BGC: {n_with_any}")
print(f"Genomes with 0 BGCs: {n_total - n_with_any}")
print(f"Total GECCO BGCs: {n_bgcs}")

with open(OUT, "w") as f:
    json.dump({"tool": "gecco", "version": "0.10.3", "n_genomes": n_total,
               "n_genomes_with_bgc": n_with_any, "n_predictions": n_bgcs,
               "predictions": dict(predictions)}, f, indent=2)

import numpy as np
lens = [b["end_bp"] - b["start_bp"] for bgcs in predictions.values() for b in bgcs]
print(f"\nLength dist: median={int(np.median(lens))} mean={int(np.mean(lens))}")
