"""Parse DeepBGC batch output to unified format."""
import json
import csv
from collections import defaultdict
from pathlib import Path

DEEPBGC_DIR = "/data/muskliu/npmaster/data/phase4/baselines/deepbgc"
OUT = "/data/muskliu/npmaster/data/phase4/baselines/deepbgc_predictions.json"

predictions = defaultdict(list)
n_total = 0
n_bgcs = 0
n_with_any = 0

for gdir in sorted(Path(DEEPBGC_DIR).iterdir()):
    if not gdir.is_dir(): continue
    gid = gdir.name
    # Output filename is {gid}.bgc.tsv
    tsv = gdir / f"{gid}.bgc.tsv"
    if not tsv.exists():
        # fallback check
        tsv_alt = list(gdir.glob("*.bgc.tsv"))
        if tsv_alt:
            tsv = tsv_alt[0]
        else:
            continue
    n_total += 1
    
    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        had_any = False
        for row in reader:
            try:
                bgc = {
                    "genome_id": gid,
                    "contig_id": row.get("sequence_id", ""),
                    "start_bp": int(row.get("nucl_start", 0)) - 1,
                    "end_bp": int(row.get("nucl_end", 0)),
                    "predicted_type": row.get("product_activity", ""),
                    "score": float(row.get("deepbgc_score", 0)),
                    "bgc_candidate_id": row.get("bgc_candidate_id", ""),
                }
                predictions[gid].append(bgc)
                n_bgcs += 1
                had_any = True
            except ValueError:
                continue
        if had_any: n_with_any += 1

print(f"Genomes processed: {n_total}")
print(f"Genomes with >=1 BGC: {n_with_any}")
print(f"Total DeepBGC BGCs: {n_bgcs}")

with open(OUT, "w") as f:
    json.dump({"tool": "deepbgc", "version": "0.1.31",
               "n_genomes": n_total, "n_genomes_with_bgc": n_with_any,
               "n_predictions": n_bgcs, "predictions": dict(predictions)}, f, indent=2)

import numpy as np
lens = [b["end_bp"] - b["start_bp"] for bgcs in predictions.values() for b in bgcs]
if lens:
    print(f"Length dist: median={int(np.median(lens))} mean={int(np.mean(lens))}")
    for lo, hi, name in [(0, 5000, "<5kb"), (5000, 10000, "5-10kb"), (10000, 20000, "10-20kb"),
                         (20000, 50000, "20-50kb"), (50000, 10**10, ">50kb")]:
        n = sum(1 for l in lens if lo <= l < hi)
        print(f"  {name}: {n}")
