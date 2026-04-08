#!/usr/bin/env python3
"""
Stage 1.3a+b: slice windows + label (multi-label, 8-dim)

Window: W=16384, S=8192, 50% overlap. No padding at contig end (drop short tail).
Labels:
  - positive: window has max_overlap_ratio (with any BGC) >= 0.5
      → bgc_types = union of classes from BGCs with overlap_ratio >= 0.3
      → is_negative = 0
  - negative: no overlap AND min distance to any BGC >= 10000 bp
  - ambiguous: otherwise

8-dim: [Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other, is_negative]

Also records `overlaps_truncated_bgc` (bool) to support eval-time exclusion.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

CLASSES = ["Polyketide","NRP","Terpene","RiPP","Saccharide","Alkaloid","Other"]

def iter_fasta(path):
    op = gzip.open if path.endswith(".gz") else open
    seq_id = None
    length = 0
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, length
                seq_id = line[1:].split()[0]
                length = 0
            else:
                length += len(line.strip())
        if seq_id is not None:
            yield seq_id, length

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bgc-csv", required=True)
    ap.add_argument("--genomes-dir", required=True, help="dir with {acc}.fna.gz")
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--step", type=int, default=8192)
    ap.add_argument("--buffer", type=int, default=10000)
    ap.add_argument("--overlap-pos", type=float, default=0.5)
    ap.add_argument("--overlap-type", type=float, default=0.3)
    ap.add_argument("--out", required=True, help="windows_metadata.csv")
    args = ap.parse_args()

    # load BGCs grouped by (genome, contig)
    bgcs_by_contig = defaultdict(list)
    genome_phylum = {}
    with open(args.bgc_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["genome_id"], row["contig_id"])
            bgcs_by_contig[key].append({
                "bgc_id": row["bgc_id"],
                "start": int(row["start_bp"]),
                "end": int(row["end_bp"]),
                "classes": row["bgc_types"].split(";") if row["bgc_types"] else [],
                "truncated": row["is_truncated"]=="1",
            })
            genome_phylum[row["genome_id"]] = row["phylum"]

    # genomes to process: union of genomes in BGC file (these are the ones with antiSMASH done)
    genomes = sorted(set(k[0] for k in bgcs_by_contig))
    # also include genomes with 0 BGCs — they still need window sampling
    # find by listing antismash_dir ... but we only get it from BGC file. For safety, also list genomes_dir
    import os
    for fn in os.listdir(args.genomes_dir):
        if fn.endswith(".fna.gz"):
            acc = fn[:-7]
            if acc not in genome_phylum:
                # genome with no BGCs detected; phylum unknown from BGC csv
                pass

    # we'll process all genomes present in genomes_dir (but phylum for 0-bgc genomes
    # needs selected_genomes.tsv; skip for now — caller provides bgc-csv which has all genomes with BGCs,
    # other ones we skip labelling but still need windows: see below.

    # Actually: we need selected_genomes.tsv for phylum lookup too. We have it via bgc_csv for genomes with BGCs.
    # For 0-bgc genomes: we MUST still produce windows (they're all negative). Load phylum from selected_genomes.tsv.
    sel = os.path.join(os.path.dirname(args.bgc_csv), "selected_genomes.tsv")
    if os.path.exists(sel):
        with open(sel) as f:
            rd = csv.DictReader(f, delimiter="\t")
            for row in rd:
                genome_phylum.setdefault(row["ncbi_accession"], row["phylum"])

    W = args.window; S = args.step; BUF = args.buffer
    OP = args.overlap_pos; OT = args.overlap_type

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_pos=n_neg=n_amb=0
    class_counts = {c:0 for c in CLASSES}
    with open(args.out, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["window_id","genome_id","contig_id","window_start","window_end",
                        "label"]+CLASSES+["is_negative","max_bgc_overlap_ratio",
                        "nearest_bgc_distance","overlaps_truncated_bgc","phylum"])
        wid = 0
        all_fastas = sorted(fn for fn in os.listdir(args.genomes_dir) if fn.endswith(".fna.gz"))
        for fn in all_fastas:
            acc = fn[:-7]
            phylum = genome_phylum.get(acc, "")
            path = os.path.join(args.genomes_dir, fn)
            for contig_id, clen in iter_fasta(path):
                contig_bgcs = bgcs_by_contig.get((acc, contig_id), [])
                # sliding window
                pos = 0
                while pos + W <= clen:
                    w_start = pos; w_end = pos+W
                    # find overlapping BGCs
                    overlaps = []
                    min_dist = None
                    overlaps_trunc = False
                    for b in contig_bgcs:
                        ov = max(0, min(w_end, b["end"]) - max(w_start, b["start"]))
                        if ov > 0:
                            overlaps.append((ov/W, b))
                            if b["truncated"]:
                                overlaps_trunc = True
                        else:
                            d = min(abs(w_end-b["start"]), abs(w_start-b["end"]))
                            if min_dist is None or d < min_dist: min_dist = d
                    if overlaps:
                        max_or = max(o for o,_ in overlaps)
                        if max_or >= OP:
                            label = "positive"
                            classes = set()
                            for o,b in overlaps:
                                if o >= OT:
                                    classes.update(b["classes"])
                            vec = [1 if c in classes else 0 for c in CLASSES]
                            writer.writerow([f"w{wid:09d}",acc,contig_id,w_start,w_end,"positive"]+vec+[0,
                                f"{max_or:.3f}","", "1" if overlaps_trunc else "0", phylum])
                            n_pos += 1
                            for c in classes: class_counts[c] += 1
                        else:
                            writer.writerow([f"w{wid:09d}",acc,contig_id,w_start,w_end,"ambiguous"]+[0]*7+[0,
                                f"{max_or:.3f}","", "1" if overlaps_trunc else "0", phylum])
                            n_amb += 1
                    else:
                        if min_dist is None or min_dist >= BUF:
                            writer.writerow([f"w{wid:09d}",acc,contig_id,w_start,w_end,"negative"]+[0]*7+[1,
                                "0", str(min_dist or "inf"), "0", phylum])
                            n_neg += 1
                        else:
                            writer.writerow([f"w{wid:09d}",acc,contig_id,w_start,w_end,"ambiguous"]+[0]*7+[0,
                                "0", str(min_dist), "0", phylum])
                            n_amb += 1
                    wid += 1
                    pos += S
    print(json.dumps({
        "total_windows": wid, "positive": n_pos, "negative": n_neg, "ambiguous": n_amb,
        "class_counts": class_counts,
    }, indent=2))

if __name__ == "__main__":
    main()
