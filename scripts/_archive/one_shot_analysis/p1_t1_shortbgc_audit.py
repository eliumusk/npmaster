#!/usr/bin/env python3
"""
P1-T1: Short-BGC Support Audit
===============================
Quantify exactly where short-BGC support is missing:
  1. How many BGCs per length bucket have ZERO windows (contig too short)?
  2. How many have windows but ZERO positive windows (label rule)?
  3. Root cause breakdown: contig-tail dropout vs label threshold.

Usage:
  python p1_t1_shortbgc_audit.py \
    --meta /data/muskliu/np_master/data/windows/windows_metadata_cap08.csv \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --genomes-dir /data/muskliu/np_master/data/genomes \
    --out /data/muskliu/np_master/data/stage4_results/p1_t1_audit
"""
import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path


LENGTH_BUCKETS = [
    ("<5kb", 0, 5000),
    ("5-10kb", 5000, 10000),
    ("10-20kb", 10000, 20000),
    ("20-50kb", 20000, 50000),
    (">50kb", 50000, float("inf")),
]

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
W = 16384
S = 8192


def iter_fasta_lengths(path):
    """Yield (contig_id, length) for each contig in a fasta file."""
    op = gzip.open if str(path).endswith(".gz") else open
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


def bucket_for(bgc_len):
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= bgc_len < hi:
            return name
    return ">50kb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="windows_metadata_cap08.csv")
    ap.add_argument("--bgc-csv", required=True, help="bgc_annotations.csv")
    ap.add_argument("--genomes-dir", required=True, help="dir with {acc}.fna.gz")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ─── 1. Load contig lengths from genome files ───
    print("Loading contig lengths from genomes...")
    contig_lengths = {}  # (genome_id, contig_id) -> length
    genome_files = sorted(fn for fn in os.listdir(args.genomes_dir) if fn.endswith(".fna.gz"))
    for fn in genome_files:
        acc = fn[:-7]
        path = os.path.join(args.genomes_dir, fn)
        for cid, clen in iter_fasta_lengths(path):
            contig_lengths[(acc, cid)] = clen
    print(f"  Loaded {len(contig_lengths)} contigs from {len(genome_files)} genomes")

    # ─── 2. Load BGC annotations ───
    print("Loading BGC annotations...")
    bgcs = []
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            bgc_len = int(row["end_bp"]) - int(row["start_bp"])
            bgcs.append({
                "bgc_id": row["bgc_id"],
                "genome_id": row["genome_id"],
                "contig_id": row["contig_id"],
                "start": int(row["start_bp"]),
                "end": int(row["end_bp"]),
                "length": bgc_len,
                "types": row["bgc_types"],
                "truncated": row.get("is_truncated", "0") == "1",
                "phylum": row.get("phylum", ""),
            })
    print(f"  Loaded {len(bgcs)} BGCs")

    # ─── 3. Load window metadata and index by (genome, contig) ───
    print("Loading window metadata...")
    windows_by_contig = defaultdict(list)
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            key = (row["genome_id"], row["contig_id"])
            windows_by_contig[key].append({
                "start": int(row["window_start"]),
                "end": int(row["window_end"]),
                "label": row["label"],
                "max_overlap": float(row["max_bgc_overlap_ratio"]) if row["max_bgc_overlap_ratio"] else 0,
            })
    print(f"  Loaded windows for {len(windows_by_contig)} contigs")

    # ─── 4. Analyze each BGC ───
    print("Analyzing per-BGC coverage...")

    results = []
    for bgc in bgcs:
        gid = bgc["genome_id"]
        cid = bgc["contig_id"]
        clen = contig_lengths.get((gid, cid), 0)

        # How many sliding windows can this contig generate?
        n_possible_windows = max(0, (clen - W) // S + 1) if clen >= W else 0

        # Find windows that overlap this BGC
        contig_windows = windows_by_contig.get((gid, cid), [])
        overlapping = []
        for w in contig_windows:
            ov = max(0, min(w["end"], bgc["end"]) - max(w["start"], bgc["start"]))
            if ov > 0:
                ov_ratio = ov / W
                bgc_capture = ov / bgc["length"] if bgc["length"] > 0 else 0
                overlapping.append({
                    "label": w["label"],
                    "overlap_ratio": ov_ratio,
                    "bgc_capture": bgc_capture,
                })

        n_overlapping = len(overlapping)
        n_positive = sum(1 for o in overlapping if o["label"] == "positive")
        n_ambiguous = sum(1 for o in overlapping if o["label"] == "ambiguous")
        n_negative = sum(1 for o in overlapping if o["label"] == "negative")

        # Root cause analysis
        if n_possible_windows == 0:
            root_cause = "contig_too_short"  # contig < W, no windows at all
        elif n_overlapping == 0:
            root_cause = "no_overlapping_windows"  # windows exist but none overlap this BGC
        elif n_positive == 0 and n_ambiguous > 0:
            root_cause = "all_ambiguous"  # windows overlap but none meet positive threshold
        elif n_positive == 0:
            root_cause = "other_no_positive"
        else:
            root_cause = "has_positive"

        # Detailed: for windows that DO overlap, what's the max overlap ratio?
        max_ov = max((o["overlap_ratio"] for o in overlapping), default=0)
        max_cap = max((o["bgc_capture"] for o in overlapping), default=0)

        # Could edge padding help? (contig < W or tail remainder < W)
        # Contig tail that doesn't fit: clen - (n_possible_windows - 1) * S - W (if any windows exist)
        tail_remainder = clen % S if n_possible_windows > 0 else clen
        # More precisely: last window ends at (n_possible_windows-1)*S + W if n_possible_windows>0
        last_window_end = ((n_possible_windows - 1) * S + W) if n_possible_windows > 0 else 0
        uncovered_tail = clen - last_window_end

        # Does BGC extend into uncovered tail?
        bgc_in_tail = bgc["end"] > last_window_end if last_window_end > 0 else (clen < W)

        results.append({
            "bgc_id": bgc["bgc_id"],
            "genome_id": gid,
            "contig_id": cid,
            "bgc_length": bgc["length"],
            "bgc_types": bgc["types"],
            "truncated": bgc["truncated"],
            "phylum": bgc["phylum"],
            "contig_length": clen,
            "n_possible_windows": n_possible_windows,
            "n_overlapping_windows": n_overlapping,
            "n_positive_windows": n_positive,
            "n_ambiguous_windows": n_ambiguous,
            "max_overlap_ratio": round(max_ov, 4),
            "max_bgc_capture": round(max_cap, 4),
            "root_cause": root_cause,
            "bgc_in_uncovered_tail": bgc_in_tail,
            "uncovered_tail_bp": uncovered_tail,
            "bucket": bucket_for(bgc["length"]),
        })

    # ─── 5. Aggregate by bucket ───
    print("\n" + "=" * 80)
    print("SHORT-BGC SUPPORT AUDIT")
    print("=" * 80)

    bucket_stats = {}
    for bname, blo, bhi in LENGTH_BUCKETS:
        in_bucket = [r for r in results if blo <= r["bgc_length"] < bhi]
        n_total = len(in_bucket)
        n_zero_windows = sum(1 for r in in_bucket if r["n_possible_windows"] == 0)
        n_zero_overlap = sum(1 for r in in_bucket if r["n_overlapping_windows"] == 0 and r["n_possible_windows"] > 0)
        n_zero_positive = sum(1 for r in in_bucket if r["n_positive_windows"] == 0)
        n_has_positive = sum(1 for r in in_bucket if r["n_positive_windows"] > 0)
        n_in_tail = sum(1 for r in in_bucket if r["bgc_in_uncovered_tail"])

        # Root cause breakdown
        rc_counts = defaultdict(int)
        for r in in_bucket:
            rc_counts[r["root_cause"]] += 1

        # Type breakdown for zero-positive
        type_zero_pos = defaultdict(int)
        for r in in_bucket:
            if r["n_positive_windows"] == 0:
                for t in r["bgc_types"].split(";"):
                    if t:
                        type_zero_pos[t] += 1

        bucket_stats[bname] = {
            "n_total": n_total,
            "n_zero_windows": n_zero_windows,
            "n_zero_overlap": n_zero_overlap,
            "n_zero_positive": n_zero_positive,
            "n_has_positive": n_has_positive,
            "zero_positive_rate": round(n_zero_positive / max(1, n_total), 4),
            "n_in_uncovered_tail": n_in_tail,
            "root_cause": dict(rc_counts),
            "zero_positive_by_type": dict(type_zero_pos),
        }

        print(f"\n  {bname} (n={n_total})")
        print(f"    Zero windows (contig too short):   {n_zero_windows}")
        print(f"    Zero overlapping windows:          {n_zero_overlap}")
        print(f"    Zero positive windows:             {n_zero_positive} ({bucket_stats[bname]['zero_positive_rate']:.1%})")
        print(f"    Has positive:                      {n_has_positive}")
        print(f"    BGC in uncovered tail:             {n_in_tail}")
        print(f"    Root cause: {dict(rc_counts)}")

    # ─── 6. Edge-padding potential ───
    print("\n\n=== EDGE-PADDING POTENTIAL ===")
    # How many BGCs could be rescued by adding padded windows?
    rescuable_by_padding = [r for r in results
                            if r["n_positive_windows"] == 0
                            and (r["root_cause"] == "contig_too_short" or r["bgc_in_uncovered_tail"])]
    print(f"  BGCs with zero positive that could gain a window from edge padding: {len(rescuable_by_padding)}")

    # Break down by bucket
    for bname, blo, bhi in LENGTH_BUCKETS:
        in_bucket = [r for r in rescuable_by_padding if blo <= r["bgc_length"] < bhi]
        if in_bucket:
            print(f"    {bname}: {len(in_bucket)}")

    # ─── 7. Truncation analysis ───
    print("\n\n=== TRUNCATION ANALYSIS ===")
    trunc = [r for r in results if r["truncated"]]
    non_trunc = [r for r in results if not r["truncated"]]
    print(f"  Truncated BGCs: {len(trunc)} ({sum(1 for r in trunc if r['n_positive_windows']==0)} with zero positive)")
    print(f"  Non-truncated:  {len(non_trunc)} ({sum(1 for r in non_trunc if r['n_positive_windows']==0)} with zero positive)")

    # Truncated BGCs by length bucket
    for bname, blo, bhi in LENGTH_BUCKETS:
        trunc_bucket = [r for r in trunc if blo <= r["bgc_length"] < bhi]
        if trunc_bucket:
            n_zp = sum(1 for r in trunc_bucket if r["n_positive_windows"] == 0)
            print(f"    Truncated {bname}: {len(trunc_bucket)} total, {n_zp} zero positive")

    # ─── 8. Save ───
    output = {
        "task_id": "P1-T1",
        "description": "Short-BGC support audit",
        "window_size": W,
        "stride": S,
        "label_config": "cap08 (short_bgc_capture=0.8)",
        "total_bgcs": len(bgcs),
        "bucket_stats": bucket_stats,
        "edge_padding_potential": {
            "n_rescuable": len(rescuable_by_padding),
            "by_bucket": {bname: len([r for r in rescuable_by_padding if blo <= r["bgc_length"] < bhi])
                         for bname, blo, bhi in LENGTH_BUCKETS},
        },
        "truncation": {
            "n_truncated": len(trunc),
            "n_truncated_zero_positive": sum(1 for r in trunc if r["n_positive_windows"] == 0),
            "n_nontrunced": len(non_trunc),
            "n_nontrunced_zero_positive": sum(1 for r in non_trunc if r["n_positive_windows"] == 0),
        },
        "per_bgc_details_path": os.path.join(args.out, "per_bgc_details.csv"),
    }

    # Save summary JSON
    with open(os.path.join(args.out, "shortbgc_audit.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Save per-BGC details CSV
    with open(os.path.join(args.out, "per_bgc_details.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bgc_id", "genome_id", "contig_id", "bgc_length", "bgc_types", "truncated",
            "phylum", "contig_length", "n_possible_windows", "n_overlapping_windows",
            "n_positive_windows", "n_ambiguous_windows", "max_overlap_ratio", "max_bgc_capture",
            "root_cause", "bgc_in_uncovered_tail", "bucket",
        ])
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["bgc_length"]):
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nResults saved to {args.out}/")
    print(f"  shortbgc_audit.json  — summary")
    print(f"  per_bgc_details.csv  — per-BGC breakdown")


if __name__ == "__main__":
    main()
