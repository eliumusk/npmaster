#!/usr/bin/env python3
"""
WS3-MVP: Re-label windows with length-adaptive overlap threshold.

Instead of re-running the full Stage 1 pipeline, this script:
1. Reads the existing windows_metadata.csv
2. Loads BGC annotations
3. For each ambiguous window, checks if it captures >= 80% of any short BGC
4. If yes, flips the label to "positive" and assigns types
5. Writes a new windows_metadata.csv

This preserves:
- All window positions (same grid)
- All splits (same genome assignments)
- All negative labels (unchanged)
- All existing positive labels (unchanged)
- Embeddings (no re-extraction needed)

Only change: some ambiguous → positive for short-BGC windows.

Usage:
  python ws3_mvp_relabel.py \
    --meta /data/muskliu/np_master/data/windows/windows_metadata.csv \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --short-bgc-capture 0.8 \
    --out /data/muskliu/np_master/data/windows/windows_metadata_ws3mvp.csv
"""
import argparse
import csv
import json
from collections import defaultdict

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="existing windows_metadata.csv")
    ap.add_argument("--bgc-csv", required=True, help="bgc_annotations.csv")
    ap.add_argument("--short-bgc-capture", type=float, default=0.8,
                    help="Fraction of BGC that must be captured to flip to positive")
    ap.add_argument("--out", required=True, help="output windows_metadata.csv")
    args = ap.parse_args()

    W = 16384  # window size (from the existing metadata)
    CAP = args.short_bgc_capture

    # Load BGC annotations grouped by (genome, contig)
    print("Loading BGC annotations...")
    bgcs_by_loc = defaultdict(list)
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            bgcs_by_loc[(row["genome_id"], row["contig_id"])].append({
                "start": int(row["start_bp"]),
                "end": int(row["end_bp"]),
                "classes": row["bgc_types"].split(";") if row["bgc_types"] else [],
                "length": int(row["end_bp"]) - int(row["start_bp"]),
            })
    print(f"  {sum(len(v) for v in bgcs_by_loc.values())} BGCs across {len(bgcs_by_loc)} contigs")

    # Process windows
    print("Processing windows...")
    n_total = 0
    n_flipped = 0
    n_already_pos = 0
    n_amb_kept = 0
    flipped_by_type = defaultdict(int)
    flipped_bgc_lengths = []

    with open(args.meta) as fin, open(args.out, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            n_total += 1

            if row["label"] == "ambiguous":
                # Check if this ambiguous window captures enough of a short BGC
                gid = row["genome_id"]
                cid = row["contig_id"]
                ws = int(row["window_start"])
                we = int(row["window_end"])

                bgcs = bgcs_by_loc.get((gid, cid), [])
                should_flip = False
                captured_classes = set()

                for b in bgcs:
                    ov = max(0, min(we, b["end"]) - max(ws, b["start"]))
                    if ov <= 0:
                        continue
                    bgc_capture = ov / b["length"] if b["length"] > 0 else 0
                    overlap_ratio = ov / W

                    # WS3-MVP: if window captures >= 80% of this BGC, flip to positive
                    if bgc_capture >= CAP:
                        should_flip = True
                        captured_classes.update(b["classes"])
                        flipped_bgc_lengths.append(b["length"])

                if should_flip:
                    # Flip to positive
                    row["label"] = "positive"
                    row["is_negative"] = "0"
                    for c in CLASSES:
                        row[c] = "1" if c in captured_classes else row[c]
                    n_flipped += 1
                    for c in captured_classes:
                        flipped_by_type[c] += 1
                else:
                    n_amb_kept += 1
            elif row["label"] == "positive":
                n_already_pos += 1

            writer.writerow(row)

    # Summary
    import numpy as np
    lens = np.array(flipped_bgc_lengths) if flipped_bgc_lengths else np.array([0])
    summary = {
        "total_windows": n_total,
        "already_positive": n_already_pos,
        "flipped_ambiguous_to_positive": n_flipped,
        "ambiguous_kept": n_amb_kept,
        "new_total_positive": n_already_pos + n_flipped,
        "flipped_by_type": dict(flipped_by_type),
        "flipped_bgc_length_stats": {
            "min": int(lens.min()),
            "max": int(lens.max()),
            "mean": int(lens.mean()),
            "median": int(np.median(lens)),
        },
        "short_bgc_capture_threshold": CAP,
    }

    print(f"\n=== WS3-MVP Re-labeling Summary ===")
    print(f"  Total windows:     {n_total}")
    print(f"  Already positive:  {n_already_pos}")
    print(f"  Flipped amb→pos:   {n_flipped}")
    print(f"  Ambiguous kept:    {n_amb_kept}")
    print(f"  New total positive: {n_already_pos + n_flipped} (+{n_flipped})")
    print(f"  Flipped by type:   {dict(flipped_by_type)}")
    if flipped_bgc_lengths:
        print(f"  Flipped BGC lengths: min={int(lens.min())} max={int(lens.max())} "
              f"mean={int(lens.mean())} median={int(np.median(lens))}")

    # Save summary
    summary_path = args.out.replace(".csv", "_relabel_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"New metadata saved to {args.out}")


if __name__ == "__main__":
    main()
