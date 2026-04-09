#!/usr/bin/env python3
"""
P1-T2: Edge-Padding MVP
========================
Add padded windows for:
  1. Short contigs (< W) that currently generate zero windows
  2. Contig tails where the remainder is too short for a full window

Padded windows are right-padded with 'N' to reach W=16384 bp.
Metadata is updated with new rows (edge_padded=1).
Embeddings must be extracted separately for the new windows only.

This script:
  - Reads existing windows_metadata + bgc_annotations + genome fastas
  - Identifies contigs that need padding
  - Generates new padded window metadata rows
  - Extracts actual DNA sequences for the padded windows (for Evo2 input)
  - Writes: updated metadata CSV, padded sequences FASTA, summary JSON

Usage:
  python p1_t2_edge_padding.py \
    --meta /data/muskliu/np_master/data/windows/windows_metadata_cap08.csv \
    --bgc-csv /data/muskliu/np_master/data/bgc_annotations.csv \
    --genomes-dir /data/muskliu/np_master/data/genomes \
    --out /data/muskliu/np_master/data/windows/edge_padding \
    --min-contig 1000
"""
import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

CLASSES = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
W = 16384
S = 8192


def load_fasta_sequences(path):
    """Load all contig sequences from a FASTA file."""
    op = gzip.open if str(path).endswith(".gz") else open
    seqs = {}
    cur_id = None
    cur_seq = []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


def label_window(w_start, w_end, contig_bgcs, overlap_pos=0.5, overlap_type=0.3,
                 short_bgc_capture=0.8, buffer=10000):
    """Label a single window using the same logic as 05_windows_and_labels.py."""
    overlaps = []
    min_dist = None
    overlaps_trunc = False

    for b in contig_bgcs:
        ov = max(0, min(w_end, b["end"]) - max(w_start, b["start"]))
        if ov > 0:
            bgc_len = b["end"] - b["start"]
            bgc_capture = ov / bgc_len if bgc_len > 0 else 0
            overlaps.append((ov / W, bgc_capture, b))
            if b.get("truncated", False):
                overlaps_trunc = True
        else:
            d = min(abs(w_end - b["start"]), abs(w_start - b["end"]))
            if min_dist is None or d < min_dist:
                min_dist = d

    if overlaps:
        max_or = max(o for o, _, _ in overlaps)
        max_bgc_cap = max(bc for _, bc, _ in overlaps)
        is_positive = (max_or >= overlap_pos) or (short_bgc_capture > 0 and max_bgc_cap >= short_bgc_capture)

        if is_positive:
            classes = set()
            for o, bc, b in overlaps:
                if o >= overlap_type or (short_bgc_capture > 0 and bc >= short_bgc_capture):
                    classes.update(b["classes"])
            return {
                "label": "positive",
                "classes": classes,
                "max_overlap": max_or,
                "nearest_dist": "",
                "overlaps_trunc": overlaps_trunc,
            }
        else:
            return {
                "label": "ambiguous",
                "classes": set(),
                "max_overlap": max_or,
                "nearest_dist": "",
                "overlaps_trunc": overlaps_trunc,
            }
    else:
        if min_dist is None or min_dist >= buffer:
            return {
                "label": "negative",
                "classes": set(),
                "max_overlap": 0,
                "nearest_dist": min_dist or "inf",
                "overlaps_trunc": False,
            }
        else:
            return {
                "label": "ambiguous",
                "classes": set(),
                "max_overlap": 0,
                "nearest_dist": min_dist,
                "overlaps_trunc": False,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="existing windows_metadata_cap08.csv")
    ap.add_argument("--bgc-csv", required=True, help="bgc_annotations.csv")
    ap.add_argument("--genomes-dir", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--min-contig", type=int, default=1000,
                    help="minimum contig length to create a padded window (default: 1000bp)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ─── 1. Load BGC annotations ───
    print("Loading BGC annotations...")
    bgcs_by_contig = defaultdict(list)
    genome_phylum = {}
    with open(args.bgc_csv) as f:
        for row in csv.DictReader(f):
            key = (row["genome_id"], row["contig_id"])
            bgcs_by_contig[key].append({
                "bgc_id": row["bgc_id"],
                "start": int(row["start_bp"]),
                "end": int(row["end_bp"]),
                "classes": row["bgc_types"].split(";") if row["bgc_types"] else [],
                "truncated": row.get("is_truncated", "0") == "1",
            })
            genome_phylum[row["genome_id"]] = row.get("phylum", "")

    # Load phylum from selected_genomes.tsv if available
    sel = os.path.join(os.path.dirname(args.bgc_csv), "selected_genomes.tsv")
    if os.path.exists(sel):
        with open(sel) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                genome_phylum.setdefault(row["ncbi_accession"], row["phylum"])

    # ─── 2. Load existing metadata to find split assignments and max window ID ───
    print("Loading existing window metadata...")
    existing_meta = []
    genome_split = {}  # genome_id -> split
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            existing_meta.append(row)
            genome_split[row["genome_id"]] = row.get("split", "")

    max_wid = max(int(row["window_id"][1:]) for row in existing_meta)
    print(f"  Existing windows: {len(existing_meta)}, max window ID: w{max_wid:09d}")

    # Index existing windows by contig to know what's already covered
    existing_coverage = defaultdict(list)  # (genome, contig) -> [(start, end)]
    for row in existing_meta:
        existing_coverage[(row["genome_id"], row["contig_id"])].append(
            (int(row["window_start"]), int(row["window_end"]))
        )

    # ─── 3. Scan genomes for contigs needing padding ───
    print("Scanning genomes for padding opportunities...")
    new_windows = []
    padded_sequences = []  # (window_id, sequence_padded_to_W)
    next_wid = max_wid + 1

    genome_files = sorted(fn for fn in os.listdir(args.genomes_dir) if fn.endswith(".fna.gz"))
    n_short_contigs = 0
    n_tail_windows = 0

    for fn in genome_files:
        acc = fn[:-7]
        phylum = genome_phylum.get(acc, "")
        split = genome_split.get(acc, "")
        if not split:
            continue  # genome not in any split, skip

        path = os.path.join(args.genomes_dir, fn)
        seqs = load_fasta_sequences(path)

        for contig_id, seq in seqs.items():
            clen = len(seq)
            contig_bgcs = bgcs_by_contig.get((acc, contig_id), [])
            existing_wins = existing_coverage.get((acc, contig_id), [])

            # Case 1: Short contig (< W) — no windows exist at all
            if clen < W and clen >= args.min_contig:
                # Create one padded window covering the entire contig
                padded_seq = seq + "N" * (W - clen)
                w_start = 0
                w_end = clen  # actual end (not W), for labeling purposes we use actual coords

                # Label using actual contig coordinates
                lbl = label_window(w_start, clen, contig_bgcs)

                wid_str = f"w{next_wid:09d}"
                new_windows.append({
                    "window_id": wid_str,
                    "genome_id": acc,
                    "contig_id": contig_id,
                    "window_start": 0,
                    "window_end": clen,  # store actual length, not padded
                    "label": lbl["label"],
                    "classes": lbl["classes"],
                    "max_overlap": lbl["max_overlap"],
                    "nearest_dist": lbl["nearest_dist"],
                    "overlaps_trunc": lbl["overlaps_trunc"],
                    "phylum": phylum,
                    "split": split,
                    "edge_padded": True,
                    "actual_length": clen,
                })
                padded_sequences.append((wid_str, padded_seq))
                next_wid += 1
                n_short_contigs += 1

            # Case 2: Contig tail — last covered position to contig end
            elif clen >= W:
                n_windows = (clen - W) // S + 1
                last_window_end = (n_windows - 1) * S + W
                tail_start = last_window_end  # start of uncovered region
                tail_len = clen - tail_start

                if tail_len >= args.min_contig:
                    # Create one padded window starting at tail_start
                    tail_seq = seq[tail_start:clen]
                    padded_seq = tail_seq + "N" * (W - tail_len)

                    lbl = label_window(tail_start, clen, contig_bgcs)

                    wid_str = f"w{next_wid:09d}"
                    new_windows.append({
                        "window_id": wid_str,
                        "genome_id": acc,
                        "contig_id": contig_id,
                        "window_start": tail_start,
                        "window_end": clen,
                        "label": lbl["label"],
                        "classes": lbl["classes"],
                        "max_overlap": lbl["max_overlap"],
                        "nearest_dist": lbl["nearest_dist"],
                        "overlaps_trunc": lbl["overlaps_trunc"],
                        "phylum": phylum,
                        "split": split,
                        "edge_padded": True,
                        "actual_length": tail_len,
                    })
                    padded_sequences.append((wid_str, padded_seq))
                    next_wid += 1
                    n_tail_windows += 1

    print(f"\n  New padded windows: {len(new_windows)}")
    print(f"    Short contigs (< {W}bp): {n_short_contigs}")
    print(f"    Contig tails: {n_tail_windows}")

    # ─── 4. Summarize new window labels ───
    n_new_pos = sum(1 for w in new_windows if w["label"] == "positive")
    n_new_neg = sum(1 for w in new_windows if w["label"] == "negative")
    n_new_amb = sum(1 for w in new_windows if w["label"] == "ambiguous")
    print(f"    Positive: {n_new_pos}")
    print(f"    Negative: {n_new_neg}")
    print(f"    Ambiguous: {n_new_amb}")

    # By split
    split_counts = defaultdict(lambda: {"pos": 0, "neg": 0, "amb": 0, "total": 0})
    for w in new_windows:
        s = w["split"]
        split_counts[s]["total"] += 1
        if w["label"] == "positive":
            split_counts[s]["pos"] += 1
        elif w["label"] == "negative":
            split_counts[s]["neg"] += 1
        else:
            split_counts[s]["amb"] += 1
    print("\n  By split:")
    for s in ["train", "val", "test_id", "test_ood"]:
        c = split_counts[s]
        print(f"    {s}: {c['total']} (pos={c['pos']}, neg={c['neg']}, amb={c['amb']})")

    # By type (for positive windows)
    type_counts = defaultdict(int)
    for w in new_windows:
        if w["label"] == "positive":
            for c in w["classes"]:
                type_counts[c] += 1
    if type_counts:
        print("\n  New positive by type:")
        for t in CLASSES:
            if type_counts.get(t, 0) > 0:
                print(f"    {t}: {type_counts[t]}")

    # ─── 5. Write updated metadata CSV ───
    print("\nWriting updated metadata...")
    out_meta = os.path.join(args.out, "windows_metadata_edgepad.csv")

    # Copy existing rows and append new ones
    fieldnames = ["window_id", "genome_id", "contig_id", "window_start", "window_end",
                  "label"] + CLASSES + ["is_negative", "max_bgc_overlap_ratio",
                  "nearest_bgc_distance", "overlaps_truncated_bgc", "phylum",
                  "split", "edge_padded", "actual_length"]

    with open(out_meta, "w", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()

        # Write existing rows (add edge_padded=0, actual_length=W)
        for row in existing_meta:
            out_row = dict(row)
            out_row["edge_padded"] = "0"
            out_row["actual_length"] = str(W)
            # Ensure all fieldnames present
            for fn in fieldnames:
                if fn not in out_row:
                    out_row[fn] = ""
            writer.writerow({k: out_row.get(k, "") for k in fieldnames})

        # Write new padded rows
        for w in new_windows:
            out_row = {
                "window_id": w["window_id"],
                "genome_id": w["genome_id"],
                "contig_id": w["contig_id"],
                "window_start": str(w["window_start"]),
                "window_end": str(w["window_end"]),
                "label": w["label"],
                "is_negative": "1" if w["label"] == "negative" else "0",
                "max_bgc_overlap_ratio": f"{w['max_overlap']:.3f}" if w["max_overlap"] else "0",
                "nearest_bgc_distance": str(w["nearest_dist"]),
                "overlaps_truncated_bgc": "1" if w["overlaps_trunc"] else "0",
                "phylum": w["phylum"],
                "split": w["split"],
                "edge_padded": "1",
                "actual_length": str(w["actual_length"]),
            }
            for c in CLASSES:
                out_row[c] = "1" if c in w.get("classes", set()) else "0"
            writer.writerow(out_row)

    print(f"  Written to {out_meta}")
    print(f"  Total rows: {len(existing_meta) + len(new_windows)}")

    # ─── 6. Write padded sequences FASTA (for Evo2 extraction) ───
    seq_fasta = os.path.join(args.out, "padded_sequences.fasta")
    with open(seq_fasta, "w") as f:
        for wid, seq in padded_sequences:
            f.write(f">{wid}\n")
            # Write in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
    print(f"  Padded sequences: {seq_fasta} ({len(padded_sequences)} sequences)")

    # ─── 7. Write window ID list for extraction ───
    wid_list = os.path.join(args.out, "new_window_ids.txt")
    with open(wid_list, "w") as f:
        for wid, _ in padded_sequences:
            f.write(wid + "\n")
    print(f"  Window ID list: {wid_list}")

    # ─── 8. Save summary ───
    summary = {
        "task_id": "P1-T2",
        "description": "Edge-padding MVP",
        "min_contig": args.min_contig,
        "window_size": W,
        "existing_windows": len(existing_meta),
        "new_windows": len(new_windows),
        "new_from_short_contigs": n_short_contigs,
        "new_from_tails": n_tail_windows,
        "new_positive": n_new_pos,
        "new_negative": n_new_neg,
        "new_ambiguous": n_new_amb,
        "by_split": {s: dict(c) for s, c in split_counts.items()},
        "by_type": dict(type_counts),
        "outputs": {
            "metadata": out_meta,
            "sequences_fasta": seq_fasta,
            "window_ids": wid_list,
        },
        "next_steps": [
            "1. Extract Evo2 embeddings for padded_sequences.fasta",
            "2. Append new embeddings to existing embeddings.npy",
            "3. Retrain with windows_metadata_edgepad.csv",
            "4. Evaluate with ws1_score_swap.py --fullwindow",
        ],
    }

    with open(os.path.join(args.out, "edge_padding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: {args.out}/edge_padding_summary.json")
    print(f"\n=== NEXT STEPS ===")
    for s in summary["next_steps"]:
        print(f"  {s}")


if __name__ == "__main__":
    main()
