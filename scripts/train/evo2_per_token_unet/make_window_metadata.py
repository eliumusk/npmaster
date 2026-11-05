#!/usr/bin/env python3
"""Generate sliced-window metadata from genome FASTA files.

This is used for clean per-token experiments where train geometry should differ
from the canonical 16kb window table while preserving its genome-level splits.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_per_token.labels import load_bgc_annotations  # noqa: E402


FIELDNAMES = [
    "window_id",
    "genome_id",
    "contig_id",
    "window_start",
    "window_end",
    "label",
    "split",
    "actual_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make fixed-width window metadata preserving genome splits.")
    parser.add_argument("--template-windows-csv", required=True, type=Path)
    parser.add_argument("--genome-dir", required=True, type=Path)
    parser.add_argument("--bgc-annot", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-subset-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="w8k")
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--n-subsets", type=int, default=16)
    return parser.parse_args()


def read_fna_lengths(path: Path) -> dict[str, int]:
    opener = gzip.open if str(path).endswith(".gz") else open
    lengths: dict[str, int] = {}
    cur_id: str | None = None
    cur_len = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_id is not None:
                    lengths[cur_id] = cur_len
                cur_id = line[1:].split()[0]
                cur_len = 0
            else:
                cur_len += len(line.strip())
    if cur_id is not None:
        lengths[cur_id] = cur_len
    return lengths


def load_split_maps(template_csv: Path, splits: set[str]) -> tuple[dict[str, str], dict[str, set[str]]]:
    genome_split: dict[str, str] = {}
    contigs_by_genome: dict[str, set[str]] = {}
    with open(template_csv) as fh:
        for row in csv.DictReader(fh):
            split = row["split"]
            if split not in splits:
                continue
            gid = row["genome_id"]
            previous = genome_split.get(gid)
            if previous is not None and previous != split:
                raise ValueError(f"genome {gid} appears in both {previous} and {split}")
            genome_split[gid] = split
            contigs_by_genome.setdefault(gid, set()).add(row["contig_id"])
    return genome_split, contigs_by_genome


def overlaps_any(regions: list[tuple[int, int]], start: int, end: int) -> bool:
    for region_start, region_end in regions:
        if region_end <= start:
            continue
        if region_start >= end:
            return False
        return True
    return False


def main() -> None:
    args = parse_args()
    splits = set(args.splits)
    genome_split, contigs_by_genome = load_split_maps(args.template_windows_csv, splits)
    bgc_map = load_bgc_annotations(args.bgc_annot)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    subset_files = []
    if args.out_subset_dir is not None:
        args.out_subset_dir.mkdir(parents=True, exist_ok=True)
        subset_files = [
            open(args.out_subset_dir / f"subset_{idx:02d}.txt", "w")
            for idx in range(args.n_subsets)
        ]

    counts: Counter = Counter()
    missing_genomes: list[str] = []
    n_windows = 0
    with open(args.out_csv, "w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for gid in sorted(genome_split):
            fna = args.genome_dir / f"{gid}.fna.gz"
            if not fna.exists():
                fna = args.genome_dir / f"{gid}.fna"
            if not fna.exists():
                missing_genomes.append(gid)
                continue
            lengths = read_fna_lengths(fna)
            split = genome_split[gid]
            for contig_id in sorted(contigs_by_genome.get(gid, ())):
                contig_len = lengths.get(contig_id)
                if contig_len is None or contig_len <= 0:
                    continue
                for start in range(0, contig_len, args.stride):
                    end = min(start + args.window, contig_len)
                    actual_length = end - start
                    if actual_length < 100:
                        continue
                    window_id = f"{args.prefix}_{n_windows:09d}"
                    label = "positive" if overlaps_any(bgc_map.get((gid, contig_id), []), start, end) else "negative"
                    writer.writerow(
                        {
                            "window_id": window_id,
                            "genome_id": gid,
                            "contig_id": contig_id,
                            "window_start": start,
                            "window_end": end,
                            "label": label,
                            "split": split,
                            "actual_length": actual_length,
                        }
                    )
                    if subset_files:
                        subset_files[n_windows % args.n_subsets].write(window_id + "\n")
                    counts[split] += 1
                    counts[label] += 1
                    n_windows += 1

    for fh in subset_files:
        fh.close()
    summary = {
        "out_csv": str(args.out_csv),
        "window": args.window,
        "stride": args.stride,
        "splits": sorted(splits),
        "n_windows": n_windows,
        "counts": dict(sorted(counts.items())),
        "missing_genomes": missing_genomes,
    }
    summary_path = args.out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
