#!/usr/bin/env python3
"""Build a pilot weak-supervision table from windows and antiSMASH regions.

This script does not create new ground truth. It converts the current
antiSMASH-only metadata into confidence buckets that can drive PU training and
data audits without touching the 9-genome benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from benchmark.nine_genome import GENOME_IDS
except Exception:  # pragma: no cover
    GENOME_IDS = [
        "GCA_000158915.1",
        "GCA_000158875.1",
        "GCA_000154945.1",
        "GCA_000158895.1",
        "GCA_000156435.1",
        "GCA_000568915.1",
        "GCA_000158815.1",
        "GCA_000568255.1",
        "GCA_000156475.1",
    ]


@dataclass(frozen=True)
class BgcRegion:
    start: int
    end: int


@dataclass(frozen=True)
class WindowRow:
    window_id: str
    genome_id: str
    contig_id: str
    start: int
    end: int
    actual_length: int
    label: str
    split: str


@dataclass(frozen=True)
class BucketParams:
    trusted_negative_margin_bp: int = 20_000
    boundary_margin_bp: int = 2_000
    min_positive_overlap_bp: int = 1_000
    min_positive_fraction: float = 0.10
    short_bgc_bp: int = 10_000


FIELDNAMES = [
    "bucket",
    "window_id",
    "genome_id",
    "contig_id",
    "start",
    "end",
    "actual_length",
    "label",
    "split",
    "overlap_bp",
    "overlap_fraction",
    "nearest_bgc_distance_bp",
    "min_overlapped_bgc_len_bp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate automatic weak-supervision candidate buckets.")
    parser.add_argument("--windows-csv", required=True, type=Path)
    parser.add_argument("--bgc-annot", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--trusted-negative-margin-bp", type=int, default=20_000)
    parser.add_argument("--boundary-margin-bp", type=int, default=2_000)
    parser.add_argument("--min-positive-overlap-bp", type=int, default=1_000)
    parser.add_argument("--min-positive-fraction", type=float, default=0.10)
    parser.add_argument("--short-bgc-bp", type=int, default=10_000)
    parser.add_argument("--sample-per-bucket", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nine-genome", action="store_true",
                        help="By default 9-genome windows are excluded from this pilot.")
    parser.add_argument("--write-full-csv", action="store_true",
                        help="Also write all classified windows; can be large.")
    return parser.parse_args()


def load_bgc_index(path: Path) -> dict[tuple[str, str], list[BgcRegion]]:
    index: dict[tuple[str, str], list[BgcRegion]] = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            index[(row["genome_id"], row["contig_id"])].append(
                BgcRegion(int(row["start_bp"]), int(row["end_bp"]))
            )
    for key in index:
        index[key].sort(key=lambda r: (r.start, r.end))
    return dict(index)


def read_window_row(row: dict[str, str]) -> WindowRow:
    return WindowRow(
        window_id=row["window_id"],
        genome_id=row["genome_id"],
        contig_id=row["contig_id"],
        start=int(row["window_start"]),
        end=int(row["window_end"]),
        actual_length=int(row["actual_length"]),
        label=row["label"],
        split=row["split"],
    )


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def distance_to_region(start: int, end: int, region: BgcRegion) -> int:
    if end < region.start:
        return region.start - end
    if start > region.end:
        return start - region.end
    return 0


def classify_window(
    window: WindowRow,
    bgc_index: dict[tuple[str, str], list[BgcRegion]],
    params: BucketParams,
) -> dict[str, str | int | float]:
    regions = bgc_index.get((window.genome_id, window.contig_id), [])
    overlaps: list[tuple[int, int]] = []
    nearest = None
    for region in regions:
        ov = overlap_len(window.start, window.end, region.start, region.end)
        if ov > 0:
            overlaps.append((ov, region.end - region.start))
        d = distance_to_region(window.start, window.end, region)
        nearest = d if nearest is None else min(nearest, d)

    overlap_bp = sum(ov for ov, _length in overlaps)
    overlap_fraction = overlap_bp / max(1, window.actual_length)
    min_overlapped_bgc_len = min((length for _ov, length in overlaps), default=0)
    nearest_distance = nearest if nearest is not None else -1

    if window.label == "ambiguous":
        bucket = "ambiguous_mask"
    elif overlap_bp > 0 and min_overlapped_bgc_len and min_overlapped_bgc_len < params.short_bgc_bp:
        bucket = "short_positive"
    elif overlap_bp >= params.min_positive_overlap_bp or overlap_fraction >= params.min_positive_fraction:
        bucket = "positive_core"
    elif overlap_bp > 0:
        bucket = "boundary_uncertain"
    elif nearest_distance >= 0 and nearest_distance <= params.boundary_margin_bp:
        bucket = "boundary_uncertain"
    elif window.label == "negative" and (
        nearest_distance < 0 or nearest_distance >= params.trusted_negative_margin_bp
    ):
        bucket = "trusted_negative"
    else:
        bucket = "weak_unlabeled"

    return {
        "bucket": bucket,
        "window_id": window.window_id,
        "genome_id": window.genome_id,
        "contig_id": window.contig_id,
        "start": window.start,
        "end": window.end,
        "actual_length": window.actual_length,
        "label": window.label,
        "split": window.split,
        "overlap_bp": int(overlap_bp),
        "overlap_fraction": round(float(overlap_fraction), 6),
        "nearest_bgc_distance_bp": int(nearest_distance),
        "min_overlapped_bgc_len_bp": int(min_overlapped_bgc_len),
    }


def reservoir_add(samples: dict[str, list[dict]], counts: Counter, row: dict, limit: int, rng: random.Random) -> None:
    bucket = str(row["bucket"])
    counts[bucket] += 1
    if limit <= 0:
        return
    current = samples[bucket]
    if len(current) < limit:
        current.append(row)
        return
    idx = rng.randrange(counts[bucket])
    if idx < limit:
        current[idx] = row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    params = BucketParams(
        trusted_negative_margin_bp=args.trusted_negative_margin_bp,
        boundary_margin_bp=args.boundary_margin_bp,
        min_positive_overlap_bp=args.min_positive_overlap_bp,
        min_positive_fraction=args.min_positive_fraction,
        short_bgc_bp=args.short_bgc_bp,
    )
    rng = random.Random(args.seed)
    bgc_index = load_bgc_index(args.bgc_annot)
    nine_ids = set(GENOME_IDS)
    counts: Counter = Counter()
    counts_by_split: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[dict]] = defaultdict(list)
    full_rows: list[dict] = []
    excluded_nine_genome = 0

    with open(args.windows_csv) as fh:
        for raw in csv.DictReader(fh):
            window = read_window_row(raw)
            if not args.include_nine_genome and window.genome_id in nine_ids:
                excluded_nine_genome += 1
                continue
            row = classify_window(window, bgc_index, params)
            reservoir_add(samples, counts, row, args.sample_per_bucket, rng)
            counts_by_split[window.split][str(row["bucket"])] += 1
            if args.write_full_csv:
                full_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = [
        row
        for bucket in sorted(samples)
        for row in sorted(samples[bucket], key=lambda r: str(r["window_id"]))
    ]
    write_csv(args.out_dir / "auto_weak_pilot_sample.csv", sample_rows)
    if args.write_full_csv:
        write_csv(args.out_dir / "auto_weak_pilot_full.csv", full_rows)

    summary = {
        "inputs": {
            "windows_csv": str(args.windows_csv),
            "bgc_annot": str(args.bgc_annot),
            "exclude_nine_genome": not args.include_nine_genome,
            "excluded_nine_genome_windows": excluded_nine_genome,
        },
        "params": params.__dict__,
        "counts_by_bucket": dict(sorted(counts.items())),
        "counts_by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(counts_by_split.items())
        },
        "sample_rows": len(sample_rows),
        "note": (
            "These are automatic weak-supervision buckets, not curated ground truth. "
            "Use trusted_negative for stronger negative weighting; keep boundary_uncertain/weak_unlabeled low-weight or masked."
        ),
    }
    (args.out_dir / "auto_weak_pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
