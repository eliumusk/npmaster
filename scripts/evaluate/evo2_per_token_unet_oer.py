#!/usr/bin/env python3
"""Compare per-token model predictions on OER against antiSMASH regions.

Loads per-token decoded region CSVs and compares with antiSMASH ground truth
(parsed from per-sample JSON files). Also loads evo2_lr_learned_decoder and evo2_lr_multiscale
region CSVs as baselines for a head-to-head comparison table.

Metrics:
  - AS-recall (asymmetric 50% overlap): a TP = antiSMASH region where
    overlap / as_region_length >= 0.5.
  - Novel count: predictions not overlapping any antiSMASH region (any-overlap).
  - Region length statistics (median, min, max).
  - Stratified AS-recall by contig length bucket (10-20kb, 20-50kb, >=50kb).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


# --------------- data types ---------------

class Region(NamedTuple):
    contig: str
    start: int
    end: int


CONTIG_LENGTH_BUCKETS = [
    ("10-20kb", 10_000, 20_000),
    ("20-50kb", 20_000, 50_000),
    (">=50kb", 50_000, None),
]


def contig_length_bucket(length: int) -> str:
    for name, lo, hi in CONTIG_LENGTH_BUCKETS:
        if length >= lo and (hi is None or length < hi):
            return name
    return "<10kb"


# --------------- antiSMASH JSON parser ---------------

def parse_antismash_dir(
    antismash_dir: Path,
    sample_glob: str = "OER*.ge10000",
) -> list[Region]:
    """Parse antiSMASH regions from per-sample JSON files.

    Layout: antismash_dir/<sample>.ge10000/<sample>.ge10000.json
    Each JSON has records[i].features where type=="region" with location
    string like "[12345:>67890](+)". Coordinates are 0-based half-open
    (BioPython convention used by antiSMASH JSON output).

    Falls back to records[i].areas if present (older antiSMASH format).
    """
    regions: list[Region] = []
    for sample_dir in sorted(antismash_dir.glob(sample_glob)):
        if not sample_dir.is_dir():
            continue
        # Find the JSON file -- try exact name match first, then any .json
        json_path = sample_dir / f"{sample_dir.name}.json"
        if not json_path.exists():
            json_files = list(sample_dir.glob("*.json"))
            if not json_files:
                continue
            json_path = json_files[0]

        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARN: skipping {json_path}: {exc}")
            continue

        records = data.get("records", [])
        for rec in records:
            contig = rec.get("id") or rec.get("name")
            if not contig:
                continue

            # Strategy 1: parse from features (antiSMASH 7.x canonical)
            found_from_features = False
            for feat in rec.get("features", []):
                if feat.get("type") != "region":
                    continue
                loc = feat.get("location", "")
                m = re.search(r"\[(\d+):[><]?(\d+)\]", loc)
                if not m:
                    continue
                s, e = int(m.group(1)), int(m.group(2))
                if e > s:
                    regions.append(Region(contig, s, e))
                    found_from_features = True

            # Strategy 2: fallback to areas array (older format)
            if not found_from_features:
                for area in rec.get("areas", []):
                    start_1 = area.get("start")
                    end_1 = area.get("end")
                    if start_1 is None or end_1 is None:
                        continue
                    # areas use 1-based inclusive coords
                    s = int(start_1) - 1
                    e = int(end_1)
                    if e > s:
                        regions.append(Region(contig, s, e))

    return regions


# --------------- region CSV parser ---------------

def read_regions_flat(csv_path: Path, strip_prefix: bool = True) -> list[Region]:
    """Read a regions CSV (genome,contig,start,end,...) -> flat list of Regions.

    If strip_prefix is True, contig "OER00425668|k141_xxx" becomes "k141_xxx"
    to match antiSMASH contig IDs.
    """
    regions: list[Region] = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            contig = row["contig"]
            if strip_prefix and "|" in contig:
                contig = contig.split("|", 1)[1]
            s = int(row["start"])
            e = int(row["end"])
            if e > s:
                regions.append(Region(contig, s, e))
    return regions


# --------------- FASTA contig lengths ---------------

def load_contig_lengths(fasta_dir: Path) -> dict[str, int]:
    """Load contig lengths from all FASTA files in a directory.

    Handles .fa, .fna, .fasta extensions. Contig IDs are the first word
    after '>' on each header line. For compound IDs like
    "OER00425668|k141_xxx", stores under the full ID.
    """
    lengths: dict[str, int] = {}
    extensions = ("*.fa", "*.fna", "*.fasta")
    for ext in extensions:
        for fa_path in sorted(fasta_dir.glob(ext)):
            cur_id = None
            cur_len = 0
            with open(fa_path) as fh:
                for line in fh:
                    if line.startswith(">"):
                        if cur_id is not None:
                            lengths[cur_id] = cur_len
                        cur_id = line[1:].split()[0]
                        cur_len = 0
                    else:
                        cur_len += len(line.rstrip())
            if cur_id is not None:
                lengths[cur_id] = cur_len
    return lengths


# --------------- overlap helpers ---------------

def interval_overlap(a_s: int, a_e: int, b_s: int, b_e: int) -> int:
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def compute_as_recall(
    as_regions: list[Region],
    pred_regions: list[Region],
    threshold: float = 0.5,
) -> dict:
    """Compute antiSMASH recall using asymmetric overlap rule.

    A TP = antiSMASH region where overlap / as_region_length >= threshold.
    Also counts novel predictions (no overlap with any AS region).
    """
    # Index predictions by contig for fast lookup
    pred_by_contig: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in pred_regions:
        pred_by_contig[r.contig].append((r.start, r.end))

    tp = 0
    for asr in as_regions:
        as_len = asr.end - asr.start
        if as_len <= 0:
            continue
        best_ov = 0
        for (ps, pe) in pred_by_contig.get(asr.contig, []):
            ov = interval_overlap(asr.start, asr.end, ps, pe)
            if ov > best_ov:
                best_ov = ov
        if best_ov / as_len >= threshold:
            tp += 1

    # Count novel: predictions with zero overlap to any AS region
    as_by_contig: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for asr in as_regions:
        as_by_contig[asr.contig].append((asr.start, asr.end))

    novel = 0
    for r in pred_regions:
        has_overlap = False
        for (a_s, a_e) in as_by_contig.get(r.contig, []):
            if interval_overlap(r.start, r.end, a_s, a_e) > 0:
                has_overlap = True
                break
        if not has_overlap:
            novel += 1

    n_as = len(as_regions)
    n_pred = len(pred_regions)
    recall = tp / n_as if n_as else 0.0

    return {
        "n_as": n_as,
        "n_pred": n_pred,
        "tp": tp,
        "as_recall": recall,
        "novel": novel,
    }


def compute_as_recall_by_contig_bucket(
    as_regions: list[Region],
    pred_regions: list[Region],
    contig_lengths: dict[str, int],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """AS-recall stratified by contig length bucket.

    contig_lengths keys may be compound "OER...|k141_xxx"; we try both
    the full key and the short key (after |) for lookup.
    """
    # Build a contig -> length map that handles prefix stripping
    contig_len_map: dict[str, int] = {}
    for cid, clen in contig_lengths.items():
        contig_len_map[cid] = clen
        # Also store short form
        if "|" in cid:
            short = cid.split("|", 1)[1]
            if short not in contig_len_map:
                contig_len_map[short] = clen

    # Index predictions by contig
    pred_by_contig: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in pred_regions:
        pred_by_contig[r.contig].append((r.start, r.end))

    # Per-bucket stats
    bucket_stats: dict[str, dict] = {}
    for bname, _, _ in CONTIG_LENGTH_BUCKETS:
        bucket_stats[bname] = {"n_as": 0, "tp": 0}

    n_unknown_contig = 0
    for asr in as_regions:
        clen = contig_len_map.get(asr.contig)
        if clen is None:
            n_unknown_contig += 1
            continue
        bname = contig_length_bucket(clen)
        if bname not in bucket_stats:
            continue  # <10kb not tracked

        bucket_stats[bname]["n_as"] += 1

        as_len = asr.end - asr.start
        if as_len <= 0:
            continue
        best_ov = 0
        for (ps, pe) in pred_by_contig.get(asr.contig, []):
            ov = interval_overlap(asr.start, asr.end, ps, pe)
            if ov > best_ov:
                best_ov = ov
        if best_ov / as_len >= threshold:
            bucket_stats[bname]["tp"] += 1

    result = {}
    for bname in bucket_stats:
        n = bucket_stats[bname]["n_as"]
        tp = bucket_stats[bname]["tp"]
        recall = tp / n if n else 0.0
        result[bname] = {
            "n_as": n,
            "tp": tp,
            "as_recall": recall,
            "label": f"{tp}/{n} ({recall:.3f})",
        }
    if n_unknown_contig > 0:
        result["_unknown_contig_len"] = n_unknown_contig
    return result


def region_length_stats(regions: list[Region]) -> dict:
    """Compute length statistics for a list of regions."""
    if not regions:
        return {"n": 0, "median": 0, "min": 0, "max": 0}
    lens = sorted(r.end - r.start for r in regions)
    n = len(lens)
    median = lens[n // 2] if n % 2 else (lens[n // 2 - 1] + lens[n // 2]) // 2
    return {"n": n, "median": int(median), "min": lens[0], "max": lens[-1]}


# --------------- evaluation for one method ---------------

def evaluate_method(
    name: str,
    pred_regions: list[Region],
    as_regions: list[Region],
    contig_lengths: dict[str, int],
) -> dict:
    """Full evaluation of one method against antiSMASH."""
    overall = compute_as_recall(as_regions, pred_regions)
    buckets = compute_as_recall_by_contig_bucket(
        as_regions, pred_regions, contig_lengths,
    )
    pred_stats = region_length_stats(pred_regions)

    return {
        "method": name,
        "n_pred": overall["n_pred"],
        "n_as": overall["n_as"],
        "tp": overall["tp"],
        "as_recall": overall["as_recall"],
        "novel": overall["novel"],
        "pred_median_len": pred_stats["median"],
        "pred_min_len": pred_stats["min"],
        "pred_max_len": pred_stats["max"],
        "bucket_recall": buckets,
    }


# --------------- CLI ---------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare per-token predictions on OER against antiSMASH.",
    )
    p.add_argument("--regions-csv", required=True, type=Path,
                   help="Per-token decoded regions CSV (genome,contig,start,end,...)")
    p.add_argument("--antismash-dir", required=True, type=Path,
                   help="Parent dir containing OER*.ge10000/ antiSMASH outputs")
    p.add_argument("--evo2-lr-learned-csv", type=Path, default=None,
                   help="evo2_lr_learned_decoder regions CSV for comparison")
    p.add_argument("--evo2-lr-hysteresis-csv", type=Path, default=None,
                   help="evo2_lr_multiscale regions CSV for comparison")
    p.add_argument("--fasta-dir", type=Path, default=None,
                   help="FASTA dir for contig lengths (for stratified recall)")
    p.add_argument("--sample-glob", default="OER*.ge10000",
                   help="Glob for antiSMASH sample subdirs (default: OER*.ge10000)")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Output JSON path (optional)")
    return p.parse_args()


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'method':<25s} {'n_pred':>7s} {'AS-recall':>10s} "
        f"{'novel':>6s} {'med_len':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['method']:<25s} {r['n_pred']:>7d} "
            f"{r['as_recall']:>10.4f} "
            f"{r['novel']:>6d} {r['pred_median_len']:>8d}"
        )


def print_bucket_table(results: list[dict]) -> None:
    """Print per-contig-length-bucket AS-recall for all methods."""
    bucket_names = [name for name, _, _ in CONTIG_LENGTH_BUCKETS]
    header = f"{'method':<25s}" + "".join(f"  {b:>12s}" for b in bucket_names)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['method']:<25s}"
        for b in bucket_names:
            bdata = r["bucket_recall"].get(b, {})
            label = bdata.get("label", "n/a")
            row += f"  {label:>12s}"
        print(row)


def main():
    args = parse_args()

    # ---- 1. Parse antiSMASH ----
    print("[1] Parsing antiSMASH regions ...")
    as_regions = parse_antismash_dir(args.antismash_dir, sample_glob=args.sample_glob)
    as_len_stats = region_length_stats(as_regions)
    print(f"    {len(as_regions)} antiSMASH regions "
          f"(median length {as_len_stats['median']})")

    # ---- 2. Load contig lengths ----
    contig_lengths: dict[str, int] = {}
    if args.fasta_dir and args.fasta_dir.exists():
        print("[2] Loading contig lengths from FASTA ...")
        contig_lengths = load_contig_lengths(args.fasta_dir)
        print(f"    {len(contig_lengths)} contigs loaded")
    else:
        print("[2] No FASTA dir provided; contig-length stratification skipped")

    # ---- 3. Evaluate per-token predictions ----
    print("[3] Evaluating per-token predictions ...")
    pt_regions = read_regions_flat(args.regions_csv, strip_prefix=True)
    pt_result = evaluate_method("per-token", pt_regions, as_regions, contig_lengths)

    results = [pt_result]

    # ---- 4. Evaluate baselines ----
    if args.evo2_lr_learned_csv and args.evo2_lr_learned_csv.exists():
        print("[4a] Evaluating evo2_lr_learned_decoder ...")
        lr_learned_regions = read_regions_flat(args.evo2_lr_learned_csv, strip_prefix=True)
        lr_learned_result = evaluate_method(
            "evo2_lr_learned_decoder", lr_learned_regions, as_regions, contig_lengths,
        )
        results.append(lr_learned_result)

    if args.evo2_lr_hysteresis_csv and args.evo2_lr_hysteresis_csv.exists():
        print("[4b] Evaluating evo2_lr_multiscale ...")
        lr_hysteresis_regions = read_regions_flat(args.evo2_lr_hysteresis_csv, strip_prefix=True)
        lr_hysteresis_result = evaluate_method(
            "evo2_lr_multiscale", lr_hysteresis_regions, as_regions, contig_lengths,
        )
        results.append(lr_hysteresis_result)

    # ---- 5. Report ----
    print()
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print()
    print_comparison_table(results)

    print()
    print(f"antiSMASH regions: {len(as_regions)} "
          f"(median length {as_len_stats['median']})")

    if contig_lengths:
        print()
        print("AS-recall by contig length bucket:")
        print_bucket_table(results)

    # ---- 6. Save JSON ----
    output = {
        "antismash": {
            "n_regions": len(as_regions),
            "median_length": as_len_stats["median"],
            "min_length": as_len_stats["min"],
            "max_length": as_len_stats["max"],
        },
        "methods": {r["method"]: r for r in results},
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nWrote {args.out_json}")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
