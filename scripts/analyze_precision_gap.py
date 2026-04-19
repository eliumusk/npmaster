#!/usr/bin/env python3
"""Diagnose why predicted regions fail the 9-genome precision rule.

This script is intentionally diagnostic-only: it decodes per-window NP-Master
predictions with the same machinery as ``evaluate_nine_genome.py``, then emits
one row per predicted region explaining whether the prediction is a bench TP or
which broad FP pattern it falls into.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_nine_genome import (  # noqa: E402
    DEFAULT_OPERATING_POINTS,
    decode_regions_hysteresis,
    load_ground_truth,
    load_windows_from_csv,
    overlap_length,
    resolve_params,
    select_genome_ids,
)


Region = tuple[str, int, int]


CSV_FIELDS = [
    "genome_id",
    "contig",
    "pred_start",
    "pred_end",
    "pred_len",
    "best_gt_id",
    "gt_contig",
    "gt_start",
    "gt_end",
    "gt_len",
    "overlap_bp",
    "overlap_over_pred",
    "overlap_over_gt",
    "overlapping_gt_count",
    "bench_precision_tp",
    "loose_tp",
    "failure_type",
]


def _region_id(genome_id: str, region: Region | None) -> str:
    if region is None:
        return ""
    contig, start, end = region
    return f"{genome_id}:{contig}:{start}-{end}"


def _best_overlap(pred_region: Region, gt_regions: Sequence[Region]) -> dict[str, Any]:
    pred_contig, pred_start, pred_end = pred_region
    pred_len = max(1, pred_end - pred_start)
    best_gt: Region | None = None
    best_overlap = 0
    overlapping_gt_count = 0

    for gt_region in gt_regions:
        gt_contig, gt_start, gt_end = gt_region
        if gt_contig != pred_contig:
            continue
        overlap = overlap_length(pred_start, pred_end, gt_start, gt_end)
        if overlap <= 0:
            continue
        overlapping_gt_count += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_gt = gt_region

    gt_len = max(1, best_gt[2] - best_gt[1]) if best_gt else 0
    return {
        "best_gt": best_gt,
        "overlap_bp": best_overlap,
        "overlap_over_pred": best_overlap / pred_len if best_overlap else 0.0,
        "overlap_over_gt": best_overlap / gt_len if best_overlap and gt_len else 0.0,
        "overlapping_gt_count": overlapping_gt_count,
    }


def classify_prediction(
    *,
    genome_id: str,
    pred_region: Region,
    gt_regions: Sequence[Region],
    min_overlap: float = 0.5,
) -> dict[str, Any]:
    """Classify one predicted region under the bench precision failure modes."""
    contig, pred_start, pred_end = pred_region
    pred_len = pred_end - pred_start
    overlap = _best_overlap(pred_region, gt_regions)
    best_gt: Region | None = overlap["best_gt"]
    overlap_bp = int(overlap["overlap_bp"])
    overlap_over_pred = float(overlap["overlap_over_pred"])
    overlap_over_gt = float(overlap["overlap_over_gt"])
    overlapping_gt_count = int(overlap["overlapping_gt_count"])

    bench_precision_tp = overlap_over_pred >= min_overlap
    loose_tp = False
    if best_gt is not None and overlap_bp > 0:
        gt_len = max(1, best_gt[2] - best_gt[1])
        loose_tp = overlap_bp / max(1, min(pred_len, gt_len)) >= min_overlap

    if bench_precision_tp:
        failure_type = "bench_tp"
    elif overlap_bp == 0:
        failure_type = "no_overlap"
    elif overlapping_gt_count >= 2:
        failure_type = "merged_multiple_gt"
    elif overlap_over_gt >= min_overlap:
        failure_type = "too_long_overlap_gt"
    elif overlap_over_gt < 0.1:
        failure_type = "short_partial_hit"
    else:
        failure_type = "flank_only_or_boundary_shift"

    gt_contig = best_gt[0] if best_gt else ""
    gt_start = best_gt[1] if best_gt else ""
    gt_end = best_gt[2] if best_gt else ""
    gt_len = best_gt[2] - best_gt[1] if best_gt else ""

    return {
        "genome_id": genome_id,
        "contig": contig,
        "pred_start": pred_start,
        "pred_end": pred_end,
        "pred_len": pred_len,
        "best_gt_id": _region_id(genome_id, best_gt),
        "gt_contig": gt_contig,
        "gt_start": gt_start,
        "gt_end": gt_end,
        "gt_len": gt_len,
        "overlap_bp": overlap_bp,
        "overlap_over_pred": overlap_over_pred,
        "overlap_over_gt": overlap_over_gt,
        "overlapping_gt_count": overlapping_gt_count,
        "bench_precision_tp": bench_precision_tp,
        "loose_tp": loose_tp,
        "failure_type": failure_type,
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["failure_type"]) for row in rows)
    stats: dict[str, dict[str, float | int]] = {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["failure_type"])].append(row)

    for failure_type, group in sorted(by_type.items()):
        pred_lens = [float(row["pred_len"]) for row in group]
        overlaps = [float(row["overlap_over_pred"]) for row in group]
        stats[failure_type] = {
            "n": len(group),
            "median_pred_len": statistics.median(pred_lens) if pred_lens else 0,
            "median_overlap_over_pred": statistics.median(overlaps) if overlaps else 0,
        }

    bench_tp = counts.get("bench_tp", 0)
    loose_tp_bench_fp = sum(
        1
        for row in rows
        if bool(row.get("loose_tp")) and str(row.get("failure_type")) != "bench_tp"
    )
    return {
        "n_pred": len(rows),
        "bench_tp": bench_tp,
        "bench_fp": len(rows) - bench_tp,
        "loose_tp_but_bench_fp": loose_tp_bench_fp,
        "failure_counts": dict(sorted(counts.items())),
        "failure_stats": stats,
    }


def analyze_results_dir(
    *,
    base: Path,
    results_dir: Path,
    genome_ids: Sequence[str],
    params: dict[str, Any],
    min_overlap: float = 0.5,
) -> dict[str, Any]:
    gt_by_genome = load_ground_truth(base / "ground_truth.json")
    rows: list[dict[str, Any]] = []

    for genome_id in genome_ids:
        csv_path = results_dir / f"{genome_id}_predictions.csv"
        windows = load_windows_from_csv(csv_path, score_family=params["score"])
        pred_regions = decode_regions_hysteresis(
            windows,
            start_threshold=params["start_threshold"],
            extend_threshold=params["extend_threshold"],
            max_gap=params["max_gap"],
            min_support_windows=params["min_support_windows"],
            length_cap=params["length_cap"],
            valley_threshold=params["valley_threshold"],
            valley_min_windows=params["valley_min_windows"],
        )
        for pred_region in pred_regions:
            rows.append(
                classify_prediction(
                    genome_id=genome_id,
                    pred_region=pred_region,
                    gt_regions=gt_by_genome.get(genome_id, []),
                    min_overlap=min_overlap,
                )
            )

    return {
        "rows": rows,
        "summary": summarize_rows(rows),
        "params": {
            "score": params["score"],
            "start_threshold": params["start_threshold"],
            "extend_threshold": params["extend_threshold"],
            "max_gap": params["max_gap"],
            "length_cap": params["length_cap"],
            "min_support_windows": params["min_support_windows"],
            "valley_threshold": params["valley_threshold"],
            "valley_min_windows": params["valley_min_windows"],
        },
    }


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/syh/benchmark/nine_genome")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--genome-ids", default=None)
    parser.add_argument("--operating-point", default=None)
    parser.add_argument("--operating-points-file", default=str(DEFAULT_OPERATING_POINTS))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--start-threshold", type=float, default=None)
    parser.add_argument("--extend-threshold", type=float, default=None)
    parser.add_argument("--max-gap", type=int, default=None)
    parser.add_argument("--length-cap", type=int, default=None)
    parser.add_argument("--min-support-windows", type=int, default=None)
    parser.add_argument("--valley-threshold", type=float, default=None)
    parser.add_argument("--valley-min-windows", type=int, default=None)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    params = resolve_params(args)
    genome_ids = select_genome_ids(args.genome_ids)
    payload = analyze_results_dir(
        base=Path(args.base),
        results_dir=Path(args.results_dir),
        genome_ids=genome_ids,
        params=params,
    )
    write_rows_csv(Path(args.out_csv), payload["rows"])
    write_summary_json(
        Path(args.out_summary),
        {
            "summary": payload["summary"],
            "params": payload["params"],
            "base": args.base,
            "results_dir": args.results_dir,
            "genome_ids": genome_ids,
        },
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
