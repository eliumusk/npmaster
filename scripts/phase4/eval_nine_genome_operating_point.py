#!/usr/bin/env python3
"""Evaluate NP-Master 9-genome predictions under a frozen operating point.

Reads precomputed per-window prediction CSVs from the 9-genome benchmark,
merges windows into regions with configurable threshold / max_gap, then
reports region metrics under loose / strict / IoU overlap rules.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


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


def overlap_length(s1: int, e1: int, s2: int, e2: int) -> int:
    return max(0, min(e1, e2) - max(s1, s2))


def load_ground_truth(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    with path.open() as f:
        data = json.load(f)
    return {
        genome_id: [(contig, int(start), int(end)) for contig, start, end in regions]
        for genome_id, regions in data["ground_truth"].items()
    }


def select_genome_ids(genome_ids_arg: str | None) -> list[str]:
    if not genome_ids_arg:
        return GENOME_IDS
    selected = {part.strip() for part in genome_ids_arg.split(",") if part.strip()}
    return [genome_id for genome_id in GENOME_IDS if genome_id in selected]


def split_regions_by_length_cap(
    regions: list[tuple[str, list[tuple[int, int]]]],
    length_cap: int | None,
) -> list[tuple[str, int, int]]:
    if not length_cap:
        return [
            (contig, windows[0][0], windows[-1][1])
            for contig, windows in regions
            if windows
        ]
    out: list[tuple[str, int, int]] = []
    for contig, windows in regions:
        if not windows:
            continue
        start = windows[0][0]
        end = windows[-1][1]
        if end - start <= length_cap:
            out.append((contig, start, end))
            continue
        current_chunk = [windows[0]]
        for window in windows[1:]:
            chunk_start = current_chunk[0][0]
            proposed_end = window[1]
            if proposed_end - chunk_start > length_cap:
                out.append((contig, current_chunk[0][0], current_chunk[-1][1]))
                current_chunk = [window]
            else:
                current_chunk.append(window)
        if current_chunk:
            out.append((contig, current_chunk[0][0], current_chunk[-1][1]))
    return out


def load_regions_from_csv(
    csv_path: Path,
    threshold: float,
    max_gap: int,
    length_cap: int | None = None,
) -> list[tuple[str, int, int]]:
    windows: list[tuple[str, int, int]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["bgc_score"]) >= threshold:
                windows.append((row["contig"], int(row["start"]), int(row["end"])))
    if not windows:
        return []
    windows.sort(key=lambda x: (x[0], x[1]))
    merged: list[dict[str, object]] = [
        {"contig": windows[0][0], "windows": [(windows[0][1], windows[0][2])]}
    ]
    for contig, start, end in windows[1:]:
        last = merged[-1]
        last_contig = str(last["contig"])
        last_windows = last["windows"]
        assert isinstance(last_windows, list)
        last_end = int(last_windows[-1][1])
        if contig == last_contig and start <= last_end + max_gap:
            last_windows.append((start, end))
        else:
            merged.append({"contig": contig, "windows": [(start, end)]})
    return split_regions_by_length_cap(
        [
            (
                str(region["contig"]),
                [(int(s), int(e)) for s, e in region["windows"]],
            )
            for region in merged
        ],
        length_cap,
    )


def score_overlap(pred: tuple[str, int, int], gt: tuple[str, int, int], mode: str) -> float:
    _, ps, pe = pred
    _, gs, ge = gt
    pl = pe - ps
    gl = ge - gs
    ov = overlap_length(ps, pe, gs, ge)
    if ov <= 0:
        return 0.0
    if mode == "loose":
        return ov / max(1, min(pl, gl))
    if mode == "strict":
        return ov / max(1, max(pl, gl))
    if mode == "iou":
        return ov / max(1, pl + gl - ov)
    raise ValueError(f"Unknown mode: {mode}")


def eval_regions(
    gt_regions: list[tuple[str, int, int]],
    pred_regions: list[tuple[str, int, int]],
    mode: str,
    min_overlap: float = 0.5,
) -> dict[str, float | int]:
    gt_by_contig: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    pred_by_contig: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for region in gt_regions:
        gt_by_contig[region[0]].append(region)
    for region in pred_regions:
        pred_by_contig[region[0]].append(region)

    tp_r = 0
    for gt in gt_regions:
        best = 0.0
        for pred in pred_by_contig.get(gt[0], []):
            best = max(best, score_overlap(pred, gt, mode))
        if best >= min_overlap:
            tp_r += 1

    tp_p = 0
    for pred in pred_regions:
        best = 0.0
        for gt in gt_by_contig.get(pred[0], []):
            best = max(best, score_overlap(pred, gt, mode))
        if best >= min_overlap:
            tp_p += 1

    n_gt = len(gt_regions)
    n_pred = len(pred_regions)
    recall = tp_r / n_gt if n_gt else 0.0
    precision = tp_p / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp_recall": tp_r,
        "tp_precision": tp_p,
        "n_gt": n_gt,
        "n_pred": n_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/syh/benchmark/nine_genome")
    parser.add_argument("--results-dir", default=None, help="Defaults to <base>/results/npmaster")
    parser.add_argument(
        "--genome-ids",
        default=None,
        help="Optional comma-separated subset of genome IDs (without .fna) to score.",
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-gap", type=int, required=True)
    parser.add_argument("--length-cap", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = Path(args.base)
    results_dir = Path(args.results_dir) if args.results_dir else base / "results" / "npmaster"
    gt = load_ground_truth(base / "ground_truth.json")
    genome_ids = select_genome_ids(args.genome_ids)

    per_genome: dict[str, dict[str, dict[str, float | int]]] = {}
    aggregate = {
        "loose": {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0},
        "strict": {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0},
        "iou": {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0},
    }

    for genome_id in genome_ids:
        csv_path = results_dir / f"{genome_id}_predictions.csv"
        pred_regions = load_regions_from_csv(
            csv_path,
            args.threshold,
            args.max_gap,
            args.length_cap,
        )
        genome_metrics = {}
        for mode in ["loose", "strict", "iou"]:
            m = eval_regions(gt.get(genome_id, []), pred_regions, mode)
            genome_metrics[mode] = m
            for key in aggregate[mode]:
                aggregate[mode][key] += int(m[key])
        per_genome[genome_id] = genome_metrics

    summary = {}
    for mode, a in aggregate.items():
        recall = a["tp_recall"] / a["n_gt"] if a["n_gt"] else 0.0
        precision = a["tp_precision"] / a["n_pred"] if a["n_pred"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary[mode] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "tp_recall": a["tp_recall"],
            "tp_precision": a["tp_precision"],
            "n_gt": a["n_gt"],
            "n_pred": a["n_pred"],
        }

    out = {
        "threshold": args.threshold,
        "max_gap": args.max_gap,
        "length_cap": args.length_cap,
        "summary": summary,
        "per_genome": per_genome,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
