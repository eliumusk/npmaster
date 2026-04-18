#!/usr/bin/env python3
"""Formal 9-genome sweep for the second-stage postprocessing decoder."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class WindowScore:
    contig: str
    start: int
    end: int
    score: float


def split_regions_by_length_cap(
    regions: list[tuple[str, list[WindowScore]]],
    length_cap: int | None,
) -> list[tuple[str, int, int]]:
    if not length_cap:
        return [
            (contig, windows[0].start, windows[-1].end)
            for contig, windows in regions
            if windows
        ]

    out: list[tuple[str, int, int]] = []
    for contig, windows in regions:
        if not windows:
            continue
        current_chunk = [windows[0]]
        for window in windows[1:]:
            chunk_start = current_chunk[0].start
            proposed_end = window.end
            if proposed_end - chunk_start > length_cap:
                out.append((contig, current_chunk[0].start, current_chunk[-1].end))
                current_chunk = [window]
            else:
                current_chunk.append(window)
        if current_chunk:
            out.append((contig, current_chunk[0].start, current_chunk[-1].end))
    return out


def split_windows_on_valleys(
    windows: list[WindowScore],
    valley_threshold: float | None,
    valley_min_windows: int,
) -> list[list[WindowScore]]:
    if not windows:
        return []
    if valley_threshold is None:
        return [windows]

    segments: list[list[WindowScore]] = []
    current: list[WindowScore] = []
    index = 0
    while index < len(windows):
        if windows[index].score >= valley_threshold:
            current.append(windows[index])
            index += 1
            continue

        run_end = index
        while run_end < len(windows) and windows[run_end].score < valley_threshold:
            run_end += 1
        valley_run = windows[index:run_end]
        if len(valley_run) >= valley_min_windows:
            if current:
                segments.append(current)
                current = []
        else:
            current.extend(valley_run)
        index = run_end

    if current:
        segments.append(current)
    return segments


def decode_regions_hysteresis(
    windows: list[WindowScore],
    start_threshold: float,
    extend_threshold: float,
    max_gap: int,
    min_support_windows: int = 1,
    length_cap: int | None = None,
    valley_threshold: float | None = None,
    valley_min_windows: int = 1,
) -> list[tuple[str, int, int]]:
    if extend_threshold > start_threshold:
        raise ValueError("extend_threshold must be <= start_threshold")

    windows = sorted(windows, key=lambda row: (row.contig, row.start))
    seeded_regions: list[tuple[str, list[WindowScore]]] = []
    current: list[WindowScore] = []
    seeded = False

    for window in windows:
        if not current:
            if window.score >= start_threshold:
                current = [window]
                seeded = True
            continue

        last = current[-1]
        contiguous = (
            window.contig == last.contig and
            window.start <= last.end + max_gap
        )
        if not contiguous:
            if seeded and len(current) >= min_support_windows:
                seeded_regions.append((current[0].contig, current))
            current = [window] if window.score >= start_threshold else []
            seeded = bool(current)
            continue

        if window.score >= extend_threshold:
            current.append(window)
            if window.score >= start_threshold:
                seeded = True
            continue

        if seeded and len(current) >= min_support_windows:
            seeded_regions.append((current[0].contig, current))
        current = [window] if window.score >= start_threshold else []
        seeded = bool(current)

    if current and seeded and len(current) >= min_support_windows:
        seeded_regions.append((current[0].contig, current))

    split_regions: list[tuple[str, list[WindowScore]]] = []
    for contig, region_windows in seeded_regions:
        for segment in split_windows_on_valleys(
            region_windows,
            valley_threshold=valley_threshold,
            valley_min_windows=valley_min_windows,
        ):
            if len(segment) >= min_support_windows:
                split_regions.append((contig, segment))

    return split_regions_by_length_cap(split_regions, length_cap)


def parse_optional_ints(raw: str) -> list[int | None]:
    out: list[int | None] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item or item == "none":
            out.append(None)
        else:
            out.append(int(item))
    return out


def parse_optional_floats(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item or item == "none":
            out.append(None)
        else:
            out.append(float(item))
    return out


def overlap_length(s1: int, e1: int, s2: int, e2: int) -> int:
    return max(0, min(e1, e2) - max(s1, s2))


def load_ground_truth(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    with path.open() as f:
        data = json.load(f)
    return {
        genome_id: [(contig, int(start), int(end)) for contig, start, end in regions]
        for genome_id, regions in data["ground_truth"].items()
    }


def load_window_scores(results_dir: Path) -> dict[str, list[WindowScore]]:
    out: dict[str, list[WindowScore]] = {}
    for genome_id in GENOME_IDS:
        rows: list[WindowScore] = []
        with (results_dir / f"{genome_id}_predictions.csv").open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    WindowScore(
                        contig=row["contig"],
                        start=int(row["start"]),
                        end=int(row["end"]),
                        score=float(row["bgc_score"]),
                    )
                )
        out[genome_id] = rows
    return out


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
    raise ValueError(mode)


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
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--start-thresholds", default="0.90,0.95,0.98")
    parser.add_argument("--extend-thresholds", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--max-gaps", default="0,8192,16384")
    parser.add_argument("--length-caps", default="30000,50000")
    parser.add_argument("--min-supports", default="1,2,3")
    parser.add_argument("--valley-thresholds", default="none,0.85,0.90")
    parser.add_argument("--valley-min-windows", default="1,2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    results_dir = Path(args.results_dir)
    gt = load_ground_truth(base / "ground_truth.json")
    window_scores = load_window_scores(results_dir)

    start_thresholds = [float(x) for x in args.start_thresholds.split(",")]
    extend_thresholds = [float(x) for x in args.extend_thresholds.split(",")]
    max_gaps = [int(x) for x in args.max_gaps.split(",")]
    length_caps = parse_optional_ints(args.length_caps)
    min_supports = [int(x) for x in args.min_supports.split(",")]
    valley_thresholds = parse_optional_floats(args.valley_thresholds)
    valley_min_windows = [int(x) for x in args.valley_min_windows.split(",")]

    rows: list[dict[str, float | int | None]] = []
    for start_threshold in start_thresholds:
        for extend_threshold in extend_thresholds:
            if extend_threshold > start_threshold:
                continue
            for max_gap in max_gaps:
                for length_cap in length_caps:
                    for min_support in min_supports:
                        for valley_threshold in valley_thresholds:
                            if valley_threshold is not None and valley_threshold <= extend_threshold:
                                continue
                            for valley_min in valley_min_windows:
                                aggregate = {
                                    "loose": {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0},
                                    "strict": {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0},
                                    "iou": {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0},
                                }
                                for genome_id in GENOME_IDS:
                                    pred_regions = decode_regions_hysteresis(
                                        window_scores[genome_id],
                                        start_threshold=start_threshold,
                                        extend_threshold=extend_threshold,
                                        max_gap=max_gap,
                                        min_support_windows=min_support,
                                        length_cap=length_cap,
                                        valley_threshold=valley_threshold,
                                        valley_min_windows=valley_min,
                                    )
                                    genome_gt = gt.get(genome_id, [])
                                    for mode in ["loose", "strict", "iou"]:
                                        m = eval_regions(genome_gt, pred_regions, mode)
                                        for key in aggregate[mode]:
                                            aggregate[mode][key] += int(m[key])

                                row: dict[str, float | int | None] = {
                                    "start_threshold": start_threshold,
                                    "extend_threshold": extend_threshold,
                                    "max_gap": max_gap,
                                    "length_cap": length_cap,
                                    "min_support_windows": min_support,
                                    "valley_threshold": valley_threshold,
                                    "valley_min_windows": valley_min,
                                }
                                for mode, agg in aggregate.items():
                                    recall = agg["tp_recall"] / agg["n_gt"] if agg["n_gt"] else 0.0
                                    precision = agg["tp_precision"] / agg["n_pred"] if agg["n_pred"] else 0.0
                                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                                    row[f"recall_{mode}"] = recall
                                    row[f"precision_{mode}"] = precision
                                    row[f"f1_{mode}"] = f1
                                    row["n_pred"] = agg["n_pred"]
                                rows.append(row)

    rows.sort(key=lambda row: (row["f1_strict"], row["f1_iou"]), reverse=True)
    out = {
        "results_dir": str(results_dir),
        "n_rows": len(rows),
        "best_strict": rows[0],
        "best_iou": max(rows, key=lambda row: row["f1_iou"]),
        "best_loose": max(rows, key=lambda row: row["f1_loose"]),
        "top10_strict": rows[:10],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
