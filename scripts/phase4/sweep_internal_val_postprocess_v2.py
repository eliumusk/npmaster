#!/usr/bin/env python3
"""Internal-val sweep for the second-stage postprocessing decoder."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOW_BP = 16384


SCORE_FNS = {
    "max_type": lambda p: p[:, :7].max(axis=1),
    "1_minus_isneg": lambda p: 1.0 - p[:, 7],
    "fused": lambda p: 0.5 * (1.0 - p[:, 7]) + 0.5 * p[:, :7].max(axis=1),
}


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


def overlap_metrics(pred: tuple[int, int], gt: tuple[int, int]) -> dict[str, float]:
    ps, pe = pred
    gs, ge = gt
    ov = overlap_length(ps, pe, gs, ge)
    plen = pe - ps
    glen = ge - gs
    return {
        "loose": ov / max(1, min(plen, glen)),
        "strict": ov / max(1, max(plen, glen)),
        "iou": ov / max(1, plen + glen - ov),
    }


def load_gt(bgc_csv: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    by = defaultdict(list)
    with bgc_csv.open() as f:
        for row in csv.DictReader(f):
            by[(row["genome_id"], row["contig_id"])].append(row)
    return by


def load_cache(cache_path: Path, split: str) -> dict[str, np.ndarray]:
    data = np.load(cache_path, allow_pickle=True)
    return {
        "probs": data[f"{split}__probs"],
        "genome_id": data[f"{split}__genome_id"],
        "contig_id": data[f"{split}__contig_id"],
        "window_start": data[f"{split}__window_start"],
    }


def build_windows(split_data: dict[str, np.ndarray], score_name: str, window_cls) -> dict[str, list[object]]:
    score = SCORE_FNS[score_name](split_data["probs"])
    per_genome = defaultdict(list)
    for i in range(len(score)):
        genome_id = str(split_data["genome_id"][i])
        per_genome[genome_id].append(
            window_cls(
                contig=str(split_data["contig_id"][i]),
                start=int(split_data["window_start"][i]),
                end=int(split_data["window_start"][i]) + WINDOW_BP,
                score=float(score[i]),
            )
        )
    for genome_id in per_genome:
        per_genome[genome_id].sort(key=lambda row: (row.contig, row.start))
    return per_genome


def evaluate_regions(
    regions: list[tuple[str, int, int]],
    gt_by_loc: dict[tuple[str, str], list[dict[str, str]]],
    genome_id: str,
    ov_thr: float = 0.5,
) -> dict[str, object]:
    matched_gt = {"loose": set(), "strict": set(), "iou": set()}
    pred_match = {"loose": 0, "strict": 0, "iou": 0}

    contigs = {region[0] for region in regions}
    for contig in contigs:
        gt_regions = gt_by_loc.get((genome_id, contig), [])
        for gt in gt_regions:
            gt_s = int(gt["start_bp"])
            gt_e = int(gt["end_bp"])
            gt_key = (genome_id, contig, gt_s, gt_e)
            best = {"loose": 0.0, "strict": 0.0, "iou": 0.0}
            for pred_contig, pred_s, pred_e in regions:
                if pred_contig != contig:
                    continue
                metrics = overlap_metrics((pred_s, pred_e), (gt_s, gt_e))
                for key in best:
                    best[key] = max(best[key], metrics[key])
            for key in best:
                if best[key] >= ov_thr:
                    matched_gt[key].add(gt_key)

    for pred_contig, pred_s, pred_e in regions:
        gt_regions = gt_by_loc.get((genome_id, pred_contig), [])
        best = {"loose": 0.0, "strict": 0.0, "iou": 0.0}
        for gt in gt_regions:
            metrics = overlap_metrics((pred_s, pred_e), (int(gt["start_bp"]), int(gt["end_bp"])))
            for key in best:
                best[key] = max(best[key], metrics[key])
        for key in best:
            if best[key] >= ov_thr:
                pred_match[key] += 1

    n_pred = len(regions)
    return {
        "n_regions": n_pred,
        "matched_gt": {key: len(value) for key, value in matched_gt.items()},
        "precision": {
            key: pred_match[key] / n_pred if n_pred else 0.0
            for key in pred_match
        },
    }


def run_config(
    windows_by_genome: dict[str, list[WindowScore]],
    gt_by_loc: dict[tuple[str, str], list[dict[str, str]]],
    cfg: dict[str, int | float | None],
) -> dict[str, object]:
    split_genomes = set(windows_by_genome.keys())
    n_gt = sum(len(v) for (genome_id, _), v in gt_by_loc.items() if genome_id in split_genomes)
    agg = {
        "loose": {"matched_gt": 0, "pred_match": 0},
        "strict": {"matched_gt": 0, "pred_match": 0},
        "iou": {"matched_gt": 0, "pred_match": 0},
    }
    n_regions = 0
    region_lengths = []

    for genome_id, windows in windows_by_genome.items():
        pred_regions = decode_regions_hysteresis(
            windows,
            start_threshold=float(cfg["start_threshold"]),
            extend_threshold=float(cfg["extend_threshold"]),
            max_gap=int(cfg["max_gap"]),
            min_support_windows=int(cfg["min_support_windows"]),
            length_cap=cfg["length_cap"],
            valley_threshold=cfg["valley_threshold"],
            valley_min_windows=int(cfg["valley_min_windows"]),
        )
        ev = evaluate_regions(pred_regions, gt_by_loc, genome_id)
        n_regions += ev["n_regions"]
        region_lengths.extend(end - start for _, start, end in pred_regions)
        for mode in agg:
            agg[mode]["matched_gt"] += int(ev["matched_gt"][mode])
            agg[mode]["pred_match"] += int(round(ev["precision"][mode] * ev["n_regions"]))

    row: dict[str, object] = {
        **cfg,
        "n_regions": n_regions,
        "median_len_kb": float(np.median(region_lengths) / 1000) if region_lengths else 0.0,
        "p95_len_kb": float(np.percentile(region_lengths, 95) / 1000) if region_lengths else 0.0,
    }
    for mode in agg:
        recall = agg[mode]["matched_gt"] / n_gt if n_gt else 0.0
        precision = agg[mode]["pred_match"] / n_regions if n_regions else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        row[f"recall_{mode}"] = recall
        row[f"p_{mode}"] = precision
        row[f"f1_{mode}"] = f1
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--bgc-csv", default="/data/muskliu/np_master/data/bgc_annotations.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-names", default="max_type,fused")
    parser.add_argument("--start-thresholds", default="0.90,0.95,0.98")
    parser.add_argument("--extend-thresholds", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--max-gaps", default="0,8192,16384")
    parser.add_argument("--length-caps", default="30000,50000")
    parser.add_argument("--min-supports", default="1,2,3")
    parser.add_argument("--valley-thresholds", default="none,0.85,0.90")
    parser.add_argument("--valley-min-windows", default="1,2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_data = load_cache(Path(args.cache), args.split)
    gt_by_loc = load_gt(Path(args.bgc_csv))
    score_names = [item.strip() for item in args.score_names.split(",") if item.strip()]

    start_thresholds = [float(x) for x in args.start_thresholds.split(",")]
    extend_thresholds = [float(x) for x in args.extend_thresholds.split(",")]
    max_gaps = [int(x) for x in args.max_gaps.split(",")]
    length_caps = parse_optional_ints(args.length_caps)
    min_supports = [int(x) for x in args.min_supports.split(",")]
    valley_thresholds = parse_optional_floats(args.valley_thresholds)
    valley_min_windows = [int(x) for x in args.valley_min_windows.split(",")]

    all_rows: dict[str, list[dict[str, object]]] = {}
    best: dict[str, dict[str, object]] = {}
    for score_name in score_names:
        windows_by_genome = build_windows(split_data, score_name, WindowScore)
        rows = []
        for start_threshold in start_thresholds:
            for extend_threshold in extend_thresholds:
                if extend_threshold > start_threshold:
                    continue
                for max_gap in max_gaps:
                    for length_cap in length_caps:
                        for min_support_windows in min_supports:
                            for valley_threshold in valley_thresholds:
                                if valley_threshold is not None and valley_threshold <= extend_threshold:
                                    continue
                                for valley_min in valley_min_windows:
                                    cfg = {
                                        "score": score_name,
                                        "start_threshold": start_threshold,
                                        "extend_threshold": extend_threshold,
                                        "max_gap": max_gap,
                                        "length_cap": length_cap,
                                        "min_support_windows": min_support_windows,
                                        "valley_threshold": valley_threshold,
                                        "valley_min_windows": valley_min,
                                    }
                                    rows.append(run_config(windows_by_genome, gt_by_loc, cfg))
        all_rows[score_name] = rows
        best[score_name] = {
            "best_strict": max(rows, key=lambda row: row["f1_strict"]),
            "best_iou": max(rows, key=lambda row: row["f1_iou"]),
            "best_loose": max(rows, key=lambda row: row["f1_loose"]),
        }

    out = {
        "cache": args.cache,
        "split": args.split,
        "score_names": score_names,
        "best": best,
        "sweep": all_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["best"], indent=2))


if __name__ == "__main__":
    main()
