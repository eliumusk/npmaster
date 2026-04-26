#!/usr/bin/env python3
"""Evaluate NP-Master 9-genome predictions under a frozen operating point.

Reads precomputed per-window prediction CSVs from the 9-genome benchmark,
merges windows into regions with the v2 hysteresis decoder, then reports
region metrics under four overlap rules:

- `bench`: recall requires overlap / gt_len   >= 0.5
           precision requires overlap / pred_len >= 0.5
           (asymmetric 50% — matches muskliu/run_nine_genome_eval.sh
           and is the claim metric used in operating_points.yaml)
- `loose`: overlap / min(pred_len, gt_len) >= 0.5
- `strict`: overlap / max(pred_len, gt_len) >= 0.5
- `iou`: IoU >= 0.5

Do NOT use the older syh/benchmark evaluate.py (any-overlap precision): it
is known to be too permissive on the precision side.

Operating points are frozen in `configs/operating_points.yaml`. Pass
`--operating-point NAME` to load parameters from there. Individual flags
still work as overrides for ad hoc diagnostics.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
from decoding.postprocess import WindowScore, decode_regions_hysteresis  # noqa: E402

DEFAULT_OPERATING_POINTS = REPO_ROOT / "configs" / "operating_points.yaml"


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


def load_genome_subset(split_path: Path, split_part: str) -> list[str]:
    with split_path.open() as f:
        split_data = json.load(f)
    if split_part not in split_data:
        raise ValueError(
            f"Split part {split_part!r} not found in {split_path}. "
            f"Available keys: {sorted(split_data)}"
        )
    subset = split_data[split_part]
    if not isinstance(subset, list) or not all(isinstance(genome_id, str) for genome_id in subset):
        raise ValueError(f"Split part {split_part!r} in {split_path} must be a list of genome IDs.")
    return subset


def load_operating_point(name: str, path: Path) -> dict:
    with path.open() as f:
        book = yaml.safe_load(f)
    if name not in book:
        raise SystemExit(
            f"Operating point {name!r} not found in {path}. Available: {sorted(book)}"
        )
    return book[name]


def compute_row_score(row: dict, score_family: str | None) -> float:
    """Resolve a per-window score under the requested family.

    Supported families:
      - None / "max_type" / "bgc_score": use the `bgc_score` column as-is
        (this column already stores max over the 7 BGC type probs).
      - "fused": 0.5 * max_type + 0.5 * (1 - is_negative)
                 = 0.5 * bgc_score + 0.5 * (1 - non_bgc_score)
      - "1_minus_isneg": 1 - non_bgc_score
    """
    bgc = float(row["bgc_score"])
    if score_family in (None, "", "max_type", "bgc_score"):
        return bgc
    non_bgc = float(row.get("non_bgc_score", 0.0))
    if score_family == "fused":
        return 0.5 * bgc + 0.5 * (1.0 - non_bgc)
    if score_family == "1_minus_isneg":
        return 1.0 - non_bgc
    raise ValueError(f"Unknown score family: {score_family!r}")


def load_windows_from_csv(csv_path: Path, score_family: str | None = None) -> list[WindowScore]:
    rows: list[WindowScore] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                WindowScore(
                    contig=row["contig"],
                    start=int(row["start"]),
                    end=int(row["end"]),
                    score=compute_row_score(row, score_family),
                )
            )
    return rows


def score_overlap_recall(pred: tuple[str, int, int], gt: tuple[str, int, int], mode: str) -> float:
    """Overlap score seen from the GT side (recall TP rule)."""
    _, ps, pe = pred
    _, gs, ge = gt
    pl = pe - ps
    gl = ge - gs
    ov = overlap_length(ps, pe, gs, ge)
    if ov <= 0:
        return 0.0
    if mode == "bench":
        return ov / max(1, gl)
    if mode == "loose":
        return ov / max(1, min(pl, gl))
    if mode == "strict":
        return ov / max(1, max(pl, gl))
    if mode == "iou":
        return ov / max(1, pl + gl - ov)
    raise ValueError(f"Unknown mode: {mode}")


def score_overlap_precision(pred: tuple[str, int, int], gt: tuple[str, int, int], mode: str) -> float:
    """Overlap score seen from the prediction side (precision TP rule).

    For symmetric modes (loose/strict/iou) this matches score_overlap_recall.
    For `bench` it switches the denominator to pred_len.
    """
    _, ps, pe = pred
    _, gs, ge = gt
    pl = pe - ps
    gl = ge - gs
    ov = overlap_length(ps, pe, gs, ge)
    if ov <= 0:
        return 0.0
    if mode == "bench":
        return ov / max(1, pl)
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
            best = max(best, score_overlap_recall(pred, gt, mode))
        if best >= min_overlap:
            tp_r += 1

    tp_p = 0
    precision_gt_hits: list[int] = []
    for pred in pred_regions:
        best = 0.0
        best_gt_index = None
        for gt_index, gt in enumerate(gt_regions):
            if gt[0] != pred[0]:
                continue
            score = score_overlap_precision(pred, gt, mode)
            if score > best:
                best = score
                best_gt_index = gt_index
        if best >= min_overlap:
            tp_p += 1
            if best_gt_index is not None:
                precision_gt_hits.append(best_gt_index)

    n_gt = len(gt_regions)
    n_pred = len(pred_regions)
    recall = tp_r / n_gt if n_gt else 0.0
    precision = tp_p / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hit_counts = Counter(precision_gt_hits)
    unique_gt_precision_tp = len(hit_counts)
    duplicate_precision_tp = sum(count - 1 for count in hit_counts.values() if count > 1)
    max_preds_per_gt = max(hit_counts.values()) if hit_counts else 0
    dedup_precision = unique_gt_precision_tp / n_pred if n_pred else 0.0
    dedup_f1 = (
        2 * dedup_precision * recall / (dedup_precision + recall)
        if (dedup_precision + recall)
        else 0.0
    )
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp_recall": tp_r,
        "tp_precision": tp_p,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "fragmentation": {
            "unique_gt_precision_tp": unique_gt_precision_tp,
            "duplicate_precision_tp": duplicate_precision_tp,
            "max_preds_per_gt": max_preds_per_gt,
            "pred_per_gt_ratio": n_pred / n_gt if n_gt else 0.0,
            "dedup_precision": dedup_precision,
            "dedup_f1": dedup_f1,
        },
    }


def resolve_params(args: object) -> dict:
    """Merge CLI flags over a yaml operating point. CLI always wins when set."""
    params: dict = {
        "start_threshold": None,
        "extend_threshold": None,
        "max_gap": None,
        "length_cap": None,
        "min_support_windows": 1,
        "valley_threshold": None,
        "valley_min_windows": 1,
        "score": None,
        "stride": None,
        "checkpoint": None,
        "expected_metrics": None,
        "operating_point": None,
    }

    if args.operating_point:
        op = load_operating_point(args.operating_point, Path(args.operating_points_file))
        params["operating_point"] = args.operating_point
        params["start_threshold"] = float(op["start_threshold"])
        params["extend_threshold"] = float(op["extend_threshold"])
        params["max_gap"] = int(op["max_gap"])
        params["length_cap"] = op.get("length_cap")
        params["min_support_windows"] = int(op.get("min_support_windows", 1))
        params["valley_threshold"] = op.get("valley_threshold")
        params["valley_min_windows"] = int(op.get("valley_min_windows", 1))
        params["score"] = op.get("score")
        params["stride"] = op.get("stride")
        params["checkpoint"] = op.get("checkpoint")
        params["expected_metrics"] = op.get("metrics")

    # CLI overrides.
    if args.start_threshold is not None:
        params["start_threshold"] = args.start_threshold
    if args.extend_threshold is not None:
        params["extend_threshold"] = args.extend_threshold
    if args.max_gap is not None:
        params["max_gap"] = args.max_gap
    if args.length_cap is not None:
        params["length_cap"] = args.length_cap
    if args.min_support_windows is not None:
        params["min_support_windows"] = args.min_support_windows
    if args.valley_threshold is not None:
        params["valley_threshold"] = args.valley_threshold
    if args.valley_min_windows is not None:
        params["valley_min_windows"] = args.valley_min_windows
    if getattr(args, "score", None) is not None:
        params["score"] = args.score

    # Legacy single-threshold shorthand.
    if args.threshold is not None:
        if params["start_threshold"] is None:
            params["start_threshold"] = args.threshold
        if params["extend_threshold"] is None:
            params["extend_threshold"] = args.threshold

    missing = [k for k in ("start_threshold", "extend_threshold", "max_gap") if params[k] is None]
    if missing:
        raise SystemExit(
            f"Missing required params: {missing}. Pass --operating-point or supply "
            "--start-threshold/--extend-threshold/--max-gap directly."
        )
    return params


def evaluate_results_dir(
    base: Path,
    results_dir: Path,
    genome_ids: Sequence[str],
    params: dict,
) -> dict:
    gt = load_ground_truth(base / "ground_truth.json")
    modes = ["bench", "loose", "strict", "iou"]
    per_genome: dict[str, dict[str, dict[str, float | int]]] = {}
    aggregate = {
        m: {
            "tp_recall": 0,
            "tp_precision": 0,
            "n_gt": 0,
            "n_pred": 0,
            "unique_gt_precision_tp": 0,
            "duplicate_precision_tp": 0,
            "max_preds_per_gt": 0,
        }
        for m in modes
    }
    pred_lengths: list[int] = []

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
        pred_lengths.extend(max(0, end - start) for _, start, end in pred_regions)
        genome_metrics = {}
        for mode in modes:
            m = eval_regions(gt.get(genome_id, []), pred_regions, mode)
            genome_metrics[mode] = m
            for key in ("tp_recall", "tp_precision", "n_gt", "n_pred"):
                aggregate[mode][key] += int(m[key])
            frag = m["fragmentation"]
            aggregate[mode]["unique_gt_precision_tp"] += int(frag["unique_gt_precision_tp"])
            aggregate[mode]["duplicate_precision_tp"] += int(frag["duplicate_precision_tp"])
            aggregate[mode]["max_preds_per_gt"] = max(
                aggregate[mode]["max_preds_per_gt"],
                int(frag["max_preds_per_gt"]),
            )
        per_genome[genome_id] = genome_metrics

    summary = {}
    for mode, a in aggregate.items():
        recall = a["tp_recall"] / a["n_gt"] if a["n_gt"] else 0.0
        precision = a["tp_precision"] / a["n_pred"] if a["n_pred"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        dedup_precision = a["unique_gt_precision_tp"] / a["n_pred"] if a["n_pred"] else 0.0
        dedup_f1 = (
            2 * dedup_precision * recall / (dedup_precision + recall)
            if (dedup_precision + recall)
            else 0.0
        )
        summary[mode] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "tp_recall": a["tp_recall"],
            "tp_precision": a["tp_precision"],
            "n_gt": a["n_gt"],
            "n_pred": a["n_pred"],
            "fragmentation": {
                "unique_gt_precision_tp": a["unique_gt_precision_tp"],
                "duplicate_precision_tp": a["duplicate_precision_tp"],
                "max_preds_per_gt": a["max_preds_per_gt"],
                "pred_per_gt_ratio": a["n_pred"] / a["n_gt"] if a["n_gt"] else 0.0,
                "dedup_precision": dedup_precision,
                "dedup_f1": dedup_f1,
            },
        }

    if pred_lengths:
        sorted_lengths = sorted(pred_lengths)
        pred_length_stats = {
            "n": len(sorted_lengths),
            "min": int(sorted_lengths[0]),
            "max": int(sorted_lengths[-1]),
            "mean": float(statistics.mean(sorted_lengths)),
            "median": float(statistics.median(sorted_lengths)),
            "p25": float(sorted_lengths[len(sorted_lengths) // 4]),
            "p75": float(sorted_lengths[(3 * len(sorted_lengths)) // 4]),
        }
    else:
        pred_length_stats = {"n": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}

    return {
        "operating_point": params["operating_point"],
        "params": {
            "score": params["score"],
            "stride": params["stride"],
            "checkpoint": params["checkpoint"],
            "start_threshold": params["start_threshold"],
            "extend_threshold": params["extend_threshold"],
            "max_gap": params["max_gap"],
            "length_cap": params["length_cap"],
            "min_support_windows": params["min_support_windows"],
            "valley_threshold": params["valley_threshold"],
            "valley_min_windows": params["valley_min_windows"],
        },
        "expected_metrics": params["expected_metrics"],
        "summary": summary,
        "pred_length_stats": pred_length_stats,
        "per_genome": per_genome,
    }

