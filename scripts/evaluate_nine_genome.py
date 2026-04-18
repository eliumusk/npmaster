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

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from postprocess_v2 import WindowScore, decode_regions_hysteresis  # noqa: E402

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
    for pred in pred_regions:
        best = 0.0
        for gt in gt_by_contig.get(pred[0], []):
            best = max(best, score_overlap_precision(pred, gt, mode))
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


def resolve_params(args: argparse.Namespace) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/syh/benchmark/nine_genome")
    parser.add_argument("--results-dir", default=None, help="Defaults to <base>/results/npmaster")
    parser.add_argument(
        "--genome-ids",
        default=None,
        help="Optional comma-separated subset of genome IDs (without .fna) to score.",
    )
    parser.add_argument(
        "--operating-point",
        default=None,
        help="Name of a frozen operating point in operating_points.yaml "
        "(e.g. val_frozen_fused, tuned_v2).",
    )
    parser.add_argument(
        "--operating-points-file",
        default=str(DEFAULT_OPERATING_POINTS),
        help="Path to operating_points.yaml (defaults to configs/operating_points.yaml).",
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Legacy single-threshold shortcut (sets both start and extend).")
    parser.add_argument("--start-threshold", type=float, default=None)
    parser.add_argument("--extend-threshold", type=float, default=None)
    parser.add_argument("--max-gap", type=int, default=None)
    parser.add_argument("--length-cap", type=int, default=None)
    parser.add_argument("--min-support-windows", type=int, default=None)
    parser.add_argument("--valley-threshold", type=float, default=None)
    parser.add_argument("--valley-min-windows", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    params = resolve_params(args)

    base = Path(args.base)
    results_dir = Path(args.results_dir) if args.results_dir else base / "results" / "npmaster"
    gt = load_ground_truth(base / "ground_truth.json")
    genome_ids = select_genome_ids(args.genome_ids)

    modes = ["bench", "loose", "strict", "iou"]
    per_genome: dict[str, dict[str, dict[str, float | int]]] = {}
    aggregate = {
        m: {"tp_recall": 0, "tp_precision": 0, "n_gt": 0, "n_pred": 0} for m in modes
    }

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
        genome_metrics = {}
        for mode in modes:
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
