#!/usr/bin/env python3
"""Diagnose score-family behavior on internal val and 9-genome windows."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


WINDOW_BP = 16384
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
THRESHOLDS = [0.50, 0.90, 0.95, 0.98]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "nine_genome", "both"], default="both")
    parser.add_argument(
        "--cache",
        default="/data/muskliu/np_master/data/phase4/postprocess/cached_preds_val_s0.npz",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--ignore-ambiguous", action="store_true", default=True)
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--base", default="/data/syh/benchmark/nine_genome")
    parser.add_argument(
        "--results-dir",
        default="/tmp/npmaster_stride8192_s0_merged",
        help="Directory containing 9-genome per-window CSVs.",
    )
    parser.add_argument(
        "--positive-overlap-bp",
        type=int,
        default=1,
        help="Minimum bp overlap with GT to label a window positive on 9-genome.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def overlap_length(s1: int, e1: int, s2: int, e2: int) -> int:
    return max(0, min(e1, e2) - max(s1, s2))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return 0.0
    ranks = average_ranks(scores.astype(np.float64))
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = int(labels.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def quantile_summary(scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0:
        return {f"q{int(q * 100):02d}": 0.0 for q in QUANTILES}
    quantiles = np.quantile(scores, QUANTILES)
    return {
        f"q{int(q * 100):02d}": float(v)
        for q, v in zip(QUANTILES, quantiles.tolist())
    }


def threshold_counts(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    pos_mask = labels == 1
    neg_mask = labels == 0
    for thr in THRESHOLDS:
        pos_hits = int(np.sum(scores[pos_mask] >= thr))
        neg_hits = int(np.sum(scores[neg_mask] >= thr))
        out[str(thr)] = {
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "pos_rate": pos_hits / int(np.sum(pos_mask)) if np.any(pos_mask) else 0.0,
            "neg_rate": neg_hits / int(np.sum(neg_mask)) if np.any(neg_mask) else 0.0,
        }
    return out


def build_records_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    records: list[dict[str, object]],
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    labels = labels.astype(np.int64)
    negative_indices = np.where(labels == 0)[0]
    positive_indices = np.where(labels == 1)[0]
    fp_order = negative_indices[np.argsort(-scores[negative_indices], kind="mergesort")]
    fn_order = positive_indices[np.argsort(scores[positive_indices], kind="mergesort")]
    return {
        "top_high_score_negatives": [
            records[int(i)] | {"score": float(scores[int(i)])}
            for i in fp_order[:top_k]
        ],
        "top_low_score_positives": [
            records[int(i)] | {"score": float(scores[int(i)])}
            for i in fn_order[:top_k]
        ],
    }


def evaluate_dataset(
    labels: np.ndarray,
    score_map: dict[str, np.ndarray],
    records: list[dict[str, object]],
    top_k: int,
) -> dict[str, object]:
    labels = labels.astype(np.int64)
    pos_mask = labels == 1
    neg_mask = labels == 0
    out: dict[str, object] = {
        "n_total": int(len(labels)),
        "n_positive": int(np.sum(pos_mask)),
        "n_negative": int(np.sum(neg_mask)),
    }
    for score_name, scores in score_map.items():
        out[score_name] = {
            "auroc": compute_auroc(labels, scores),
            "auprc": compute_average_precision(labels, scores),
            "positives": quantile_summary(scores[pos_mask]),
            "negatives": quantile_summary(scores[neg_mask]),
            "threshold_counts": threshold_counts(labels, scores),
            "tails": build_records_summary(labels, scores, records, top_k),
        }
    return out


def load_val_dataset(cache_path: Path, split: str, ignore_ambiguous: bool) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
    data = np.load(cache_path, allow_pickle=True)
    probs = data[f"{split}__probs"]
    genome_id = data[f"{split}__genome_id"].astype(str)
    contig_id = data[f"{split}__contig_id"].astype(str)
    window_start = data[f"{split}__window_start"].astype(np.int64)
    label = data[f"{split}__label"].astype(str)
    is_bgc = data[f"{split}__is_bgc"].astype(np.int64)

    mask = np.ones(len(probs), dtype=bool)
    if ignore_ambiguous:
        mask &= label != "ambiguous"

    probs = probs[mask]
    genome_id = genome_id[mask]
    contig_id = contig_id[mask]
    window_start = window_start[mask]
    label = label[mask]
    is_bgc = is_bgc[mask]

    score_map = {
        "max_type": probs[:, :7].max(axis=1),
        "1_minus_isneg": 1.0 - probs[:, 7],
        "fused": 0.5 * probs[:, :7].max(axis=1) + 0.5 * (1.0 - probs[:, 7]),
    }
    records = [
        {
            "genome_id": genome_id[i],
            "contig_id": contig_id[i],
            "start": int(window_start[i]),
            "end": int(window_start[i] + WINDOW_BP),
            "label": label[i],
            "is_bgc": int(is_bgc[i]),
        }
        for i in range(len(probs))
    ]
    return is_bgc, score_map, records


def load_ground_truth(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    with path.open() as f:
        data = json.load(f)
    return {
        genome_id: [(contig, int(start), int(end)) for contig, start, end in regions]
        for genome_id, regions in data["ground_truth"].items()
    }


def best_gt_overlap(
    contig: str,
    start: int,
    end: int,
    gt_regions: list[tuple[str, int, int]],
) -> tuple[int, float]:
    best_bp = 0
    best_strict = 0.0
    pred_len = max(1, end - start)
    for gt_contig, gt_start, gt_end in gt_regions:
        if gt_contig != contig:
            continue
        ov = overlap_length(start, end, gt_start, gt_end)
        if ov <= 0:
            continue
        best_bp = max(best_bp, ov)
        gt_len = max(1, gt_end - gt_start)
        best_strict = max(best_strict, ov / max(pred_len, gt_len))
    return best_bp, best_strict


def load_nine_genome_dataset(
    base: Path,
    results_dir: Path,
    positive_overlap_bp: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
    gt = load_ground_truth(base / "ground_truth.json")
    labels: list[int] = []
    max_type: list[float] = []
    one_minus_isneg: list[float] = []
    fused: list[float] = []
    records: list[dict[str, object]] = []

    for genome_id in GENOME_IDS:
        csv_path = results_dir / f"{genome_id}_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing predictions for {genome_id}: {csv_path}")
        gt_regions = gt.get(genome_id, [])
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = int(row["start"])
                end = int(row["end"])
                ov_bp, ov_strict = best_gt_overlap(
                    row["contig"],
                    start,
                    end,
                    gt_regions,
                )
                label = 1 if ov_bp >= positive_overlap_bp else 0
                bgc_score = float(row["bgc_score"])
                non_bgc_score = float(row["non_bgc_score"])
                labels.append(label)
                max_type.append(bgc_score)
                one_minus_isneg.append(1.0 - non_bgc_score)
                fused.append(0.5 * bgc_score + 0.5 * (1.0 - non_bgc_score))
                records.append(
                    {
                        "genome_id": genome_id,
                        "contig_id": row["contig"],
                        "start": start,
                        "end": end,
                        "label": label,
                        "best_overlap_bp": ov_bp,
                        "best_overlap_strict": ov_strict,
                    }
                )

    return (
        np.asarray(labels, dtype=np.int64),
        {
            "max_type": np.asarray(max_type, dtype=np.float64),
            "1_minus_isneg": np.asarray(one_minus_isneg, dtype=np.float64),
            "fused": np.asarray(fused, dtype=np.float64),
        },
        records,
    )


def main() -> None:
    args = parse_args()
    ignore_ambiguous = args.ignore_ambiguous and not args.include_ambiguous

    out: dict[str, object] = {
        "config": {
            "mode": args.mode,
            "split": args.split,
            "ignore_ambiguous": ignore_ambiguous,
            "positive_overlap_bp": args.positive_overlap_bp,
            "top_k": args.top_k,
        }
    }

    if args.mode in {"val", "both"}:
        labels, score_map, records = load_val_dataset(
            Path(args.cache),
            args.split,
            ignore_ambiguous=ignore_ambiguous,
        )
        out["internal_val"] = evaluate_dataset(labels, score_map, records, args.top_k)

    if args.mode in {"nine_genome", "both"}:
        labels, score_map, records = load_nine_genome_dataset(
            Path(args.base),
            Path(args.results_dir),
            positive_overlap_bp=args.positive_overlap_bp,
        )
        out["nine_genome"] = evaluate_dataset(labels, score_map, records, args.top_k)

    payload = json.dumps(out, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
