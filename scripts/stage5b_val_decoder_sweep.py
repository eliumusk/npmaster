#!/usr/bin/env python3
"""Stage 5b: internal-val decoder sweep with fragmentation gate.

Given a directory of per-genome val predictions (produced by
``scripts/dump_val_predictions.py``) and the shared internal val GT
(= antiSMASH regions from ``bgc_annotations.csv`` restricted to val-split
genomes), this script sweeps a decoder grid and records, for every
(checkpoint, decoder) cell:

  * primary metrics: ``dedup_f1``, ``bench_precision``, ``bench_recall``
  * fragmentation gate: ``frag_ratio = dedup_precision / bench_precision``
    (≥ threshold → candidate passes; rejects reward-hacking configs)
  * length-bucket recall + precision for ``<5kb / 5-10kb / 10-20kb /
    20-50kb / >=50kb``
  * secondary: bench_f1, dedup_precision, n_pred, n_gt, max_preds_per_gt

Operating-point picker: for each checkpoint, pick the decoder cell with
the highest ``dedup_f1`` that clears ``frag_ratio >= --gate``; ties
broken by ``bench_precision``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_nine_genome import load_windows_from_csv, eval_regions  # noqa: E402
from postprocess_v2 import decode_regions_hysteresis  # noqa: E402


BUCKETS = ("<5kb", "5-10kb", "10-20kb", "20-50kb", ">=50kb")


def assign_length_bucket(length_bp: int) -> str:
    if length_bp < 5000:
        return "<5kb"
    if length_bp < 10000:
        return "5-10kb"
    if length_bp < 20000:
        return "10-20kb"
    if length_bp < 50000:
        return "20-50kb"
    return ">=50kb"


def _overlap(a: tuple[str, int, int], b: tuple[str, int, int]) -> int:
    if a[0] != b[0]:
        return 0
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def compute_length_buckets(
    gt_regions: list[tuple[str, int, int]],
    pred_regions: list[tuple[str, int, int]],
    recall_frac: float = 0.5,
    precision_frac: float = 0.5,
) -> dict[str, dict[str, int | float]]:
    """Length-bucket recall + precision using the same asymmetric rule
    as bench mode."""
    counts: dict[str, dict[str, float]] = {
        b: {"n_gt": 0, "recall_tp": 0, "n_pred": 0, "precision_tp": 0}
        for b in BUCKETS
    }

    for gt in gt_regions:
        b = assign_length_bucket(gt[2] - gt[1])
        counts[b]["n_gt"] += 1
        gt_len = max(1, gt[2] - gt[1])
        if any(_overlap(gt, p) / gt_len >= recall_frac for p in pred_regions):
            counts[b]["recall_tp"] += 1
    for p in pred_regions:
        b = assign_length_bucket(p[2] - p[1])
        counts[b]["n_pred"] += 1
        p_len = max(1, p[2] - p[1])
        if any(_overlap(p, g) / p_len >= precision_frac for g in gt_regions):
            counts[b]["precision_tp"] += 1

    for b in BUCKETS:
        row = counts[b]
        row["recall"] = (
            row["recall_tp"] / row["n_gt"] if row["n_gt"] else 0.0
        )
        row["precision"] = (
            row["precision_tp"] / row["n_pred"] if row["n_pred"] else 0.0
        )
    return counts


def load_internal_val_gt(
    bgc_csv: Path, val_genomes: set[str]
) -> dict[str, list[tuple[str, int, int]]]:
    df = pd.read_csv(bgc_csv)
    df = df[df["genome_id"].isin(val_genomes)].copy()
    out: dict[str, list[tuple[str, int, int]]] = {}
    for _, r in df.iterrows():
        out.setdefault(str(r["genome_id"]), []).append(
            (str(r["contig_id"]), int(r["start_bp"]), int(r["end_bp"]))
        )
    return out


def collect_val_genomes(meta_path: Path, split: str = "val") -> set[str]:
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    splits = np.asarray(meta["split"])
    gids = np.asarray(meta["genome_id"])
    return set(str(g) for g in np.unique(gids[splits == split]))


def decode_and_eval_one(
    val_pred_dir: Path,
    genome_ids: list[str],
    gt_by_genome: dict[str, list[tuple[str, int, int]]],
    params: dict,
) -> dict:
    """Decode → aggregate bench + dedup + length buckets across val genomes."""
    agg = {
        "tp_recall": 0,
        "tp_precision": 0,
        "n_gt": 0,
        "n_pred": 0,
        "unique_gt_precision_tp": 0,
        "duplicate_precision_tp": 0,
        "max_preds_per_gt": 0,
    }
    bucket_agg = {
        b: {"n_gt": 0, "recall_tp": 0, "n_pred": 0, "precision_tp": 0}
        for b in BUCKETS
    }

    for gid in genome_ids:
        csv_path = val_pred_dir / f"{gid}_predictions.csv"
        if not csv_path.is_file():
            continue
        windows = load_windows_from_csv(csv_path, score_family=params["score"])
        preds = decode_regions_hysteresis(
            windows,
            start_threshold=params["start_threshold"],
            extend_threshold=params["extend_threshold"],
            max_gap=params.get("max_gap", 0),
            min_support_windows=params.get("min_support_windows", 1),
            length_cap=params.get("length_cap"),
            valley_threshold=params.get("valley_threshold"),
            valley_min_windows=params.get("valley_min_windows", 1),
        )
        gts = gt_by_genome.get(gid, [])
        m = eval_regions(gts, preds, mode="bench")
        for k in ("tp_recall", "tp_precision", "n_gt", "n_pred"):
            agg[k] += int(m[k])
        frag = m["fragmentation"]
        agg["unique_gt_precision_tp"] += int(frag["unique_gt_precision_tp"])
        agg["duplicate_precision_tp"] += int(frag["duplicate_precision_tp"])
        agg["max_preds_per_gt"] = max(
            agg["max_preds_per_gt"], int(frag["max_preds_per_gt"])
        )
        buckets = compute_length_buckets(gts, preds)
        for b in BUCKETS:
            for k in ("n_gt", "recall_tp", "n_pred", "precision_tp"):
                bucket_agg[b][k] += buckets[b][k]

    recall = agg["tp_recall"] / agg["n_gt"] if agg["n_gt"] else 0.0
    precision = (
        agg["tp_precision"] / agg["n_pred"] if agg["n_pred"] else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    dedup_precision = (
        agg["unique_gt_precision_tp"] / agg["n_pred"] if agg["n_pred"] else 0.0
    )
    dedup_f1 = (
        2 * dedup_precision * recall / (dedup_precision + recall)
        if (dedup_precision + recall)
        else 0.0
    )
    frag_ratio = (
        dedup_precision / precision if precision > 0 else 0.0
    )

    result = {
        "bench_recall": recall,
        "bench_precision": precision,
        "bench_f1": f1,
        "dedup_precision": dedup_precision,
        "dedup_f1": dedup_f1,
        "frag_ratio": frag_ratio,
        "n_pred": agg["n_pred"],
        "n_gt": agg["n_gt"],
        "max_preds_per_gt": agg["max_preds_per_gt"],
        "duplicate_precision_tp": agg["duplicate_precision_tp"],
        "unique_gt_precision_tp": agg["unique_gt_precision_tp"],
    }
    for b in BUCKETS:
        row = bucket_agg[b]
        result[f"bucket_{b}_n_gt"] = row["n_gt"]
        result[f"bucket_{b}_n_pred"] = row["n_pred"]
        result[f"bucket_{b}_recall"] = (
            row["recall_tp"] / row["n_gt"] if row["n_gt"] else 0.0
        )
        result[f"bucket_{b}_precision"] = (
            row["precision_tp"] / row["n_pred"] if row["n_pred"] else 0.0
        )
    return result


DEFAULT_GRID = {
    "score": ["fused"],
    "start_threshold": [0.90, 0.95, 0.97, 0.99],
    "extend_threshold": [0.85, 0.90, 0.95],
    "max_gap": [0],
    "min_support_windows": [1, 2, 3, 5, 8, 10, 15],
    "length_cap": [None, 60000, 100000],
    "valley_threshold": [None],
    "valley_min_windows": [1],
}


def iter_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    out = []
    for combo in product(*values):
        row = dict(zip(keys, combo))
        # Extend threshold cannot exceed start threshold.
        if row["extend_threshold"] > row["start_threshold"]:
            continue
        out.append(row)
    return out


def run_sweep_for_checkpoint(
    val_pred_dir: Path,
    genome_ids: list[str],
    gt_by_genome: dict,
    grid: list[dict],
) -> list[dict]:
    rows = []
    for params in grid:
        metrics = decode_and_eval_one(
            val_pred_dir, genome_ids, gt_by_genome, params
        )
        row = {**params, **metrics}
        rows.append(row)
    return rows


def pick_operating_point(rows: list[dict], gate: float) -> dict | None:
    eligible = [r for r in rows if r["frag_ratio"] >= gate]
    if not eligible:
        return None
    eligible.sort(
        key=lambda r: (-r["dedup_f1"], -r["bench_precision"], -r["bench_recall"])
    )
    return eligible[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-pred-root",
        type=Path,
        required=True,
        help="Directory containing per-checkpoint val-prediction subdirs. "
        "Each subdir is named val_predictions_<tag>/ and holds "
        "{gid}_predictions.csv.",
    )
    parser.add_argument(
        "--bgc-csv",
        type=Path,
        default=Path("/data/muskliu/npmaster/data/bgc_annotations.csv"),
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path(
            "/data/muskliu/npmaster/data/phase3/route_a/meta_arrays.pkl"
        ),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--gate",
        type=float,
        default=0.75,
        help="Minimum dedup_precision / bench_precision ratio (the "
        "fragmentation-consistency gate).",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=REPO_ROOT / "experiments" / "stage5b_val_decoder_sweep",
    )
    parser.add_argument(
        "--only-checkpoint-tags",
        nargs="*",
        default=None,
        help="Limit sweep to these checkpoint tags (substring match on "
        "val_predictions_<tag>).",
    )
    args = parser.parse_args(argv)

    print("[sweep] collecting val-split genomes...", flush=True)
    val_genomes = collect_val_genomes(args.meta, args.split)
    print(f"[sweep] val genomes = {len(val_genomes)}", flush=True)

    print("[sweep] loading internal val GT (antiSMASH regions)...", flush=True)
    gt_by_genome = load_internal_val_gt(args.bgc_csv, val_genomes)
    print(
        f"[sweep] val GT: {len(gt_by_genome)} genomes with "
        f"{sum(len(v) for v in gt_by_genome.values())} BGCs",
        flush=True,
    )

    grid = iter_grid(DEFAULT_GRID)
    print(f"[sweep] decoder grid size = {len(grid)}", flush=True)

    val_pred_dirs = sorted(args.val_pred_root.glob("val_predictions_*"))
    # Also accept nested experiment dirs with val_predictions_*
    if not val_pred_dirs:
        val_pred_dirs = sorted(
            args.val_pred_root.glob("*/artifacts/val_predictions_*")
        )
    if args.only_checkpoint_tags:
        keep = args.only_checkpoint_tags
        val_pred_dirs = [
            d for d in val_pred_dirs if any(k in d.name for k in keep)
        ]
    if not val_pred_dirs:
        print(
            f"[sweep] no val_predictions_* dirs found under "
            f"{args.val_pred_root}",
            file=sys.stderr,
        )
        return 1
    print(f"[sweep] checkpoints to sweep: {len(val_pred_dirs)}", flush=True)

    all_rows = []
    selections = []
    genome_ids = sorted(gt_by_genome.keys())

    for i, pred_dir in enumerate(val_pred_dirs, start=1):
        tag = pred_dir.name.replace("val_predictions_", "")
        print(
            f"[sweep] ({i}/{len(val_pred_dirs)}) sweeping tag={tag}",
            flush=True,
        )
        rows = run_sweep_for_checkpoint(
            pred_dir, genome_ids, gt_by_genome, grid
        )
        for r in rows:
            r["tag"] = tag
            all_rows.append(r)
        best = pick_operating_point(rows, args.gate)
        if best is not None:
            selected = dict(best)
            selected["tag"] = tag
            selections.append(selected)
            print(
                f"[sweep]   selected: start={best['start_threshold']} "
                f"ms={best['min_support_windows']} "
                f"cap={best['length_cap']} | "
                f"dedup_f1={best['dedup_f1']:.4f} "
                f"bench_P={best['bench_precision']:.4f} "
                f"bench_R={best['bench_recall']:.4f} "
                f"frag_ratio={best['frag_ratio']:.3f}",
                flush=True,
            )
        else:
            print(
                f"[sweep]   NO cell passes the gate={args.gate}",
                flush=True,
            )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_path = args.out_prefix.parent / (args.out_prefix.name + "_all_cells.csv")
    sel_path = args.out_prefix.parent / (args.out_prefix.name + "_selected.csv")
    pd.DataFrame(all_rows).to_csv(all_path, index=False)
    pd.DataFrame(selections).to_csv(sel_path, index=False)
    print(f"[sweep] wrote {all_path} ({len(all_rows)} rows)", flush=True)
    print(f"[sweep] wrote {sel_path} ({len(selections)} winners)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
