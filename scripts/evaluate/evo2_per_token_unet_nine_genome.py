#!/usr/bin/env python3
"""Evaluate per-token region predictions against the 9-genome GT.

Given a regions CSV with columns: genome,contig,start,end,score[,type]
scores it under `bench` (asymmetric 50%) and `strict` modes, aggregated
over the 9 canonical genomes. Reports:
  - recall, precision, f1 (bench, strict)
  - dedup_precision, dedup_f1 (bench)
  - fragmentation stats
  - per-length-bucket recall
  - bootstrapped 95% CI on dedup_f1 (over GT regions)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.nine_genome import (  # noqa: E402
    GENOME_IDS, eval_regions, load_ground_truth,
)


LENGTH_BUCKETS = [
    ("<5kb", 0, 5_000),
    ("5-10kb", 5_000, 10_000),
    ("10-20kb", 10_000, 20_000),
    ("20-50kb", 20_000, 50_000),
    (">=50kb", 50_000, None),
]


def length_bucket(L: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if L >= lo and (hi is None or L < hi):
            return name
    return "?"


def read_regions(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    by_genome: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            g = row.get("genome") or row.get("genome_id") or ""
            by_genome[g].append((row["contig"], int(row["start"]), int(row["end"])))
    return dict(by_genome)


def aggregate(
    gt: dict[str, list[tuple[str, int, int]]],
    pred: dict[str, list[tuple[str, int, int]]],
    genome_ids: list[str],
    modes: tuple[str, ...] = ("bench", "strict"),
) -> dict:
    agg = {
        m: {
            "tp_recall": 0, "tp_precision": 0,
            "n_gt": 0, "n_pred": 0,
            "unique_gt_precision_tp": 0,
            "duplicate_precision_tp": 0,
            "max_preds_per_gt": 0,
        } for m in modes
    }
    for gid in genome_ids:
        for mode in modes:
            m = eval_regions(gt.get(gid, []), pred.get(gid, []), mode)
            for key in ("tp_recall", "tp_precision", "n_gt", "n_pred"):
                agg[mode][key] += int(m[key])
            frag = m["fragmentation"]
            agg[mode]["unique_gt_precision_tp"] += int(frag["unique_gt_precision_tp"])
            agg[mode]["duplicate_precision_tp"] += int(frag["duplicate_precision_tp"])
            agg[mode]["max_preds_per_gt"] = max(
                agg[mode]["max_preds_per_gt"], int(frag["max_preds_per_gt"]),
            )

    summary: dict[str, dict] = {}
    for mode, a in agg.items():
        n_gt = a["n_gt"]; n_pred = a["n_pred"]
        recall = a["tp_recall"] / n_gt if n_gt else 0.0
        precision = a["tp_precision"] / n_pred if n_pred else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        dedup_p = a["unique_gt_precision_tp"] / n_pred if n_pred else 0.0
        dedup_f1 = (
            2 * dedup_p * recall / (dedup_p + recall) if (dedup_p + recall) else 0.0
        )
        summary[mode] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "dedup_precision": dedup_p,
            "dedup_f1": dedup_f1,
            "tp_recall": a["tp_recall"],
            "tp_precision": a["tp_precision"],
            "n_gt": n_gt,
            "n_pred": n_pred,
            "unique_gt_precision_tp": a["unique_gt_precision_tp"],
            "duplicate_precision_tp": a["duplicate_precision_tp"],
            "max_preds_per_gt": a["max_preds_per_gt"],
        }
    return summary


def bucket_recall(
    gt: dict[str, list[tuple[str, int, int]]],
    pred: dict[str, list[tuple[str, int, int]]],
    genome_ids: list[str],
) -> dict[str, str]:
    """Bench-mode bucket recall: a GT is hit if any pred has overlap/gt_len >= 0.5."""
    gt_tot: dict[str, int] = Counter()
    hit: dict[str, int] = Counter()
    for gid in genome_ids:
        gts = gt.get(gid, [])
        preds = pred.get(gid, [])
        for (c, gs, ge) in gts:
            L = ge - gs
            b = length_bucket(L)
            gt_tot[b] += 1
            best = 0
            for (pc, ps, pe) in preds:
                if pc != c:
                    continue
                ov = max(0, min(pe, ge) - max(ps, gs))
                if ov > best:
                    best = ov
            if L > 0 and (best / L) >= 0.5:
                hit[b] += 1
    return {
        b: f"{hit[b]}/{gt_tot[b]} ({hit[b] / max(1, gt_tot[b]):.3f})"
        for b, _, _ in LENGTH_BUCKETS
    }


def bootstrap_dedup_f1_ci(
    gt: dict[str, list[tuple[str, int, int]]],
    pred: dict[str, list[tuple[str, int, int]]],
    genome_ids: list[str],
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for bench dedup_f1 by resampling GT genomes (with replacement).

    Keeps predictions fixed per genome; resamples which genomes are scored.
    Returns (low, high) percentile CI.
    """
    rnd = random.Random(seed)
    vals: list[float] = []
    gids = list(genome_ids)
    for _ in range(n_boot):
        sample = [rnd.choice(gids) for _ in gids]
        s = aggregate(gt, pred, sample, modes=("bench",))
        vals.append(s["bench"]["dedup_f1"])
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return lo, hi


def region_length_stats(pred: dict[str, list[tuple[str, int, int]]]) -> dict:
    lens = [e - s for regs in pred.values() for _, s, e in regs]
    lens.sort()
    if not lens:
        return {"n": 0}
    n = len(lens)
    median = lens[n // 2] if n % 2 else (lens[n // 2 - 1] + lens[n // 2]) // 2
    return {"n": n, "median": int(median), "min": lens[0], "max": lens[-1]}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-csv", required=True, type=Path)
    p.add_argument("--method-name", required=True)
    p.add_argument("--gt-json", type=Path,
                   default=Path("/data/syh/benchmark/nine_genome/ground_truth.json"))
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--dataset", choices=["9g"], default="9g")
    p.add_argument("--n-boot", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    pred = read_regions(args.regions_csv)
    gt = load_ground_truth(args.gt_json)
    genome_ids = list(GENOME_IDS)

    print(f"loaded {sum(len(v) for v in pred.values())} regions across "
          f"{len(pred)} genomes")

    summary = aggregate(gt, pred, genome_ids)
    buckets = bucket_recall(gt, pred, genome_ids)
    ci_lo, ci_hi = bootstrap_dedup_f1_ci(
        gt, pred, genome_ids, n_boot=args.n_boot, seed=0,
    )

    result = {
        "method": args.method_name,
        "source_csv": str(args.regions_csv),
        "nine_genome": {
            "summary": summary,
            "bucket_recall_bench": buckets,
            "dedup_f1_ci95_bench": [ci_lo, ci_hi],
            "length_stats": region_length_stats(pred),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["nine_genome"]["summary"], indent=2))
    print("buckets:", buckets)
    print(f"dedup_f1 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
