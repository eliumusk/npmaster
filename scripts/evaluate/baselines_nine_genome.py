#!/usr/bin/env python3
"""Run the full 9-genome comparison table locally.

Scores each configured tool + a list of NP-Master operating points under the
same region-level ``bench`` metric as ``evaluate_nine_genome.py`` and writes a
single JSON + markdown table.

Baseline inputs stay external (``/data/syh/benchmark/nine_genome/...``) and are
read-only here; all parsing / aggregation logic lives under this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.nine_genome import (  # noqa: E402
    DEFAULT_OPERATING_POINTS,
    GENOME_IDS,
    eval_regions,
    load_ground_truth,
    load_operating_point,
    load_windows_from_csv,
)
from decoding.postprocess import decode_regions_hysteresis  # noqa: E402

from baselines.parse_antismash import parse_antismash_dir  # noqa: E402
from baselines.parse_bgc_prophet import parse_bgc_prophet_file  # noqa: E402
from baselines.parse_deepbgc import parse_deepbgc_dir  # noqa: E402
from baselines.parse_gecco import parse_gecco_dir  # noqa: E402


BASELINE_PARSERS = {
    "antismash": ("dir", parse_antismash_dir),
    "deepbgc": ("dir", parse_deepbgc_dir),
    "gecco": ("dir", parse_gecco_dir),
    "bgc_prophet": ("file", parse_bgc_prophet_file),
}


def predictions_to_tuples(
    preds: dict[str, list[dict]],
    genome_ids: list[str],
) -> dict[str, list[tuple[str, int, int]]]:
    out: dict[str, list[tuple[str, int, int]]] = {}
    for gid in genome_ids:
        out[gid] = [
            (p["contig_id"], int(p["start_bp"]), int(p["end_bp"]))
            for p in preds.get(gid, [])
        ]
    return out


def aggregate_over_genomes(
    gt: dict[str, list[tuple[str, int, int]]],
    pred: dict[str, list[tuple[str, int, int]]],
    genome_ids: list[str],
    modes: tuple[str, ...] = ("bench", "loose", "strict", "iou"),
) -> dict:
    per_genome: dict[str, dict] = {}
    agg = {
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
    for gid in genome_ids:
        per_genome[gid] = {}
        for mode in modes:
            m = eval_regions(gt.get(gid, []), pred.get(gid, []), mode)
            per_genome[gid][mode] = m
            for key in ("tp_recall", "tp_precision", "n_gt", "n_pred"):
                agg[mode][key] += int(m[key])
            frag = m["fragmentation"]
            agg[mode]["unique_gt_precision_tp"] += int(frag["unique_gt_precision_tp"])
            agg[mode]["duplicate_precision_tp"] += int(frag["duplicate_precision_tp"])
            agg[mode]["max_preds_per_gt"] = max(
                agg[mode]["max_preds_per_gt"],
                int(frag["max_preds_per_gt"]),
            )
    summary: dict[str, dict] = {}
    for mode, a in agg.items():
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
    return {"summary": summary, "per_genome": per_genome}


def decode_npmaster_operating_point(
    results_dir: Path,
    op: dict,
    genome_ids: list[str],
) -> dict[str, list[tuple[str, int, int]]]:
    out: dict[str, list[tuple[str, int, int]]] = {}
    for gid in genome_ids:
        csv_path = results_dir / f"{gid}_predictions.csv"
        windows = load_windows_from_csv(csv_path, score_family=op.get("score"))
        out[gid] = decode_regions_hysteresis(
            windows,
            start_threshold=float(op["start_threshold"]),
            extend_threshold=float(op["extend_threshold"]),
            max_gap=int(op["max_gap"]),
            min_support_windows=int(op.get("min_support_windows", 1)),
            length_cap=op.get("length_cap"),
            valley_threshold=op.get("valley_threshold"),
            valley_min_windows=int(op.get("valley_min_windows", 1)),
        )
    return out


def region_length_stats(pred: dict[str, list[tuple[str, int, int]]]) -> dict:
    lens = [e - s for regs in pred.values() for _, s, e in regs]
    lens.sort()
    if not lens:
        return {"n": 0}
    n = len(lens)
    median = lens[n // 2] if n % 2 else (lens[n // 2 - 1] + lens[n // 2]) // 2
    return {"n": n, "min": lens[0], "max": lens[-1], "median": int(median),
            "mean": int(sum(lens) / n)}


def format_markdown_table(rows: list[dict]) -> str:
    header = (
        "| Tool | R | P | F1 | dedup P | dedup F1 | n_pred | median pred len |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        b = r["summary"]["bench"]
        frag = b.get("fragmentation", {})
        ll = r["length_stats"]
        median = f"{ll['median']:,} bp" if ll.get("n") else "—"
        lines.append(
            f"| {r['label']} | {b['recall']:.4f} | {b['precision']:.4f} | "
            f"{b['f1']:.4f} | {frag.get('dedup_precision', 0.0):.4f} | "
            f"{frag.get('dedup_f1', 0.0):.4f} | {b['n_pred']} | {median} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path,
                        default=Path("/data/syh/benchmark/nine_genome/ground_truth.json"))
    parser.add_argument("--antismash-dir", type=Path, default=None)
    parser.add_argument("--deepbgc-dir", type=Path, default=None)
    parser.add_argument("--gecco-dir", type=Path, default=None)
    parser.add_argument("--bgc-prophet-json", type=Path, default=None)
    parser.add_argument("--npmaster-results-dir", type=Path, default=None,
                        help="Per-window CSV dir (e.g. stride8192_s0_merged).")
    parser.add_argument("--operating-points", default="val_frozen_fused,tuned_v2",
                        help="Comma-separated operating points from operating_points.yaml.")
    parser.add_argument("--operating-points-file", default=str(DEFAULT_OPERATING_POINTS),
                        type=Path)
    parser.add_argument("--genome-ids", default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if args.genome_ids:
        requested = {g.strip() for g in args.genome_ids.split(",") if g.strip()}
        genome_ids = [g for g in GENOME_IDS if g in requested]
    else:
        genome_ids = list(GENOME_IDS)

    gt = load_ground_truth(args.gt)
    rows: list[dict] = []

    baseline_inputs = [
        ("antiSMASH", "antismash", args.antismash_dir),
        ("DeepBGC", "deepbgc", args.deepbgc_dir),
        ("GECCO", "gecco", args.gecco_dir),
        ("BGC-Prophet", "bgc_prophet", args.bgc_prophet_json),
    ]
    for label, key, path in baseline_inputs:
        if path is None:
            continue
        kind, parser_fn = BASELINE_PARSERS[key]
        if kind == "dir":
            raw = parser_fn(path, genome_ids=genome_ids)
        else:
            raw = parser_fn(path, genome_ids=genome_ids)
        pred = predictions_to_tuples(raw, genome_ids)
        scored = aggregate_over_genomes(gt, pred, genome_ids)
        rows.append({
            "tool": key,
            "label": label,
            "source": str(path),
            "length_stats": region_length_stats(pred),
            **scored,
        })

    if args.npmaster_results_dir is not None:
        for op_name in [s.strip() for s in args.operating_points.split(",") if s.strip()]:
            op = load_operating_point(op_name, args.operating_points_file)
            pred = decode_npmaster_operating_point(args.npmaster_results_dir, op, genome_ids)
            scored = aggregate_over_genomes(gt, pred, genome_ids)
            rows.append({
                "tool": f"npmaster:{op_name}",
                "label": f"NP-Master `{op_name}`",
                "source": str(args.npmaster_results_dir),
                "operating_point": op_name,
                "length_stats": region_length_stats(pred),
                **scored,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"genome_ids": genome_ids, "rows": rows}, f, indent=2)

    md = format_markdown_table(rows)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
