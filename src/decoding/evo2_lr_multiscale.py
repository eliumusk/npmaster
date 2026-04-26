#!/usr/bin/env python3
"""evo2_lr_multiscale ensemble honest readout on 9-genome.

Two-scale union-dedup ensemble:
  - Primary: LR at w=8192/s=2048, OP s=0.95/e=0.80/ms=3 (v4 alt)
  - Short rescue: LR at w=4096/s=1024, OP s=0.95/e=0.80/ms=5, filter L<10kb

All OPs pre-declared from first principles (see plan §13).
Produces per-bucket recall + pooled metrics + bootstrap CI over genomes.

Usage:
  python src/decoding/evo2_lr_multiscale.py \\
    --primary-preds experiments/evo2_lr_multiscale/9g_preds_lrA1_w8192 \\
    --short-preds  experiments/evo2_lr_multiscale/short_bgc/9g_preds_lrA1_w4096 \\
    --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \\
    --out-json experiments/evo2_lr_multiscale/v4_1_readout.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
from decoding.postprocess import WindowScore, decode_regions_hysteresis  # noqa: E402
from benchmark.nine_genome import eval_regions  # noqa: E402


# --- evo2_lr_multiscale frozen operating points ---------------------------------------------

PRIMARY_OP = {"start_threshold": 0.95, "extend_threshold": 0.80,
              "min_support_windows": 3, "max_gap": 0}
SHORT_OP = {"start_threshold": 0.95, "extend_threshold": 0.80,
            "min_support_windows": 5, "max_gap": 0}  # min region = 8 kb at w=4096/s=1024
SHORT_LENGTH_CAP = 10_000   # only keep short-scale regions shorter than this (5-10 kb rescue)
IOU_DEDUP = 0.3


def decode_dir(preds_dir: Path, op: dict) -> dict[str, list[tuple]]:
    """Decode regions from a directory of per-genome window CSVs.

    Returns {genome: [(contig, start, end, max_score)]}.
    """
    out: dict[str, list[tuple]] = {}
    for csv_path in sorted(preds_dir.glob("*_predictions.csv")):
        genome = csv_path.name[: -len("_predictions.csv")]
        df = pd.read_csv(csv_path)
        wins = [WindowScore(str(r["contig"]), int(r["start"]), int(r["end"]),
                             float(r["bgc_score"]))
                for _, r in df.iterrows()]
        regs = decode_regions_hysteresis(wins,
                                          start_threshold=op["start_threshold"],
                                          extend_threshold=op["extend_threshold"],
                                          min_support_windows=op["min_support_windows"],
                                          max_gap=op["max_gap"])
        result = []
        for r in regs:
            c, rs, re_ = r[0], int(r[1]), int(r[2])
            mx = 0.0
            for w in wins:
                if w.contig == c and w.end > rs and w.start < re_:
                    mx = max(mx, w.score)
            result.append((c, rs, re_, mx))
        out[genome] = result
    return out


def iou(a: tuple, b: tuple) -> float:
    if a[0] != b[0]:
        return 0.0
    o = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    return o / max(1, max(a[2], b[2]) - min(a[1], b[1]))


def union_dedup(primary: list, short: list, iou_thresh: float = IOU_DEDUP) -> list:
    """Union of two region sets. Where regions overlap (IoU >= iou_thresh),
    keep the higher-score one (equal score → primary wins by sort order)."""
    allr = sorted(list(primary) + list(short), key=lambda r: r[3], reverse=True)
    kept = []
    for r in allr:
        if not any(iou(r, k) >= iou_thresh for k in kept):
            kept.append(r)
    return kept


def bucket_for(length: int) -> str:
    if length < 5000: return "<5kb"
    if length < 10000: return "5-10kb"
    if length < 20000: return "10-20kb"
    if length < 50000: return "20-50kb"
    return ">=50kb"


BUCKETS = ["<5kb", "5-10kb", "10-20kb", "20-50kb", ">=50kb"]


def bucket_recall(preds_by_g: dict, gt: dict) -> dict[str, str]:
    stats = {b: {"n_gt": 0, "hits": 0} for b in BUCKETS}
    for g, gts in gt.items():
        pr = preds_by_g.get(g, [])
        for gc, gs, ge in gts:
            gl = ge - gs
            b = bucket_for(gl)
            stats[b]["n_gt"] += 1
            hit = any(p[0] == gc and max(0, min(p[2], ge) - max(p[1], gs)) / max(1, gl) >= 0.5
                      for p in pr)
            if hit: stats[b]["hits"] += 1
    return {b: f"{stats[b]['hits']}/{stats[b]['n_gt']} ({stats[b]['hits']/max(1,stats[b]['n_gt']):.3f})"
            for b in BUCKETS}


def per_genome_counts(preds_by_g: dict, gt: dict) -> dict:
    pg = {}
    strict_t_rec = strict_t_prec = 0
    for g, gts in gt.items():
        pp = [(p[0], p[1], p[2]) for p in preds_by_g.get(g, [])]
        mb = eval_regions(list(gts), pp, mode="bench", min_overlap=0.5)
        ms_ = eval_regions(list(gts), pp, mode="strict", min_overlap=0.5)
        pg[g] = {
            "n_gt": mb["n_gt"], "n_pred": mb["n_pred"],
            "tp_r": mb["tp_recall"], "tp_p": mb["tp_precision"],
            "u_gt": mb["fragmentation"]["unique_gt_precision_tp"],
            "strict_tp_r": int(round(ms_["recall"] * mb["n_gt"])),
            "strict_tp_p": int(round(ms_["precision"] * mb["n_pred"])),
        }
        strict_t_rec += pg[g]["strict_tp_r"]
        strict_t_prec += pg[g]["strict_tp_p"]
    return pg


def pool_metrics(pg_counts: dict, genomes: list) -> dict:
    tr = sum(pg_counts[g]["tp_r"] for g in genomes)
    tp = sum(pg_counts[g]["tp_p"] for g in genomes)
    ng = sum(pg_counts[g]["n_gt"] for g in genomes)
    nn = sum(pg_counts[g]["n_pred"] for g in genomes)
    ug = sum(pg_counts[g]["u_gt"] for g in genomes)
    str_tr = sum(pg_counts[g]["strict_tp_r"] for g in genomes)
    str_tp = sum(pg_counts[g]["strict_tp_p"] for g in genomes)
    r = tr / max(1, ng); p = tp / max(1, nn); bf = 2*p*r / max(1e-12, p + r)
    dp = ug / max(1, nn); df1 = 2*dp*r / max(1e-12, dp + r)
    sr = str_tr / max(1, ng); sp = str_tp / max(1, nn); sf = 2*sp*sr / max(1e-12, sp + sr)
    return {"n_gt": ng, "n_pred": nn,
            "bench_precision": p, "bench_recall": r, "bench_f1": bf,
            "dedup_f1": df1, "strict_f1": sf, "strict_precision": sp, "strict_recall": sr}


def bootstrap_ci(pg_counts: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    genomes = list(pg_counts.keys())
    samples = {k: [] for k in ("dedup_f1", "strict_f1", "bench_f1",
                                "bench_precision", "bench_recall")}
    for _ in range(n_boot):
        sample = list(rng.choice(genomes, size=len(genomes), replace=True))
        m = pool_metrics(pg_counts, sample)
        for k in samples:
            samples[k].append(m[k])
    return {k: {"mean": float(np.mean(v)),
                "ci_lo": float(np.quantile(v, 0.025)),
                "ci_hi": float(np.quantile(v, 0.975))}
            for k, v in samples.items()}


def run_readout(
    primary_preds: Path,
    short_preds: Path,
    gt_json: Path,
    out_json: Path | None = None,
    bootstrap: int = 2000,
) -> dict:
    """Evaluate the active Evo2-LR multiscale recipe on a ground-truth JSON."""
    gt = json.loads(gt_json.read_text())["ground_truth"]
    print(f"[gt] {len(gt)} genomes, {sum(len(v) for v in gt.values())} regions")

    print(f"[decode] primary ({primary_preds}) with OP {PRIMARY_OP}")
    primary = decode_dir(primary_preds, PRIMARY_OP)
    primary_total = sum(len(v) for v in primary.values())
    print(f"[decode] primary regions: {primary_total}")

    print(f"[decode] short ({short_preds}) with OP {SHORT_OP}")
    short = decode_dir(short_preds, SHORT_OP)
    short_all = sum(len(v) for v in short.values())
    # Apply length cap
    short_filt = {g: [r for r in regs if (r[2] - r[1]) < SHORT_LENGTH_CAP]
                  for g, regs in short.items()}
    short_kept = sum(len(v) for v in short_filt.values())
    print(f"[decode] short regions: {short_all} total; {short_kept} after L<{SHORT_LENGTH_CAP}")

    # evo2_lr_multiscale ensemble
    ensemble = {g: union_dedup(primary.get(g, []), short_filt.get(g, []), IOU_DEDUP)
                for g in gt.keys()}

    # Baselines for comparison
    print("\n[eval] three configurations:")
    results = {}
    for name, preds in [("primary_only (v4 baseline)", primary),
                        ("short_only (gated L<10k)", short_filt),
                        ("evo2_lr_multiscale ensemble", ensemble)]:
        pg = per_genome_counts(preds, gt)
        pooled = pool_metrics(pg, list(pg.keys()))
        ci = bootstrap_ci(pg, n_boot=bootstrap)
        buckets = bucket_recall(preds, gt)
        results[name] = {"pooled": pooled, "bootstrap_ci_95": ci, "bucket_recall": buckets}
        print(f"\n  [{name}]  n_pred={pooled['n_pred']}")
        print(f"    dedup_F1  = {pooled['dedup_f1']:.4f}  "
              f"[CI95: {ci['dedup_f1']['ci_lo']:.3f}..{ci['dedup_f1']['ci_hi']:.3f}]")
        print(f"    strict_F1 = {pooled['strict_f1']:.4f}  "
              f"[CI95: {ci['strict_f1']['ci_lo']:.3f}..{ci['strict_f1']['ci_hi']:.3f}]")
        print(f"    bench_F1  = {pooled['bench_f1']:.4f}  "
              f"[CI95: {ci['bench_f1']['ci_lo']:.3f}..{ci['bench_f1']['ci_hi']:.3f}]")
        print(f"    per-bucket recall:")
        for b in BUCKETS:
            print(f"      {b:>8s}: {buckets[b]}")

    # Pretty summary
    b = results["primary_only (v4 baseline)"]["pooled"]
    e = results["evo2_lr_multiscale ensemble"]["pooled"]
    print("\n=== evo2_lr_multiscale vs v4 baseline ===")
    print(f"  Δ dedup_F1:  {e['dedup_f1'] - b['dedup_f1']:+.4f}")
    print(f"  Δ strict_F1: {e['strict_f1'] - b['strict_f1']:+.4f}")
    print(f"  Δ bench_F1:  {e['bench_f1'] - b['bench_f1']:+.4f}")
    print(f"  Δ n_pred:    {e['n_pred'] - b['n_pred']:+d}")

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "config": {
                "primary_op": PRIMARY_OP, "short_op": SHORT_OP,
                "short_length_cap": SHORT_LENGTH_CAP, "iou_dedup": IOU_DEDUP,
            },
            "results": results,
        }, indent=2))
        print(f"\n[done] {out_json}")
    return results
