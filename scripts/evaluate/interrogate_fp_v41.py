#!/usr/bin/env python3
"""Interrogate v4.1 false-positive predictions against antiSMASH calls.

Runs the same FP/antiSMASH overlap analysis as interrogate_fp.py,
but using v4.1's hysteresis decoder on per-window CSV predictions.

Usage:
  python scripts/evaluate/interrogate_fp_v41.py \
    --primary-dir experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
    --short-dir   experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
    --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
    --antismash-dir /data/syh/benchmark/nine_genome/results/antismash
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from benchmark.nine_genome import load_ground_truth
from decoding.postprocess import decode_regions_hysteresis, WindowScore


PRIMARY_OP = dict(start_threshold=0.95, extend_threshold=0.80,
                  min_support_windows=3, max_gap=0)
SHORT_OP = dict(start_threshold=0.95, extend_threshold=0.80,
                min_support_windows=5, max_gap=0)
SHORT_LENGTH_CAP = 10_000
IOU_DEDUP = 0.3


def load_window_scores(csv_path):
    """Load per-window CSV → {contig: [WindowScore, ...]}."""
    by_contig = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_contig[row["contig"]].append(WindowScore(
                contig=row["contig"],
                start=int(row["start"]),
                end=int(row["end"]),
                score=float(row["bgc_score"]),
            ))
    for c in by_contig:
        by_contig[c].sort(key=lambda w: w.start)
    return dict(by_contig)


def decode_genome(contig_windows, op):
    """Decode all contigs for one genome → [(contig, start, end, score), ...]."""
    all_windows = []
    for windows in contig_windows.values():
        all_windows.extend(windows)

    decoded = decode_regions_hysteresis(
        all_windows,
        start_threshold=op["start_threshold"],
        extend_threshold=op["extend_threshold"],
        min_support_windows=op["min_support_windows"],
        max_gap=op["max_gap"],
    )
    regions = []
    for contig, s, e in decoded:
        matching = [w.score for w in all_windows
                    if w.contig == contig and w.start >= s and w.end <= e]
        max_score = max(matching) if matching else 0
        regions.append((contig, s, e, max_score))
    return regions


def union_dedup(primary_regions, short_regions, iou_thr=0.3):
    """Merge primary + short (capped) regions, dedup by IoU."""
    short_filtered = [(c, s, e, sc) for c, s, e, sc in short_regions
                      if e - s < SHORT_LENGTH_CAP]

    all_regions = list(primary_regions) + short_filtered
    if not all_regions:
        return []

    all_regions.sort(key=lambda x: -x[3])
    kept = []
    for c, s, e, sc in all_regions:
        duplicate = False
        for kc, ks, ke, ksc in kept:
            if kc != c:
                continue
            ovl = max(0, min(e, ke) - max(s, ks))
            union = (e - s) + (ke - ks) - ovl
            if union > 0 and ovl / union >= iou_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append((c, s, e, sc))
    return kept


def load_antismash_regions(antismash_dir):
    regions_by_genome = {}
    for genome_dir in sorted(antismash_dir.iterdir()):
        if not genome_dir.is_dir():
            continue
        genome_id = genome_dir.name
        json_path = genome_dir / f"{genome_id}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        regions = []
        for rec in data.get("records", []):
            contig = rec.get("id", rec.get("name", ""))
            for area in rec.get("areas", []):
                regions.append((contig, area["start"], area["end"],
                                area.get("products", [])))
        regions_by_genome[genome_id] = regions
    return regions_by_genome


def overlap_bp(s1, e1, s2, e2):
    return max(0, min(e1, e2) - max(s1, s2))


def classify_and_check(preds, gt_regions, antismash_regions):
    tps, fps = [], []
    for contig, ps, pe, score in preds:
        pred_len = pe - ps
        is_tp = False
        for gt_contig, gs, ge in gt_regions:
            if gt_contig != contig:
                continue
            ovl = overlap_bp(ps, pe, gs, ge)
            if pred_len > 0 and ovl / pred_len >= 0.5:
                is_tp = True
                break
        if is_tp:
            tps.append((contig, ps, pe, score))
        else:
            fps.append((contig, ps, pe, score))

    fp_analysis = []
    for contig, ps, pe, score in fps:
        pred_len = pe - ps
        best_ovl = 0
        best_products = []
        for as_c, as_s, as_e, products in antismash_regions:
            if as_c != contig:
                continue
            ovl = overlap_bp(ps, pe, as_s, as_e)
            if ovl > best_ovl:
                best_ovl = ovl
                best_products = products
        fp_analysis.append({
            "contig": contig, "start": ps, "end": pe,
            "length_bp": pred_len, "score": round(score, 4),
            "antismash_overlap": best_ovl > 0,
            "overlap_bp": best_ovl,
            "overlap_frac": round(best_ovl / pred_len, 3) if pred_len > 0 else 0,
            "antismash_products": ", ".join(best_products) if best_products else "",
        })
    return tps, fps, fp_analysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary-dir", required=True, type=Path)
    p.add_argument("--short-dir", type=Path, default=None)
    p.add_argument("--gt-json", required=True, type=Path)
    p.add_argument("--antismash-dir", required=True, type=Path)
    args = p.parse_args()

    print("Loading ground truth...")
    gt = load_ground_truth(args.gt_json)
    n_gt = sum(len(v) for v in gt.values())
    print(f"  {n_gt} GT regions across {len(gt)} genomes")

    print("Loading antiSMASH regions...")
    antismash = load_antismash_regions(args.antismash_dir)
    n_as = sum(len(v) for v in antismash.values())
    print(f"  {n_as} antiSMASH areas across {len(antismash)} genomes")

    print("Decoding v4.1 predictions...")
    all_tps, all_fps, all_fp_analysis = [], [], []

    genomes = sorted(set(
        p.name.replace("_predictions.csv", "")
        for p in args.primary_dir.glob("*_predictions.csv")
    ))

    for genome_id in genomes:
        primary_csv = args.primary_dir / f"{genome_id}_predictions.csv"
        primary_windows = load_window_scores(primary_csv)
        primary_regions = decode_genome(primary_windows, PRIMARY_OP)

        short_regions = []
        if args.short_dir:
            short_csv = args.short_dir / f"{genome_id}_predictions.csv"
            if short_csv.exists():
                short_windows = load_window_scores(short_csv)
                short_regions = decode_genome(short_windows, SHORT_OP)

        preds = union_dedup(primary_regions, short_regions, IOU_DEDUP)

        gt_regions = gt.get(genome_id, [])
        as_regions = antismash.get(genome_id, [])
        tps, fps, fp_analysis = classify_and_check(preds, gt_regions, as_regions)

        for r in fp_analysis:
            r["genome"] = genome_id
        all_tps.extend(tps)
        all_fps.extend(fps)
        all_fp_analysis.extend(fp_analysis)

        n_fp_as = sum(1 for r in fp_analysis if r["antismash_overlap"])
        print(f"  {genome_id}: {len(preds)} pred, {len(tps)} TP, "
              f"{len(fps)} FP, {n_fp_as} FP overlap antiSMASH")

    n_fp = len(all_fps)
    n_fp_as = sum(1 for r in all_fp_analysis if r["antismash_overlap"])
    n_fp_as_50 = sum(1 for r in all_fp_analysis if r["overlap_frac"] >= 0.5)
    n_total = len(all_tps) + n_fp

    print(f"\n{'='*70}")
    print(f"v4.1 FP INTERROGATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total predictions:        {n_total}")
    print(f"True positives (vs GT):   {len(all_tps)}")
    print(f"False positives (vs GT):  {n_fp}")
    if n_fp > 0:
        print(f"")
        print(f"FP with ANY antiSMASH overlap:  {n_fp_as}/{n_fp} "
              f"({100*n_fp_as/n_fp:.1f}%)")
        print(f"FP with >=50% antiSMASH overlap: {n_fp_as_50}/{n_fp} "
              f"({100*n_fp_as_50/n_fp:.1f}%)")
        print(f"Pure FP (no antiSMASH support): {n_fp - n_fp_as}/{n_fp} "
              f"({100*(n_fp - n_fp_as)/n_fp:.1f}%)")

    print(f"\n{'='*70}")
    print(f"DETAILED FP TABLE (sorted by antiSMASH overlap)")
    print(f"{'='*70}")
    all_fp_analysis.sort(key=lambda x: -x["overlap_bp"])
    print(f"{'genome':<20s} {'contig':<15s} {'start':>10s} {'end':>10s} "
          f"{'len':>7s} {'score':>6s} {'AS_ovl':>7s} {'frac':>5s} {'products'}")
    print("-" * 110)
    for r in all_fp_analysis:
        print(f"{r['genome']:<20s} {r['contig']:<15s} {r['start']:>10d} "
              f"{r['end']:>10d} {r['length_bp']:>7d} {r['score']:>6.3f} "
              f"{r['overlap_bp']:>7d} {r['overlap_frac']:>5.2f} "
              f"{r['antismash_products']}")

    if all_fp_analysis:
        fp_lens = [r["length_bp"] for r in all_fp_analysis]
        as_lens = [r["length_bp"] for r in all_fp_analysis if r["antismash_overlap"]]
        pure_fp_lens = [r["length_bp"] for r in all_fp_analysis
                        if not r["antismash_overlap"]]
        print(f"\nLength distribution:")
        print(f"  All FP:   median={int(np.median(fp_lens)):,} bp, "
              f"mean={int(np.mean(fp_lens)):,} bp")
        if as_lens:
            print(f"  AS-overlap FP: median={int(np.median(as_lens)):,} bp, "
                  f"mean={int(np.mean(as_lens)):,} bp")
        if pure_fp_lens:
            print(f"  Pure FP:  median={int(np.median(pure_fp_lens)):,} bp, "
                  f"mean={int(np.mean(pure_fp_lens)):,} bp")

    # Side-by-side comparison hint
    print(f"\n{'='*70}")
    print("COMPARISON (paste per-token results alongside)")
    print(f"{'='*70}")
    print(f"v4.1:       {n_total} pred, {len(all_tps)} TP, {n_fp} FP, "
          f"{n_fp_as} AS-overlap ({100*n_fp_as/n_fp:.1f}%), "
          f"{n_fp - n_fp_as} pure FP" if n_fp > 0 else "")


if __name__ == "__main__":
    main()
