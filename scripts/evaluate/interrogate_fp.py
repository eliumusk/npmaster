#!/usr/bin/env python3
"""Interrogate per-token false-positive predictions against antiSMASH calls.

For each per-token prediction that doesn't match any 9g GT region (bench-mode
50% overlap), check whether it overlaps an antiSMASH area on the same contig.

Usage:
  python scripts/evaluate/interrogate_fp.py \
    --probs-dir experiments/step2_per_token/round2/E10_h64_lr1e3_b32/9g_probs \
    --gt-json /data/syh/benchmark/nine_genome/ground_truth.json \
    --antismash-dir /data/syh/benchmark/nine_genome/results/antismash
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from per_token.decode import assemble_per_contig_track
from benchmark.nine_genome import load_ground_truth


def load_antismash_regions(antismash_dir: Path):
    """Parse antiSMASH JSON outputs → {genome_id: [(contig, start, end, products), ...]}."""
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
                s, e = area["start"], area["end"]
                products = area.get("products", [])
                regions.append((contig, s, e, products))
        regions_by_genome[genome_id] = regions
    return regions_by_genome


def decode_baseline(track_sum, track_count, threshold=0.5,
                    min_len_bp=2000, max_gap_bp=500):
    with np.errstate(invalid="ignore", divide="ignore"):
        track = np.where(track_count > 0,
                         track_sum / np.maximum(track_count, 1), 0.0)

    above = (track >= threshold).astype(np.int8)
    if above.sum() == 0:
        return []

    diff = np.diff(above, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if max_gap_bp > 0 and len(starts) > 1:
        merged_s, merged_e = [starts[0]], [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - merged_e[-1] < max_gap_bp:
                merged_e[-1] = e
            else:
                merged_s.append(s); merged_e.append(e)
        starts, ends = np.array(merged_s), np.array(merged_e)

    regions = []
    for s, e in zip(starts, ends):
        if e - s < min_len_bp:
            continue
        seg = track[s:e]
        regions.append((int(s), int(e), float(seg.mean())))
    return regions


def overlap_frac(s1, e1, s2, e2):
    ovl = max(0, min(e1, e2) - max(s1, s2))
    return ovl


def classify_predictions(preds_by_contig, gt_by_genome, genome_id):
    """Split predictions into TP and FP using bench-mode 50% overlap."""
    gt_regions = gt_by_genome.get(genome_id, [])

    tps, fps = [], []
    for contig, ps, pe, score in preds_by_contig:
        pred_len = pe - ps
        is_tp = False
        for gt_entry in gt_regions:
            gt_contig, gs, ge = gt_entry[0], gt_entry[1], gt_entry[2]
            if gt_contig != contig:
                continue
            ovl = overlap_frac(ps, pe, gs, ge)
            if ovl / pred_len >= 0.5:
                is_tp = True
                break
        if is_tp:
            tps.append((contig, ps, pe, score))
        else:
            fps.append((contig, ps, pe, score))
    return tps, fps


def check_antismash_overlap(fps, antismash_regions):
    """For each FP, check if it overlaps any antiSMASH area."""
    results = []
    for contig, ps, pe, score in fps:
        pred_len = pe - ps
        best_ovl = 0
        best_products = []
        any_overlap = False
        for as_contig, as_s, as_e, products in antismash_regions:
            if as_contig != contig:
                continue
            ovl = overlap_frac(ps, pe, as_s, as_e)
            if ovl > 0:
                any_overlap = True
                if ovl > best_ovl:
                    best_ovl = ovl
                    best_products = products
        ovl_frac_pred = best_ovl / pred_len if pred_len > 0 else 0
        results.append({
            "contig": contig,
            "start": ps,
            "end": pe,
            "length_bp": pred_len,
            "score": round(score, 4),
            "antismash_overlap": any_overlap,
            "overlap_bp": best_ovl,
            "overlap_frac": round(ovl_frac_pred, 3),
            "antismash_products": ", ".join(best_products) if best_products else "",
        })
    return results


def load_probs_and_build_tracks(probs_dir, upsample_k=8):
    tracks_by_genome = {}
    for npz_path in sorted(probs_dir.glob("*.probs.npz")):
        stem = npz_path.name.replace(".probs.npz", "")
        coords_path = probs_dir / f"{stem}.coords.csv"
        if not coords_path.exists():
            continue
        coords = {}
        with open(coords_path) as fh:
            for row in csv.DictReader(fh):
                coords[row["window_id"]] = (row["contig"], int(row["start"]),
                                             int(row["end"]))
        probs = np.load(npz_path)
        win_probs = {}
        for wid in probs.files:
            p_latent = probs[wid].astype(np.float32)
            p_nt = np.repeat(p_latent, upsample_k)
            win_probs[wid] = p_nt
        track_data = assemble_per_contig_track(win_probs, coords)
        tracks_by_genome[stem] = track_data
    return tracks_by_genome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probs-dir", required=True, type=Path)
    p.add_argument("--gt-json", required=True, type=Path)
    p.add_argument("--antismash-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--upsample-k", type=int, default=8)
    args = p.parse_args()

    print("Loading ground truth...")
    gt = load_ground_truth(args.gt_json)
    n_gt = sum(len(v) for v in gt.values())
    print(f"  {n_gt} GT regions across {len(gt)} genomes")

    print("Loading antiSMASH regions...")
    antismash = load_antismash_regions(args.antismash_dir)
    n_as = sum(len(v) for v in antismash.values())
    print(f"  {n_as} antiSMASH areas across {len(antismash)} genomes")

    print("Building per-contig probability tracks...")
    tracks = load_probs_and_build_tracks(args.probs_dir, args.upsample_k)
    print(f"  {len(tracks)} genomes loaded")

    print(f"\nDecoding with threshold={args.threshold}...")
    all_tps, all_fps = [], []
    all_fp_analysis = []

    for genome_id, contigs in sorted(tracks.items()):
        preds = []
        for contig, (s, cnt) in contigs.items():
            regions = decode_baseline(s, cnt, threshold=args.threshold)
            for (rs, re_, sc) in regions:
                preds.append((contig, rs, re_, sc))

        tps, fps = classify_predictions(preds, gt, genome_id)
        all_tps.extend(tps)
        all_fps.extend(fps)

        as_regions = antismash.get(genome_id, [])
        fp_analysis = check_antismash_overlap(fps, as_regions)
        for r in fp_analysis:
            r["genome"] = genome_id
        all_fp_analysis.extend(fp_analysis)

        n_fp_with_as = sum(1 for r in fp_analysis if r["antismash_overlap"])
        print(f"  {genome_id}: {len(preds)} pred, {len(tps)} TP, "
              f"{len(fps)} FP, {n_fp_with_as} FP overlap antiSMASH")

    # Summary
    n_fp = len(all_fps)
    n_fp_as = sum(1 for r in all_fp_analysis if r["antismash_overlap"])
    n_fp_as_50 = sum(1 for r in all_fp_analysis
                     if r["overlap_frac"] >= 0.5)

    print(f"\n{'='*70}")
    print(f"FP INTERROGATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total predictions:        {len(all_tps) + n_fp}")
    print(f"True positives (vs GT):   {len(all_tps)}")
    print(f"False positives (vs GT):  {n_fp}")
    print(f"")
    print(f"FP with ANY antiSMASH overlap:  {n_fp_as}/{n_fp} "
          f"({100*n_fp_as/n_fp:.1f}%)" if n_fp > 0 else "")
    print(f"FP with ≥50% antiSMASH overlap: {n_fp_as_50}/{n_fp} "
          f"({100*n_fp_as_50/n_fp:.1f}%)" if n_fp > 0 else "")
    print(f"Pure FP (no antiSMASH support): {n_fp - n_fp_as}/{n_fp} "
          f"({100*(n_fp - n_fp_as)/n_fp:.1f}%)" if n_fp > 0 else "")

    if n_fp_as >= 30:
        print(f"\n>>> {n_fp_as} FPs overlap antiSMASH — strong evidence of GT under-annotation <<<")
    elif n_fp_as < 10:
        print(f"\n>>> Only {n_fp_as} FPs overlap antiSMASH — these are likely true FPs <<<")
    else:
        print(f"\n>>> Mixed signal: {n_fp_as} FPs overlap antiSMASH — need deeper look <<<")

    # Detailed table
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

    # Length distribution of FPs
    fp_lens = [r["length_bp"] for r in all_fp_analysis]
    as_lens = [r["length_bp"] for r in all_fp_analysis if r["antismash_overlap"]]
    pure_fp_lens = [r["length_bp"] for r in all_fp_analysis if not r["antismash_overlap"]]

    print(f"\nLength distribution:")
    print(f"  All FP:   median={int(np.median(fp_lens)):,} bp, "
          f"mean={int(np.mean(fp_lens)):,} bp")
    if as_lens:
        print(f"  AS-overlap FP: median={int(np.median(as_lens)):,} bp, "
              f"mean={int(np.mean(as_lens)):,} bp")
    if pure_fp_lens:
        print(f"  Pure FP:  median={int(np.median(pure_fp_lens)):,} bp, "
              f"mean={int(np.mean(pure_fp_lens)):,} bp")


if __name__ == "__main__":
    main()
