#!/usr/bin/env python3
"""Sweep decoder parameters on existing per-token 9g probabilities.

Tests: smoothing, hysteresis, min-support, score filtering.
Evaluates each config against 9g ground truth and prints a comparison table.

Usage:
  python scripts/evaluate/decoder_sweep.py \
    --probs-dir experiments/step2_per_token/round2/E10_h64_lr1e3_b32/9g_probs \
    --gt-json /data/syh/benchmark/nine_genome/ground_truth.json
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from per_token.decode import assemble_per_contig_track
from benchmark.nine_genome import load_ground_truth, eval_regions


def decode_smooth_threshold(
    track_sum, track_count,
    smooth_size=0, threshold=0.5,
    min_len_bp=2000, max_gap_bp=500,
):
    with np.errstate(invalid="ignore", divide="ignore"):
        track = np.where(track_count > 0, track_sum / np.maximum(track_count, 1), 0.0)

    if smooth_size > 0:
        kernel = np.ones(smooth_size) / smooth_size
        track = np.convolve(track, kernel, mode="same")

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


def decode_hysteresis(
    track_sum, track_count,
    start_thr=0.7, extend_thr=0.3,
    min_len_bp=2000, max_gap_bp=500,
    smooth_size=0,
):
    with np.errstate(invalid="ignore", divide="ignore"):
        track = np.where(track_count > 0, track_sum / np.maximum(track_count, 1), 0.0)

    if smooth_size > 0:
        kernel = np.ones(smooth_size) / smooth_size
        track = np.convolve(track, kernel, mode="same")

    N = len(track)
    raw_regions = []
    i = 0
    while i < N:
        if track[i] >= start_thr:
            s = i
            i += 1
            while i < N and track[i] >= extend_thr:
                i += 1
            raw_regions.append((s, i))
        else:
            i += 1

    if not raw_regions:
        return []

    # Gap-fill: merge regions separated by < max_gap_bp
    if max_gap_bp > 0:
        merged = [list(raw_regions[0])]
        for s, e in raw_regions[1:]:
            if s - merged[-1][1] < max_gap_bp:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        raw_regions = [(s, e) for s, e in merged]

    regions = []
    for s, e in raw_regions:
        if e - s < min_len_bp:
            continue
        seg = track[s:e]
        regions.append((int(s), int(e), float(seg.mean())))
    return regions


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
                coords[row["window_id"]] = (row["contig"], int(row["start"]), int(row["end"]))

        probs = np.load(npz_path)
        win_probs = {}
        for wid in probs.files:
            p_latent = probs[wid].astype(np.float32)
            p_nt = np.repeat(p_latent, upsample_k)
            win_probs[wid] = p_nt

        track_data = assemble_per_contig_track(win_probs, coords)
        tracks_by_genome[stem] = track_data
    return tracks_by_genome


def evaluate_decoder(tracks_by_genome, gt, decode_fn, **kwargs):
    all_gt = []
    all_pred = []
    for genome_id, contigs in tracks_by_genome.items():
        gt_regions = gt.get(genome_id, [])
        all_gt.extend(gt_regions)
        for contig, (s, cnt) in contigs.items():
            regions = decode_fn(s, cnt, **kwargs)
            for (rs, re_, sc) in regions:
                all_pred.append((contig, rs, re_))

    if not all_pred:
        return {"recall": 0, "precision": 0, "dedup_f1": 0, "n_pred": 0,
                "fragmentation": {"dedup_f1": 0, "max_preds_per_gt": 0}}

    m = eval_regions(all_gt, all_pred, mode="bench")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probs-dir", required=True, type=Path)
    p.add_argument("--gt-json", required=True, type=Path)
    p.add_argument("--upsample-k", type=int, default=8)
    args = p.parse_args()

    print("loading ground truth...")
    gt = load_ground_truth(args.gt_json)
    n_gt = sum(len(v) for v in gt.values())
    print(f"  {n_gt} ground truth regions across {len(gt)} genomes")

    print("building per-contig probability tracks...")
    tracks = load_probs_and_build_tracks(args.probs_dir, args.upsample_k)
    print(f"  {len(tracks)} genomes loaded\n")

    results = []

    def run(tag, fn, **kw):
        m = evaluate_decoder(tracks, gt, fn, **kw)
        df1 = m["fragmentation"]["dedup_f1"]
        results.append((tag, df1, m["precision"], m["recall"], m["n_pred"],
                         m["fragmentation"]["max_preds_per_gt"]))
        return df1

    # ── 1. Baselines ──
    print("=== Baseline (current decoder) ===")
    for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
        df1 = run(f"base_thr{thr}", decode_smooth_threshold, threshold=thr)
        print(f"  thr={thr}: dedup_F1={df1:.4f}")

    # ── 2. Smoothing ──
    print("\n=== Smoothing + threshold ===")
    for sz in [64, 128, 256, 512, 1024]:
        for thr in [0.3, 0.35, 0.4, 0.45, 0.5]:
            df1 = run(f"smooth{sz}_thr{thr}", decode_smooth_threshold,
                       smooth_size=sz, threshold=thr)
            if df1 > 0.48:
                print(f"  ** smooth={sz} thr={thr}: dedup_F1={df1:.4f}")

    # ── 3. Hysteresis ──
    print("\n=== Hysteresis ===")
    for s_thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for e_thr in [0.1, 0.2, 0.3, 0.4]:
            if e_thr >= s_thr:
                continue
            df1 = run(f"hyst_s{s_thr}_e{e_thr}", decode_hysteresis,
                       start_thr=s_thr, extend_thr=e_thr)
            if df1 > 0.48:
                print(f"  ** start={s_thr} extend={e_thr}: dedup_F1={df1:.4f}")

    # ── 4. Hysteresis + smoothing ──
    print("\n=== Hysteresis + smoothing ===")
    for sz in [64, 128, 256, 512]:
        for s_thr in [0.4, 0.5, 0.6, 0.7]:
            for e_thr in [0.1, 0.2, 0.3]:
                if e_thr >= s_thr:
                    continue
                df1 = run(f"sm{sz}_hyst_s{s_thr}_e{e_thr}", decode_hysteresis,
                           start_thr=s_thr, extend_thr=e_thr, smooth_size=sz)
                if df1 > 0.48:
                    print(f"  ** smooth={sz} start={s_thr} extend={e_thr}: dedup_F1={df1:.4f}")

    # ── 5. Min-length sweep ──
    print("\n=== Min-length sweep (thr=0.5) ===")
    for ml in [3000, 4000, 5000, 8000]:
        df1 = run(f"minlen{ml}", decode_smooth_threshold,
                   threshold=0.5, min_len_bp=ml)
        print(f"  min_len={ml}: dedup_F1={df1:.4f}")

    # ── 6. Gap sweep ──
    print("\n=== Gap sweep (thr=0.5) ===")
    for gap in [0, 200, 500, 1000, 2000, 4000]:
        df1 = run(f"gap{gap}", decode_smooth_threshold,
                   threshold=0.5, max_gap_bp=gap)
        print(f"  gap={gap}: dedup_F1={df1:.4f}")

    # ── Summary ──
    results.sort(key=lambda x: -x[1])
    print("\n" + "=" * 85)
    print("TOP 20 DECODER CONFIGURATIONS (E10)")
    print("=" * 85)
    print(f"{'config':<40s} {'dedup_F1':>8s} {'bench_P':>8s} {'bench_R':>8s} {'n_pred':>6s} {'max/gt':>6s}")
    print("-" * 85)
    for tag, df1, prec, rec, n_pred, mpg in results[:20]:
        print(f"{tag:<40s} {df1:8.4f} {prec:8.3f} {rec:8.3f} {n_pred:6d} {mpg:6d}")
    print()
    print(f"{'v4.1-hysteresis':<40s} {'0.4878':>8s} {'0.598':>8s} {'0.459':>8s} {'246':>6s}")
    print(f"{'current best (base_thr0.5)':<40s} {'0.4751':>8s} {'0.498':>8s} {'0.485':>8s} {'303':>6s}")

    best = results[0]
    gain = best[1] - 0.4751
    print(f"\nBest decoder: {best[0]} → dedup_F1={best[1]:.4f} (Δ={gain:+.4f} vs baseline)")
    if best[1] > 0.4878:
        print(">>> BEATS v4.1-hysteresis! <<<")
    else:
        gap_to_v41 = 0.4878 - best[1]
        print(f"Gap to v4.1-hysteresis: {gap_to_v41:.4f}")


if __name__ == "__main__":
    main()
