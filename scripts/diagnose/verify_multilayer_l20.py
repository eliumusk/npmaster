#!/usr/bin/env python3
"""Sanity-verify that the L20 features from scripts/extract/multi_layer.py
match the existing v4.1 9g L20 mean-pool features (modulo bf16/fp16 quant).

If this fails, the multi-layer extractor has a bug and the probe results are
not trustworthy.

Usage:
  python scripts/diagnose/verify_multilayer_l20.py \
      --new-feat-dir   experiments/multilayer_probe/9g \
      --ref-feat-dir   experiments/evo2_first_principles/9g_embeddings_w8192_s2048
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_per_row(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32); B = B.astype(np.float32)
    dot = (A * B).sum(axis=1)
    nA = np.linalg.norm(A, axis=1)
    nB = np.linalg.norm(B, axis=1)
    return dot / np.maximum(1e-9, nA * nB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-feat-dir", type=Path, required=True,
                    help="dir with {genome}.L20.npy + {genome}_windows.csv")
    ap.add_argument("--ref-feat-dir", type=Path, required=True,
                    help="dir with {genome}_embeddings.npy + {genome}_windows.csv")
    ap.add_argument("--cosine-floor", type=float, default=0.999,
                    help="min per-row cosine. fp16 round-trip should give >0.9999")
    args = ap.parse_args()

    new_csvs = sorted(args.new_feat_dir.glob("*_windows.csv"))
    if not new_csvs:
        print(f"[fail] no windows.csv in {args.new_feat_dir}", file=sys.stderr)
        sys.exit(2)

    any_fail = False
    for new_csv in new_csvs:
        stem = new_csv.name.replace("_windows.csv", "")
        new_npy = args.new_feat_dir / f"{stem}.L20.npy"
        ref_npy = args.ref_feat_dir / f"{stem}_embeddings.npy"
        ref_csv = args.ref_feat_dir / f"{stem}_windows.csv"
        if not (new_npy.exists() and ref_npy.exists() and ref_csv.exists()):
            print(f"[skip] {stem}: missing files (new={new_npy.exists()}, "
                  f"ref={ref_npy.exists()}, ref_csv={ref_csv.exists()})")
            continue

        new_df = pd.read_csv(new_csv)
        ref_df = pd.read_csv(ref_csv)
        if len(new_df) != len(ref_df):
            print(f"[FAIL] {stem}: row count {len(new_df)} != ref {len(ref_df)}")
            any_fail = True
            continue
        # Check coord alignment row-by-row.
        same_coords = ((new_df["contig"].astype(str).values
                        == ref_df["contig"].astype(str).values).all()
                       and (new_df["start"].astype(int).values
                            == ref_df["start"].astype(int).values).all()
                       and (new_df["end"].astype(int).values
                            == ref_df["end"].astype(int).values).all())
        if not same_coords:
            print(f"[FAIL] {stem}: window coords differ between dirs")
            any_fail = True
            continue

        new_X = np.load(new_npy)
        ref_X = np.load(ref_npy)
        if new_X.shape != ref_X.shape:
            print(f"[FAIL] {stem}: shape {new_X.shape} vs ref {ref_X.shape}")
            any_fail = True
            continue

        cos = cosine_per_row(new_X, ref_X)
        cos_lo = float(cos.min())
        cos_med = float(np.median(cos))
        diff_abs = np.abs(new_X.astype(np.float32) - ref_X.astype(np.float32)).mean()
        verdict = "OK" if cos_lo >= args.cosine_floor else "FAIL"
        print(f"[{verdict}] {stem}: n={len(new_X)} cos_min={cos_lo:.5f} "
              f"cos_med={cos_med:.5f} mean_abs_diff={diff_abs:.6f}")
        if cos_lo < args.cosine_floor:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
