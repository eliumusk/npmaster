#!/usr/bin/env python3
"""Decode per-token BGC probabilities into region calls.

Reads per-stem outputs from `scripts/inference/per_token/infer_shards.py`:
  <probs_dir>/<stem>.probs.npz    window_id -> (T_latent,) fp16
  <probs_dir>/<stem>.coords.csv   window_id, contig, start, end, clean_len

For 9g the stem IS the genome_id (e.g. `GCA_000154945.1`). Each stem's
windows are decoded independently via `per_token.decode.decode_all`.

Per-latent-token probs are expanded to per-nt by nearest-neighbour upsample
(repeat by `upsample_k`) before being placed on the per-nt per-contig track.
This matches how the labels were built (per-nt ground truth, downsampled for
supervision) and keeps region coords in native bp.

Output:
  <out_regions_csv>  columns: genome, contig, start, end, score, type
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from per_token.decode import decode_all  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probs-dir", required=True, type=Path,
                   help="Dir containing <stem>.probs.npz + <stem>.coords.csv files")
    p.add_argument("--out-regions-csv", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-len-bp", type=int, default=2000)
    p.add_argument("--max-gap-bp", type=int, default=500)
    p.add_argument("--upsample-k", type=int, default=8,
                   help="bp per latent token (must match extractor downsample)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_regions_csv.parent.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(args.probs_dir.glob("*.probs.npz"))
    if not npz_paths:
        raise SystemExit(f"no probs files in {args.probs_dir}")

    all_rows: list[dict] = []
    K = args.upsample_k

    for npz_path in npz_paths:
        stem = npz_path.name.replace(".probs.npz", "")
        coords_path = args.probs_dir / f"{stem}.coords.csv"
        if not coords_path.exists():
            print(f"WARN: missing coords for {stem}; skipping")
            continue

        coords: dict[str, tuple[str, int, int]] = {}
        with open(coords_path) as fh:
            for row in csv.DictReader(fh):
                coords[row["window_id"]] = (row["contig"], int(row["start"]), int(row["end"]))

        probs = np.load(npz_path)
        # Upsample latent probs to per-nt (repeat by K).
        win_probs: dict[str, np.ndarray] = {}
        for wid in probs.files:
            p_latent = probs[wid].astype(np.float32)     # (T_latent,)
            p_nt = np.repeat(p_latent, K)                 # (T_latent * K,)
            win_probs[wid] = p_nt

        regions = decode_all(
            win_probs, coords,
            threshold=args.threshold,
            min_len_bp=args.min_len_bp,
            max_gap_bp=args.max_gap_bp,
        )
        for r in regions:
            all_rows.append({
                "genome": stem,
                "contig": r.contig,
                "start": r.start,
                "end": r.end,
                "score": f"{r.score:.4f}",
                "type": "",
            })
        print(f"[{stem}] {len(regions)} regions @ threshold={args.threshold}")

    with open(args.out_regions_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["genome", "contig", "start", "end", "score", "type"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {args.out_regions_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
