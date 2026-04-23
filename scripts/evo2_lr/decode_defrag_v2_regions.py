#!/usr/bin/env python3
"""Decode defrag_v2_ms10_s099 regions from ContextFullModel per-window predictions.

Reads {genome}_predictions.csv (contig,start,end,bgc_score,non_bgc_score,pred_class,probs
with `bgc_score = max(probs[:7])` and `non_bgc_score = probs[7]`) and applies the
frozen defrag_v2_ms10_s099 operating point from `configs/operating_points.yaml`:
  score_family = fused = 0.5 * max_type + 0.5 * (1 - non_bgc_score)
  start_threshold = 0.99
  extend_threshold = 0.95
  min_support_windows = 10
  max_gap = 0

Outputs {genome}_regions.csv + all_regions.csv + manifest.json, same schema
as decode_v4_1_predictions.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from postprocess_v2 import WindowScore, decode_regions_hysteresis  # noqa: E402


OP = dict(
    start_threshold=0.99,
    extend_threshold=0.95,
    min_support_windows=10,
    max_gap=0,
)


def decode_one(csv_path: Path) -> list[tuple]:
    df = pd.read_csv(csv_path)
    fused = 0.5 * df["bgc_score"] + 0.5 * (1.0 - df["non_bgc_score"])
    wins = [
        WindowScore(str(r.contig), int(r.start), int(r.end), float(f))
        for (_, r), f in zip(df.iterrows(), fused)
    ]
    regs = decode_regions_hysteresis(wins, **OP)
    out = []
    for r in regs:
        contig, start, end = r[0], int(r[1]), int(r[2])
        mx = 0.0
        for w in wins:
            if w.contig == contig and w.end > start and w.start < end:
                mx = max(mx, w.score)
        out.append((contig, start, end, mx))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    manifest = {"op": OP, "score_family": "fused", "genomes": {}}
    for csv_path in sorted(args.preds_dir.glob("*_predictions.csv")):
        genome = csv_path.name[: -len("_predictions.csv")]
        regions = decode_one(csv_path)
        df = pd.DataFrame(regions, columns=["contig", "start", "end", "max_score"])
        df.insert(0, "genome", genome)
        df.to_csv(args.out_dir / f"{genome}_regions.csv", index=False)
        all_rows.append(df)
        manifest["genomes"][genome] = {"n_regions": len(regions)}
        print(f"[{genome}] {len(regions)} regions", flush=True)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            args.out_dir / "all_regions.csv", index=False
        )
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
