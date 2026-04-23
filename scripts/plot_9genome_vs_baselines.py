#!/usr/bin/env python3
"""Render 6-track BGC boundary plots for all 9 benchmark genomes.

Tracks (top → bottom): GT / antiSMASH / BGC-Prophet / DeepBGC /
defrag_v2_ms10_s099 / v4.1 ensemble.

Decodes v4.1 primary+short once and defrag_v2 once, then loops over genomes —
avoids re-decoding each preds dir per-genome (which `plot_genome_bgc_boundaries
--v41-track` would do if invoked 9×).

Canonical invocation (on server):
  conda activate bgc
  python scripts/plot_9genome_vs_baselines.py \
      --gt /data/syh/benchmark/nine_genome/ground_truth.json \
      --antismash-dir /data/syh/benchmark/nine_genome/results/antismash \
      --bgc-prophet-json /data/syh/benchmark/nine_genome/results/bgc_prophet/bgc_prophet_regions.json \
      --deepbgc-dir /data/syh/benchmark/nine_genome/results/deepbgc \
      --defrag-v2-preds-dir /data/muskliu/npmaster/data/phase4/defrag_v2_ms10_s099_preds_stride2048 \
      --v41-primary-dir /data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
      --v41-short-dir /data/muskliu/npmaster/experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
      --out-dir figures/9genome_comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

import evaluate_nine_genome as ev  # noqa: E402
from baselines.parse_antismash import parse_antismash_dir  # noqa: E402
from baselines.parse_bgc_prophet import parse_bgc_prophet_file  # noqa: E402
from baselines.parse_deepbgc import parse_deepbgc_dir  # noqa: E402
from plot_genome_bgc_boundaries import plot_genome_boundaries, NPMasterTrack  # noqa: E402
from decoding.postprocess import WindowScore, decode_regions_hysteresis  # noqa: E402
from decoding.v4_1 import (  # noqa: E402
    PRIMARY_OP as V41_PRIMARY_OP,
    SHORT_OP as V41_SHORT_OP,
    SHORT_LENGTH_CAP as V41_SHORT_LENGTH_CAP,
    IOU_DEDUP as V41_IOU_DEDUP,
    decode_dir as v41_decode_dir,
    union_dedup as v41_union_dedup,
)


# defrag_v2_ms10_s099 OP, mirrors configs/operating_points.yaml
DEFRAG_V2_OP = dict(
    score_family="fused",
    start_threshold=0.99,
    extend_threshold=0.95,
    min_support_windows=10,
    length_cap=None,
    max_gap=0,
)


def decode_defrag_v2(preds_dir: Path) -> dict[str, list[tuple[str, int, int]]]:
    out: dict[str, list[tuple[str, int, int]]] = {}
    for csv_path in sorted(preds_dir.glob("*_predictions.csv")):
        genome = csv_path.name.replace("_predictions.csv", "")
        windows = ev.load_windows_from_csv(csv_path, score_family=DEFRAG_V2_OP["score_family"])
        regions = decode_regions_hysteresis(
            windows,
            start_threshold=DEFRAG_V2_OP["start_threshold"],
            extend_threshold=DEFRAG_V2_OP["extend_threshold"],
            max_gap=DEFRAG_V2_OP["max_gap"],
            min_support_windows=DEFRAG_V2_OP["min_support_windows"],
            length_cap=DEFRAG_V2_OP["length_cap"],
        )
        out[genome] = [(r[0], int(r[1]), int(r[2])) for r in regions]
    return out


def decode_v41(primary_dir: Path, short_dir: Path) -> dict[str, list[tuple[str, int, int]]]:
    primary = v41_decode_dir(primary_dir, V41_PRIMARY_OP)
    short = v41_decode_dir(short_dir, V41_SHORT_OP)
    short_filt = {g: [r for r in regs if (r[2] - r[1]) < V41_SHORT_LENGTH_CAP]
                  for g, regs in short.items()}
    genomes = set(primary) | set(short_filt)
    out: dict[str, list[tuple[str, int, int]]] = {}
    for g in genomes:
        merged = v41_union_dedup(primary.get(g, []), short_filt.get(g, []), V41_IOU_DEDUP)
        out[g] = [(r[0], int(r[1]), int(r[2])) for r in merged]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--antismash-dir", required=True, type=Path)
    p.add_argument("--bgc-prophet-json", required=True, type=Path)
    p.add_argument("--deepbgc-dir", required=True, type=Path)
    p.add_argument("--defrag-v2-preds-dir", required=True, type=Path)
    p.add_argument("--v41-primary-dir", required=True, type=Path)
    p.add_argument("--v41-short-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--genome-ids", default=None,
                   help="Comma-separated subset; default: all genomes in GT")
    p.add_argument("--max-contigs", type=int, default=None,
                   help="Only plot the N most-populated contigs per genome")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(args.gt.read_text())["ground_truth"]
    genomes = sorted(gt.keys())
    if args.genome_ids:
        want = {s.strip() for s in args.genome_ids.split(",") if s.strip()}
        genomes = [g for g in genomes if g in want]

    print(f"[decode] antiSMASH / BGC-Prophet / DeepBGC ...", flush=True)
    antismash = parse_antismash_dir(args.antismash_dir, genome_ids=genomes)
    bgc_prophet = parse_bgc_prophet_file(args.bgc_prophet_json, genome_ids=genomes)
    deepbgc = parse_deepbgc_dir(args.deepbgc_dir, genome_ids=genomes)

    print(f"[decode] defrag_v2_ms10_s099 from {args.defrag_v2_preds_dir}", flush=True)
    defrag = decode_defrag_v2(args.defrag_v2_preds_dir)

    print(f"[decode] v4.1 ensemble (primary={args.v41_primary_dir}, short={args.v41_short_dir})",
          flush=True)
    v41 = decode_v41(args.v41_primary_dir, args.v41_short_dir)

    def _unified(preds: dict[str, list[dict]], g: str) -> list[tuple[str, int, int]]:
        return [(r["contig_id"], int(r["start_bp"]), int(r["end_bp"])) for r in preds.get(g, [])]

    for g in genomes:
        baseline_tracks = [
            ("antiSMASH", _unified(antismash, g)),
            ("BGC-Prophet", _unified(bgc_prophet, g)),
            ("DeepBGC", _unified(deepbgc, g)),
            ("defrag_v2_ms10_s099", defrag.get(g, [])),
            ("v4.1 ensemble", v41.get(g, [])),
        ]
        out_png = args.out_dir / f"{g}.png"
        out_json = args.out_dir / f"{g}.json"
        plot_genome_boundaries(
            genome_id=g,
            gt_path=args.gt,
            out_png=out_png,
            out_json=out_json,
            baseline_tracks=baseline_tracks,
            npmaster_tracks=[],  # all NP-Master tracks are precomputed above
            max_contigs=args.max_contigs,
        )
        print(f"  [{g}] saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
