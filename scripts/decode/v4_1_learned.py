#!/usr/bin/env python3
"""Decode v4.1-learned ensemble regions from primary + short per-window preds.

Replaces the hysteresis decoder in `scripts/decode/v4_1.py` with the learned
1D CNN decoder (`src/decoding/learned.py`). Everything else matches v4.1
multi-scale: primary (w=8192/s=2048) + short rescue (w=4096/s=1024, L<10 kb)
merged via union-dedup at IoU=0.3.

Frozen OP: primary threshold 0.95, short threshold 0.98. See
`docs/learned-decoder.md` for derivation.

Emits per-genome regions CSVs + aggregate all_regions.csv + manifest.json.
To add per-type classification, chain with:
    scripts/classify/v4_1_regions.py    (Stage-2 per-type LR)
    scripts/classify/defrag_v2_regions.py (CNN 8-class probs)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoding.learned import (  # noqa: E402
    PRIMARY_THRESHOLD, SHORT_THRESHOLD, SHORT_LENGTH_CAP, IOU_DEDUP,
    decode_dir, load_decoder, union_dedup,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--primary-preds", type=Path, required=True,
                   help="dir of per-genome CSVs at w=8192/s=2048")
    p.add_argument("--short-preds", type=Path, required=True,
                   help="dir of per-genome CSVs at w=4096/s=1024")
    p.add_argument("--decoder-ckpt", type=Path, required=True,
                   help="path to trained Decoder1DCNN state_dict (.pt)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--primary-threshold", type=float, default=PRIMARY_THRESHOLD)
    p.add_argument("--short-threshold", type=float, default=SHORT_THRESHOLD)
    p.add_argument("--short-length-cap", type=int, default=SHORT_LENGTH_CAP)
    p.add_argument("--iou-dedup", type=float, default=IOU_DEDUP)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[decoder] loading {args.decoder_ckpt}", flush=True)
    model = load_decoder(args.decoder_ckpt, device=device,
                         depth=args.depth, hidden=args.hidden)

    print(f"[decode] primary ({args.primary_preds}) thr={args.primary_threshold}", flush=True)
    primary = decode_dir(args.primary_preds, model, args.primary_threshold, device)
    print(f"  {sum(len(v) for v in primary.values())} primary regions", flush=True)

    print(f"[decode] short ({args.short_preds}) thr={args.short_threshold}", flush=True)
    short = decode_dir(args.short_preds, model, args.short_threshold, device)

    # union_dedup expects per-genome lists of tuples; apply per-genome
    genomes = sorted(set(primary) | set(short))
    manifest = {
        "primary_threshold": args.primary_threshold,
        "short_threshold": args.short_threshold,
        "short_length_cap": args.short_length_cap,
        "iou_dedup": args.iou_dedup,
        "decoder_ckpt": str(args.decoder_ckpt),
        "decoder_depth": args.depth, "decoder_hidden": args.hidden,
        "genomes": genomes,
    }

    total_prim = 0
    total_short = 0
    all_rows: list[dict] = []
    for g in genomes:
        p_regs = primary.get(g, [])
        s_regs = short.get(g, [])
        merged = union_dedup(p_regs, s_regs, args.short_length_cap, args.iou_dedup)
        # merged tuples: (contig, start, end, score, source)
        out_csv = args.out_dir / f"{g}_regions.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["genome", "contig", "start", "end", "max_score", "source_scale"])
            for (c, s, e, sc, src) in merged:
                w.writerow([g, c, s, e, sc, src])
                all_rows.append({"genome": g, "contig": c, "start": s, "end": e,
                                 "max_score": sc, "source_scale": src})
                if src == "primary": total_prim += 1
                else: total_short += 1

    agg = args.out_dir / "all_regions.csv"
    if all_rows:
        with open(agg, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows: w.writerow(r)

    with open(args.out_dir / "manifest.json", "w") as fh:
        json.dump({**manifest, "total_regions": len(all_rows),
                   "primary_count": total_prim, "short_rescue_count": total_short}, fh, indent=2)

    print(f"\n[write] {agg}  ({len(all_rows)} regions: {total_prim} primary + {total_short} short)",
          flush=True)


if __name__ == "__main__":
    main()
