#!/usr/bin/env python3
"""Decode v4.1 ensemble regions from primary + short per-window preds (no GT).

Applies the frozen v4.1 multi-scale recipe:
  - Primary: decode with PRIMARY_OP (w=8192 scale window scores)
  - Short:   decode with SHORT_OP then filter L<SHORT_LENGTH_CAP (the 5–10 kb rescue arm)
  - Ensemble: union-dedup at IOU_DEDUP (higher-score wins)

Emits per-genome regions CSVs + an aggregate `all_regions.csv` + manifest.json.
Use this for novel genomes / metagenomes where no ground truth exists.
For benchmark eval with GT, use scripts/evaluate/evo2_lr_multiscale.py instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoding.evo2_lr_multiscale import (  # noqa: E402
    PRIMARY_OP,
    SHORT_OP,
    SHORT_LENGTH_CAP,
    IOU_DEDUP,
    decode_dir,
    union_dedup,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--primary-preds", type=Path, required=True,
                   help="dir of per-genome window CSVs at w=8192/s=2048")
    p.add_argument("--short-preds", type=Path, required=True,
                   help="dir of per-genome window CSVs at w=4096/s=1024")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[decode] primary ({args.primary_preds}) OP={PRIMARY_OP}", flush=True)
    primary = decode_dir(args.primary_preds, PRIMARY_OP)
    print(f"  {sum(len(v) for v in primary.values())} primary regions", flush=True)

    print(f"[decode] short ({args.short_preds}) OP={SHORT_OP}", flush=True)
    short = decode_dir(args.short_preds, SHORT_OP)
    short_filt = {g: [r for r in regs if (r[2] - r[1]) < SHORT_LENGTH_CAP]
                  for g, regs in short.items()}
    print(f"  {sum(len(v) for v in short_filt.values())} short kept "
          f"(L<{SHORT_LENGTH_CAP})", flush=True)

    genomes = sorted(set(primary) | set(short_filt))
    manifest = {
        "primary_op": PRIMARY_OP,
        "short_op": SHORT_OP,
        "short_length_cap": SHORT_LENGTH_CAP,
        "iou_dedup": IOU_DEDUP,
        "genomes": {},
    }

    all_rows: list[dict] = []
    for g in genomes:
        regs = union_dedup(primary.get(g, []), short_filt.get(g, []), IOU_DEDUP)
        # Tag each region by its source scale; check short first to match the
        # original handoff convention (if a region exists in both lists at
        # identical coords, label it as short_rescue).
        short_set = {(r[0], r[1], r[2]) for r in short_filt.get(g, [])}
        per_genome = args.out_dir / f"{g}_regions.csv"
        with per_genome.open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["genome", "contig", "start", "end", "max_score", "source_scale"])
            for r in regs:
                contig, start, end, score = r[0], int(r[1]), int(r[2]), float(r[3])
                scale = "short_rescue" if (contig, start, end) in short_set else "primary"
                w.writerow([g, contig, start, end, score, scale])
                all_rows.append({"genome": g, "contig": contig, "start": start,
                                 "end": end, "max_score": score, "source_scale": scale})
        manifest["genomes"][g] = {
            "n_primary_regions": len(primary.get(g, [])),
            "n_short_regions_raw": len([r for r in short.get(g, []) if True]),
            "n_short_regions_kept": len(short_filt.get(g, [])),
            "n_regions": len(regs),
            "regions_csv": per_genome.name,
        }
        print(f"[{g}] {len(regs)} regions", flush=True)

    all_path = args.out_dir / "regions.csv"
    with all_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["genome", "contig", "start", "end", "max_score", "source_scale"],
            lineterminator="\n")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    with (args.out_dir / "all_regions.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["genome", "contig", "start", "end", "max_score", "source_scale"],
            lineterminator="\n")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {args.out_dir} ({sum(len(v) for v in manifest['genomes'].values() if isinstance(v, dict)) or len(all_rows)} total rows)")


if __name__ == "__main__":
    main()
