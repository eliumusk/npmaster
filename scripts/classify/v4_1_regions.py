#!/usr/bin/env python3
"""Add v4.1 per-type classification to v4.1 ensemble regions.

For each v4.1 region, apply the 7 per-type LR heads to the cached w=8192
embeddings, aggregate by MAX across windows in the region (9-genome validation
showed max beats mean substantially: top-1 0.71 vs 0.37), and take argmax over
the 7 BGC classes to get a dominant predicted type. Also record top-2 for
ambiguous hybrid cases.

Input:
  --regions-csv         v4.1 regions with columns [genome, contig, start, end, ...]
  --emb-dir             w=8192 cached embeddings dir (matches region genomes)
  --type-lr-dir         dir with lr_type_{Alkaloid|Terpene|NRP|Polyketide|RiPP|Saccharide|Other}.npz
  --out-csv             copy of regions CSV with added columns:
                          v4_1_type         (argmax)
                          v4_1_type_score   (score for argmax)
                          v4_1_type_top2    (second best)
                          v4_1_type_scores  (7 scores, semicolon-joined rounded)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classification.per_type import (  # noqa: E402
    BGC_TYPES, load_type_lrs, type_lr_forward,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regions-csv", type=Path, required=True)
    p.add_argument("--emb-dir", type=Path, required=True)
    p.add_argument("--type-lr-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    regions = pd.read_csv(args.regions_csv)
    print(f"[load] {len(regions)} v4.1 regions from {args.regions_csv}", flush=True)

    print("[load] 7 per-type LR heads", flush=True)
    type_lrs = load_type_lrs(args.type_lr_dir)

    # Process per genome to avoid loading all embeddings at once
    out_rows = []
    for genome, grp in regions.groupby("genome", sort=False):
        emb_p = args.emb_dir / f"{genome}_embeddings.npy"
        win_p = args.emb_dir / f"{genome}_windows.csv"
        if not (emb_p.exists() and win_p.exists()):
            print(f"[skip] {genome}: no cached emb", flush=True)
            for _, r in grp.iterrows():
                r = r.copy()
                r["v4_1_type"] = None
                r["v4_1_type_score"] = None
                r["v4_1_type_top2"] = None
                r["v4_1_type_scores"] = None
                out_rows.append(r)
            continue
        emb = np.load(emb_p).astype(np.float32)
        wins = pd.read_csv(win_p)
        type_probs = np.stack(
            [type_lr_forward(emb, type_lrs[t]) for t in BGC_TYPES], axis=1
        )  # (N_windows, 7)

        contig_arr = wins["contig"].to_numpy()
        start_arr = wins["start"].to_numpy()
        end_arr = wins["end"].to_numpy()
        for _, r in grp.iterrows():
            mask = (
                (contig_arr == r["contig"])
                & (end_arr > r["start"])
                & (start_arr < r["end"])
            )
            if not mask.any():
                vt = None; vs = None; v2 = None; scs = None
            else:
                per_type = type_probs[mask].max(axis=0)  # (7,)
                order = np.argsort(-per_type)
                vt = BGC_TYPES[int(order[0])]
                vs = float(per_type[order[0]])
                v2 = BGC_TYPES[int(order[1])]
                scs = ";".join(f"{BGC_TYPES[i]}={per_type[i]:.3f}" for i in range(7))
            r = r.copy()
            r["v4_1_type"] = vt
            r["v4_1_type_score"] = vs
            r["v4_1_type_top2"] = v2
            r["v4_1_type_scores"] = scs
            out_rows.append(r)
        print(f"[{genome}] labeled {len(grp)} regions", flush=True)

    out_df = pd.DataFrame(out_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    # Print type distribution summary
    print("\n[type distribution]")
    vc = out_df["v4_1_type"].value_counts(dropna=False)
    for t, n in vc.items():
        print(f"  {t!s:<12}  {n}")
    print(f"\n[done] {args.out_csv}")


if __name__ == "__main__":
    main()
