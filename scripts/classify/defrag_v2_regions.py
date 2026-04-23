#!/usr/bin/env python3
"""Add per-type classification to defrag_v2 regions.

defrag_v2's ContextFullModel natively emits 8-class per-window probabilities
(already in `{genome}_predictions.csv` produced by inference_stride.py, in the
`probs` JSON column: `[Polyketide, NRP, Terpene, RiPP, Saccharide, Alkaloid, Other, non-BGC]`
— empirically verified 2026-04-23; earlier files wrote these indices assuming a
different order that turned out to be wrong, but the numeric values are correct).

For each defrag_v2 region, we max-aggregate the 7 BGC-type probs across
windows overlapping the region (same aggregation as scripts/classify/v4_1_regions.py
for fair apples-to-apples), then argmax to get dominant type.

Output: copy of regions CSV with added columns v41-style:
  defrag_v2_type, defrag_v2_type_score, defrag_v2_type_top2, defrag_v2_type_scores
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# CNN probs[:7] model-output order (verified against meta_arrays.pkl 2026-04-23).
CNN_TYPE_ORDER = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
BGC_TYPES = CNN_TYPE_ORDER  # alias for readability below


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regions-csv", type=Path, required=True)
    p.add_argument("--preds-dir", type=Path, required=True,
                   help="dir with {genome}_predictions.csv from inference_stride.py")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    regions = pd.read_csv(args.regions_csv)
    print(f"[load] {len(regions)} defrag_v2 regions from {args.regions_csv}", flush=True)

    out_rows = []
    for genome, grp in regions.groupby("genome", sort=False):
        pred_path = args.preds_dir / f"{genome}_predictions.csv"
        if not pred_path.exists():
            print(f"[skip] {genome}: no preds CSV", flush=True)
            for _, r in grp.iterrows():
                r = r.copy()
                r["defrag_v2_type"] = None
                r["defrag_v2_type_score"] = None
                r["defrag_v2_type_top2"] = None
                r["defrag_v2_type_scores"] = None
                out_rows.append(r)
            continue

        pred = pd.read_csv(pred_path)
        # Parse probs JSON column once, take columns 0..6 (7 BGC types)
        probs_arr = np.array([json.loads(p)[:7] for p in pred["probs"]], dtype=np.float32)
        # Indexing helpers
        c = pred["contig"].to_numpy()
        s = pred["start"].to_numpy()
        e = pred["end"].to_numpy()

        for _, r in grp.iterrows():
            mask = (c == r["contig"]) & (e > r["start"]) & (s < r["end"])
            if not mask.any():
                vt = vs = v2 = scs = None
            else:
                per_type = probs_arr[mask].max(axis=0)        # (7,)
                order = np.argsort(-per_type)
                vt = BGC_TYPES[int(order[0])]
                vs = float(per_type[order[0]])
                v2 = BGC_TYPES[int(order[1])]
                scs = ";".join(f"{BGC_TYPES[i]}={per_type[i]:.3f}" for i in range(7))
            r = r.copy()
            r["defrag_v2_type"] = vt
            r["defrag_v2_type_score"] = vs
            r["defrag_v2_type_top2"] = v2
            r["defrag_v2_type_scores"] = scs
            out_rows.append(r)
        print(f"[{genome}] labeled {len(grp)} regions", flush=True)

    out_df = pd.DataFrame(out_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print("\n[type distribution]")
    vc = out_df["defrag_v2_type"].value_counts(dropna=False)
    for t, n in vc.items():
        print(f"  {t!s:<12}  {n}")
    print(f"\n[done] {args.out_csv}")


if __name__ == "__main__":
    main()
