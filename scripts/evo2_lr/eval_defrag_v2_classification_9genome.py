#!/usr/bin/env python3
"""Evaluate defrag_v2 CNN's per-type classification vs antiSMASH on 9-genome.

Same pairing + confusion matrix logic as eval_v41_classification_9genome.py,
but the type source is the ContextFullModel CNN head's native 8-class output
stored in per-window prediction CSVs (column `probs` = JSON list of 8).

We pair v4.1 ensemble regions ↔ antiSMASH regions (same as v4.1 eval), then
at each matched region, take the CNN's 7 BGC-type probs from every window in
the region, max-aggregate, and argmax → defrag_v2 dominant type. This makes
the result directly comparable to the v4.1 confusion matrix on the same
regions.

Typical invocation (on server):
  python scripts/evo2_lr/eval_defrag_v2_classification_9genome.py \
      --v41-primary-preds .../9g_preds_lrA1_w8192 \
      --v41-short-preds   .../short_bgc/9g_preds_lrA1_w4096 \
      --defrag-v2-preds-dir /data/muskliu/npmaster/data/phase4/defrag_v2_ms10_s099_preds_stride2048 \
      --antismash-dir /data/syh/benchmark/nine_genome/results/antismash \
      --bgc-annotations /data/muskliu/npmaster/data/bgc_annotations.csv \
      --agg max \
      --out-json experiments/defrag_v2_classification_9genome.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evo2_lr"))

from baselines.parse_antismash import parse_antismash_dir  # noqa: E402
from eval_v41_classification_9genome import (  # noqa: E402
    BGC_TYPES,
    build_product_to_type_map,
    iou_pair,
    map_antismash_products,
)

# CNN probs[:7] model-output order (verified 2026-04-23 from meta_arrays.pkl).
# Stage 2 per-type LRs (used by v4.1 eval) use BGC_TYPES directly; the CNN head
# uses a different column order that we have to respect here.
CNN_TYPE_ORDER = ["Polyketide", "NRP", "Terpene", "RiPP", "Saccharide", "Alkaloid", "Other"]
from v4_1_multiscale_readout import (  # noqa: E402
    PRIMARY_OP as V41_PRIMARY_OP,
    SHORT_OP as V41_SHORT_OP,
    SHORT_LENGTH_CAP as V41_SHORT_LENGTH_CAP,
    IOU_DEDUP as V41_IOU_DEDUP,
    decode_dir as v41_decode_dir,
    union_dedup as v41_union_dedup,
)


def defrag_type_for_region(region, pred_df, agg="max"):
    """Aggregate defrag_v2 CNN's 7 BGC-type probs across windows in region.

    pred_df must have columns contig, start, end, probs (JSON 8-list).
    Returns (top1_type, 7 aggregated scores) or (None, None) if no windows.
    """
    contig, rs, re_ = region[0], int(region[1]), int(region[2])
    c = pred_df["contig"].to_numpy()
    s = pred_df["start"].to_numpy()
    e = pred_df["end"].to_numpy()
    mask = (c == contig) & (e > rs) & (s < re_)
    if not mask.any():
        return None, None
    idx = np.where(mask)[0]
    probs = np.array([json.loads(pred_df["probs"].iat[i])[:7] for i in idx],
                     dtype=np.float32)     # (k, 7)
    if agg == "mean":
        per_type = probs.mean(axis=0)
    elif agg == "max":
        per_type = probs.max(axis=0)
    else:
        raise ValueError(agg)
    order = np.argsort(-per_type)
    return CNN_TYPE_ORDER[int(order[0])], per_type.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v41-primary-preds", type=Path, required=True)
    p.add_argument("--v41-short-preds", type=Path, required=True)
    p.add_argument("--defrag-v2-preds-dir", type=Path, required=True,
                   help="dir with {genome}_predictions.csv from ContextFullModel inference")
    p.add_argument("--antismash-dir", type=Path, required=True)
    p.add_argument("--bgc-annotations", type=Path, required=True)
    p.add_argument("--iou-min", type=float, default=0.3)
    p.add_argument("--agg", choices=["mean", "max"], default="max")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    print("[load] product→type map from training annotations", flush=True)
    prod_to_type = build_product_to_type_map(args.bgc_annotations)
    print(f"  {len(prod_to_type)} unique products → 7-class", flush=True)

    print("[decode] v4.1 ensemble regions", flush=True)
    primary = v41_decode_dir(args.v41_primary_preds, V41_PRIMARY_OP)
    short = v41_decode_dir(args.v41_short_preds, V41_SHORT_OP)
    short_filt = {g: [r for r in regs if (r[2] - r[1]) < V41_SHORT_LENGTH_CAP]
                  for g, regs in short.items()}
    genomes = sorted(set(primary) | set(short_filt))
    v41 = {g: v41_union_dedup(primary.get(g, []), short_filt.get(g, []), V41_IOU_DEDUP)
           for g in genomes}
    print(f"  {sum(len(v) for v in v41.values())} ensemble regions", flush=True)

    print("[load] antiSMASH 9-genome", flush=True)
    antis = parse_antismash_dir(args.antismash_dir, genome_ids=genomes)

    pairs: list[dict] = []
    unmatched = 0
    for g in genomes:
        pred_path = args.defrag_v2_preds_dir / f"{g}_predictions.csv"
        if not pred_path.exists():
            print(f"[skip] {g}: no defrag_v2 preds", flush=True)
            continue
        pred_df = pd.read_csv(pred_path)
        a_regs = [(r["contig_id"], int(r["start_bp"]), int(r["end_bp"]),
                   r["predicted_type"]) for r in antis.get(g, [])]

        for vr in v41.get(g, []):
            best_iou = 0.0
            best_a = None
            for ar in a_regs:
                iou = iou_pair(vr, ar)
                if iou > best_iou:
                    best_iou = iou
                    best_a = ar
            dv2_type, dv2_scores = defrag_type_for_region(vr, pred_df, agg=args.agg)
            if best_a is None or best_iou < args.iou_min:
                unmatched += 1
                continue
            a_types = map_antismash_products(best_a[3], prod_to_type)
            if not a_types:
                continue
            pairs.append({
                "genome": g,
                "contig": vr[0],
                "v41_start": int(vr[1]),
                "v41_end": int(vr[2]),
                "antismash_products": best_a[3],
                "antismash_types": a_types,
                "antismash_primary_type": a_types[0],
                "defrag_v2_type": dv2_type,
                "defrag_v2_scores": [round(s, 4) for s in dv2_scores],
                "iou": round(best_iou, 3),
            })

    print(f"[pairs] matched: {len(pairs)}, unmatched v4.1 regions: {unmatched}",
          flush=True)

    cm = {a: {v: 0 for v in BGC_TYPES} for a in BGC_TYPES}
    top_any_hits = 0
    for pr in pairs:
        cm[pr["antismash_primary_type"]][pr["defrag_v2_type"]] += 1
        if pr["defrag_v2_type"] in pr["antismash_types"]:
            top_any_hits += 1

    n_total = len(pairs)
    acc_top1 = sum(cm[t][t] for t in BGC_TYPES) / max(1, n_total)
    acc_top_any = top_any_hits / max(1, n_total)

    per_type = {}
    for t in BGC_TYPES:
        tp = cm[t][t]
        fp = sum(cm[a][t] for a in BGC_TYPES if a != t)
        fn = sum(cm[t][v] for v in BGC_TYPES if v != t)
        p_ = tp / max(1, tp + fp)
        r_ = tp / max(1, tp + fn)
        f1 = 2 * p_ * r_ / max(1e-9, p_ + r_)
        per_type[t] = {"n_antismash": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                       "precision": round(p_, 3), "recall": round(r_, 3),
                       "f1": round(f1, 3)}

    out = {
        "type_source": "defrag_v2_cnn_head",
        "n_matched_pairs": n_total,
        "iou_min": args.iou_min,
        "agg": args.agg,
        "accuracy_top1": round(acc_top1, 3),
        "accuracy_top_any": round(acc_top_any, 3),
        "confusion_matrix": cm,
        "per_type": per_type,
        "pairs_sample": pairs[:50],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    print(f"\n[result] matched pairs: {n_total}  |  top-1 acc: {acc_top1:.3f}  |  "
          f"top-any acc: {acc_top_any:.3f}")
    print(f"\n[confusion matrix]  rows=antiSMASH primary  cols=defrag_v2 argmax  agg={args.agg}")
    print(f"           {'  '.join(f'{t[:6]:>6}' for t in BGC_TYPES)}")
    for a in BGC_TYPES:
        row = "  ".join(f"{cm[a][v]:>6}" for v in BGC_TYPES)
        total = sum(cm[a].values())
        print(f"{a[:9]:>9}  {row}   (n={total})")
    print("\n[per-type]")
    for t in BGC_TYPES:
        s = per_type[t]
        print(f"  {t:<12} n_antiSMASH={s['n_antismash']:>4}  "
              f"P={s['precision']:.2f}  R={s['recall']:.2f}  F1={s['f1']:.2f}")
    print(f"\n[done] {args.out_json}")


if __name__ == "__main__":
    main()
