#!/usr/bin/env python3
"""Evaluate v4.1's per-type classification vs antiSMASH on 9-genome.

Pipeline:
  1. Load 9g w=8192 Evo2 embeddings (cached).
  2. Apply 7 per-type LR heads (`stage2/type_lr/lr_type_*.npz`) → per-window 7 scores.
  3. Regenerate v4.1 ensemble regions at frozen OPs (primary w=8192 + short w=4096, union-dedup).
  4. For each v4.1 region, aggregate type scores over its windows (mean + max) and
     take argmax over 7 BGC classes → predicted type.
  5. Load antiSMASH calls on 9-genome; for each v4.1 region, find best-IoU antiSMASH
     region on the same contig. If IoU ≥ --iou-min, pair them.
  6. Build `antismash_type → v4.1_type` confusion matrix + per-type P/R + overall
     accuracy. Also report top-any (does v4.1's top-1 match any of antiSMASH's
     listed product types when it's a hybrid call).

Caveats baked into the report:
  - Teacher-student fidelity test: both antiSMASH (gold here) and v4.1 training
    labels originate from antiSMASH. High accuracy = faithful type generalization,
    NOT "true novelty". Low accuracy = v4.1's type head drifted from teacher.
  - Product→type mapping is derived empirically from `bgc_annotations.csv`
    (majority vote per product string across training pool), so it reflects the
    same convention used during training.

Typical invocation (on server):
  python scripts/evo2_lr/eval_v41_classification_9genome.py \
      --emb-dir /data/muskliu/npmaster/experiments/evo2_first_principles/9g_embeddings_w8192_s2048 \
      --type-lr-dir /data/muskliu/npmaster/experiments/evo2_first_principles/stage2/type_lr \
      --v41-primary-preds /data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1_w8192 \
      --v41-short-preds /data/muskliu/npmaster/experiments/evo2_first_principles/short_bgc/9g_preds_lrA1_w4096 \
      --antismash-dir /data/syh/benchmark/nine_genome/results/antismash \
      --bgc-annotations /data/muskliu/npmaster/data/bgc_annotations.csv \
      --out-json experiments/classification_eval_9genome.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evo2_lr"))

from baselines.parse_antismash import parse_antismash_dir  # noqa: E402
from v4_1_multiscale_readout import (  # noqa: E402
    PRIMARY_OP as V41_PRIMARY_OP,
    SHORT_OP as V41_SHORT_OP,
    SHORT_LENGTH_CAP as V41_SHORT_LENGTH_CAP,
    IOU_DEDUP as V41_IOU_DEDUP,
    decode_dir as v41_decode_dir,
    union_dedup as v41_union_dedup,
)


BGC_TYPES = ["Alkaloid", "Terpene", "NRP", "Polyketide", "RiPP", "Saccharide", "Other"]


def build_product_to_type_map(annotations_csv: Path) -> dict[str, str]:
    """Empirically derive raw-product → canonical 7-class from training annotations.

    `bgc_types` in training data is ';'-separated when hybrid (e.g. 'NRP;Polyketide').
    We take the first type listed per row (the canonical primary), group by
    `bgc_product` (raw antiSMASH product string), and majority-vote.
    """
    df = pd.read_csv(annotations_csv, usecols=["bgc_types", "bgc_product"])
    df = df.dropna(subset=["bgc_types", "bgc_product"])
    df["primary_type"] = df["bgc_types"].str.split(";").str[0]
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, r in df.iterrows():
        counts[str(r["bgc_product"]).strip()][r["primary_type"]] += 1
    return {prod: c.most_common(1)[0][0] for prod, c in counts.items()}


def map_antismash_products(products_str: str, prod_to_type: dict[str, str]) -> list[str]:
    """Map an antiSMASH area's `products` string (comma-joined) to unique 7-classes.

    Unknown products fall back to 'Other' (rather than drop) so nothing is silently
    discarded — the report will flag how many unknowns there were.
    """
    out = []
    for p in (products_str or "").split(","):
        p = p.strip()
        if not p:
            continue
        t = prod_to_type.get(p, "Other")
        if t not in out:
            out.append(t)
    return out


def load_type_lrs(type_lr_dir: Path) -> dict[str, dict]:
    """Load the 7 per-type LR heads. Each .npz contains at least 'W' and 'b'."""
    lrs = {}
    for t in BGC_TYPES:
        p = type_lr_dir / f"lr_type_{t}.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing per-type LR: {p}")
        z = np.load(p)
        # Support both single-arr (2,D) and W/b layouts; we'll detect at use.
        lrs[t] = dict(z)
    return lrs


def type_lr_forward(emb: np.ndarray, weights: dict) -> np.ndarray:
    """Apply a binary LR to (N, D) embeddings → probability-of-positive (N,).

    Handles both sklearn-style (coef_ / intercept_ keys) and npz-custom
    (W / b keys) weight formats.
    """
    if "coef_" in weights and "intercept_" in weights:
        W = weights["coef_"].reshape(-1)            # (D,)
        b = float(weights["intercept_"].ravel()[0])
    elif "W" in weights and "b" in weights:
        W = weights["W"].reshape(-1)
        b = float(np.ravel(weights["b"])[0])
    elif "w" in weights and "b" in weights:
        W = weights["w"].reshape(-1)
        b = float(np.ravel(weights["b"])[0])
    else:
        raise KeyError(f"unknown LR npz layout; keys={list(weights.keys())}")
    z = emb @ W + b                                   # (N,)
    return 1.0 / (1.0 + np.exp(-z))


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def iou_pair(a, b) -> float:
    """a, b are (contig, start, end, ...)."""
    if a[0] != b[0]:
        return 0.0
    ov = overlap_len(a[1], a[2], b[1], b[2])
    if ov == 0:
        return 0.0
    return ov / max(1, max(a[2], b[2]) - min(a[1], b[1]))


def find_v41_type_for_region(
    region, emb_all, win_df, type_probs_cache, agg="mean",
):
    """For a v4.1 region, aggregate per-type scores over its windows, return ranked types.

    type_probs_cache: (N_windows, 7) array in same order as win_df rows.
    """
    contig, rs, re_ = region[0], int(region[1]), int(region[2])
    in_region = (
        (win_df["contig"].to_numpy() == contig)
        & (win_df["end"].to_numpy() > rs)
        & (win_df["start"].to_numpy() < re_)
    )
    if not in_region.any():
        return None, None
    scores = type_probs_cache[in_region]             # (k, 7)
    if agg == "mean":
        per_type = scores.mean(axis=0)
    elif agg == "max":
        per_type = scores.max(axis=0)
    else:
        raise ValueError(agg)
    order = np.argsort(-per_type)                    # descending
    return BGC_TYPES[int(order[0])], per_type.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", type=Path, required=True)
    p.add_argument("--type-lr-dir", type=Path, required=True)
    p.add_argument("--v41-primary-preds", type=Path, required=True)
    p.add_argument("--v41-short-preds", type=Path, required=True)
    p.add_argument("--antismash-dir", type=Path, required=True)
    p.add_argument("--bgc-annotations", type=Path, required=True,
                   help="Training bgc_annotations.csv for product→type mapping")
    p.add_argument("--gt-json", type=Path, default=Path("/data/syh/benchmark/nine_genome/ground_truth.json"))
    p.add_argument("--iou-min", type=float, default=0.3)
    p.add_argument("--agg", choices=["mean", "max"], default="mean")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    print("[load] product→type map from training annotations", flush=True)
    prod_to_type = build_product_to_type_map(args.bgc_annotations)
    print(f"  {len(prod_to_type)} unique products → 7-class", flush=True)

    print("[load] 7 per-type LR heads", flush=True)
    type_lrs = load_type_lrs(args.type_lr_dir)

    print("[decode] v4.1 primary regions", flush=True)
    primary = v41_decode_dir(args.v41_primary_preds, V41_PRIMARY_OP)
    print(f"  {sum(len(v) for v in primary.values())} primary regions", flush=True)

    print("[decode] v4.1 short regions (L<10kb)", flush=True)
    short = v41_decode_dir(args.v41_short_preds, V41_SHORT_OP)
    short_filt = {g: [r for r in regs if (r[2] - r[1]) < V41_SHORT_LENGTH_CAP]
                  for g, regs in short.items()}
    print(f"  {sum(len(v) for v in short_filt.values())} short regions kept", flush=True)

    print("[ensemble] union-dedup at IoU=%.2f" % V41_IOU_DEDUP, flush=True)
    genomes = sorted(set(primary) | set(short_filt))
    v41 = {
        g: v41_union_dedup(primary.get(g, []), short_filt.get(g, []), V41_IOU_DEDUP)
        for g in genomes
    }
    print(f"  {sum(len(v) for v in v41.values())} ensemble regions", flush=True)

    print("[load] antiSMASH 9-genome", flush=True)
    antis = parse_antismash_dir(args.antismash_dir, genome_ids=genomes)
    for g in genomes:
        n_unknown = 0
        for r in antis.get(g, []):
            for p_ in (r["predicted_type"] or "").split(","):
                p_ = p_.strip()
                if p_ and p_ not in prod_to_type:
                    n_unknown += 1
        if n_unknown:
            print(f"  {g}: {n_unknown} unknown antiSMASH products (→ 'Other' fallback)",
                  flush=True)

    # Pair up v4.1 ↔ antiSMASH regions; compute v4.1 type per matched region
    pairs: list[dict] = []
    unmatched_v41 = 0
    for g in genomes:
        emb_p = args.emb_dir / f"{g}_embeddings.npy"
        win_p = args.emb_dir / f"{g}_windows.csv"
        if not (emb_p.exists() and win_p.exists()):
            print(f"[skip] {g} missing cached emb", flush=True)
            continue
        emb = np.load(emb_p).astype(np.float32)      # (N, 4096)
        win_df = pd.read_csv(win_p)
        # Score all windows with the 7 type LRs
        type_probs = np.stack(
            [type_lr_forward(emb, type_lrs[t]) for t in BGC_TYPES], axis=1
        )  # (N, 7)

        a_regs = [(r["contig_id"], int(r["start_bp"]), int(r["end_bp"]),
                   r["predicted_type"]) for r in antis.get(g, [])]

        for vr in v41.get(g, []):
            # best-IoU antiSMASH region on same contig
            best_iou = 0.0
            best_a = None
            for ar in a_regs:
                iou = iou_pair(vr, ar)
                if iou > best_iou:
                    best_iou = iou
                    best_a = ar
            v_type, v_scores = find_v41_type_for_region(vr, emb, win_df, type_probs, agg=args.agg)
            if best_a is None or best_iou < args.iou_min:
                unmatched_v41 += 1
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
                "v41_type": v_type,
                "v41_scores": [round(s, 4) for s in v_scores],
                "iou": round(best_iou, 3),
            })

    print(f"[pairs] matched: {len(pairs)}, unmatched v4.1 regions: {unmatched_v41}",
          flush=True)

    # Confusion matrix: rows = antiSMASH primary type, cols = v4.1 predicted type
    cm = {a: {v: 0 for v in BGC_TYPES} for a in BGC_TYPES}
    top_any_hits = 0
    for pr in pairs:
        cm[pr["antismash_primary_type"]][pr["v41_type"]] += 1
        if pr["v41_type"] in pr["antismash_types"]:
            top_any_hits += 1

    n_total = len(pairs)
    accuracy_top1 = sum(cm[t][t] for t in BGC_TYPES) / max(1, n_total)
    accuracy_top_any = top_any_hits / max(1, n_total)

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
        "n_v41_regions": sum(len(v) for v in v41.values()),
        "n_unmatched_v41": unmatched_v41,
        "n_matched_pairs": n_total,
        "iou_min": args.iou_min,
        "agg": args.agg,
        "accuracy_top1": round(accuracy_top1, 3),
        "accuracy_top_any": round(accuracy_top_any, 3),
        "confusion_matrix": cm,
        "per_type": per_type,
        "pairs_sample": pairs[:50],  # keep a sample for inspection
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    # Pretty print confusion matrix
    print(f"\n[result] matched pairs: {n_total}  |  top-1 acc: {accuracy_top1:.3f}  |  "
          f"top-any acc: {accuracy_top_any:.3f}")
    print(f"\n[confusion matrix]  rows=antiSMASH primary  cols=v4.1 argmax  agg={args.agg}")
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
