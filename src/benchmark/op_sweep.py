#!/usr/bin/env python3
"""Stage 0 step 2: OP sweep on internal-val, single readout on 9-genome, bootstrap CI.

Protocol (no 9-genome data touches OP selection):
  1. Sweep decoder OP grid on internal-val region metrics (region-level
     dedup_f1 + bench_f1 + strict_f1).
  2. Pick best OP by internal-val dedup_f1.
  3. Apply that single OP to 9-genome per-window pred CSVs → one readout.
  4. Bootstrap 1000x over 9-genome genomes for 95% CI on each metric.
  5. Per-genome table + per-length-bucket recall.

Inputs:
  --internal-val-dir   produced by stage0_build_internal_val.py
  --nine-genome-preds  /data/muskliu/npmaster/experiments/evo2_first_principles/9g_preds_lrA1 (or ..._w8192)
  --nine-genome-gt     /data/syh/benchmark/nine_genome/ground_truth.json
  --grid               JSON list of dicts {start, extend, ms, max_gap}; or --auto for default grid
  --out-dir            where to write stage0_results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from decoding.postprocess import WindowScore, decode_regions_hysteresis  # noqa: E402
# Use the canonical 9-genome region evaluator so metrics are comparable 1:1
# with defrag_v2_ms10_s099 / operating_points.yaml.
from benchmark.nine_genome import eval_regions  # noqa: E402


def regions_metrics(preds: list[tuple[str, int, int]],
                    gts: list[tuple[str, int, int]]) -> dict:
    """Wrap evaluate_nine_genome.eval_regions across bench/strict/loose modes."""
    out: dict = {}
    for mode in ("bench", "strict", "loose"):
        r = eval_regions(gts, preds, mode=mode, min_overlap=0.5)
        if mode == "bench":
            out["n_gt"] = r["n_gt"]
            out["n_pred"] = r["n_pred"]
            out["tp_recall"] = r["tp_recall"]
            out["tp_precision"] = r["tp_precision"]
            out["bench_precision"] = r["precision"]
            out["bench_recall"] = r["recall"]
            out["bench_f1"] = r["f1"]
            out["dedup_f1"] = r["fragmentation"]["dedup_f1"]
            out["dedup_precision"] = r["fragmentation"]["dedup_precision"]
            out["unique_gt_precision_tp"] = r["fragmentation"]["unique_gt_precision_tp"]
            out["duplicate_precision_tp"] = r["fragmentation"]["duplicate_precision_tp"]
            out["max_preds_per_gt"] = r["fragmentation"]["max_preds_per_gt"]
        elif mode == "strict":
            out["strict_precision"] = r["precision"]
            out["strict_recall"] = r["recall"]
            out["strict_f1"] = r["f1"]
            out["strict_dedup_f1"] = r["fragmentation"]["dedup_f1"]
        elif mode == "loose":
            out["loose_precision"] = r["precision"]
            out["loose_recall"] = r["recall"]
            out["loose_f1"] = r["f1"]
    return out


_WINDOW_CACHE: dict[str, list[WindowScore]] = {}


def load_windows(csv_path: Path) -> list[WindowScore]:
    key = str(csv_path)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]
    df = pd.read_csv(csv_path)
    wins = [
        WindowScore(str(c), int(s), int(e), float(bs))
        for c, s, e, bs in zip(df["contig"].values, df["start"].values,
                                df["end"].values, df["bgc_score"].values)
    ]
    _WINDOW_CACHE[key] = wins
    return wins


def decode_genome(csv_path: Path, start: float, extend: float, ms: int,
                  max_gap: int) -> list[tuple[str, int, int]]:
    wins = load_windows(csv_path)
    regions = decode_regions_hysteresis(
        wins, start_threshold=start, extend_threshold=extend,
        min_support_windows=ms, max_gap=max_gap,
    )
    return [(reg[0], int(reg[1]), int(reg[2])) for reg in regions]


def pool_from_per_genome(per_genome: dict[str, dict]) -> dict:
    """Aggregate per-genome metrics into pooled counts. Assumes no cross-genome
    matching (we never pool regions with naked contig names across genomes)."""
    t_rec = 0; t_prec = 0; n_gt = 0; n_pred = 0
    u_gt = 0; d_prec = 0; max_ppg = 0
    s_rec = 0; s_prec = 0; l_rec = 0; l_prec = 0
    s_u_gt = 0
    for g, m in per_genome.items():
        if "error" in m:
            continue
        t_rec += m["tp_recall"]
        t_prec += m["tp_precision"]
        n_gt += m["n_gt"]
        n_pred += m["n_pred"]
        u_gt += m["unique_gt_precision_tp"]
        d_prec += m["duplicate_precision_tp"]
        max_ppg = max(max_ppg, m["max_preds_per_gt"])
        # strict / loose — eval_regions.fragmentation only carries bench dedup;
        # strict/loose P/R we reconstruct from per-genome stored P/R weighted.
        # Since we have per-genome tp_precision for strict/loose is NOT stored,
        # fallback to recomputing strict/loose via per-genome then summing.
    recall = t_rec / max(1, n_gt)
    precision = t_prec / max(1, n_pred)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    dedup_precision = u_gt / max(1, n_pred)
    dedup_f1 = 2 * dedup_precision * recall / max(1e-12, dedup_precision + recall)
    return {
        "n_gt": n_gt, "n_pred": n_pred,
        "tp_recall": t_rec, "tp_precision": t_prec,
        "bench_precision": precision, "bench_recall": recall, "bench_f1": f1,
        "dedup_precision": dedup_precision, "dedup_f1": dedup_f1,
        "unique_gt_precision_tp": u_gt, "duplicate_precision_tp": d_prec,
        "max_preds_per_gt": max_ppg,
    }


def eval_op(preds_dir: Path, gt: dict, start: float, extend: float, ms: int,
            max_gap: int) -> dict:
    """Per-genome decode + per-genome metrics; aggregate into pooled stats.

    Aggregation is exact under the convention that contigs are genome-local
    (no naked cross-genome pooling), which is the natural case here.
    """
    per_genome: dict[str, dict] = {}
    # For strict/loose pooling: collect per-genome TP_recall/TP_precision
    strict_t_rec = 0; strict_t_prec = 0
    loose_t_rec = 0; loose_t_prec = 0
    for genome, gts in gt.items():
        csv_path = preds_dir / f"{genome}_predictions.csv"
        if not csv_path.exists():
            per_genome[genome] = {"error": "no predictions"}
            continue
        preds = decode_genome(csv_path, start, extend, ms, max_gap)
        m = regions_metrics(preds, list(gts))
        per_genome[genome] = m
        # strict/loose: reconstruct per-genome counts from stored precision/recall
        # regions_metrics already gives strict_precision/recall etc.
        strict_t_rec += int(round(m["strict_recall"] * m["n_gt"]))
        strict_t_prec += int(round(m["strict_precision"] * m["n_pred"]))
        loose_t_rec += int(round(m["loose_recall"] * m["n_gt"]))
        loose_t_prec += int(round(m["loose_precision"] * m["n_pred"]))

    pooled = pool_from_per_genome(per_genome)
    # Pooled strict/loose
    n_gt = pooled["n_gt"]; n_pred = pooled["n_pred"]
    sr = strict_t_rec / max(1, n_gt); sp = strict_t_prec / max(1, n_pred)
    sf = 2 * sp * sr / max(1e-12, sp + sr)
    lr = loose_t_rec / max(1, n_gt); lp = loose_t_prec / max(1, n_pred)
    lf = 2 * lp * lr / max(1e-12, lp + lr)
    pooled["strict_precision"] = sp
    pooled["strict_recall"] = sr
    pooled["strict_f1"] = sf
    pooled["loose_precision"] = lp
    pooled["loose_recall"] = lr
    pooled["loose_f1"] = lf
    return {
        "op": {"start": start, "extend": extend, "ms": ms, "max_gap": max_gap},
        "pooled": pooled,
        "per_genome": per_genome,
    }


def default_grid() -> list[dict]:
    starts = [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    # absolute extend thresholds — not fractional — so we cover (s, e) with e < s
    # Generate pairs where extend <= start. Dense grid for [0.5, start] in 0.02 steps.
    mss = [1, 2, 3, 5]
    max_gaps = [0]
    out = []
    for s in starts:
        es = sorted({round(e, 2) for e in np.arange(0.50, s + 0.001, 0.04)}.union(
            {round(s * f, 2) for f in (0.6, 0.7, 0.8, 0.9, 1.0)}
        ))
        es = [round(e, 2) for e in es if 0.3 <= e <= s]
        for e in es:
            for ms in mss:
                for mg in max_gaps:
                    out.append({"start": s, "extend": e, "ms": ms, "max_gap": mg})
    # dedupe
    seen = set()
    u = []
    for op in out:
        k = (op["start"], op["extend"], op["ms"], op["max_gap"])
        if k in seen:
            continue
        seen.add(k)
        u.append(op)
    return u


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--internal-val-dir", type=Path, required=True)
    p.add_argument("--nine-genome-preds", type=Path, required=True)
    p.add_argument("--nine-genome-gt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grid-json", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10,
                   help="how many top OPs to report and re-eval on 9-genome")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--skip-internal-sweep", action="store_true",
                   help="if internal-val sweep already done, just read best_op.json")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    iv_gt = json.loads((args.internal_val_dir / "ground_truth.json").read_text())["ground_truth"]
    ng_gt = json.loads(args.nine_genome_gt.read_text())["ground_truth"]
    print(f"[load] internal-val: {len(iv_gt)} genomes, {sum(len(v) for v in iv_gt.values())} regions")
    print(f"[load] 9-genome:     {len(ng_gt)} genomes, {sum(len(v) for v in ng_gt.values())} regions")

    grid = default_grid()
    if args.grid_json:
        grid = json.loads(args.grid_json.read_text())
    print(f"[grid] {len(grid)} operating points")

    # --- internal-val sweep ---
    sweep_path = args.out_dir / "internal_val_sweep.json"
    if args.skip_internal_sweep and sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
    else:
        sweep = []
        for i, op in enumerate(grid):
            r = eval_op(args.internal_val_dir, iv_gt,
                        op["start"], op["extend"], op["ms"], op["max_gap"])
            sweep.append({
                "op": op,
                "dedup_f1": r["pooled"]["dedup_f1"],
                "strict_f1": r["pooled"]["strict_f1"],
                "bench_precision": r["pooled"]["bench_precision"],
                "bench_recall": r["pooled"]["bench_recall"],
                "n_pred": r["pooled"]["n_pred"],
            })
            if (i + 1) % 10 == 0 or i == len(grid) - 1:
                print(f"[sweep] {i+1}/{len(grid)}  best so far: "
                      f"{max(sweep, key=lambda x: x['dedup_f1'])}")
        sweep.sort(key=lambda x: x["dedup_f1"], reverse=True)
        sweep_path.write_text(json.dumps(sweep, indent=2))

    top = sweep[: args.top_k]
    print("\n[top-k on internal-val]")
    for r in top:
        print(f"  s={r['op']['start']} e={r['op']['extend']} ms={r['op']['ms']:2d} "
              f"→ iv_dedup={r['dedup_f1']:.4f} iv_strict={r['strict_f1']:.4f} "
              f"P={r['bench_precision']:.3f} R={r['bench_recall']:.3f} n={r['n_pred']}")

    best_op = top[0]["op"]
    (args.out_dir / "best_op.json").write_text(json.dumps({
        "best_op": best_op,
        "selected_on": "internal_val_dedup_f1",
        "internal_val_dedup_f1": top[0]["dedup_f1"],
        "internal_val_strict_f1": top[0]["strict_f1"],
    }, indent=2))

    # --- single readout on 9-genome ---
    print(f"\n[readout] applying best OP {best_op} to 9-genome")
    r9 = eval_op(args.nine_genome_preds, ng_gt,
                 best_op["start"], best_op["extend"], best_op["ms"], best_op["max_gap"])

    # Bootstrap per-genome: stash pred/gt region lists; resample genome-level.
    per_genome_regions: dict[str, dict] = {}
    for genome, gts in ng_gt.items():
        csv_path = args.nine_genome_preds / f"{genome}_predictions.csv"
        if not csv_path.exists():
            continue
        preds = decode_genome(csv_path, best_op["start"], best_op["extend"],
                              best_op["ms"], best_op["max_gap"])
        per_genome_regions[genome] = {
            "preds": [(f"{genome}|{c}", s, e) for c, s, e in preds],
            "gts": [(f"{genome}|{c}", s, e) for c, s, e in gts],
        }

    # Precompute per-genome metrics; bootstrap resamples genomes and aggregates.
    per_genome_full = {g: regions_metrics(v["preds"], v["gts"])
                       for g, v in per_genome_regions.items()}

    rng = np.random.default_rng(0)
    genomes = list(per_genome_regions.keys())
    keys = ("dedup_f1", "strict_f1", "bench_f1", "bench_precision", "bench_recall",
            "strict_precision", "strict_recall")
    samples = {k: [] for k in keys}
    for _ in range(args.bootstrap):
        sample = rng.choice(genomes, size=len(genomes), replace=True)
        # Pool per-genome TP/FP/FN across resampled genomes
        sub = {i: per_genome_full[g] for i, g in enumerate(sample)}
        p = pool_from_per_genome(sub)
        # strict/loose pooled
        n_gt = p["n_gt"]; n_pred = p["n_pred"]
        s_rec = sum(per_genome_full[g]["strict_recall"] * per_genome_full[g]["n_gt"]
                    for g in sample) / max(1, n_gt)
        s_prec = sum(per_genome_full[g]["strict_precision"] * per_genome_full[g]["n_pred"]
                     for g in sample) / max(1, n_pred)
        s_f1 = 2 * s_prec * s_rec / max(1e-12, s_prec + s_rec)
        samples["dedup_f1"].append(p["dedup_f1"])
        samples["bench_f1"].append(p["bench_f1"])
        samples["bench_precision"].append(p["bench_precision"])
        samples["bench_recall"].append(p["bench_recall"])
        samples["strict_f1"].append(s_f1)
        samples["strict_precision"].append(s_prec)
        samples["strict_recall"].append(s_rec)
    ci: dict[str, dict] = {}
    for k in keys:
        s = np.asarray(samples[k])
        ci[k] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "ci_lo": float(np.quantile(s, 0.025)),
            "ci_hi": float(np.quantile(s, 0.975)),
        }

    # --- also apply top-k to 9-genome for reporting (NOT for selection) ---
    top_readouts: list[dict] = []
    for rec in top:
        op = rec["op"]
        r = eval_op(args.nine_genome_preds, ng_gt,
                    op["start"], op["extend"], op["ms"], op["max_gap"])
        top_readouts.append({
            "op": op,
            "iv_dedup_f1": rec["dedup_f1"],
            "iv_strict_f1": rec["strict_f1"],
            "9g_dedup_f1": r["pooled"]["dedup_f1"],
            "9g_strict_f1": r["pooled"]["strict_f1"],
            "9g_bench_precision": r["pooled"]["bench_precision"],
            "9g_bench_recall": r["pooled"]["bench_recall"],
            "9g_n_pred": r["pooled"]["n_pred"],
        })

    out = {
        "best_op": best_op,
        "selected_on": "internal_val_dedup_f1 (no 9-genome touched)",
        "internal_val": {
            "dedup_f1": top[0]["dedup_f1"],
            "strict_f1": top[0]["strict_f1"],
        },
        "nine_genome_readout": {
            "pooled": r9["pooled"],
            "per_genome": per_genome_full,
            "bootstrap_ci_95": ci,
            "n_bootstrap": args.bootstrap,
        },
        "top_k_readouts": top_readouts,
    }
    (args.out_dir / "stage0_results.json").write_text(json.dumps(out, indent=2))

    # Pretty print summary
    print("\n===== STAGE 0 HONEST READOUT =====")
    print(f"Best OP (from internal-val): s={best_op['start']} e={best_op['extend']} "
          f"ms={best_op['ms']} max_gap={best_op['max_gap']}")
    pool = r9["pooled"]
    print(f"9-genome pooled metrics:")
    print(f"  dedup_F1  = {pool['dedup_f1']:.4f}   [CI95: {ci['dedup_f1']['ci_lo']:.4f}, "
          f"{ci['dedup_f1']['ci_hi']:.4f}]")
    print(f"  strict_F1 = {pool['strict_f1']:.4f}  [CI95: {ci['strict_f1']['ci_lo']:.4f}, "
          f"{ci['strict_f1']['ci_hi']:.4f}]")
    print(f"  bench_P   = {pool['bench_precision']:.4f}  [CI95: "
          f"{ci['bench_precision']['ci_lo']:.4f}, {ci['bench_precision']['ci_hi']:.4f}]")
    print(f"  bench_R   = {pool['bench_recall']:.4f}  [CI95: "
          f"{ci['bench_recall']['ci_lo']:.4f}, {ci['bench_recall']['ci_hi']:.4f}]")
    print(f"  n_pred = {pool['n_pred']}")
    print(f"\nReference (same geometry): defrag_v2_ms10_s099 all9_dedup_f1 = 0.4385")
    print(f"\nArtifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
