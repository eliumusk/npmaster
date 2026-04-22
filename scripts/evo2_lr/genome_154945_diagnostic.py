#!/usr/bin/env python3
"""Stage 0 diagnostic: GCA_000154945.1 — the shared blind-spot genome.

All three systems (LR_A1, defrag_v2_ms10_s099, BGC-Prophet) score near 0
on this genome in the v3 night run. Before investing in any architectural
fix, figure out whether this is:

  (a) Data/GT issue    — antiSMASH GT disagrees with true BGC content
                         (truncated contigs, weird products, etc.)
  (b) Evo2 signal gap  — per-window BGC scores are low at the GT locations
                         (foundation model can't see these BGCs)
  (c) Decoder issue    — scores fire correctly but decoder with the chosen
                         OP misses them (threshold / min_support issue)
  (d) Window boundary  — GT regions are tiny/contig-edge and our window
                         size can't resolve them

Outputs: docs/reports/genome_154945_diagnostic.md (markdown report)
         artifacts: distribution plot (ascii), per-region score traces

Inputs:
  --preds-csv    /data/.../9g_preds_lrA1/GCA_000154945.1_predictions.csv
  --gt-json      /data/syh/benchmark/nine_genome/ground_truth.json
  --compare-preds optional comma-sep list of other methods to cross-check
                  e.g. 9g_preds_lrA1_w8192:w8192,9g_preds_ctxCNN:ctx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENOME = "GCA_000154945.1"


def load_preds(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["bgc_score"] = df["bgc_score"].astype(float)
    return df


def window_score_at_pos(df: pd.DataFrame, contig: str, s: int, e: int) -> pd.DataFrame:
    """Return windows overlapping region (contig, s, e)."""
    mask = (df["contig"] == contig) & (df["end"] > s) & (df["start"] < e)
    return df[mask].copy()


def ascii_hist(vals: np.ndarray, bins: int = 20, label: str = "") -> str:
    if len(vals) == 0:
        return f"{label}: (empty)\n"
    h, edges = np.histogram(vals, bins=bins, range=(0, 1))
    mx = h.max() if h.max() > 0 else 1
    out = [f"{label} (n={len(vals)}, min={vals.min():.3f}, max={vals.max():.3f}, "
           f"mean={vals.mean():.3f}, median={np.median(vals):.3f})"]
    for i, c in enumerate(h):
        bar = "#" * int(30 * c / mx)
        out.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}]  {c:6d}  {bar}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preds-csv", type=Path, required=True)
    p.add_argument("--gt-json", type=Path, required=True)
    p.add_argument("--antismash-dir", type=Path, default=None,
                   help="optional: dir containing per-genome antiSMASH JSON")
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--compare-preds", type=str, default="",
                   help="comma-separated: path:label pairs")
    args = p.parse_args()

    gt_data = json.loads(args.gt_json.read_text())["ground_truth"]
    gts = gt_data.get(GENOME, [])
    print(f"[load] GT regions for {GENOME}: {len(gts)}")
    for contig, s, e in gts:
        print(f"  {contig}:{s}-{e}  len={e-s}")

    df = load_preds(args.preds_csv)
    print(f"[load] preds: {len(df)} windows across "
          f"{df['contig'].nunique()} contigs; contigs:"
          f" {df['contig'].unique()[:5].tolist()}...")

    overall = df["bgc_score"].to_numpy()
    print(ascii_hist(overall, label="overall bgc_score"))

    # Per-region detail
    region_reports = []
    gt_scores_flat: list[float] = []
    for contig, s, e in gts:
        sub = window_score_at_pos(df, contig, s, e)
        arr = sub["bgc_score"].to_numpy()
        gt_scores_flat.extend(arr.tolist())
        topk = sub.sort_values("bgc_score", ascending=False).head(5)
        region_reports.append({
            "contig": contig, "start": s, "end": e, "length": e - s,
            "n_windows": int(len(sub)),
            "mean_score": float(arr.mean()) if len(arr) else 0.0,
            "max_score": float(arr.max()) if len(arr) else 0.0,
            "p90_score": float(np.percentile(arr, 90)) if len(arr) else 0.0,
            "top5_scores": topk["bgc_score"].round(3).tolist(),
        })

    # Non-GT windows for comparison
    # For each pred window, mark whether it overlaps any GT
    is_in_gt = np.zeros(len(df), dtype=bool)
    for contig, s, e in gts:
        mask = (df["contig"] == contig) & (df["end"] > s) & (df["start"] < e)
        is_in_gt |= mask.to_numpy()

    gt_arr = df.loc[is_in_gt, "bgc_score"].to_numpy()
    nongt_arr = df.loc[~is_in_gt, "bgc_score"].to_numpy()

    # High-scoring non-GT windows (potential FPs or missed true BGCs)
    top_nongt = df.loc[~is_in_gt].sort_values("bgc_score", ascending=False).head(15)

    # Cross-check with other methods
    compare_data = {}
    for entry in args.compare_preds.split(","):
        entry = entry.strip()
        if not entry:
            continue
        path, label = entry.split(":")
        p2 = Path(path) / f"{GENOME}_predictions.csv"
        if not p2.exists():
            continue
        df2 = load_preds(p2)
        is_in_gt2 = np.zeros(len(df2), dtype=bool)
        for contig, s, e in gts:
            mask = (df2["contig"] == contig) & (df2["end"] > s) & (df2["start"] < e)
            is_in_gt2 |= mask.to_numpy()
        compare_data[label] = {
            "total_n": len(df2),
            "gt_scores": df2.loc[is_in_gt2, "bgc_score"].to_numpy(),
            "nongt_scores": df2.loc[~is_in_gt2, "bgc_score"].to_numpy(),
        }

    # Build markdown
    md = ["# GCA_000154945.1 Diagnostic", "",
          "Shared blind-spot across LR_A1, defrag_v2, BGC-Prophet (all near 0 F1).",
          "", "## GT regions", ""]
    md.append("| contig | start | end | length | n_windows | mean | max | p90 | top-5 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in region_reports:
        md.append(f"| {r['contig']} | {r['start']} | {r['end']} | {r['length']} | "
                  f"{r['n_windows']} | {r['mean_score']:.3f} | {r['max_score']:.3f} | "
                  f"{r['p90_score']:.3f} | {r['top5_scores']} |")
    md.append("")

    md.extend([
        "## Score distribution: GT windows vs non-GT windows", "",
        "```", ascii_hist(gt_arr, label="GT-overlap windows"), "", "",
        ascii_hist(nongt_arr, label="non-GT windows"), "```", "",
    ])

    md.append("## Top-15 high-scoring non-GT windows (possible FPs or missed-GT)")
    md.append("")
    md.append("| contig | start | end | bgc_score |")
    md.append("|---|---:|---:|---:|")
    for _, r in top_nongt.iterrows():
        md.append(f"| {r['contig']} | {r['start']} | {r['end']} | {r['bgc_score']:.3f} |")
    md.append("")

    # Cross-method
    if compare_data:
        md.append("## Cross-method comparison on this genome")
        md.append("")
        md.append("| method | n_win | GT mean | GT max | GT p90 | non-GT p99 |")
        md.append("|---|---:|---:|---:|---:|---:|")
        md.append(f"| LR_A1 (this) | {len(df)} | {gt_arr.mean():.3f} | "
                  f"{gt_arr.max() if len(gt_arr) else 0:.3f} | "
                  f"{np.percentile(gt_arr, 90) if len(gt_arr) else 0:.3f} | "
                  f"{np.percentile(nongt_arr, 99) if len(nongt_arr) else 0:.3f} |")
        for label, d in compare_data.items():
            g = d["gt_scores"]; n = d["nongt_scores"]
            md.append(f"| {label} | {d['total_n']} | "
                      f"{g.mean() if len(g) else 0:.3f} | "
                      f"{g.max() if len(g) else 0:.3f} | "
                      f"{np.percentile(g, 90) if len(g) else 0:.3f} | "
                      f"{np.percentile(n, 99) if len(n) else 0:.3f} |")
        md.append("")

    # Diagnosis summary
    md.append("## Diagnosis")
    md.append("")
    diag: list[str] = []
    if len(gt_arr) == 0:
        diag.append("- **GT regions have zero window coverage** — contig-boundary issue or contig naming mismatch.")
    else:
        gt_max = gt_arr.max()
        nongt_p99 = np.percentile(nongt_arr, 99) if len(nongt_arr) else 0
        if gt_max < 0.5:
            diag.append(f"- **(b) Evo2 signal gap**: max GT-window score = {gt_max:.3f}, well below 0.5. "
                        "Evo2 mean-pool does not see these BGCs at all. Decoder cannot recover.")
        elif gt_max < 0.9 and gt_max > nongt_p99:
            diag.append(f"- **(c) Decoder threshold too aggressive**: GT max {gt_max:.3f} is above non-GT p99 {nongt_p99:.3f} "
                        "but below production start threshold (~0.9–0.96). Lowering start threshold could recover recall.")
        else:
            diag.append(f"- **GT windows fire ({gt_max:.3f})** but may be sparse in count ({len(gt_arr)} windows across "
                        f"{len(gts)} GT regions = {len(gt_arr)/max(1,len(gts)):.1f} win/region). "
                        "Check min_support_windows sensitivity.")
    any_edge = any(r["n_windows"] < 2 for r in region_reports)
    if any_edge:
        diag.append("- **(d) Short-region coverage**: some GT regions have < 2 overlapping windows "
                    "(window size cap at this geometry).")
    for line in diag:
        md.append(line)
    md.append("")
    md.append("## Next actions")
    md.append("- If (b): requires a different layer, different pooling, or different foundation model; LR head is not the lever.")
    md.append("- If (c): lower start threshold on this genome only — but we don't know per-genome which to lower; consider learned calibration.")
    md.append("- If (d): short-window head (w=2048) is indicated.")

    args.out_md.write_text("\n".join(md))
    print(f"[done] wrote {args.out_md}")


if __name__ == "__main__":
    main()
