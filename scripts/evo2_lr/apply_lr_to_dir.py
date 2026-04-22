#!/usr/bin/env python3
"""Apply a Baseline-A linear probe (W, b) to extracted 9-genome embeddings and
emit per-window prediction CSVs compatible with evaluate_nine_genome.py.

Input:
  --lr-npz       lr_{variant}.npz produced by scripts/evo2_lr/train_lr.py
  --emb-dir      dir with {genome}_embeddings.npy + {genome}_windows.csv
  --out-dir      prediction CSVs

CSV schema (matches phase4/stride8192_s0_merged/*_predictions.csv):
  contig,start,end,bgc_score,non_bgc_score,pred_class,probs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr-npz", type=Path, required=True)
    p.add_argument("--emb-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--variant-tag", default="lrA1",
                   help="label written into pred_class column (diagnostic only)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lr = np.load(args.lr_npz)
    W = lr["W"].astype(np.float32)
    b = float(lr["b"])
    print(f"[lr] W shape={W.shape} b={b:.4f}", flush=True)

    manifest = {"variant": args.variant_tag, "genomes": {}}

    for emb_path in sorted(args.emb_dir.glob("*_embeddings.npy")):
        genome_id = emb_path.stem.replace("_embeddings", "")
        win_path = args.emb_dir / f"{genome_id}_windows.csv"
        if not win_path.exists():
            print(f"[skip] no windows csv for {genome_id}", flush=True)
            continue

        emb = np.load(emb_path).astype(np.float32)
        wins = pd.read_csv(win_path)
        assert len(emb) == len(wins), f"{genome_id}: emb {len(emb)} vs wins {len(wins)}"

        logits = emb @ W + b
        probs = sigmoid(logits.astype(np.float64)).astype(np.float32)
        non_bgc = 1.0 - probs

        out_csv = args.out_dir / f"{genome_id}_predictions.csv"
        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["contig", "start", "end", "bgc_score", "non_bgc_score",
                         "pred_class", "probs"])
            for i, row in wins.iterrows():
                cls = "BGC" if probs[i] >= 0.5 else "non-BGC"
                wr.writerow([
                    row["contig"],
                    int(row["start"]),
                    int(row["end"]),
                    f"{probs[i]:.6f}",
                    f"{non_bgc[i]:.6f}",
                    cls,
                    f"[{probs[i]:.6f}, {non_bgc[i]:.6f}]",
                ])
        manifest["genomes"][genome_id] = {
            "n_windows": int(len(emb)),
            "csv": out_csv.name,
            "pos_rate_thr0.5": float((probs >= 0.5).mean()),
            "score_mean": float(probs.mean()),
            "score_p95": float(np.quantile(probs, 0.95)),
        }
        print(f"[{genome_id}] n={len(emb)} pos_rate(thr=0.5)={manifest['genomes'][genome_id]['pos_rate_thr0.5']:.3f} "
              f"mean={manifest['genomes'][genome_id]['score_mean']:.3f}",
              flush=True)

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
