#!/usr/bin/env python3
"""Apply a multi-layer LR probe (from lr_multilayer.py) to 9g multi-layer
features and emit per-genome predictions.csv compatible with honest_readout.py.

Inputs:
  --lr-npz       lr_{combo}.npz produced by scripts/train/lr_multilayer.py.
                 Contains W, b, combo (list of layer shorts), and per-layer
                 mean/std arrays (same standardization applied at train).
  --feat-dir     dir with {genome}.{L13,L20,L27}.npy + {genome}_windows.csv
                 produced by scripts/extract/multi_layer.py --source genome.
  --out-dir      output dir; one {genome}_predictions.csv per genome.

CSV schema matches scripts/inference/apply_lr.py exactly so existing readout
scripts work unchanged: contig,start,end,bgc_score,non_bgc_score,pred_class,probs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-npz", type=Path, required=True)
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variant-tag", default="multilayer")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.lr_npz)
    W = z["W"].astype(np.float32)
    b = float(z["b"])
    combo = list(z["combo"])
    scalers = {ln: {"mean": z[f"mean_{ln}"].astype(np.float32),
                    "std": z[f"std_{ln}"].astype(np.float32)} for ln in combo}
    print(f"[lr] combo={combo} W={W.shape} b={b:.4f}", flush=True)

    manifest = {"variant": args.variant_tag, "combo": combo, "genomes": {}}

    win_csvs = sorted(args.feat_dir.glob("*_windows.csv"))
    if not win_csvs:
        raise FileNotFoundError(f"no *_windows.csv in {args.feat_dir}")

    for win_path in win_csvs:
        stem = win_path.name.replace("_windows.csv", "")
        # Load only the layers this combo needs; standardize each, concat.
        wins = pd.read_csv(win_path)
        pieces = []
        for ln in combo:
            npy = args.feat_dir / f"{stem}.{ln}.npy"
            if not npy.exists():
                raise FileNotFoundError(npy)
            X = np.load(npy).astype(np.float32)
            X = (X - scalers[ln]["mean"]) / scalers[ln]["std"]
            pieces.append(X)
        X = np.concatenate(pieces, axis=1)
        if X.shape[0] != len(wins):
            raise RuntimeError(f"{stem}: feat rows {X.shape[0]} vs windows {len(wins)}")
        if X.shape[1] != W.shape[0]:
            raise RuntimeError(f"{stem}: feat dim {X.shape[1]} vs W dim {W.shape[0]}")

        logits = X @ W + b
        probs = stable_sigmoid(logits)
        non_bgc = (1.0 - probs).astype(np.float32)

        out_csv = args.out_dir / f"{stem}_predictions.csv"
        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["contig", "start", "end", "bgc_score",
                         "non_bgc_score", "pred_class", "probs"])
            for i, row in wins.iterrows():
                cls = "BGC" if probs[i] >= 0.5 else "non-BGC"
                wr.writerow([
                    row["contig"], int(row["start"]), int(row["end"]),
                    f"{probs[i]:.6f}", f"{non_bgc[i]:.6f}", cls,
                    f"[{probs[i]:.6f}, {non_bgc[i]:.6f}]",
                ])
        manifest["genomes"][stem] = {
            "n_windows": int(len(wins)),
            "csv": out_csv.name,
            "pos_rate_thr0.5": float((probs >= 0.5).mean()),
            "score_mean": float(probs.mean()),
            "score_p95": float(np.quantile(probs, 0.95)),
        }
        print(f"[{stem}] n={len(wins)} pos_rate(.5)={manifest['genomes'][stem]['pos_rate_thr0.5']:.3f} "
              f"mean={manifest['genomes'][stem]['score_mean']:.3f}", flush=True)

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
