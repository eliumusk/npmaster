#!/usr/bin/env python3
"""Train four LR probes from sharded multi-layer Evo2 features.

Inputs (server-side):
  --shard-dir   dir with shard{NN}.{L13,L20,L27}.npy + shard{NN}_windows.csv
                produced by scripts/extract/multi_layer.py --source sliced
  --master-csv  windows_metadata_hardneg.csv (provides label, split,
                hard_negative, overlaps_truncated_bgc per window_id)

Trains four logistic-regression probes — L13-only, L20-only, L27-only,
concat(L13,L20,L27) — using sklearn with class_weight="balanced", C=1.0,
500 max_iter. All four are A1-style: positives vs negatives, drop ambiguous.

Each layer is per-dimension standardized on the train split; the scaler
(mean, std) is saved alongside W, b so inference applies the same transform.
The concat probe standardizes EACH layer independently, then concatenates,
to make every layer contribute equally to L2 regularization.

Outputs (per layer combo, into --out-dir):
  lr_{combo}.npz       W (D,), b (), per-layer mean/std arrays
  metrics_{combo}.json val AUROC / AP / threshold-0.5 P,R,F1
  config.json          paths, n_train, n_val, layer combo list
  summary.json         all four val metrics in one file (for downstream sweep)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


LAYER_SHORT = ("L13", "L20", "L27")
COMBOS: dict[str, tuple[str, ...]] = {
    "L13": ("L13",),
    "L20": ("L20",),
    "L27": ("L27",),
    "concat": ("L13", "L20", "L27"),
}


def load_shards(shard_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Concat all shard{NN}.{layer}.npy + shard{NN}_windows.csv in NN order.

    Returns (windows_df, {layer: (N, 4096) fp32}). NaN/zero rows raise.
    """
    shard_csvs = sorted(shard_dir.glob("shard*_windows.csv"))
    if not shard_csvs:
        raise FileNotFoundError(f"no shard*_windows.csv in {shard_dir}")
    print(f"[load] {len(shard_csvs)} shards in {shard_dir}", flush=True)

    dfs: list[pd.DataFrame] = []
    layer_blocks: dict[str, list[np.ndarray]] = {ln: [] for ln in LAYER_SHORT}
    for csv_path in shard_csvs:
        stem = csv_path.name.replace("_windows.csv", "")
        df = pd.read_csv(csv_path)
        if "window_id" not in df.columns:
            raise RuntimeError(f"{csv_path} missing window_id column")
        n = len(df)
        for ln in LAYER_SHORT:
            npy_path = shard_dir / f"{stem}.{ln}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(npy_path)
            arr = np.load(npy_path)
            if arr.shape[0] != n:
                raise RuntimeError(f"{npy_path} has {arr.shape[0]} rows but csv has {n}")
            layer_blocks[ln].append(arr)
        dfs.append(df)

    windows_df = pd.concat(dfs, ignore_index=True)
    layers = {ln: np.concatenate(layer_blocks[ln], axis=0) for ln in LAYER_SHORT}
    for ln, X in layers.items():
        if not np.isfinite(X).all():
            raise RuntimeError(f"non-finite values in {ln}")
        zero_rows = int((X.sum(axis=1) == 0).sum())
        if zero_rows > 0:
            print(f"[warn] {ln}: {zero_rows} all-zero rows (likely failed forwards)",
                  flush=True)
    print(f"[load] windows={len(windows_df)} L13={layers['L13'].shape} "
          f"L20={layers['L20'].shape} L27={layers['L27'].shape}", flush=True)
    return windows_df, layers


def attach_labels(windows_df: pd.DataFrame, master_csv: Path) -> pd.DataFrame:
    master = pd.read_csv(master_csv,
                         usecols=["window_id", "label", "split",
                                  "hard_negative", "overlaps_truncated_bgc"])
    merged = windows_df.merge(master, on="window_id", how="left", validate="m:1")
    missing = merged["label"].isna().sum()
    if missing > 0:
        raise RuntimeError(f"{missing} window_ids in shards not found in {master_csv}")
    return merged


def fit_one(
    X_layers_train: dict[str, np.ndarray],
    y_train: np.ndarray,
    X_layers_val: dict[str, np.ndarray],
    y_val: np.ndarray,
    combo: tuple[str, ...],
    C: float,
    max_iter: int,
) -> dict:
    """Fit one combo. Returns dict with W, b, per-layer scaler, val metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Per-layer standardization on train, applied to train+val.
    scalers: dict[str, dict[str, np.ndarray]] = {}
    pieces_tr = []
    pieces_va = []
    for ln in combo:
        Xtr = X_layers_train[ln].astype(np.float32, copy=False)
        Xva = X_layers_val[ln].astype(np.float32, copy=False)
        mu = Xtr.mean(axis=0).astype(np.float32)
        sd = Xtr.std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        scalers[ln] = {"mean": mu, "std": sd}
        pieces_tr.append((Xtr - mu) / sd)
        pieces_va.append((Xva - mu) / sd)
    Xtr = np.concatenate(pieces_tr, axis=1)
    Xva = np.concatenate(pieces_va, axis=1)

    t0 = time.time()
    lr = LogisticRegression(
        C=C, solver="lbfgs", max_iter=max_iter, n_jobs=-1,
        class_weight="balanced",
    )
    lr.fit(Xtr, y_train)
    fit_s = time.time() - t0
    W = lr.coef_[0].astype(np.float32)
    b = float(lr.intercept_[0])

    # Val metrics
    logits_va = Xva @ W + b
    probs_va = 1.0 / (1.0 + np.exp(-logits_va.astype(np.float64)))
    yhat = (probs_va >= 0.5).astype(np.int32)
    tp = int(((yhat == 1) & (y_val == 1)).sum())
    fp = int(((yhat == 1) & (y_val == 0)).sum())
    fn = int(((yhat == 0) & (y_val == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    metrics = {
        "fit_seconds": fit_s,
        "n_train": int(len(y_train)),
        "n_train_pos": int(y_train.sum()),
        "n_val": int(len(y_val)),
        "n_val_pos": int(y_val.sum()),
        "auroc": float(roc_auc_score(y_val, probs_va)),
        "ap": float(average_precision_score(y_val, probs_va)),
        "thr0.5_precision": prec,
        "thr0.5_recall": rec,
        "thr0.5_f1": f1,
        "feature_dim": int(Xtr.shape[1]),
    }
    return {"W": W, "b": b, "scalers": scalers, "metrics": metrics}


def save_lr(path: Path, fit: dict, combo: tuple[str, ...]) -> None:
    payload = {"W": fit["W"], "b": np.float32(fit["b"]),
               "combo": np.array(list(combo))}
    for ln in combo:
        payload[f"mean_{ln}"] = fit["scalers"][ln]["mean"]
        payload[f"std_{ln}"] = fit["scalers"][ln]["std"]
    np.savez(path, **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--combos", nargs="+", default=list(COMBOS.keys()))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    windows_df, layers = load_shards(args.shard_dir)
    merged = attach_labels(windows_df, args.master_csv)

    keep = merged["label"].isin({"positive", "negative"}).to_numpy()
    split = merged["split"].astype(str).to_numpy()
    y_full = (merged["label"].to_numpy() == "positive").astype(np.int32)

    train_mask = (split == "train") & keep
    val_mask = (split == "val") & keep
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"[split] keep={int(keep.sum())} train={len(train_idx)} val={len(val_idx)} "
          f"train.pos={int(y_full[train_idx].sum())} val.pos={int(y_full[val_idx].sum())}",
          flush=True)

    X_train = {ln: layers[ln][train_idx] for ln in LAYER_SHORT}
    X_val = {ln: layers[ln][val_idx] for ln in LAYER_SHORT}
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]

    summary: dict[str, dict] = {}
    for name in args.combos:
        combo = COMBOS[name]
        print(f"\n==== combo {name} ({combo}) ====", flush=True)
        fit = fit_one(X_train, y_train, X_val, y_val, combo,
                      C=args.C, max_iter=args.max_iter)
        lr_path = args.out_dir / f"lr_{name}.npz"
        save_lr(lr_path, fit, combo)
        with open(args.out_dir / f"metrics_{name}.json", "w") as f:
            json.dump(fit["metrics"], f, indent=2)
        summary[name] = fit["metrics"]
        m = fit["metrics"]
        print(f"  AUROC={m['auroc']:.4f}  AP={m['ap']:.4f}  "
              f"P@.5={m['thr0.5_precision']:.4f}  R@.5={m['thr0.5_recall']:.4f}  "
              f"F1@.5={m['thr0.5_f1']:.4f}  D={m['feature_dim']}", flush=True)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({"shard_dir": str(args.shard_dir),
                   "master_csv": str(args.master_csv),
                   "C": args.C, "max_iter": args.max_iter,
                   "combos": args.combos}, f, indent=2)
    print(f"\n[done] summary -> {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
