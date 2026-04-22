#!/usr/bin/env python3
"""Baseline A: train a linear probe on existing 4096-d Evo2 mean-pool embeddings.

Input:
  data/embeddings/edge_padding/embeddings_merged.npy   (N, 4096) fp32
  data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv

Output (to --out-dir):
  lr_{variant}.npz         weights (W, b)
  metrics_{variant}.json   val AUROC, AP, thresholded P/R/F1
  config.json              run config

Variants:
  A1  all positives vs all negatives (drop ambiguous)
  A2  drop ambiguous + drop truncated positives
  A3  positives + hard_negatives only (sharper signal set)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(embeddings_path: Path, csv_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    print(f"[load] mmap embeddings {embeddings_path}", flush=True)
    emb = np.load(embeddings_path, mmap_mode="r")
    print(f"[load] embeddings shape={emb.shape} dtype={emb.dtype}", flush=True)

    print(f"[load] csv {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"[load] csv rows={len(df)}", flush=True)
    assert len(df) == emb.shape[0], f"row count mismatch: {len(df)} vs {emb.shape[0]}"
    return emb, df


def mask_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    """Boolean mask of rows that contribute to training for a given variant."""
    label = df["label"].astype(str)
    if variant == "A1":
        return label.isin({"positive", "negative"})
    if variant == "A2":
        keep_label = label.isin({"positive", "negative"})
        keep_trunc = ~((label == "positive") & (df["overlaps_truncated_bgc"] == 1))
        return keep_label & keep_trunc
    if variant == "A3":
        is_pos = label == "positive"
        is_hardneg = (df["hard_negative"] == 1)
        return is_pos | is_hardneg
    raise ValueError(f"unknown variant {variant}")


def fit_lr(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    C: float,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    """sklearn logistic regression on fp32 embeddings. Returns (W_1, b)."""
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    lr = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=-1,
        class_weight="balanced",
        verbose=0,
    )
    lr.fit(X, y, sample_weight=sample_weight)
    dt = time.time() - t0
    print(f"[fit] variant fit in {dt:.1f}s n={len(y)} pos={int(y.sum())}", flush=True)
    W = lr.coef_[0].astype(np.float32)
    b = float(lr.intercept_[0])
    return W, b


def score(W: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    logits = X @ W + b
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return probs.astype(np.float32)


def metrics_binary(y_true: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    out = {}
    out["n"] = int(len(y_true))
    out["n_pos"] = int(y_true.sum())
    out["auroc"] = float(roc_auc_score(y_true, p))
    out["ap"] = float(average_precision_score(y_true, p))
    # threshold @ 0.5
    yhat = (p >= 0.5).astype(np.int32)
    tp = int(((yhat == 1) & (y_true == 1)).sum())
    fp = int(((yhat == 1) & (y_true == 0)).sum())
    fn = int(((yhat == 0) & (y_true == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out["thr0.5_precision"] = prec
    out["thr0.5_recall"] = rec
    out["thr0.5_f1"] = f1
    # best-F1 threshold on this set (diagnostic only)
    order = np.argsort(-p)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    n_pos = y_sorted.sum()
    prec_arr = tp_cum / (tp_cum + fp_cum + 1e-9)
    rec_arr = tp_cum / max(1, n_pos)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    bi = int(np.argmax(f1_arr))
    out["bestF1"] = float(f1_arr[bi])
    out["bestF1_thr"] = float(p[order][bi])
    out["bestF1_precision"] = float(prec_arr[bi])
    out["bestF1_recall"] = float(rec_arr[bi])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["A1", "A2", "A3"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--subsample-neg", type=int, default=0,
                    help="if >0, downsample negatives in the train split to this many")
    ap.add_argument("--hardneg-weight", type=float, default=1.0,
                    help="sample weight multiplier for hard_negative==1 in train (A1/A2 only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    emb, df = load_data(args.embeddings, args.csv)
    split = df["split"].astype(str).to_numpy()

    # preload full fp32 matrix into RAM slice by slice to avoid giant copy.
    # we always need the train + val slices in memory.
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    testid_idx = np.where(split == "test_id")[0]
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test_id={len(testid_idx)}", flush=True)

    # Lift to fp32 small slices; emb is memmap so keep slicing lazy.
    def slice_mat(idx: np.ndarray) -> np.ndarray:
        t0 = time.time()
        x = np.asarray(emb[idx], dtype=np.float32)
        print(f"[slice] {len(idx)} rows in {time.time()-t0:.1f}s, mem≈{x.nbytes/1e9:.2f}GB", flush=True)
        return x

    # Labels
    def label_vec(idx: np.ndarray, variant: str) -> np.ndarray:
        mask = mask_variant(df.iloc[idx], variant).to_numpy()
        lab = (df["label"].iloc[idx].to_numpy() == "positive").astype(np.int32)
        return mask, lab

    # Load val once (used across all variants); use ALL val rows (incl ambiguous)
    # but we'll only eval on label ∈ {pos, neg}. Ambiguous excluded.
    val_mask_all = df["label"].iloc[val_idx].isin({"positive", "negative"}).to_numpy()
    val_idx_kept = val_idx[val_mask_all]
    X_val = slice_mat(val_idx_kept)
    y_val = (df["label"].iloc[val_idx_kept].to_numpy() == "positive").astype(np.int32)
    print(f"[val] kept={len(val_idx_kept)} pos={int(y_val.sum())}", flush=True)

    all_metrics = {}
    for variant in args.variants:
        print(f"\n==== variant {variant} ====", flush=True)
        mask, lab = label_vec(train_idx, variant)
        tr_keep = train_idx[mask]
        y_tr = lab[mask]
        # optional subsample negatives
        if args.subsample_neg > 0:
            neg_pos = np.where(y_tr == 0)[0]
            pos_pos = np.where(y_tr == 1)[0]
            if len(neg_pos) > args.subsample_neg:
                neg_keep = rng.choice(neg_pos, size=args.subsample_neg, replace=False)
                keep = np.concatenate([pos_pos, neg_keep])
                keep.sort()
                tr_keep = tr_keep[keep]
                y_tr = y_tr[keep]
                print(f"[ss] down-sampled neg → train n={len(y_tr)}", flush=True)
        print(f"[train/{variant}] n={len(y_tr)} pos={int(y_tr.sum())}", flush=True)
        X_tr = slice_mat(tr_keep)

        # sample weights (hard-neg up-weight for A1/A2)
        sw = None
        if variant in {"A1", "A2"} and args.hardneg_weight != 1.0:
            sw = np.ones(len(y_tr), dtype=np.float32)
            hn = (df["hard_negative"].iloc[tr_keep].to_numpy() == 1)
            sw[hn & (y_tr == 0)] = float(args.hardneg_weight)
            print(f"[sw] applied hardneg_weight={args.hardneg_weight} on {int(hn.sum())} rows", flush=True)

        W, b = fit_lr(X_tr, y_tr, sw, C=args.C, max_iter=args.max_iter)
        np.savez(args.out_dir / f"lr_{variant}.npz", W=W, b=np.float32(b))

        # Val metrics
        p_val = score(W, b, X_val)
        m_val = metrics_binary(y_val, p_val)
        # test_id metrics (auxiliary)
        testid_mask = df["label"].iloc[testid_idx].isin({"positive", "negative"}).to_numpy()
        testid_kept = testid_idx[testid_mask]
        X_testid = slice_mat(testid_kept)
        y_testid = (df["label"].iloc[testid_kept].to_numpy() == "positive").astype(np.int32)
        p_testid = score(W, b, X_testid)
        m_testid = metrics_binary(y_testid, p_testid)
        print(f"[val  /{variant}] AUROC={m_val['auroc']:.4f} AP={m_val['ap']:.4f} bestF1={m_val['bestF1']:.4f}", flush=True)
        print(f"[t_id/{variant}] AUROC={m_testid['auroc']:.4f} AP={m_testid['ap']:.4f} bestF1={m_testid['bestF1']:.4f}", flush=True)

        all_metrics[variant] = {
            "val": m_val,
            "test_id": m_testid,
            "n_train": int(len(y_tr)),
            "n_train_pos": int(y_tr.sum()),
        }
        del X_tr, X_testid
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(vars(args) | {"embeddings_shape": list(emb.shape)},
                  f, indent=2, default=str)
    print(f"\n[done] metrics → {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
