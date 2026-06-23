#!/usr/bin/env python3
"""Evaluation for the multi-label product-type head.

Reports the metrics standard in the BGC-classification literature:
  - per-class precision / recall / F1 / support (threshold on P(present))
  - macro-F1 and micro-F1 (macro is the headline — it exposes minority-class
    and PKS/NRP failures that accuracy hides)
  - macro-AUROC (threshold-free, the metric the SOTA benchmarks rank on)
  - a single-label confusion matrix over clusters that have exactly one true
    class — this is the clean view of the PKS↔NRP confusion that motivated the
    rebuild.
"""
from __future__ import annotations

import numpy as np


def multilabel_report(
    y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], threshold: float = 0.5
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    per_class = {}
    f1s = []
    for j, c in enumerate(classes):
        tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
        fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
        fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(y_true[:, j].sum()),
        }
        f1s.append(f1)

    tp_all = int(((y_pred == 1) & (y_true == 1)).sum())
    fp_all = int(((y_pred == 1) & (y_true == 0)).sum())
    fn_all = int(((y_pred == 0) & (y_true == 1)).sum())
    micro_p = tp_all / max(1, tp_all + fp_all)
    micro_r = tp_all / max(1, tp_all + fn_all)
    micro_f1 = 2 * micro_p * micro_r / max(1e-9, micro_p + micro_r)

    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_auroc": macro_auroc(y_true, y_prob),
        "per_class": per_class,
    }


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Mean per-class AUROC, skipping classes with no positive or no negative."""
    from sklearn.metrics import roc_auc_score

    aurocs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        aurocs.append(roc_auc_score(col, y_prob[:, j]))
    return round(float(np.mean(aurocs)), 4) if aurocs else None


def single_label_confusion(
    y_true: np.ndarray, y_prob: np.ndarray, classes: list[str]
) -> dict:
    """Confusion matrix over single-true-label clusters (predicted = argmax prob).

    Restricting to single-label truth makes the PKS↔NRP cell unambiguous; hybrid
    clusters are excluded here and judged by the multi-label report instead.
    """
    single = np.where(y_true.sum(axis=1) == 1)[0]
    n = len(classes)
    mat = np.zeros((n, n), dtype=int)
    for i in single:
        t = int(np.argmax(y_true[i]))
        p = int(np.argmax(y_prob[i]))
        mat[t, p] += 1
    return {
        "classes": classes,
        "matrix": mat.tolist(),
        "n_single_label": int(len(single)),
        "accuracy": round(float(np.trace(mat) / max(1, mat.sum())), 4),
    }


def format_confusion(conf: dict) -> str:
    """Pretty ASCII confusion matrix (rows = true, cols = predicted)."""
    classes = conf["classes"]
    mat = conf["matrix"]
    w = max(4, max(len(c) for c in classes))
    head = " " * (w + 2) + " ".join(f"{c[:w]:>{w}}" for c in classes)
    lines = [head]
    for i, c in enumerate(classes):
        row = " ".join(f"{mat[i][j]:>{w}}" for j in range(len(classes)))
        lines.append(f"{c:>{w}}  {row}")
    lines.append(f"(single-label acc = {conf['accuracy']}, n = {conf['n_single_label']})")
    return "\n".join(lines)
