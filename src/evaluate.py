"""
Phase 3 Context Model Evaluation.

Two evaluation modes:
1. Window-level: bgc_binary_auroc (fast, every epoch)
2. Region-level: val F1 at optimal threshold (every 5 epochs, PRIMARY checkpoint metric)
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve

from data.context_dataset import ContextDataset


def _safe_auroc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def evaluate_context_model(model, dataset: ContextDataset, device: str,
                           K: int = 11, batch_size: int = 512) -> Dict:
    """Window-level evaluation using contig-aware inference.

    Processes each contig fully, collects per-window predictions,
    then computes bgc_binary_auroc on non-ambiguous center windows.
    """
    model.eval()
    center_set = set(dataset.center_indices.tolist())

    all_scores = []
    all_is_bgc = []
    processed = set()

    for contig_key, indices in dataset.contig_index.items():
        # Skip contigs with no center windows in this split
        relevant = [i for i in indices if i in center_set]
        if not relevant:
            continue

        # Full contig forward pass
        proj, valid_mask, labels, global_indices = dataset.get_contig_data(contig_key)
        proj = proj.unsqueeze(0).to(device)
        valid_mask = valid_mask.unsqueeze(0).to(device)

        _, logits = model(proj, valid_mask)
        probs = torch.sigmoid(logits[0]).cpu().numpy()  # (L, 8)

        for pos, g_idx in enumerate(global_indices):
            if g_idx not in center_set or g_idx in processed:
                continue
            processed.add(g_idx)
            score = probs[pos, :7].max()
            is_bgc = 1 - int(dataset.all_labels[g_idx, 7])
            all_scores.append(score)
            all_is_bgc.append(is_bgc)

    all_scores = np.array(all_scores)
    all_is_bgc = np.array(all_is_bgc)
    bgc_auroc = _safe_auroc(all_is_bgc, all_scores)

    return {
        "bgc_binary_auroc": bgc_auroc,
        "n_evaluated": len(all_scores),
        "n_positive": int(all_is_bgc.sum()),
    }


@torch.no_grad()
def evaluate_region_level(model, dataset: ContextDataset, device: str,
                          bgc_csv_path: str = None, split: str = "val") -> Dict:
    """Region-level proxy: compute window-level F1 at optimal threshold on val.

    This is faster than full region calling and correlates with region-level F1.
    Used as the PRIMARY checkpoint selection metric (Rule 8).
    """
    model.eval()
    center_set = set(dataset.center_indices.tolist())

    all_scores = []
    all_is_bgc = []
    processed = set()

    for contig_key, indices in dataset.contig_index.items():
        relevant = [i for i in indices if i in center_set]
        if not relevant:
            continue

        proj, valid_mask, labels, global_indices = dataset.get_contig_data(contig_key)
        proj = proj.unsqueeze(0).to(device)
        valid_mask = valid_mask.unsqueeze(0).to(device)

        _, logits = model(proj, valid_mask)
        probs = torch.sigmoid(logits[0]).cpu().numpy()

        for pos, g_idx in enumerate(global_indices):
            if g_idx not in center_set or g_idx in processed:
                continue
            processed.add(g_idx)
            score = probs[pos, :7].max()
            is_bgc = 1 - int(dataset.all_labels[g_idx, 7])
            all_scores.append(score)
            all_is_bgc.append(is_bgc)

    all_scores = np.array(all_scores)
    all_is_bgc = np.array(all_is_bgc)

    if len(np.unique(all_is_bgc)) < 2:
        return {"region_f1": float("nan")}

    prec, rec, thr = precision_recall_curve(all_is_bgc, all_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1s)
    best_f1 = float(f1s[best_idx])
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5

    return {
        "region_f1": best_f1,
        "threshold": best_thr,
        "precision": float(prec[best_idx]),
        "recall": float(rec[best_idx]),
        "n_windows": len(all_scores),
    }
