#!/usr/bin/env python3
"""Mine fresh hard negatives for Route B from training negatives."""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def score_fn_from_probs(probs: np.ndarray, score_name: str) -> np.ndarray:
    max_type = probs[:, :7].max(axis=1)
    one_minus_isneg = 1.0 - probs[:, 7]
    if score_name == "max_type":
        return max_type
    if score_name == "1_minus_isneg":
        return one_minus_isneg
    if score_name == "fused":
        return 0.5 * max_type + 0.5 * one_minus_isneg
    raise ValueError(f"Unknown score_name={score_name}")


def select_hard_negative_indices(
    split: np.ndarray,
    label: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    min_score: float,
) -> np.ndarray:
    mask = (split == "train") & (label == "negative") & (scores >= min_score)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-scores[indices], kind="mergesort")
    if top_k > 0:
        order = order[:top_k]
    return indices[order]


def patch_meta_arrays(meta: dict, hard_indices: np.ndarray) -> dict:
    patched = dict(meta)
    hard_negative = np.zeros(len(meta["label"]), dtype=np.int32)
    hard_negative[hard_indices] = 1
    patched["hard_negative"] = hard_negative
    return patched


def resolve_src_dir(cli_value: str | None) -> Path:
    if cli_value:
        candidate = Path(cli_value).expanduser().resolve()
        if (candidate / "model" / "context_full_model.py").exists():
            return candidate
        raise FileNotFoundError(f"Invalid --src-dir: {candidate}")

    candidates = []
    try:
        current_file = Path(__file__).resolve()
        candidates.extend([
            current_file.parent.parent / "src",
            current_file.parent.parent.parent / "src",
            Path.cwd() / "src",
            Path.cwd().parent / "src",
        ])
    except Exception:
        pass
    for candidate in candidates:
        if (candidate / "model" / "context_full_model.py").exists():
            return candidate
    searched = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not locate src/. Pass --src-dir explicitly. "
        f"Searched: {searched}"
    )


def load_route_b_model(ckpt_path: str | Path, device: str, src_dir: str | Path | None):
    src_path = str(resolve_src_dir(str(src_dir) if src_dir else None))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from model.context_full_model import ContextFullModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ContextFullModel(
        input_dim=4096,
        proj_hidden=2048,
        proj_out=256,
        proj_dropout=0.0,
        n_labels=8,
        cnn_dropout=0.0,
        init_beta=-4.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def infer_all_scores(
    model,
    embeddings_path: str | Path,
    contig_index_path: str | Path,
) -> np.ndarray:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    with open(contig_index_path, "rb") as f:
        contig_index = pickle.load(f)

    scores = np.zeros((embeddings.shape[0], 8), dtype=np.float32)
    device = next(model.parameters()).device
    for indices in contig_index.values():
        emb = np.stack([embeddings[i] for i in indices])
        emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)
        valid = torch.ones(1, len(indices), dtype=torch.bool, device=device)
        _, logits = model(emb_t, valid)
        probs = torch.sigmoid(logits[0]).cpu().numpy().astype(np.float32)
        scores[np.asarray(indices, dtype=np.int64)] = probs
    return scores


def write_candidates_csv(
    out_path: Path,
    meta: dict,
    score_family: np.ndarray,
    type_probs: np.ndarray,
    hard_indices: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "global_idx",
                "genome_id",
                "contig_id",
                "window_start",
                "split",
                "label",
                "phylum",
                "score",
                "max_type",
                "top_type_idx",
                "top_type_score",
                "is_negative_prob",
            ],
        )
        writer.writeheader()
        for rank, g_idx in enumerate(hard_indices, start=1):
            row_probs = type_probs[g_idx]
            top_type_idx = int(np.argmax(row_probs[:7]))
            writer.writerow({
                "rank": rank,
                "global_idx": int(g_idx),
                "genome_id": str(meta["genome_id"][g_idx]),
                "contig_id": str(meta["contig_id"][g_idx]),
                "window_start": int(meta["window_start"][g_idx]),
                "split": str(meta["split"][g_idx]),
                "label": str(meta["label"][g_idx]),
                "phylum": str(meta["phylum"][g_idx]),
                "score": float(score_family[g_idx]),
                "max_type": float(row_probs[:7].max()),
                "top_type_idx": top_type_idx,
                "top_type_score": float(row_probs[top_type_idx]),
                "is_negative_prob": float(row_probs[7]),
            })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--meta-arrays", required=True)
    p.add_argument("--contig-index", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--score-name", choices=["max_type", "1_minus_isneg", "fused"], default="fused")
    p.add_argument("--top-k", type=int, default=5000)
    p.add_argument("--min-score", type=float, default=0.95)
    p.add_argument("--device", default="cuda")
    p.add_argument("--src-dir", default=None, help="Path to repo src/ (auto-detected if omitted).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.meta_arrays, "rb") as f:
        meta = pickle.load(f)

    model, ckpt = load_route_b_model(args.ckpt, device=device, src_dir=args.src_dir)
    probs = infer_all_scores(model, args.embeddings, args.contig_index)
    scores = score_fn_from_probs(probs, args.score_name)
    hard_indices = select_hard_negative_indices(
        split=meta["split"],
        label=meta["label"],
        scores=scores,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    patched_meta = patch_meta_arrays(meta, hard_indices)

    with (out_dir / "meta_arrays.pkl").open("wb") as f:
        pickle.dump(patched_meta, f)

    write_candidates_csv(
        out_dir / "hardneg_candidates.csv",
        meta=meta,
        score_family=scores,
        type_probs=probs,
        hard_indices=hard_indices,
    )

    phylum_counts: dict[str, int] = {}
    if len(hard_indices) > 0:
        unique, counts = np.unique(meta["phylum"][hard_indices].astype(str), return_counts=True)
        phylum_counts = {
            str(k): int(v)
            for k, v in zip(unique.tolist(), counts.tolist())
        }

    summary = {
        "checkpoint": str(args.ckpt),
        "epoch": int(ckpt.get("epoch", -1)),
        "score_name": args.score_name,
        "top_k": int(args.top_k),
        "min_score": float(args.min_score),
        "n_total_windows": int(len(meta["label"])),
        "n_train_negative": int(np.sum((meta["split"] == "train") & (meta["label"] == "negative"))),
        "n_hard_negative": int(len(hard_indices)),
        "score_min": float(scores[hard_indices[-1]]) if len(hard_indices) > 0 else None,
        "score_max": float(scores[hard_indices[0]]) if len(hard_indices) > 0 else None,
        "phylum_counts": phylum_counts,
        "output_meta_arrays": str(out_dir / "meta_arrays.pkl"),
        "output_candidates_csv": str(out_dir / "hardneg_candidates.csv"),
    }
    (out_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
