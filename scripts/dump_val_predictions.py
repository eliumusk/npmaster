#!/usr/bin/env python3
"""Dump per-window predictions for the val split using cached embeddings.

Stage 5b needs per-window scores on internal val for decoder sweeping,
but re-running Evo2 on 169 val genomes × 24 checkpoints is prohibitive.
The training pipeline already emits and caches a full embeddings array,
so this script just runs the context model's **CNN + classifier** on
those cached embeddings — zero Evo2 calls.

Output format matches ``scripts/inference_stride.py`` so the resulting
CSVs are drop-in inputs to ``scripts/evaluate_nine_genome.py`` and
``scripts/postprocess_v2.decode_regions_hysteresis``.

Usage:
    python scripts/dump_val_predictions.py \\
        --ckpt experiments/.../artifacts/checkpoints/best_tight_pos_099_v1_s0.pt \\
        --meta /data/muskliu/npmaster/data/phase3/route_a/meta_arrays.pkl \\
        --embeddings /data/muskliu/npmaster/data/embeddings/edge_padding/embeddings_merged.npy \\
        --contig-index /data/muskliu/npmaster/data/phase3/route_a/contig_index.pkl \\
        --out-dir experiments/.../artifacts/val_predictions_tight_pos_099_v1_s0 \\
        --split val \\
        --window-size 8192
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_NAMES = (
    "Polyketide",
    "NRP",
    "Terpene",
    "RiPP",
    "Saccharide",
    "Alkaloid",
    "Other",
    "non_bgc",
)


def _load_model(ckpt_path: Path, device: str):
    import torch

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from model.context_full_model import ContextFullModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def _group_val_contigs(
    meta: dict, split: str
) -> dict[tuple[str, str], list[int]]:
    """Map (genome_id, contig_id) -> list of row indices in the meta arrays
    that belong to the requested split and that contig, in window_start
    order."""
    splits = np.asarray(meta["split"])
    gids = np.asarray(meta["genome_id"])
    cids = np.asarray(meta["contig_id"])
    starts = np.asarray(meta["window_start"])

    keep = splits == split
    if not keep.any():
        raise ValueError(f"No rows in meta_arrays with split=={split!r}")

    idxs_all = np.nonzero(keep)[0]
    groups: dict[tuple[str, str], list[int]] = {}
    for i in idxs_all:
        key = (str(gids[i]), str(cids[i]))
        groups.setdefault(key, []).append(int(i))

    # Sort each contig's indices by window_start so predictions are
    # ordered along the genome axis.
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(starts[r]))
    return groups


def _forward_contig(
    model, embeddings: np.ndarray, contig_idxs: list[int], device: str
) -> np.ndarray:
    """Return (L, 8) probs for one contig."""
    import torch

    emb = embeddings[contig_idxs]  # (L, 4096)
    emb_tensor = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(device)
    valid_mask = torch.ones(
        1, emb_tensor.shape[1], dtype=torch.bool, device=device
    )
    with torch.no_grad():
        _, logits = model(emb_tensor, valid_mask)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
    return probs


def dump_predictions(
    ckpt: Path,
    meta: Path,
    embeddings: Path,
    contig_index: Path,
    out_dir: Path,
    split: str = "val",
    window_size: int = 8192,
    max_contig_windows: int | None = None,
) -> dict[str, int]:
    """Run the forward pass over every val contig and dump per-genome CSVs.

    ``max_contig_windows`` optionally chunks very long contigs to bound
    peak GPU memory (default: no chunking). The model's receptive field
    along the window axis is only 15, so chunking with a 15-window
    overlap and trimming would work, but in practice 80 GB A800 handles
    even the longest contigs in one shot so we leave this off.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[dump_val] loading model {ckpt.name}", flush=True)
    model, ckpt_blob = _load_model(ckpt, device)
    print(
        f"[dump_val] model loaded alpha={model.gate.alpha.item():.4f} "
        f"epoch={ckpt_blob.get('epoch', '?')}",
        flush=True,
    )

    print(f"[dump_val] loading meta_arrays {meta}", flush=True)
    with meta.open("rb") as f:
        meta_blob = pickle.load(f)
    print(f"[dump_val] loading embeddings mmap {embeddings}", flush=True)
    emb_arr = np.load(embeddings, mmap_mode="r")
    print(f"[dump_val] embeddings shape={emb_arr.shape} dtype={emb_arr.dtype}", flush=True)

    groups = _group_val_contigs(meta_blob, split)
    print(
        f"[dump_val] {split} contigs={len(groups)} "
        f"{split} windows={sum(len(v) for v in groups.values())}",
        flush=True,
    )

    counts_per_genome: dict[str, int] = {}

    # Accumulate per-genome rows, flush on genome change.
    by_genome: dict[str, list[dict]] = {}
    starts_arr = np.asarray(meta_blob["window_start"])

    for n_done, (key, rows) in enumerate(sorted(groups.items()), start=1):
        gid, cid = key
        probs = _forward_contig(model, emb_arr, rows, device)
        for idx_in_contig, row_idx in enumerate(rows):
            p = probs[idx_in_contig]
            start = int(starts_arr[row_idx])
            by_genome.setdefault(gid, []).append(
                {
                    "contig": cid,
                    "start": start,
                    "end": start + window_size,
                    "bgc_score": round(float(p[:7].max()), 6),
                    "non_bgc_score": round(float(p[7]), 6),
                    "pred_class": DEFAULT_CLASS_NAMES[int(p.argmax())],
                    "probs": json.dumps(
                        [round(float(x), 6) for x in p.tolist()]
                    ),
                }
            )
        if n_done % 500 == 0:
            print(
                f"[dump_val] processed {n_done}/{len(groups)} contigs "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    for gid, rows in by_genome.items():
        out_csv = out_dir / f"{gid}_predictions.csv"
        rows.sort(key=lambda r: (r["contig"], r["start"]))
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "contig",
                    "start",
                    "end",
                    "bgc_score",
                    "non_bgc_score",
                    "pred_class",
                    "probs",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        counts_per_genome[gid] = len(rows)

    print(
        f"[dump_val] done {len(counts_per_genome)} genomes "
        f"{sum(counts_per_genome.values())} rows "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return counts_per_genome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--contig-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--window-size", type=int, default=8192)
    args = parser.parse_args(argv)

    counts = dump_predictions(
        ckpt=args.ckpt,
        meta=args.meta,
        embeddings=args.embeddings,
        contig_index=args.contig_index,
        out_dir=args.out_dir,
        split=args.split,
        window_size=args.window_size,
    )
    # Compact summary on stderr.
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "split": args.split,
                "n_genomes": len(counts),
                "n_windows": sum(counts.values()),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
