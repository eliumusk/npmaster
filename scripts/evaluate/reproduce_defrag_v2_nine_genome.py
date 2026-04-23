#!/usr/bin/env python3
"""Reproduce defrag_v2_ms10_s099 9-genome per-window predictions.

Exists because the canonical preds dir was cleaned up in the 2026-04-19 purge
(lived under a `phase4/nine_genome_sweep_*` name that matched the sweep-cleanup
filter). The Route B CNN checkpoint + cached Evo2 embeddings at w=8192/s=2048
are both still on disk, so we can regenerate the inference CSVs cheaply (no
Evo2 forward needed — a few seconds total on one GPU).

Produces the same CSV schema as `data/phase4/stride8192_s0_merged/*` so
scoring via
  scripts/evaluate_nine_genome.py --operating-point defrag_v2_ms10_s099
reproduces the published pooled dedup_F1 = 0.4385.

Canonical invocation (on server):
  conda activate bgc
  python scripts/evaluate/reproduce_defrag_v2_nine_genome.py \
      --ckpt /data/muskliu/npmaster/data/phase3/stage3_hardneg_v1/best/best_tk100-ms0-hnw4.pt \
      --emb-dir /data/muskliu/npmaster/experiments/evo2_first_principles/9g_embeddings_w8192_s2048 \
      --out-dir /data/muskliu/npmaster/data/phase4/defrag_v2_ms10_s099_preds_stride2048
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# CNN output class order — empirically verified 2026-04-23 by correlating
# per-column positive counts in meta_arrays.pkl with single-type clusters in
# bgc_annotations.csv. EARLIER VERSIONS OF THIS FILE HAD THE WRONG ORDER
# (copied from inference_stride.py), which caused pred_class strings to be
# silently mislabeled (Polyketide→Alkaloid, NRP→Terpene, etc). probs values
# themselves were always correct — only the string mapping was wrong.
DEFAULT_CLASS_NAMES = [
    "Polyketide",  # col 0
    "NRP",         # col 1
    "Terpene",     # col 2
    "RiPP",        # col 3
    "Saccharide",  # col 4
    "Alkaloid",    # col 5
    "Other",       # col 6
    "non-BGC",     # col 7
]


def resolve_src_dir(repo_root: Path) -> Path:
    src = repo_root / "src"
    if not (src / "model" / "context_full_model.py").exists():
        raise FileNotFoundError(f"src/ not found at {src}")
    return src


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--emb-dir", required=True, type=Path,
                   help="Dir with GCA_*_embeddings.npy + GCA_*_windows.csv")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--genome-ids", default=None,
                   help="Comma-separated subset (default: all in emb-dir)")
    p.add_argument("--repo-root", default=None, type=Path,
                   help="Path to repo root (for src/ import; auto-detected if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(resolve_src_dir(repo_root)))
    from model.context_full_model import ContextFullModel  # noqa: E402

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ckpt: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ContextFullModel(
        input_dim=4096, proj_hidden=2048, proj_out=256,
        proj_dropout=0.0, n_labels=8, cnn_dropout=0.0, init_beta=-4.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to("cuda").eval()
    print(f"  alpha={model.gate.alpha.item():.4f} epoch={ckpt.get('epoch','?')}", flush=True)

    selected = None
    if args.genome_ids:
        selected = {s.strip() for s in args.genome_ids.split(",") if s.strip()}

    emb_files = sorted(args.emb_dir.glob("*_embeddings.npy"))
    if selected is not None:
        emb_files = [f for f in emb_files if f.name.replace("_embeddings.npy", "") in selected]
    if not emb_files:
        raise SystemExit(f"No matching embeddings under {args.emb_dir}")

    summary: dict[str, int] = {}
    for emb_path in emb_files:
        genome = emb_path.name.replace("_embeddings.npy", "")
        win_csv = args.emb_dir / f"{genome}_windows.csv"
        if not win_csv.exists():
            print(f"[skip] {genome}: no windows.csv", flush=True)
            continue

        t0 = time.time()
        emb = np.load(emb_path).astype(np.float32)                     # (N, 4096)
        win_df = pd.read_csv(win_csv)                                   # contig,start,end
        if len(win_df) != emb.shape[0]:
            raise RuntimeError(f"{genome}: windows.csv ({len(win_df)}) != emb rows ({emb.shape[0]})")

        rows: list[dict] = []
        for contig_id, sub in win_df.groupby("contig", sort=False):
            idx = sub.index.to_numpy()
            sub_emb = emb[idx]                                          # (L, 4096)
            L = sub_emb.shape[0]
            x = torch.from_numpy(sub_emb).unsqueeze(0).to("cuda")       # (1, L, 4096)
            mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
            with torch.no_grad():
                _, logits = model(x, mask)
                probs = torch.sigmoid(logits[0]).cpu().numpy()          # (L, 8)
            for i, (_, r) in enumerate(sub.iterrows()):
                bgc_score = float(probs[i, :7].max())
                pred_class = DEFAULT_CLASS_NAMES[int(probs[i].argmax())]
                rows.append({
                    "contig": contig_id,
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "bgc_score": round(bgc_score, 6),
                    "non_bgc_score": round(float(probs[i, 7]), 6),
                    "pred_class": pred_class,
                    "probs": json.dumps([round(float(p), 6) for p in probs[i].tolist()]),
                })

        out_csv = args.out_dir / f"{genome}_predictions.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["contig", "start", "end", "bgc_score",
                            "non_bgc_score", "pred_class", "probs"],
            )
            w.writeheader()
            w.writerows(rows)
        summary[genome] = len(rows)
        print(f"  {genome}: {len(rows)} windows  [{time.time()-t0:.1f}s]", flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "source_ckpt": str(args.ckpt),
        "source_emb_dir": str(args.emb_dir),
        "window": 8192,
        "stride": 2048,
        "n_windows_per_genome": summary,
    }, indent=2))
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
