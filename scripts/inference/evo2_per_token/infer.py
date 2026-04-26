#!/usr/bin/env python3
"""Apply a trained per-token head to cached per-token Evo2 features.

Output: per-window NPZ with (window_id, token_probs) where token_probs has shape (T,).
Downstream decode reads these into regions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evo2_per_token.model import PerTokenHead  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out-npz", required=True, type=Path)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--window-ids-txt", type=Path, default=None,
                   help="Optional: text file of window IDs to process (one per line). If omitted, process all .npy in features-dir.")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["args"]
    model = PerTokenHead(
        d_in=cfg["d_reduced"], hidden=cfg["hidden"], dilations=tuple(cfg["dilations"]),
    )
    model.load_state_dict(ckpt["model"])
    model.eval().cuda()

    if args.window_ids_txt and args.window_ids_txt.exists():
        ids = [line.strip() for line in args.window_ids_txt.open() if line.strip()]
    else:
        ids = sorted(p.stem for p in args.features_dir.glob("*.npy"))
    print(f"{len(ids)} windows to score")

    probs_out = {}
    with torch.no_grad():
        for start in range(0, len(ids), args.batch):
            batch_ids = ids[start:start + args.batch]
            xs = [np.load(args.features_dir / f"{w}.npy").astype(np.float32) for w in batch_ids]
            max_T = max(x.shape[0] for x in xs)
            D = xs[0].shape[1]
            padded = np.zeros((len(xs), max_T, D), dtype=np.float32)
            for i, x in enumerate(xs):
                padded[i, :x.shape[0]] = x
            x_t = torch.from_numpy(padded).cuda()
            logits = model(x_t)                   # (B, max_T)
            p = torch.sigmoid(logits).cpu().numpy()
            for i, wid in enumerate(batch_ids):
                probs_out[wid] = p[i, :xs[i].shape[0]].astype(np.float16)
            if (start // args.batch) % 50 == 0:
                print(f"  scored {start + len(batch_ids)} / {len(ids)}")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **probs_out)
    print(f"wrote {args.out_npz} with {len(probs_out)} windows")


if __name__ == "__main__":
    main()
