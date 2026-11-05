#!/usr/bin/env python3
"""Build PU-style target/weight arrays for prepared per-token data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_per_token.label_buckets import build_pu_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create {split}_targets.npy and {split}_weights.npy.")
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--weak-negative-weight", type=float, default=0.2)
    parser.add_argument("--trusted-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.7)
    parser.add_argument("--boundary-soften-tokens", type=int, default=0)
    parser.add_argument("--boundary-target", type=float, default=0.75)
    parser.add_argument("--trusted-mask-suffix", default="trusted_negative_mask",
                        help="Optional {split}_<suffix>.npy boolean mask name.")
    parser.add_argument("--hard-mask-suffix", default="hard_negative_mask",
                        help="Optional {split}_<suffix>.npy boolean mask name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, dict] = {}
    for split in args.splits:
        labels_path = args.prepared_dir / f"{split}_labels.npy"
        if not labels_path.exists():
            raise SystemExit(f"missing labels: {labels_path}")
        labels = np.load(labels_path, mmap_mode="r")
        trusted_path = args.prepared_dir / f"{split}_{args.trusted_mask_suffix}.npy"
        hard_path = args.prepared_dir / f"{split}_{args.hard_mask_suffix}.npy"
        trusted = np.load(trusted_path, mmap_mode="r") if trusted_path.exists() else None
        hard = np.load(hard_path, mmap_mode="r") if hard_path.exists() else None
        targets, weights = build_pu_targets(
            labels,
            trusted_negative_mask=trusted,
            hard_negative_mask=hard,
            weak_negative_weight=args.weak_negative_weight,
            trusted_negative_weight=args.trusted_negative_weight,
            hard_negative_weight=args.hard_negative_weight,
            boundary_soften_tokens=args.boundary_soften_tokens,
            boundary_target=args.boundary_target,
        )
        np.save(args.prepared_dir / f"{split}_targets.npy", targets)
        np.save(args.prepared_dir / f"{split}_weights.npy", weights)
        summary[split] = {
            "shape": list(labels.shape),
            "mean_weight": float(weights.mean()),
            "positive_tokens": int((labels == 1).sum()),
            "weighted_tokens": int((weights > 0).sum()),
            "trusted_mask_loaded": trusted is not None,
            "hard_mask_loaded": hard is not None,
        }
    (args.prepared_dir / "pu_weights_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
