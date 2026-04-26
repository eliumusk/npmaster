#!/usr/bin/env python3
"""Evo2-LR multiscale honest readout on 9-genome.

CLI wrapper around :func:`decoding.evo2_lr_multiscale.run_readout`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decoding.evo2_lr_multiscale import run_readout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evo2-LR multiscale 9-genome readout")
    parser.add_argument("--primary-preds", type=Path, required=True)
    parser.add_argument("--short-preds", type=Path, required=True)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run_readout(
        primary_preds=args.primary_preds,
        short_preds=args.short_preds,
        gt_json=args.gt_json,
        out_json=args.out_json,
        bootstrap=args.bootstrap,
    )

if __name__ == "__main__":
    main()
