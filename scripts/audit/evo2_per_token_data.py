#!/usr/bin/env python3
"""Audit per-token U-Net training metadata before benchmark claims."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_per_token.audit import audit_training_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit per-token U-Net metadata and 9-genome exposure.")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--bgc-annot", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--fail-on-nine-genome-overlap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_training_metadata(args.labels_csv, args.bgc_annot)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["nine_genome_overlap"], indent=2))
    print(f"wrote {args.out_json}")
    if args.fail_on_nine_genome_overlap and audit["nine_genome_overlap"]["window_count"] > 0:
        raise SystemExit("9-genome windows are present in training metadata")


if __name__ == "__main__":
    main()
