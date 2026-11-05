#!/usr/bin/env python3
"""Select and freeze per-token U-Net operating points without 9-genome input."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo2_per_token.selection import select_operating_point, write_frozen_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select non-9g operating point and write protocol JSON.")
    parser.add_argument("--candidates-json", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-regions", required=True, type=int)
    parser.add_argument("--min-as-recall", type=float, default=0.0)
    parser.add_argument("--selection-source", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.candidates_json.read_text())
    candidates = payload["candidates"] if isinstance(payload, dict) else payload
    chosen = select_operating_point(
        candidates,
        max_regions=args.max_regions,
        min_as_recall=args.min_as_recall,
    )
    metrics = chosen.get("metrics", {})
    op = {
        "name": chosen["name"],
        "threshold": chosen.get("threshold"),
        "min_len_bp": chosen.get("min_len_bp"),
        "max_gap_bp": chosen.get("max_gap_bp", 500),
        "selection_metrics": metrics,
    }
    sources = args.selection_source or sorted({str(c.get("source_dataset", "unknown")) for c in candidates})
    write_frozen_protocol(
        args.out_json,
        checkpoint=args.checkpoint,
        operating_points=[op],
        selection_sources=sources,
    )
    print(json.dumps({"chosen": op, "protocol": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
