"""Parse BGC-Prophet's pre-computed per-genome regions JSON.

BGC-Prophet publishes its 9-genome predictions as a single JSON of the form::

    {"<genome_id>": [[contig, start, end], ...], ...}

with 0-based half-open coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_bgc_prophet_file(
    regions_json: Path,
    genome_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    with Path(regions_json).open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level of {regions_json}, got {type(data).__name__}")
    out: dict[str, list[dict]] = {}
    for gid, regs in data.items():
        if genome_ids is not None and gid not in genome_ids:
            continue
        unified: list[dict] = []
        for item in regs:
            if len(item) < 3:
                continue
            contig, start, end = item[0], int(item[1]), int(item[2])
            if end <= start:
                continue
            unified.append({
                "genome_id": gid,
                "contig_id": contig,
                "start_bp": start,
                "end_bp": end,
            })
        out[gid] = unified
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions-json", required=True, type=Path,
                        help="Path to bgc_prophet_regions.json")
    parser.add_argument("--genome-ids", default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", default="unknown")
    args = parser.parse_args()

    genome_ids = None
    if args.genome_ids:
        genome_ids = [g.strip() for g in args.genome_ids.split(",") if g.strip()]

    preds = parse_bgc_prophet_file(args.regions_json, genome_ids=genome_ids)
    n_pred = sum(len(v) for v in preds.values())
    n_with = sum(1 for v in preds.values() if v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "tool": "bgc_prophet",
            "version": args.version,
            "n_genomes": len(preds),
            "n_genomes_with_bgc": n_with,
            "n_predictions": n_pred,
            "predictions": preds,
        }, f, indent=2)
    print(f"bgc_prophet: {len(preds)} genomes, {n_with} with BGC, {n_pred} regions -> {args.out}")


if __name__ == "__main__":
    main()
