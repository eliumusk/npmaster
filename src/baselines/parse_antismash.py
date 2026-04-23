"""Parse antiSMASH per-genome output directory into a unified region format.

Handles the native antiSMASH 7.x ``<gca>.json`` layout, where each record has
``areas`` with 1-based inclusive ``start`` / ``end`` coordinates. The contig id
is read from ``rec.id``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_antismash_dir(
    results_dir: Path,
    genome_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Return ``{genome_id: [region_dict, ...]}`` with 0-based half-open coords.

    Layout expected: ``results_dir/<genome_id>/<genome_id>.json``.
    """
    predictions: dict[str, list[dict]] = defaultdict(list)
    for gdir in sorted(Path(results_dir).iterdir()):
        if not gdir.is_dir():
            continue
        gid = gdir.name
        if genome_ids is not None and gid not in genome_ids:
            continue
        json_path = gdir / f"{gid}.json"
        if not json_path.exists():
            predictions[gid] = []
            continue
        with json_path.open() as f:
            data = json.load(f)
        for rec in data.get("records", []):
            contig = rec.get("id", "")
            for area in rec.get("areas", []):
                start_1based = area.get("start")
                end_1based = area.get("end")
                if start_1based is None or end_1based is None:
                    continue
                start = int(start_1based) - 1
                end = int(end_1based)
                if end <= start:
                    continue
                products = area.get("products") or []
                predictions[gid].append({
                    "genome_id": gid,
                    "contig_id": contig,
                    "start_bp": start,
                    "end_bp": end,
                    "predicted_type": ",".join(products) if products else "",
                })
    return dict(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--genome-ids", default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", default="unknown")
    args = parser.parse_args()

    genome_ids = None
    if args.genome_ids:
        genome_ids = [g.strip() for g in args.genome_ids.split(",") if g.strip()]

    preds = parse_antismash_dir(args.results_dir, genome_ids=genome_ids)
    n_pred = sum(len(v) for v in preds.values())
    n_with = sum(1 for v in preds.values() if v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "tool": "antismash",
            "version": args.version,
            "n_genomes": len(preds),
            "n_genomes_with_bgc": n_with,
            "n_predictions": n_pred,
            "predictions": preds,
        }, f, indent=2)
    print(f"antismash: {len(preds)} genomes, {n_with} with BGC, {n_pred} regions -> {args.out}")


if __name__ == "__main__":
    main()
