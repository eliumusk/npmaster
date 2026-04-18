"""Parse GECCO per-genome output directory into a unified region format."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_gecco_dir(
    results_dir: Path,
    genome_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Read a GECCO batch output dir and return {genome_id: [region_dict, ...]}.

    Layout expected: ``results_dir/<genome_id>/<genome_id>.clusters.tsv``.
    Genomes with a ``*.features.tsv`` but no ``*.clusters.tsv`` are treated as
    zero-prediction runs (GECCO completed and found nothing).
    """
    predictions: dict[str, list[dict]] = defaultdict(list)
    for gdir in sorted(Path(results_dir).iterdir()):
        if not gdir.is_dir():
            continue
        gid = gdir.name
        if genome_ids is not None and gid not in genome_ids:
            continue
        if not (gdir / f"{gid}.features.tsv").exists():
            continue
        cluster_tsv = gdir / f"{gid}.clusters.tsv"
        if not cluster_tsv.exists():
            predictions[gid] = []
            continue
        with cluster_tsv.open() as f:
            lines = f.readlines()
        if len(lines) < 2:
            predictions[gid] = []
            continue
        headers = lines[0].strip().split("\t")
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            row = dict(zip(headers, parts))
            try:
                start = int(row.get("start", 0)) - 1
                end = int(row.get("end", 0))
            except ValueError:
                continue
            if end <= start:
                continue
            predictions[gid].append({
                "genome_id": gid,
                "contig_id": row.get("sequence_id", ""),
                "start_bp": start,
                "end_bp": end,
                "predicted_type": row.get("type", "Unknown"),
                "score": float(row.get("max_p", 0) or 0),
                "cluster_id": row.get("cluster_id", ""),
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

    preds = parse_gecco_dir(args.results_dir, genome_ids=genome_ids)
    n_pred = sum(len(v) for v in preds.values())
    n_with = sum(1 for v in preds.values() if v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "tool": "gecco",
            "version": args.version,
            "n_genomes": len(preds),
            "n_genomes_with_bgc": n_with,
            "n_predictions": n_pred,
            "predictions": preds,
        }, f, indent=2)
    print(f"gecco: {len(preds)} genomes, {n_with} with BGC, {n_pred} regions -> {args.out}")


if __name__ == "__main__":
    main()
