#!/usr/bin/env python3
"""Parse antiSMASH output JSONs into a tight-core BGC annotation CSV.

Walks every genome directory under ``--antismash-root``, reads the
per-genome JSON, and for every feature of type ``proto_core`` records a
row with ``(genome_id, contig_id, start_bp, end_bp, bgc_length,
bgc_types, bgc_product)``. Output schema is a superset of
``bgc_annotations.csv`` so downstream scripts can swap one for the
other via ``--bgc-csv``.

Rationale: antiSMASH emits three nested coordinate layers —
``proto_core`` (actual biosynthetic core genes, ~1-20 kb),
``cand_cluster`` / ``protocluster`` (same coords wrapped with metadata),
and ``region`` (core + configurable flank, typically +10 kb each side).
``bgc_annotations.csv`` was built from ``region`` → wide labels.
``proto_core`` is the layer that matches the 9-genome human-curated
tight-core GT style, so it's the correct val-GT source for Stage 5c-0.

Parallelized across CPUs. Typical runtime on 1 965 genomes with 128
workers: a few minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
from pathlib import Path


_LOC_RE = re.compile(r"\[(\d+):(\d+)\](?:\(([+-?])\))?")


def _parse_location(loc: str) -> tuple[int, int] | None:
    """antiSMASH locations look like '[760663:781635](+)'. Handle
    compound / join locations by taking the outer envelope."""
    if not loc:
        return None
    # Pick the leftmost start and rightmost end across all sub-intervals.
    matches = _LOC_RE.findall(loc)
    if not matches:
        return None
    starts = [int(m[0]) for m in matches]
    ends = [int(m[1]) for m in matches]
    return min(starts), max(ends)


def _extract_product(feature: dict) -> str:
    qual = feature.get("qualifiers") or {}
    products = qual.get("product") or []
    if isinstance(products, list):
        return ";".join(str(p) for p in products)
    return str(products)


def _process_one_genome(job: tuple[str, Path]) -> list[dict]:
    genome_id, json_path = job
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    rows: list[dict] = []
    for record in data.get("records") or []:
        contig_id = record.get("id") or record.get("name")
        if not contig_id:
            continue
        core_idx = 0
        for feat in record.get("features") or []:
            if feat.get("type") != "proto_core":
                continue
            loc = _parse_location(str(feat.get("location", "")))
            if loc is None:
                continue
            start, end = loc
            if end <= start:
                continue
            core_idx += 1
            rows.append(
                {
                    "bgc_id": f"{genome_id}_CORE_{core_idx:03d}",
                    "genome_id": genome_id,
                    "contig_id": contig_id,
                    "start_bp": int(start),
                    "end_bp": int(end),
                    "bgc_length": int(end - start),
                    "bgc_types": _extract_product(feat),
                    "bgc_product": _extract_product(feat),
                    "is_truncated": 0,
                    "phylum": "",
                    "source": "antismash_proto_core",
                }
            )
    return rows


def iter_genome_jobs(antismash_root: Path) -> list[tuple[str, Path]]:
    jobs = []
    for genome_dir in sorted(antismash_root.iterdir()):
        if not genome_dir.is_dir():
            continue
        gid = genome_dir.name
        for cand in (genome_dir / f"{gid}.json", genome_dir / "regions.json"):
            if cand.is_file():
                jobs.append((gid, cand))
                break
    return jobs


def merge_phylum(
    rows: list[dict], bgc_csv: Path | None
) -> list[dict]:
    """Propagate phylum labels from the original bgc_annotations.csv when
    available (downstream scripts expect it)."""
    if bgc_csv is None or not bgc_csv.is_file():
        return rows
    try:
        import pandas as pd

        src = pd.read_csv(bgc_csv)
        lookup = dict(
            zip(src["genome_id"].astype(str), src["phylum"].astype(str))
        )
    except Exception:
        return rows
    for r in rows:
        r["phylum"] = lookup.get(r["genome_id"], "")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--antismash-root",
        type=Path,
        default=Path("/data/muskliu/npmaster/data/antismash_results"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/data/muskliu/npmaster/data/bgc_annotations_core.csv"
        ),
    )
    parser.add_argument(
        "--merge-phylum-from",
        type=Path,
        default=Path("/data/muskliu/npmaster/data/bgc_annotations.csv"),
        help="Optional: fill phylum column from an existing "
        "bgc_annotations.csv (per-genome lookup).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
    )
    args = parser.parse_args()

    jobs = iter_genome_jobs(args.antismash_root)
    if not jobs:
        print(f"no antiSMASH JSONs found under {args.antismash_root}")
        return 1
    print(f"[parse_core] {len(jobs)} genomes, {args.n_workers} workers")

    rows: list[dict] = []
    with mp.Pool(args.n_workers) as pool:
        for result in pool.imap_unordered(_process_one_genome, jobs, chunksize=8):
            rows.extend(result)

    print(
        f"[parse_core] extracted {len(rows)} proto_core intervals across "
        f"{len({r['genome_id'] for r in rows})} genomes"
    )

    rows = merge_phylum(rows, args.merge_phylum_from)

    # Length distribution peek.
    import statistics

    lengths = [r["bgc_length"] for r in rows]
    if lengths:
        print(
            "[parse_core] length bp — min={} p25={} median={} p75={} "
            "p95={} max={}".format(
                min(lengths),
                int(statistics.quantiles(lengths, n=4)[0]),
                int(statistics.median(lengths)),
                int(statistics.quantiles(lengths, n=4)[2]),
                int(statistics.quantiles(lengths, n=20)[-1]),
                max(lengths),
            )
        )
        # Bucket counts
        for lo, hi, name in [
            (0, 5000, "<5kb"),
            (5000, 10000, "5-10kb"),
            (10000, 20000, "10-20kb"),
            (20000, 50000, "20-50kb"),
            (50000, 10**9, ">=50kb"),
        ]:
            n = sum(1 for L in lengths if lo <= L < hi)
            print(f"[parse_core]   {name}: {n}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "bgc_id",
            "genome_id",
            "contig_id",
            "start_bp",
            "end_bp",
            "bgc_length",
            "bgc_types",
            "bgc_product",
            "is_truncated",
            "phylum",
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort for determinism.
        rows.sort(key=lambda r: (r["genome_id"], r["contig_id"], r["start_bp"]))
        writer.writerows(rows)
    print(f"[parse_core] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
