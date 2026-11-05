"""Audit helpers for per-token U-Net training metadata.

The local repository does not carry the full server-side data tables, so these
helpers keep the checks file-based and deterministic. They are intended to run
against copied or server-resident CSV metadata before promoting a 9-genome claim.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

try:
    from benchmark.nine_genome import GENOME_IDS
except Exception:  # pragma: no cover - keeps the helper importable in isolation.
    GENOME_IDS = [
        "GCA_000158915.1",
        "GCA_000158875.1",
        "GCA_000154945.1",
        "GCA_000158895.1",
        "GCA_000156435.1",
        "GCA_000568915.1",
        "GCA_000158815.1",
        "GCA_000568255.1",
        "GCA_000156475.1",
    ]


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _counter_dict(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def _length_buckets(rows: list[dict[str, str]]) -> dict[str, int]:
    buckets = Counter()
    for row in rows:
        start = int(row.get("window_start", row.get("start", 0)))
        end = int(row.get("window_end", row.get("end", start)))
        length = max(0, end - start)
        if length < 4_096:
            buckets["<4kb"] += 1
        elif length < 8_192:
            buckets["4-8kb"] += 1
        elif length < 16_384:
            buckets["8-16kb"] += 1
        else:
            buckets[">=16kb"] += 1
    return dict(sorted(buckets.items()))


def audit_training_metadata(
    windows_csv: str | Path,
    bgc_annotations_csv: str | Path,
    nine_genome_ids: set[str] | None = None,
) -> dict:
    """Summarize training metadata and detect 9-genome exposure.

    The audit intentionally avoids interpreting antiSMASH strict/relaxed mode
    unless that provenance exists in the input files; it reports this as
    ``unknown`` so promotion decisions do not accidentally assume cleanliness.
    """
    windows = _read_csv_rows(windows_csv)
    bgc_rows = _read_csv_rows(bgc_annotations_csv)
    nine_ids = set(nine_genome_ids or GENOME_IDS)

    overlap_ids = sorted({row["genome_id"] for row in windows if row.get("genome_id") in nine_ids})
    overlap_count = sum(1 for row in windows if row.get("genome_id") in nine_ids)

    return {
        "windows": {
            "n_total": len(windows),
            "by_split": _counter_dict(windows, "split"),
            "by_label": _counter_dict(windows, "label"),
            "by_length_bucket": _length_buckets(windows),
            "n_genomes": len({row.get("genome_id", "") for row in windows}),
        },
        "bgc_annotations": {
            "n_regions": len(bgc_rows),
            "n_genomes": len({row.get("genome_id", "") for row in bgc_rows}),
        },
        "nine_genome_overlap": {
            "window_count": overlap_count,
            "genome_ids": overlap_ids,
        },
        "provenance": {
            "antismash_mode": "unknown",
            "note": "Verify antiSMASH version and strict/relaxed settings on the server run manifest.",
        },
    }
