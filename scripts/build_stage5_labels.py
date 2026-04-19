#!/usr/bin/env python3
"""Stage 5 label-variant builder.

Reads an existing window-metadata CSV (with columns ``label``,
``max_bgc_overlap_ratio``, ``nearest_bgc_distance``,
``overlaps_truncated_bgc``, ``hard_negative``, plus optional
``bgc_length_max``) and emits the same rows augmented with three
columns: ``stage5_label``, ``stage5_weight``, ``stage5_variant``.

Implemented variants (ladder documented in
``docs/stage5-data-audit.md §5``):

  baseline_v1
      Pass-through of existing labels. Weights: positive 1.0,
      negative 1.0, ambiguous 0.0, hard_negative 1.0.
  tight_pos_075_v1, tight_pos_090_v1
      Relabel by overlap threshold. Windows below the threshold
      become ambiguous (masked); negatives unchanged.
  short_bgc_upweight_v1
      Upweight positives whose source BGC is shorter than
      ``--short-max-bp`` (default 10 000 bp) by ``--short-weight``
      (default 4.0).
  truncated_exclude_v1
      Drop positives whose source BGC is ``overlaps_truncated_bgc == 1``
      by masking them (weight 0).
  hardpos_hardneg_v2
      Combines short_bgc_upweight_v1 + truncated_exclude_v1 and also
      upweights ``hard_negative == 1`` rows by ``--hardneg-weight``
      (default 4.0).

Embargo guard: if ``--embargo`` or ``embargoed_genome_ids`` is given,
the builder refuses to emit labels for rows belonging to an embargoed
assembly — defence-in-depth against 9-genome leakage into Stage 5
training data.

Example::

    python scripts/build_stage5_labels.py \\
        --input /data/muskliu/npmaster/data/stage4_results/p2_t2_hard_neg/windows_metadata_hardneg.csv \\
        --output /data/muskliu/npmaster/data/stage5/no9_labels/tight_pos_075_v1.csv \\
        --variant tight_pos_075_v1 \\
        --embargo configs/splits/nine_genome_embargo_v1.json \\
        --bgc-csv /data/muskliu/npmaster/data/bgc_annotations.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Variant implementations
# ---------------------------------------------------------------------------


def _start_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Initialise stage5 columns from the existing labels.

    The weight convention used throughout:
      - original positive → 1.0
      - original ambiguous → 0.0 (masked from loss)
      - original negative → 1.0
    """
    label = df["label"].astype(str).copy()
    weight = pd.Series(
        np.where(label == "ambiguous", 0.0, 1.0),
        index=df.index,
        dtype=float,
    )
    return label, weight


def _baseline_v1(df: pd.DataFrame, **_: object) -> tuple[pd.Series, pd.Series]:
    return _start_columns(df)


def _tight_threshold(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.Series, pd.Series]:
    label, weight = _start_columns(df)
    overlap = df["max_bgc_overlap_ratio"].astype(float)
    # Any row the CSV calls "positive" but whose overlap is below the new
    # stricter threshold is demoted to ambiguous (mask from training).
    demote = (label == "positive") & (overlap < threshold)
    label = label.mask(demote, "ambiguous")
    weight = weight.mask(demote, 0.0)
    return label, weight


def _tight_pos_075_v1(
    df: pd.DataFrame, **_: object
) -> tuple[pd.Series, pd.Series]:
    return _tight_threshold(df, 0.75)


def _tight_pos_090_v1(
    df: pd.DataFrame, **_: object
) -> tuple[pd.Series, pd.Series]:
    return _tight_threshold(df, 0.90)


def _short_bgc_upweight_v1(
    df: pd.DataFrame,
    short_max_bp: float = 10_000.0,
    short_positive_weight: float = 4.0,
    **_: object,
) -> tuple[pd.Series, pd.Series]:
    label, weight = _start_columns(df)
    if "bgc_length_max" not in df.columns:
        raise KeyError(
            "short_bgc_upweight_v1 needs a 'bgc_length_max' column "
            "(join with bgc_annotations.csv before calling)."
        )
    bgc_len = df["bgc_length_max"].astype(float)
    is_short_pos = (label == "positive") & (bgc_len.fillna(np.inf) < short_max_bp)
    weight = weight.mask(is_short_pos, float(short_positive_weight))
    return label, weight


def _truncated_exclude_v1(
    df: pd.DataFrame, **_: object
) -> tuple[pd.Series, pd.Series]:
    label, weight = _start_columns(df)
    truncated = df["overlaps_truncated_bgc"].fillna(0).astype(int) == 1
    demote = (label == "positive") & truncated
    label = label.mask(demote, "ambiguous")
    weight = weight.mask(demote, 0.0)
    return label, weight


def _hardpos_hardneg_v2(
    df: pd.DataFrame,
    short_max_bp: float = 10_000.0,
    short_positive_weight: float = 4.0,
    hardneg_weight: float = 4.0,
    **_: object,
) -> tuple[pd.Series, pd.Series]:
    # truncated-exclude first (must happen before upweight so that
    # truncated short positives are masked, not upweighted).
    label, weight = _truncated_exclude_v1(df)

    bgc_len = df.get("bgc_length_max")
    if bgc_len is None:
        raise KeyError(
            "hardpos_hardneg_v2 needs a 'bgc_length_max' column."
        )
    bgc_len = bgc_len.astype(float)
    is_short_pos = (label == "positive") & (
        bgc_len.fillna(np.inf) < short_max_bp
    )
    weight = weight.mask(is_short_pos, float(short_positive_weight))

    hard_neg = df["hard_negative"].fillna(0).astype(int) == 1
    is_hardneg = (label == "negative") & hard_neg
    weight = weight.mask(is_hardneg, float(hardneg_weight))
    return label, weight


VARIANTS: dict[str, Callable[..., tuple[pd.Series, pd.Series]]] = {
    "baseline_v1": _baseline_v1,
    "tight_pos_075_v1": _tight_pos_075_v1,
    "tight_pos_090_v1": _tight_pos_090_v1,
    "short_bgc_upweight_v1": _short_bgc_upweight_v1,
    "truncated_exclude_v1": _truncated_exclude_v1,
    "hardpos_hardneg_v2": _hardpos_hardneg_v2,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _check_embargo(df: pd.DataFrame, embargoed: set[str]) -> None:
    if not embargoed or "genome_id" not in df.columns:
        return
    hits = df.loc[df["genome_id"].isin(embargoed), "genome_id"].unique()
    if len(hits):
        raise ValueError(
            "Refusing to emit Stage 5 labels: embargoed genome IDs found in "
            f"input: {sorted(hits)}"
        )


def apply_variant(
    df: pd.DataFrame,
    variant: str,
    *,
    embargoed_genome_ids: set[str] | None = None,
    **variant_kwargs: object,
) -> pd.DataFrame:
    """Apply a named Stage 5 variant and return a new frame.

    The returned frame has the same rows as ``df`` with three appended
    columns: ``stage5_label``, ``stage5_weight``, ``stage5_variant``.
    """
    if variant not in VARIANTS:
        raise KeyError(
            f"Unknown variant '{variant}'. Known: {sorted(VARIANTS)}"
        )
    _check_embargo(df, embargoed_genome_ids or set())

    label, weight = VARIANTS[variant](df, **variant_kwargs)
    out = df.copy()
    out["stage5_label"] = label.to_numpy()
    out["stage5_weight"] = weight.astype(float).to_numpy()
    out["stage5_variant"] = variant
    return out


def _join_bgc_length(
    windows: pd.DataFrame, bgc_csv: Path
) -> pd.DataFrame:
    """Add ``bgc_length_max`` column = max length of any BGC overlapping a
    positive window. Only required by variants that need per-window
    source-BGC length (``short_bgc_upweight_v1``, ``hardpos_hardneg_v2``).

    Approximation: joins at the (genome_id, contig_id) level and
    uses a window-vs-BGC interval overlap test. This is O(windows ×
    BGCs_in_contig) — acceptable for the real table size (1 M windows,
    20 k BGCs distributed across 2 k genomes; each contig has at most
    a few BGCs).
    """
    bgc = pd.read_csv(bgc_csv)
    required = {"genome_id", "contig_id", "start_bp", "end_bp", "bgc_length"}
    missing = required - set(bgc.columns)
    if missing:
        raise KeyError(f"bgc_csv missing columns: {sorted(missing)}")

    if {"genome_id", "contig_id", "window_start", "window_end"} - set(
        windows.columns
    ):
        raise KeyError("windows CSV missing required coordinate columns")

    # Group BGCs by (genome_id, contig_id) for quick per-window lookup.
    grouped = {
        key: sub[["start_bp", "end_bp", "bgc_length"]].to_numpy()
        for key, sub in bgc.groupby(["genome_id", "contig_id"], sort=False)
    }

    lengths = np.full(len(windows), np.nan, dtype=float)
    keys = windows[["genome_id", "contig_id"]].to_numpy()
    starts = windows["window_start"].to_numpy()
    ends = windows["window_end"].to_numpy()

    for i in range(len(windows)):
        entry = grouped.get((keys[i, 0], keys[i, 1]))
        if entry is None:
            continue
        # rows = (start, end, length); overlap iff start<we AND end>ws
        mask = (entry[:, 0] < ends[i]) & (entry[:, 1] > starts[i])
        if mask.any():
            lengths[i] = float(entry[mask, 2].max())

    out = windows.copy()
    out["bgc_length_max"] = lengths
    return out


def _load_embargo_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data.get("embargoed_genome_ids")
    if not isinstance(ids, list):
        raise ValueError(f"{path} missing 'embargoed_genome_ids' list")
    return set(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", required=True, choices=sorted(VARIANTS.keys())
    )
    parser.add_argument(
        "--embargo",
        type=Path,
        default=None,
        help="Optional embargo-config JSON; fail if any embargoed genome "
        "ID is present in the input CSV.",
    )
    parser.add_argument(
        "--bgc-csv",
        type=Path,
        default=None,
        help="Optional BGC annotation CSV to join for bgc_length_max; "
        "required by variants that look at source BGC length "
        "(short_bgc_upweight_v1, hardpos_hardneg_v2).",
    )
    parser.add_argument("--short-max-bp", type=float, default=10_000.0)
    parser.add_argument("--short-weight", type=float, default=4.0)
    parser.add_argument("--hardneg-weight", type=float, default=4.0)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)

    variants_needing_bgc_len = {
        "short_bgc_upweight_v1",
        "hardpos_hardneg_v2",
    }
    if args.variant in variants_needing_bgc_len and "bgc_length_max" not in df.columns:
        if args.bgc_csv is None:
            parser.error(
                f"Variant {args.variant} requires --bgc-csv or a "
                "'bgc_length_max' column already in the input."
            )
        df = _join_bgc_length(df, args.bgc_csv)

    embargoed = _load_embargo_ids(args.embargo)

    out = apply_variant(
        df,
        args.variant,
        embargoed_genome_ids=embargoed,
        short_max_bp=args.short_max_bp,
        short_positive_weight=args.short_weight,
        hardneg_weight=args.hardneg_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Emit a compact summary on stderr for logging pipelines.
    summary = {
        "variant": args.variant,
        "rows_total": int(len(out)),
        "rows_positive": int((out["stage5_label"] == "positive").sum()),
        "rows_ambiguous": int((out["stage5_label"] == "ambiguous").sum()),
        "rows_negative": int((out["stage5_label"] == "negative").sum()),
        "sum_weight_positive": float(
            out.loc[out["stage5_label"] == "positive", "stage5_weight"].sum()
        ),
        "sum_weight_negative": float(
            out.loc[out["stage5_label"] == "negative", "stage5_weight"].sum()
        ),
        "rows_masked": int((out["stage5_weight"] == 0.0).sum()),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
