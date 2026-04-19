#!/usr/bin/env python3
"""Patch an existing ``meta_arrays.pkl`` with Stage 5 label variant output.

Reads:
  --meta           the base meta_arrays pickle (schema: split, label,
                   genome_id, contig_id, window_start, phylum,
                   edge_padded, labels(N,8), hard_negative(N))
  --stage5-csv     the output of ``build_stage5_labels.py`` (same row
                   order as the base windows CSV — mandatory precondition)

Writes ``--output``: a new meta-arrays pickle with:
  - ``labels``  patched row-by-row so that ``stage5_label == "ambiguous"``
    rows become all-zero (mask), "positive" rows keep their original
    type one-hot, and "negative" rows keep channel 7 = 1.
  - ``label`` (string) updated to match ``stage5_label``.
  - ``hard_negative`` OR'd with rows whose ``stage5_weight > 1``, so
    train.py's existing ``--hard-neg-weight`` scalar produces the same
    effect as a per-row weight multiplier (legitimate because each
    variant only uses one upweight factor).
  - All other arrays pass through unchanged.

The applier refuses to run if the CSV row order does not match the meta
arrays (verified by (genome_id, contig_id, window_start) triple equality
on a sample).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _verify_alignment(meta: dict, frame: pd.DataFrame) -> None:
    n_meta = int(meta["labels"].shape[0])
    if len(frame) != n_meta:
        raise ValueError(
            f"row count mismatch: meta has {n_meta} rows, "
            f"stage5 CSV has {len(frame)} — re-run build_stage5_labels "
            "on the same windows CSV the meta was built from."
        )
    # Spot-check row alignment on (genome_id, contig_id, window_start).
    required_cols = {"genome_id", "contig_id", "window_start"}
    if not required_cols.issubset(frame.columns):
        missing = sorted(required_cols - set(frame.columns))
        raise ValueError(
            f"stage5 CSV missing alignment columns: {missing}"
        )
    # Use a deterministic sample of indices across the array.
    sample_idx = np.linspace(
        0, n_meta - 1, num=min(256, n_meta), dtype=int
    )
    for col in ("genome_id", "contig_id"):
        left = meta[col][sample_idx].astype(str)
        right = frame[col].to_numpy()[sample_idx].astype(str)
        if not np.array_equal(left, right):
            bad = int(np.argmax(left != right))
            raise ValueError(
                f"row-order mismatch at sample idx {sample_idx[bad]} "
                f"on column {col!r}: meta={left[bad]!r}, "
                f"csv={right[bad]!r}. Re-run build_stage5_labels on "
                "the matching windows CSV."
            )
    left_ws = meta["window_start"][sample_idx].astype(np.int64)
    right_ws = (
        frame["window_start"].to_numpy()[sample_idx].astype(np.int64)
    )
    if not np.array_equal(left_ws, right_ws):
        raise ValueError("row-order mismatch on window_start samples.")


def apply_stage5_to_meta(
    meta: dict,
    frame: pd.DataFrame,
) -> dict:
    """Return a new meta-arrays dict with Stage 5 labels applied."""
    _verify_alignment(meta, frame)

    required_cols = {"stage5_label", "stage5_weight"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(
            f"stage5 CSV missing columns: {sorted(missing)}"
        )

    stage5_label = frame["stage5_label"].astype(str).to_numpy()
    stage5_weight = frame["stage5_weight"].astype(float).to_numpy()

    new_labels = meta["labels"].copy()
    new_label_str = np.array(meta["label"], dtype=object).copy()

    # Mask any row the variant demoted to ambiguous.
    mask_rows = stage5_label == "ambiguous"
    new_labels[mask_rows] = 0.0
    new_label_str[mask_rows] = "ambiguous"

    # No promotion: a row that was not originally positive cannot become
    # positive just because stage5_label says so. The variants in our
    # ladder only ever demote or mask, so this is a safety invariant
    # rather than a limitation.
    promote_rows = (stage5_label == "positive") & (
        np.array(meta["label"], dtype=object) != "positive"
    )
    if promote_rows.any():
        raise ValueError(
            f"{int(promote_rows.sum())} rows request promotion from "
            "non-positive to positive — current variants should never "
            "do this. Bug in the label builder or a schema drift."
        )

    # Negatives stay as negatives (channel 7 = 1). Our variants do not
    # currently move rows between negative and ambiguous; assert this.
    neg_drift = (np.array(meta["label"], dtype=object) == "negative") & (
        stage5_label != "negative"
    )
    if neg_drift.any():
        raise ValueError(
            f"{int(neg_drift.sum())} originally-negative rows changed "
            "label under the variant — unexpected."
        )

    # Route upweight into the existing hard_negative loss path.
    new_hard_neg = meta["hard_negative"].astype(np.int32).copy()
    upweighted = stage5_weight > 1.0
    new_hard_neg = np.where(upweighted, 1, new_hard_neg).astype(np.int32)

    new_meta = dict(meta)
    new_meta["labels"] = new_labels
    new_meta["label"] = new_label_str
    new_meta["hard_negative"] = new_hard_neg

    return new_meta


def _load_meta(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _dump_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(meta, f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--stage5-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a summary JSON describing the patch.",
    )
    args = parser.parse_args(argv)

    meta = _load_meta(args.meta)
    frame = pd.read_csv(args.stage5_csv)
    out = apply_stage5_to_meta(meta, frame)
    _dump_meta(out, args.output)

    summary = {
        "base_meta": str(args.meta),
        "stage5_csv": str(args.stage5_csv),
        "output": str(args.output),
        "rows": int(out["labels"].shape[0]),
        "positive_rows_before": int(
            (np.array(meta["label"], dtype=object) == "positive").sum()
        ),
        "positive_rows_after": int(
            (np.array(out["label"], dtype=object) == "positive").sum()
        ),
        "ambiguous_rows_before": int(
            (np.array(meta["label"], dtype=object) == "ambiguous").sum()
        ),
        "ambiguous_rows_after": int(
            (np.array(out["label"], dtype=object) == "ambiguous").sum()
        ),
        "hard_negative_before": int(
            np.sum(np.asarray(meta["hard_negative"]) == 1)
        ),
        "hard_negative_after": int(np.sum(out["hard_negative"] == 1)),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
