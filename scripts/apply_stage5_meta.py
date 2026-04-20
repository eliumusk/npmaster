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
    allow_promotion: bool = False,
) -> dict:
    """Return a new meta-arrays dict with Stage 5 labels applied.

    ``allow_promotion`` (Stage 5c-1+): set True when the variant's label
    source is an entirely new GT set (e.g. proto_core coords) that
    legitimately promotes some rows from negative/ambiguous to positive.
    Promoted rows that have all-zero type channels get
    ``Other`` (channel 6) = 1 as a default training target; the
    downstream `bgc_score = max over types` eval still treats them as
    "BGC" without needing per-core type labels.
    """
    _verify_alignment(meta, frame)

    required_cols = {"stage5_label", "stage5_weight"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(
            f"stage5 CSV missing columns: {sorted(missing)}"
        )

    stage5_label = frame["stage5_label"].astype(str).to_numpy()
    stage5_weight = frame["stage5_weight"].astype(float).to_numpy()
    orig_label = np.array(meta["label"], dtype=object)

    new_labels = meta["labels"].copy()
    new_label_str = orig_label.copy()

    # Mask any row the variant demoted to ambiguous.
    mask_rows = stage5_label == "ambiguous"
    new_labels[mask_rows] = 0.0
    new_label_str[mask_rows] = "ambiguous"

    promote_rows = (stage5_label == "positive") & (orig_label != "positive")
    if promote_rows.any():
        if not allow_promotion:
            raise ValueError(
                f"{int(promote_rows.sum())} rows request promotion from "
                "non-positive to positive — pass allow_promotion=True "
                "when the variant replaces the GT source (e.g. Stage "
                "5c-1 proto_core labels)."
            )
        # Assign type-channel Other (index 6) = 1, clear channel 7
        # (is_negative), zero the rest. Preserves any existing type
        # multi-hot only if it was already positive-typed (which we
        # skip by construction since promote_rows excludes originals).
        new_labels[promote_rows] = 0.0
        new_labels[promote_rows, 6] = 1.0
        new_label_str[promote_rows] = "positive"

    # Originally-negative rows that are still negative stay; rows going
    # negative → ambiguous or negative → positive are handled above.
    # We allow the "negative → ambiguous" drift under allow_promotion=True
    # (proto_core flips more windows into ambiguous because distance
    # checks change).
    neg_drift = (orig_label == "negative") & (stage5_label != "negative")
    if neg_drift.any() and not allow_promotion:
        raise ValueError(
            f"{int(neg_drift.sum())} originally-negative rows changed "
            "label under the variant — unexpected under the Stage 5a/5b "
            "variant ladder. Pass allow_promotion=True for Stage 5c-1+."
        )
    # Under allow_promotion, if a row was originally negative but is now
    # ambiguous under the new GT, zero out the channels (mask).
    neg_to_amb = (orig_label == "negative") & (stage5_label == "ambiguous")
    new_labels[neg_to_amb] = 0.0
    # And if originally ambiguous but now negative, mark as negative by
    # setting channel 7 = 1 and others to 0.
    amb_to_neg = (orig_label == "ambiguous") & (stage5_label == "negative")
    new_labels[amb_to_neg] = 0.0
    new_labels[amb_to_neg, 7] = 1.0
    new_label_str[amb_to_neg] = "negative"
    # And if originally positive but the new GT calls it negative (e.g. a
    # region-positive window that doesn't overlap any proto_core), demote
    # to negative cleanly.
    pos_to_neg = (orig_label == "positive") & (stage5_label == "negative")
    new_labels[pos_to_neg] = 0.0
    new_labels[pos_to_neg, 7] = 1.0
    new_label_str[pos_to_neg] = "negative"

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
    parser.add_argument(
        "--allow-promotion",
        action="store_true",
        help="Allow non-positive → positive promotion. Required for "
        "Stage 5c-1 core-GT variants where the GT source changes.",
    )
    args = parser.parse_args(argv)

    meta = _load_meta(args.meta)
    frame = pd.read_csv(args.stage5_csv)
    out = apply_stage5_to_meta(meta, frame, allow_promotion=args.allow_promotion)
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
