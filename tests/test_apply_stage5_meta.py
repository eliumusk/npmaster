"""Tests for the Stage 5 meta-array applier.

The applier takes an existing ``meta_arrays.pkl`` (produced by the
mainline mining pipeline, schema:
``split``, ``label``, ``genome_id``, ``contig_id``, ``window_start``,
``phylum``, ``edge_padded``, ``labels`` (N, 8), ``hard_negative``)
and a Stage 5 labelled window CSV (emitted by ``build_stage5_labels``
with ``stage5_label`` / ``stage5_weight`` / ``stage5_variant``).

It emits a new meta-arrays dict with ``labels`` (N, 8) patched so that:
  - rows with ``stage5_label == "ambiguous"`` become masked (all zeros)
  - rows with ``stage5_label == "positive"`` retain their original
    positive type one-hot if they were positive, or are skipped if
    they were not already positive (no new positives manufactured —
    our variants only demote, never promote)
  - rows with ``stage5_label == "negative"`` keep channel 7 = 1

It also merges ``stage5_weight > 1`` rows into ``hard_negative = 1``
so the existing train.py loss-weighting code path is reused.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_applier = _load_module(
    "scripts/apply_stage5_meta.py", "apply_stage5_meta"
)
apply_stage5_to_meta = _applier.apply_stage5_to_meta


def _base_meta(n: int = 4) -> dict:
    return {
        "split": np.array(["train"] * n, dtype=object),
        "label": np.array(
            ["positive", "positive", "ambiguous", "negative"], dtype=object
        ),
        "genome_id": np.array([f"G{i}" for i in range(n)], dtype=object),
        "contig_id": np.array([f"C{i}" for i in range(n)], dtype=object),
        "window_start": np.arange(n, dtype=np.int64) * 1000,
        "phylum": np.array(["p"] * n, dtype=object),
        "edge_padded": np.zeros(n, dtype=np.int32),
        "labels": np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0],   # positive Terpene
                [1, 0, 0, 0, 0, 0, 0, 0],   # positive Polyketide
                [0, 0, 0, 0, 0, 0, 0, 0],   # ambiguous
                [0, 0, 0, 0, 0, 0, 0, 1],   # negative
            ],
            dtype=np.float32,
        ),
        "hard_negative": np.zeros(n, dtype=np.int32),
    }


def _stage5_frame(labels: list[str], weights: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": [f"G{i}" for i in range(len(labels))],
            "contig_id": [f"C{i}" for i in range(len(labels))],
            "window_start": np.arange(len(labels)) * 1000,
            "stage5_label": labels,
            "stage5_weight": weights,
            "stage5_variant": "test_variant",
        }
    )


class ApplyTests(unittest.TestCase):
    def test_identity_passes_through_when_labels_match(self):
        meta = _base_meta()
        frame = _stage5_frame(
            ["positive", "positive", "ambiguous", "negative"],
            [1.0, 1.0, 0.0, 1.0],
        )
        out = apply_stage5_to_meta(meta, frame)
        np.testing.assert_array_equal(out["labels"], meta["labels"])
        np.testing.assert_array_equal(
            out["hard_negative"], meta["hard_negative"]
        )

    def test_demoted_positive_becomes_ambiguous(self):
        meta = _base_meta()
        frame = _stage5_frame(
            ["ambiguous", "positive", "ambiguous", "negative"],
            [0.0, 1.0, 0.0, 1.0],
        )
        out = apply_stage5_to_meta(meta, frame)
        # row 0: was positive Terpene → now all zeros
        np.testing.assert_array_equal(
            out["labels"][0], np.zeros(8, dtype=np.float32)
        )
        # row 1: unchanged
        np.testing.assert_array_equal(
            out["labels"][1], meta["labels"][1]
        )
        # label string also updated
        self.assertEqual(out["label"][0], "ambiguous")

    def test_upweighted_rows_flagged_as_hard_negative(self):
        """stage5_weight > 1 → hard_negative = 1 so train.py --hard-neg-weight
        picks them up through the existing loss-weight code path."""
        meta = _base_meta()
        frame = _stage5_frame(
            ["positive", "positive", "ambiguous", "negative"],
            [4.0, 1.0, 0.0, 4.0],
        )
        out = apply_stage5_to_meta(meta, frame)
        self.assertEqual(int(out["hard_negative"][0]), 1)
        self.assertEqual(int(out["hard_negative"][1]), 0)
        self.assertEqual(int(out["hard_negative"][3]), 1)

    def test_mismatched_row_order_raises(self):
        meta = _base_meta()
        # Shuffle frame by reversing rows
        frame = _stage5_frame(
            ["negative", "ambiguous", "positive", "positive"],
            [1.0, 0.0, 1.0, 1.0],
        )
        frame = frame.iloc[[3, 2, 1, 0]].reset_index(drop=True)
        # Intentionally mis-align genome_id with the meta
        frame["genome_id"] = ["X"] * 4
        with self.assertRaisesRegex(ValueError, "mismatch"):
            apply_stage5_to_meta(meta, frame)

    def test_refuses_if_frame_has_different_row_count(self):
        meta = _base_meta(4)
        frame = _stage5_frame(["positive"] * 3, [1.0] * 3)
        with self.assertRaisesRegex(ValueError, "row count"):
            apply_stage5_to_meta(meta, frame)


class CliSmokeTests(unittest.TestCase):
    def test_cli_writes_output_pkl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta = _base_meta()
            meta_path = tmp_path / "meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)

            frame = _stage5_frame(
                ["positive", "ambiguous", "ambiguous", "negative"],
                [1.0, 0.0, 0.0, 1.0],
            )
            csv_path = tmp_path / "labels.csv"
            frame.to_csv(csv_path, index=False)

            out_path = tmp_path / "out.pkl"
            rc = _applier.main(
                [
                    "--meta",
                    str(meta_path),
                    "--stage5-csv",
                    str(csv_path),
                    "--output",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())

            with open(out_path, "rb") as f:
                loaded = pickle.load(f)
            # Row 1 should be masked now
            np.testing.assert_array_equal(
                loaded["labels"][1], np.zeros(8, dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
