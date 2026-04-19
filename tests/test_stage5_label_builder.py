"""Tests for the Stage 5 label-variant builder.

The builder reads an existing window-metadata CSV (with a real schema:
``label``, ``max_bgc_overlap_ratio``, ``nearest_bgc_distance``,
``overlaps_truncated_bgc``, ``hard_negative``, ``bgc_length_max``, ...)
and emits the same rows plus three additional columns:

    stage5_label    — {"positive", "negative", "ambiguous"}
    stage5_weight   — float, 0.0 for masked rows
    stage5_variant  — the variant id

Dropped rows are still emitted but with ``stage5_weight = 0.0`` and
``stage5_label = "ambiguous"``, so downstream training can filter by
weight without re-keying the dataset.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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


_builder = _load_module(
    "scripts/build_stage5_labels.py", "build_stage5_labels"
)
apply_variant = _builder.apply_variant
VARIANTS = _builder.VARIANTS


def _sample_frame() -> pd.DataFrame:
    """Minimal fixture covering every code path."""
    rows = [
        # full overlap positive, non-truncated, normal BGC length
        {"window_id": 1, "label": "positive", "max_bgc_overlap_ratio": 1.00,
         "nearest_bgc_distance": 0.0, "overlaps_truncated_bgc": 0,
         "hard_negative": 0, "bgc_length_max": 20000.0},
        # positive, overlap 0.80 (clears 0.75 but not 0.90)
        {"window_id": 2, "label": "positive", "max_bgc_overlap_ratio": 0.80,
         "nearest_bgc_distance": 0.0, "overlaps_truncated_bgc": 0,
         "hard_negative": 0, "bgc_length_max": 20000.0},
        # positive, overlap 0.60 (edge of current rule, below 0.75 threshold)
        {"window_id": 3, "label": "positive", "max_bgc_overlap_ratio": 0.60,
         "nearest_bgc_distance": 0.0, "overlaps_truncated_bgc": 0,
         "hard_negative": 0, "bgc_length_max": 20000.0},
        # positive, truncated BGC
        {"window_id": 4, "label": "positive", "max_bgc_overlap_ratio": 1.00,
         "nearest_bgc_distance": 0.0, "overlaps_truncated_bgc": 1,
         "hard_negative": 0, "bgc_length_max": 20000.0},
        # short-BGC positive (< 10 kb)
        {"window_id": 5, "label": "positive", "max_bgc_overlap_ratio": 1.00,
         "nearest_bgc_distance": 0.0, "overlaps_truncated_bgc": 0,
         "hard_negative": 0, "bgc_length_max": 4000.0},
        # ambiguous (0.3 overlap)
        {"window_id": 6, "label": "ambiguous",
         "max_bgc_overlap_ratio": 0.30, "nearest_bgc_distance": 0.0,
         "overlaps_truncated_bgc": 0, "hard_negative": 0,
         "bgc_length_max": 20000.0},
        # clean negative
        {"window_id": 7, "label": "negative", "max_bgc_overlap_ratio": 0.0,
         "nearest_bgc_distance": 500000.0, "overlaps_truncated_bgc": 0,
         "hard_negative": 0, "bgc_length_max": float("nan")},
        # hard negative
        {"window_id": 8, "label": "negative", "max_bgc_overlap_ratio": 0.0,
         "nearest_bgc_distance": 5_000_000.0, "overlaps_truncated_bgc": 0,
         "hard_negative": 1, "bgc_length_max": float("nan")},
    ]
    return pd.DataFrame(rows)


class VariantCoverageTests(unittest.TestCase):
    def test_variants_registry_exposes_expected_ids(self):
        self.assertIn("baseline_v1", VARIANTS)
        self.assertIn("tight_pos_075_v1", VARIANTS)
        self.assertIn("tight_pos_090_v1", VARIANTS)
        self.assertIn("short_bgc_upweight_v1", VARIANTS)
        self.assertIn("truncated_exclude_v1", VARIANTS)
        self.assertIn("hardpos_hardneg_v2", VARIANTS)

    def test_unknown_variant_raises(self):
        df = _sample_frame()
        with self.assertRaises(KeyError):
            apply_variant(df, "does_not_exist")


class BaselineV1Tests(unittest.TestCase):
    def test_preserves_original_labels(self):
        df = _sample_frame()
        out = apply_variant(df, "baseline_v1")
        self.assertEqual(list(out["stage5_label"]), list(df["label"]))
        # positives + hard-neg all weighted 1.0; ambiguous masked at 0
        self.assertEqual(
            out.loc[out["label"] == "ambiguous", "stage5_weight"].iloc[0],
            0.0,
        )
        self.assertEqual(
            out.loc[out["label"] == "positive", "stage5_weight"].iloc[0],
            1.0,
        )


class TightPos075V1Tests(unittest.TestCase):
    def test_threshold_075_only_high_overlap_remain_positive(self):
        df = _sample_frame()
        out = apply_variant(df, "tight_pos_075_v1")
        by_id = out.set_index("window_id")["stage5_label"]
        self.assertEqual(by_id[1], "positive")  # overlap 1.00
        self.assertEqual(by_id[2], "positive")  # overlap 0.80
        self.assertEqual(by_id[3], "ambiguous")  # overlap 0.60
        self.assertEqual(by_id[6], "ambiguous")  # original ambiguous stays
        self.assertEqual(by_id[7], "negative")

    def test_ambiguous_rows_have_zero_weight(self):
        df = _sample_frame()
        out = apply_variant(df, "tight_pos_075_v1")
        amb = out[out["stage5_label"] == "ambiguous"]
        self.assertTrue((amb["stage5_weight"] == 0.0).all())


class TightPos090V1Tests(unittest.TestCase):
    def test_only_overlap_at_or_above_090_remain_positive(self):
        df = _sample_frame()
        out = apply_variant(df, "tight_pos_090_v1")
        by_id = out.set_index("window_id")["stage5_label"]
        self.assertEqual(by_id[1], "positive")  # 1.00
        self.assertEqual(by_id[2], "ambiguous")  # 0.80 no longer passes
        self.assertEqual(by_id[3], "ambiguous")  # 0.60


class ShortBgcUpweightV1Tests(unittest.TestCase):
    def test_short_source_bgc_gets_upweight(self):
        df = _sample_frame()
        out = apply_variant(df, "short_bgc_upweight_v1")
        by_id = out.set_index("window_id")
        # window 5 has bgc_length_max = 4000, should be upweighted
        self.assertGreater(by_id.loc[5, "stage5_weight"], 1.0)
        # window 1 has 20000, baseline weight 1.0
        self.assertEqual(by_id.loc[1, "stage5_weight"], 1.0)

    def test_upweight_respects_custom_threshold_and_factor(self):
        df = _sample_frame()
        out = apply_variant(
            df,
            "short_bgc_upweight_v1",
            short_max_bp=5000,
            short_positive_weight=8.0,
        )
        by_id = out.set_index("window_id")
        self.assertEqual(by_id.loc[5, "stage5_weight"], 8.0)


class TruncatedExcludeV1Tests(unittest.TestCase):
    def test_truncated_positives_are_masked(self):
        df = _sample_frame()
        out = apply_variant(df, "truncated_exclude_v1")
        by_id = out.set_index("window_id")
        # window 4 is truncated positive → mask
        self.assertEqual(by_id.loc[4, "stage5_weight"], 0.0)
        self.assertEqual(by_id.loc[4, "stage5_label"], "ambiguous")
        # window 1 is clean positive → kept
        self.assertEqual(by_id.loc[1, "stage5_weight"], 1.0)
        self.assertEqual(by_id.loc[1, "stage5_label"], "positive")


class HardposHardnegV2Tests(unittest.TestCase):
    def test_combines_short_upweight_and_truncated_mask_and_hardneg_upweight(self):
        df = _sample_frame()
        out = apply_variant(df, "hardpos_hardneg_v2")
        by_id = out.set_index("window_id")
        # short BGC positive upweighted
        self.assertGreater(by_id.loc[5, "stage5_weight"], 1.0)
        # truncated positive masked
        self.assertEqual(by_id.loc[4, "stage5_weight"], 0.0)
        # hard negative upweighted (>1.0)
        self.assertGreater(by_id.loc[8, "stage5_weight"], 1.0)
        # plain negative stays at 1.0
        self.assertEqual(by_id.loc[7, "stage5_weight"], 1.0)


class EmbargoGuardTests(unittest.TestCase):
    """The builder must refuse to emit labels for embargoed 9-genome rows."""

    def test_refuses_if_embargoed_genome_in_input(self):
        df = _sample_frame()
        df.loc[:, "genome_id"] = "GCA_999999999.1"
        df.loc[0, "genome_id"] = "GCA_000154945.1"  # embargoed

        with self.assertRaisesRegex(ValueError, "embargoed"):
            apply_variant(
                df,
                "baseline_v1",
                embargoed_genome_ids={"GCA_000154945.1"},
            )


class CliSmokeTests(unittest.TestCase):
    def test_cli_writes_output_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "in.csv"
            output_csv = tmp_path / "out.csv"
            df = _sample_frame()
            df["genome_id"] = "GCA_999999999.1"
            df.to_csv(input_csv, index=False)

            rc = _builder.main(
                [
                    "--input",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                    "--variant",
                    "tight_pos_075_v1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(output_csv.exists())
            out = pd.read_csv(output_csv)
            self.assertIn("stage5_label", out.columns)
            self.assertIn("stage5_weight", out.columns)
            self.assertIn("stage5_variant", out.columns)
            self.assertTrue(
                (out["stage5_variant"] == "tight_pos_075_v1").all()
            )


if __name__ == "__main__":
    unittest.main()
