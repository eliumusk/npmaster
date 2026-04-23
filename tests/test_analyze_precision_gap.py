import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzePrecisionGapTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = load_module(
            "scripts/diagnose/precision_gap.py",
            "analyze_precision_gap",
        )

    def test_classifies_prediction_failure_modes(self):
        gt_regions = [
            ("ctg1", 100, 200),
            ("ctg1", 400, 500),
        ]
        cases = [
            (("ctg1", 110, 190), "bench_tp"),
            (("ctg1", 0, 300), "too_long_overlap_gt"),
            (("ctg1", 50, 550), "merged_multiple_gt"),
            (("ctg1", 900, 1000), "no_overlap"),
            (("ctg1", 195, 260), "short_partial_hit"),
            (("ctg1", 180, 260), "flank_only_or_boundary_shift"),
        ]

        for pred_region, expected in cases:
            with self.subTest(pred_region=pred_region):
                row = self.analyzer.classify_prediction(
                    genome_id="GCA_TEST",
                    pred_region=pred_region,
                    gt_regions=gt_regions,
                    min_overlap=0.5,
                )
                self.assertEqual(row["failure_type"], expected)

    def test_summary_counts_failure_modes_and_length_stats(self):
        rows = [
            {"failure_type": "bench_tp", "pred_len": 80, "overlap_over_pred": 0.75},
            {"failure_type": "too_long_overlap_gt", "pred_len": 200, "overlap_over_pred": 0.45},
            {"failure_type": "too_long_overlap_gt", "pred_len": 300, "overlap_over_pred": 0.30},
            {"failure_type": "no_overlap", "pred_len": 100, "overlap_over_pred": 0.0},
        ]

        summary = self.analyzer.summarize_rows(rows)

        self.assertEqual(summary["n_pred"], 4)
        self.assertEqual(summary["failure_counts"]["bench_tp"], 1)
        self.assertEqual(summary["failure_counts"]["too_long_overlap_gt"], 2)
        self.assertEqual(summary["failure_counts"]["no_overlap"], 1)
        self.assertEqual(summary["bench_tp"], 1)
        self.assertEqual(summary["bench_fp"], 3)
        self.assertEqual(summary["failure_stats"]["too_long_overlap_gt"]["median_pred_len"], 250)


if __name__ == "__main__":
    unittest.main()
