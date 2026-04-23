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


class PrecisionConstrainedDecoderSweepTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module(
            "scripts/run_precision_constrained_decoder_sweep.py",
            "run_precision_constrained_decoder_sweep",
        )

    def test_resolve_params_grid_overrides_fixed_for_max_gap(self):
        decoder_params = {
            "score_family": "fused",
            "start_threshold": 0.99,
            "extend_threshold": 0.95,
            "max_gap": 2,
            "min_support_windows": 6,
            "length_cap": None,
            "valley_threshold": None,
            "valley_min_windows": 1,
        }
        fixed = {"max_gap": 0, "length_cap": 50000}

        resolved = self.runner.resolve_params(decoder_params, fixed)

        self.assertEqual(resolved["max_gap"], 2)
        self.assertIsNone(resolved["length_cap"])
        self.assertEqual(resolved["min_support_windows"], 6)

    def test_resolve_params_falls_back_to_fixed_when_grid_key_missing(self):
        decoder_params = {
            "score_family": "fused",
            "start_threshold": 0.99,
            "extend_threshold": 0.95,
        }
        fixed = {"max_gap": 3, "length_cap": 45000, "min_support_windows": 4}

        resolved = self.runner.resolve_params(decoder_params, fixed)

        self.assertEqual(resolved["max_gap"], 3)
        self.assertEqual(resolved["length_cap"], 45000)
        self.assertEqual(resolved["min_support_windows"], 4)

    def test_generate_decoder_trials_filters_invalid_hysteresis_pairs(self):
        config = {
            "decoder_grid": {
                "score_family": ["fused"],
                "start_threshold": [0.95, 0.99],
                "extend_threshold": [0.97],
                "min_support_windows": [2],
                "length_cap": [50000],
                "valley_threshold": [None],
                "valley_min_windows": [1],
            },
            "decoder_fixed": {"max_gap": 0},
        }

        trials = self.runner.generate_decoder_trials(config)

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["params"]["start_threshold"], 0.99)
        self.assertEqual(trials[0]["params"]["extend_threshold"], 0.97)

    def test_rank_trials_prioritizes_precision_when_recall_constraint_is_met(self):
        results = [
            {
                "trial_id": "high-f1-low-p",
                "summary": {"bench": {"recall": 0.60, "precision": 0.30, "f1": 0.40}},
            },
            {
                "trial_id": "low-f1-high-p",
                "summary": {"bench": {"recall": 0.36, "precision": 0.50, "f1": 0.42}},
            },
            {
                "trial_id": "below-recall",
                "summary": {"bench": {"recall": 0.30, "precision": 0.90, "f1": 0.45}},
            },
        ]

        ranked = self.runner.rank_precision_constrained(results, min_recall=0.3475)

        self.assertEqual([row["trial_id"] for row in ranked], ["low-f1-high-p", "high-f1-low-p"])

    def test_rank_trials_can_filter_pathological_fragmentation_by_n_pred(self):
        results = [
            {
                "trial_id": "fragmented",
                "summary": {"bench": {"recall": 0.36, "precision": 0.55, "f1": 0.43, "n_pred": 1200}},
            },
            {
                "trial_id": "credible",
                "summary": {"bench": {"recall": 0.50, "precision": 0.40, "f1": 0.44, "n_pred": 293}},
            },
        ]

        ranked = self.runner.rank_precision_constrained(
            results,
            min_recall=0.3475,
            max_n_pred=500,
        )

        self.assertEqual([row["trial_id"] for row in ranked], ["credible"])

    def test_write_csv_primary_sort_by_dedup_f1_and_emits_frag_columns(self):
        import csv
        import tempfile

        rows = [
            {
                "trial_id": "frag-heavy",
                "prediction_set_id": "set-a",
                "decoder_id": "decoder-001",
                "results_dir": "tmp/fake",
                "decoder_params": {
                    "score_family": "fused",
                    "start_threshold": 0.99,
                    "extend_threshold": 0.95,
                    "max_gap": 0,
                    "min_support_windows": 2,
                    "length_cap": 30000,
                    "valley_threshold": None,
                    "valley_min_windows": 1,
                },
                "pred_length_stats": {"median": 10240.0, "p75": 14336.0},
                "bench": {
                    "recall": 0.50,
                    "precision": 0.43,
                    "f1": 0.46,
                    "n_pred": 468,
                    "tp_precision": 203,
                    "fragmentation": {
                        "dedup_precision": 0.16,
                        "dedup_f1": 0.25,
                        "unique_gt_precision_tp": 76,
                        "duplicate_precision_tp": 127,
                        "max_preds_per_gt": 10,
                    },
                },
                "summary": {
                    "bench": {
                        "recall": 0.50,
                        "precision": 0.43,
                        "f1": 0.46,
                        "n_pred": 468,
                        "tp_precision": 203,
                        "fragmentation": {
                            "dedup_precision": 0.16,
                            "dedup_f1": 0.25,
                            "unique_gt_precision_tp": 76,
                            "duplicate_precision_tp": 127,
                            "max_preds_per_gt": 10,
                        },
                    }
                },
            },
            {
                "trial_id": "frag-light",
                "prediction_set_id": "set-a",
                "decoder_id": "decoder-002",
                "results_dir": "tmp/fake",
                "decoder_params": {
                    "score_family": "fused",
                    "start_threshold": 0.95,
                    "extend_threshold": 0.80,
                    "max_gap": 0,
                    "min_support_windows": 1,
                    "length_cap": None,
                    "valley_threshold": None,
                    "valley_min_windows": 1,
                },
                "pred_length_stats": {"median": 21000.0, "p75": 30000.0},
                "bench": {
                    "recall": 0.45,
                    "precision": 0.35,
                    "f1": 0.39,
                    "n_pred": 200,
                    "tp_precision": 70,
                    "fragmentation": {
                        "dedup_precision": 0.33,
                        "dedup_f1": 0.38,
                        "unique_gt_precision_tp": 66,
                        "duplicate_precision_tp": 4,
                        "max_preds_per_gt": 2,
                    },
                },
                "summary": {
                    "bench": {
                        "recall": 0.45,
                        "precision": 0.35,
                        "f1": 0.39,
                        "n_pred": 200,
                        "tp_precision": 70,
                        "fragmentation": {
                            "dedup_precision": 0.33,
                            "dedup_f1": 0.38,
                            "unique_gt_precision_tp": 66,
                            "duplicate_precision_tp": 4,
                            "max_preds_per_gt": 2,
                        },
                    }
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.csv"
            self.runner.write_csv(out_path, rows, min_recall=0.30, max_n_pred=500)
            with out_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows_read = list(reader)

        self.assertIn("dedup_f1", fieldnames)
        self.assertIn("dedup_precision", fieldnames)
        self.assertIn("unique_gt_precision_tp", fieldnames)
        self.assertIn("max_preds_per_gt", fieldnames)
        self.assertIn("median_pred_len", fieldnames)
        self.assertIn("max_gap", fieldnames)
        self.assertEqual(rows_read[0]["trial_id"], "frag-light")
        self.assertEqual(rows_read[1]["trial_id"], "frag-heavy")

    def test_write_csv_puts_eligible_rows_before_ineligible(self):
        import csv
        import tempfile

        def _make_row(trial_id: str, recall: float, n_pred: int, dedup_f1: float) -> dict:
            bench = {
                "recall": recall,
                "precision": dedup_f1,
                "f1": dedup_f1,
                "n_pred": n_pred,
                "tp_precision": int(dedup_f1 * n_pred),
                "fragmentation": {
                    "dedup_precision": dedup_f1,
                    "dedup_f1": dedup_f1,
                    "unique_gt_precision_tp": int(dedup_f1 * n_pred),
                    "duplicate_precision_tp": 0,
                    "max_preds_per_gt": 1,
                },
            }
            return {
                "trial_id": trial_id,
                "prediction_set_id": "set-a",
                "decoder_id": trial_id,
                "results_dir": "tmp/fake",
                "decoder_params": {
                    "score_family": "fused",
                    "start_threshold": 0.99,
                    "extend_threshold": 0.95,
                    "max_gap": 0,
                    "min_support_windows": 10,
                    "length_cap": None,
                    "valley_threshold": None,
                    "valley_min_windows": 1,
                },
                "pred_length_stats": {"median": 30000.0, "p75": 40000.0},
                "bench": bench,
                "summary": {"bench": bench},
            }

        rows = [
            _make_row("ineligible-high-dedup", recall=0.20, n_pred=40, dedup_f1=0.55),
            _make_row("eligible-moderate", recall=0.40, n_pred=80, dedup_f1=0.42),
            _make_row("eligible-low", recall=0.35, n_pred=90, dedup_f1=0.38),
            _make_row("ineligible-n-pred", recall=0.50, n_pred=1200, dedup_f1=0.50),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.csv"
            self.runner.write_csv(out_path, rows, min_recall=0.30, max_n_pred=500)
            with out_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows_read = list(reader)

        self.assertEqual(rows_read[0]["trial_id"], "eligible-moderate")
        self.assertEqual(rows_read[1]["trial_id"], "eligible-low")
        # Ineligible rows come after eligible ones, still in dedup_f1 order among themselves.
        self.assertEqual(rows_read[2]["trial_id"], "ineligible-high-dedup")
        self.assertEqual(rows_read[3]["trial_id"], "ineligible-n-pred")
        # Gates are reported so a human reading the CSV can see why a row is demoted.
        self.assertEqual(rows_read[0]["meets_recall_floor"], "True")
        self.assertEqual(rows_read[2]["meets_recall_floor"], "False")
        self.assertEqual(rows_read[3]["meets_n_pred_cap"], "False")

    def test_annotate_gate_flags_fragmentation_hacks(self):
        gated = self.runner._annotate_gate(
            {"summary": {"bench": {"precision": 0.69, "fragmentation": {"dedup_precision": 0.50}}}},
            threshold=0.75,
        )
        self.assertAlmostEqual(gated["fragmentation_gate"]["dedup_over_bench_p"], 0.50 / 0.69, places=4)
        self.assertFalse(gated["fragmentation_gate"]["gate_pass"])

        clean = self.runner._annotate_gate(
            {"summary": {"bench": {"precision": 0.47, "fragmentation": {"dedup_precision": 0.47}}}},
            threshold=0.75,
        )
        self.assertEqual(clean["fragmentation_gate"]["dedup_over_bench_p"], 1.0)
        self.assertTrue(clean["fragmentation_gate"]["gate_pass"])

        self.assertIsNone(self.runner._annotate_gate(None, threshold=0.75))

    def test_rank_trials_uses_dedup_f1_before_raw_precision(self):
        results = [
            {
                "trial_id": "fragmented-high-raw",
                "summary": {
                    "bench": {
                        "recall": 0.50,
                        "precision": 0.45,
                        "f1": 0.47,
                        "n_pred": 300,
                        "fragmentation": {
                            "dedup_f1": 0.25,
                            "dedup_precision": 0.16,
                            "duplicate_precision_tp": 80,
                            "max_preds_per_gt": 10,
                        },
                    }
                },
            },
            {
                "trial_id": "less-fragmented",
                "summary": {
                    "bench": {
                        "recall": 0.45,
                        "precision": 0.32,
                        "f1": 0.37,
                        "n_pred": 220,
                        "fragmentation": {
                            "dedup_f1": 0.30,
                            "dedup_precision": 0.23,
                            "duplicate_precision_tp": 10,
                            "max_preds_per_gt": 2,
                        },
                    }
                },
            },
        ]

        ranked = self.runner.rank_dedup_constrained(results, min_recall=0.3475)

        self.assertEqual([row["trial_id"] for row in ranked], ["less-fragmented", "fragmented-high-raw"])


if __name__ == "__main__":
    unittest.main()
