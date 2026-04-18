import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class NineGenomeCaseStudySelectionTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module(
            "scripts/plot_nine_genome_case_studies.py",
            "plot_nine_genome_case_studies",
        )

    def test_rank_genomes_by_typicality_prefers_middle_profile(self):
        stats_by_genome = {
            "extreme_long": {
                "gt_regions": 2,
                "formal_pred_ratio": 4.0,
                "formal_f1": 0.05,
                "tuned_f1": 0.10,
                "antismash_f1": 0.12,
                "deepbgc_f1": 0.02,
                "bgc_prophet_f1": 0.20,
            },
            "typical": {
                "gt_regions": 3,
                "formal_pred_ratio": 1.5,
                "formal_f1": 0.28,
                "tuned_f1": 0.34,
                "antismash_f1": 0.25,
                "deepbgc_f1": 0.11,
                "bgc_prophet_f1": 0.39,
            },
            "extreme_short": {
                "gt_regions": 6,
                "formal_pred_ratio": 0.3,
                "formal_f1": 0.75,
                "tuned_f1": 0.80,
                "antismash_f1": 0.62,
                "deepbgc_f1": 0.41,
                "bgc_prophet_f1": 0.70,
            },
        }

        ranked = self.mod.rank_genomes_by_typicality(stats_by_genome)

        self.assertEqual(ranked[0], "typical")

    def test_choose_representative_locus_prefers_supported_median_length_gt(self):
        gt_regions = [
            ("contig1", 0, 10_000),
            ("contig1", 40_000, 62_000),
            ("contig1", 90_000, 160_000),
        ]
        predictions = {
            "antiSMASH": [("contig1", 42_000, 63_000)],
            "DeepBGC": [("contig1", 39_000, 60_000)],
            "BGC-Prophet": [("contig1", 45_000, 61_000)],
            "NP-Master formal": [("contig1", 38_000, 70_000)],
            "NP-Master tuned": [("contig1", 44_000, 60_000)],
        }

        locus = self.mod.choose_representative_locus(
            gt_regions=gt_regions,
            predictions_by_tool=predictions,
        )

        self.assertEqual(locus, ("contig1", 40_000, 62_000))

    def test_build_plot_window_expands_to_include_nearby_predictions(self):
        locus = ("contigA", 50_000, 70_000)
        predictions = {
            "antiSMASH": [("contigA", 48_000, 72_000)],
            "NP-Master formal": [("contigA", 30_000, 90_000)],
            "NP-Master tuned": [("contigA", 51_000, 69_000)],
        }

        start, end = self.mod.build_plot_window(
            locus=locus,
            predictions_by_tool=predictions,
            flank_bp=20_000,
            min_window_bp=80_000,
            neighbor_slack_bp=5_000,
        )

        self.assertEqual((start, end), (10_000, 110_000))


if __name__ == "__main__":
    unittest.main()
