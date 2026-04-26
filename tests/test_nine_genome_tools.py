import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class EvalNineGenomeOperatingPointTests(unittest.TestCase):
    def setUp(self):
        self.eval_mod = load_module(
            "src/benchmark/nine_genome.py",
            "evaluate_nine_genome",
        )

    def test_strict_metric_penalizes_oversized_region(self):
        gt = [("contig1", 100, 200)]
        pred = [("contig1", 0, 500)]

        loose = self.eval_mod.eval_regions(gt, pred, "loose")
        strict = self.eval_mod.eval_regions(gt, pred, "strict")
        iou = self.eval_mod.eval_regions(gt, pred, "iou")

        self.assertEqual(loose["recall"], 1.0)
        self.assertEqual(loose["precision"], 1.0)
        self.assertEqual(strict["recall"], 0.0)
        self.assertEqual(strict["precision"], 0.0)
        self.assertEqual(iou["recall"], 0.0)
        self.assertEqual(iou["precision"], 0.0)

    def test_window_loading_then_decode_merges_by_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "preds.csv"
            csv_path.write_text(
                "contig,start,end,bgc_score\n"
                "c1,0,100,0.9\n"
                "c1,120,220,0.95\n"
                "c1,500,600,0.91\n",
                encoding="utf-8",
            )
            windows = self.eval_mod.load_windows_from_csv(csv_path)
            merged = self.eval_mod.decode_regions_hysteresis(
                windows,
                start_threshold=0.8,
                extend_threshold=0.8,
                max_gap=32,
            )
            self.assertEqual(merged, [("c1", 0, 220), ("c1", 500, 600)])

    def test_bench_mode_uses_asymmetric_50pct_rule(self):
        gt = [("c1", 100, 200)]          # gt_len = 100
        pred = [("c1", 150, 450)]        # pred_len = 300, overlap = 50

        bench = self.eval_mod.eval_regions(gt, pred, "bench")
        # recall: overlap/gt_len = 50/100 = 0.5 -> TP
        # precision: overlap/pred_len = 50/300 ~ 0.17 -> not TP
        self.assertEqual(bench["recall"], 1.0)
        self.assertEqual(bench["precision"], 0.0)

    def test_bench_mode_recall_requires_50pct_gt_coverage(self):
        gt = [("c1", 0, 1000)]           # gt_len = 1000
        pred = [("c1", 0, 400)]          # pred_len = 400, overlap = 400
        # recall: 400/1000 = 0.4 -> not TP
        # precision: 400/400 = 1.0 -> TP
        bench = self.eval_mod.eval_regions(gt, pred, "bench")
        self.assertEqual(bench["recall"], 0.0)
        self.assertEqual(bench["precision"], 1.0)

    def test_fragmentation_metrics_penalize_duplicate_precision_hits(self):
        gt = [("c1", 100, 400)]
        pred = [
            ("c1", 100, 260),
            ("c1", 160, 320),
            ("c1", 240, 400),
            ("c1", 700, 800),
        ]

        bench = self.eval_mod.eval_regions(gt, pred, "bench")
        frag = bench["fragmentation"]

        self.assertEqual(bench["tp_precision"], 3)
        self.assertEqual(frag["unique_gt_precision_tp"], 1)
        self.assertEqual(frag["duplicate_precision_tp"], 2)
        self.assertEqual(frag["max_preds_per_gt"], 3)
        self.assertAlmostEqual(frag["dedup_precision"], 0.25)
        self.assertAlmostEqual(frag["dedup_f1"], 0.4)

    def test_fused_score_computed_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "preds.csv"
            csv_path.write_text(
                "contig,start,end,bgc_score,non_bgc_score\n"
                "c1,0,100,0.80,0.10\n",
                encoding="utf-8",
            )
            maxtype = self.eval_mod.load_windows_from_csv(csv_path, score_family="max_type")
            fused = self.eval_mod.load_windows_from_csv(csv_path, score_family="fused")
            self.assertAlmostEqual(maxtype[0].score, 0.80)
            # fused = 0.5*0.80 + 0.5*(1-0.10) = 0.40 + 0.45 = 0.85
            self.assertAlmostEqual(fused[0].score, 0.85)

    def test_operating_point_loader_reads_yaml(self):
        op = self.eval_mod.load_operating_point(
            "evo2_lr_multiscale", self.eval_mod.DEFAULT_OPERATING_POINTS
        )
        self.assertEqual(op["start_threshold"], 0.95)
        self.assertEqual(op["extend_threshold"], 0.80)
        self.assertEqual(op["max_gap"], 0)
        self.assertIsNone(op.get("length_cap"))
        self.assertEqual(op["min_support_windows"], 3)
        self.assertEqual(op["score"], "bgc_score")

    def test_resolve_params_accepts_score_override_without_operating_point(self):
        args = SimpleNamespace(
            operating_point=None,
            operating_points_file=self.eval_mod.DEFAULT_OPERATING_POINTS,
            start_threshold=0.99,
            extend_threshold=0.95,
            max_gap=0,
            length_cap=None,
            min_support_windows=10,
            valley_threshold=None,
            valley_min_windows=None,
            threshold=None,
            score="fused",
        )

        params = self.eval_mod.resolve_params(args)

        self.assertEqual(params["score"], "fused")
        self.assertEqual(params["min_support_windows"], 10)

    def test_select_genome_ids_preserves_benchmark_order(self):
        selected = self.eval_mod.select_genome_ids("GCA_000154945.1,GCA_000158915.1")
        self.assertEqual(selected, ["GCA_000158915.1", "GCA_000154945.1"])



class PostprocessV2Tests(unittest.TestCase):
    def setUp(self):
        self.pp_mod = load_module(
            "src/decoding/postprocess.py",
            "postprocess_v2",
        )

    def make_window(self, start: int, end: int, score: float):
        return self.pp_mod.WindowScore("c1", start, end, score)

    def test_hysteresis_requires_high_confidence_seed(self):
        windows = [
            self.make_window(0, 100, 0.85),
            self.make_window(100, 200, 0.96),
            self.make_window(200, 300, 0.87),
        ]
        regions = self.pp_mod.decode_regions_hysteresis(
            windows,
            start_threshold=0.95,
            extend_threshold=0.80,
            max_gap=0,
        )
        self.assertEqual(regions, [("c1", 100, 300)])

    def test_min_support_drops_single_window_seed(self):
        windows = [
            self.make_window(0, 100, 0.97),
            self.make_window(200, 300, 0.99),
            self.make_window(300, 400, 0.90),
        ]
        regions = self.pp_mod.decode_regions_hysteresis(
            windows,
            start_threshold=0.95,
            extend_threshold=0.85,
            max_gap=0,
            min_support_windows=2,
        )
        self.assertEqual(regions, [("c1", 200, 400)])

    def test_valley_split_breaks_sustained_low_bridge(self):
        windows = [
            self.make_window(0, 100, 0.98),
            self.make_window(100, 200, 0.91),
            self.make_window(200, 300, 0.82),
            self.make_window(300, 400, 0.81),
            self.make_window(400, 500, 0.92),
            self.make_window(500, 600, 0.97),
        ]
        regions = self.pp_mod.decode_regions_hysteresis(
            windows,
            start_threshold=0.95,
            extend_threshold=0.80,
            max_gap=0,
            valley_threshold=0.85,
            valley_min_windows=2,
        )
        self.assertEqual(regions, [("c1", 0, 200), ("c1", 400, 600)])

    def test_length_cap_still_applies_after_hysteresis(self):
        windows = [
            self.make_window(0, 100, 0.99),
            self.make_window(100, 200, 0.93),
            self.make_window(200, 300, 0.92),
            self.make_window(300, 400, 0.91),
        ]
        regions = self.pp_mod.decode_regions_hysteresis(
            windows,
            start_threshold=0.95,
            extend_threshold=0.90,
            max_gap=0,
            length_cap=250,
        )
        self.assertEqual(regions, [("c1", 0, 200), ("c1", 200, 400)])



if __name__ == "__main__":
    unittest.main()
