import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bgc-gcd") not in sys.path:
    sys.path.insert(0, str(ROOT / "bgc-gcd"))


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
            "scripts/phase4/eval_nine_genome_operating_point.py",
            "eval_nine_genome_operating_point",
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
            "val_frozen_fused", self.eval_mod.DEFAULT_OPERATING_POINTS
        )
        self.assertEqual(op["start_threshold"], 0.98)
        self.assertEqual(op["extend_threshold"], 0.95)
        self.assertEqual(op["max_gap"], 0)
        self.assertEqual(op["length_cap"], 50000)
        self.assertEqual(op["min_support_windows"], 1)
        self.assertEqual(op["score"], "fused")

    def test_select_genome_ids_preserves_benchmark_order(self):
        selected = self.eval_mod.select_genome_ids("GCA_000154945.1,GCA_000158915.1")
        self.assertEqual(selected, ["GCA_000158915.1", "GCA_000154945.1"])


class NineGenomeStrideInferenceTests(unittest.TestCase):
    def setUp(self):
        self.infer_mod = load_module(
            "scripts/phase4/run_nine_genome_stride_inference.py",
            "run_nine_genome_stride_inference",
        )

    def test_make_windows_supports_overlap_stride(self):
        windows = self.infer_mod.make_windows("A" * 20000, window=16384, stride=8192)
        starts = [w[0] for w in windows]
        ends = [w[1] for w in windows]
        seq_lens = [len(w[2]) for w in windows]

        self.assertEqual(starts, [0, 8192, 16384])
        self.assertEqual(ends, [16384, 20000, 20000])
        self.assertEqual(seq_lens, [16384, 16384, 16384])

    def test_resolve_bgc_gcd_dir_accepts_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bgc_gcd = Path(tmpdir) / "bgc-gcd"
            model_dir = bgc_gcd / "model"
            model_dir.mkdir(parents=True)
            (model_dir / "context_full_model.py").write_text("# stub\n", encoding="utf-8")

            resolved = self.infer_mod.resolve_bgc_gcd_dir(str(bgc_gcd))
            self.assertEqual(resolved, bgc_gcd.resolve())

    def test_parse_genome_subset_argument(self):
        selected = {
            part.strip()
            for part in "GCA_000154945.1,GCA_000158915.1".split(",")
            if part.strip()
        }
        self.assertEqual(selected, {"GCA_000154945.1", "GCA_000158915.1"})


class PostprocessV2Tests(unittest.TestCase):
    def setUp(self):
        self.pp_mod = load_module(
            "scripts/phase4/postprocess_v2.py",
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


class ScoreFamilyDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.diag_mod = load_module(
            "scripts/phase4/diagnose_score_families.py",
            "diagnose_score_families",
        )

    def test_auc_and_ap_perfect_ranking(self):
        labels = self.diag_mod.np.asarray([0, 0, 1, 1], dtype=self.diag_mod.np.int64)
        scores = self.diag_mod.np.asarray([0.1, 0.2, 0.8, 0.9], dtype=self.diag_mod.np.float64)
        self.assertAlmostEqual(self.diag_mod.compute_auroc(labels, scores), 1.0)
        self.assertAlmostEqual(self.diag_mod.compute_average_precision(labels, scores), 1.0)

    def test_best_gt_overlap_uses_matching_contig_only(self):
        ov_bp, ov_strict = self.diag_mod.best_gt_overlap(
            "c1",
            100,
            200,
            [
                ("c2", 100, 300),
                ("c1", 50, 180),
                ("c1", 120, 240),
            ],
        )
        self.assertEqual(ov_bp, 80)
        self.assertAlmostEqual(ov_strict, 80 / 120)

    def test_nine_genome_loader_builds_fused_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "benchmark"
            results = Path(tmpdir) / "results"
            base.mkdir()
            results.mkdir()
            (base / "ground_truth.json").write_text(
                '{"ground_truth":{"GCA_000158915.1":[["c1",0,100]],'
                '"GCA_000158875.1":[],"GCA_000154945.1":[],"GCA_000158895.1":[],'
                '"GCA_000156435.1":[],"GCA_000568915.1":[],"GCA_000158815.1":[],'
                '"GCA_000568255.1":[],"GCA_000156475.1":[]}}',
                encoding="utf-8",
            )
            for genome_id in self.diag_mod.GENOME_IDS:
                rows = "contig,start,end,bgc_score,non_bgc_score\n"
                if genome_id == "GCA_000158915.1":
                    rows += "c1,0,100,0.8,0.1\n"
                else:
                    rows += "c1,0,100,0.2,0.9\n"
                (results / f"{genome_id}_predictions.csv").write_text(rows, encoding="utf-8")

            labels, score_map, records = self.diag_mod.load_nine_genome_dataset(
                base=base,
                results_dir=results,
                positive_overlap_bp=1,
            )

            self.assertEqual(int(labels.sum()), 1)
            self.assertAlmostEqual(score_map["max_type"][0], 0.8)
            self.assertAlmostEqual(score_map["1_minus_isneg"][0], 0.9)
            self.assertAlmostEqual(score_map["fused"][0], 0.85)
            self.assertEqual(records[0]["best_overlap_bp"], 100)


class RouteBHardNegativeTests(unittest.TestCase):
    def setUp(self):
        self.train_mod = load_module(
            "bgc-gcd/train_context_e2e.py",
            "train_context_e2e_for_tests",
        )
        self.mine_mod = load_module(
            "scripts/phase4/mine_route_b_hardneg.py",
            "mine_route_b_hardneg",
        )

    def test_apply_hard_negative_weights_upweights_only_flagged_positions(self):
        per_pos_loss = self.train_mod.torch.tensor([[1.0, 2.0, 3.0]])
        loss_mask = self.train_mod.torch.tensor([[True, True, False]])
        hard_neg_mask = self.train_mod.torch.tensor([[False, True, True]])

        loss, n_hard = self.train_mod.apply_hard_negative_weights(
            per_pos_loss,
            loss_mask,
            hard_neg_mask,
            hard_neg_weight=4.0,
        )

        self.assertEqual(n_hard, 1)
        self.assertAlmostEqual(loss.item(), (1.0 + 2.0 * 4.0) / 5.0)

    def test_select_hard_negative_indices_filters_to_train_negatives(self):
        split = self.mine_mod.np.asarray(["train", "train", "val", "train"])
        label = self.mine_mod.np.asarray(["negative", "positive", "negative", "negative"])
        scores = self.mine_mod.np.asarray([0.97, 0.99, 0.98, 0.96])

        picked = self.mine_mod.select_hard_negative_indices(
            split=split,
            label=label,
            scores=scores,
            top_k=2,
            min_score=0.95,
        )

        self.assertEqual(picked.tolist(), [0, 3])

    def test_patch_meta_arrays_writes_binary_flag(self):
        meta = {
            "label": self.mine_mod.np.asarray(["negative", "negative", "positive"]),
            "split": self.mine_mod.np.asarray(["train", "train", "train"]),
        }
        patched = self.mine_mod.patch_meta_arrays(meta, self.mine_mod.np.asarray([1]))
        self.assertEqual(patched["hard_negative"].tolist(), [0, 1, 0])

    def test_score_fn_from_probs_supports_fused(self):
        probs = self.mine_mod.np.asarray([[0.2, 0.4, 0.1, 0.3, 0, 0, 0, 0.8]])
        fused = self.mine_mod.score_fn_from_probs(probs, "fused")
        self.assertAlmostEqual(float(fused[0]), 0.5 * 0.4 + 0.5 * 0.2)


if __name__ == "__main__":
    unittest.main()
