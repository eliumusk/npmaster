import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


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


class DataAuditTests(unittest.TestCase):
    def test_audit_reports_split_counts_and_nine_genome_overlap(self):
        from evo2_per_token.audit import audit_training_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            windows_csv = tmp / "windows.csv"
            windows_csv.write_text(
                "window_id,genome_id,contig_id,window_start,window_end,label,split,actual_length\n"
                "w1,GCA_000158915.1,c1,0,100,positive,train,100\n"
                "w2,GENOME_X,c2,0,100,negative,val,100\n"
                "w3,GENOME_X,c2,100,200,ambiguous,val,100\n",
                encoding="utf-8",
            )
            bgc_csv = tmp / "bgc.csv"
            bgc_csv.write_text(
                "genome_id,contig_id,start_bp,end_bp\n"
                "GCA_000158915.1,c1,10,80\n"
                "GENOME_X,c2,120,180\n",
                encoding="utf-8",
            )

            audit = audit_training_metadata(windows_csv, bgc_csv)

        self.assertEqual(audit["windows"]["n_total"], 3)
        self.assertEqual(audit["windows"]["by_split"], {"train": 1, "val": 2})
        self.assertEqual(audit["windows"]["by_label"], {"positive": 1, "negative": 1, "ambiguous": 1})
        self.assertEqual(audit["nine_genome_overlap"]["window_count"], 1)
        self.assertEqual(audit["nine_genome_overlap"]["genome_ids"], ["GCA_000158915.1"])
        self.assertEqual(audit["bgc_annotations"]["n_regions"], 2)


class AutoWeakDatasetTests(unittest.TestCase):
    def test_classify_window_assigns_positive_boundary_and_trusted_negative_buckets(self):
        weak = load_module(
            "scripts/audit/evo2_per_token_auto_weak_dataset.py",
            "auto_weak_dataset_for_test",
        )
        bgc_index = {
            ("g1", "c1"): [
                weak.BgcRegion(start=100, end=180),
                weak.BgcRegion(start=400, end=460),
            ]
        }
        params = weak.BucketParams(
            trusted_negative_margin_bp=100,
            boundary_margin_bp=50,
            min_positive_overlap_bp=20,
            short_bgc_bp=100,
        )

        positive = weak.classify_window(
            weak.WindowRow("w_pos", "g1", "c1", 90, 190, 100, "positive", "train"),
            bgc_index,
            params,
        )
        boundary = weak.classify_window(
            weak.WindowRow("w_boundary", "g1", "c1", 181, 220, 39, "negative", "train"),
            bgc_index,
            params,
        )
        trusted = weak.classify_window(
            weak.WindowRow("w_neg", "g1", "c1", 290, 300, 10, "negative", "train"),
            bgc_index,
            params,
        )

        self.assertEqual(positive["bucket"], "short_positive")
        self.assertEqual(positive["overlap_bp"], 80)
        self.assertEqual(boundary["bucket"], "boundary_uncertain")
        self.assertEqual(boundary["nearest_bgc_distance_bp"], 1)
        self.assertEqual(trusted["bucket"], "trusted_negative")
        self.assertEqual(trusted["nearest_bgc_distance_bp"], 100)


class LabelBucketTests(unittest.TestCase):
    def test_build_pu_targets_treats_background_as_weak_unlabeled_by_default(self):
        from evo2_per_token.label_buckets import build_pu_targets

        labels = np.array([[1, 1, 0, 0, -1]], dtype=np.int8)
        targets, weights = build_pu_targets(labels, weak_negative_weight=0.25)

        np.testing.assert_array_equal(targets, np.array([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(weights, np.array([[1.0, 1.0, 0.25, 0.25, 0.0]], dtype=np.float32))

    def test_trusted_and_hard_negative_masks_override_weak_background(self):
        from evo2_per_token.label_buckets import build_pu_targets

        labels = np.array([[1, 0, 0, 0]], dtype=np.int8)
        trusted = np.array([[False, True, False, False]])
        hard = np.array([[False, False, True, False]])
        _targets, weights = build_pu_targets(
            labels,
            trusted_negative_mask=trusted,
            hard_negative_mask=hard,
            weak_negative_weight=0.2,
            trusted_negative_weight=1.0,
            hard_negative_weight=0.7,
        )

        np.testing.assert_array_equal(weights, np.array([[1.0, 1.0, 0.7, 0.2]], dtype=np.float32))

    def test_boundary_softening_marks_positive_edges(self):
        from evo2_per_token.label_buckets import build_pu_targets

        labels = np.array([[0, 1, 1, 1, 0]], dtype=np.int8)
        targets, _weights = build_pu_targets(labels, boundary_soften_tokens=1, boundary_target=0.6)

        np.testing.assert_array_equal(targets, np.array([[0.0, 0.6, 1.0, 0.6, 0.0]], dtype=np.float32))


class WeightedTrainingTests(unittest.TestCase):
    def test_prepared_array_dataset_can_return_token_weights(self):
        from evo2_per_token.training_data import PreparedArrayDataset

        feats = np.zeros((2, 3, 4), dtype=np.float16)
        labels = np.array([[1, 0, -1], [0, 1, 0]], dtype=np.int8)
        weights = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.2]], dtype=np.float32)
        ds = PreparedArrayDataset(feats, labels, weights=weights)

        x, y, w = ds[0]

        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.int64)
        self.assertEqual(w.dtype, torch.float32)
        np.testing.assert_array_equal(w.numpy(), weights[0])

    def test_weighted_bce_ignores_zero_weight_positions(self):
        from evo2_per_token.model import LOSS_REGISTRY, weighted_bce_loss

        logits = torch.tensor([[0.0, 0.0, 10.0]])
        targets = torch.tensor([[1.0, 0.0, 0.0]])
        weights = torch.tensor([[1.0, 1.0, 0.0]])

        loss = weighted_bce_loss(logits, targets, weights)

        self.assertAlmostEqual(float(loss), 0.693147, places=5)
        self.assertIs(LOSS_REGISTRY["weighted_bce"], weighted_bce_loss)

    def test_ddp_loader_uses_prepared_weights_when_present(self):
        train_ddp = load_module("scripts/train/evo2_per_token_unet/train_ddp.py", "train_ddp_for_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "train_feats.npy", np.zeros((2, 3, 4), dtype=np.float16))
            np.save(tmp / "train_labels.npy", np.zeros((2, 3), dtype=np.int8))
            np.save(tmp / "train_weights.npy", np.ones((2, 3), dtype=np.float32))

            ds = train_ddp.load_prepared_dataset(tmp, "train", subsample=0, seed=0)

        self.assertEqual(len(ds[0]), 3)


class PreparedMetadataTests(unittest.TestCase):
    def test_prepare_data_writes_split_keep_indices_and_coords(self):
        prepare_data = load_module(
            "scripts/train/evo2_per_token_unet/prepare_data.py",
            "prepare_data_for_metadata_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            windows = [
                prepare_data.Window("w_train", "g1", "c1", 0, 16, "positive", "train", 16),
                prepare_data.Window("w_val0", "g2", "c2", 32, 48, "negative", "val", 16),
                prepare_data.Window("w_val1", "g2", "c2", 48, 64, "positive", "val", 12),
            ]
            keep = [2, 1]

            prepare_data.write_split_metadata(tmp, "val", windows, keep)

            np.testing.assert_array_equal(np.load(tmp / "val_keep.npy"), np.array([2, 1], dtype=np.int64))
            self.assertEqual(
                (tmp / "val_window_ids.txt").read_text(encoding="utf-8").splitlines(),
                ["w_val1", "w_val0"],
            )
            self.assertEqual(
                (tmp / "val_coords.csv").read_text(encoding="utf-8").splitlines(),
                [
                    "window_id,genome,contig,start,end,actual_length,label,split",
                    "w_val1,g2,c2,48,64,12,positive,val",
                    "w_val0,g2,c2,32,48,16,negative,val",
                ],
            )


class InternalValSweepTests(unittest.TestCase):
    def test_evaluate_decode_grid_reports_region_metrics_on_internal_val(self):
        val_sweep = load_module(
            "scripts/evaluate/evo2_per_token_val_sweep.py",
            "val_sweep_for_test",
        )

        window_probs = {
            "w1": np.array([0.1, 0.8, 0.9, 0.1], dtype=np.float32),
            "w2": np.array([0.1, 0.2, 0.9, 0.9], dtype=np.float32),
        }
        window_coords = {
            "w1": ("g1|c1", 0, 4),
            "w2": ("g1|c1", 4, 8),
        }
        gt_regions = [("g1|c1", 1, 3), ("g1|c1", 6, 8)]

        rows = val_sweep.evaluate_decode_grid(
            window_probs,
            window_coords,
            gt_regions,
            thresholds=[0.5],
            min_lens=[1, 4],
            max_gaps=[0],
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["threshold"], 0.5)
        self.assertEqual(rows[0]["min_len_bp"], 1)
        self.assertEqual(rows[0]["n_pred"], 2)
        self.assertEqual(rows[0]["bench_recall"], 1.0)
        self.assertEqual(rows[0]["bench_precision"], 1.0)
        self.assertEqual(rows[0]["dedup_f1"], 1.0)
        self.assertEqual(rows[1]["min_len_bp"], 4)
        self.assertEqual(rows[1]["n_pred"], 0)

    def test_load_probs_dir_reads_per_stem_npz_outputs(self):
        val_sweep = load_module(
            "scripts/evaluate/evo2_per_token_val_sweep.py",
            "val_sweep_probs_for_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.savez_compressed(
                tmp / "stem.probs.npz",
                w1=np.array([0.1, 0.2], dtype=np.float16),
                w2=np.array([0.3, 0.4], dtype=np.float16),
            )

            probs = val_sweep.load_probs_dir(tmp)

        self.assertEqual(sorted(probs), ["w1", "w2"])
        np.testing.assert_array_equal(probs["w2"], np.array([0.3, 0.4], dtype=np.float16))


class SelectionProtocolTests(unittest.TestCase):
    def test_selector_refuses_nine_genome_sources(self):
        from evo2_per_token.selection import select_operating_point

        candidates = [
            {
                "name": "leaky_variant",
                "source_dataset": "nine genome diagnostic",
                "metrics": {"as_recall": 0.8, "n_pred": 100, "novel": 10},
            }
        ]

        with self.assertRaises(ValueError):
            select_operating_point(candidates, max_regions=200)

    def test_selector_prefers_recall_under_candidate_budget(self):
        from evo2_per_token.selection import select_operating_point

        candidates = [
            {
                "name": "conservative",
                "source_dataset": "oer_proxy",
                "metrics": {"as_recall": 0.4, "n_pred": 100, "novel": 20},
            },
            {
                "name": "recall_first",
                "source_dataset": "oer_proxy",
                "metrics": {"as_recall": 0.55, "n_pred": 180, "novel": 40},
            },
            {
                "name": "too_many",
                "source_dataset": "oer_proxy",
                "metrics": {"as_recall": 0.8, "n_pred": 500, "novel": 100},
            },
        ]

        chosen = select_operating_point(candidates, max_regions=200)

        self.assertEqual(chosen["name"], "recall_first")

    def test_protocol_writer_records_frozen_recipe(self):
        from evo2_per_token.selection import load_protocol_operating_point, write_frozen_protocol

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "protocol.json"
            write_frozen_protocol(
                out,
                checkpoint="experiments/evo2_per_token_unet/train_runs/pu/best.pt",
                operating_points=[
                    {"name": "oer_discovery", "threshold": 0.5, "min_len_bp": 2000},
                    {"name": "qc_conservative", "threshold": 0.7, "min_len_bp": 8000},
                ],
                selection_sources=["internal_val", "oer_proxy"],
            )
            protocol = json.loads(out.read_text())
            op = load_protocol_operating_point(out, "oer_discovery")

        self.assertEqual(protocol["checkpoint"], "experiments/evo2_per_token_unet/train_runs/pu/best.pt")
        self.assertEqual(protocol["selection_sources"], ["internal_val", "oer_proxy"])
        self.assertTrue(protocol["nine_genome_allowed_final_readout_only"])
        self.assertEqual(op["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
