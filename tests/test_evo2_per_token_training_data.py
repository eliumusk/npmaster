import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class PreparedArrayDatasetTests(unittest.TestCase):
    def test_memmap_dataset_returns_selected_rows_with_training_dtypes(self):
        from evo2_per_token.training_data import PreparedArrayDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            feats_path = tmp / "feats.npy"
            labels_path = tmp / "labels.npy"
            feats = np.arange(4 * 3 * 2, dtype=np.float16).reshape(4, 3, 2)
            labels = np.array(
                [
                    [0, 1, -1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [-1, 1, 1],
                ],
                dtype=np.int8,
            )
            np.save(feats_path, feats)
            np.save(labels_path, labels)

            feat_mm = np.load(feats_path, mmap_mode="r")
            label_mm = np.load(labels_path, mmap_mode="r")
            ds = PreparedArrayDataset(feat_mm, label_mm, indices=np.array([2, 0]))

            self.assertEqual(len(ds), 2)
            x, y = ds[0]
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.int64)
            np.testing.assert_array_equal(x.numpy(), feats[2].astype(np.float32))
            np.testing.assert_array_equal(y.numpy(), labels[2].astype(np.int64))

    def test_deterministic_subset_is_stable_and_sorted(self):
        from evo2_per_token.training_data import deterministic_subset

        first = deterministic_subset(total=10, requested=4, seed=7)
        second = deterministic_subset(total=10, requested=4, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.tolist(), sorted(first.tolist()))
        self.assertIsNone(deterministic_subset(total=10, requested=0, seed=7))
        self.assertIsNone(deterministic_subset(total=10, requested=10, seed=7))


if __name__ == "__main__":
    unittest.main()
