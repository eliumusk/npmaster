import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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


class Stage1DecoderSweepTests(unittest.TestCase):
    def setUp(self):
        self.sweep_mod = load_module(
            "scripts/run_decoder_sweep.py",
            "run_decoder_sweep",
        )
        self.eval_mod = load_module(
            "scripts/evaluate_nine_genome.py",
            "evaluate_nine_genome_for_sweep",
        )

    def test_load_sweep_config_reads_decoder_grid_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "sweep.yaml"
            sweep_path.write_text(
                "\n".join(
                    [
                        "decoder_grid:",
                        "  score_family: [fused, max_type]",
                        "  start_threshold: [0.98, 0.99]",
                        "  extend_threshold: [0.95]",
                        "  max_gap: [0, 128]",
                        "  min_support_windows: [1]",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = self.sweep_mod.load_sweep_config(sweep_path)

            self.assertEqual(cfg["decoder_grid"]["score_family"], ["fused", "max_type"])
            self.assertEqual(cfg["decoder_grid"]["max_gap"], [0, 128])

    def test_generate_decoder_trials_expands_cartesian_product(self):
        sweep_config = {
            "decoder_grid": {
                "score_family": ["fused", "max_type"],
                "start_threshold": [0.98],
                "extend_threshold": [0.95, 0.9],
                "max_gap": [0, 128],
            }
        }

        trials = self.sweep_mod.generate_decoder_trials(sweep_config)

        self.assertEqual(len(trials), 8)
        self.assertEqual(trials[0]["trial_id"], "decoder-001")
        self.assertEqual(
            trials[0]["params"],
            {
                "score_family": "fused",
                "start_threshold": 0.98,
                "extend_threshold": 0.95,
                "max_gap": 0,
            },
        )
        self.assertEqual(trials[-1]["trial_id"], "decoder-008")

    def test_split_file_selects_only_requested_part(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "split_name": "nine-genome-dev-lockbox-v1",
                        "dev": ["GCA_DEV_2", "GCA_DEV_1"],
                        "lockbox": ["GCA_LOCK_1", "GCA_LOCK_2"],
                    }
                ),
                encoding="utf-8",
            )

            dev_subset = self.eval_mod.load_genome_subset(split_path, "dev")
            lockbox_subset = self.eval_mod.load_genome_subset(split_path, "lockbox")

            self.assertEqual(dev_subset, ["GCA_DEV_2", "GCA_DEV_1"])
            self.assertEqual(lockbox_subset, ["GCA_LOCK_1", "GCA_LOCK_2"])

    def test_write_ranked_summary_outputs_sorted_json_and_csv(self):
        ranked_trials = self.sweep_mod.rank_trials(
            [
                {
                    "trial_id": "decoder-003",
                    "params": {"score_family": "fused"},
                    "summary": {"bench": {"recall": 0.35, "precision": 0.43}},
                },
                {
                    "trial_id": "decoder-001",
                    "params": {"score_family": "max_type"},
                    "summary": {"bench": {"recall": 0.41, "precision": 0.39}},
                },
                {
                    "trial_id": "decoder-002",
                    "params": {"score_family": "fused"},
                    "summary": {"bench": {"recall": 0.41, "precision": 0.40}},
                },
            ]
        )

        self.assertEqual(
            [trial["trial_id"] for trial in ranked_trials],
            ["decoder-002", "decoder-001", "decoder-003"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            self.sweep_mod.write_ranked_summary(out_dir, ranked_trials)

            summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            summary_csv = (out_dir / "summary.csv").read_text(encoding="utf-8").splitlines()

            self.assertEqual(summary_json["ranking_metric"], ["recall", "precision"])
            self.assertEqual(summary_json["trials"][0]["trial_id"], "decoder-002")
            self.assertEqual(
                summary_csv[0],
                "rank,trial_id,recall,precision,score_family",
            )
            self.assertEqual(summary_csv[1], "1,decoder-002,0.410000,0.400000,fused")


if __name__ == "__main__":
    unittest.main()
