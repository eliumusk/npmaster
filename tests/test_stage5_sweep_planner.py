"""Tests for the Stage 5 sweep planner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_planner = _load_module(
    "scripts/run_stage5_data_sweep.py", "run_stage5_data_sweep"
)
load_sweep_config = _planner.load_sweep_config
plan_trial = _planner.plan_trial


MINIMAL_YAML = """
sweep_id: stage5_test_v1
defaults:
  input_windows_csv: /srv/windows.csv
  base_meta_arrays: /srv/meta.pkl
  bgc_csv: /srv/bgc.csv
  embargo_config: configs/splits/nine_genome_embargo_v1.json
  base_checkpoint: /srv/warm.pt
  chunk_index: /srv/chunk.pkl
  contig_index: /srv/contig.pkl
  embeddings: /srv/emb.npy
  projections: /srv/proj.npy
  hard_neg_weight: 4.0
  seed: 0
  epochs: 60
  batch_size: 64
  stage5_data_root: /srv/stage5
trials:
  baseline_v1:
    variant: baseline_v1
  tight_pos_075_v1:
    variant: tight_pos_075_v1
  short_bgc_upweight_v1:
    variant: short_bgc_upweight_v1
    short_max_bp: 10000
    short_weight: 4.0
"""


class LoadSweepConfigTests(unittest.TestCase):
    def test_parses_minimal_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.yaml"
            p.write_text(MINIMAL_YAML, encoding="utf-8")
            cfg = load_sweep_config(p)
            self.assertEqual(cfg["sweep_id"], "stage5_test_v1")
            self.assertEqual(len(cfg["trials"]), 3)
            variants = {t["variant"] for t in cfg["trials"]}
            self.assertEqual(
                variants,
                {"baseline_v1", "tight_pos_075_v1", "short_bgc_upweight_v1"},
            )

    def test_unknown_variant_rejected(self):
        yaml_text = MINIMAL_YAML + "  bogus_v1:\n    variant: bogus_v1\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.yaml"
            p.write_text(yaml_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bogus_v1"):
                load_sweep_config(p)

    def test_missing_defaults_rejected(self):
        import yaml as _yaml

        parsed = _yaml.safe_load(MINIMAL_YAML)
        parsed["defaults"].pop("base_checkpoint")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.yaml"
            p.write_text(_yaml.safe_dump(parsed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base_checkpoint"):
                load_sweep_config(p)


class PlanTrialTests(unittest.TestCase):
    def test_plan_emits_five_stage_command_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p = tmp_path / "s.yaml"
            p.write_text(MINIMAL_YAML, encoding="utf-8")
            cfg = load_sweep_config(p)

            experiments_root = tmp_path / "experiments"
            trial = cfg["trials"][0]  # baseline_v1
            result = plan_trial(
                cfg, trial, experiments_root=experiments_root
            )

            command_text = result["command_text"]
            self.assertIn("check_no_nine_genome_leakage.py", command_text)
            self.assertIn("build_stage5_labels.py", command_text)
            self.assertIn("apply_stage5_meta.py", command_text)
            self.assertIn("src/train.py", command_text)
            self.assertIn("baseline_v1", command_text)

            manifest = result["manifest"]
            self.assertEqual(
                manifest["nine_genome_policy"], "final_report_only"
            )
            self.assertFalse(manifest["uses_nine_genome_for_selection"])
            self.assertEqual(manifest["variant"], "baseline_v1")

    def test_per_variant_meta_path_and_csv_path_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p = tmp_path / "s.yaml"
            p.write_text(MINIMAL_YAML, encoding="utf-8")
            cfg = load_sweep_config(p)

            paths = set()
            for trial in cfg["trials"]:
                r = plan_trial(
                    cfg,
                    trial,
                    experiments_root=tmp_path / "experiments",
                )
                command_text = r["command_text"]
                # Find unique per-trial output artefacts
                self.assertIn(trial["variant"], command_text)
                paths.add(r["trial_dir"].as_posix())
            # Every trial has a distinct output directory
            self.assertEqual(len(paths), len(cfg["trials"]))

    def test_embargo_check_is_first_stage(self):
        """Leakage check must run before any other command so a
        tainted config kills the trial before training starts."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p = tmp_path / "s.yaml"
            p.write_text(MINIMAL_YAML, encoding="utf-8")
            cfg = load_sweep_config(p)
            r = plan_trial(
                cfg,
                cfg["trials"][0],
                experiments_root=tmp_path / "experiments",
            )
            lines = r["command_text"].splitlines()
            leakage_idx = next(
                i
                for i, line in enumerate(lines)
                if "check_no_nine_genome_leakage.py" in line
            )
            build_idx = next(
                i
                for i, line in enumerate(lines)
                if "build_stage5_labels.py" in line
            )
            train_idx = next(
                i for i, line in enumerate(lines) if "src/train.py" in line
            )
            self.assertLess(leakage_idx, build_idx)
            self.assertLess(build_idx, train_idx)


class ArtefactWritingTests(unittest.TestCase):
    def test_plan_main_writes_command_and_manifest_per_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p = tmp_path / "s.yaml"
            p.write_text(MINIMAL_YAML, encoding="utf-8")
            experiments_root = tmp_path / "experiments"

            rc = _planner.main(
                [
                    "--config",
                    str(p),
                    "--out-root",
                    str(experiments_root),
                ]
            )
            self.assertEqual(rc, 0)

            # baseline_v1 trial dir should exist with command.txt + manifest.json
            trial_dir = (
                experiments_root
                / "stage5_test_v1-baseline_v1"
            )
            self.assertTrue((trial_dir / "command.txt").exists())
            self.assertTrue((trial_dir / "manifest.json").exists())
            manifest = json.loads(
                (trial_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["variant"], "baseline_v1")


if __name__ == "__main__":
    unittest.main()
