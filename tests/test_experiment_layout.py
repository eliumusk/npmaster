import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from experiments.layout import (  # noqa: E402
    create_experiment_layout,
    write_command,
    write_resolved_config,
)
from experiments.runbook import build_runbook_entry  # noqa: E402


class ExperimentLayoutTests(unittest.TestCase):
    def test_create_experiment_layout_builds_canonical_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            layout = create_experiment_layout(root, "exp-001")

            expected_root = root / "experiments" / "exp-001"
            self.assertEqual(layout["root"], expected_root)

            expected_paths = [
                expected_root / "manifest.json",
                expected_root / "command.txt",
                expected_root / "config.resolved.yaml",
                expected_root / "logs",
                expected_root / "metrics",
                expected_root / "artifacts" / "checkpoints",
                expected_root / "artifacts" / "predictions",
                expected_root / "artifacts" / "plots",
                expected_root / "artifacts" / "tables",
                expected_root / "inputs",
            ]
            for path in expected_paths:
                self.assertTrue(path.exists(), path)

    def test_create_experiment_layout_resolves_path_deterministically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            first = create_experiment_layout(root, "exp-001")
            second = create_experiment_layout(root, "exp-001")

            self.assertEqual(first["root"], second["root"])

    def test_create_experiment_layout_accepts_experiments_dir_directly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = Path(tmpdir) / "experiments"

            layout = create_experiment_layout(experiments_root, "exp-001")

            self.assertEqual(layout["root"], experiments_root / "exp-001")

    def test_writers_create_manifest_command_and_resolved_config_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            layout = create_experiment_layout(root, "exp-001")

            manifest_path = layout["root"] / "manifest.json"
            manifest_path.write_text(
                json.dumps({"experiment_id": "exp-001"}),
                encoding="utf-8",
            )
            write_command(layout["command"], "python scripts/run.py --dry-run")
            write_resolved_config(
                layout["config"],
                {"decoder": {"start_threshold": 0.95}},
            )

            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                {"experiment_id": "exp-001"},
            )
            self.assertEqual(
                layout["command"].read_text(encoding="utf-8"),
                "python scripts/run.py --dry-run\n",
            )
            self.assertEqual(
                yaml.safe_load(layout["config"].read_text(encoding="utf-8")),
                {"decoder": {"start_threshold": 0.95}},
            )


class ExperimentRunbookTests(unittest.TestCase):
    def test_build_runbook_entry_returns_minimal_metadata(self):
        entry = build_runbook_entry(
            experiment_id="exp-001",
            server="server-a",
            command="python scripts/run.py --dry-run",
            manifest_path=Path("experiments/exp-001/manifest.json"),
        )

        self.assertEqual(
            entry,
            {
                "experiment_id": "exp-001",
                "server": "server-a",
                "command": "python scripts/run.py --dry-run",
                "manifest_path": "experiments/exp-001/manifest.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
