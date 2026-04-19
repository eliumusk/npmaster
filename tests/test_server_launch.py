import importlib.util
import json
import sys
import tempfile
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


class ServerLaunchTests(unittest.TestCase):
    def setUp(self):
        self.sync_mod = load_module(
            "scripts/sync_snapshot_to_server.py",
            "sync_snapshot_to_server",
        )
        self.launch_mod = load_module(
            "scripts/launch_experiment_batch.py",
            "launch_experiment_batch",
        )

    def test_build_run_bundle_spec_requires_commit_config_and_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            experiment_dir = repo_root / "experiments" / "trial-001"
            inputs_dir = experiment_dir / "inputs"
            inputs_dir.mkdir(parents=True)
            (experiment_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiment_id": "trial-001",
                        "git_commit": "abc1234",
                        "command": "python train.py --config configs/sweeps/demo.yaml",
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "config.resolved.yaml").write_text(
                "decoder:\n  start_threshold: 0.95\n",
                encoding="utf-8",
            )
            (experiment_dir / "command.txt").write_text(
                "python train.py --config configs/sweeps/demo.yaml\n",
                encoding="utf-8",
            )
            (inputs_dir / "manifest.json").write_text(
                json.dumps({"split_version": "nine_genome_dev_lockbox_v1"}),
                encoding="utf-8",
            )

            bundle = self.sync_mod.build_run_bundle_spec(
                experiment_dir=experiment_dir,
                repo_root=repo_root,
            )

            self.assertEqual(bundle["experiment_id"], "trial-001")
            self.assertEqual(bundle["git_commit"], "abc1234")
            self.assertEqual(bundle["snapshot_id"], "abc1234")
            self.assertEqual(
                bundle["relative_experiment_path"],
                "experiments/trial-001",
            )
            self.assertEqual(
                bundle["required_files"],
                [
                    "manifest.json",
                    "config.resolved.yaml",
                    "command.txt",
                    "inputs/manifest.json",
                ],
            )

            (experiment_dir / "command.txt").unlink()
            with self.assertRaisesRegex(ValueError, "command.txt"):
                self.sync_mod.build_run_bundle_spec(
                    experiment_dir=experiment_dir,
                    repo_root=repo_root,
                )

    def test_assign_trials_spreads_jobs_across_servers_and_gpus_before_stacking(self):
        hosts = {
            "hosts": [
                {"name": "alpha", "ssh_target": "alpha", "gpus": [0, 1]},
                {"name": "beta", "ssh_target": "beta", "gpus": [0]},
            ]
        }
        bundles = [
            {"experiment_id": "trial-001"},
            {"experiment_id": "trial-002"},
            {"experiment_id": "trial-003"},
            {"experiment_id": "trial-004"},
        ]

        assignments = self.launch_mod.assign_trials_to_devices(bundles, hosts)

        self.assertEqual(
            [(item["experiment_id"], item["server"], item["gpu"]) for item in assignments],
            [
                ("trial-001", "alpha", 0),
                ("trial-002", "alpha", 1),
                ("trial-003", "beta", 0),
                ("trial-004", "alpha", 0),
            ],
        )
        self.assertEqual(assignments[3]["slot_index"], 1)

    def test_build_remote_launch_plan_uses_host_paths_from_config(self):
        bundle = {
            "experiment_id": "trial-009",
            "git_commit": "deadbeef",
            "relative_experiment_path": "experiments/trial-009",
            "command": "python scripts/run_tiling_sweep.py --trial trial-009",
        }
        host = {
            "name": "alpha",
            "ssh_target": "runner@alpha",
            "workspace_root": "/srv/npmaster",
            "snapshot_dirname": "snapshots",
            "gpus": [0, 1],
        }
        assignment = {
            "experiment_id": "trial-009",
            "server": "alpha",
            "gpu": 1,
            "slot_index": 0,
        }

        plan = self.launch_mod.build_remote_launch_plan(bundle, host, assignment)

        self.assertEqual(plan["server"], "alpha")
        self.assertEqual(plan["ssh_target"], "runner@alpha")
        self.assertEqual(
            plan["remote_experiment_dir"],
            "/srv/npmaster/snapshots/deadbeef/repo/experiments/trial-009",
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=1", plan["remote_command"])
        self.assertIn("/srv/npmaster/snapshots/deadbeef/repo/experiments/trial-009/command.txt", plan["remote_command"])
        self.assertNotIn(str(ROOT), plan["remote_command"])

    def test_build_sync_plan_keeps_repo_snapshot_commit_scoped(self):
        bundle = {
            "experiment_id": "trial-011",
            "git_commit": "cafebabe",
            "relative_experiment_path": "experiments/trial-011",
        }
        host = {
            "name": "beta",
            "ssh_target": "runner@beta",
            "workspace_root": "/srv/npmaster",
            "snapshot_dirname": "snapshots",
        }

        plan = self.sync_mod.build_sync_plan(bundle, host, ROOT)

        self.assertEqual(plan["server"], "beta")
        self.assertEqual(
            plan["remote_snapshot_root"],
            "/srv/npmaster/snapshots/cafebabe",
        )
        self.assertIn("runner@beta:/srv/npmaster/snapshots/cafebabe/repo/", plan["commands"][0])


if __name__ == "__main__":
    unittest.main()
