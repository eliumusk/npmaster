import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
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


class Stage2TilingSweepTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module(
            "scripts/run_stage2_tiling_sweep.py",
            "run_stage2_tiling_sweep",
        )

    def test_repo_sweep_config_defines_multiple_trials(self):
        config = self.runner.load_sweep_config(ROOT / "configs" / "sweeps" / "stage2_tiling_v1.yaml")

        self.assertEqual(config["sweep_id"], "stage2_tiling_v1")
        self.assertGreaterEqual(len(config["trials"]), 2)
        self.assertTrue(all(isinstance(trial["window"], int) for trial in config["trials"]))
        self.assertTrue(all(isinstance(trial["stride"], int) for trial in config["trials"]))

    def test_build_trial_id_is_stable_and_unique(self):
        trial_a = self.runner.build_trial_id(window=16384, stride=8192)
        trial_b = self.runner.build_trial_id(window=16384, stride=4096)

        self.assertEqual(trial_a, "w16384-s8192")
        self.assertNotEqual(trial_a, trial_b)

    def test_validate_trial_rejects_stride_larger_than_window(self):
        with self.assertRaisesRegex(ValueError, "stride"):
            self.runner.validate_trial({"window": 4096, "stride": 8192})

    def test_plan_trials_writes_remote_safe_command_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage2.yaml"
            experiments_root = tmp / "experiments"

            split_path.write_text(
                json.dumps(
                    {
                        "version": "nine_genome_dev_lockbox_v1",
                        "dev": [
                            "GCA_000158915.1",
                            "GCA_000154945.1",
                            "GCA_000158875.1",
                            "GCA_000156475.1",
                            "GCA_000568915.1",
                        ],
                        "lockbox": [
                            "GCA_000156435.1",
                            "GCA_000158815.1",
                            "GCA_000158895.1",
                            "GCA_000568255.1",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sweep_path.write_text(
                "\n".join(
                    [
                        "sweep_id: stage2_tiling_tmp",
                        "defaults:",
                        "  checkpoint: ckpts/best.pt",
                        "  base_results_dir: /srv/nine/results",
                        "  benchmark_base: /srv/nine",
                        "  src_dir: /srv/npmaster/src",
                        "  fna_dir: /srv/nine/fna",
                        "  operating_point: val_frozen_fused",
                        "  server: gpu-a",
                        "trials:",
                        "  - window: 16384",
                        "    stride: 8192",
                        "  - window: 12288",
                        "    stride: 4096",
                    ]
                ),
                encoding="utf-8",
            )

            plans = self.runner.plan_tiling_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )

            self.assertEqual([plan["trial_id"] for plan in plans], ["w16384-s8192", "w12288-s4096"])
            first = plans[0]
            self.assertEqual(first["manifest"]["server"], "gpu-a")
            self.assertEqual(first["manifest"]["split_part"], "dev")
            self.assertEqual(
                first["manifest"]["input_genome_ids"],
                [
                    "GCA_000158915.1",
                    "GCA_000154945.1",
                    "GCA_000158875.1",
                    "GCA_000156475.1",
                    "GCA_000568915.1",
                ],
            )
            self.assertEqual(first["command_specs"]["inference"]["argv"][0], "python")
            self.assertIn("--window", first["command_specs"]["inference"]["argv"])
            self.assertIn("--stride", first["command_specs"]["inference"]["argv"])
            self.assertIn("--genome-ids", first["command_specs"]["inference"]["argv"])
            self.assertIn("--operating-point", first["command_specs"]["evaluate"]["argv"])
            self.assertEqual(
                first["command_specs"]["evaluate"]["outputs"]["summary_json"],
                str(Path("experiments") / first["experiment_id"] / "metrics" / "summary.json"),
            )
            self.assertEqual(
                first["command_specs"]["inference"]["outputs"]["predictions_dir"],
                str(Path("experiments") / first["experiment_id"] / "artifacts" / "predictions"),
            )

            manifest_path = experiments_root / first["experiment_id"] / "manifest.json"
            command_path = experiments_root / first["experiment_id"] / "command.txt"
            resolved_config_path = experiments_root / first["experiment_id"] / "config.resolved.yaml"
            inputs_manifest_path = experiments_root / first["experiment_id"] / "inputs" / "manifest.json"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(command_path.exists())
            self.assertTrue(resolved_config_path.exists())
            self.assertTrue(inputs_manifest_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["command_specs"]["inference"]["argv"], first["command_specs"]["inference"]["argv"])
            self.assertEqual(
                manifest["command"],
                f"bash experiments/{first['experiment_id']}/command.txt",
            )
            command_txt = command_path.read_text(encoding="utf-8")
            self.assertIn("python", command_txt)
            self.assertIn("--window 16384", command_txt)
            self.assertIn("--stride 8192", command_txt)


    def _write_split(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "version": "nine_genome_dev_lockbox_v1",
                    "dev": [
                        "GCA_000158915.1",
                        "GCA_000154945.1",
                        "GCA_000158875.1",
                        "GCA_000156475.1",
                        "GCA_000568915.1",
                    ],
                    "lockbox": [
                        "GCA_000156435.1",
                        "GCA_000158815.1",
                        "GCA_000158895.1",
                        "GCA_000568255.1",
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_skip_existing_preserves_manifest_and_adds_new_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage2.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)

            initial_yaml = "\n".join(
                [
                    "sweep_id: stage2_tiling_tmp",
                    "defaults:",
                    "  checkpoint: ckpts/best.pt",
                    "  base_results_dir: /srv/nine/results",
                    "  benchmark_base: /srv/nine",
                    "  src_dir: /srv/npmaster/src",
                    "  fna_dir: /srv/nine/fna",
                    "  operating_point: val_frozen_fused",
                    "  server: gpu-a",
                    "trials:",
                    "  - window: 16384",
                    "    stride: 8192",
                    "  - window: 12288",
                    "    stride: 4096",
                ]
            )
            sweep_path.write_text(initial_yaml, encoding="utf-8")

            first_plans = self.runner.plan_tiling_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )
            self.assertEqual(len(first_plans), 2)

            manifest_paths = {
                plan["experiment_id"]: experiments_root / plan["experiment_id"] / "manifest.json"
                for plan in first_plans
            }
            pre_state = {
                eid: (p.read_text(encoding="utf-8"), p.stat().st_mtime_ns)
                for eid, p in manifest_paths.items()
            }

            # Small sleep to guarantee a detectable mtime difference if we
            # accidentally rewrote the file.
            time.sleep(0.05)

            # Extend the sweep with one new trial; re-run with --skip-existing.
            extended_yaml = initial_yaml + "\n".join(
                [
                    "",
                    "  - window: 24576",
                    "    stride: 24576",
                ]
            )
            sweep_path.write_text(extended_yaml, encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                second_plans = self.runner.plan_tiling_trials(
                    split_path=split_path,
                    sweep_path=sweep_path,
                    split_part="dev",
                    experiments_root=experiments_root,
                    skip_existing=True,
                )
            log_output = buf.getvalue()

            # Only the new trial should be emitted in plans.
            self.assertEqual(len(second_plans), 1)
            self.assertEqual(second_plans[0]["trial_id"], "w24576-s24576")

            new_manifest = (
                experiments_root
                / second_plans[0]["experiment_id"]
                / "manifest.json"
            )
            self.assertTrue(new_manifest.exists())

            # Existing manifests must be untouched (content + mtime).
            for eid, path in manifest_paths.items():
                content_before, mtime_before = pre_state[eid]
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    content_before,
                    f"manifest content changed for {eid}",
                )
                self.assertEqual(
                    path.stat().st_mtime_ns,
                    mtime_before,
                    f"manifest mtime changed for {eid}",
                )
                self.assertIn(eid, log_output)
                self.assertIn("skip-existing", log_output)

    def test_per_trial_server_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage2.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)

            sweep_path.write_text(
                "\n".join(
                    [
                        "sweep_id: stage2_tiling_tmp",
                        "defaults:",
                        "  checkpoint: ckpts/best.pt",
                        "  base_results_dir: /srv/nine/results",
                        "  benchmark_base: /srv/nine",
                        "  src_dir: /srv/npmaster/src",
                        "  fna_dir: /srv/nine/fna",
                        "  operating_point: val_frozen_fused",
                        "  server: gpu-default",
                        "trials:",
                        "  - window: 16384",
                        "    stride: 8192",
                        "  - window: 12288",
                        "    stride: 4096",
                        "    server: gpu-override",
                    ]
                ),
                encoding="utf-8",
            )

            plans = self.runner.plan_tiling_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )

            self.assertEqual(len(plans), 2)
            default_plan = plans[0]
            override_plan = plans[1]
            self.assertEqual(default_plan["manifest"]["server"], "gpu-default")
            self.assertEqual(override_plan["manifest"]["server"], "gpu-override")

            # Round-trip: the manifest on disk must match.
            disk_manifest = json.loads(
                (experiments_root / override_plan["experiment_id"] / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(disk_manifest["server"], "gpu-override")


if __name__ == "__main__":
    unittest.main()
