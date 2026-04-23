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


MINIMAL_DEFAULTS = [
    "  base_checkpoint: /warm/H_hardneg_r10.pt",
    "  meta_arrays_path: /pre-mined/stage3_winner/meta_arrays.pkl",
    "  chunk_index: /data/chunk_index.pkl",
    "  contig_index: /data/contig_index.pkl",
    "  embeddings: /data/embeddings.npy",
    "  projections: /data/projections_256.npy",
    "  bgc_csv: /data/bgc_annotations.csv",
    "  hard_neg_weight: 4.0",
    "  benchmark_base: /srv/nine",
    "  fna_dir: /srv/nine/fna",
    "  src_dir: src",
    "  operating_point: val_frozen_fused",
    "  window: 8192",
    "  stride: 2048",
    "  server: gpu-a",
]


def _split_payload() -> dict:
    return {
        "version": "nine_genome_dev_lockbox_v1",
        "dev": [
            "GCA_000154945.1",
            "GCA_000156475.1",
            "GCA_000158875.1",
            "GCA_000158915.1",
            "GCA_000568915.1",
        ],
        "lockbox": [
            "GCA_000156435.1",
            "GCA_000158815.1",
            "GCA_000158895.1",
            "GCA_000568255.1",
        ],
    }


class Stage4TrainingSweepTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module(
            "scripts/run_stage4_training_sweep.py",
            "run_stage4_training_sweep",
        )

    # ---------- yaml / trial parsing ----------

    def test_repo_sweep_config_loads_with_defaults_and_16_trials(self):
        config = self.runner.load_sweep_config(
            ROOT / "configs" / "sweeps" / "stage4_training_v1.yaml"
        )
        self.assertEqual(config["sweep_id"], "stage4_training_v1")
        self.assertEqual(len(config["trials"]), 16)
        self.assertEqual(config["defaults"]["operating_point"], "val_frozen_fused")
        self.assertEqual(config["defaults"]["window"], 8192)
        self.assertEqual(config["defaults"]["stride"], 2048)
        self.assertEqual(float(config["defaults"]["hard_neg_weight"]), 4.0)
        self.assertEqual(
            config["defaults"]["meta_arrays_path"],
            "/data/muskliu/npmaster_exec/snapshots/"
            "f66bdc8d4ce01ab0ec545ffce3043eb5a1d5b166-dirty-gpuenvfix1/repo/"
            "experiments/stage3_hardneg_v1-dev-tk100-ms0-hnw4/"
            "artifacts/mining/meta_arrays.pkl",
        )
        # Default server fixed (per-trial server can override at launch time).
        self.assertEqual(config["defaults"]["server"], "voc_gpu_node7_a800")

    def test_repo_sweep_trial_ids_match_yaml_keys_and_are_unique(self):
        config = self.runner.load_sweep_config(
            ROOT / "configs" / "sweeps" / "stage4_training_v1.yaml"
        )
        ids = [t["trial_id"] for t in config["trials"]]
        self.assertEqual(len(ids), len(set(ids)))
        # The exact 16 ids the user specified.
        expected = {
            "lr-x0p1", "lr-x0p3", "lr-x3p0", "lr-x10p0",
            "wd-0", "wd-1em5", "wd-1em3", "wd-1em2",
            "warm-0", "warm-10", "warm-15", "warm-20",
            "pdrop-0p0", "pdrop-0p5", "cdrop-0p2", "cdrop-0p3",
        }
        self.assertEqual(set(ids), expected)

    def test_build_trial_id_returns_yaml_key_unchanged(self):
        self.assertEqual(self.runner.build_trial_id(trial_id="lr-x0p1"), "lr-x0p1")
        self.assertEqual(self.runner.build_trial_id(trial_id="wd-0"), "wd-0")

    # ---------- validate_trial: hyperparam ranges ----------

    def test_validate_trial_rejects_negative_lr(self):
        with self.assertRaisesRegex(ValueError, "lr_proj"):
            self.runner.validate_trial("bad", {"lr_proj": -1e-5})
        with self.assertRaisesRegex(ValueError, "lr_context"):
            self.runner.validate_trial("bad", {"lr_context": -3e-4})
        with self.assertRaisesRegex(ValueError, "lr_classifier"):
            self.runner.validate_trial("bad", {"lr_classifier": -1.0})

    def test_validate_trial_rejects_negative_wd(self):
        with self.assertRaisesRegex(ValueError, "wd"):
            self.runner.validate_trial("bad", {"wd": -1e-5})

    def test_validate_trial_rejects_negative_warmup(self):
        with self.assertRaisesRegex(ValueError, "warmup_epochs"):
            self.runner.validate_trial("bad", {"warmup_epochs": -1})

    def test_validate_trial_rejects_dropout_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "proj_dropout"):
            self.runner.validate_trial("bad", {"proj_dropout": -0.1})
        with self.assertRaisesRegex(ValueError, "proj_dropout"):
            self.runner.validate_trial("bad", {"proj_dropout": 1.5})
        with self.assertRaisesRegex(ValueError, "cnn_dropout"):
            self.runner.validate_trial("bad", {"cnn_dropout": -0.01})
        with self.assertRaisesRegex(ValueError, "cnn_dropout"):
            self.runner.validate_trial("bad", {"cnn_dropout": 1.01})

    def test_validate_trial_rejects_unknown_key(self):
        with self.assertRaisesRegex(ValueError, "unknown override key"):
            self.runner.validate_trial("bad", {"batch_size": 128})

    def test_validate_trial_accepts_zero_wd_and_zero_warmup_and_zero_dropout(self):
        # wd=0, warmup=0, dropout=0 are valid edge cases.
        self.runner.validate_trial("ok", {"wd": 0.0})
        self.runner.validate_trial("ok", {"warmup_epochs": 0})
        self.runner.validate_trial("ok", {"proj_dropout": 0.0})
        self.runner.validate_trial("ok", {"cnn_dropout": 0.0})

    # ---------- plan / manifest / command.txt ----------

    def _write_split(self, path: Path) -> None:
        path.write_text(json.dumps(_split_payload()), encoding="utf-8")

    def _write_sweep(self, path: Path, extra_lines: list[str] | None = None) -> None:
        lines = [
            "sweep_id: stage4_training_tmp",
            "defaults:",
            *MINIMAL_DEFAULTS,
            "trials:",
            "  lr-x3p0:",
            "    lr_proj: 3.0e-4",
            "    lr_context: 3.0e-3",
            "    lr_classifier: 3.0e-3",
            "  wd-0:",
            "    wd: 0.0",
        ]
        if extra_lines:
            lines.extend(extra_lines)
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_plan_emits_three_stage_command_txt_no_mine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage4.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)
            self._write_sweep(sweep_path)

            plans = self.runner.plan_training_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )

            self.assertEqual(len(plans), 2)
            first = plans[0]
            self.assertEqual(first["trial_id"], "lr-x3p0")

            # Manifest required + Stage-4-specific fields.
            manifest = first["manifest"]
            self.assertEqual(manifest["task_id"], "task8_stage4_training_sweep")
            self.assertEqual(manifest["warm_start"], "/warm/H_hardneg_r10.pt")
            self.assertEqual(
                manifest["base_meta_arrays_path"],
                "/pre-mined/stage3_winner/meta_arrays.pkl",
            )
            self.assertEqual(manifest["split_part"], "dev")
            self.assertEqual(manifest["server"], "gpu-a")
            self.assertEqual(float(manifest["hard_neg_weight"]), 4.0)
            self.assertEqual(manifest["window"], 8192)
            self.assertEqual(manifest["stride"], 2048)
            self.assertEqual(manifest["operating_point"], "val_frozen_fused")
            self.assertIn("git_commit", manifest)
            self.assertIn("git_dirty", manifest)
            self.assertEqual(
                manifest["input_genome_ids"], _split_payload()["dev"]
            )
            # Manifest mirrors the per-trial overrides (only the swept keys).
            self.assertEqual(
                set(manifest["overrides"].keys()),
                {"lr_proj", "lr_context", "lr_classifier"},
            )

            # command_specs has only 3 stages — no `mine` key.
            cs = first["command_specs"]
            self.assertNotIn("mine", cs)
            self.assertIn("train", cs)
            self.assertIn("inference", cs)
            self.assertIn("evaluate", cs)

            # train: warm start via --ckpt, --meta-arrays points DIRECTLY at
            # defaults.meta_arrays_path (no per-trial mining stage).
            train_argv = cs["train"]["argv"]
            self.assertEqual(train_argv[1], "src/train.py")
            for flag in (
                "--embeddings",
                "--projections",
                "--chunk-index",
                "--meta-arrays",
                "--contig-index",
                "--bgc-csv",
                "--ckpt",
                "--out-dir",
                "--tag",
                "--epochs",
                "--hard-neg-weight",
            ):
                self.assertIn(flag, train_argv)
            self.assertNotIn("--warm-start", train_argv)
            self.assertNotIn("--output-ckpt", train_argv)
            meta_idx = train_argv.index("--meta-arrays")
            self.assertEqual(
                train_argv[meta_idx + 1],
                "/pre-mined/stage3_winner/meta_arrays.pkl",
            )
            ckpt_idx = train_argv.index("--ckpt")
            self.assertEqual(train_argv[ckpt_idx + 1], "/warm/H_hardneg_r10.pt")
            hnw_idx = train_argv.index("--hard-neg-weight")
            self.assertAlmostEqual(float(train_argv[hnw_idx + 1]), 4.0)

            # The lr-x3p0 trial overrides the 3 LR flags — they MUST appear.
            for flag in ("--lr-proj", "--lr-context", "--lr-classifier"):
                self.assertIn(flag, train_argv)
            lp_idx = train_argv.index("--lr-proj")
            self.assertAlmostEqual(float(train_argv[lp_idx + 1]), 3.0e-4)
            lc_idx = train_argv.index("--lr-context")
            self.assertAlmostEqual(float(train_argv[lc_idx + 1]), 3.0e-3)
            lcl_idx = train_argv.index("--lr-classifier")
            self.assertAlmostEqual(float(train_argv[lcl_idx + 1]), 3.0e-3)

            # Knobs NOT overridden by lr-x3p0 must NOT appear, so train.py
            # falls back to its own defaults.
            for flag in (
                "--wd",
                "--warmup-epochs",
                "--proj-dropout",
                "--cnn-dropout",
            ):
                self.assertNotIn(flag, train_argv)

            # inference: best_<tag>.pt under checkpoints/.
            inf_argv = cs["inference"]["argv"]
            self.assertEqual(inf_argv[1], "scripts/inference/stride.py")
            ckpt_flag_idx = inf_argv.index("--ckpt")
            self.assertIn("best_lr-x3p0.pt", inf_argv[ckpt_flag_idx + 1])
            window_idx = inf_argv.index("--window")
            self.assertEqual(inf_argv[window_idx + 1], "8192")
            stride_idx = inf_argv.index("--stride")
            self.assertEqual(inf_argv[stride_idx + 1], "2048")

            # evaluate: frozen op + summary.json.
            ev_argv = cs["evaluate"]["argv"]
            self.assertEqual(ev_argv[1], "scripts/evaluate_nine_genome.py")
            op_idx = ev_argv.index("--operating-point")
            self.assertEqual(ev_argv[op_idx + 1], "val_frozen_fused")
            out_idx = ev_argv.index("--out")
            self.assertTrue(ev_argv[out_idx + 1].endswith("metrics/summary.json"))

            # On-disk artifacts.
            experiment_dir = experiments_root / first["experiment_id"]
            manifest_path = experiment_dir / "manifest.json"
            command_path = experiment_dir / "command.txt"
            resolved_config_path = experiment_dir / "config.resolved.yaml"
            inputs_manifest_path = experiment_dir / "inputs" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(command_path.exists())
            self.assertTrue(resolved_config_path.exists())
            self.assertTrue(inputs_manifest_path.exists())

            disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(disk_manifest["warm_start"], "/warm/H_hardneg_r10.pt")
            self.assertEqual(
                disk_manifest["base_meta_arrays_path"],
                "/pre-mined/stage3_winner/meta_arrays.pkl",
            )

            # command.txt: 3 stages only — no `mine_hard_negatives.py` line.
            command_txt = command_path.read_text(encoding="utf-8")
            self.assertNotIn("mine_hard_negatives.py", command_txt)
            train_pos = command_txt.find("src/train.py")
            inf_pos = command_txt.find("scripts/inference/stride.py")
            ev_pos = command_txt.find("scripts/evaluate_nine_genome.py")
            self.assertGreaterEqual(train_pos, 0)
            self.assertGreaterEqual(inf_pos, 0)
            self.assertGreaterEqual(ev_pos, 0)
            self.assertLess(train_pos, inf_pos)
            self.assertLess(inf_pos, ev_pos)
            # The Stage 3 winner pkl must be referenced verbatim in train cmd.
            self.assertIn(
                "/pre-mined/stage3_winner/meta_arrays.pkl",
                command_txt,
            )
            # Override flags appear; non-overrides do not.
            self.assertIn("--lr-proj", command_txt)
            self.assertIn("--lr-context", command_txt)
            self.assertIn("--lr-classifier", command_txt)

            # Second trial (wd-0): only --wd should appear, not LR flags.
            second = plans[1]
            self.assertEqual(second["trial_id"], "wd-0")
            second_train = second["command_specs"]["train"]["argv"]
            self.assertIn("--wd", second_train)
            self.assertNotIn("--lr-proj", second_train)
            self.assertNotIn("--warmup-epochs", second_train)
            self.assertNotIn("--proj-dropout", second_train)
            self.assertNotIn("--cnn-dropout", second_train)
            wd_idx = second_train.index("--wd")
            self.assertEqual(float(second_train[wd_idx + 1]), 0.0)

    def test_plan_repo_yaml_yields_16_unique_experiment_ids_and_trial_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = Path(tmpdir) / "experiments"
            plans = self.runner.plan_training_trials(
                split_path=ROOT / "configs" / "splits" / "nine_genome_dev_lockbox_v1.json",
                sweep_path=ROOT / "configs" / "sweeps" / "stage4_training_v1.yaml",
                split_part="dev",
                experiments_root=experiments_root,
            )
            self.assertEqual(len(plans), 16)
            exp_ids = [p["experiment_id"] for p in plans]
            trial_ids = [p["trial_id"] for p in plans]
            self.assertEqual(len(set(exp_ids)), 16)
            self.assertEqual(len(set(trial_ids)), 16)
            # Every experiment_id is prefixed with sweep_id and split_part.
            for eid in exp_ids:
                self.assertTrue(eid.startswith("stage4_training_v1-dev-"))
            # Spot-check: lr-x10p0 must reference 1e-2 lr_classifier.
            lr_x10 = next(p for p in plans if p["trial_id"] == "lr-x10p0")
            argv = lr_x10["command_specs"]["train"]["argv"]
            lcl_idx = argv.index("--lr-classifier")
            self.assertAlmostEqual(float(argv[lcl_idx + 1]), 1.0e-2)
            # Every command.txt has exactly 3 stages (no mine).
            for plan in plans:
                cmd_txt = (
                    Path(plan["experiment_dir"]) / "command.txt"
                ).read_text(encoding="utf-8")
                self.assertNotIn("mine_hard_negatives.py", cmd_txt)
                self.assertIn("src/train.py", cmd_txt)
                self.assertIn("scripts/inference/stride.py", cmd_txt)
                self.assertIn("scripts/evaluate_nine_genome.py", cmd_txt)

    def test_skip_existing_preserves_manifest_and_adds_new_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage4.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)
            self._write_sweep(sweep_path)

            first_plans = self.runner.plan_training_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )
            self.assertEqual(len(first_plans), 2)

            manifest_paths = {
                plan["experiment_id"]: experiments_root
                / plan["experiment_id"]
                / "manifest.json"
                for plan in first_plans
            }
            pre_state = {
                eid: (p.read_text(encoding="utf-8"), p.stat().st_mtime_ns)
                for eid, p in manifest_paths.items()
            }

            time.sleep(0.05)

            # Append a new trial; re-plan with --skip-existing.
            self._write_sweep(
                sweep_path,
                extra_lines=[
                    "  warm-15:",
                    "    warmup_epochs: 15",
                ],
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                second_plans = self.runner.plan_training_trials(
                    split_path=split_path,
                    sweep_path=sweep_path,
                    split_part="dev",
                    experiments_root=experiments_root,
                    skip_existing=True,
                )
            log_output = buf.getvalue()

            self.assertEqual(len(second_plans), 1)
            self.assertEqual(second_plans[0]["trial_id"], "warm-15")

            new_manifest = (
                experiments_root
                / second_plans[0]["experiment_id"]
                / "manifest.json"
            )
            self.assertTrue(new_manifest.exists())

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
            sweep_path = tmp / "stage4.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)

            sweep_path.write_text(
                "\n".join(
                    [
                        "sweep_id: stage4_training_tmp",
                        "defaults:",
                        *MINIMAL_DEFAULTS,
                        "trials:",
                        "  lr-x3p0:",
                        "    lr_proj: 3.0e-4",
                        "    lr_context: 3.0e-3",
                        "    lr_classifier: 3.0e-3",
                        "  wd-1em3:",
                        "    wd: 1.0e-3",
                        "    server: gpu-override",
                    ]
                ),
                encoding="utf-8",
            )

            plans = self.runner.plan_training_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )
            self.assertEqual(len(plans), 2)
            self.assertEqual(plans[0]["manifest"]["server"], "gpu-a")
            self.assertEqual(plans[1]["manifest"]["server"], "gpu-override")

            disk_manifest = json.loads(
                (
                    experiments_root
                    / plans[1]["experiment_id"]
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(disk_manifest["server"], "gpu-override")


if __name__ == "__main__":
    unittest.main()
