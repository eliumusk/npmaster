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
    "  mining_seed_checkpoint: /seed/best_route_b_s0.pt",
    "  base_meta_arrays: /data/meta_arrays.pkl",
    "  chunk_index: /data/chunk_index.pkl",
    "  contig_index: /data/contig_index.pkl",
    "  embeddings: /data/embeddings.npy",
    "  projections: /data/projections_256.npy",
    "  bgc_csv: /data/bgc_annotations.csv",
    "  score_name: fused",
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


class Stage3HardnegSweepTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module(
            "scripts/run_stage3_hardneg_sweep.py",
            "run_stage3_hardneg_sweep",
        )

    # ---------- yaml / trial parsing ----------

    def test_repo_sweep_config_loads_with_defaults_and_trials(self):
        config = self.runner.load_sweep_config(
            ROOT / "configs" / "sweeps" / "stage3_hardneg_v1.yaml"
        )
        self.assertEqual(config["sweep_id"], "stage3_hardneg_v1")
        self.assertGreaterEqual(len(config["trials"]), 1)
        trial = config["trials"][0]
        self.assertIsInstance(trial["top_k"], int)
        self.assertIsInstance(trial["min_score"], float)
        self.assertIsInstance(trial["hard_neg_weight"], float)
        self.assertEqual(config["defaults"]["score_name"], "fused")
        self.assertEqual(config["defaults"]["window"], 8192)
        self.assertEqual(config["defaults"]["stride"], 2048)

    # ---------- trial_id generation ----------

    def test_build_trial_id_is_deterministic(self):
        a = self.runner.build_trial_id(top_k=5000, min_score=0.80, hard_neg_weight=1.0)
        self.assertEqual(a, "tk5000-ms0p8-hnw1")
        b = self.runner.build_trial_id(
            top_k=10000, min_score=0.95, hard_neg_weight=4.0
        )
        self.assertEqual(b, "tk10000-ms0p95-hnw4")
        # Same inputs => same id.
        again = self.runner.build_trial_id(
            top_k=5000, min_score=0.80, hard_neg_weight=1.0
        )
        self.assertEqual(again, a)

    def test_build_trial_id_is_unique_across_axes(self):
        seen: set[str] = set()
        cases = [
            (5000, 0.80, 1.0),
            (5000, 0.80, 4.0),
            (5000, 0.90, 1.0),
            (10000, 0.80, 1.0),
            (5000, 0.95, 2.0),
        ]
        for top_k, min_score, hnw in cases:
            tid = self.runner.build_trial_id(
                top_k=top_k, min_score=min_score, hard_neg_weight=hnw
            )
            self.assertNotIn(tid, seen, f"duplicate trial_id: {tid}")
            seen.add(tid)

    # ---------- validate_trial ----------

    def test_validate_trial_rejects_nonpositive_hard_neg_weight(self):
        with self.assertRaisesRegex(ValueError, "hard_neg_weight"):
            self.runner.validate_trial(
                {"top_k": 5000, "min_score": 0.8, "hard_neg_weight": 0.0}
            )
        with self.assertRaisesRegex(ValueError, "hard_neg_weight"):
            self.runner.validate_trial(
                {"top_k": 5000, "min_score": 0.8, "hard_neg_weight": -1.0}
            )

    def test_validate_trial_rejects_nonpositive_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.runner.validate_trial(
                {"top_k": 0, "min_score": 0.8, "hard_neg_weight": 1.0}
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.runner.validate_trial(
                {"top_k": -10, "min_score": 0.8, "hard_neg_weight": 1.0}
            )

    def test_validate_trial_rejects_min_score_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "min_score"):
            self.runner.validate_trial(
                {"top_k": 5000, "min_score": -0.1, "hard_neg_weight": 1.0}
            )
        with self.assertRaisesRegex(ValueError, "min_score"):
            self.runner.validate_trial(
                {"top_k": 5000, "min_score": 1.01, "hard_neg_weight": 1.0}
            )

    # ---------- plan / manifest / command.txt ----------

    def _write_split(self, path: Path) -> None:
        path.write_text(json.dumps(_split_payload()), encoding="utf-8")

    def _write_sweep(self, path: Path, extra_trials: list[str] | None = None) -> None:
        lines = [
            "sweep_id: stage3_hardneg_tmp",
            "defaults:",
            *MINIMAL_DEFAULTS,
            "trials:",
            "  - top_k: 5000",
            "    min_score: 0.80",
            "    hard_neg_weight: 1.0",
            "  - top_k: 10000",
            "    min_score: 0.90",
            "    hard_neg_weight: 4.0",
        ]
        if extra_trials:
            lines.extend(extra_trials)
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_plan_emits_four_stage_command_txt_and_manifest_with_warm_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage3.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)
            self._write_sweep(sweep_path)

            plans = self.runner.plan_hardneg_trials(
                split_path=split_path,
                sweep_path=sweep_path,
                split_part="dev",
                experiments_root=experiments_root,
            )

            self.assertEqual(len(plans), 2)
            first = plans[0]
            self.assertEqual(first["trial_id"], "tk5000-ms0p8-hnw1")

            # Manifest has required + task-specific fields.
            manifest = first["manifest"]
            self.assertEqual(manifest["task_id"], "task7_stage3_hardneg_sweep")
            self.assertEqual(manifest["warm_start"], "/warm/H_hardneg_r10.pt")
            self.assertEqual(
                manifest["mining_seed_checkpoint"], "/seed/best_route_b_s0.pt"
            )
            self.assertEqual(manifest["split_part"], "dev")
            self.assertEqual(manifest["server"], "gpu-a")
            self.assertEqual(manifest["top_k"], 5000)
            self.assertAlmostEqual(manifest["min_score"], 0.80)
            self.assertAlmostEqual(manifest["hard_neg_weight"], 1.0)
            self.assertEqual(manifest["score_name"], "fused")
            self.assertEqual(manifest["window"], 8192)
            self.assertEqual(manifest["stride"], 2048)
            self.assertIn("git_commit", manifest)
            self.assertIn("git_dirty", manifest)
            self.assertEqual(
                manifest["input_genome_ids"], _split_payload()["dev"]
            )

            # command_specs has all four stages with correct flag surfaces.
            cs = first["command_specs"]
            self.assertIn("mine", cs)
            self.assertIn("train", cs)
            self.assertIn("inference", cs)
            self.assertIn("evaluate", cs)

            # mine: real flags only (--score-name, --top-k, --min-score,
            # --out-dir, --meta-arrays, --embeddings, --contig-index, --ckpt).
            mine_argv = cs["mine"]["argv"]
            self.assertEqual(mine_argv[0], "python")
            self.assertEqual(mine_argv[1], "scripts/mine_hard_negatives.py")
            for flag in (
                "--ckpt",
                "--embeddings",
                "--meta-arrays",
                "--contig-index",
                "--out-dir",
                "--score-name",
                "--top-k",
                "--min-score",
            ):
                self.assertIn(flag, mine_argv)
            # Must NOT fabricate non-existent flags.
            self.assertNotIn("--input", mine_argv)
            self.assertNotIn("--output", mine_argv)

            # train: warm start via --ckpt (not --warm-start), --out-dir + --tag.
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
            self.assertNotIn("--log-csv", train_argv)
            # Training reads the freshly mined meta from the mining stage
            # output (not the base_meta_arrays).
            meta_idx = train_argv.index("--meta-arrays")
            self.assertIn(
                "artifacts/mining/meta_arrays.pkl",
                train_argv[meta_idx + 1],
            )
            # Warm-start points at base_checkpoint.
            ckpt_idx = train_argv.index("--ckpt")
            self.assertEqual(train_argv[ckpt_idx + 1], "/warm/H_hardneg_r10.pt")
            # hard_neg_weight value is plumbed through.
            hnw_idx = train_argv.index("--hard-neg-weight")
            self.assertAlmostEqual(float(train_argv[hnw_idx + 1]), 1.0)

            # inference: uses best_<tag>.pt under checkpoints/.
            inf_argv = cs["inference"]["argv"]
            self.assertEqual(inf_argv[1], "scripts/inference/stride.py")
            ckpt_flag_idx = inf_argv.index("--ckpt")
            self.assertIn("best_tk5000-ms0p8-hnw1.pt", inf_argv[ckpt_flag_idx + 1])
            window_idx = inf_argv.index("--window")
            self.assertEqual(inf_argv[window_idx + 1], "8192")
            stride_idx = inf_argv.index("--stride")
            self.assertEqual(inf_argv[stride_idx + 1], "2048")

            # evaluate: frozen operating point + summary.json.
            ev_argv = cs["evaluate"]["argv"]
            self.assertEqual(ev_argv[1], "scripts/evaluate_nine_genome.py")
            op_idx = ev_argv.index("--operating-point")
            self.assertEqual(ev_argv[op_idx + 1], "val_frozen_fused")
            out_idx = ev_argv.index("--out")
            self.assertTrue(ev_argv[out_idx + 1].endswith("metrics/summary.json"))

            # On-disk artifacts exist and round-trip.
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
                disk_manifest["command"],
                f"bash experiments/{first['experiment_id']}/command.txt",
            )

            # command.txt must contain all 4 stages in order.
            command_txt = command_path.read_text(encoding="utf-8")
            mine_pos = command_txt.find("scripts/mine_hard_negatives.py")
            train_pos = command_txt.find("src/train.py")
            inf_pos = command_txt.find("scripts/inference/stride.py")
            ev_pos = command_txt.find("scripts/evaluate_nine_genome.py")
            self.assertGreaterEqual(mine_pos, 0)
            self.assertGreaterEqual(train_pos, 0)
            self.assertGreaterEqual(inf_pos, 0)
            self.assertGreaterEqual(ev_pos, 0)
            self.assertLess(mine_pos, train_pos)
            self.assertLess(train_pos, inf_pos)
            self.assertLess(inf_pos, ev_pos)
            # The key hardneg knobs must be literally present.
            self.assertIn("--hard-neg-weight", command_txt)
            self.assertIn("--top-k 5000", command_txt)
            self.assertIn("--min-score 0.8000", command_txt)

    def test_skip_existing_preserves_manifest_and_adds_new_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_path = tmp / "split.json"
            sweep_path = tmp / "stage3.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)
            self._write_sweep(sweep_path)

            first_plans = self.runner.plan_hardneg_trials(
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

            # Append a new trial and re-plan with --skip-existing.
            self._write_sweep(
                sweep_path,
                extra_trials=[
                    "  - top_k: 5000",
                    "    min_score: 0.80",
                    "    hard_neg_weight: 2.0",
                ],
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                second_plans = self.runner.plan_hardneg_trials(
                    split_path=split_path,
                    sweep_path=sweep_path,
                    split_part="dev",
                    experiments_root=experiments_root,
                    skip_existing=True,
                )
            log_output = buf.getvalue()

            self.assertEqual(len(second_plans), 1)
            self.assertEqual(second_plans[0]["trial_id"], "tk5000-ms0p8-hnw2")

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
            sweep_path = tmp / "stage3.yaml"
            experiments_root = tmp / "experiments"
            self._write_split(split_path)

            sweep_path.write_text(
                "\n".join(
                    [
                        "sweep_id: stage3_hardneg_tmp",
                        "defaults:",
                        *MINIMAL_DEFAULTS,
                        "trials:",
                        "  - top_k: 5000",
                        "    min_score: 0.80",
                        "    hard_neg_weight: 1.0",
                        "  - top_k: 10000",
                        "    min_score: 0.90",
                        "    hard_neg_weight: 4.0",
                        "    server: gpu-override",
                    ]
                ),
                encoding="utf-8",
            )

            plans = self.runner.plan_hardneg_trials(
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
