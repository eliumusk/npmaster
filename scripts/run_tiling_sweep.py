#!/usr/bin/env python3
"""Local planner for tiling_v1 sweep trials and command specs."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.layout import create_experiment_layout, write_command, write_resolved_config
from experiments.manifest import load_split, write_manifest

DEFAULT_SPLIT_PATH = REPO_ROOT / "configs" / "splits" / "nine_genome_dev_lockbox_v1.json"
DEFAULT_SWEEP_PATH = REPO_ROOT / "configs" / "sweeps" / "tiling_v1.yaml"
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def load_sweep_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a mapping.")
    if not data.get("sweep_id"):
        raise ValueError("Sweep config requires a non-empty sweep_id.")
    trials = data.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("Sweep config requires at least one trial.")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("Sweep defaults must be a mapping.")

    normalized_trials = []
    for raw_trial in trials:
        if not isinstance(raw_trial, dict):
            raise ValueError("Each trial must be a mapping.")
        trial = {
            "window": int(raw_trial["window"]),
            "stride": int(raw_trial["stride"]),
        }
        if "server" in raw_trial and raw_trial["server"] is not None:
            server = raw_trial["server"]
            if not isinstance(server, str) or not server.strip():
                raise ValueError("Per-trial server override must be a non-empty string.")
            trial["server"] = server
        validate_trial(trial)
        normalized_trials.append(trial)

    return {
        "sweep_id": str(data["sweep_id"]),
        "defaults": defaults,
        "trials": normalized_trials,
    }


def validate_trial(trial: dict) -> dict:
    window = int(trial["window"])
    stride = int(trial["stride"])
    if window <= 0:
        raise ValueError("window must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > window:
        raise ValueError("stride must be less than or equal to window")
    normalized = {"window": window, "stride": stride}
    if "server" in trial and trial["server"] is not None:
        server = trial["server"]
        if not isinstance(server, str) or not server.strip():
            raise ValueError("Per-trial server override must be a non-empty string.")
        normalized["server"] = server
    return normalized


def build_trial_id(*, window: int, stride: int) -> str:
    return f"w{int(window)}-s{int(stride)}"


def resolve_git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def resolve_git_dirty() -> bool:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return False
    return bool(proc.stdout.strip())


def repo_relative_experiment_dir(experiment_dir: Path) -> Path:
    try:
        return experiment_dir.relative_to(REPO_ROOT)
    except ValueError:
        return Path("experiments") / experiment_dir.name


def build_command_specs(
    *,
    defaults: dict,
    experiment_rel_dir: Path,
    genome_ids: list[str],
    trial: dict,
) -> dict:
    predictions_dir = experiment_rel_dir / "artifacts" / "predictions"
    summary_json = experiment_rel_dir / "metrics" / "summary.json"
    genome_ids_arg = ",".join(genome_ids)

    inference_argv = [
        "python",
        "scripts/inference_stride.py",
        "--fna-dir",
        str(defaults["fna_dir"]),
        "--out-dir",
        predictions_dir.as_posix(),
        "--ckpt",
        str(defaults["checkpoint"]),
        "--src-dir",
        str(defaults["src_dir"]),
        "--genome-ids",
        genome_ids_arg,
        "--window",
        str(trial["window"]),
        "--stride",
        str(trial["stride"]),
        "--layer",
        str(defaults.get("layer", "blocks.20.mlp.l3")),
        "--evo-model",
        str(defaults.get("evo_model", "evo2_7b")),
        "--cache-dir",
        str(defaults.get("cache_dir", "/data/public_models/hub")),
    ]
    evaluate_argv = [
        "python",
        "scripts/evaluate_nine_genome.py",
        "--base",
        str(defaults["benchmark_base"]),
        "--results-dir",
        predictions_dir.as_posix(),
        "--genome-ids",
        genome_ids_arg,
        "--operating-point",
        str(defaults["operating_point"]),
        "--out",
        summary_json.as_posix(),
    ]
    return {
        "inference": {
            "argv": inference_argv,
            "shell": shlex.join(inference_argv),
            "outputs": {
                "predictions_dir": predictions_dir.as_posix(),
            },
        },
        "evaluate": {
            "argv": evaluate_argv,
            "shell": shlex.join(evaluate_argv),
            "outputs": {
                "summary_json": summary_json.as_posix(),
            },
        },
    }


def build_manifest(
    *,
    experiment_id: str,
    experiment_rel_dir: Path,
    split_data: dict,
    split_part: str,
    defaults: dict,
    trial: dict,
    command_specs: dict,
) -> dict:
    server = trial.get("server") or defaults["server"]
    return {
        "experiment_id": experiment_id,
        "task_id": "task4_tiling_sweep",
        "sweep_id": defaults["sweep_id"],
        "trial_id": build_trial_id(window=trial["window"], stride=trial["stride"]),
        "git_commit": resolve_git_commit(),
        "git_dirty": resolve_git_dirty(),
        "split_version": split_data["version"],
        "split_part": split_part,
        "input_genome_ids": list(split_data[split_part]),
        "server": server,
        "window": trial["window"],
        "stride": trial["stride"],
        "operating_point": defaults["operating_point"],
        "checkpoint": defaults["checkpoint"],
        "command": f"bash {experiment_rel_dir.as_posix()}/command.txt",
        "command_specs": command_specs,
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def plan_tiling_trials(
    *,
    split_path: Path,
    sweep_path: Path,
    split_part: str,
    experiments_root: Path,
    skip_existing: bool = False,
) -> list[dict]:
    split_data = load_split(split_path)
    if split_part not in split_data:
        raise ValueError(f"Split file does not define part {split_part!r}")
    genome_ids = list(split_data[split_part])

    sweep = load_sweep_config(sweep_path)
    defaults = deepcopy(sweep["defaults"])
    defaults["sweep_id"] = sweep["sweep_id"]

    required_defaults = (
        "checkpoint",
        "benchmark_base",
        "base_results_dir",
        "fna_dir",
        "src_dir",
        "operating_point",
        "server",
    )
    missing_defaults = [name for name in required_defaults if not defaults.get(name)]
    if missing_defaults:
        raise ValueError(f"Missing required defaults in sweep config: {missing_defaults}")

    plans = []
    seen_ids: set[str] = set()
    experiments_root = experiments_root.resolve()
    for trial in sweep["trials"]:
        trial = validate_trial(trial)
        trial_id = build_trial_id(window=trial["window"], stride=trial["stride"])
        if trial_id in seen_ids:
            raise ValueError(f"Duplicate trial_id generated for {trial_id}")
        seen_ids.add(trial_id)

        experiment_id = f"{sweep['sweep_id']}-{split_part}-{trial_id}"
        if skip_existing:
            candidate_manifest = experiments_root / experiment_id / "manifest.json"
            if candidate_manifest.exists() and candidate_manifest.stat().st_size > 0:
                print(
                    f"[skip-existing] {experiment_id}: manifest already exists at "
                    f"{candidate_manifest}; skipping.",
                    flush=True,
                )
                continue
        layout = create_experiment_layout(experiments_root, experiment_id)
        experiment_dir = layout["root"]
        experiment_rel_dir = repo_relative_experiment_dir(experiment_dir)
        command_specs = build_command_specs(
            defaults=defaults,
            experiment_rel_dir=experiment_rel_dir,
            genome_ids=genome_ids,
            trial=trial,
        )
        manifest = build_manifest(
            experiment_id=experiment_id,
            experiment_rel_dir=experiment_rel_dir,
            split_data=split_data,
            split_part=split_part,
            defaults=defaults,
            trial=trial,
            command_specs=command_specs,
        )

        inputs_manifest = {
            "split_path": str(split_path),
            "split_version": split_data["version"],
            "split_part": split_part,
            "genome_ids": genome_ids,
        }
        resolved_config = {
            "sweep_id": sweep["sweep_id"],
            "split_part": split_part,
            "defaults": defaults,
            "trial": trial,
        }
        command_text = "\n".join(
            [
                command_specs["inference"]["shell"],
                command_specs["evaluate"]["shell"],
            ]
        )

        write_manifest(layout["manifest"], manifest)
        write_text(layout["input_manifest"], json.dumps(inputs_manifest, indent=2, sort_keys=True))
        write_resolved_config(layout["config"], resolved_config)
        write_command(layout["command"], command_text)

        plans.append(
            {
                "experiment_id": experiment_id,
                "trial_id": trial_id,
                "experiment_dir": str(experiment_dir),
                "manifest": manifest,
                "command_specs": command_specs,
            }
        )
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--sweep-path", default=str(DEFAULT_SWEEP_PATH))
    parser.add_argument("--split-part", default="dev", choices=["dev", "lockbox"])
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument("--out", default=None, help="Optional plan summary JSON path.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip trials whose experiments/<experiment_id>/manifest.json already exists. "
            "Existing manifests are not overwritten."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = plan_tiling_trials(
        split_path=Path(args.split_path),
        sweep_path=Path(args.sweep_path),
        split_part=args.split_part,
        experiments_root=Path(args.experiments_root),
        skip_existing=args.skip_existing,
    )
    payload = {
        "split_path": str(Path(args.split_path)),
        "sweep_path": str(Path(args.sweep_path)),
        "split_part": args.split_part,
        "experiments_root": str(Path(args.experiments_root)),
        "skip_existing": args.skip_existing,
        "plans": plans,
    }
    if args.out:
        write_text(Path(args.out), json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
