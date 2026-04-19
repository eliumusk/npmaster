#!/usr/bin/env python3
"""Local planner for Stage 3a hardneg sweep trials and command specs.

Per trial, emits a ``command.txt`` with four sequential stages:

  1. ``scripts/mine_hard_negatives.py`` — patches ``meta_arrays.pkl`` with a
     fresh ``hard_negative`` column under the configured ``score_name``,
     ``top_k`` and ``min_score``. Output lands under
     ``<experiment_dir>/artifacts/mining/``.
  2. ``src/train.py`` — warm-starts from ``base_checkpoint`` and trains with
     ``--hard-neg-weight`` applied to the freshly mined meta. Output lands
     under ``<experiment_dir>/artifacts/checkpoints/`` (best_<tag>.pt,
     train_log_<tag>.csv).
  3. ``scripts/inference_stride.py`` — runs the trained best_<tag>.pt over
     dev genomes at the Stage 2 winning tiling (w8192/s2048 by default).
  4. ``scripts/evaluate_nine_genome.py`` — scores the predictions under the
     frozen ``val_frozen_fused`` operating point.

All flag surfaces have been verified against the real scripts:

  * ``scripts/mine_hard_negatives.py:171-183``
  * ``src/train.py:333-358``
  * ``scripts/inference_stride.py:77-92``
  * ``scripts/evaluate_nine_genome.py:358-388``

The runner does NOT execute anything; it only writes manifest + config +
command.txt, mirroring the Stage 2 tiling runner's discipline.
"""
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
DEFAULT_SWEEP_PATH = REPO_ROOT / "configs" / "sweeps" / "stage3_hardneg_v1.yaml"
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


# Required keys in the sweep `defaults:` block. The runner refuses to plan if
# any are missing — these are all paths / knobs the mining + training + eval
# commands must have to be reproducible.
REQUIRED_DEFAULTS = (
    "base_checkpoint",
    "mining_seed_checkpoint",
    "base_meta_arrays",
    "chunk_index",
    "contig_index",
    "embeddings",
    "projections",
    "bgc_csv",
    "score_name",
    "benchmark_base",
    "fna_dir",
    "src_dir",
    "operating_point",
    "window",
    "stride",
    "server",
)


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
            "top_k": int(raw_trial["top_k"]),
            "min_score": float(raw_trial["min_score"]),
            "hard_neg_weight": float(raw_trial["hard_neg_weight"]),
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
    top_k = int(trial["top_k"])
    min_score = float(trial["min_score"])
    hard_neg_weight = float(trial["hard_neg_weight"])
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not (0.0 <= min_score <= 1.0):
        raise ValueError("min_score must be in [0.0, 1.0]")
    if hard_neg_weight <= 0:
        raise ValueError("hard_neg_weight must be positive")
    normalized = {
        "top_k": top_k,
        "min_score": min_score,
        "hard_neg_weight": hard_neg_weight,
    }
    if "server" in trial and trial["server"] is not None:
        server = trial["server"]
        if not isinstance(server, str) or not server.strip():
            raise ValueError("Per-trial server override must be a non-empty string.")
        normalized["server"] = server
    return normalized


def _fmt_float_tag(value: float) -> str:
    """Format floats for trial_id: strip trailing zeros, dot -> 'p'."""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace(".", "p")


def build_trial_id(*, top_k: int, min_score: float, hard_neg_weight: float) -> str:
    return (
        f"tk{int(top_k)}"
        f"-ms{_fmt_float_tag(min_score)}"
        f"-hnw{_fmt_float_tag(hard_neg_weight)}"
    )


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
    trial_tag: str,
) -> dict:
    mining_dir = experiment_rel_dir / "artifacts" / "mining"
    mined_meta = mining_dir / "meta_arrays.pkl"

    train_out_dir = experiment_rel_dir / "artifacts" / "checkpoints"
    best_ckpt = train_out_dir / f"best_{trial_tag}.pt"

    predictions_dir = experiment_rel_dir / "artifacts" / "predictions"
    summary_json = experiment_rel_dir / "metrics" / "summary.json"
    genome_ids_arg = ",".join(genome_ids)

    # -------- stage 1: mine hardnegs --------
    # Flags verified at scripts/mine_hard_negatives.py:173-180.
    mine_argv = [
        "python",
        "scripts/mine_hard_negatives.py",
        "--ckpt",
        str(defaults["mining_seed_checkpoint"]),
        "--embeddings",
        str(defaults["embeddings"]),
        "--meta-arrays",
        str(defaults["base_meta_arrays"]),
        "--contig-index",
        str(defaults["contig_index"]),
        "--out-dir",
        mining_dir.as_posix(),
        "--score-name",
        str(defaults["score_name"]),
        "--top-k",
        str(trial["top_k"]),
        "--min-score",
        f"{trial['min_score']:.4f}",
    ]

    # -------- stage 2: train --------
    # Flags verified at src/train.py:335-357. Note: warm-start flag is --ckpt
    # (not --warm-start); output ckpt + log CSV paths are derived from
    # --out-dir + --tag, so we can't pass --output-ckpt / --log-csv directly.
    train_argv = [
        "python",
        "src/train.py",
        "--embeddings",
        str(defaults["embeddings"]),
        "--projections",
        str(defaults["projections"]),
        "--chunk-index",
        str(defaults["chunk_index"]),
        "--meta-arrays",
        mined_meta.as_posix(),
        "--contig-index",
        str(defaults["contig_index"]),
        "--bgc-csv",
        str(defaults["bgc_csv"]),
        "--ckpt",
        str(defaults["base_checkpoint"]),
        "--out-dir",
        train_out_dir.as_posix(),
        "--seed",
        str(int(defaults.get("seed", 0))),
        "--tag",
        trial_tag,
        "--batch-size",
        str(int(defaults.get("batch_size", 64))),
        "--epochs",
        str(int(defaults.get("epochs", 100))),
        "--hard-neg-weight",
        f"{trial['hard_neg_weight']:.4f}",
    ]

    # -------- stage 3: inference --------
    # Flags verified at scripts/inference_stride.py:78-92.
    inference_argv = [
        "python",
        "scripts/inference_stride.py",
        "--fna-dir",
        str(defaults["fna_dir"]),
        "--out-dir",
        predictions_dir.as_posix(),
        "--ckpt",
        best_ckpt.as_posix(),
        "--src-dir",
        str(defaults["src_dir"]),
        "--genome-ids",
        genome_ids_arg,
        "--window",
        str(int(defaults["window"])),
        "--stride",
        str(int(defaults["stride"])),
        "--layer",
        str(defaults.get("layer", "blocks.20.mlp.l3")),
        "--evo-model",
        str(defaults.get("evo_model", "evo2_7b")),
        "--cache-dir",
        str(defaults.get("cache_dir", "/data/public_models/hub")),
    ]

    # -------- stage 4: evaluate --------
    # Flags verified at scripts/evaluate_nine_genome.py:359-387.
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
        "mine": {
            "argv": mine_argv,
            "shell": shlex.join(mine_argv),
            "outputs": {
                "mining_dir": mining_dir.as_posix(),
                "mined_meta_arrays": mined_meta.as_posix(),
            },
        },
        "train": {
            "argv": train_argv,
            "shell": shlex.join(train_argv),
            "outputs": {
                "train_out_dir": train_out_dir.as_posix(),
                "best_checkpoint": best_ckpt.as_posix(),
            },
        },
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
    trial_tag: str,
    command_specs: dict,
) -> dict:
    server = trial.get("server") or defaults["server"]
    return {
        "experiment_id": experiment_id,
        "task_id": "task7_stage3_hardneg_sweep",
        "sweep_id": defaults["sweep_id"],
        "trial_id": trial_tag,
        "git_commit": resolve_git_commit(),
        "git_dirty": resolve_git_dirty(),
        "split_version": split_data["version"],
        "split_part": split_part,
        "input_genome_ids": list(split_data[split_part]),
        "server": server,
        "top_k": trial["top_k"],
        "min_score": trial["min_score"],
        "hard_neg_weight": trial["hard_neg_weight"],
        "score_name": defaults["score_name"],
        "warm_start": defaults["base_checkpoint"],
        "mining_seed_checkpoint": defaults["mining_seed_checkpoint"],
        "base_meta_arrays": defaults["base_meta_arrays"],
        "window": int(defaults["window"]),
        "stride": int(defaults["stride"]),
        "operating_point": defaults["operating_point"],
        "command": f"bash {experiment_rel_dir.as_posix()}/command.txt",
        "command_specs": command_specs,
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def plan_hardneg_trials(
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

    missing_defaults = [name for name in REQUIRED_DEFAULTS if defaults.get(name) in (None, "")]
    if missing_defaults:
        raise ValueError(f"Missing required defaults in sweep config: {missing_defaults}")

    plans = []
    seen_ids: set[str] = set()
    experiments_root = experiments_root.resolve()
    for trial in sweep["trials"]:
        trial = validate_trial(trial)
        trial_id = build_trial_id(
            top_k=trial["top_k"],
            min_score=trial["min_score"],
            hard_neg_weight=trial["hard_neg_weight"],
        )
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
            trial_tag=trial_id,
        )
        manifest = build_manifest(
            experiment_id=experiment_id,
            experiment_rel_dir=experiment_rel_dir,
            split_data=split_data,
            split_part=split_part,
            defaults=defaults,
            trial=trial,
            trial_tag=trial_id,
            command_specs=command_specs,
        )

        inputs_manifest = {
            "split_path": str(split_path),
            "split_version": split_data["version"],
            "split_part": split_part,
            "genome_ids": genome_ids,
            "base_meta_arrays": defaults["base_meta_arrays"],
            "chunk_index": defaults["chunk_index"],
            "contig_index": defaults["contig_index"],
            "embeddings": defaults["embeddings"],
            "projections": defaults["projections"],
            "base_checkpoint": defaults["base_checkpoint"],
            "mining_seed_checkpoint": defaults["mining_seed_checkpoint"],
        }
        resolved_config = {
            "sweep_id": sweep["sweep_id"],
            "split_part": split_part,
            "defaults": defaults,
            "trial": trial,
        }
        command_text = "\n".join(
            [
                command_specs["mine"]["shell"],
                command_specs["train"]["shell"],
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
    plans = plan_hardneg_trials(
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
