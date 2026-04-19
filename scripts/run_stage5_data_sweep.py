#!/usr/bin/env python3
"""Local planner for the Stage 5 no-9-genome label-variant sweep.

Mirrors ``scripts/run_stage4_training_sweep.py`` in spirit but with a
simpler contract: per trial, write a single ``command.txt`` that runs,
in order:

  0. ``scripts/check_no_nine_genome_leakage.py`` — fail-fast guard on
     the generated trial directory (manifest/config) against the
     embargo list.
  1. ``scripts/build_stage5_labels.py`` — build the Stage 5 labelled
     windows CSV into ``<trial>/inputs/labels_<variant>.csv``.
  2. ``scripts/apply_stage5_meta.py`` — patch ``base_meta_arrays``
     with the new labels, writing
     ``<trial>/inputs/meta_arrays_<variant>.pkl``.
  3. ``python src/train.py`` — warm-start from ``base_checkpoint`` and
     train with the patched meta. Produces
     ``<trial>/artifacts/checkpoints/best_<variant>_s<seed>.pt`` and
     ``train_log_<variant>_s<seed>.csv`` — the train log contains the
     internal-val metric used for Stage 5 selection.

No 9-genome inference / evaluation is planned — that step is run
separately, only once per pre-registered candidate after internal
selection (per ``docs/stage5-no-nine-genome-tuning.md §1.2``).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.layout import (  # noqa: E402
    create_experiment_layout,
    write_command,
)


ALLOWED_VARIANTS = {
    "baseline_v1",
    "tight_pos_075_v1",
    "tight_pos_085_v1",
    "tight_pos_090_v1",
    "tight_pos_095_v1",
    "tight_pos_099_v1",
    "short_bgc_upweight_v1",
    "truncated_exclude_v1",
    "hardpos_hardneg_v2",
}

REQUIRED_DEFAULTS = (
    "input_windows_csv",
    "base_meta_arrays",
    "bgc_csv",
    "embargo_config",
    "base_checkpoint",
    "chunk_index",
    "contig_index",
    "embeddings",
    "projections",
    "hard_neg_weight",
    "stage5_data_root",
)

ALLOWED_TRIAL_KEYS = {
    "variant",
    "short_max_bp",
    "short_weight",
    "hardneg_weight",
    "seed",
    "epochs",
    "lr_proj",
    "lr_context",
    "lr_classifier",
    "server",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_sweep_config(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Stage 5 sweep config must be a mapping.")
    if not data.get("sweep_id"):
        raise ValueError("sweep_id is required.")

    defaults = data.get("defaults") or {}
    for key in REQUIRED_DEFAULTS:
        if key not in defaults:
            raise ValueError(f"defaults.{key} is required.")

    raw_trials = data.get("trials") or {}
    if not raw_trials:
        raise ValueError("trials must contain at least one entry.")

    trials = []
    for trial_id, body in raw_trials.items():
        if not isinstance(body, dict):
            raise ValueError(f"trial {trial_id!r} body must be a mapping.")
        unknown = set(body) - ALLOWED_TRIAL_KEYS
        if unknown:
            raise ValueError(
                f"trial {trial_id!r}: unknown keys {sorted(unknown)}."
            )
        variant = body.get("variant")
        if variant not in ALLOWED_VARIANTS:
            raise ValueError(
                f"trial {trial_id!r}: unknown variant {variant!r}. "
                f"Allowed: {sorted(ALLOWED_VARIANTS)}"
            )
        trials.append({"trial_id": str(trial_id), **body})

    return {
        "sweep_id": str(data["sweep_id"]),
        "defaults": defaults,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Per-trial command construction
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def plan_trial(
    sweep: dict,
    trial: dict,
    *,
    experiments_root: Path,
) -> dict:
    """Build all artefacts for a single Stage 5 trial.

    Returns a dict with keys ``command_text``, ``manifest``,
    ``trial_dir`` — the caller is responsible for writing them.
    """
    defaults = sweep["defaults"]
    sweep_id = sweep["sweep_id"]
    variant = trial["variant"]
    trial_id = trial["trial_id"]
    seed = int(trial.get("seed", defaults.get("seed", 0)))
    epochs = int(trial.get("epochs", defaults.get("epochs", 60)))

    experiment_id = f"{sweep_id}-{trial_id}"
    trial_dir = experiments_root / experiment_id

    # Commands use repo-relative paths for trial outputs so the script
    # runs unchanged on the server (cwd = repo root). The planner still
    # records the absolute trial_dir on disk for writing manifests.
    try:
        rel_trial_dir = trial_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel_trial_dir = Path("experiments") / experiment_id

    tag = f"{variant}_s{seed}"
    labels_csv_rel = rel_trial_dir / "inputs" / f"labels_{variant}.csv"
    meta_pkl_rel = rel_trial_dir / "inputs" / f"meta_arrays_{variant}.pkl"
    train_out_rel = rel_trial_dir / "artifacts" / "checkpoints"
    patch_summary_rel = (
        rel_trial_dir / "metrics" / f"meta_patch_summary_{variant}.json"
    )
    # Keep absolute paths only for downstream file-writing bookkeeping.
    labels_csv = trial_dir / "inputs" / f"labels_{variant}.csv"
    meta_pkl = trial_dir / "inputs" / f"meta_arrays_{variant}.pkl"
    train_out = trial_dir / "artifacts" / "checkpoints"

    # Stage 0: leakage check over the trial directory after it's written.
    # We scan the trial dir itself (which will contain config + manifest)
    # to ensure no embargoed ID slipped in.
    leakage_argv = [
        "python",
        "scripts/check_no_nine_genome_leakage.py",
        "--embargo",
        str(defaults["embargo_config"]),
        "--paths",
        rel_trial_dir.as_posix(),
    ]

    # Stage 1: build labelled CSV
    build_argv = [
        "python",
        "scripts/build_stage5_labels.py",
        "--input",
        str(defaults["input_windows_csv"]),
        "--output",
        labels_csv_rel.as_posix(),
        "--variant",
        variant,
        "--embargo",
        str(defaults["embargo_config"]),
        "--bgc-csv",
        str(defaults["bgc_csv"]),
    ]
    if "short_max_bp" in trial:
        build_argv += ["--short-max-bp", str(float(trial["short_max_bp"]))]
    if "short_weight" in trial:
        build_argv += ["--short-weight", str(float(trial["short_weight"]))]
    if "hardneg_weight" in trial:
        build_argv += [
            "--hardneg-weight",
            str(float(trial["hardneg_weight"])),
        ]

    # Stage 2: patch meta arrays
    apply_argv = [
        "python",
        "scripts/apply_stage5_meta.py",
        "--meta",
        str(defaults["base_meta_arrays"]),
        "--stage5-csv",
        labels_csv_rel.as_posix(),
        "--output",
        meta_pkl_rel.as_posix(),
        "--summary-json",
        patch_summary_rel.as_posix(),
    ]

    # Stage 3: train
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
        meta_pkl_rel.as_posix(),
        "--contig-index",
        str(defaults["contig_index"]),
        "--bgc-csv",
        str(defaults["bgc_csv"]),
        "--ckpt",
        str(defaults["base_checkpoint"]),
        "--out-dir",
        train_out_rel.as_posix(),
        "--seed",
        str(seed),
        "--tag",
        tag,
        "--batch-size",
        str(int(defaults.get("batch_size", 64))),
        "--epochs",
        str(epochs),
        "--hard-neg-weight",
        f"{float(defaults['hard_neg_weight']):.4f}",
    ]
    for key in ("lr_proj", "lr_context", "lr_classifier"):
        if key in trial:
            flag = "--" + key.replace("_", "-")
            train_argv += [flag, f"{float(trial[key]):.10g}"]

    command_text = "\n".join(
        [
            "# Stage 5 no-9 trial: " + experiment_id,
            "# Variant: " + variant,
            "set -euo pipefail",
            "",
            "# 0. leakage check (trial dir vs embargo list)",
            shlex.join(leakage_argv),
            "",
            "# 1. build stage5 labelled CSV",
            shlex.join(build_argv),
            "",
            "# 2. patch meta arrays with new labels/weights",
            shlex.join(apply_argv),
            "",
            "# 3. train (warm-start; train.py logs internal val metrics)",
            shlex.join(train_argv),
        ]
    )

    manifest = {
        "experiment_id": experiment_id,
        "sweep_id": sweep_id,
        "trial_id": trial_id,
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "tag": tag,
        "server": trial.get("server"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "frozen": True,
        "nine_genome_policy": "final_report_only",
        "uses_nine_genome_for_selection": False,
        "inputs": {
            "windows_csv": str(defaults["input_windows_csv"]),
            "base_meta_arrays": str(defaults["base_meta_arrays"]),
            "bgc_csv": str(defaults["bgc_csv"]),
            "base_checkpoint": str(defaults["base_checkpoint"]),
            "embargo_config": str(defaults["embargo_config"]),
        },
        "per_trial_artefacts": {
            "labels_csv": str(labels_csv),
            "meta_arrays_pkl": str(meta_pkl),
            "train_out_dir": str(train_out),
            "best_checkpoint": str(train_out / f"best_{tag}.pt"),
            "train_log_csv": str(train_out / f"train_log_{tag}.csv"),
        },
    }

    return {
        "command_text": command_text,
        "manifest": manifest,
        "trial_dir": trial_dir,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sweeps" / "stage5_no9_data_v1.yaml",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "experiments",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="If set, only plan these variant ids (filter by trial_id).",
    )
    args = parser.parse_args(argv)

    sweep = load_sweep_config(args.config)
    out_root = args.out_root

    written = []
    for trial in sweep["trials"]:
        if args.variants and trial["trial_id"] not in args.variants:
            continue
        result = plan_trial(sweep, trial, experiments_root=out_root)
        trial_dir = result["trial_dir"]
        paths = create_experiment_layout(
            out_root, result["manifest"]["experiment_id"]
        )
        write_command(paths["command"], result["command_text"])
        paths["manifest"].write_text(
            json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Also write a compact config.resolved.yaml for auditability.
        paths["config"].write_text(
            yaml.safe_dump(
                {
                    "sweep_id": sweep["sweep_id"],
                    "trial": trial,
                    "defaults": sweep["defaults"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        written.append(trial_dir)

    print(
        json.dumps(
            {
                "sweep_id": sweep["sweep_id"],
                "trials_written": [str(p) for p in written],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
