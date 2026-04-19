#!/usr/bin/env python3
"""tiling_decoder_v1: decoder micro-sweep applied to top tiling_v1 tilings.

CPU-only. Consumes frozen per-window predictions produced by the tiling_v1
sweep and redecodes them under a grid of decoder parameters.

Ranks trials lex (recall, precision) under the `bench` scorer on dev, writes
one pooled summary.{json,csv} plus per-trial immutable experiment dirs that
match the tiling_v1 layout (manifest.json + config.resolved.yaml + command.txt
+ metrics/summary.json).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate_nine_genome as eval_nine_genome  # noqa: E402
from experiments.layout import create_experiment_layout, write_command, write_resolved_config  # noqa: E402
from experiments.manifest import load_split, write_manifest  # noqa: E402


DEFAULT_SWEEP_PATH = REPO_ROOT / "configs" / "sweeps" / "tiling_decoder_v1.yaml"
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def resolve_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
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


def load_sweep_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a mapping.")
    for key in ("sweep_id", "tilings", "decoder_grid", "decoder_fixed", "benchmark", "split"):
        if key not in data:
            raise ValueError(f"Sweep config missing required key: {key}")
    if not isinstance(data["tilings"], list) or not data["tilings"]:
        raise ValueError("Sweep config requires at least one tiling.")
    decoder_grid = data["decoder_grid"]
    if not isinstance(decoder_grid, dict) or not decoder_grid:
        raise ValueError("decoder_grid must be a non-empty mapping.")
    for key, values in decoder_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"decoder_grid[{key!r}] must be a non-empty list.")
    return data


def decoder_trial_id(index: int) -> str:
    return f"decoder-{index:03d}"


def generate_decoder_trials(sweep_config: dict) -> list[dict]:
    grid = sweep_config["decoder_grid"]
    keys = list(grid)
    trials = []
    skipped = 0
    for combo in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, combo, strict=True))
        # Hysteresis decoder requires extend <= start.
        if float(params["extend_threshold"]) > float(params["start_threshold"]):
            skipped += 1
            continue
        trials.append({"decoder_id": decoder_trial_id(len(trials) + 1), "params": params})
    if skipped:
        print(
            f"generate_decoder_trials: skipped {skipped} combos where "
            "extend_threshold > start_threshold",
            flush=True,
        )
    return trials


def resolve_trial_params(decoder_params: dict, fixed: dict) -> dict:
    return {
        "score": decoder_params["score_family"],
        "start_threshold": float(decoder_params["start_threshold"]),
        "extend_threshold": float(decoder_params["extend_threshold"]),
        "max_gap": int(fixed.get("max_gap", 0)),
        "length_cap": fixed.get("length_cap"),
        "min_support_windows": int(decoder_params.get("min_support_windows", 1)),
        "valley_threshold": fixed.get("valley_threshold"),
        "valley_min_windows": int(fixed.get("valley_min_windows", 1)),
        "stride": None,
        "checkpoint": None,
        "expected_metrics": None,
        "operating_point": None,
    }


def run_single_trial(
    *,
    tiling: dict,
    decoder: dict,
    sweep_config: dict,
    split_data: dict,
    split_part: str,
    experiments_root: Path,
    sweep_id: str,
    git_commit: str,
    git_dirty: bool,
) -> dict:
    predictions_dir = (REPO_ROOT / tiling["predictions_dir"]).resolve()
    benchmark_base = (REPO_ROOT / sweep_config["benchmark"]["base"]).resolve()

    params = resolve_trial_params(decoder["params"], sweep_config["decoder_fixed"])
    genome_ids = list(split_data[split_part])

    eval_output = eval_nine_genome.evaluate_results_dir(
        benchmark_base,
        predictions_dir,
        genome_ids,
        params,
    )

    experiment_id = f"{sweep_id}-{split_part}-{tiling['trial_id']}-{decoder['decoder_id']}"
    layout = create_experiment_layout(experiments_root, experiment_id)
    experiment_dir = layout["root"]
    try:
        experiment_rel = experiment_dir.relative_to(REPO_ROOT)
    except ValueError:
        experiment_rel = Path("experiments") / experiment_dir.name

    summary_rel = experiment_rel / "metrics" / "summary.json"
    predictions_rel = Path(tiling["predictions_dir"])
    benchmark_rel = Path(sweep_config["benchmark"]["base"])

    # Reproducibility: record a concrete evaluate_nine_genome.py invocation
    # that would regenerate metrics/summary.json. We pass decoder flags
    # explicitly (no yaml operating point) so the call is self-contained.
    evaluate_argv = [
        "python",
        "scripts/evaluate_nine_genome.py",
        "--base",
        benchmark_rel.as_posix(),
        "--results-dir",
        predictions_rel.as_posix(),
        "--genome-ids",
        ",".join(genome_ids),
        "--start-threshold",
        str(params["start_threshold"]),
        "--extend-threshold",
        str(params["extend_threshold"]),
        "--max-gap",
        str(params["max_gap"]),
        "--min-support-windows",
        str(params["min_support_windows"]),
        "--out",
        summary_rel.as_posix(),
    ]
    if params["length_cap"] is not None:
        evaluate_argv.extend(["--length-cap", str(params["length_cap"])])

    command_specs = {
        "evaluate": {
            "argv": evaluate_argv,
            "shell": shlex.join(evaluate_argv),
            "outputs": {"summary_json": summary_rel.as_posix()},
        },
        # Decoder flags alone don't pin score_family; record it separately.
        "score_family": decoder["params"]["score_family"],
    }

    manifest = {
        "experiment_id": experiment_id,
        "task_id": "task5_tiling_decoder_sweep",
        "sweep_id": sweep_id,
        "trial_id": f"{tiling['trial_id']}-{decoder['decoder_id']}",
        "tiling_trial_id": tiling["trial_id"],
        "decoder_trial_id": decoder["decoder_id"],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "split_version": split_data["version"],
        "split_part": split_part,
        "input_genome_ids": genome_ids,
        "server": "local_cpu",
        "window": int(tiling["window"]),
        "stride": int(tiling["stride"]),
        "source_experiment": tiling["source_experiment"],
        "predictions_dir": predictions_rel.as_posix(),
        "operating_point": None,
        "checkpoint": "best_route_b_s0",
        "decoder_params": decoder["params"],
        "resolved_params": {
            "score": params["score"],
            "start_threshold": params["start_threshold"],
            "extend_threshold": params["extend_threshold"],
            "max_gap": params["max_gap"],
            "length_cap": params["length_cap"],
            "min_support_windows": params["min_support_windows"],
            "valley_threshold": params["valley_threshold"],
            "valley_min_windows": params["valley_min_windows"],
        },
        "command": f"bash {experiment_rel.as_posix()}/command.txt",
        "command_specs": command_specs,
    }

    inputs_manifest = {
        "predictions_dir": predictions_rel.as_posix(),
        "benchmark_base": benchmark_rel.as_posix(),
        "split_path": sweep_config["split"]["path"],
        "split_version": split_data["version"],
        "split_part": split_part,
        "genome_ids": genome_ids,
        "source_experiment": tiling["source_experiment"],
    }

    resolved_config = {
        "sweep_id": sweep_id,
        "split_part": split_part,
        "tiling": {
            "trial_id": tiling["trial_id"],
            "window": int(tiling["window"]),
            "stride": int(tiling["stride"]),
            "source_experiment": tiling["source_experiment"],
        },
        "decoder": {
            "decoder_id": decoder["decoder_id"],
            "params": decoder["params"],
            "fixed": sweep_config["decoder_fixed"],
        },
    }

    write_manifest(layout["manifest"], manifest)
    layout["input_manifest"].write_text(
        json.dumps(inputs_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_resolved_config(layout["config"], resolved_config)
    write_command(layout["command"], command_specs["evaluate"]["shell"])

    # Persist the evaluation output in the per-trial experiment dir.
    summary_path = layout["metrics"] / "summary.json"
    summary_path.write_text(json.dumps(eval_output, indent=2), encoding="utf-8")

    return {
        "experiment_id": experiment_id,
        "tiling_trial_id": tiling["trial_id"],
        "decoder_trial_id": decoder["decoder_id"],
        "window": int(tiling["window"]),
        "stride": int(tiling["stride"]),
        "params": {
            **decoder["params"],
            "max_gap": sweep_config["decoder_fixed"].get("max_gap", 0),
            "length_cap": sweep_config["decoder_fixed"].get("length_cap"),
        },
        "summary": eval_output["summary"],
    }


def rank_trials(trial_results: list[dict]) -> list[dict]:
    ranked = sorted(
        trial_results,
        key=lambda trial: (
            -float(trial["summary"]["bench"]["recall"]),
            -float(trial["summary"]["bench"]["precision"]),
            trial["tiling_trial_id"],
            trial["decoder_trial_id"],
        ),
    )
    for index, trial in enumerate(ranked, start=1):
        trial["rank"] = index
    return ranked


def write_pooled_summary(out_dir: Path, ranked_trials: list[dict], header: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **header,
        "trial_count": len(ranked_trials),
        "trials": ranked_trials,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    # CSV columns: rank, tiling, decoder_id, recall, precision, f1, tp_recall,
    # tp_precision, n_gt, n_pred, score_family, start_threshold,
    # extend_threshold, min_support_windows, window, stride.
    fieldnames = [
        "rank",
        "tiling_trial_id",
        "decoder_trial_id",
        "window",
        "stride",
        "recall",
        "precision",
        "f1",
        "tp_recall",
        "tp_precision",
        "n_gt",
        "n_pred",
        "score_family",
        "start_threshold",
        "extend_threshold",
        "min_support_windows",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in ranked_trials:
            bench = trial["summary"]["bench"]
            params = trial["params"]
            writer.writerow(
                {
                    "rank": trial["rank"],
                    "tiling_trial_id": trial["tiling_trial_id"],
                    "decoder_trial_id": trial["decoder_trial_id"],
                    "window": trial["window"],
                    "stride": trial["stride"],
                    "recall": f"{float(bench['recall']):.6f}",
                    "precision": f"{float(bench['precision']):.6f}",
                    "f1": f"{float(bench['f1']):.6f}",
                    "tp_recall": bench["tp_recall"],
                    "tp_precision": bench["tp_precision"],
                    "n_gt": bench["n_gt"],
                    "n_pred": bench["n_pred"],
                    "score_family": params["score_family"],
                    "start_threshold": params["start_threshold"],
                    "extend_threshold": params["extend_threshold"],
                    "min_support_windows": params["min_support_windows"],
                },
            )


def print_topk(ranked_trials: list[dict], k: int = 5) -> None:
    print(f"\nTop {k} by bench lex (recall, precision):")
    print(
        f"{'rank':>4}  {'tiling':<14}  {'decoder':<12}  "
        f"{'recall':>8}  {'precision':>9}  {'f1':>6}  {'score':<8}  "
        f"{'start':>6}  {'extend':>6}  {'msw':>3}"
    )
    for trial in ranked_trials[:k]:
        bench = trial["summary"]["bench"]
        params = trial["params"]
        print(
            f"{trial['rank']:>4}  {trial['tiling_trial_id']:<14}  "
            f"{trial['decoder_trial_id']:<12}  "
            f"{bench['recall']:>8.4f}  {bench['precision']:>9.4f}  "
            f"{bench['f1']:>6.4f}  {params['score_family']:<8}  "
            f"{params['start_threshold']:>6.2f}  {params['extend_threshold']:>6.2f}  "
            f"{params['min_support_windows']:>3}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP_PATH))
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Pooled summary dir. Defaults to <experiments_root>/<sweep_id>.",
    )
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep_config)
    experiments_root = Path(args.experiments_root).resolve()

    sweep_config = load_sweep_config(sweep_path)
    sweep_id = str(sweep_config["sweep_id"])

    split_path = REPO_ROOT / sweep_config["split"]["path"]
    split_part = sweep_config["split"]["part"]
    split_data = load_split(split_path)
    if split_part not in split_data:
        raise SystemExit(f"split part {split_part!r} missing from {split_path}")

    decoder_trials = generate_decoder_trials(sweep_config)
    tilings = sweep_config["tilings"]
    total_trials = len(decoder_trials) * len(tilings)

    git_commit = resolve_git_commit()
    git_dirty = resolve_git_dirty()

    print(
        f"Running {total_trials} trials "
        f"({len(tilings)} tilings x {len(decoder_trials)} decoder configs)",
        flush=True,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (experiments_root / sweep_id)

    results: list[dict] = []
    for ti, tiling in enumerate(tilings, start=1):
        for di, decoder in enumerate(decoder_trials, start=1):
            trial_result = run_single_trial(
                tiling=tiling,
                decoder=decoder,
                sweep_config=sweep_config,
                split_data=split_data,
                split_part=split_part,
                experiments_root=experiments_root,
                sweep_id=sweep_id,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
            results.append(trial_result)
            if (len(results) % 10 == 0) or (len(results) == total_trials):
                print(
                    f"  progress {len(results)}/{total_trials} "
                    f"(tiling {ti}/{len(tilings)}, decoder {di}/{len(decoder_trials)})",
                    flush=True,
                )

    ranked = rank_trials(results)

    header = {
        "sweep_id": sweep_id,
        "ranking_metric": list(sweep_config.get("ranking_metric", ["recall", "precision"])),
        "scorer": sweep_config.get("scorer", "bench"),
        "split": {
            "path": sweep_config["split"]["path"],
            "part": split_part,
            "version": split_data["version"],
            "genome_ids": split_data[split_part],
        },
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "tilings": [
            {
                "trial_id": t["trial_id"],
                "window": int(t["window"]),
                "stride": int(t["stride"]),
                "predictions_dir": t["predictions_dir"],
                "source_experiment": t["source_experiment"],
            }
            for t in tilings
        ],
        "decoder_grid": sweep_config["decoder_grid"],
        "decoder_fixed": sweep_config["decoder_fixed"],
    }
    write_pooled_summary(out_dir, ranked, header)
    print_topk(ranked, k=max(1, args.topk))
    print(f"\nPooled summary: {out_dir}/summary.json")
    print(f"Pooled ranking: {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
