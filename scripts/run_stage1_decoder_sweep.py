#!/usr/bin/env python3
"""Plan and execute Stage 1 decoder sweeps on frozen 9-genome predictions."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_nine_genome as eval_nine_genome  # noqa: E402


DEFAULT_SPLIT = REPO_ROOT / "configs" / "splits" / "nine_genome_dev_lockbox_v1.json"
DEFAULT_SWEEP = REPO_ROOT / "configs" / "sweeps" / "stage1_decoder_v1.yaml"


def load_sweep_config(path: Path) -> dict:
    with path.open() as f:
        config = yaml.safe_load(f) or {}
    decoder_grid = config.get("decoder_grid")
    if not isinstance(decoder_grid, dict) or not decoder_grid:
        raise ValueError(f"Sweep config {path} must define a non-empty decoder_grid mapping.")
    for key, values in decoder_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"decoder_grid[{key!r}] must be a non-empty list.")
    return config


def generate_decoder_trials(sweep_config: dict) -> list[dict]:
    decoder_grid = sweep_config["decoder_grid"]
    keys = list(decoder_grid)
    trials = []
    for index, combo in enumerate(itertools.product(*(decoder_grid[key] for key in keys)), start=1):
        params = dict(zip(keys, combo, strict=True))
        trials.append(
            {
                "trial_id": f"decoder-{index:03d}",
                "params": params,
            }
        )
    return trials


def rank_trials(trial_results: list[dict]) -> list[dict]:
    ranked = sorted(
        trial_results,
        key=lambda trial: (
            -float(trial["summary"]["bench"]["recall"]),
            -float(trial["summary"]["bench"]["precision"]),
            trial["trial_id"],
        ),
    )
    for index, trial in enumerate(ranked, start=1):
        trial["rank"] = index
    return ranked


def write_ranked_summary(out_dir: Path, ranked_trials: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    param_keys = sorted({key for trial in ranked_trials for key in trial.get("params", {})})

    payload = {
        "ranking_metric": ["recall", "precision"],
        "trial_count": len(ranked_trials),
        "trials": ranked_trials,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["rank", "trial_id", "recall", "precision", *param_keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in ranked_trials:
            row = {
                "rank": trial["rank"],
                "trial_id": trial["trial_id"],
                "recall": f'{float(trial["summary"]["bench"]["recall"]):.6f}',
                "precision": f'{float(trial["summary"]["bench"]["precision"]):.6f}',
            }
            for key in param_keys:
                row[key] = trial.get("params", {}).get(key, "")
            writer.writerow(row)


def run_trial(
    trial: dict,
    *,
    base: Path,
    results_dir: Path,
    genome_ids: list[str],
    base_params: dict,
) -> dict:
    params = dict(base_params)
    trial_params = trial["params"]
    params.update(
        {
            "score": trial_params.get("score_family", params.get("score")),
            "start_threshold": trial_params["start_threshold"],
            "extend_threshold": trial_params["extend_threshold"],
            "max_gap": trial_params["max_gap"],
            "min_support_windows": trial_params.get(
                "min_support_windows",
                params.get("min_support_windows", 1),
            ),
            "length_cap": trial_params.get("length_cap", params.get("length_cap")),
            "valley_threshold": trial_params.get(
                "valley_threshold",
                params.get("valley_threshold"),
            ),
            "valley_min_windows": trial_params.get(
                "valley_min_windows",
                params.get("valley_min_windows", 1),
            ),
        }
    )
    return {
        "trial_id": trial["trial_id"],
        "params": trial_params,
        "summary": eval_nine_genome.evaluate_results_dir(base, results_dir, genome_ids, params)[
            "summary"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/syh/benchmark/nine_genome")
    parser.add_argument("--results-dir", default=None, help="Defaults to <base>/results/npmaster")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT))
    parser.add_argument("--split-part", choices=("dev", "lockbox"), required=True)
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP))
    parser.add_argument("--operating-point", default=None)
    parser.add_argument(
        "--operating-points-file",
        default=str(eval_nine_genome.DEFAULT_OPERATING_POINTS),
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    results_dir = Path(args.results_dir) if args.results_dir else base / "results" / "npmaster"
    split_path = Path(args.split_file)
    sweep_path = Path(args.sweep_config)
    out_dir = Path(args.out_dir)

    genome_ids = eval_nine_genome.load_genome_subset(split_path, args.split_part)
    sweep_config = load_sweep_config(sweep_path)
    trials = generate_decoder_trials(sweep_config)

    params = {
        "start_threshold": 0.0,
        "extend_threshold": 0.0,
        "max_gap": 0,
        "length_cap": None,
        "min_support_windows": 1,
        "valley_threshold": None,
        "valley_min_windows": 1,
        "score": None,
        "stride": None,
        "checkpoint": None,
        "expected_metrics": None,
        "operating_point": None,
    }
    if args.operating_point:
        op = eval_nine_genome.load_operating_point(
            args.operating_point,
            Path(args.operating_points_file),
        )
        params.update(
            {
                "start_threshold": float(op["start_threshold"]),
                "extend_threshold": float(op["extend_threshold"]),
                "max_gap": int(op["max_gap"]),
                "length_cap": op.get("length_cap"),
                "min_support_windows": int(op.get("min_support_windows", 1)),
                "valley_threshold": op.get("valley_threshold"),
                "valley_min_windows": int(op.get("valley_min_windows", 1)),
                "score": op.get("score"),
                "stride": op.get("stride"),
                "checkpoint": op.get("checkpoint"),
                "expected_metrics": op.get("metrics"),
                "operating_point": args.operating_point,
            }
        )

    ranked_trials = rank_trials(
        [
            run_trial(
                trial,
                base=base,
                results_dir=results_dir,
                genome_ids=genome_ids,
                base_params=params,
            )
            for trial in trials
        ]
    )
    write_ranked_summary(out_dir, ranked_trials)
    print(json.dumps({"split_part": args.split_part, "trial_count": len(ranked_trials)}, indent=2))


if __name__ == "__main__":
    main()
