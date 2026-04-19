#!/usr/bin/env python3
"""Run a decoder sweep ranked by de-duplicated F1 under safety gates.

This is for the current tight-core question: if NP-Master already has recall
headroom over BGC-Prophet, how much precision can be recovered by stricter
region decoding without rewarding fragmented predictions?
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_nine_genome as eval_nine_genome  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "sweeps" / "stage3b_defrag_v2.yaml"
DEFAULT_SPLIT = REPO_ROOT / "configs" / "splits" / "nine_genome_dev_lockbox_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "stage3b_defrag_v2"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping.")
    return data


def load_split(path: Path, split_part: str) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    values = data.get(split_part)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"Split part {split_part!r} must be a list of genome IDs.")
    return values


def generate_decoder_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config.get("decoder_grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Config requires non-empty decoder_grid.")
    keys = list(grid)
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"decoder_grid[{key!r}] must be a non-empty list.")

    trials: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, combo, strict=True))
        if float(params["extend_threshold"]) > float(params["start_threshold"]):
            continue
        trials.append({"decoder_id": f"decoder-{len(trials) + 1:03d}", "params": params})
    return trials


def resolve_params(decoder_params: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    # decoder_grid keys always win over decoder_fixed (swept values are not silently dropped).
    def _grid_over_fixed(key: str, default: Any = None) -> Any:
        if key in decoder_params:
            return decoder_params[key]
        return fixed.get(key, default)

    return {
        "score": decoder_params["score_family"],
        "start_threshold": float(decoder_params["start_threshold"]),
        "extend_threshold": float(decoder_params["extend_threshold"]),
        "max_gap": int(_grid_over_fixed("max_gap", 0)),
        "length_cap": _grid_over_fixed("length_cap"),
        "min_support_windows": int(_grid_over_fixed("min_support_windows", 1)),
        "valley_threshold": _grid_over_fixed("valley_threshold"),
        "valley_min_windows": int(_grid_over_fixed("valley_min_windows", 1)),
        "stride": None,
        "checkpoint": None,
        "expected_metrics": None,
        "operating_point": None,
    }


# Fragmentation gate threshold: a decoder whose raw bench precision is inflated
# by splitting GT BGCs into fragments will have dedup_P much lower than bench_P.
# Any candidate that fails this gate is a diagnostic upper bound only, never a
# deployable operating point. See docs/evaluation-protocol.md.
DEFAULT_FRAG_GATE = 0.75


def _bench(result: dict[str, Any]) -> dict[str, float | int]:
    return result["summary"]["bench"]


def _frag_gate_ratio(result: dict[str, Any]) -> float:
    bench = _bench(result)
    bench_p = float(bench.get("precision", 0.0))
    dedup_p = float(bench.get("fragmentation", {}).get("dedup_precision", 0.0))
    if bench_p <= 0:
        return 0.0
    return dedup_p / bench_p


def _annotate_gate(result: dict[str, Any] | None, threshold: float) -> dict[str, Any] | None:
    if result is None:
        return None
    # Copy shallowly and attach gate annotation so downstream readers of the
    # summary.json can see at a glance whether a diagnostic-best is deployable.
    annotated = dict(result)
    ratio = _frag_gate_ratio(result)
    annotated["fragmentation_gate"] = {
        "threshold": threshold,
        "dedup_over_bench_p": ratio,
        "gate_pass": bool(ratio >= threshold),
    }
    return annotated


def rank_precision_constrained(
    results: list[dict[str, Any]],
    min_recall: float,
    max_n_pred: int | None = None,
) -> list[dict[str, Any]]:
    eligible = [row for row in results if float(_bench(row)["recall"]) >= min_recall]
    if max_n_pred is not None:
        eligible = [row for row in eligible if int(_bench(row)["n_pred"]) <= max_n_pred]
    return sorted(
        eligible,
        key=lambda row: (
            float(_bench(row)["precision"]),
            float(_bench(row)["f1"]),
            float(_bench(row)["recall"]),
        ),
        reverse=True,
    )


def rank_dedup_constrained(
    results: list[dict[str, Any]],
    min_recall: float,
    max_n_pred: int | None = None,
) -> list[dict[str, Any]]:
    eligible = [row for row in results if float(_bench(row)["recall"]) >= min_recall]
    if max_n_pred is not None:
        eligible = [row for row in eligible if int(_bench(row)["n_pred"]) <= max_n_pred]
    return sorted(
        eligible,
        key=lambda row: (
            float(_bench(row).get("fragmentation", {}).get("dedup_f1", 0.0)),
            float(_bench(row).get("fragmentation", {}).get("dedup_precision", 0.0)),
            -float(_bench(row).get("fragmentation", {}).get("duplicate_precision_tp", 0.0)),
            -float(_bench(row).get("fragmentation", {}).get("max_preds_per_gt", 0.0)),
            float(_bench(row)["f1"]),
        ),
        reverse=True,
    )


def rank_f1(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda row: (
            float(_bench(row)["f1"]),
            float(_bench(row)["precision"]),
            float(_bench(row)["recall"]),
        ),
        reverse=True,
    )


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_sweep(
    *,
    config: dict[str, Any],
    split_path: Path,
    split_part: str,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_base = (REPO_ROOT / config["benchmark"]["base"]).resolve()
    genome_ids = load_split(split_path, split_part)
    decoder_trials = generate_decoder_trials(config)
    fixed = config.get("decoder_fixed") or {}
    prediction_sets = config.get("prediction_sets")
    if not isinstance(prediction_sets, list) or not prediction_sets:
        raise ValueError("Config requires non-empty prediction_sets.")

    rows: list[dict[str, Any]] = []
    for pred_set in prediction_sets:
        pred_set_id = str(pred_set["id"])
        results_dir = (REPO_ROOT / pred_set["results_dir"]).resolve()
        for decoder in decoder_trials:
            params = resolve_params(decoder["params"], fixed)
            result = eval_nine_genome.evaluate_results_dir(
                benchmark_base,
                results_dir,
                genome_ids,
                params,
            )
            bench = result["summary"]["bench"]
            rows.append(
                {
                    "trial_id": f"{pred_set_id}-{decoder['decoder_id']}",
                    "prediction_set_id": pred_set_id,
                    "decoder_id": decoder["decoder_id"],
                    "results_dir": _display_path(results_dir),
                    "decoder_params": decoder["params"],
                    "summary": result["summary"],
                    "pred_length_stats": result.get("pred_length_stats", {}),
                    "bench": bench,
                }
            )

    min_recall = float(config.get("ranking", {}).get("min_recall", 0.3475))
    max_n_pred_raw = config.get("ranking", {}).get("max_n_pred")
    max_n_pred = int(max_n_pred_raw) if max_n_pred_raw is not None else None
    precision_ranked = rank_precision_constrained(
        rows,
        min_recall=min_recall,
        max_n_pred=max_n_pred,
    )
    dedup_ranked = rank_dedup_constrained(
        rows,
        min_recall=min_recall,
        max_n_pred=max_n_pred,
    )
    f1_ranked = rank_f1(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "summary.json", config, rows, precision_ranked, dedup_ranked, f1_ranked, min_recall, max_n_pred)
    write_csv(output_dir / "summary.csv", rows, min_recall, max_n_pred)
    return {
        "n_trials": len(rows),
        "min_recall": min_recall,
        "max_n_pred": max_n_pred,
        "fragmentation_gate_threshold": DEFAULT_FRAG_GATE,
        "best_dedup_constrained": _annotate_gate(dedup_ranked[0] if dedup_ranked else None, DEFAULT_FRAG_GATE),
        "diagnostic_best_precision_constrained": _annotate_gate(precision_ranked[0] if precision_ranked else None, DEFAULT_FRAG_GATE),
        "diagnostic_best_f1": _annotate_gate(f1_ranked[0] if f1_ranked else None, DEFAULT_FRAG_GATE),
    }


def write_summary(
    path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    precision_ranked: list[dict[str, Any]],
    dedup_ranked: list[dict[str, Any]],
    f1_ranked: list[dict[str, Any]],
    min_recall: float,
    max_n_pred: int | None,
) -> None:
    gate = DEFAULT_FRAG_GATE
    payload = {
        "sweep_id": config.get("sweep_id"),
        "min_recall": min_recall,
        "max_n_pred": max_n_pred,
        "n_trials": len(rows),
        "fragmentation_gate_threshold": gate,
        # Primary: gate-aware, fragmentation-safe. Deployable if it passes the recall floor.
        "best_dedup_constrained": _annotate_gate(dedup_ranked[0] if dedup_ranked else None, gate),
        "top_dedup_constrained": [_annotate_gate(row, gate) for row in dedup_ranked[:20]],
        # Diagnostic only: these rank by raw bench metrics and may exploit fragmentation
        # or length_cap short-core tricks. Never promote a diagnostic_best_* to an
        # operating point without manual gate inspection.
        "diagnostic_best_precision_constrained": _annotate_gate(precision_ranked[0] if precision_ranked else None, gate),
        "diagnostic_top_precision_constrained": [_annotate_gate(row, gate) for row in precision_ranked[:20]],
        "diagnostic_best_f1": _annotate_gate(f1_ranked[0] if f1_ranked else None, gate),
        "diagnostic_top_f1": [_annotate_gate(row, gate) for row in f1_ranked[:20]],
        "trials": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], min_recall: float, max_n_pred: int | None) -> None:
    fields = [
        "trial_id",
        "prediction_set_id",
        "decoder_id",
        "dedup_f1",
        "dedup_precision",
        "recall",
        "bench_precision",
        "bench_f1",
        "n_pred",
        "tp_precision",
        "unique_gt_precision_tp",
        "duplicate_precision_tp",
        "max_preds_per_gt",
        "median_pred_len",
        "p75_pred_len",
        "meets_recall_floor",
        "meets_n_pred_cap",
        "score_family",
        "start_threshold",
        "extend_threshold",
        "max_gap",
        "min_support_windows",
        "length_cap",
        "valley_threshold",
        "valley_min_windows",
        "results_dir",
    ]
    def _meets_gates(row: dict[str, Any]) -> bool:
        bench = row["bench"]
        if float(bench["recall"]) < min_recall:
            return False
        if max_n_pred is not None and int(bench["n_pred"]) > max_n_pred:
            return False
        return True

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        # Primary sort: eligible (meets gate) dedup_f1-best first, then ineligible
        # rows also by dedup_f1 for diagnostic traceability. Ensures CSV head
        # matches summary.json::best_dedup_constrained.
        all_ranked = rank_dedup_constrained(rows, min_recall=0.0, max_n_pred=None)
        eligible = [r for r in all_ranked if _meets_gates(r)]
        ineligible = [r for r in all_ranked if not _meets_gates(r)]
        for row in eligible + ineligible:
            bench = row["bench"]
            frag = bench.get("fragmentation", {}) or {}
            params = row["decoder_params"]
            length_stats = row.get("pred_length_stats") or {}
            writer.writerow(
                {
                    "trial_id": row["trial_id"],
                    "prediction_set_id": row["prediction_set_id"],
                    "decoder_id": row["decoder_id"],
                    "dedup_f1": frag.get("dedup_f1"),
                    "dedup_precision": frag.get("dedup_precision"),
                    "recall": bench["recall"],
                    "bench_precision": bench["precision"],
                    "bench_f1": bench["f1"],
                    "n_pred": bench["n_pred"],
                    "tp_precision": bench["tp_precision"],
                    "unique_gt_precision_tp": frag.get("unique_gt_precision_tp"),
                    "duplicate_precision_tp": frag.get("duplicate_precision_tp"),
                    "max_preds_per_gt": frag.get("max_preds_per_gt"),
                    "median_pred_len": length_stats.get("median"),
                    "p75_pred_len": length_stats.get("p75"),
                    "meets_recall_floor": float(bench["recall"]) >= min_recall,
                    "meets_n_pred_cap": max_n_pred is None or int(bench["n_pred"]) <= max_n_pred,
                    "score_family": params["score_family"],
                    "start_threshold": params["start_threshold"],
                    "extend_threshold": params["extend_threshold"],
                    "max_gap": params.get("max_gap"),
                    "min_support_windows": params.get("min_support_windows"),
                    "length_cap": params.get("length_cap"),
                    "valley_threshold": params.get("valley_threshold"),
                    "valley_min_windows": params.get("valley_min_windows"),
                    "results_dir": row["results_dir"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--split-part", default="dev")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = run_sweep(
        config=load_yaml(Path(args.config)),
        split_path=Path(args.split),
        split_part=args.split_part,
        output_dir=Path(args.out_dir),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
