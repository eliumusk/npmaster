#!/usr/bin/env python3
"""Generate representative 9-genome case-study PNGs.

The script picks "typical" genomes algorithmically, then plots one
representative GT-centered locus per genome with:

- GT
- antiSMASH
- DeepBGC
- BGC-Prophet
- NP-Master `val_frozen_fused`
- NP-Master `tuned_v2`
- NP-Master window scores (`fused` and `max_type`)

GECCO is intentionally omitted from the default figure because the current
server snapshot does not contain matching 9-genome outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_repo_candidates: list[Path] = []
try:
    _repo_candidates.append(Path(__file__).resolve().parents[2])
except NameError:
    pass
_repo_candidates.append(Path.cwd())
REPO_ROOT = next(
    (
        candidate
        for candidate in _repo_candidates
        if (candidate / "scripts" / "evaluate_nine_genome.py").exists()
    ),
    Path.cwd(),
)
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate_nine_genome as ev  # noqa: E402
from decoding.postprocess import decode_regions_hysteresis  # noqa: E402


TRACK_ORDER = [
    "GT",
    "antiSMASH",
    "DeepBGC",
    "BGC-Prophet",
    "NP-Master formal",
    "NP-Master tuned",
]

TRACK_COLORS = {
    "GT": "#111827",
    "antiSMASH": "#2563EB",
    "DeepBGC": "#D97706",
    "BGC-Prophet": "#059669",
    "NP-Master formal": "#DC2626",
    "NP-Master tuned": "#7C3AED",
}


def parse_antismash_dir(
    results_dir: Path,
    genome_ids: list[str] | None = None,
) -> dict[str, list[tuple[str, int, int]]]:
    predictions: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for gdir in sorted(Path(results_dir).iterdir()):
        if not gdir.is_dir():
            continue
        gid = gdir.name
        if genome_ids is not None and gid not in genome_ids:
            continue
        json_path = gdir / f"{gid}.json"
        if not json_path.exists():
            predictions[gid] = []
            continue
        with json_path.open() as f:
            data = json.load(f)
        for rec in data.get("records", []):
            contig = rec.get("id", "")
            for area in rec.get("areas", []):
                start_1 = area.get("start")
                end_1 = area.get("end")
                if start_1 is None or end_1 is None:
                    continue
                start = int(start_1) - 1
                end = int(end_1)
                if end <= start:
                    continue
                predictions[gid].append((contig, start, end))
    return dict(predictions)


def parse_deepbgc_dir(
    results_dir: Path,
    genome_ids: list[str] | None = None,
) -> dict[str, list[tuple[str, int, int]]]:
    predictions: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for gdir in sorted(Path(results_dir).iterdir()):
        if not gdir.is_dir():
            continue
        gid = gdir.name
        if genome_ids is not None and gid not in genome_ids:
            continue
        tsv = gdir / f"{gid}.bgc.tsv"
        if not tsv.exists():
            alts = list(gdir.glob("*.bgc.tsv"))
            if not alts:
                predictions[gid] = []
                continue
            tsv = alts[0]
        with tsv.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    start = int(row.get("nucl_start", 0)) - 1
                    end = int(row.get("nucl_end", 0))
                except ValueError:
                    continue
                if end <= start:
                    continue
                predictions[gid].append((row.get("sequence_id", ""), start, end))
    return dict(predictions)


def parse_bgc_prophet_file(
    regions_json: Path,
    genome_ids: list[str] | None = None,
) -> dict[str, list[tuple[str, int, int]]]:
    with Path(regions_json).open() as f:
        data = json.load(f)
    predictions: dict[str, list[tuple[str, int, int]]] = {}
    for gid, regions in data.items():
        if genome_ids is not None and gid not in genome_ids:
            continue
        rows: list[tuple[str, int, int]] = []
        for item in regions:
            if len(item) < 3:
                continue
            contig, start, end = item[0], int(item[1]), int(item[2])
            if end <= start:
                continue
            rows.append((contig, start, end))
        predictions[gid] = rows
    return predictions


def load_npmaster_windows(
    csv_path: Path,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            bgc_score = float(row["bgc_score"])
            non_bgc_score = float(row.get("non_bgc_score", 0.0))
            rows.append({
                "contig": row["contig"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "bgc_score": bgc_score,
                "non_bgc_score": non_bgc_score,
                "fused_score": 0.5 * bgc_score + 0.5 * (1.0 - non_bgc_score),
            })
    return rows


def decode_npmaster_regions(
    csv_path: Path,
    operating_point: dict,
) -> list[tuple[str, int, int]]:
    windows = ev.load_windows_from_csv(csv_path, score_family=operating_point.get("score"))
    return decode_regions_hysteresis(
        windows,
        start_threshold=float(operating_point["start_threshold"]),
        extend_threshold=float(operating_point["extend_threshold"]),
        max_gap=int(operating_point["max_gap"]),
        min_support_windows=int(operating_point.get("min_support_windows", 1)),
        length_cap=operating_point.get("length_cap"),
        valley_threshold=operating_point.get("valley_threshold"),
        valley_min_windows=int(operating_point.get("valley_min_windows", 1)),
    )


def median_or_zero(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _robust_scale(values: list[float]) -> float:
    if not values:
        return 1.0
    median = statistics.median(values)
    deviations = [abs(v - median) for v in values]
    mad = statistics.median(deviations)
    return float(mad if mad > 1e-9 else 1.0)


def rank_genomes_by_typicality(
    stats_by_genome: dict[str, dict[str, float]],
) -> list[str]:
    if not stats_by_genome:
        return []
    feature_names = sorted(next(iter(stats_by_genome.values())).keys())
    medians = {
        name: statistics.median([stats[name] for stats in stats_by_genome.values()])
        for name in feature_names
    }
    scales = {
        name: _robust_scale([stats[name] for stats in stats_by_genome.values()])
        for name in feature_names
    }

    def score(gid: str) -> tuple[float, str]:
        stats = stats_by_genome[gid]
        dist = 0.0
        for name in feature_names:
            dist += abs(stats[name] - medians[name]) / scales[name]
        return dist, gid

    ranked = sorted(stats_by_genome, key=lambda gid: score(gid))
    return ranked


def regions_overlap(
    a: tuple[str, int, int],
    b: tuple[str, int, int],
    slack_bp: int = 0,
) -> bool:
    if a[0] != b[0]:
        return False
    return min(a[2], b[2] + slack_bp) > max(a[1], b[1] - slack_bp)


def choose_representative_locus(
    gt_regions: list[tuple[str, int, int]],
    predictions_by_tool: dict[str, list[tuple[str, int, int]]],
) -> tuple[str, int, int]:
    if not gt_regions:
        raise ValueError("Cannot choose a representative locus without GT regions.")
    gt_lengths = [end - start for _, start, end in gt_regions]
    target_len = median_or_zero(gt_lengths)
    best_key = None
    best_region = None
    for region in gt_regions:
        support_tools = 0
        for tool_regions in predictions_by_tool.values():
            if any(regions_overlap(region, pred, slack_bp=10_000) for pred in tool_regions):
                support_tools += 1
        length_distance = abs((region[2] - region[1]) - target_len)
        key = (-support_tools, length_distance, region[1], region[2])
        if best_key is None or key < best_key:
            best_key = key
            best_region = region
    assert best_region is not None
    return best_region


def build_plot_window(
    locus: tuple[str, int, int],
    predictions_by_tool: dict[str, list[tuple[str, int, int]]],
    flank_bp: int = 25_000,
    min_window_bp: int = 120_000,
    neighbor_slack_bp: int = 20_000,
) -> tuple[int, int]:
    contig, start, end = locus
    left = start
    right = end
    for regions in predictions_by_tool.values():
        for pred in regions:
            if pred[0] != contig:
                continue
            if pred[2] < start - neighbor_slack_bp or pred[1] > end + neighbor_slack_bp:
                continue
            left = min(left, pred[1])
            right = max(right, pred[2])
    left = max(0, left - flank_bp)
    right = right + flank_bp
    if right - left < min_window_bp:
        center = (left + right) // 2
        half = min_window_bp // 2
        left = max(0, center - half)
        right = left + min_window_bp
    return left, right


def compute_genome_stats(
    genome_id: str,
    gt_regions: list[tuple[str, int, int]],
    tool_regions: dict[str, list[tuple[str, int, int]]],
) -> dict[str, float]:
    n_gt = max(1, len(gt_regions))

    def bench_f1(regions: list[tuple[str, int, int]]) -> float:
        return float(ev.eval_regions(gt_regions, regions, "bench")["f1"])

    return {
        "gt_regions": float(len(gt_regions)),
        "formal_pred_ratio": len(tool_regions["NP-Master formal"]) / n_gt,
        "formal_f1": bench_f1(tool_regions["NP-Master formal"]),
        "tuned_f1": bench_f1(tool_regions["NP-Master tuned"]),
        "antismash_f1": bench_f1(tool_regions["antiSMASH"]),
        "deepbgc_f1": bench_f1(tool_regions["DeepBGC"]),
        "bgc_prophet_f1": bench_f1(tool_regions["BGC-Prophet"]),
    }


def select_case_studies(
    gt_by_genome: dict[str, list[tuple[str, int, int]]],
    tool_regions_by_genome: dict[str, dict[str, list[tuple[str, int, int]]]],
    n_cases: int,
) -> list[str]:
    stats_by_genome = {
        gid: compute_genome_stats(gid, gt_by_genome.get(gid, []), tool_regions_by_genome[gid])
        for gid in tool_regions_by_genome
        if gt_by_genome.get(gid)
    }
    ranked = rank_genomes_by_typicality(stats_by_genome)
    return ranked[:n_cases]


def plot_case_studies(
    out_png: Path,
    manifest_path: Path,
    selected_genomes: list[str],
    gt_by_genome: dict[str, list[tuple[str, int, int]]],
    tool_regions_by_genome: dict[str, dict[str, list[tuple[str, int, int]]]],
    window_rows_by_genome: dict[str, list[dict[str, float | int | str]]],
    operating_points: dict[str, dict],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_cases = len(selected_genomes)
    fig, axes = plt.subplots(
        nrows=n_cases * 2,
        ncols=1,
        figsize=(15, 3.8 * n_cases),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 2.2] * n_cases},
    )
    if n_cases == 1:
        axes = [axes[0], axes[1]]

    manifest = {"selected_genomes": []}

    for case_idx, gid in enumerate(selected_genomes):
        score_ax = axes[case_idx * 2]
        track_ax = axes[case_idx * 2 + 1]
        gt_regions = gt_by_genome[gid]
        predictions = tool_regions_by_genome[gid]
        locus = choose_representative_locus(gt_regions, predictions)
        x0, x1 = build_plot_window(locus, predictions)
        contig = locus[0]

        locus_predictions = {
            name: [
                reg for reg in regions
                if reg[0] == contig and reg[2] > x0 and reg[1] < x1
            ]
            for name, regions in predictions.items()
        }
        locus_gt = [
            reg for reg in gt_regions
            if reg[0] == contig and reg[2] > x0 and reg[1] < x1
        ]

        rows = [
            row for row in window_rows_by_genome[gid]
            if row["contig"] == contig and int(row["end"]) > x0 and int(row["start"]) < x1
        ]
        xs = [int(row["start"]) for row in rows]
        fused = [float(row["fused_score"]) for row in rows]
        max_type = [float(row["bgc_score"]) for row in rows]

        score_ax.plot(xs, fused, color="#DC2626", linewidth=1.6, label="fused score")
        score_ax.plot(xs, max_type, color="#7C3AED", linewidth=1.2, label="max_type score")
        formal = operating_points["val_frozen_fused"]
        tuned = operating_points["tuned_v2"]
        score_ax.axhline(
            float(formal["start_threshold"]),
            color="#DC2626",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="formal start thr",
        )
        score_ax.axhline(
            float(tuned["start_threshold"]),
            color="#7C3AED",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="tuned start thr",
        )
        score_ax.set_xlim(x0, x1)
        score_ax.set_ylim(0.0, 1.05)
        score_ax.set_ylabel("Score")
        score_ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        score_ax.set_title(
            f"{gid} | {contig} | locus {locus[1]:,}-{locus[2]:,} bp | "
            f"window {x0:,}-{x1:,} bp",
            loc="left",
            fontsize=11,
        )
        if case_idx == 0:
            score_ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)

        y_positions = list(range(len(TRACK_ORDER)))
        for y, track_name in zip(y_positions, TRACK_ORDER):
            regions = locus_gt if track_name == "GT" else locus_predictions.get(track_name, [])
            for _, start, end in regions:
                rect = Rectangle(
                    (max(start, x0), y - 0.3),
                    min(end, x1) - max(start, x0),
                    0.6,
                    facecolor=TRACK_COLORS[track_name],
                    edgecolor="none",
                    alpha=0.92,
                )
                track_ax.add_patch(rect)

        track_ax.set_xlim(x0, x1)
        track_ax.set_ylim(-0.8, len(TRACK_ORDER) - 0.2)
        track_ax.set_yticks(y_positions)
        track_ax.set_yticklabels(TRACK_ORDER)
        track_ax.invert_yaxis()
        track_ax.grid(axis="x", alpha=0.15, linewidth=0.6)
        track_ax.set_xlabel("Genomic position (bp)")
        track_ax.set_ylabel("Track")

        manifest["selected_genomes"].append({
            "genome_id": gid,
            "contig": contig,
            "locus": {"start": locus[1], "end": locus[2]},
            "plot_window": {"start": x0, "end": x1},
        })

    fig.suptitle(
        "9-genome BGC boundary case studies\n"
        "Representative genomes selected by median-profile ranking, not by manual cherry-picking",
        fontsize=14,
        y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--antismash-dir", type=Path, required=True)
    parser.add_argument("--deepbgc-dir", type=Path, required=True)
    parser.add_argument("--bgc-prophet-json", type=Path, required=True)
    parser.add_argument("--npmaster-results-dir", type=Path, required=True)
    parser.add_argument(
        "--operating-points-file",
        type=Path,
        default=REPO_ROOT / "configs" / "operating_points.yaml",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.operating_points_file.open() as f:
        operating_points = yaml.safe_load(f)

    gt_by_genome = ev.load_ground_truth(args.gt)
    antismash = parse_antismash_dir(args.antismash_dir, genome_ids=ev.GENOME_IDS)
    deepbgc = parse_deepbgc_dir(args.deepbgc_dir, genome_ids=ev.GENOME_IDS)
    bgc_prophet = parse_bgc_prophet_file(args.bgc_prophet_json, genome_ids=ev.GENOME_IDS)

    tool_regions_by_genome: dict[str, dict[str, list[tuple[str, int, int]]]] = {}
    window_rows_by_genome: dict[str, list[dict[str, float | int | str]]] = {}

    for gid in ev.GENOME_IDS:
        csv_path = args.npmaster_results_dir / f"{gid}_predictions.csv"
        window_rows_by_genome[gid] = load_npmaster_windows(csv_path)
        tool_regions_by_genome[gid] = {
            "antiSMASH": antismash.get(gid, []),
            "DeepBGC": deepbgc.get(gid, []),
            "BGC-Prophet": bgc_prophet.get(gid, []),
            "NP-Master formal": decode_npmaster_regions(csv_path, operating_points["val_frozen_fused"]),
            "NP-Master tuned": decode_npmaster_regions(csv_path, operating_points["tuned_v2"]),
        }

    selected_genomes = select_case_studies(
        gt_by_genome=gt_by_genome,
        tool_regions_by_genome=tool_regions_by_genome,
        n_cases=args.num_cases,
    )

    plot_case_studies(
        out_png=args.out_png,
        manifest_path=args.out_manifest,
        selected_genomes=selected_genomes,
        gt_by_genome=gt_by_genome,
        tool_regions_by_genome=tool_regions_by_genome,
        window_rows_by_genome=window_rows_by_genome,
        operating_points=operating_points,
    )

    print(json.dumps({"selected_genomes": selected_genomes}, indent=2))


if __name__ == "__main__":
    main()
