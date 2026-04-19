#!/usr/bin/env python3
"""Plot BGC boundary tracks for one genome.

The plot is intentionally simple: one PNG per genome, one row per tool, and
horizontal bars showing each predicted/annotated BGC boundary. It is meant for
visual inspection of boundary behavior, not for score diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

import evaluate_nine_genome as ev  # noqa: E402
from baselines.parse_antismash import parse_antismash_dir  # noqa: E402
from baselines.parse_bgc_prophet import parse_bgc_prophet_file  # noqa: E402
from baselines.parse_deepbgc import parse_deepbgc_dir  # noqa: E402
from postprocess_v2 import decode_regions_hysteresis  # noqa: E402


Region = tuple[str, int, int]


@dataclass(frozen=True)
class NPMasterTrack:
    label: str
    results_dir: Path
    score_family: str | None
    start_threshold: float
    extend_threshold: float
    min_support_windows: int
    length_cap: int | None
    max_gap: int = 0
    valley_threshold: float | None = None
    valley_min_windows: int = 1


@dataclass(frozen=True)
class BaselineTrack:
    label: str
    regions_by_genome: dict[str, list[Region]]


TRACK_COLORS = [
    "#111827",
    "#059669",
    "#2563EB",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
    "#9333EA",
    "#4D7C0F",
]

GT_EDGE_COLOR = "#5DAE49"


def parse_npmaster_track_spec(spec: str) -> NPMasterTrack:
    """Parse `label|results_dir|score|start|extend|min_support|length_cap`.

    Use `none` for length_cap when no cap is desired. `score` accepts
    `fused`, `max_type`, `bgc_score`, or `1_minus_isneg`.
    """
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) != 7:
        raise ValueError(
            "--npmaster-track must be "
            "'label|results_dir|score|start|extend|min_support|length_cap'"
        )
    label, results_dir, score, start, extend, min_support, length_cap = parts
    if score in {"", "none", "None", "null"}:
        score_family = None
    else:
        score_family = score
    if length_cap in {"", "none", "None", "null"}:
        parsed_length_cap = None
    else:
        parsed_length_cap = int(length_cap)
    return NPMasterTrack(
        label=label,
        results_dir=Path(results_dir),
        score_family=score_family,
        start_threshold=float(start),
        extend_threshold=float(extend),
        min_support_windows=int(min_support),
        length_cap=parsed_length_cap,
    )


def _unified_predictions_to_regions(preds: dict[str, list[dict]], genome_id: str) -> list[Region]:
    return [
        (row["contig_id"], int(row["start_bp"]), int(row["end_bp"]))
        for row in preds.get(genome_id, [])
    ]


def load_baseline_tracks(
    *,
    genome_id: str,
    bgc_prophet_json: Path | None,
    antismash_dir: Path | None,
    deepbgc_dir: Path | None,
) -> list[tuple[str, list[Region]]]:
    tracks: list[tuple[str, list[Region]]] = []
    genome_ids = [genome_id]
    if bgc_prophet_json is not None:
        raw = parse_bgc_prophet_file(bgc_prophet_json, genome_ids=genome_ids)
        tracks.append(("BGC-Prophet", _unified_predictions_to_regions(raw, genome_id)))
    if antismash_dir is not None:
        raw = parse_antismash_dir(antismash_dir, genome_ids=genome_ids)
        tracks.append(("antiSMASH", _unified_predictions_to_regions(raw, genome_id)))
    if deepbgc_dir is not None:
        raw = parse_deepbgc_dir(deepbgc_dir, genome_ids=genome_ids)
        tracks.append(("DeepBGC", _unified_predictions_to_regions(raw, genome_id)))
    return tracks


def decode_npmaster_track(genome_id: str, track: NPMasterTrack) -> list[Region]:
    csv_path = track.results_dir / f"{genome_id}_predictions.csv"
    windows = ev.load_windows_from_csv(csv_path, score_family=track.score_family)
    return decode_regions_hysteresis(
        windows,
        start_threshold=track.start_threshold,
        extend_threshold=track.extend_threshold,
        max_gap=track.max_gap,
        min_support_windows=track.min_support_windows,
        length_cap=track.length_cap,
        valley_threshold=track.valley_threshold,
        valley_min_windows=track.valley_min_windows,
    )


def active_contigs(tracks: list[tuple[str, list[Region]]], max_contigs: int | None) -> list[str]:
    by_contig: dict[str, int] = defaultdict(int)
    for _, regions in tracks:
        for contig, _, _ in regions:
            by_contig[contig] += 1
    ranked = sorted(by_contig, key=lambda contig: (-by_contig[contig], contig))
    if max_contigs is not None:
        ranked = ranked[:max_contigs]
    return ranked


def contig_xlim(contig: str, tracks: list[tuple[str, list[Region]]]) -> tuple[int, int]:
    starts: list[int] = []
    ends: list[int] = []
    for _, regions in tracks:
        for region_contig, start, end in regions:
            if region_contig == contig:
                starts.append(start)
                ends.append(end)
    if not ends:
        return 0, 1
    right = max(ends)
    left = 0
    pad = max(10_000, int((right - min(starts or [0])) * 0.03))
    return left, right + pad


def plot_genome_boundaries(
    *,
    genome_id: str,
    gt_path: Path,
    out_png: Path,
    out_json: Path,
    baseline_tracks: list[tuple[str, list[Region]]],
    npmaster_tracks: list[NPMasterTrack],
    max_contigs: int | None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    gt = ev.load_ground_truth(gt_path).get(genome_id, [])
    tracks: list[tuple[str, list[Region]]] = [("GT", gt)]
    tracks.extend(baseline_tracks)
    for np_track in npmaster_tracks:
        tracks.append((np_track.label, decode_npmaster_track(genome_id, np_track)))

    contigs = active_contigs(tracks, max_contigs=max_contigs)
    if not contigs:
        raise ValueError(f"No GT or prediction regions found for {genome_id}")

    n_tracks = len(tracks)
    fig_height = max(3.5, 1.5 + len(contigs) * (0.55 * n_tracks + 0.9))
    fig, axes = plt.subplots(
        nrows=len(contigs),
        ncols=1,
        figsize=(18, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    color_by_track = {
        label: TRACK_COLORS[index % len(TRACK_COLORS)]
        for index, (label, _) in enumerate(tracks)
    }

    for ax, contig in zip(axes[:, 0], contigs, strict=True):
        x0, x1 = contig_xlim(contig, tracks)
        for y, (label, regions) in enumerate(tracks):
            for region_contig, start, end in regions:
                if region_contig != contig:
                    continue
                width = max(1, end - start)
                ax.add_patch(
                    Rectangle(
                        (start, y - 0.34),
                        width,
                        0.68,
                        facecolor=color_by_track[label],
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.92,
                    )
                )
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.8, n_tracks - 0.2)
        ax.set_yticks(list(range(n_tracks)))
        ax.set_yticklabels([label for label, _ in tracks])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.16, linewidth=0.6)
        ax.set_title(contig, loc="left", fontsize=10)
        ax.set_xlabel("Genomic position (bp)")

    fig.suptitle(
        f"BGC boundary calls for {genome_id}",
        fontsize=15,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    region_counts = {label: len(regions) for label, regions in tracks}
    contig_counts = {
        contig: {
            label: sum(1 for region_contig, _, _ in regions if region_contig == contig)
            for label, regions in tracks
        }
        for contig in contigs
    }
    manifest = {
        "genome_id": genome_id,
        "out_png": str(out_png),
        "tracks": [label for label, _ in tracks],
        "contigs": contigs,
        "region_counts": region_counts,
        "contig_region_counts": contig_counts,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _regions_in_window(regions: list[Region], contig: str, start_bp: int, end_bp: int) -> list[Region]:
    return [
        region
        for region in regions
        if region[0] == contig and region[2] > start_bp and region[1] < end_bp
    ]


def _kb_label(start: int, end: int) -> str:
    return f"{round((end - start) / 1000):.0f} kb"


def plot_zoom_boundaries(
    *,
    genome_id: str,
    gt_path: Path,
    out_png: Path,
    out_json: Path,
    baseline_tracks: list[tuple[str, list[Region]]],
    npmaster_tracks: list[NPMasterTrack],
    contig: str,
    start_bp: int,
    end_bp: int,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if end_bp <= start_bp:
        raise ValueError("end_bp must be greater than start_bp")

    gt = ev.load_ground_truth(gt_path).get(genome_id, [])
    tracks: list[tuple[str, list[Region]]] = [("Ground Truth", gt)]
    tracks.extend(baseline_tracks)
    for np_track in npmaster_tracks:
        tracks.append((np_track.label, decode_npmaster_track(genome_id, np_track)))

    window_tracks = [
        (label, _regions_in_window(regions, contig, start_bp, end_bp))
        for label, regions in tracks
    ]
    n_tracks = len(window_tracks)
    fig, ax = plt.subplots(figsize=(16, max(4.0, 1.2 + 0.72 * n_tracks)), constrained_layout=True)
    color_by_track = {
        label: TRACK_COLORS[index % len(TRACK_COLORS)]
        for index, (label, _) in enumerate(window_tracks)
    }
    color_by_track["Ground Truth"] = GT_EDGE_COLOR

    gt_edges: list[int] = []
    for _, gt_start, gt_end in window_tracks[0][1]:
        gt_edges.extend([gt_start, gt_end])
    for edge in gt_edges:
        if start_bp <= edge <= end_bp:
            ax.axvline(
                edge / 1_000_000,
                color=GT_EDGE_COLOR,
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
                zorder=0,
            )

    min_label_width_bp = max(18_000, int((end_bp - start_bp) * 0.035))
    for y, (label, regions) in enumerate(window_tracks):
        ax.axhline(y, color="#9CA3AF", linewidth=0.8, alpha=0.25, zorder=0)
        for _, region_start, region_end in regions:
            clipped_start = max(region_start, start_bp)
            clipped_end = min(region_end, end_bp)
            if clipped_end <= clipped_start:
                continue
            x = clipped_start / 1_000_000
            width = (clipped_end - clipped_start) / 1_000_000
            ax.add_patch(
                Rectangle(
                    (x, y - 0.36),
                    width,
                    0.72,
                    facecolor=color_by_track[label],
                    edgecolor="#374151",
                    linewidth=0.6,
                    alpha=0.92,
                    zorder=2,
                )
            )
            if clipped_end - clipped_start >= min_label_width_bp:
                ax.text(
                    x + width / 2,
                    y,
                    _kb_label(region_start, region_end),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                    zorder=3,
                )

    ax.set_xlim(start_bp / 1_000_000, end_bp / 1_000_000)
    ax.set_ylim(-0.7, n_tracks - 0.3)
    ax.set_yticks(list(range(n_tracks)))
    ax.set_yticklabels([label for label, _ in window_tracks], fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#9CA3AF", alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_xlabel(
        f"Genomic position (Mb) — zoom {start_bp / 1_000_000:.2f}-{end_bp / 1_000_000:.2f} Mb",
        fontsize=10,
    )
    ax.set_title(
        "Boundary disagreement in a BGC-dense region  "
        "(green dotted lines = GT BGC edges)",
        fontsize=12,
        fontweight="bold",
    )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#6B7280")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    region_counts = {label: len(regions) for label, regions in window_tracks}
    manifest = {
        "mode": "zoom",
        "genome_id": genome_id,
        "contig": contig,
        "window": {"start_bp": start_bp, "end_bp": end_bp},
        "out_png": str(out_png),
        "tracks": [label for label, _ in window_tracks],
        "region_counts": region_counts,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["overview", "zoom"], default="overview")
    parser.add_argument("--genome-id", required=True)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--bgc-prophet-json", type=Path, default=None)
    parser.add_argument("--antismash-dir", type=Path, default=None)
    parser.add_argument("--deepbgc-dir", type=Path, default=None)
    parser.add_argument(
        "--npmaster-track",
        action="append",
        default=[],
        help=(
            "Repeatable: label|results_dir|score|start|extend|min_support|length_cap. "
            "Example: 'Best F1|tmp/preds|max_type|0.995|0.99|4|30000'"
        ),
    )
    parser.add_argument("--max-contigs", type=int, default=None)
    parser.add_argument("--contig", default=None)
    parser.add_argument("--start-bp", type=int, default=None)
    parser.add_argument("--end-bp", type=int, default=None)
    parser.add_argument("--out-png", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_json = args.out_json
    if out_json is None:
        out_json = args.out_png.with_suffix(".json")
    baseline_tracks = load_baseline_tracks(
        genome_id=args.genome_id,
        bgc_prophet_json=args.bgc_prophet_json,
        antismash_dir=args.antismash_dir,
        deepbgc_dir=args.deepbgc_dir,
    )
    npmaster_tracks = [parse_npmaster_track_spec(spec) for spec in args.npmaster_track]
    if args.mode == "zoom":
        if args.contig is None or args.start_bp is None or args.end_bp is None:
            raise SystemExit("--mode zoom requires --contig, --start-bp, and --end-bp")
        manifest = plot_zoom_boundaries(
            genome_id=args.genome_id,
            gt_path=args.gt,
            out_png=args.out_png,
            out_json=out_json,
            baseline_tracks=baseline_tracks,
            npmaster_tracks=npmaster_tracks,
            contig=args.contig,
            start_bp=args.start_bp,
            end_bp=args.end_bp,
        )
    else:
        manifest = plot_genome_boundaries(
            genome_id=args.genome_id,
            gt_path=args.gt,
            out_png=args.out_png,
            out_json=out_json,
            baseline_tracks=baseline_tracks,
            npmaster_tracks=npmaster_tracks,
            max_contigs=args.max_contigs,
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
