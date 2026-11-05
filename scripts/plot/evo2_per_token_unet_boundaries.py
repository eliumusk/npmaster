#!/usr/bin/env python3
"""Plot 9-genome BGC boundary tracks across tools, including U-Net.

This follows the old qualitative comparison format: one figure per genome, one
panel per active contig, and one row per tool (GT, baselines, NP-Master lines).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark import nine_genome as ev  # noqa: E402
from decoding.postprocess import decode_regions_hysteresis  # noqa: E402

Region = tuple[str, int, int]


TRACK_COLORS = [
    "#111827",
    "#059669",
    "#2563EB",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
]


def parse_antismash_dir(results_dir: Path, genome_ids: list[str]) -> dict[str, list[Region]]:
    out: dict[str, list[Region]] = defaultdict(list)
    for genome_id in genome_ids:
        json_path = results_dir / genome_id / f"{genome_id}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        for rec in data.get("records", []):
            contig = rec.get("id", "")
            for area in rec.get("areas", []):
                start_1 = area.get("start")
                end_1 = area.get("end")
                if start_1 is None or end_1 is None:
                    continue
                start = int(start_1) - 1
                end = int(end_1)
                if end > start:
                    out[genome_id].append((contig, start, end))
    return out


def parse_bgc_prophet_file(regions_json: Path, genome_ids: list[str]) -> dict[str, list[Region]]:
    raw = json.loads(regions_json.read_text())
    keep = set(genome_ids)
    out: dict[str, list[Region]] = defaultdict(list)
    for genome_id, regions in raw.items():
        if genome_id not in keep:
            continue
        for item in regions:
            if len(item) < 3:
                continue
            contig, start, end = item[0], int(item[1]), int(item[2])
            if end > start:
                out[genome_id].append((contig, start, end))
    return out


def parse_deepbgc_dir(results_dir: Path, genome_ids: list[str]) -> dict[str, list[Region]]:
    out: dict[str, list[Region]] = defaultdict(list)
    for genome_id in genome_ids:
        gdir = results_dir / genome_id
        if not gdir.exists():
            continue
        candidates = [gdir / f"{genome_id}.bgc.tsv", *gdir.glob("*.bgc.tsv")]
        tsv = next((p for p in candidates if p.exists()), None)
        if tsv is None:
            continue
        with tsv.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    start = int(row.get("nucl_start", 0)) - 1
                    end = int(row.get("nucl_end", 0))
                except ValueError:
                    continue
                if end > start:
                    out[genome_id].append((row.get("sequence_id", ""), start, end))
    return out


def decode_window_preds(
    preds_dir: Path,
    *,
    score_family: str | None,
    start_threshold: float,
    extend_threshold: float,
    min_support_windows: int,
    max_gap: int = 0,
    length_cap: int | None = None,
) -> dict[str, list[Region]]:
    out: dict[str, list[Region]] = {}
    for csv_path in sorted(preds_dir.glob("*_predictions.csv")):
        genome_id = csv_path.name.replace("_predictions.csv", "")
        windows = ev.load_windows_from_csv(csv_path, score_family=score_family)
        regions = decode_regions_hysteresis(
            windows,
            start_threshold=start_threshold,
            extend_threshold=extend_threshold,
            max_gap=max_gap,
            min_support_windows=min_support_windows,
            length_cap=length_cap,
        )
        out[genome_id] = [(contig, int(start), int(end)) for contig, start, end in regions]
    return out


def load_region_csv(pred_csv: Path) -> dict[str, list[Region]]:
    out: dict[str, list[Region]] = defaultdict(list)
    with pred_csv.open() as handle:
        for row in csv.DictReader(handle):
            start = int(row["start"])
            end = int(row["end"])
            if end > start:
                out[row["genome"]].append((row["contig"], start, end))
    return out


def active_contigs(tracks: list[tuple[str, list[Region]]], max_contigs: int | None) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for _, regions in tracks:
        for contig, _, _ in regions:
            counts[contig] += 1
    ranked = sorted(counts, key=lambda contig: (-counts[contig], contig))
    return ranked if max_contigs is None else ranked[:max_contigs]


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
    pad = max(10_000, int((max(ends) - min(starts or [0])) * 0.03))
    return 0, max(ends) + pad


def plot_genome(
    genome_id: str,
    tracks: list[tuple[str, list[Region]]],
    out_png: Path,
    out_json: Path,
    max_contigs: int | None,
) -> None:
    contigs = active_contigs(tracks, max_contigs=max_contigs)
    if not contigs:
        raise ValueError(f"No regions found for {genome_id}")

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
                ax.add_patch(
                    Rectangle(
                        (start, y - 0.34),
                        max(1, end - start),
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
    fig.suptitle(f"BGC boundary calls for {genome_id}", fontsize=15)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "genome_id": genome_id,
        "tracks": [label for label, _ in tracks],
        "contigs": contigs,
        "region_counts": {label: len(regions) for label, regions in tracks},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 9-genome BGC boundary comparisons")
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--antismash-dir", type=Path, default=None)
    parser.add_argument("--bgc-prophet-json", type=Path, default=None)
    parser.add_argument("--deepbgc-dir", type=Path, default=None)
    parser.add_argument("--evo2-lr-preds-dir", type=Path, default=None)
    parser.add_argument("--evo2-lr-label", default="Evo2-LR primary")
    parser.add_argument("--evo2-lr-multiscale-csv", type=Path, default=None)
    parser.add_argument("--evo2-lr-learned-csv", type=Path, default=None)
    parser.add_argument("--unet-regions-csv", type=Path, required=True)
    parser.add_argument("--unet-label", default="E10 U-Net")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--genome-ids", default=None,
                        help="Comma-separated subset; default: all genomes in GT")
    parser.add_argument("--max-contigs", type=int, default=None)
    args = parser.parse_args()

    gt = ev.load_ground_truth(args.gt_json)
    genomes = sorted(gt)
    if args.genome_ids:
        keep = {item.strip() for item in args.genome_ids.split(",") if item.strip()}
        genomes = [genome for genome in genomes if genome in keep]

    antismash = parse_antismash_dir(args.antismash_dir, genomes) if args.antismash_dir else {}
    bgc_prophet = (
        parse_bgc_prophet_file(args.bgc_prophet_json, genomes) if args.bgc_prophet_json else {}
    )
    deepbgc = parse_deepbgc_dir(args.deepbgc_dir, genomes) if args.deepbgc_dir else {}
    evo2_lr = (
        decode_window_preds(
            args.evo2_lr_preds_dir,
            score_family=None,
            start_threshold=0.95,
            extend_threshold=0.80,
            min_support_windows=3,
        )
        if args.evo2_lr_preds_dir else {}
    )
    evo2_lr_multiscale = (
        load_region_csv(args.evo2_lr_multiscale_csv) if args.evo2_lr_multiscale_csv else {}
    )
    evo2_lr_learned = (
        load_region_csv(args.evo2_lr_learned_csv) if args.evo2_lr_learned_csv else {}
    )
    unet = load_region_csv(args.unet_regions_csv)

    for genome in genomes:
        tracks = [
            ("GT", gt.get(genome, [])),
        ]
        if antismash:
            tracks.append(("antiSMASH", antismash.get(genome, [])))
        if bgc_prophet:
            tracks.append(("BGC-Prophet", bgc_prophet.get(genome, [])))
        if deepbgc:
            tracks.append(("DeepBGC", deepbgc.get(genome, [])))
        tracks.append((args.unet_label, unet.get(genome, [])))
        if evo2_lr_multiscale:
            tracks.insert(4, ("Evo2-LR multiscale", evo2_lr_multiscale.get(genome, [])))
        elif evo2_lr:
            tracks.insert(4, (args.evo2_lr_label, evo2_lr.get(genome, [])))
        if evo2_lr_learned:
            tracks.insert(-1, ("Evo2-LR learned decoder", evo2_lr_learned.get(genome, [])))
        out_png = args.out_dir / f"{genome}.png"
        out_json = args.out_dir / f"{genome}.json"
        plot_genome(genome, tracks, out_png, out_json, args.max_contigs)
        print(out_png)


if __name__ == "__main__":
    main()
