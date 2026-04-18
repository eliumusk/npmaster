#!/usr/bin/env python3
"""Plot a one-genome 9-genome boundary comparison case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROW_STYLES = [
    ("gt", "Ground Truth", "#5ba84f"),
    ("antismash", "antiSMASH", "#4c86c6"),
    ("deepbgc", "DeepBGC", "#f39a3d"),
    ("bgc_prophet", "BGC-Prophet", "#d6504a"),
    ("npmaster_orig", "NP-Master (orig)", "#8f74bf"),
    ("npmaster_tuned", "NP-Master (tuned)", "#5e3c99"),
]


def format_kb(start: int, end: int) -> str:
    return f"{(end - start) / 1000:.0f} kb"


def draw_regions(ax, y: float, regions: list[list[int]], color: str, x0: int, x1: int) -> None:
    for _, start, end in regions:
        if end <= x0 or start >= x1:
            continue
        clip_start = max(start, x0)
        clip_end = min(end, x1)
        width = clip_end - clip_start
        rect = ax.barh(
            [y],
            [width],
            left=[clip_start],
            height=0.72,
            color=color,
            edgecolor="#555555",
            linewidth=0.9,
            align="center",
        )
        if width >= 22000:
            ax.text(
                clip_start + width / 2,
                y,
                format_kb(start, end),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    data = json.loads(Path(args.input_json).read_text())
    x0, x1 = data["zoom"]

    fig, ax = plt.subplots(figsize=(15.5, 6.8), dpi=170)

    gt_regions = data["gt"]
    for _, start, end in gt_regions:
        if start >= x0:
            ax.axvline(start, color="#77c26a", linestyle=":", linewidth=0.8, alpha=0.55)
        if end <= x1:
            ax.axvline(end, color="#77c26a", linestyle=":", linewidth=0.8, alpha=0.55)

    yticks = []
    ylabels = []
    for idx, (key, label, color) in enumerate(ROW_STYLES):
        y = len(ROW_STYLES) - idx
        yticks.append(y)
        ylabels.append(label)
        draw_regions(ax, y, data[key], color, x0, x1)

    prec_old = data["strict_precision"]["npmaster_orig_default"]
    prec_new = data["strict_precision"]["npmaster_tuned_8192"]
    rec_old = data["strict_recall"]["npmaster_orig_default"]
    rec_new = data["strict_recall"]["npmaster_tuned_8192"]

    title = (
        f"9-genome case: {data['genome_id']} {data['contig']}  "
        f"({x0/1e6:.2f}-{x1/1e6:.2f} Mb)"
    )
    subtitle = (
        "Green dotted lines = GT BGC edges. "
        f"NP-Master strict precision {prec_old:.3f} -> {prec_new:.3f}, "
        f"strict recall {rec_old:.3f} -> {rec_new:.3f}."
    )

    ax.set_title(title + "\n" + subtitle, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(x0, x1)
    ax.set_ylim(0.4, len(ROW_STYLES) + 0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Genomic position (bp)", fontsize=10.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
