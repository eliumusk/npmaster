#!/usr/bin/env python3
"""Pick Stage 5 winners from summary.csv and write a selection freeze.

Selection protocol (per docs/stage5-no-nine-genome-tuning.md §1.2):

  1. Group trial rows by ``variant``.
  2. Within each variant, average ``peak_val_f1`` across completed seeds
     (NaNs propagate — variants without any completed seed are dropped).
  3. Rank variants by the 3-seed mean descending, ties broken by mean
     AUROC.
  4. Top ``--top`` variants are written to a freeze file
     (``selection_decision.md``) recording the variant id, 3-seed mean,
     min/max across seeds, and the best checkpoint for each seed.

The freeze file is a commitment artifact: once written, 9-genome
evaluation runs ONLY on the listed checkpoints. Rerunning this script
after 9-genome results arrive is a protocol violation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def pick_winners(
    summary_csv: Path, top: int = 3, min_epochs: int = 30
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df[df["epochs_completed"].fillna(0) >= min_epochs].copy()
    if df.empty:
        return df

    agg = (
        df.groupby("variant", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_val_f1=("peak_val_f1", "mean"),
            min_val_f1=("peak_val_f1", "min"),
            max_val_f1=("peak_val_f1", "max"),
            mean_val_auroc=("peak_val_auroc", "mean"),
            mean_epochs=("epochs_completed", "mean"),
        )
        .dropna(subset=["mean_val_f1"])
    )

    agg = agg.sort_values(
        ["mean_val_f1", "mean_val_auroc"], ascending=[False, False]
    )
    return agg.head(top)


def _variant_checkpoints(summary_df: pd.DataFrame, variant: str) -> list[dict]:
    sub = summary_df[summary_df["variant"] == variant].sort_values("seed")
    rows = []
    for _, r in sub.iterrows():
        trial_id = r["trial_id"]
        tag = f"{variant}_s{int(r['seed'])}"
        ckpt = (
            REPO_ROOT
            / "experiments"
            / f"{r['sweep_id']}-{trial_id}"
            / "artifacts"
            / "checkpoints"
            / f"best_{tag}.pt"
        )
        rows.append(
            {
                "seed": int(r["seed"]),
                "trial_id": trial_id,
                "tag": tag,
                "peak_val_f1": r.get("peak_val_f1"),
                "best_checkpoint": ckpt.as_posix(),
            }
        )
    return rows


def render_freeze(
    winners: pd.DataFrame,
    full_summary: pd.DataFrame,
    selection_note: str,
) -> str:
    lines = [
        "# Stage 5 Selection Decision",
        "",
        "**Protocol:** winners picked on `peak val_region_f1` averaged",
        "across seeds of the internal val split (169 genomes, antiSMASH",
        "labels under the variant's own tight threshold). 9-genome data",
        "was NOT inspected during selection.",
        "",
        f"**Selection note:** {selection_note}",
        "",
        "## Frozen winners",
        "",
        "| rank | variant | n_seeds | mean val F1 | min | max | mean AUROC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, r) in enumerate(winners.iterrows(), start=1):
        lines.append(
            f"| {rank} | {r['variant']} | {int(r['n_seeds'])} | "
            f"{r['mean_val_f1']:.4f} | {r['min_val_f1']:.4f} | "
            f"{r['max_val_f1']:.4f} | {r['mean_val_auroc']:.4f} |"
        )

    lines.append("")
    lines.append("## Frozen checkpoints per winner")
    lines.append("")
    for _, r in winners.iterrows():
        lines.append(f"### {r['variant']}")
        lines.append("")
        lines.append(
            "| seed | trial_id | peak val F1 | checkpoint |"
        )
        lines.append("|---:|---|---:|---|")
        for ck in _variant_checkpoints(full_summary, r["variant"]):
            f1_text = (
                f"{ck['peak_val_f1']:.4f}"
                if ck["peak_val_f1"] is not None
                and not pd.isna(ck["peak_val_f1"])
                else "—"
            )
            lines.append(
                f"| {ck['seed']} | {ck['trial_id']} | {f1_text} | "
                f"`{ck['best_checkpoint']}` |"
            )
        lines.append("")

    lines.append("## Rules for 9-genome evaluation")
    lines.append("")
    lines.append(
        "1. Run inference + eval on ONLY the checkpoints listed above."
    )
    lines.append(
        "2. Use the frozen `val_frozen_fused` operating point; do NOT"
    )
    lines.append("   search a new decoder.")
    lines.append(
        "3. After final 9-genome results are recorded, this file is"
    )
    lines.append("   locked. Do not re-run winner selection.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "stage5_no9_data_v1-summary.csv",
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=30,
        help="Only consider seeds that completed at least this many epochs.",
    )
    parser.add_argument(
        "--freeze-out",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "stage5_no9_data_v1-selection_decision.md",
    )
    parser.add_argument(
        "--selection-note",
        default="Rungs 0/1/2 sweep; no 9-genome inspection during selection.",
    )
    args = parser.parse_args(argv)

    full_summary = pd.read_csv(args.summary_csv)
    winners = pick_winners(args.summary_csv, args.top, args.min_epochs)
    if winners.empty:
        print(
            f"No winners — no trials completed ≥{args.min_epochs} epochs.",
        )
        return 1

    freeze_text = render_freeze(winners, full_summary, args.selection_note)
    args.freeze_out.parent.mkdir(parents=True, exist_ok=True)
    args.freeze_out.write_text(freeze_text, encoding="utf-8")
    print(f"wrote {args.freeze_out}")
    print(winners.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
