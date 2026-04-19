#!/usr/bin/env python3
"""Aggregate Stage 5 trial outputs into a single summary table.

Reads every ``experiments/<sweep>-<trial>/manifest.json`` under an
experiments root, joins with the corresponding ``train_log_<tag>.csv``
to extract the peak internal-val metric, and emits:

  * ``<out>.csv`` — one row per trial with key fields
  * ``<out>.md``  — the same information as a ranked Markdown table

Peak metric used for Stage 5 selection = ``val_region_f1`` (train.py
writes this every 5 epochs). Falls back to last-non-null observation
if no completed 5-epoch interval yet (handy for mid-run peeks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_train_log(manifest: dict, trial_dir: Path) -> Path | None:
    """Manifest may record an absolute local path from planning time; the
    real log lives under the trial directory. Prefer resolving by shape."""
    ckpt_dir = trial_dir / "artifacts" / "checkpoints"
    tag = manifest.get("tag")
    if tag:
        candidate = ckpt_dir / f"train_log_{tag}.csv"
        if candidate.is_file():
            return candidate
    # Fallback: any train_log_*.csv in the dir.
    candidates = list(ckpt_dir.glob("train_log_*.csv")) if ckpt_dir.is_dir() else []
    if candidates:
        return candidates[0]
    return None


def _peek_best(df: pd.DataFrame) -> dict:
    metric = "val_region_f1"
    if metric not in df.columns or df[metric].notna().sum() == 0:
        return {
            "peak_val_f1": None,
            "peak_val_auroc": float(df["val_bgc_auroc"].max())
            if "val_bgc_auroc" in df.columns
            else None,
            "peak_epoch": None,
        }
    idx = df[metric].idxmax()
    return {
        "peak_val_f1": float(df.loc[idx, metric]),
        "peak_val_auroc": float(df.loc[idx, "val_bgc_auroc"])
        if "val_bgc_auroc" in df.columns
        else None,
        "peak_epoch": int(df.loc[idx, "epoch"])
        if "epoch" in df.columns
        else None,
    }


def summarize_trial(manifest_path: Path) -> dict | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    trial_dir = manifest_path.parent

    summary = {
        "experiment_id": manifest.get("experiment_id"),
        "sweep_id": manifest.get("sweep_id"),
        "trial_id": manifest.get("trial_id"),
        "variant": manifest.get("variant"),
        "seed": manifest.get("seed"),
        "epochs_planned": manifest.get("epochs"),
    }

    log_path = _resolve_train_log(manifest, trial_dir)
    if log_path is None or not log_path.is_file():
        summary.update(
            {
                "epochs_completed": 0,
                "peak_val_f1": None,
                "peak_val_auroc": None,
                "peak_epoch": None,
                "final_L_bce": None,
                "total_sec": None,
                "status": "no_train_log",
            }
        )
        return summary

    try:
        df = pd.read_csv(log_path)
    except Exception:
        summary["status"] = "unreadable_log"
        return summary

    summary.update(_peek_best(df))
    summary["epochs_completed"] = int(len(df))
    summary["final_L_bce"] = (
        float(df["L_bce"].iloc[-1]) if "L_bce" in df.columns and len(df) else None
    )
    summary["total_sec"] = (
        float(df["sec"].sum()) if "sec" in df.columns else None
    )
    summary["status"] = "running" if len(df) < (
        manifest.get("epochs") or 0
    ) else "done"
    return summary


def _collect_manifests(root: Path, sweep_prefix: str) -> list[Path]:
    manifests = []
    for trial_dir in sorted(root.glob(f"{sweep_prefix}-*")):
        if not trial_dir.is_dir():
            continue
        manifest_path = trial_dir / "manifest.json"
        if manifest_path.is_file():
            manifests.append(manifest_path)
    return manifests


def _render_md(rows: list[dict]) -> str:
    header = (
        "| trial_id | variant | seed | epochs | peak val F1 | "
        "peak val AUROC | peak epoch | final L_bce | status |"
    )
    sep = "|" + "---|" * 9
    lines = [header, sep]
    # Rank by peak_val_f1 desc, None last
    sortable = sorted(
        rows,
        key=lambda r: (
            -(r["peak_val_f1"] if r["peak_val_f1"] is not None else -1.0),
            r["trial_id"] or "",
        ),
    )
    for r in sortable:
        lines.append(
            "| {trial_id} | {variant} | {seed} | {ec}/{ep} | {pf} | "
            "{pa} | {pe} | {flb} | {st} |".format(
                trial_id=r.get("trial_id") or "",
                variant=r.get("variant") or "",
                seed=r.get("seed") if r.get("seed") is not None else "",
                ec=r.get("epochs_completed"),
                ep=r.get("epochs_planned"),
                pf=f"{r['peak_val_f1']:.4f}"
                if r.get("peak_val_f1") is not None
                else "—",
                pa=f"{r['peak_val_auroc']:.4f}"
                if r.get("peak_val_auroc") is not None
                else "—",
                pe=r.get("peak_epoch") if r.get("peak_epoch") is not None else "—",
                flb=f"{r['final_L_bce']:.4f}"
                if r.get("final_L_bce") is not None
                else "—",
                st=r.get("status") or "",
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=REPO_ROOT / "experiments",
    )
    parser.add_argument(
        "--sweep-prefix",
        default="stage5_no9_data_v1",
        help="Prefix match for trial dirs; default matches the Stage 5 sweep.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "stage5_no9_data_v1-summary.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "experiments" / "stage5_no9_data_v1-summary.md",
    )
    args = parser.parse_args(argv)

    manifests = _collect_manifests(args.experiments_root, args.sweep_prefix)
    if not manifests:
        print(
            f"No manifests found under {args.experiments_root} "
            f"with prefix {args.sweep_prefix!r}."
        )
        return 1

    rows = []
    for manifest_path in manifests:
        summary = summarize_trial(manifest_path)
        if summary is not None:
            rows.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    args.out_md.write_text(_render_md(rows), encoding="utf-8")
    print(f"wrote {args.out}\nwrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
