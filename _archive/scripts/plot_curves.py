"""Plot training-loss curves from train_log_*.csv."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_one(csv_path: Path, out_png: Path):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(df["epoch"], df["L_cls"], label="L_cls")
    axes[0, 0].set_title("L_cls"); axes[0, 0].set_xlabel("epoch")
    axes[0, 1].plot(df["epoch"], df["L_con"], label="L_con", color="tab:orange")
    axes[0, 1].set_title("L_con"); axes[0, 1].set_xlabel("epoch")
    axes[1, 0].plot(df["epoch"], df["L_dann"], label="L_dann", color="tab:green")
    axes[1, 0].plot(df["epoch"], df["lambda_dann"], label="λ", color="tab:red",
                    linestyle="--")
    axes[1, 0].set_title("L_dann (and λ warmup)"); axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 1].plot(df["epoch"], df["val_mean_auroc"], label="val mean AUROC")
    axes[1, 1].plot(df["epoch"], df["val_bgc_auroc"], label="val BGC AUROC")
    axes[1, 1].plot(df["epoch"], df["val_dann_acc"], label="DANN phylum acc",
                    linestyle="--")
    axes[1, 1].set_title("Validation"); axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    fig.suptitle(csv_path.stem)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default="results", help="scan recursively")
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()
    root = Path(args.logs_dir)
    for csv in root.rglob("train_log_*.csv"):
        out = Path(args.out_dir) / (csv.stem + ".png")
        plot_one(csv, out)


if __name__ == "__main__":
    main()
