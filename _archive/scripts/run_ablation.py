"""Run all 4 ablation configs × N seeds, aggregate ablation_table.csv."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from utils import load_config
from train import train


ABLATIONS = ["A_baseline", "B_contrastive", "C_full", "D_with_nll"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--configs", nargs="+", default=ABLATIONS)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip_d", action="store_true",
                   help="Skip D_with_nll if NLL features are unavailable")
    args = p.parse_args()

    rows = []
    for cname in args.configs:
        if args.skip_d and cname == "D_with_nll":
            continue
        ab_path = f"configs/ablations/{cname}.yaml"
        for seed in args.seeds:
            cfg = load_config(args.base, ab_path)
            if args.epochs is not None:
                cfg["train"]["n_epochs"] = args.epochs
            # write each ablation into its own out_dir subdir
            cfg["log"]["out_dir"] = str(
                Path(cfg["log"]["out_dir"]).parent / cname)
            tag = f"{cname}_s{seed}"
            res = train(cfg, seed=seed, log_tag=tag)
            res["config"] = cname
            rows.append(res)

    # write aggregate table
    out = Path("results") / "ablation_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["config", "seed", "best_val_mean_auroc",
            "val_mean_auroc", "val_bgc_auroc",
            "test_id_mean_auroc", "test_id_bgc_auroc",
            "test_ood_mean_auroc", "test_ood_bgc_auroc",
            "val_dann_acc", "test_id_dann_acc"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
