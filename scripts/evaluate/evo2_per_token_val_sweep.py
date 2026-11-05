#!/usr/bin/env python3
"""Internal validation region-level sweep for per-token U-Net checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.nine_genome import eval_regions  # noqa: E402
from evo2_per_token.decode import assemble_per_contig_track, decode_per_contig  # noqa: E402
from evo2_per_token.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep per-token decode params on internal val regions.")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probs-dir", type=Path,
                        help="Optional existing <stem>.probs.npz directory; skips checkpoint scoring.")
    parser.add_argument("--coords-csv", type=Path,
                        help="Coordinate CSV to use with --probs-dir. Defaults to {split}_coords.csv for prepared data.")
    parser.add_argument("--bgc-annot", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--min-lens", default="2000,5000,8000,10000")
    parser.add_argument("--max-gaps", default="0,500,1000")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--upsample-k", type=int, default=None)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def composite_contig(genome: str, contig: str) -> str:
    return f"{genome}|{contig}"


def read_split_coords(coords_csv: Path) -> tuple[list[str], dict[str, tuple[str, int, int]]]:
    window_ids: list[str] = []
    coords: dict[str, tuple[str, int, int]] = {}
    with open(coords_csv) as fh:
        for row in csv.DictReader(fh):
            genome = row.get("genome") or row.get("genome_id")
            contig = row.get("contig") or row.get("contig_id")
            if genome is None or contig is None:
                raise ValueError(f"{coords_csv} must contain genome/contig columns")
            start = row.get("start") or row.get("window_start")
            end = row.get("end") or row.get("window_end")
            if start is None or end is None:
                raise ValueError(f"{coords_csv} must contain start/end columns")
            window_id = row["window_id"]
            window_ids.append(window_id)
            coords[window_id] = (
                composite_contig(genome, contig),
                int(start),
                int(end),
            )
    return window_ids, coords


def load_val_ground_truth(
    bgc_csv: Path,
    window_coords: dict[str, tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    covered_contigs = {contig for contig, _start, _end in window_coords.values()}
    gt: list[tuple[str, int, int]] = []
    with open(bgc_csv) as fh:
        for row in csv.DictReader(fh):
            contig = composite_contig(row["genome_id"], row["contig_id"])
            if contig in covered_contigs:
                gt.append((contig, int(row["start_bp"]), int(row["end_bp"])))
    return gt


def score_prepared_split(
    prepared_dir: Path,
    split: str,
    checkpoint: Path,
    window_ids: list[str],
    batch: int,
) -> dict[str, np.ndarray]:
    feats = np.load(prepared_dir / f"{split}_feats.npy", mmap_mode="r")
    if feats.shape[0] != len(window_ids):
        raise ValueError(
            f"{split}_coords.csv rows ({len(window_ids)}) do not match {split}_feats.npy rows ({feats.shape[0]})"
        )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    model_type = ckpt.get("model_type", cfg.get("model", "pertoken"))
    model_kwargs: dict = {"d_in": cfg["d_reduced"], "hidden": cfg["hidden"]}
    if model_type == "pertoken":
        model_kwargs["dilations"] = tuple(cfg["dilations"])
    model = build_model(model_type, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval().cuda()

    probs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, feats.shape[0], batch):
            end = min(start + batch, feats.shape[0])
            x = np.asarray(feats[start:end], dtype=np.float32)
            logits = model(torch.from_numpy(x).cuda(non_blocking=True))
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float16)
            for offset, window_id in enumerate(window_ids[start:end]):
                probs[window_id] = p[offset]
    return probs


def load_probs_dir(probs_dir: Path) -> dict[str, np.ndarray]:
    probs: dict[str, np.ndarray] = {}
    for npz_path in sorted(probs_dir.glob("*.probs.npz")):
        npz = np.load(npz_path)
        for window_id in npz.files:
            if window_id in probs:
                raise ValueError(f"duplicate window_id in probs: {window_id}")
            probs[window_id] = npz[window_id]
    if not probs:
        raise ValueError(f"no *.probs.npz files found in {probs_dir}")
    return probs


def upsample_window_probs(
    window_probs: dict[str, np.ndarray],
    upsample_k: int,
) -> dict[str, np.ndarray]:
    if upsample_k == 1:
        return {key: value.astype(np.float32, copy=False) for key, value in window_probs.items()}
    return {
        key: np.repeat(value.astype(np.float32), upsample_k)
        for key, value in window_probs.items()
    }


def evaluate_decode_grid(
    window_probs: dict[str, np.ndarray],
    window_coords: dict[str, tuple[str, int, int]],
    gt_regions: list[tuple[str, int, int]],
    thresholds: list[float],
    min_lens: list[int],
    max_gaps: list[int],
    upsample_k: int = 1,
) -> list[dict]:
    tracks = assemble_per_contig_track(upsample_window_probs(window_probs, upsample_k), window_coords)
    rows: list[dict] = []
    for threshold in thresholds:
        for min_len_bp in min_lens:
            for max_gap_bp in max_gaps:
                pred_regions: list[tuple[str, int, int]] = []
                for contig, (track_sum, track_count) in tracks.items():
                    for start, end, _score in decode_per_contig(
                        track_sum,
                        track_count,
                        threshold=threshold,
                        min_len_bp=min_len_bp,
                        max_gap_bp=max_gap_bp,
                    ):
                        pred_regions.append((contig, start, end))
                bench = eval_regions(gt_regions, pred_regions, mode="bench", min_overlap=0.5)
                strict = eval_regions(gt_regions, pred_regions, mode="strict", min_overlap=0.5)
                rows.append(
                    {
                        "threshold": float(threshold),
                        "min_len_bp": int(min_len_bp),
                        "max_gap_bp": int(max_gap_bp),
                        "n_gt": int(bench["n_gt"]),
                        "n_pred": int(bench["n_pred"]),
                        "bench_precision": float(bench["precision"]),
                        "bench_recall": float(bench["recall"]),
                        "bench_f1": float(bench["f1"]),
                        "dedup_f1": float(bench["fragmentation"]["dedup_f1"]),
                        "dedup_precision": float(bench["fragmentation"]["dedup_precision"]),
                        "strict_f1": float(strict["f1"]),
                    }
                )
    return rows


def choose_best(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty sweep")
    return max(
        rows,
        key=lambda row: (
            row["dedup_f1"],
            row["bench_f1"],
            row["bench_recall"],
            -abs(math.log(max(row["n_pred"], 1) / max(row["n_gt"], 1))),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.probs_dir is None:
        if args.prepared_dir is None or args.checkpoint is None:
            raise SystemExit("prepared scoring requires --prepared-dir and --checkpoint")
        meta = json.loads((args.prepared_dir / "meta.json").read_text())
        upsample_k = int(args.upsample_k or meta.get("downsample_k", 8))
        coords_csv = args.coords_csv or (args.prepared_dir / f"{args.split}_coords.csv")
    else:
        if args.coords_csv is None:
            raise SystemExit("--probs-dir requires --coords-csv")
        upsample_k = int(args.upsample_k or 8)
        coords_csv = args.coords_csv

    window_ids, window_coords = read_split_coords(coords_csv)
    gt_regions = load_val_ground_truth(args.bgc_annot, window_coords)
    if args.probs_dir is None:
        window_probs = score_prepared_split(
            args.prepared_dir,
            args.split,
            args.checkpoint,
            window_ids,
            args.batch,
        )
        checkpoint = str(args.checkpoint)
        prepared_dir = str(args.prepared_dir)
    else:
        window_probs = load_probs_dir(args.probs_dir)
        checkpoint = ""
        prepared_dir = ""
        missing = [window_id for window_id in window_ids if window_id not in window_probs]
        if missing:
            raise SystemExit(f"{len(missing)} coords rows missing probabilities; first={missing[0]}")
    rows = evaluate_decode_grid(
        window_probs,
        window_coords,
        gt_regions,
        thresholds=parse_float_list(args.thresholds),
        min_lens=parse_int_list(args.min_lens),
        max_gaps=parse_int_list(args.max_gaps),
        upsample_k=upsample_k,
    )
    best = choose_best(rows)
    result = {
        "checkpoint": checkpoint,
        "prepared_dir": prepared_dir,
        "probs_dir": str(args.probs_dir) if args.probs_dir else "",
        "coords_csv": str(coords_csv),
        "split": args.split,
        "selection_source": "internal_val",
        "upsample_k": upsample_k,
        "best": best,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"best": best, "out_json": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
