#!/usr/bin/env python3
"""Apply a trained per-token head to shard-format per-token features.

Input layout (9g extraction):
  <features_dir>/<stem>.part*.npy                 (N_rows, T_latent, D) fp16
  <features_dir>/<stem>.part*_windows.csv         columns: window_id, contig, start, end, clean_len
  <features_dir>/<stem>.manifest.json             summary (optional, for info)

Output:
  <out_dir>/<stem>.probs.npz   keys = window_id, values = (T_latent,) fp16 sigmoid probs
  <out_dir>/<stem>.coords.csv  columns: window_id, contig, start, end, clean_len

Model is length-agnostic (1D convs): a ckpt trained at T_latent=2048 can score
T_latent=1024 inputs without modification.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evo2_per_token.model import PerTokenHead, build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--coords-dir", type=Path, default=None,
                   help="Fallback dir with <stem>.coords.csv when per-part windows CSVs are absent.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--stems", nargs="*", default=None,
                   help="Optional: restrict to these stems (e.g. genome IDs).")
    return p.parse_args()


def iter_shards(features_dir: Path, stems: list[str] | None):
    """Yield (stem, [(npy_path, csv_path), ...]) grouped by stem."""
    all_csvs = sorted(features_dir.glob("*.part*_windows.csv"))
    by_stem: dict[str, list[tuple[Path, Path]]] = {}
    for csv_path in all_csvs:
        npy_path = csv_path.with_name(csv_path.name.replace("_windows.csv", ".npy"))
        if not npy_path.exists():
            continue
        # Derive stem: strip ".partNNNN_windows.csv"
        name = csv_path.name
        stem = name.rsplit(".part", 1)[0]
        by_stem.setdefault(stem, []).append((npy_path, csv_path))
    for stem in sorted(by_stem):
        if stems and stem not in stems:
            continue
        yield stem, by_stem[stem]


def iter_npy_parts(features_dir: Path, stems: list[str] | None):
    """Yield (stem, [npy_path, ...]) when per-part windows CSVs are unavailable."""
    by_stem: dict[str, list[Path]] = {}
    for npy_path in sorted(features_dir.glob("*.part*.npy")):
        stem = npy_path.name.rsplit(".part", 1)[0]
        by_stem.setdefault(stem, []).append(npy_path)
    for stem in sorted(by_stem):
        if stems and stem not in stems:
            continue
        yield stem, by_stem[stem]


def read_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def score_stem(
    model: torch.nn.Module,
    shards: list[tuple[Path, Path]],
    out_npz: Path,
    out_csv: Path,
    batch: int,
) -> int:
    """Score all windows of one stem; returns count of windows written."""
    probs_by_id: dict[str, np.ndarray] = {}
    coords_rows: list[dict] = []
    device = next(model.parameters()).device

    for (npy_path, csv_path) in shards:
        mm = np.load(npy_path, mmap_mode="r")   # (N, T, D) fp16
        rows = read_csv_rows(csv_path)
        if len(rows) != mm.shape[0]:
            raise RuntimeError(f"length mismatch: {npy_path.name} npy={mm.shape[0]} csv={len(rows)}")
        N, T, D = mm.shape

        for start in range(0, N, batch):
            end = min(N, start + batch)
            x = np.asarray(mm[start:end], dtype=np.float32)  # (b, T, D)
            xt = torch.from_numpy(x).to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(xt)                           # (b, T)
                p = torch.sigmoid(logits).cpu().numpy().astype(np.float16)  # (b, T)
            for bi in range(end - start):
                row = rows[start + bi]
                wid = row["window_id"]
                probs_by_id[wid] = p[bi]
                coords_rows.append(row)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **probs_by_id)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        if coords_rows:
            w = csv.DictWriter(fh, fieldnames=list(coords_rows[0].keys()))
            w.writeheader()
            w.writerows(coords_rows)
    return len(probs_by_id)


def score_stem_from_coords(
    model: torch.nn.Module,
    npy_parts: list[Path],
    coords_csv: Path,
    out_npz: Path,
    out_csv: Path,
    batch: int,
) -> int:
    """Score shards when only a per-stem coords CSV is available.

    The coords CSV is assumed to be in the same row order as the concatenated
    part files. This is how earlier per-token inference wrote coordinates.
    """
    all_rows = read_csv_rows(coords_csv)
    probs_by_id: dict[str, np.ndarray] = {}
    coords_rows: list[dict] = []
    device = next(model.parameters()).device
    offset = 0

    for npy_path in npy_parts:
        mm = np.load(npy_path, mmap_mode="r")
        N, T, D = mm.shape
        rows = all_rows[offset:offset + N]
        if len(rows) != N:
            raise RuntimeError(
                f"length mismatch: {npy_path.name} npy={N} coords_remaining={len(rows)}"
            )
        offset += N

        for start in range(0, N, batch):
            end = min(N, start + batch)
            x = np.asarray(mm[start:end], dtype=np.float32)
            xt = torch.from_numpy(x).to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(xt)
                p = torch.sigmoid(logits).cpu().numpy().astype(np.float16)
            for bi in range(end - start):
                row = rows[start + bi]
                wid = row["window_id"]
                probs_by_id[wid] = p[bi]
                coords_rows.append(row)

    if offset != len(all_rows):
        raise RuntimeError(f"unused coords rows for {coords_csv.name}: {len(all_rows) - offset}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **probs_by_id)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        if coords_rows:
            w = csv.DictWriter(fh, fieldnames=list(coords_rows[0].keys()))
            w.writeheader()
            w.writerows(coords_rows)
    return len(probs_by_id)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    model_type = ckpt.get("model_type", cfg.get("model", "pertoken"))
    model_kwargs: dict = {"d_in": cfg["d_reduced"], "hidden": cfg["hidden"]}
    if model_type == "pertoken":
        model_kwargs["dilations"] = tuple(cfg["dilations"])
    model = build_model(model_type, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval().cuda()
    print(f"loaded {model_type} model from {args.ckpt.name}")

    total = 0
    t0 = time.time()
    part_csv_groups = list(iter_shards(args.features_dir, args.stems))
    if part_csv_groups:
        for stem, shards in part_csv_groups:
            out_npz = args.out_dir / f"{stem}.probs.npz"
            out_csv = args.out_dir / f"{stem}.coords.csv"
            n = score_stem(model, shards, out_npz, out_csv, args.batch)
            total += n
            print(f"[{stem}] scored {n} windows -> {out_npz.name}")
    elif args.coords_dir is not None:
        for stem, npy_parts in iter_npy_parts(args.features_dir, args.stems):
            coords_csv = args.coords_dir / f"{stem}.coords.csv"
            if not coords_csv.exists():
                print(f"WARN: missing coords for {stem}; skipping")
                continue
            out_npz = args.out_dir / f"{stem}.probs.npz"
            out_csv = args.out_dir / f"{stem}.coords.csv"
            n = score_stem_from_coords(model, npy_parts, coords_csv, out_npz, out_csv, args.batch)
            total += n
            print(f"[{stem}] scored {n} windows from coords -> {out_npz.name}")
    else:
        raise SystemExit(
            f"no per-part windows CSVs in {args.features_dir}; pass --coords-dir for fallback"
        )

    print(f"done: {total} windows in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
