#!/usr/bin/env python3
"""Configurable 9-genome inference for NP-Master Route B (defrag_v2-style).

Two paths, transparently chosen per genome:
  (1) **Embedding cache hit** — if `--emb-cache-dir` has `{genome}_embeddings.npy`
      + `{genome}_windows.csv` at the requested geometry, Evo2 forward is
      **skipped entirely**. Only the CNN head runs (seconds per genome).
  (2) **Cache miss / no --emb-cache-dir** — Evo2 forward runs on the FASTA. Uses
      the optimized path from `src/evo2_lr/extract_batch.py`
      (batched + layer-truncated). Optionally writes cache for next run.

Evo2 re-extraction is ~2-3 orders of magnitude more expensive than the CNN
head. Always prefer the cached path — the 2026-04-22 optimization bench
measured full-forward bs=1 at 1.3 windows/s/GPU, vs. the CNN head at
~thousands of windows/s.

Typical use after a cached extract run (e.g. the 16-GPU `evo2_lr.extract_batch`
pipeline):
  python scripts/inference/stride.py \
    --fna-dir        <source FASTAs> \
    --emb-cache-dir  <dir produced by evo2_lr.extract_batch> \
    --ckpt           <defrag_v2-style ckpt> \
    --out-dir        <preds target> \
    --window 8192 --stride 2048 --layer blocks.20.mlp.l3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evo2_lr.extract_batch import (  # noqa: E402
    extract_mean_pool,
    make_windows,
    read_fna,
    truncate_blocks_to_layer,
)


# CNN output class order — verified 2026-04-23 against training meta_arrays.pkl.
# Must match the ContextFullModel's 8 output dims exactly.
DEFAULT_CLASS_NAMES = [
    "Polyketide", "NRP", "Terpene", "RiPP",
    "Saccharide", "Alkaloid", "Other", "non-BGC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fna-dir", default="/data/syh/benchmark/nine_genome/fna")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--src-dir", default=None)
    parser.add_argument("--emb-cache-dir", type=Path, default=None,
                        help="Dir with {genome}_embeddings.npy + {genome}_windows.csv "
                             "from evo2_lr.extract_batch. If present for a genome, "
                             "Evo2 forward is skipped for that genome.")
    parser.add_argument("--save-to-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="On cache miss, write produced embeddings to --emb-cache-dir "
                             "for next run. Only active if --emb-cache-dir is provided.")
    parser.add_argument("--skip-existing-preds", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip a genome whose {genome}_predictions.csv already exists "
                             "in --out-dir. Default on.")
    parser.add_argument("--genome-ids", default=None,
                        help="Optional comma-separated subset of genome IDs (without .fna).")
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--stride", type=int, default=16384)
    parser.add_argument("--layer", default="blocks.20.mlp.l3")
    parser.add_argument("--evo-model", default="evo2_7b")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Evo2 forward batch size (cache-miss path only).")
    parser.add_argument("--truncate-blocks", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop Evo2 blocks after --layer (saves ~34%% of forward; "
                             "bit-identical embeddings). Cache-miss path only.")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--cache-dir", default="/data/public_models/hub",
                        help="HF_HUB_CACHE path (Evo2 weights on disk, not embeddings).")
    return parser.parse_args()


def resolve_src_dir(cli_value: str | None) -> Path:
    if cli_value:
        candidate = Path(cli_value).expanduser().resolve()
        if (candidate / "model" / "context_full_model.py").exists():
            return candidate
        raise FileNotFoundError(f"Invalid --src-dir: {candidate}")
    candidates = []
    try:
        current_file = Path(__file__).resolve()
        candidates.extend([
            current_file.parent.parent.parent / "src",
            current_file.parent.parent / "src",
        ])
    except Exception:
        pass
    cwd = Path.cwd().resolve()
    candidates.extend([cwd / "src", cwd.parent / "src"])
    for candidate in candidates:
        if (candidate / "model" / "context_full_model.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate src/. Pass --src-dir explicitly.")


def cache_paths(cache_dir: Path, genome_id: str) -> tuple[Path, Path]:
    return (cache_dir / f"{genome_id}_embeddings.npy",
            cache_dir / f"{genome_id}_windows.csv")


def load_from_cache(cache_dir: Path, genome_id: str):
    emb_p, win_p = cache_paths(cache_dir, genome_id)
    if not (emb_p.exists() and win_p.exists()):
        return None, None
    emb = np.load(emb_p).astype(np.float32)
    wins = pd.read_csv(win_p)
    if len(wins) != emb.shape[0]:
        raise RuntimeError(
            f"cache mismatch for {genome_id}: emb rows={emb.shape[0]} "
            f"windows_csv rows={len(wins)}"
        )
    return emb, wins


def save_to_cache(cache_dir: Path, genome_id: str, emb: np.ndarray,
                  win_rows: list[tuple[str, int, int]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_p, win_p = cache_paths(cache_dir, genome_id)
    np.save(emb_p, emb.astype(np.float16))  # match evo2_lr.extract_batch default
    with open(win_p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["contig", "start", "end"])
        for contig, start, end in win_rows:
            wr.writerow([contig, start, end])


def load_context_model(ckpt_path: str, src_dir: Path):
    import torch
    sys.path.insert(0, str(src_dir))
    from model.context_full_model import ContextFullModel
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ContextFullModel(
        input_dim=4096, proj_hidden=2048, proj_out=256,
        proj_dropout=0.0, n_labels=8, cnn_dropout=0.0, init_beta=-4.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to("cuda").eval()
    return model, ckpt


def run_context_on_contig(context_model, emb_array_contig: np.ndarray) -> np.ndarray:
    """emb: (L, 4096). Returns probs (L, 8)."""
    import torch
    L = emb_array_contig.shape[0]
    emb_tensor = torch.from_numpy(emb_array_contig).float().unsqueeze(0).to("cuda")
    valid_mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        _, logits = context_model(emb_tensor, valid_mask)
        probs = torch.sigmoid(logits[0]).cpu().numpy()
    return probs


def write_predictions(genome_id: str, emb_all: np.ndarray, wins_df: pd.DataFrame,
                      context_model, out_dir: Path) -> int:
    out_csv = out_dir / f"{genome_id}_predictions.csv"
    rows: list[dict] = []
    # groupby(sort=False) preserves FASTA order as recorded in the windows csv.
    for contig_id, sub in wins_df.groupby("contig", sort=False):
        idx = sub.index.to_numpy()
        emb_contig = emb_all[idx]
        probs = run_context_on_contig(context_model, emb_contig)
        for i, (_, r) in enumerate(sub.iterrows()):
            bgc_score = float(probs[i, :7].max())
            pred_class = int(probs[i].argmax())
            rows.append({
                "contig": contig_id,
                "start": int(r["start"]),
                "end": int(r["end"]),
                "bgc_score": round(bgc_score, 6),
                "non_bgc_score": round(float(probs[i, 7]), 6),
                "pred_class": DEFAULT_CLASS_NAMES[pred_class],
                "probs": json.dumps([round(float(p), 6) for p in probs[i].tolist()]),
            })
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "contig", "start", "end", "bgc_score",
            "non_bgc_score", "pred_class", "probs",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    if args.window <= 0 or args.stride <= 0 or args.stride > args.window:
        raise SystemExit(f"Invalid window/stride: {args.window}/{args.stride}")

    selected = None
    if args.genome_ids:
        selected = {s.strip() for s in args.genome_ids.split(",") if s.strip()}

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fna_paths = sorted(Path(args.fna_dir).glob("*.fna"))
    if selected is not None:
        fna_paths = [p for p in fna_paths if p.stem in selected]

    # Plan per genome: skip-existing / cache-hit / needs-evo2
    skip_existing: list[str] = []
    cached: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    need_evo2: list[Path] = []
    for p in fna_paths:
        g = p.stem
        out_csv = out_dir / f"{g}_predictions.csv"
        if args.skip_existing_preds and out_csv.exists():
            skip_existing.append(g)
            continue
        emb, wins = (None, None)
        if args.emb_cache_dir is not None:
            emb, wins = load_from_cache(args.emb_cache_dir, g)
        if emb is not None:
            cached[g] = (emb, wins)
        else:
            need_evo2.append(p)

    print(f"[plan] total genomes: {len(fna_paths)}", flush=True)
    print(f"[plan]   skip (preds exist): {len(skip_existing)}", flush=True)
    print(f"[plan]   cache hit (skip Evo2): {len(cached)}", flush=True)
    print(f"[plan]   need Evo2 forward: {len(need_evo2)}", flush=True)
    if skip_existing:
        print(f"[plan]   skipped: {','.join(skip_existing[:10])}"
              f"{' ...' if len(skip_existing) > 10 else ''}", flush=True)

    # Load CNN head (always needed unless every genome is skip-existing)
    if not cached and not need_evo2:
        print("[done] nothing to do", flush=True)
        return

    src_dir = resolve_src_dir(args.src_dir)
    context_model, ckpt = load_context_model(args.ckpt, src_dir)
    print(f"[ctxmodel] loaded alpha={context_model.gate.alpha.item():.4f} "
          f"epoch={ckpt.get('epoch', '?')}", flush=True)

    # Load Evo2 only if something misses cache
    evo2_model = None
    if need_evo2:
        print("[evo2] loading...", flush=True)
        t0 = time.time()
        from evo2 import Evo2
        evo2_model = Evo2(args.evo_model)
        print(f"[evo2] loaded in {time.time() - t0:.1f}s", flush=True)
        if args.truncate_blocks:
            info = truncate_blocks_to_layer(evo2_model, args.layer)
            if info.get("truncated"):
                print(f"[evo2] truncated to blocks[0..{info['kept_blocks']-1}] "
                      f"(dropped {info['dropped_blocks']}/{info['total_blocks']})",
                      flush=True)
            else:
                print(f"[evo2] truncate skipped: {info.get('reason')}", flush=True)

    results: dict[str, int] = {}

    # Fast path: cached genomes
    for g, (emb, wins) in cached.items():
        t0 = time.time()
        n = write_predictions(g, emb, wins, context_model, out_dir)
        results[g] = n
        print(f"[cached] {g}: {n} windows → preds in {time.time()-t0:.1f}s", flush=True)

    # Slow path: cache miss, run Evo2
    for fna_path in need_evo2:
        g = fna_path.stem
        print(f"\n[extract] {g}", flush=True)
        t_genome = time.time()
        per_contig_embs = []
        per_contig_wins = []
        for contig_id, seq in read_fna(fna_path):
            windows = make_windows(seq, window=args.window, stride=args.stride)
            if not windows:
                continue
            print(f"  {contig_id}: {len(seq)} bp → {len(windows)} windows", flush=True)
            seqs = [w[2] for w in windows]
            emb_contig = extract_mean_pool(
                evo2_model, seqs, args.layer, batch_size=args.batch_size,
            )
            per_contig_embs.append(emb_contig)
            for start, end, _ in windows:
                per_contig_wins.append((contig_id, start, end))

        if not per_contig_embs:
            print(f"[warn] {g}: no windows, skipped", flush=True)
            continue

        emb_all = np.concatenate(per_contig_embs, axis=0)
        wins_df = pd.DataFrame(per_contig_wins, columns=["contig", "start", "end"])

        if args.emb_cache_dir is not None and args.save_to_cache:
            save_to_cache(args.emb_cache_dir, g, emb_all, per_contig_wins)
            print(f"[cache] wrote {g} to {args.emb_cache_dir}", flush=True)

        n = write_predictions(g, emb_all, wins_df, context_model, out_dir)
        results[g] = n
        print(f"[extract] {g}: {n} windows → preds in {time.time()-t_genome:.1f}s",
              flush=True)

    print(json.dumps({
        "n_windows_per_genome": results,
        "skipped_existing": skip_existing,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
