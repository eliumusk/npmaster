#!/usr/bin/env python3
"""Train the per-token BGC head from SHARD-concatenated feature files.

Unlike `train.py` (which expects one `.npy` per window_id), this loader reads
from the shard outputs of `src/evo2_per_token/extract/production_extract.py`:

    <out_dir>/<stem>.partNNNN.npy          # shape (N_rows, T, D), fp16
    <out_dir>/<stem>.partNNNN_windows.csv  # N_rows rows, columns:
                                           #   window_id, contig, start, end, clean_len

Design:
  - Scan every `*.part*_windows.csv` in --features-dir, build a global index
    mapping window_id -> (npy_path, row_idx). Each npy is mmap'd on demand.
  - Intersect with --labels-csv (the canonical windows_metadata_edgepad_filtered.csv)
    to pick up `split`, `label`, `genome_id`, `contig_id`, `actual_length` per window.
  - Build dense per-nt labels then DOWNSAMPLE by K = T_latent / T_raw_actual.
    (Training windows are w=16384, T_latent=2048, so K=8.)
    Rule: y_down[t] = -1 if any mask in block, else 1 if any positive, else 0.
  - DataLoader yields (features, labels) per window.

Outputs identical to train.py (best.pt, train_log.jsonl, config.json).
Model, loss, decoder are unchanged — only the Dataset class is new.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evo2_per_token.labels import (Window, load_bgc_annotations,  # noqa: E402
                              load_windows, per_nt_labels)
from evo2_per_token.model import (build_model, count_params,  # noqa: E402
                             LOSS_REGISTRY, masked_bce_loss)


def scan_shards(features_dir: Path) -> dict[str, tuple[Path, int]]:
    """Return {window_id: (npy_path, row_idx)} for every window in all shards.

    Duplicates (same window_id in multiple shards) are resolved by keeping the
    first occurrence (sorted path order), so runs are deterministic.
    """
    index: dict[str, tuple[Path, int]] = {}
    csvs = sorted(features_dir.glob("*.part*_windows.csv"))
    for csv_path in csvs:
        npy_path = csv_path.with_name(csv_path.name.replace("_windows.csv", ".npy"))
        if not npy_path.exists():
            continue
        with open(csv_path) as fh:
            for row_idx, row in enumerate(csv.DictReader(fh)):
                wid = row["window_id"]
                if wid not in index:
                    index[wid] = (npy_path, row_idx)
    return index


def downsample_labels(y_raw: np.ndarray, K: int) -> np.ndarray:
    """Downsample per-nt labels by factor K to per-latent-token labels.

    y_raw: (N, T_raw) int8 in {-1, 0, 1}
    returns: (N, T_latent) int8 in {-1, 0, 1}, T_latent = T_raw // K

    Rule per block of K positions:
      - If any position == -1 → -1 (mask)
      - Else if any position == 1 → 1
      - Else → 0
    """
    N, T_raw = y_raw.shape
    T_latent = T_raw // K
    y_block = y_raw[:, :T_latent * K].reshape(N, T_latent, K)
    has_mask = (y_block == -1).any(axis=2)
    has_pos = (y_block == 1).any(axis=2)
    out = np.zeros((N, T_latent), dtype=np.int8)
    out[has_pos] = 1
    out[has_mask] = -1
    return out


def preload_features(
    windows: list[Window],
    shard_index: dict[str, tuple[Path, int]],
    d_in: int,
    t_latent: int,
) -> tuple[np.ndarray, list[int]]:
    """Concatenate per-window features into one big fp16 array in RAM.

    Reads shards sequentially (shard at a time, not per window) so VePFS
    sees large sequential IO. Returns:
      feats: (N_kept, t_latent, d_in) fp16
      keep:  indices into `windows` that were actually loaded
    """
    # Group kept (window_index_in_list, shard_path, row_in_shard) by shard.
    by_shard: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for i, w in enumerate(windows):
        entry = shard_index.get(w.window_id)
        if entry is None:
            continue
        by_shard[entry[0]].append((i, entry[1]))

    total = sum(len(v) for v in by_shard.values())
    feats = np.empty((total, t_latent, d_in), dtype=np.float16)
    keep: list[int] = [-1] * total
    write = 0
    n_shards = len(by_shard)
    t0 = time.time()
    for k, (shard_path, items) in enumerate(sorted(by_shard.items())):
        arr = np.load(shard_path)   # full load, seq IO
        if arr.shape[1] < t_latent:
            raise ValueError(f"{shard_path.name}: T={arr.shape[1]} < {t_latent}")
        if arr.shape[2] != d_in:
            raise ValueError(f"{shard_path.name}: D={arr.shape[2]} != {d_in}")
        # Sort by row so we read contiguously within the shard.
        items.sort(key=lambda t: t[1])
        rows = [r for (_, r) in items]
        block = arr[rows, :t_latent, :].astype(np.float16, copy=False)
        feats[write:write + len(rows)] = block
        for (win_i, _), wr in zip(items, range(write, write + len(rows))):
            keep[wr] = win_i
        write += len(rows)
        if (k + 1) % 50 == 0 or (k + 1) == n_shards:
            elapsed = time.time() - t0
            print(f"  preload [{k+1}/{n_shards}] shards, "
                  f"{write}/{total} windows, {elapsed:.0f}s "
                  f"({write / max(1e-6, elapsed):.0f} wins/s)")
    assert all(i >= 0 for i in keep), "some rows unfilled"
    return feats, keep


def preload_features_paired(
    train_windows: list[Window],
    val_windows: list[Window],
    shard_index: dict[str, tuple[Path, int]],
    d_in: int,
    t_latent: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Load each shard exactly once; split rows into train and val buffers."""
    # Per shard: (train_row_items [(win_i, row)], val_row_items [(win_i, row)])
    shards: dict[Path, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}
    for i, w in enumerate(train_windows):
        entry = shard_index.get(w.window_id)
        if entry is None:
            continue
        p, r = entry
        shards.setdefault(p, ([], []))[0].append((i, r))
    for i, w in enumerate(val_windows):
        entry = shard_index.get(w.window_id)
        if entry is None:
            continue
        p, r = entry
        shards.setdefault(p, ([], []))[1].append((i, r))

    n_train = sum(len(t[0]) for t in shards.values())
    n_val = sum(len(t[1]) for t in shards.values())

    train_feats = np.empty((n_train, t_latent, d_in), dtype=np.float16)
    val_feats = np.empty((n_val, t_latent, d_in), dtype=np.float16)
    train_keep = [-1] * n_train
    val_keep = [-1] * n_val
    w_tr = 0; w_va = 0
    n_shards = len(shards)
    t0 = time.time()
    for k, (shard_path, (tr_items, va_items)) in enumerate(sorted(shards.items())):
        arr = np.load(shard_path)
        if arr.shape[1] < t_latent:
            raise ValueError(f"{shard_path.name}: T={arr.shape[1]} < {t_latent}")
        if arr.shape[2] != d_in:
            raise ValueError(f"{shard_path.name}: D={arr.shape[2]} != {d_in}")
        if tr_items:
            tr_items.sort(key=lambda t: t[1])
            rows = [r for (_, r) in tr_items]
            block = arr[rows, :t_latent, :].astype(np.float16, copy=False)
            train_feats[w_tr:w_tr + len(rows)] = block
            for (win_i, _), wr in zip(tr_items, range(w_tr, w_tr + len(rows))):
                train_keep[wr] = win_i
            w_tr += len(rows)
        if va_items:
            va_items.sort(key=lambda t: t[1])
            rows = [r for (_, r) in va_items]
            block = arr[rows, :t_latent, :].astype(np.float16, copy=False)
            val_feats[w_va:w_va + len(rows)] = block
            for (win_i, _), wr in zip(va_items, range(w_va, w_va + len(rows))):
                val_keep[wr] = win_i
            w_va += len(rows)
        del arr
        if (k + 1) % 50 == 0 or (k + 1) == n_shards:
            elapsed = time.time() - t0
            print(f"  preload [{k+1}/{n_shards}] shards  "
                  f"train={w_tr}/{n_train}  val={w_va}/{n_val}  "
                  f"{elapsed:.0f}s")
    return train_feats, val_feats, train_keep, val_keep


class InMemDataset(Dataset):
    """All features preloaded; O(1) lookup. No workers needed."""

    def __init__(
        self,
        feats: np.ndarray,          # (N, T, D) fp16
        latent_labels: np.ndarray,   # (N, T) int8
    ) -> None:
        assert feats.shape[0] == latent_labels.shape[0]
        self.feats = feats
        self.labels = latent_labels

    def __len__(self) -> int:
        return self.feats.shape[0]

    def __getitem__(self, i: int):
        x = self.feats[i].astype(np.float32, copy=False)
        y = self.labels[i].astype(np.int64, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_crop_collate(min_t: int, max_t: int):
    """Return a collate_fn that random-crops each batch to a uniform length."""
    def collate(batch):
        crop_len = random.randint(min_t, max_t)
        xs, ys = [], []
        for x, y in batch:
            T = x.shape[0]
            if T > crop_len:
                start = random.randint(0, T - crop_len)
                x = x[start:start + crop_len]
                y = y[start:start + crop_len]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)
    return collate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, default=None,
                   help="Shard directory (contains <stem>.partNNNN.npy / _windows.csv)")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="windows_metadata_edgepad_filtered.csv (canonical splits/labels)")
    p.add_argument("--bgc-annot", type=Path, default=None)
    p.add_argument("--prepared-dir", type=Path, default=None,
                   help="Dir with pre-built {train,val}_{feats,labels}.npy from prepare_data.py. "
                        "Skips shard scanning + label building; data is mmap'd (zero RAM).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--d-reduced", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--model", choices=["pertoken", "unet"], default="pertoken",
                   help="Head architecture.")
    p.add_argument("--loss", choices=["bce", "dice_bce"], default="bce",
                   help="Loss function.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--pos-weight", type=float, default=3.0,
                   help="Only used when --loss=bce.")
    p.add_argument("--crop-min", type=int, default=0,
                   help="Min crop length in tokens. 0 = no crop (fixed windows).")
    p.add_argument("--crop-max", type=int, default=0,
                   help="Max crop length in tokens. 0 = no crop.")
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["val"])
    p.add_argument("--downsample-k", type=int, default=8,
                   help="Raw bp / latent token (must match extractor --downsample).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsample-train", type=int, default=0,
                   help="If >0, sample this many train windows (smoke test).")
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--max-wall-minutes", type=float, default=120.0,
                   help="Abort new epochs past this wall time (minutes).")
    return p.parse_args()


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    t_global = time.time()

    if args.prepared_dir is not None:
        # ── Fast path: load pre-built arrays from /dev/shm ───────────
        print(f"loading prepared data from {args.prepared_dir}...")
        import json as _json
        meta = _json.loads((args.prepared_dir / "meta.json").read_text())
        t0_load = time.time()
        n_full = meta["n_train"]
        do_sub = args.subsample_train and args.subsample_train < n_full
        if do_sub:
            # mmap → subsample → copy: avoids loading full 297 GiB per process
            rng = np.random.RandomState(args.seed)
            idx = np.sort(rng.permutation(n_full)[:args.subsample_train])
            mm = np.load(args.prepared_dir / "train_feats.npy", mmap_mode="r")
            train_feats = mm[idx].copy(); del mm
            mm = np.load(args.prepared_dir / "train_labels.npy", mmap_mode="r")
            train_labels = mm[idx].copy(); del mm
            print(f"  subsampled train: {args.subsample_train}/{n_full}")
        else:
            train_feats = np.load(args.prepared_dir / "train_feats.npy")
            train_labels = np.load(args.prepared_dir / "train_labels.npy")
        if args.subsample_val and args.subsample_val < meta["n_val"]:
            rng_v = np.random.RandomState(args.seed + 1)
            vidx = np.sort(rng_v.permutation(meta["n_val"])[:args.subsample_val])
            mm = np.load(args.prepared_dir / "val_feats.npy", mmap_mode="r")
            val_feats = mm[vidx].copy(); del mm
            mm = np.load(args.prepared_dir / "val_labels.npy", mmap_mode="r")
            val_labels = mm[vidx].copy(); del mm
            print(f"  subsampled val: {args.subsample_val}/{meta['n_val']}")
        else:
            val_feats = np.load(args.prepared_dir / "val_feats.npy")
            val_labels = np.load(args.prepared_dir / "val_labels.npy")
        args.d_reduced = meta["d_in"]
        print(f"  train: {train_feats.shape}  val: {val_feats.shape}  "
              f"pos_rate={meta['pos_rate']:.4f}  "
              f"loaded in {time.time()-t0_load:.0f}s  "
              f"({train_feats.nbytes/1024**3:.1f} GiB)")
    else:
        # ── Original path: scan shards + build labels ────────────────
        if args.features_dir is None or args.labels_csv is None or args.bgc_annot is None:
            raise RuntimeError("Need --features-dir/--labels-csv/--bgc-annot or --prepared-dir")

        print("scanning shards...")
        shard_index = scan_shards(args.features_dir)
        print(f"  shard_index: {len(shard_index)} windows across "
              f"{len(set(v[0] for v in shard_index.values()))} npy files")

        print("loading canonical labels csv...")
        all_windows = load_windows(args.labels_csv)
        window_by_id = {w.window_id: w for w in all_windows}
        bgc_map = load_bgc_annotations(args.bgc_annot)

        train_w: list[Window] = []
        val_w: list[Window] = []
        for wid in shard_index:
            w = window_by_id.get(wid)
            if w is None:
                continue
            if w.split in args.train_splits:
                train_w.append(w)
            elif w.split in args.val_splits:
                val_w.append(w)

        if args.subsample_train and args.subsample_train < len(train_w):
            rnd = random.Random(args.seed); rnd.shuffle(train_w)
            train_w = train_w[:args.subsample_train]
        if args.subsample_val and args.subsample_val < len(val_w):
            rnd = random.Random(args.seed + 1); rnd.shuffle(val_w)
            val_w = val_w[:args.subsample_val]
        print(f"train windows: {len(train_w)}  val windows: {len(val_w)}")
        if not train_w or not val_w:
            raise RuntimeError("Empty train or val set after filtering.")

        t_raw = int(args.downsample_k * 2048)
        from dataclasses import replace
        def _fix(ws: list[Window]) -> list[Window]:
            out = []
            for w in ws:
                if (w.end - w.start) != t_raw:
                    out.append(replace(w, end=w.start + t_raw))
                else:
                    out.append(w)
            return out
        train_w = _fix(train_w)
        val_w = _fix(val_w)

        print("building raw dense labels...")
        y_train_raw = per_nt_labels(train_w, bgc_map)
        y_val_raw = per_nt_labels(val_w, bgc_map)
        print("  train raw label shape:", y_train_raw.shape,
              "  val raw label shape:", y_val_raw.shape)

        print(f"downsampling labels by K={args.downsample_k}...")
        y_train = downsample_labels(y_train_raw, args.downsample_k)
        y_val = downsample_labels(y_val_raw, args.downsample_k)
        print("  train latent label shape:", y_train.shape,
              "  pos rate:",
              f"{(y_train == 1).sum() / max(1, (y_train >= 0).sum()):.4f}")
        print("  val latent label shape:", y_val.shape,
              "  pos rate:",
              f"{(y_val == 1).sum() / max(1, (y_val >= 0).sum()):.4f}")

        print("preloading train+val features into RAM (shared shard pass)...")
        train_feats, val_feats, train_keep, val_keep = preload_features_paired(
            train_w, val_w, shard_index, args.d_reduced,
            t_latent=y_train.shape[1],
        )
        train_labels = y_train[train_keep]
        val_labels = y_val[val_keep]
        print(f"  train feats: {train_feats.shape} "
              f"({train_feats.nbytes / 1024**3:.1f} GiB)")
        print(f"  val feats: {val_feats.shape} "
              f"({val_feats.nbytes / 1024**3:.1f} GiB)")

    train_ds = InMemDataset(train_feats, train_labels)
    val_ds = InMemDataset(val_feats, val_labels)
    print(f"dataset sizes: train={len(train_ds)} val={len(val_ds)}")

    use_crop = args.crop_min > 0 and args.crop_max > 0
    collate_fn = make_crop_collate(args.crop_min, args.crop_max) if use_crop else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    device = torch.device("cuda")
    model_kwargs: dict = {"d_in": args.d_reduced, "hidden": args.hidden}
    if args.model == "pertoken":
        model_kwargs["dilations"] = tuple(args.dilations)
    model = build_model(args.model, **model_kwargs).to(device)
    print(f"model={args.model}  loss={args.loss}  params={count_params(model)}")

    loss_fn = LOSS_REGISTRY[args.loss]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    log_path = args.out_dir / "train_log.jsonl"
    # Truncate any stale log.
    log_path.write_text("")

    with open(args.out_dir / "config.json", "w") as fh:
        json.dump(
            {**vars(args), "model_type": args.model,
             "param_count": count_params(model),
             "n_train_windows": len(train_ds),
             "n_val_windows": len(val_ds),
             "t_latent": int(train_labels.shape[1]),
             "downsample_k": args.downsample_k,
             "label_rule": ("y_down[t] = -1 if any y_raw[t*K:(t+1)*K] == -1; "
                            "else 1 if any == 1; else 0")},
            fh, indent=2, default=str,
        )

    for epoch in range(args.epochs):
        wall_min = (time.time() - t_global) / 60.0
        if wall_min > args.max_wall_minutes:
            print(f"[wall {wall_min:.1f}min > budget {args.max_wall_minutes}]: stopping.")
            break

        model.train()
        t0 = time.time()
        tr_loss = 0.0; tr_n = 0
        n_steps = len(train_loader)
        log_every = max(1, n_steps // 5)
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if args.loss == "bce":
                loss = loss_fn(logits, y, pos_weight=args.pos_weight)
            else:
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.shape[0]; tr_n += x.shape[0]
            if step % log_every == 0 or step == n_steps - 1:
                print(f"  ep{epoch} step {step}/{n_steps} loss={loss.item():.4f} "
                      f"wall={((time.time()-t_global)/60):.0f}m", flush=True)

        model.eval()
        va_loss = 0.0; va_n = 0
        va_pos_preds = 0; va_pos_true = 0; va_tp = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                if args.loss == "bce":
                    vl = masked_bce_loss(logits, y, pos_weight=args.pos_weight)
                else:
                    vl = loss_fn(logits, y)
                va_loss += vl.item() * x.shape[0]
                va_n += x.shape[0]
                preds = (torch.sigmoid(logits) > 0.5).int()
                mask = (y >= 0)
                tp = ((preds == 1) & (y == 1) & mask).sum().item()
                va_tp += tp
                va_pos_preds += ((preds == 1) & mask).sum().item()
                va_pos_true += ((y == 1) & mask).sum().item()

        tr_loss /= max(1, tr_n); va_loss /= max(1, va_n)
        prec = va_tp / max(1, va_pos_preds); rec = va_tp / max(1, va_pos_true)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        elapsed = time.time() - t0
        rec_line = {
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "val_token_precision": prec, "val_token_recall": rec,
            "val_token_f1": f1, "elapsed_s": elapsed,
            "wall_min_total": (time.time() - t_global) / 60.0,
        }
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec_line) + "\n")
        print(rec_line)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args),
                 "model_type": args.model,
                 "val_loss": va_loss, "epoch": epoch},
                args.out_dir / "best.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs.")
                break

    print(f"done. best_epoch={best_epoch} best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
