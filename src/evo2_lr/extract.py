"""FASTA → Evo2 mean-pool embeddings at configured (window, stride).

This is a thin wrapper around the Evo2 Python SDK. It mirrors the
`scripts/inference_stride.py` logic but without the Route B CNN step —
output is raw 4096-d mean-pool embeddings per window.

Supported input: plain .fna and .fna.gz.
"""
from __future__ import annotations

import csv
import gzip
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_fna(path: Path | str) -> list[tuple[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    contigs: list[tuple[str, str]] = []
    cur_id = None
    cur: list[str] = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    contigs.append((cur_id, "".join(cur)))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip().upper())
    if cur_id is not None:
        contigs.append((cur_id, "".join(cur)))
    return contigs


def make_windows(seq: str, window: int, stride: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for start in range(0, len(seq), stride):
        end = min(start + window, len(seq))
        sub = seq[start:end]
        if len(sub) < 100:
            continue
        if len(sub) < window:
            sub = sub + ("N" * (window - len(sub)))
        out.append((start, end, sub))
    return out


def _clean_seq(seq: str) -> str:
    return "".join(c for c in seq if c in "ATCG")


def _extract_mean_pool_batched(
    evo2_model,
    sequences: list[str],
    layer: str,
    *,
    batch_size: int = 1,
) -> np.ndarray:
    """Mean-pool Evo2 embeddings, batching equal token-length windows.

    Windows are usually fixed-length, but tail windows and internal Ns can
    produce shorter cleaned token streams. Grouping by token length avoids
    padding-token assumptions while still batching the common full windows.
    """
    import torch

    batch_size = max(1, int(batch_size))
    out: list[np.ndarray | None] = [None] * len(sequences)
    by_len: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)

    for i, seq in enumerate(sequences):
        clean = _clean_seq(seq)
        if len(clean) < 10:
            out[i] = np.zeros(4096, dtype=np.float32)
            continue
        toks = evo2_model.tokenizer.tokenize(clean)
        by_len[len(toks)].append((i, toks))

    for tok_len, items in sorted(by_len.items(), key=lambda kv: kv[0], reverse=True):
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            ids = torch.tensor([t for _, t in chunk], dtype=torch.int).to("cuda")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, emb_dict = evo2_model(
                    ids, return_embeddings=True, layer_names=[layer]
                )
            h = emb_dict[layer].float()
            if h.ndim != 3:
                raise RuntimeError(f"unexpected embedding tensor shape: {tuple(h.shape)}")
            pooled = h.mean(dim=1).cpu().numpy().astype(np.float32)
            for (orig_i, _), emb in zip(chunk, pooled):
                out[orig_i] = emb

    missing = sum(v is None for v in out)
    if missing:
        raise RuntimeError(f"internal error: {missing} embeddings were not filled")
    return np.asarray(out, dtype=np.float32)


def extract_embeddings(
    fna_path: Path | str,
    out_emb_npy: Path | str,
    out_win_csv: Path | str,
    *,
    window: int = 8192,
    stride: int = 2048,
    layer: str = "blocks.20.mlp.l3",
    evo_model: str = "evo2_7b",
    cache_dir: str = "/data/public_models/hub",
    dtype: str = "fp16",
    min_contig_len: int = 0,
    batch_size: int = 1,
) -> dict:
    """Extract Evo2 mean-pool embeddings for a single FASTA.

    Writes `{out_emb_npy}` of shape (N_windows, 4096) and `{out_win_csv}`
    with columns contig,start,end. Returns a small manifest dict.
    """
    import os
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from evo2 import Evo2

    print(f"[extract] loading Evo2 ({evo_model})...", flush=True)
    t0 = time.time()
    evo2_model = Evo2(evo_model)
    print(f"[extract] loaded in {time.time() - t0:.1f}s", flush=True)

    dtype_np = np.float16 if dtype == "fp16" else np.float32

    all_emb: list[np.ndarray] = []
    rows: list[tuple[str, int, int]] = []
    n_contigs = 0
    n_contigs_kept = 0
    n_bases_kept = 0
    for contig_id, seq in read_fna(fna_path):
        n_contigs += 1
        if len(seq) < min_contig_len:
            continue
        n_contigs_kept += 1
        n_bases_kept += len(seq)
        windows = make_windows(seq, window, stride)
        if not windows:
            continue
        print(f"[extract] {contig_id}: {len(seq)} bp → {len(windows)} windows",
              flush=True)
        emb_arr = _extract_mean_pool_batched(
            evo2_model,
            [sub for _, _, sub in windows],
            layer,
            batch_size=batch_size,
        ).astype(dtype_np)
        all_emb.append(emb_arr)
        rows.extend([(contig_id, w[0], w[1]) for w in windows])

    if not all_emb:
        raise RuntimeError(f"no windows extracted from {fna_path}")
    emb_cat = np.concatenate(all_emb, axis=0)
    Path(out_emb_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb_npy, emb_cat)
    with Path(out_win_csv).open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["contig", "start", "end"])
        wr.writerows(rows)
    manifest = {
        "fna": str(fna_path),
        "embeddings": str(out_emb_npy),
        "windows_csv": str(out_win_csv),
        "n_windows": int(emb_cat.shape[0]),
        "n_contigs": n_contigs,
        "n_contigs_kept": n_contigs_kept,
        "n_bases_kept": n_bases_kept,
        "window": window, "stride": stride, "layer": layer,
        "evo_model": evo_model, "dtype": dtype,
        "min_contig_len": int(min_contig_len),
        "batch_size": int(batch_size),
    }
    return manifest
