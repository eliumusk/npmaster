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
) -> dict:
    """Extract Evo2 mean-pool embeddings for a single FASTA.

    Writes `{out_emb_npy}` of shape (N_windows, 4096) and `{out_win_csv}`
    with columns contig,start,end. Returns a small manifest dict.
    """
    import os
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from evo2 import Evo2

    print(f"[extract] loading Evo2 ({evo_model})...", flush=True)
    t0 = time.time()
    evo2_model = Evo2(evo_model)
    print(f"[extract] loaded in {time.time() - t0:.1f}s", flush=True)

    dtype_np = np.float16 if dtype == "fp16" else np.float32

    all_emb: list[np.ndarray] = []
    rows: list[tuple[str, int, int]] = []
    for contig_id, seq in read_fna(fna_path):
        windows = make_windows(seq, window, stride)
        if not windows:
            continue
        print(f"[extract] {contig_id}: {len(seq)} bp → {len(windows)} windows",
              flush=True)
        embs = []
        for _, _, sub in windows:
            clean = "".join(c for c in sub if c in "ATCG")
            if len(clean) < 10:
                embs.append(np.zeros(4096, dtype=np.float32))
                continue
            input_ids = torch.tensor(
                evo2_model.tokenizer.tokenize(clean), dtype=torch.int
            ).unsqueeze(0).to("cuda")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, emb_dict = evo2_model(
                    input_ids, return_embeddings=True, layer_names=[layer]
                )
            emb = emb_dict[layer][0].float().mean(dim=0).cpu().numpy()
            embs.append(emb)
        emb_arr = np.asarray(embs, dtype=np.float32).astype(dtype_np)
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
        "window": window, "stride": stride, "layer": layer,
        "evo_model": evo_model, "dtype": dtype,
    }
    return manifest
