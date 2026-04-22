#!/usr/bin/env python3
"""Extract Evo2 7B mean-pool embeddings on 9-genome FASTA at chosen (window, stride).

This is a stripped-down mirror of scripts/inference_stride.py that emits ONLY
the per-window embedding array + window coord CSV. No downstream ContextModel.

Output per genome (to --out-dir):
  {genome_id}_embeddings.npy   (N_win, 4096) fp16
  {genome_id}_windows.csv      contig,start,end
Plus:
  manifest.json                window/stride/layer/model/timing
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


def read_fna(fna_path):
    contigs = []
    cur_id = None
    cur = []
    with open(fna_path) as f:
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


def make_windows(seq, window, stride):
    out = []
    for start in range(0, len(seq), stride):
        end = min(start + window, len(seq))
        sub = seq[start:end]
        if len(sub) < 100:
            continue
        if len(sub) < window:
            sub = sub + ("N" * (window - len(sub)))
        out.append((start, end, sub))
    return out


def extract_mean_pool(evo2_model, sequences, layer_name):
    import torch
    embs = []
    for seq in sequences:
        clean = "".join(c for c in seq if c in "ATCG")
        if len(clean) < 10:
            embs.append(np.zeros(4096, dtype=np.float32))
            continue
        input_ids = torch.tensor(
            evo2_model.tokenizer.tokenize(clean), dtype=torch.int
        ).unsqueeze(0).to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, emb_dict = evo2_model(
                input_ids, return_embeddings=True, layer_names=[layer_name]
            )
        emb = emb_dict[layer_name][0].float().mean(dim=0).cpu().numpy()
        embs.append(emb)
    return np.asarray(embs, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fna-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--layer", default="blocks.20.mlp.l3")
    p.add_argument("--evo-model", default="evo2_7b")
    p.add_argument("--cache-dir", default="/data/muskliu/npmaster/data/hf_cache")
    p.add_argument("--genome-ids", default=None,
                   help="comma-separated subset of genome_ids to process")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", default="fp16",
                   help="storage dtype for per-window embeddings: fp16 or fp32")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"

    selected = None
    if args.genome_ids:
        selected = {s.strip() for s in args.genome_ids.split(",") if s.strip()}

    print("Loading Evo2...", flush=True)
    t0 = time.time()
    from evo2 import Evo2
    evo2_model = Evo2(args.evo_model)
    print(f"Evo2 loaded in {time.time() - t0:.1f}s", flush=True)

    dtype_np = np.float16 if args.dtype == "fp16" else np.float32

    manifest = {
        "window": args.window,
        "stride": args.stride,
        "layer": args.layer,
        "evo_model": args.evo_model,
        "dtype": args.dtype,
        "genomes": {},
    }

    for fna_path in sorted(Path(args.fna_dir).glob("*.fna")):
        genome_id = fna_path.stem
        if selected is not None and genome_id not in selected:
            continue
        out_emb = args.out_dir / f"{genome_id}_embeddings.npy"
        out_win = args.out_dir / f"{genome_id}_windows.csv"
        if out_emb.exists() and out_win.exists():
            print(f"[skip] {genome_id} already present", flush=True)
            continue

        g_t0 = time.time()
        all_emb = []
        rows = []
        for contig_id, seq in read_fna(fna_path):
            windows = make_windows(seq, args.window, args.stride)
            if not windows:
                continue
            print(f"  {genome_id}/{contig_id}: {len(seq)} bp → {len(windows)} windows",
                  flush=True)
            seqs = [w[2] for w in windows]
            emb = extract_mean_pool(evo2_model, seqs, args.layer)
            all_emb.append(emb.astype(dtype_np))
            rows.extend([(contig_id, w[0], w[1]) for w in windows])

        if not all_emb:
            print(f"[warn] no windows for {genome_id}", flush=True)
            continue
        emb_cat = np.concatenate(all_emb, axis=0)
        np.save(out_emb, emb_cat)
        with open(out_win, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["contig", "start", "end"])
            wr.writerows(rows)

        dt = time.time() - g_t0
        manifest["genomes"][genome_id] = {
            "n_windows": int(emb_cat.shape[0]),
            "seconds": dt,
        }
        print(f"[{genome_id}] {emb_cat.shape} fp{8 * emb_cat.dtype.itemsize} "
              f"in {dt:.1f}s → {out_emb.name}", flush=True)

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
