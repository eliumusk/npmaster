#!/usr/bin/env python3
"""Extract Evo2 7B mean-pool embeddings on 9-genome FASTA at chosen (window, stride).

This is a stripped-down mirror of scripts/inference/stride.py that emits ONLY
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
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_fna(fna_path):
    contigs = []
    cur_id = None
    cur = []
    opener = gzip.open if str(fna_path).endswith(".gz") else open
    with opener(fna_path, "rt") as f:
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


def clean_seq(seq):
    return "".join(c for c in seq if c in "ATCG")


# Set to True on first batch so we print shape/dtype once per process.
_DEBUG_BATCH_SHAPE_PRINTED = False


def extract_mean_pool(evo2_model, sequences, layer_name, batch_size=1, forward_s_accum=None):
    """Extract mean-pooled embeddings. Returns (N, 4096) np.float32.

    - Groups by token length so every batch is same-length (no padding).
    - On first call per process, prints the actual batched shape reaching the model
      (validates that --batch-size is doing real batching, not outer chunking).
    - If `forward_s_accum` is provided (a single-element list), accumulates the
      wall-clock time spent purely inside model forward into it, excluding tokenize
      and CPU-GPU transfer. Lets us separate "forward" from "everything else".
    """
    global _DEBUG_BATCH_SHAPE_PRINTED
    import torch
    batch_size = max(1, int(batch_size))
    out = [None] * len(sequences)
    by_len = defaultdict(list)
    for i, seq in enumerate(sequences):
        clean = clean_seq(seq)
        if len(clean) < 10:
            out[i] = np.zeros(4096, dtype=np.float32)
            continue
        toks = evo2_model.tokenizer.tokenize(clean)
        by_len[len(toks)].append((i, toks))

    for _, items in sorted(by_len.items(), key=lambda kv: kv[0], reverse=True):
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            input_ids = torch.tensor([t for _, t in chunk], dtype=torch.int).to("cuda")
            if not _DEBUG_BATCH_SHAPE_PRINTED:
                print(f"[debug] first forward: input_ids.shape={tuple(input_ids.shape)} "
                      f"dtype={input_ids.dtype} (requested batch_size={batch_size})",
                      flush=True)
                _DEBUG_BATCH_SHAPE_PRINTED = True
            if forward_s_accum is not None:
                torch.cuda.synchronize()
                t_fwd = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, emb_dict = evo2_model(
                    input_ids, return_embeddings=True, layer_names=[layer_name]
                )
                h = emb_dict[layer_name]
            if forward_s_accum is not None:
                torch.cuda.synchronize()
                forward_s_accum[0] += time.time() - t_fwd
            h = h.float()
            if h.ndim != 3:
                raise RuntimeError(f"unexpected embedding tensor shape: {tuple(h.shape)}")
            pooled = h.mean(dim=1).cpu().numpy().astype(np.float32)
            for (orig_i, _), emb in zip(chunk, pooled):
                out[orig_i] = emb

    missing = sum(v is None for v in out)
    if missing:
        raise RuntimeError(f"internal error: {missing} embeddings were not filled")
    return np.asarray(out, dtype=np.float32)


def truncate_blocks_to_layer(evo2_model, layer_name: str) -> dict:
    """Drop Evo2 blocks strictly after the block containing `layer_name`.

    Evo2 is StripedHyena with 32 blocks; our target `blocks.20.mlp.l3` needs
    only blocks [0..20]. Blocks 21-31 are ~34% of forward compute and HBM
    bandwidth — pure waste for our use case. The `norm` + `unembed` tail still
    runs on block-20 output (producing garbage logits we discard), adding ~5%
    overhead, but there is no clean way to skip them via the public API and
    the gain from skipping them is small.

    Returns a dict describing what was dropped; {} if no truncation possible.
    """
    import re
    import torch.nn as nn
    m = re.match(r"blocks\.(\d+)\.", layer_name)
    if not m:
        return {"truncated": False, "reason": f"layer_name '{layer_name}' not of 'blocks.N.*' form"}
    target_block = int(m.group(1))
    inner = getattr(evo2_model, "model", None)
    if inner is None or not hasattr(inner, "blocks"):
        return {"truncated": False, "reason": "model has no .model.blocks attribute"}
    blocks = inner.blocks
    if not isinstance(blocks, nn.ModuleList):
        return {"truncated": False, "reason": f"blocks is {type(blocks).__name__}, not ModuleList"}
    n_before = len(blocks)
    if target_block + 1 >= n_before:
        return {"truncated": False, "reason": f"target block {target_block} is last block"}
    inner.blocks = blocks[:target_block + 1]
    return {
        "truncated": True,
        "kept_blocks": target_block + 1,
        "dropped_blocks": n_before - (target_block + 1),
        "total_blocks": n_before,
    }


def fasta_paths(fna_dir: Path):
    exts = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")
    return sorted(p for p in fna_dir.iterdir() if p.is_file() and str(p).endswith(exts))


def genome_stem(path: Path) -> str:
    name = path.name
    for ext in (".fasta.gz", ".fna.gz", ".fa.gz", ".fasta", ".fna", ".fa"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return path.stem


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
    p.add_argument("--min-contig-len", type=int, default=0,
                   help="skip contigs shorter than this many bp before windowing")
    p.add_argument("--max-windows", type=int, default=0,
                   help="optional cap on total windows per process; for benchmarking")
    p.add_argument("--truncate-blocks", action=argparse.BooleanOptionalAction, default=True,
                   help="drop Evo2 blocks after the one containing --layer (default: on)")
    p.add_argument("--manifest-name", default=None,
                   help="manifest filename under --out-dir (default auto: "
                        "'manifest_{genome_id}.json' if --genome-ids selects exactly one, "
                        "else 'manifest.json'). Prevents shared-dir overwrite across parallel shards.")
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
    load_s = time.time() - t0
    print(f"Evo2 loaded in {load_s:.1f}s", flush=True)

    truncation_info = {"truncated": False, "reason": "flag off"}
    if args.truncate_blocks:
        truncation_info = truncate_blocks_to_layer(evo2_model, args.layer)
        if truncation_info.get("truncated"):
            print(f"[truncate] kept blocks [0..{truncation_info['kept_blocks']-1}], "
                  f"dropped {truncation_info['dropped_blocks']}/{truncation_info['total_blocks']}",
                  flush=True)
        else:
            print(f"[truncate] skipped: {truncation_info.get('reason')}", flush=True)

    dtype_np = np.float16 if args.dtype == "fp16" else np.float32

    forward_s_accum = [0.0]  # mutable, passed to extract_mean_pool
    manifest = {
        "window": args.window,
        "stride": args.stride,
        "layer": args.layer,
        "evo_model": args.evo_model,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "load_seconds": load_s,
        "truncate_blocks": truncation_info,
        "genomes": {},
    }

    for fna_path in fasta_paths(Path(args.fna_dir)):
        genome_id = genome_stem(fna_path)
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
        n_contigs = 0
        n_contigs_kept = 0
        n_bases_kept = 0
        reached_window_cap = False
        for contig_id, seq in read_fna(fna_path):
            n_contigs += 1
            if len(seq) < args.min_contig_len:
                continue
            n_contigs_kept += 1
            n_bases_kept += len(seq)
            windows = make_windows(seq, args.window, args.stride)
            if not windows:
                continue
            if args.max_windows > 0:
                remaining = args.max_windows - len(rows)
                if remaining <= 0:
                    reached_window_cap = True
                    break
                if len(windows) > remaining:
                    windows = windows[:remaining]
                    reached_window_cap = True
            print(f"  {genome_id}/{contig_id}: {len(seq)} bp → {len(windows)} windows",
                  flush=True)
            seqs = [w[2] for w in windows]
            emb = extract_mean_pool(evo2_model, seqs, args.layer, args.batch_size,
                                    forward_s_accum=forward_s_accum)
            all_emb.append(emb.astype(dtype_np))
            rows.extend([(contig_id, w[0], w[1]) for w in windows])
            if reached_window_cap:
                break

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
            "n_contigs": n_contigs,
            "n_contigs_kept": n_contigs_kept,
            "n_bases_kept": n_bases_kept,
            "max_windows": int(args.max_windows),
            "reached_window_cap": reached_window_cap,
            "seconds": dt,
        }
        print(f"[{genome_id}] {emb_cat.shape} fp{8 * emb_cat.dtype.itemsize} "
              f"in {dt:.1f}s → {out_emb.name}", flush=True)

    manifest["forward_seconds_total"] = forward_s_accum[0]
    total_windows = sum(g["n_windows"] for g in manifest["genomes"].values())
    manifest["total_windows"] = total_windows
    if total_windows:
        manifest["forward_seconds_per_window"] = forward_s_accum[0] / total_windows
        manifest["forward_windows_per_second"] = total_windows / max(1e-9, forward_s_accum[0])
    # Resolve manifest filename: explicit > single-genome auto > default.
    if args.manifest_name:
        manifest_path = args.out_dir / args.manifest_name
    elif len(manifest["genomes"]) == 1:
        only_id = next(iter(manifest["genomes"]))
        manifest_path = args.out_dir / f"manifest_{only_id}.json"
    else:
        manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] load={load_s:.1f}s  forward={forward_s_accum[0]:.1f}s  "
          f"windows={total_windows}  "
          f"forward_w/s={total_windows / max(1e-9, forward_s_accum[0]):.2f}", flush=True)


if __name__ == "__main__":
    main()
