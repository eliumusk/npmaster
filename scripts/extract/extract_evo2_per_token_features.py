#!/usr/bin/env python3
"""Per-token Evo2 production extraction worker (step2_per_token).

One worker = one GPU = one (or many) genome/shard.

Supports two modes via --source:
  1. genome  : iterate contigs of a single genome FASTA, emit (w, s)-windowed
               per-token projected embeddings. One .npy + _windows.csv per
               genome. Used for the 9g inference arm (w=8192/s=2048).
  2. fasta   : each FASTA record is ONE window (record id == window_id). Used
               for the training pool when all windows are pre-materialised.
  3. sliced  : read windows_metadata CSV and slice sequences from per-genome
               .fna.gz files. Fallback for training windows that were NOT
               pre-materialised into padded_sequences.fasta. Each shard is
               one modulo class of window_id.

Output (per worker):
  <out-dir>/<shard_stem>.npy             (N_kept, T_kept, D_reduced) fp16
  <out-dir>/<shard_stem>_windows.csv     window metadata rows
  <out-dir>/<shard_stem>.manifest.json   per-worker manifest

Locked config (override only if you know why):
  layer           = blocks.20.mlp.l3
  D_reduced       = 128
  downsample K    = 8            (=> T_kept = 1024 for w=8192; 2048 for w=16384)
  proj_seed       = 0xE2E2
  dtype           = fp16
  truncate_blocks = True         (drop blocks 21..31)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def read_fna(fna_path):
    opener = gzip.open if str(fna_path).endswith(".gz") else open
    cur_id, cur = None, []
    with opener(fna_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(cur)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip().upper())
    if cur_id is not None:
        yield cur_id, "".join(cur)


def iter_fasta_records(fasta_path):
    """Yield (record_id, sequence) streaming, memory-safe on 2GB FASTAs."""
    cur_id, cur = None, []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(cur)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip().upper())
    if cur_id is not None:
        yield cur_id, "".join(cur)


def make_windows(seq, window, stride, contig_id):
    """Emit windows over one contig. Tail-pad with N if final window < window."""
    out = []
    L = len(seq)
    for start in range(0, L, stride):
        end = min(start + window, L)
        sub = seq[start:end]
        clean_len = sum(1 for c in sub if c in "ATCG")
        if clean_len < 100:
            continue
        if len(sub) < window:
            sub = sub + ("N" * (window - len(sub)))
        out.append({
            "contig": contig_id,
            "start": start,
            "end": end,
            "clean_len": clean_len,
            "seq": sub,
        })
    return out


def clean_for_tokenizer(seq):
    """Evo2 tokenizer wants only ATCGN. Upper-case + strip others to N."""
    allowed = set("ATCGN")
    return "".join(c if c in allowed else "N" for c in seq.upper())


def build_projection(d_in: int, d_out: int, seed: int, device, torch):
    g = torch.Generator(device=device).manual_seed(seed)
    R = torch.randn(d_in, d_out, generator=g, device=device, dtype=torch.bfloat16)
    R = R / (d_out ** 0.5)
    return R


def run_forward(model, input_ids, layer, R, downsample_k, torch):
    """One forward pass → (T_kept, D_reduced) bfloat16 on GPU.

    Projects BEFORE cpu transfer; mean-pools every K tokens BEFORE fp16 cast.
    """
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, emb_dict = model(input_ids, return_embeddings=True, layer_names=[layer])
        h = emb_dict[layer]  # (1, T, 4096) bf16
        h_proj = torch.matmul(h, R)  # (1, T, d_out) bf16
        if downsample_k > 1:
            K = downsample_k
            T_eff = (h_proj.shape[1] // K) * K
            h_proj = h_proj[:, :T_eff, :].reshape(
                h_proj.shape[0], T_eff // K, K, h_proj.shape[2]
            ).mean(dim=2)
    return h_proj.squeeze(0).to(torch.float16).cpu().numpy()


def flush_shard(buffer_emb, buffer_meta, shard_idx, out_dir, stem):
    """Append flush: write one numbered shard chunk so we never buffer
    everything. Shards get concatenated into the final .npy by a merge step
    downstream (or kept as separate files — manifest tracks both)."""
    if not buffer_emb:
        return None, None
    arr = np.stack(buffer_emb, axis=0)
    npy = out_dir / f"{stem}.part{shard_idx:04d}.npy"
    csv_p = out_dir / f"{stem}.part{shard_idx:04d}_windows.csv"
    np.save(npy, arr)
    with open(csv_p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(list(buffer_meta[0].keys()))
        for row in buffer_meta:
            wr.writerow(list(row.values()))
    return npy, csv_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["genome", "fasta", "sliced"], required=True)
    ap.add_argument("--fasta", type=Path, default=None,
                    help="for --source=genome: one genome .fna[.gz]; "
                         "for --source=fasta: windows FASTA")
    ap.add_argument("--genome-dir", type=Path, default=None,
                    help="for --source=sliced: dir of {genome_id}.fna.gz")
    ap.add_argument("--metadata-csv", type=Path, default=None,
                    help="for --source=sliced: windows_metadata CSV "
                         "(requires window_id,genome_id,contig_id,window_start,window_end)")
    ap.add_argument("--window-ids-subset", type=Path, default=None,
                    help="txt file of window_ids this worker handles "
                         "(required for --source=fasta/sliced)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, required=True,
                    help="output filename stem, e.g. genome id or 'gpu_node7_0'")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--layer", default="blocks.20.mlp.l3")
    ap.add_argument("--d-reduced", type=int, default=128)
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--proj-seed", type=int, default=0xE2E2)
    ap.add_argument("--flush-every", type=int, default=1000,
                    help="flush a shard file every N windows")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="cap for dry runs; 0 = unlimited")
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(line + "\n")

    os.environ.setdefault("HF_HUB_CACHE", "/data/public_models/hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    log(f"worker start pid={os.getpid()} source={args.source} stem={args.stem} "
        f"w={args.window} s={args.stride} K={args.downsample} d={args.d_reduced}")

    log("loading Evo2-7b...")
    t0 = time.time()
    import torch
    from evo2 import Evo2
    from evo2_lr.extract import truncate_blocks_to_layer
    model = Evo2("evo2_7b")
    load_s = time.time() - t0
    log(f"Evo2 loaded in {load_s:.1f}s; truncating blocks...")
    info = truncate_blocks_to_layer(model, args.layer)
    log(f"truncate: {info}")

    R = build_projection(4096, args.d_reduced, args.proj_seed, "cuda", torch)

    # ---- window producer -----------------------------------------------
    def gen_windows():
        """Yield dicts with keys: window_id, contig, start, end, clean_len, seq."""
        if args.source == "genome":
            assert args.fasta is not None
            win_counter = 0
            for contig_id, seq in read_fna(args.fasta):
                for w in make_windows(seq, args.window, args.stride, contig_id):
                    w["window_id"] = f"{args.stem}_{win_counter:07d}"
                    win_counter += 1
                    yield w
        elif args.source == "fasta":
            assert args.fasta is not None
            keep = None
            if args.window_ids_subset is not None:
                keep = set(args.window_ids_subset.read_text().split())
            for rec_id, seq in iter_fasta_records(args.fasta):
                if keep is not None and rec_id not in keep:
                    continue
                seq_len = len(seq)
                clean_len = sum(1 for c in seq.upper() if c in "ATCG")
                if seq_len != args.window:
                    # tail-pad or truncate
                    if seq_len < args.window:
                        seq = seq + ("N" * (args.window - seq_len))
                    else:
                        seq = seq[: args.window]
                yield {
                    "window_id": rec_id,
                    "contig": "",
                    "start": 0,
                    "end": seq_len,
                    "clean_len": clean_len,
                    "seq": seq,
                }
        elif args.source == "sliced":
            assert args.metadata_csv is not None and args.genome_dir is not None
            assert args.window_ids_subset is not None
            keep = set(args.window_ids_subset.read_text().split())
            # group CSV rows by genome_id for cache-friendly FASTA reads.
            import csv as _csv
            by_genome = {}
            with open(args.metadata_csv, "r") as f:
                rdr = _csv.DictReader(f)
                for row in rdr:
                    if row["window_id"] not in keep:
                        continue
                    by_genome.setdefault(row["genome_id"], []).append(row)
            log(f"sliced: {sum(len(v) for v in by_genome.values())} windows "
                f"over {len(by_genome)} genomes")
            for gid, rows in by_genome.items():
                fna = args.genome_dir / f"{gid}.fna.gz"
                if not fna.exists():
                    fna = args.genome_dir / f"{gid}.fna"
                if not fna.exists():
                    log(f"MISSING genome {gid} at {args.genome_dir}; skipping {len(rows)} windows")
                    continue
                contigs = {cid: seq for cid, seq in read_fna(fna)}
                for row in rows:
                    cid = row["contig_id"]
                    if cid not in contigs:
                        log(f"MISSING contig {gid}/{cid}; skip")
                        continue
                    seq = contigs[cid]
                    start = int(row["window_start"])
                    end = int(row["window_end"])
                    sub = seq[start:end]
                    clean_len = sum(1 for c in sub if c in "ATCG")
                    if len(sub) < args.window:
                        sub = sub + ("N" * (args.window - len(sub)))
                    yield {
                        "window_id": row["window_id"],
                        "contig": cid,
                        "start": start,
                        "end": end,
                        "clean_len": clean_len,
                        "seq": sub,
                    }
        else:
            raise ValueError(args.source)

    # ---- main loop -----------------------------------------------------
    buffer_emb = []
    buffer_meta = []
    shard_idx = 0
    shard_files = []
    n_done = 0
    fwd_s = 0.0
    t_loop0 = time.time()

    for w in gen_windows():
        if args.max_windows and n_done >= args.max_windows:
            break
        seq = clean_for_tokenizer(w["seq"])
        toks = model.tokenizer.tokenize(seq)
        input_ids = torch.tensor([toks], dtype=torch.int, device="cuda")

        torch.cuda.synchronize()
        tf = time.time()
        try:
            arr = run_forward(model, input_ids, args.layer, R, args.downsample, torch)
        except Exception as e:
            log(f"[error] window {w.get('window_id')} forward failed: {e}")
            continue
        torch.cuda.synchronize()
        fwd_s += time.time() - tf

        buffer_emb.append(arr)
        buffer_meta.append({
            "window_id": w["window_id"],
            "contig": w["contig"],
            "start": w["start"],
            "end": w["end"],
            "clean_len": w["clean_len"],
        })
        n_done += 1

        if n_done % 50 == 0:
            elapsed = time.time() - t_loop0
            log(f"progress n={n_done} fwd_s={fwd_s:.1f} "
                f"rate={n_done/max(1e-9,elapsed):.3f} win/s "
                f"last_shape={arr.shape}")

        if len(buffer_emb) >= args.flush_every:
            npy, csv_p = flush_shard(buffer_emb, buffer_meta, shard_idx,
                                     args.out_dir, args.stem)
            shard_files.append({"npy": str(npy), "csv": str(csv_p),
                                "n": len(buffer_emb)})
            log(f"flushed shard {shard_idx} ({len(buffer_emb)} windows) -> {npy.name}")
            buffer_emb, buffer_meta = [], []
            shard_idx += 1

    # final flush
    if buffer_emb:
        npy, csv_p = flush_shard(buffer_emb, buffer_meta, shard_idx,
                                 args.out_dir, args.stem)
        shard_files.append({"npy": str(npy), "csv": str(csv_p),
                            "n": len(buffer_emb)})
        log(f"flushed final shard {shard_idx} ({len(buffer_emb)} windows)")

    total_elapsed = time.time() - t_loop0
    manifest = {
        "stem": args.stem,
        "source": args.source,
        "window": args.window,
        "stride": args.stride,
        "layer": args.layer,
        "d_reduced": args.d_reduced,
        "downsample": args.downsample,
        "proj_seed": args.proj_seed,
        "n_windows": n_done,
        "shard_files": shard_files,
        "load_seconds": load_s,
        "forward_seconds": fwd_s,
        "loop_seconds": total_elapsed,
        "windows_per_sec": n_done / max(1e-9, total_elapsed),
    }
    manifest_path = args.out_dir / f"{args.stem}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"DONE n={n_done} fwd_s={fwd_s:.1f} total_s={total_elapsed:.1f} "
        f"rate={n_done/max(1e-9,total_elapsed):.3f} win/s -> {manifest_path}")


if __name__ == "__main__":
    main()
