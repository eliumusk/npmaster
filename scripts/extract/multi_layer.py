#!/usr/bin/env python3
"""Multi-layer Evo2 mean-pool extraction worker for the layer-probe diagnostic.

One process = one GPU. Single full forward (no block truncation) per window;
extract three layers' mean-pool in one pass:
    blocks.13.mlp.l3   blocks.20.mlp.l3   blocks.27.mlp.l3

Two sources:

  --source genome   --fasta <one .fna[.gz]>
        9g style: window over contigs at (window, stride). One stem per genome.

  --source sliced   --metadata-csv <windows_metadata_hardneg.csv>
                    --genome-dir <dir of {genome_id}.fna.gz>
                    --window-ids-subset <txt, one window_id per line>
        Training-pool style: slice exact (genome_id, contig_id, window_start,
        window_end) tuples from per-genome FASTAs. window_id == row id; the
        output preserves CSV row order so the resulting .npy can be merged
        in the same order as the existing L20 embeddings_merged.npy.

Output (per worker, per layer):
    <out-dir>/{stem}.{layer_short}.npy             (N, 4096) fp16
    <out-dir>/{stem}_windows.csv                   shared metadata
    <out-dir>/{stem}.manifest.json                 timing + layers + n_windows

`layer_short` = `L13` | `L20` | `L27`.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

LAYER_TRIPLE = ("blocks.13.mlp.l3", "blocks.20.mlp.l3", "blocks.27.mlp.l3")
LAYER_SHORT = {"blocks.13.mlp.l3": "L13", "blocks.20.mlp.l3": "L20", "blocks.27.mlp.l3": "L27"}


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


def gen_genome_windows(fasta, window, stride):
    """For --source genome: yield list[(contig, start, end, seq)]."""
    rows = []
    seqs = []
    for contig_id, seq in read_fna(fasta):
        for start, end, sub in make_windows(seq, window, stride):
            rows.append((contig_id, start, end))
            seqs.append(sub)
    return rows, seqs


def gen_sliced_windows(metadata_csv, genome_dir, keep_window_ids, window):
    """For --source sliced: read CSV in order, slice from per-genome FASTAs.

    Returns ordered (rows, seqs) where rows[i] = (window_id, genome_id,
    contig_id, window_start, window_end) and seqs[i] is the (possibly
    N-padded) window string of length `window`. Rows are filtered to
    keep_window_ids and otherwise emitted in the CSV's original order
    (this is what guarantees alignment with embeddings_merged.npy).
    """
    keep = set(keep_window_ids)
    # group rows by genome for cache-friendly FASTA reads, but emit in CSV order
    csv_rows = []
    by_genome = {}
    with open(metadata_csv, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            wid = row["window_id"]
            if wid not in keep:
                continue
            csv_rows.append(row)
            by_genome.setdefault(row["genome_id"], []).append(wid)

    # cache: genome_id -> {contig_id: seq}. Drop a genome's cache when its
    # last window is consumed to keep peak memory bounded.
    cache = {}
    last_use = {}
    for i, row in enumerate(csv_rows):
        last_use[row["window_id"]] = i

    rows = []
    seqs = []
    for i, row in enumerate(csv_rows):
        gid = row["genome_id"]
        if gid not in cache:
            fna = Path(genome_dir) / f"{gid}.fna.gz"
            if not fna.exists():
                fna = Path(genome_dir) / f"{gid}.fna"
            if not fna.exists():
                rows.append((row["window_id"], gid, row["contig_id"],
                             int(row["window_start"]), int(row["window_end"])))
                seqs.append("N" * window)
                continue
            cache[gid] = {cid: seq for cid, seq in read_fna(fna)}
        seq = cache[gid].get(row["contig_id"], "")
        s, e = int(row["window_start"]), int(row["window_end"])
        sub = seq[s:e]
        if len(sub) < window:
            sub = sub + ("N" * (window - len(sub)))
        else:
            sub = sub[:window]
        rows.append((row["window_id"], gid, row["contig_id"], s, e))
        seqs.append(sub)
        # purge cache when last window for this genome was just consumed
        is_last_for_genome = all(
            last_use[wid] <= i for wid in by_genome[gid]
        )
        if is_last_for_genome:
            cache.pop(gid, None)
    return rows, seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["genome", "sliced"], required=True)
    ap.add_argument("--fasta", type=Path, default=None)
    ap.add_argument("--metadata-csv", type=Path, default=None)
    ap.add_argument("--genome-dir", type=Path, default=None)
    ap.add_argument("--window-ids-subset", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--stride", type=int, default=0,
                    help="for --source genome only")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--cache-dir", default="/data/muskliu/npmaster/data/hf_cache")
    ap.add_argument("--evo-model", default="evo2_7b")
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(line + "\n")

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"

    log(f"worker pid={os.getpid()} source={args.source} stem={args.stem} "
        f"layers={LAYER_TRIPLE}")

    log("loading Evo2 (no truncation; need full forward through L27)...")
    t0 = time.time()
    from evo2 import Evo2
    from evo2_lr.extract_batch import extract_mean_pool_multi  # type: ignore
    model = Evo2(args.evo_model)
    load_s = time.time() - t0
    log(f"Evo2 loaded in {load_s:.1f}s")

    # Gather windows in deterministic order.
    if args.source == "genome":
        assert args.fasta is not None and args.stride > 0
        meta, seqs = gen_genome_windows(args.fasta, args.window, args.stride)
        meta_keys = ["contig", "start", "end"]
    else:
        assert (args.metadata_csv is not None and args.genome_dir is not None
                and args.window_ids_subset is not None)
        keep = args.window_ids_subset.read_text().split()
        meta, seqs = gen_sliced_windows(
            args.metadata_csv, args.genome_dir, keep, args.window,
        )
        meta_keys = ["window_id", "genome_id", "contig_id", "start", "end"]

    if args.max_windows and len(seqs) > args.max_windows:
        meta = meta[: args.max_windows]
        seqs = seqs[: args.max_windows]
    n = len(seqs)
    log(f"prepared {n} windows; first forward pass coming up...")

    # Single multi-layer forward over all sequences.
    forward_s_accum = [0.0]
    t_loop = time.time()
    out = extract_mean_pool_multi(
        model, seqs, list(LAYER_TRIPLE),
        batch_size=args.batch_size, forward_s_accum=forward_s_accum,
    )
    loop_s = time.time() - t_loop

    # Write per-layer .npy in matching row order.
    written = {}
    for ln in LAYER_TRIPLE:
        arr = out[ln].astype(np.float16)
        name = f"{args.stem}.{LAYER_SHORT[ln]}.npy"
        path = args.out_dir / name
        np.save(path, arr)
        written[LAYER_SHORT[ln]] = {"path": str(path), "shape": list(arr.shape)}
        log(f"wrote {name} shape={arr.shape}")

    # Shared metadata (one CSV for all three layers — same row order).
    csv_path = args.out_dir / f"{args.stem}_windows.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(meta_keys)
        for row in meta:
            wr.writerow(list(row))

    manifest = {
        "stem": args.stem,
        "source": args.source,
        "window": args.window,
        "stride": args.stride,
        "layers": list(LAYER_TRIPLE),
        "n_windows": n,
        "load_seconds": load_s,
        "forward_seconds": forward_s_accum[0],
        "loop_seconds": loop_s,
        "windows_per_sec": n / max(1e-9, loop_s),
        "forward_windows_per_sec": n / max(1e-9, forward_s_accum[0]),
        "outputs": written,
        "metadata_csv": str(csv_path),
    }
    with open(args.out_dir / f"{args.stem}.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"DONE n={n} fwd_s={forward_s_accum[0]:.1f} loop_s={loop_s:.1f} "
        f"rate={n / max(1e-9, loop_s):.3f} win/s")


if __name__ == "__main__":
    main()
