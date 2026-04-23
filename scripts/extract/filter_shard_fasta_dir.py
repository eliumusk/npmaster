#!/usr/bin/env python3
"""Filter FASTA contigs by length and stream them into balanced shards.

This is intended for large metagenome assemblies where millions of short
contigs dominate file size/index overhead but cannot produce useful BGC calls.
It does not keep all contigs in memory: each retained record is assigned to the
currently smallest shard as it streams through the input files.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def fasta_paths(in_dir: Path) -> list[Path]:
    exts = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")
    return sorted(p for p in in_dir.iterdir() if p.is_file() and str(p).endswith(exts))


def sample_name(path: Path) -> str:
    name = path.name
    for suffix in (".final.contigs.fa", ".final.contigs.fna", ".fasta.gz",
                   ".fna.gz", ".fa.gz", ".fasta", ".fna", ".fa"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def iter_fasta(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    cid = None
    seq: list[str] = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cid is not None:
                    yield cid, "".join(seq)
                cid = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip().upper())
        if cid is not None:
            yield cid, "".join(seq)


def write_record(handle, cid: str, seq: str) -> None:
    handle.write(f">{cid}\n")
    for i in range(0, len(seq), 80):
        handle.write(seq[i:i + 80] + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Filter large FASTA dir and write balanced shards")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="shard")
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--min-contig-len", type=int, default=10_000)
    p.add_argument("--prefix-contig-with-sample", action="store_true", default=True)
    p.add_argument("--no-prefix-contig-with-sample", action="store_false",
                   dest="prefix_contig_with_sample")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [args.out_dir / f"{args.prefix}{i:02d}.fna" for i in range(args.n_shards)]
    handles = [p.open("w") for p in shard_paths]
    shard_bp = [0] * args.n_shards
    shard_contigs = [0] * args.n_shards
    stats = {
        "input_dir": str(args.input_dir),
        "min_contig_len": args.min_contig_len,
        "n_shards": args.n_shards,
        "inputs": {},
        "shards": {},
    }

    try:
        for path in fasta_paths(args.input_dir):
            sname = sample_name(path)
            n = kept = bp = kept_bp = 0
            for cid, seq in iter_fasta(path):
                n += 1
                bp += len(seq)
                if len(seq) < args.min_contig_len:
                    continue
                kept += 1
                kept_bp += len(seq)
                out_cid = f"{sname}|{cid}" if args.prefix_contig_with_sample else cid
                shard_i = min(range(args.n_shards), key=lambda i: shard_bp[i])
                write_record(handles[shard_i], out_cid, seq)
                shard_bp[shard_i] += len(seq)
                shard_contigs[shard_i] += 1
            stats["inputs"][path.name] = {
                "n_contigs": n,
                "n_contigs_kept": kept,
                "bp": bp,
                "bp_kept": kept_bp,
            }
            print(f"[{path.name}] kept {kept}/{n} contigs, {kept_bp:,}/{bp:,} bp", flush=True)
    finally:
        for h in handles:
            h.close()

    for i, path in enumerate(shard_paths):
        stats["shards"][path.name] = {
            "n_contigs": shard_contigs[i],
            "bp": shard_bp[i],
        }
        print(f"[shard{i:02d}] {shard_contigs[i]} contigs, {shard_bp[i]:,} bp -> {path}", flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
