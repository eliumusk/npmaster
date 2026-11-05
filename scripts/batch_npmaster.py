#!/usr/bin/env python3
"""Batch NP-Master pipeline on a directory of FASTAs (no Supabase).

For each *.fa in --in-dir, runs:
  1. parallel_extract across 16 GPUs (node7 + node8)
  2. U-Net inference (infer_shards.py subprocess)
  3. Region decode (evo2_per_token_regions.py subprocess)
  4. LR type classify (evo2_lr_regions.py subprocess)
  5. GenBank generation (prodigal)
  6. Pfam annotation per CDS (hmmscan)
  7. MIBiG nearest-neighbor (DIAMOND blastp)

Writes one subdir per genome under --out-dir with:
  regions.csv         (typed)
  regions.gbk
  regions.bed
  cds_features.json   { region_idx → list of CDS dicts }
  mibig_hits.json     { region_idx → list of hits }
  manifest.json       (timings, counts)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from serve.parallel_extract import run_parallel as run_parallel_extract
from serve.parallel_extract import run_parallel_lr
from serve.parallel_extract import walk_fasta_to_metadata
from serve.genbank import generate_genbank
from serve.score_track import write_bedgraph
from serve.pfam import annotate_regions_gbk
from serve.mibig import search_regions_against_mibig
from serve.fasta import csv_regions_to_bed, write_faidx, sha256_file


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_subproc(cmd: list[str], stage: str, cwd: Path, env: dict | None = None) -> None:
    log(f"  [{stage}] $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    log(f"  [{stage}] exit={proc.returncode} in {time.time()-t0:.1f}s")
    if proc.returncode != 0:
        log(f"    stderr tail: {proc.stderr[-800:]}")
        raise RuntimeError(f"{stage} failed")


def _subprocess_env(python_bin: Path) -> dict:
    import os
    env = os.environ.copy()
    env_root = python_bin.parent.parent
    candidates = list(env_root.glob("lib/python*/site-packages/torch/lib"))
    if candidates:
        torch_lib = str(candidates[0])
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_genome(*, fa_path: Path, out_dir: Path, cache_dir: Path,
               settings: dict, manifest_only_existing: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"genome": fa_path.stem, "fa_bytes": fa_path.stat().st_size}

    sha = sha256_file(fa_path)
    stem = sha[:16]
    manifest["sha256"] = sha
    manifest["stem"] = stem

    log(f"=== {fa_path.name}  sha={sha[:8]}  stem={stem} ===")

    # Stage 1: parallel 128-d extract (cached by sha)
    features_dir = cache_dir / "features" / sha
    done_marker = features_dir / "DONE"
    if not done_marker.exists():
        log(f"  parallel_extract on 16 GPUs…")
        t0 = time.time()
        n_parts = run_parallel_extract(
            fasta_path=fa_path, features_dir=features_dir, stem=stem,
            repo_root=settings["npmaster_repo_root"],
            python_bin=settings["python_bin"],
            torch_lib=settings["torch_lib"],
            window=8192, stride=2048,
            hosts=settings["extract_hosts"], gpus_per_host=8,
            work_dir=features_dir / "_meta",
        )
        done_marker.touch()
        manifest["t_extract_128d"] = time.time() - t0
        manifest["n_parts_128d"] = n_parts
    else:
        log(f"  128-d features cache hit ({features_dir})")
        manifest["t_extract_128d"] = 0.0

    # Stage 2: U-Net inference
    probs_dir = out_dir / "probs"
    probs_dir.mkdir(exist_ok=True)
    t0 = time.time()
    run_subproc([
        str(settings["python_bin"]),
        "scripts/inference/evo2_per_token/infer_shards.py",
        "--features-dir", str(features_dir),
        "--ckpt", str(settings["model_unet_ckpt"]),
        "--out-dir", str(probs_dir),
        "--stems", stem,
    ], "infer", settings["npmaster_repo_root"], env=_subprocess_env(settings["python_bin"]))
    manifest["t_infer"] = time.time() - t0

    # Stage 3: Region decode (using OER discovery thresholds 0.50 / 2 kb default)
    raw_csv = out_dir / "regions_raw.csv"
    t0 = time.time()
    run_subproc([
        str(settings["python_bin"]),
        "scripts/decode/evo2_per_token_regions.py",
        "--probs-dir", str(probs_dir),
        "--out-regions-csv", str(raw_csv),
        "--threshold", "0.50", "--min-len-bp", "2000",
        "--upsample-k", "8",
    ], "decode", settings["npmaster_repo_root"], env=_subprocess_env(settings["python_bin"]))
    manifest["t_decode"] = time.time() - t0

    n_regions = sum(1 for _ in open(raw_csv)) - 1
    manifest["n_regions"] = n_regions
    log(f"  decoded {n_regions} regions")
    if n_regions == 0:
        # short-circuit
        regions_csv = out_dir / "regions.csv"
        shutil.copy(raw_csv, regions_csv)
        manifest["t_classify"] = 0.0
        manifest["t_genbank"] = 0.0
        manifest["t_pfam"] = 0.0
        manifest["t_mibig"] = 0.0
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    # Stage 4: LR type classify — needs 4096-d features (separate cache)
    lr_emb_dir = cache_dir / "features_lr" / sha
    lr_done = lr_emb_dir / "DONE"
    if not lr_done.exists():
        meta_csv = features_dir / "_meta" / "windows_metadata.csv"
        if not meta_csv.exists():
            meta_csv = lr_emb_dir / "windows_metadata.csv"
            meta_csv.parent.mkdir(parents=True, exist_ok=True)
            walk_fasta_to_metadata(fa_path, meta_csv, window=8192, stride=2048, stem=stem)
        log(f"  parallel_extract LR 4096-d on 16 GPUs…")
        t0 = time.time()
        n_lr = run_parallel_lr(
            fasta_path=fa_path, metadata_csv=meta_csv,
            emb_dir=lr_emb_dir, stem=stem,
            work_dir=lr_emb_dir / "_work",
            serve_root=REPO_ROOT / "serve",
            npmaster_root=settings["npmaster_repo_root"],
            python_bin=settings["python_bin"],
            torch_lib=settings["torch_lib"],
            hosts=settings["extract_hosts"], gpus_per_host=8,
        )
        lr_done.touch()
        manifest["t_extract_4096d"] = time.time() - t0
        manifest["n_windows_4096d"] = n_lr
    else:
        log(f"  4096-d features cache hit")
        manifest["t_extract_4096d"] = 0.0

    regions_csv = out_dir / "regions.csv"
    t0 = time.time()
    run_subproc([
        str(settings["python_bin"]),
        "scripts/classify/evo2_lr_regions.py",
        "--regions-csv", str(raw_csv),
        "--emb-dir", str(lr_emb_dir),
        "--type-lr-dir", str(settings["lr_type_ckpt_dir"]),
        "--out-csv", str(regions_csv),
    ], "classify", settings["npmaster_repo_root"], env=_subprocess_env(settings["python_bin"]))
    manifest["t_classify"] = time.time() - t0

    # Stage 5: GenBank generation (prodigal CDS)
    gbk_path = out_dir / "regions.gbk"
    t0 = time.time()
    try:
        n_gbk = generate_genbank(
            fasta_path=fa_path, regions_csv=regions_csv,
            out_gbk_path=gbk_path, prodigal_bin=settings["prodigal_bin"],
            work_dir=out_dir / "_genbank_work",
        )
        manifest["t_genbank"] = time.time() - t0
        manifest["n_gbk_records"] = n_gbk
    except Exception as e:
        log(f"  genbank failed (non-fatal): {e}")
        manifest["t_genbank"] = time.time() - t0
        gbk_path = None

    # Stage 6: BED + FAI + bedgraph
    bed_path = out_dir / "regions.bed"
    csv_regions_to_bed(regions_csv, bed_path)
    fai_path = out_dir / "input.fasta.fai"
    write_faidx(fa_path, fai_path)
    try:
        wig_path = out_dir / "scores.bedgraph"
        write_bedgraph(
            probs_npz=probs_dir / f"{stem}.probs.npz",
            coords_csv=probs_dir / f"{stem}.coords.csv",
            out_path=wig_path,
        )
    except Exception as e:
        log(f"  bedgraph failed (non-fatal): {e}")

    # Stage 7: Pfam annotation
    cds_features = {}
    if gbk_path and gbk_path.exists():
        t0 = time.time()
        try:
            cds_features = annotate_regions_gbk(
                regions_gbk=gbk_path,
                pfam_db=settings["pfam_db_path"],
                work_dir=out_dir / "_pfam_work",
                hmmer_bin=settings["hmmer_bin"],
                threads=settings["hmmer_threads"],
            )
            manifest["t_pfam"] = time.time() - t0
        except Exception as e:
            log(f"  pfam failed (non-fatal): {e}")
            manifest["t_pfam"] = time.time() - t0
    with open(out_dir / "cds_features.json", "w") as f:
        json.dump(cds_features, f)

    # Stage 8: MIBiG nearest-neighbor
    mibig_hits = {}
    if gbk_path and gbk_path.exists() and settings["mibig_dmnd_path"].exists():
        t0 = time.time()
        try:
            mibig_hits = search_regions_against_mibig(
                regions_gbk=gbk_path,
                dmnd_db=settings["mibig_dmnd_path"],
                meta_json=settings["mibig_meta_path"],
                work_dir=out_dir / "_mibig_work",
                diamond_bin=settings["diamond_bin"],
                top_k=3,
            )
            manifest["t_mibig"] = time.time() - t0
        except Exception as e:
            log(f"  mibig failed (non-fatal): {e}")
            manifest["t_mibig"] = time.time() - t0
    with open(out_dir / "mibig_hits.json", "w") as f:
        json.dump(mibig_hits, f)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"  DONE: {n_regions} regions, "
        f"{sum(len(v) for v in cds_features.values()) if cds_features else 0} CDS, "
        f"{sum(1 for v in mibig_hits.values() if v) if mibig_hits else 0} regions with mibig hits")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("/data/syh/bench_34/_cache"))
    p.add_argument("--hosts", default="localhost,node8")
    args = p.parse_args()

    settings = {
        "python_bin": Path("/data/conda/miniconda/envs/evo2_bgc/bin/python"),
        "torch_lib": "/data/conda/miniconda/envs/evo2_bgc/lib/python3.12/site-packages/torch/lib",
        "prodigal_bin": Path("/root/miniconda3/envs/bgc/bin/prodigal"),
        "hmmer_bin": Path("/root/miniconda3/envs/bgc/bin/hmmscan"),
        "diamond_bin": Path("/root/miniconda3/envs/bgc/bin/diamond"),
        "pfam_db_path": Path("/data/syh/NP-Master-web/data/pfam/Pfam-A.hmm"),
        "mibig_dmnd_path": Path("/data/syh/NP-Master-web/data/mibig/mibig.dmnd"),
        "mibig_meta_path": Path("/data/syh/NP-Master-web/data/mibig/mibig_meta.json"),
        "hmmer_threads": 16,
        "extract_hosts": [h.strip() for h in args.hosts.split(",") if h.strip()],
        "npmaster_repo_root": Path("/data/muskliu/npmaster"),
        "model_unet_ckpt": Path("/data/muskliu/npmaster/experiments/evo2_per_token_unet/train_runs/full_weakneg_bce_w05_ddp4_gb64_seed0/best.pt"),
        "lr_type_ckpt_dir": Path("/data/muskliu/npmaster/data/evo2_lr_multiscale/type_lr"),
    }

    fasta_files = sorted(args.in_dir.glob("*.fa"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"BATCH NP-Master: {len(fasta_files)} genomes")
    log(f"  in:  {args.in_dir}")
    log(f"  out: {args.out_dir}")
    log(f"  hosts: {settings['extract_hosts']}")

    all_manifests = []
    t_total = time.time()
    for i, fa in enumerate(fasta_files, start=1):
        log(f"\n[{i}/{len(fasta_files)}] {fa.name}")
        genome_out = args.out_dir / fa.stem
        if (genome_out / "manifest.json").exists():
            log(f"  already done, skipping")
            with open(genome_out / "manifest.json") as fh:
                all_manifests.append(json.load(fh))
            continue
        try:
            m = run_genome(
                fa_path=fa, out_dir=genome_out,
                cache_dir=args.cache_dir, settings=settings,
            )
            all_manifests.append(m)
        except Exception as e:
            log(f"  FAILED: {e}")
            all_manifests.append({"genome": fa.stem, "error": str(e)})

    summary_path = args.out_dir / "batch_summary.json"
    with open(summary_path, "w") as fh:
        json.dump({
            "total_time_sec": time.time() - t_total,
            "n_genomes": len(fasta_files),
            "manifests": all_manifests,
        }, fh, indent=2)
    log(f"\nALL DONE in {(time.time() - t_total)/60:.1f} min  →  {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
