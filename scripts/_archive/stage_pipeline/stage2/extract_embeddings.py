#!/usr/bin/env python3
"""
Stage 2.2: Extract Evo2 7B embeddings + NLL features for all windows.

Usage:
  CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py \
    --gpu_id 0 --n_gpus 8 \
    --genomes_dir data/genomes/ \
    --metadata_csv data/windows/windows_metadata.csv \
    --output_dir data/embeddings/per_genome/ \
    --layer_name blocks.20.mlp.l3

Each GPU processes genomes[gpu_id::n_gpus] (interleaved for load balance).
Checkpoint per genome — safe to restart.
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def load_fasta_index(path):
    """Load gzipped fasta, return dict: contig_id -> sequence string."""
    contigs = {}
    opener = gzip.open if path.endswith(".gz") else open
    cur_id = None
    chunks = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    contigs[cur_id] = "".join(chunks)
                cur_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if cur_id is not None:
            contigs[cur_id] = "".join(chunks)
    return contigs


def compute_nll_features(logits, input_ids):
    """
    8 NLL statistics from logits.
    logits: (L, vocab_size), input_ids: (L,)
    """
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
    targets = input_ids[1:]
    per_pos_nll = -log_probs[torch.arange(len(targets)), targets]
    nll = per_pos_nll.cpu().numpy()

    diff_var = np.diff(nll).var() if len(nll) > 1 else 0.0
    if len(nll) > 100:
        ac100 = np.corrcoef(nll[:-100], nll[100:])[0, 1]
    else:
        ac100 = 0.0
    if len(nll) > 1000:
        ac1000 = np.corrcoef(nll[:-1000], nll[1000:])[0, 1]
    else:
        ac1000 = 0.0
    hist, _ = np.histogram(nll, bins=50, density=True)
    hist = hist + 1e-10
    nll_entropy = float(-np.sum(hist * np.log(hist)) * (nll.max() - nll.min()) / 50)

    return np.array([
        nll.mean(), nll.std(), np.median(nll), np.percentile(nll, 75),
        diff_var, ac100, ac1000, nll_entropy,
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, required=True)
    ap.add_argument("--n_gpus", type=int, required=True)
    ap.add_argument("--genomes_dir", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--layer_name", default="blocks.20.mlp.l3")
    ap.add_argument("--extract_nll", action="store_true", default=True)
    ap.add_argument("--no_nll", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="run single-sample test then exit")
    args = ap.parse_args()
    extract_nll = args.extract_nll and not args.no_nll

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES maps to 0

    # load model
    t0 = time.time()
    print(f"[GPU{args.gpu_id}] Loading Evo2 7B...", flush=True)
    os.environ["HF_HUB_CACHE"] = "/data/public_models/hub"
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    model.model.eval()
    load_time = time.time() - t0
    print(f"[GPU{args.gpu_id}] Model loaded in {load_time:.0f}s, eval={not model.model.training}", flush=True)

    # register hook for intermediate layer
    hook_output = {}
    def hook_fn(module, input, output):
        hook_output["value"] = output

    target_module = None
    for name, module in model.model.named_modules():
        if name == args.layer_name:
            target_module = module
            break
    if target_module is None:
        # try searching
        available = [n for n, _ in model.model.named_modules() if "mlp" in n and "l3" in n]
        print(f"[GPU{args.gpu_id}] ERROR: layer '{args.layer_name}' not found. Available mlp.l3: {available[:10]}")
        sys.exit(1)
    target_module.register_forward_hook(hook_fn)
    print(f"[GPU{args.gpu_id}] Hook registered on {args.layer_name}", flush=True)

    # preflight: single sample test
    test_seq = "ACGT" * 4096  # 16384 bp
    input_ids = torch.tensor(model.tokenizer.tokenize(test_seq), dtype=torch.int).unsqueeze(0).to(device)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        result = model(input_ids)
    t_single = time.time() - t1
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # handle various return types from Evo2
    if isinstance(result, tuple):
        logits = result[0]
        if isinstance(logits, tuple):
            logits = logits[0]
    else:
        logits = result
    print(f"[GPU{args.gpu_id}] raw return type: {type(result)}, logits type: {type(logits)}, "
          f"logits shape: {logits.shape if hasattr(logits, 'shape') else 'N/A'}", flush=True)

    hidden = hook_output["value"]
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    emb = hidden.float().mean(dim=1).cpu().numpy()
    print(f"[GPU{args.gpu_id}] Preflight: hidden={hidden.shape}, "
          f"emb={emb.shape}, time={t_single:.2f}s, peak_mem={peak_mem:.1f}GB", flush=True)

    if extract_nll:
        nll_feat = compute_nll_features(logits.squeeze(0), input_ids[0])
        print(f"[GPU{args.gpu_id}] NLL features: {nll_feat}", flush=True)

    if args.preflight:
        # write preflight report
        os.makedirs(args.output_dir, exist_ok=True)
        report = {
            "evo2_version": "evo2_7b",
            "single_sample_time_sec": round(t_single, 2),
            "single_sample_gpu_memory_gb": round(peak_mem, 1),
            "model_load_time_sec": round(load_time, 0),
            "embedding_shape": list(emb.shape),
            "hidden_shape": list(hidden.shape),
            "total_windows": 948362,
            "windows_per_gpu": 948362 // args.n_gpus,
            "estimated_total_hours": round(948362 * t_single / args.n_gpus / 3600, 1),
            "estimated_output_size_gb": round(948362 * 4096 * 4 / 1e9, 1),
        }
        with open(os.path.join(args.output_dir, "stage2_preflight.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    # === Full extraction ===
    os.makedirs(args.output_dir, exist_ok=True)

    # load metadata grouped by genome
    genome_windows = defaultdict(list)
    with open(args.metadata_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            genome_windows[row["genome_id"]].append(row)

    all_genomes = sorted(genome_windows.keys())
    my_genomes = all_genomes[args.gpu_id::args.n_gpus]
    print(f"[GPU{args.gpu_id}] Assigned {len(my_genomes)} genomes, "
          f"{sum(len(genome_windows[g]) for g in my_genomes)} windows", flush=True)

    log_path = os.path.join(args.output_dir, f"gpu{args.gpu_id}_log.csv")
    log_f = open(log_path, "a")

    total_done = 0
    total_start = time.time()

    for gi, genome_id in enumerate(my_genomes):
        flag = os.path.join(args.output_dir, f"{genome_id}_done.flag")
        if os.path.exists(flag):
            total_done += len(genome_windows[genome_id])
            continue

        # load genome fasta
        fasta_gz = os.path.join(args.genomes_dir, f"{genome_id}.fna.gz")
        if not os.path.exists(fasta_gz):
            log_f.write(f"{genome_id},{args.gpu_id},0,0,error,fasta_not_found\n")
            log_f.flush()
            continue

        try:
            contigs = load_fasta_index(fasta_gz)
        except Exception as e:
            log_f.write(f"{genome_id},{args.gpu_id},0,0,error,{str(e)[:100]}\n")
            log_f.flush()
            continue

        windows = genome_windows[genome_id]
        embeddings = []
        nll_features = []
        t_genome = time.time()
        n_err = 0

        for row in windows:
            seq_full = contigs.get(row["contig_id"], "")
            start, end = int(row["window_start"]), int(row["window_end"])
            seq = seq_full[start:end].upper()

            if len(seq) < 1000:
                # too short, pad or skip
                embeddings.append(np.zeros(4096, dtype=np.float32))
                if extract_nll:
                    nll_features.append(np.zeros(8, dtype=np.float32))
                n_err += 1
                continue

            # replace non-ACGT with random
            seq = "".join(c if c in "ACGT" else "ACGT"[hash(c) % 4] for c in seq)

            try:
                input_ids = torch.tensor(
                    model.tokenizer.tokenize(seq), dtype=torch.int
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    result = model(input_ids)
                    logits = result[0] if isinstance(result, tuple) else result
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    hidden = hook_output["value"]
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                    emb = hidden.float().mean(dim=1).squeeze(0).cpu().numpy()  # (4096,)
                    embeddings.append(emb.astype(np.float32))

                    if extract_nll:
                        nll_feat = compute_nll_features(logits.squeeze(0), input_ids[0])
                        nll_features.append(nll_feat)

            except Exception as e:
                embeddings.append(np.zeros(4096, dtype=np.float32))
                if extract_nll:
                    nll_features.append(np.zeros(8, dtype=np.float32))
                n_err += 1

        dt = time.time() - t_genome
        total_done += len(windows)

        # save
        np.save(os.path.join(args.output_dir, f"emb_{genome_id}.npy"),
                np.vstack(embeddings))
        if extract_nll and nll_features:
            np.save(os.path.join(args.output_dir, f"nll_{genome_id}.npy"),
                    np.vstack(nll_features))
        open(flag, "w").close()

        elapsed = time.time() - total_start
        rate = total_done / elapsed if elapsed > 0 else 0
        eta = (sum(len(genome_windows[g]) for g in my_genomes) - total_done) / rate if rate > 0 else 0

        log_f.write(f"{genome_id},{args.gpu_id},{len(windows)},{dt:.1f},success,err={n_err}\n")
        log_f.flush()

        if (gi + 1) % 10 == 0 or gi == 0:
            print(f"[GPU{args.gpu_id}] {gi+1}/{len(my_genomes)} genomes, "
                  f"{total_done} windows, {rate:.1f} win/s, ETA {eta/3600:.1f}h", flush=True)

    log_f.close()
    print(f"[GPU{args.gpu_id}] DONE. {total_done} windows in {(time.time()-total_start)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
