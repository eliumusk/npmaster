#!/usr/bin/env python3
"""
P1-T2b: Multi-GPU parallel Evo2 extraction for padded windows
==============================================================
Splits padded_sequences.fasta across N GPUs, extracts in parallel using
multiprocessing, then merges results.

Usage:
  python p1_t2_extract_parallel.py \
    --fasta /data/muskliu/np_master/data/windows/edge_padding/padded_sequences.fasta \
    --existing-emb /data/muskliu/np_master/data/embeddings/embeddings.npy \
    --existing-nll /data/muskliu/np_master/data/embeddings/nll_features.npy \
    --out /data/muskliu/np_master/data/embeddings/edge_padding \
    --gpus 0,1,2,3,5,6,7 \
    --layer blocks.20.mlp.l3
"""
import argparse
import json
import os
import time
import random
from multiprocessing import Process, Queue

import numpy as np


def load_fasta(path):
    seqs = []
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def clean_sequence(seq):
    bases = "ACGT"
    return "".join(c if c in bases else random.choice(bases) for c in seq.upper())


def extract_worker(gpu_id, shard_seqs, layer_name, out_dir, result_queue):
    """Worker process: load model on one GPU, extract embeddings for assigned sequences."""
    import torch

    device = f"cuda:{gpu_id}"
    shard_id = f"gpu{gpu_id}"
    n = len(shard_seqs)
    print(f"[{shard_id}] Starting: {n} sequences on {device}")

    # Check for existing checkpoint
    emb_path = os.path.join(out_dir, f"shard_{shard_id}_emb.npy")
    nll_path = os.path.join(out_dir, f"shard_{shard_id}_nll.npy")
    ids_path = os.path.join(out_dir, f"shard_{shard_id}_ids.json")
    done_flag = os.path.join(out_dir, f"shard_{shard_id}_done.flag")

    if os.path.exists(done_flag):
        print(f"[{shard_id}] Already done (checkpoint found). Skipping.")
        result_queue.put({"gpu": gpu_id, "status": "skipped", "n": n})
        return

    try:
        from evo2 import Evo2
        model = Evo2("evo2_7b")
        model.model.eval()

        hidden_states = {}

        def hook_fn(module, input, output):
            hidden_states["out"] = output.detach()

        parts = layer_name.split(".")
        layer = model.model
        for p in parts:
            if p.isdigit():
                layer = layer[int(p)]
            else:
                layer = getattr(layer, p)
        hook = layer.register_forward_hook(hook_fn)

        embeddings = []
        nll_features = []
        window_ids = []

        t_start = time.time()
        for i, (wid, seq) in enumerate(shard_seqs):
            seq_clean = clean_sequence(seq)
            input_ids = model.tokenizer(seq_clean, return_tensors="pt")["input_ids"].to(device)

            with torch.no_grad():
                logits = model.model(input_ids).logits
                h = hidden_states["out"]
                emb = h[0].mean(dim=0).cpu().numpy() if h.dim() == 3 else h.mean(dim=0).cpu().numpy()
                embeddings.append(emb)

                # NLL features
                log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
                target_ids = input_ids[0, 1:]
                nll = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).cpu().numpy()

                nll_mean = float(np.mean(nll))
                nll_std = float(np.std(nll))
                nll_median = float(np.median(nll))
                nll_p75 = float(np.percentile(nll, 75))
                diffs = np.diff(nll)
                diff_var = float(np.var(diffs)) if len(diffs) > 0 else 0.0

                def autocorr(x, lag):
                    if len(x) <= lag:
                        return 0.0
                    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

                ac100 = autocorr(nll, 100)
                ac1000 = autocorr(nll, 1000)
                probs = torch.softmax(logits[0, :-1], dim=-1)
                entropy = float((-probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().cpu())
                nll_features.append([nll_mean, nll_std, nll_median, nll_p75,
                                    diff_var, ac100, ac1000, entropy])

            window_ids.append(wid)

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate / 3600
                print(f"[{shard_id}] {i+1}/{n} ({rate:.1f}/s, ETA {eta:.1f}h)")

        hook.remove()

        emb_arr = np.stack(embeddings).astype(np.float32)
        nll_arr = np.array(nll_features, dtype=np.float32)

        np.save(emb_path, emb_arr)
        np.save(nll_path, nll_arr)
        with open(ids_path, "w") as f:
            json.dump(window_ids, f)

        # Write done flag
        with open(done_flag, "w") as f:
            f.write(f"done {len(window_ids)}\n")

        elapsed = time.time() - t_start
        print(f"[{shard_id}] Done: {n} seqs in {elapsed/3600:.1f}h ({elapsed/n:.2f}s/seq)")
        result_queue.put({"gpu": gpu_id, "status": "done", "n": n, "time_h": elapsed / 3600})

    except Exception as e:
        print(f"[{shard_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({"gpu": gpu_id, "status": "error", "error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--existing-emb", required=True)
    ap.add_argument("--existing-nll", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,5,6,7", help="comma-separated GPU IDs")
    ap.add_argument("--layer", default="blocks.20.mlp.l3")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load sequences
    print("Loading sequences...")
    seqs = load_fasta(args.fasta)
    print(f"  Total: {len(seqs)} sequences")

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    print(f"  GPUs: {gpu_ids} ({n_gpus} total)")

    # Split sequences across GPUs (interleaved for balance)
    shards = {g: [] for g in gpu_ids}
    for i, seq in enumerate(seqs):
        shards[gpu_ids[i % n_gpus]].append(seq)

    for g in gpu_ids:
        print(f"  GPU {g}: {len(shards[g])} sequences")

    # Launch workers
    print("\nLaunching extraction workers...")
    result_queue = Queue()
    workers = []
    for g in gpu_ids:
        p = Process(target=extract_worker,
                    args=(g, shards[g], args.layer, args.out, result_queue))
        p.start()
        workers.append(p)

    # Wait for all workers
    for p in workers:
        p.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    print(f"\nAll workers done. Results: {results}")

    # Merge shards in correct order
    print("\nMerging shards...")
    # We need to reconstruct the original order (interleaved)
    all_emb = [None] * len(seqs)
    all_nll = [None] * len(seqs)
    all_ids = [None] * len(seqs)

    for g in gpu_ids:
        emb_path = os.path.join(args.out, f"shard_gpu{g}_emb.npy")
        nll_path = os.path.join(args.out, f"shard_gpu{g}_nll.npy")
        ids_path = os.path.join(args.out, f"shard_gpu{g}_ids.json")

        if not os.path.exists(emb_path):
            print(f"  WARNING: Missing shard for GPU {g}")
            continue

        emb = np.load(emb_path)
        nll = np.load(nll_path)
        with open(ids_path) as f:
            ids = json.load(f)

        # Reconstruct original indices
        shard_indices = list(range(gpu_ids.index(g), len(seqs), n_gpus))
        for j, idx in enumerate(shard_indices):
            if j < len(emb):
                all_emb[idx] = emb[j]
                all_nll[idx] = nll[j]
                all_ids[idx] = ids[j]

    # Check for missing
    missing = sum(1 for e in all_emb if e is None)
    if missing > 0:
        print(f"  WARNING: {missing} sequences missing from merge!")

    # Filter out None entries
    valid = [(e, n, i) for e, n, i in zip(all_emb, all_nll, all_ids) if e is not None]
    new_emb = np.stack([v[0] for v in valid]).astype(np.float32)
    new_nll = np.stack([v[1] for v in valid]).astype(np.float32)
    new_ids = [v[2] for v in valid]

    print(f"  New embeddings: {new_emb.shape}")

    # Save merged new-only
    np.save(os.path.join(args.out, "padded_embeddings.npy"), new_emb)
    np.save(os.path.join(args.out, "padded_nll_features.npy"), new_nll)
    with open(os.path.join(args.out, "padded_window_ids.json"), "w") as f:
        json.dump(new_ids, f)

    # Merge with existing
    print("Merging with existing embeddings...")
    existing_emb = np.load(args.existing_emb)
    merged = np.vstack([existing_emb, new_emb])
    merged_path = os.path.join(args.out, "embeddings_merged.npy")
    np.save(merged_path, merged)
    print(f"  Merged: {merged.shape} -> {merged_path}")

    if args.existing_nll and os.path.exists(args.existing_nll):
        existing_nll = np.load(args.existing_nll)
        merged_nll = np.vstack([existing_nll, new_nll])
        merged_nll_path = os.path.join(args.out, "nll_features_merged.npy")
        np.save(merged_nll_path, merged_nll)
        print(f"  Merged NLL: {merged_nll.shape}")

    # Summary
    summary = {
        "task_id": "P1-T2b",
        "n_total": len(seqs),
        "n_extracted": len(valid),
        "n_missing": missing,
        "gpus": gpu_ids,
        "layer": args.layer,
        "merged_shape": list(merged.shape),
        "worker_results": results,
    }
    with open(os.path.join(args.out, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Summary: {args.out}/extraction_summary.json")


if __name__ == "__main__":
    main()
